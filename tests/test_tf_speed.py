"""Benchmark tests for transformer (TF) performance optimizations.

Tests proposed optimizations for correctness (exact numerical match) and
speed improvement before applying to production code.

Optimizations tested:
1. Batched log-prob computation (single actor call vs sequential for-loop)
2. jax.lax.scan for SDE sampling (vs Python for-loop)
3. jax.lax.fori_loop for ODE sampling (vs Python for-loop)
4. Conditional denoiser in TF actor (skip when not needed)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import copy
from typing import Callable, Tuple, Optional, Dict, Any
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax

# ── Existing production code imports ──
from ogpo.networks.actors import ActorVectorFieldTF, ActorVectorField
from ogpo.networks.critics import ValueTF, Value
from ogpo.networks.modules.attention import (
    AdaLNLayer, AdaLNFinalLayer, TimestepEmbedder, ConditioningMLP,
)
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode as prod_ode,
    sample_flow_actions_sde as prod_sde,
    compute_flow_log_prob as prod_log_prob,
)

# ──────────────────────────────────────────────────────────────────────
# Test configuration
# ──────────────────────────────────────────────────────────────────────
BATCH_SIZE = 256
OBS_DIM = 23
ACTION_DIM = 7
HORIZON = 4
FLOW_STEPS = 10
HIDDEN_DIM = 256
NUM_LAYERS = 3
NUM_HEADS = 4
NUM_QS = 10
SEED = 42
N_WARMUP = 3
N_BENCH = 20

FULL_ACT_DIM = ACTION_DIM * HORIZON  # 28

# ──────────────────────────────────────────────────────────────────────
# Helper: initialize networks and get params
# ──────────────────────────────────────────────────────────────────────
def make_tf_actor(use_denoiser=True):
    return ActorVectorFieldTF(
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        action_chunk_size=HORIZON,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout_rate=0.0,
        use_denoiser=use_denoiser,
    )


def make_mlp_actor(use_denoiser=True):
    return ActorVectorField(
        hidden_dims=[512, 512, 512, 512],
        action_dim=FULL_ACT_DIM,
        layer_norm=True,
        use_denoiser=use_denoiser,
    )


def init_actor(actor_def, rng, use_denoiser=True):
    obs = jnp.ones((1, OBS_DIM))
    act = jnp.ones((1, FULL_ACT_DIM))
    t = jnp.ones((1, 1))
    init_kwargs = dict(observations=obs, actions=act, times=t)
    if use_denoiser:
        init_kwargs['return_denoiser'] = True
        if isinstance(actor_def, ActorVectorFieldTF):
            init_kwargs['deterministic'] = True
    variables = actor_def.init(rng, **init_kwargs)
    return variables['params']


# ──────────────────────────────────────────────────────────────────────
# Proposed optimization 1: Batched compute_flow_log_prob
# ──────────────────────────────────────────────────────────────────────
def batched_compute_flow_log_prob(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        chains: jnp.ndarray,
        flow_steps: int,
        use_constant_noise: bool = False,
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        normalize_horizon: bool = False,
        normalize_dim: bool = False,
        is_encoded: bool = False,
        params: Any = None,
        get_entropy: bool = False,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        ft_flow_steps: int = 0,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict]:
    """Batched version: one actor call for all flow steps instead of a for-loop."""
    batch_size = chains.shape[0]
    act_dim = chains.shape[-1]
    dt = 1.0 / flow_steps

    effective_ft = ft_flow_steps if (0 < ft_flow_steps < flow_steps) else flow_steps
    ft_start = flow_steps - effective_ft

    logprob = 0.0
    joint_entropy = 0.0
    logprob_steps = 0

    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    if ft_start == 0:
        logprob_init = init_dist.log_prob(chains[:, 0]).sum(-1)
        logprob = logprob + logprob_init
        logprob_steps += 1
        if get_entropy:
            entropy_init = init_dist.entropy().sum(-1)
            joint_entropy = joint_entropy + entropy_init

    chains_prev = chains[:, :-1, :]  # [B, S, act_dim]
    chains_next = chains[:, 1:, :]   # [B, S, act_dim]

    # ── KEY CHANGE: batch all flow steps into a single actor call ──
    S = flow_steps
    all_obs = jnp.repeat(observations, S, axis=0)              # (B*S, obs_dim)
    all_x_t = chains_prev.reshape(-1, act_dim)                  # (B*S, act_dim)
    t_vals  = jnp.arange(S, dtype=jnp.float32) / S             # (S,)
    # Tile: for each batch element, repeat all S time values
    all_t = jnp.tile(t_vals, (batch_size,))[:, None]            # (B*S, 1)

    if params is not None:
        all_v = actor_fn(all_obs, all_x_t, all_t, params=params, is_encoded=is_encoded)
    else:
        all_v = actor_fn(all_obs, all_x_t, all_t, is_encoded=is_encoded)
    chains_vel = all_v.reshape(batch_size, S, act_dim)

    if use_constant_noise:
        chains_stds = jnp.full_like(chains_prev, constant_noise_std)
    else:
        if params is not None:
            all_sigma = noise_fn(all_obs, all_t, params=params)
        else:
            all_sigma = noise_fn(all_obs, all_t)
        all_sigma = jnp.maximum(all_sigma, min_noise_std)
        chains_stds = all_sigma.reshape(batch_size, S, act_dim)

    # ── Rest is identical to production code ──
    chains_mean = chains_prev + chains_vel * dt
    if clip_intermediate:
        chains_mean = jnp.clip(chains_mean, -clip_value, clip_value)

    chains_dist = distrax.Normal(chains_mean, chains_stds)
    logprob_trans = chains_dist.log_prob(chains_next).sum(-1)
    logprob_trans_ft = logprob_trans[:, ft_start:]
    logprob = logprob + logprob_trans_ft.sum(axis=-1)
    logprob_steps += effective_ft

    if get_entropy:
        entropy_trans = chains_dist.entropy().sum(-1)
        entropy_trans_ft = entropy_trans[:, ft_start:]
        joint_entropy = joint_entropy + entropy_trans_ft.sum(axis=-1)
        entropy_rate = joint_entropy / logprob_steps / act_dim
    else:
        entropy_rate = None

    if normalize_horizon:
        logprob = logprob / logprob_steps
    if normalize_dim:
        logprob = logprob / act_dim

    info = {
        'vel_mean': chains_vel.mean(),
        'sigma_mean': chains_stds.mean(),
        'logprob_mean': logprob.mean(),
        'logprob_steps': logprob_steps,
    }
    if get_entropy:
        info['joint_entropy'] = joint_entropy.mean()
        info['entropy_rate'] = entropy_rate.mean()

    return logprob, entropy_rate, info


# ──────────────────────────────────────────────────────────────────────
# Proposed optimization 2: jax.lax.scan for SDE sampling
# ──────────────────────────────────────────────────────────────────────
def scan_sample_flow_actions_sde(
        actor_fn: Callable,
        noise_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        flow_steps: int,
        act_dim: int,
        use_constant_noise: bool = False,
        constant_noise_std: float = 0.01,
        min_noise_std: float = 0.08,
        randn_clip_value: float = 3.0,
        act_min: float = -1.0,
        act_max: float = 1.0,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        is_encoded: bool = False,
        params: Any = None,
        ft_flow_steps: int = 0,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """SDE sampling using jax.lax.scan instead of Python for-loop."""
    batch_size = observations.shape[0]
    dt = 1.0 / flow_steps

    effective_ft = ft_flow_steps if (0 < ft_flow_steps < flow_steps) else flow_steps
    ft_start = flow_steps - effective_ft

    rng, key = jax.random.split(rng)
    actions = jax.random.normal(key, (batch_size, act_dim))

    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    if ft_start == 0:
        logprob_init = init_dist.log_prob(actions).sum(-1)
    else:
        logprob_init = jnp.zeros(batch_size)

    def scan_body(carry, step_idx):
        actions, logprob, rng = carry
        rng, key = jax.random.split(rng)
        t = jnp.full((batch_size, 1), step_idx / flow_steps)

        if params is not None:
            vels = actor_fn(observations, actions, t, params=params, is_encoded=is_encoded)
        else:
            vels = actor_fn(observations, actions, t, is_encoded=is_encoded)

        mean_next = actions + vels * dt
        if clip_intermediate:
            mean_next = jnp.clip(mean_next, -clip_value, clip_value)

        if use_constant_noise:
            sigma = constant_noise_std * jnp.ones(actions.shape)
        else:
            if params is not None:
                sigma = noise_fn(observations, t, params=params)
            else:
                sigma = noise_fn(observations, t)
            sigma = jnp.maximum(sigma, min_noise_std)

        step_dist = distrax.Normal(mean_next, sigma)
        rng, noise_key = jax.random.split(rng)
        new_actions = step_dist.sample(seed=noise_key)
        new_actions = jnp.clip(
            new_actions,
            mean_next - randn_clip_value * sigma,
            mean_next + randn_clip_value * sigma,
        )

        # Accumulate logprob only for finetuned transitions
        step_logprob = step_dist.log_prob(new_actions).sum(-1)
        logprob = logprob + jnp.where(step_idx >= ft_start, step_logprob, 0.0)

        # Final step clipping
        new_actions = jnp.where(
            step_idx == flow_steps - 1,
            jnp.clip(new_actions, act_min, act_max),
            new_actions,
        )

        return (new_actions, logprob, rng), (new_actions, sigma)

    step_indices = jnp.arange(flow_steps, dtype=jnp.float32)
    (final_actions, logprob, _), (all_actions, all_sigmas) = jax.lax.scan(
        scan_body, (actions, logprob_init, rng), step_indices,
    )

    # Build chains: prepend initial noise, append scan outputs
    chains = jnp.concatenate(
        [actions[:, None, :], jnp.transpose(all_actions, (1, 0, 2))],
        axis=1,
    )  # (B, flow_steps+1, act_dim)
    sigmas = jnp.transpose(all_sigmas, (1, 0, 2))  # (B, flow_steps, act_dim)

    return final_actions, chains, logprob, sigmas


# ──────────────────────────────────────────────────────────────────────
# Proposed optimization 3: jax.lax.fori_loop for ODE sampling
# ──────────────────────────────────────────────────────────────────────
def fori_sample_flow_actions_ode(
        actor_fn: Callable,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        flow_steps: int,
        act_dim: int,
        act_min: float = -1.0,
        act_max: float = 1.0,
        clip_intermediate: bool = True,
        clip_value: float = 1.0,
        noises: jnp.ndarray = None,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
    """ODE sampling using jax.lax.fori_loop instead of Python for-loop."""
    batch_size = observations.shape[0]

    if noises is None:
        actions = jax.random.normal(rng, (batch_size, act_dim))
    else:
        actions = noises

    dt = 1.0 / flow_steps

    def body_fn(i, actions):
        t = jnp.full((batch_size, 1), i / flow_steps)
        vels = actor_fn(observations, actions, t, is_encoded=is_encoded)
        actions = actions + vels * dt
        if clip_intermediate:
            actions = jnp.clip(actions, -clip_value, clip_value)
        return actions

    actions = jax.lax.fori_loop(0, flow_steps, body_fn, actions)
    actions = jnp.clip(actions, act_min, act_max)
    return actions


# ──────────────────────────────────────────────────────────────────────
# Proposed optimization 4: conditional denoiser in TF actor
# ──────────────────────────────────────────────────────────────────────
class ActorVectorFieldTF_OptDenoiser(ActorVectorFieldTF):
    """TF actor that skips denoiser computation when return_denoiser=False."""

    @nn.compact
    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False,
                 return_denoiser=False, deterministic=True, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        actions = actions.reshape(-1, self.action_chunk_size, self.action_dim)
        x = self.action_proj(actions) + self.pos_embed
        cond = self.cond_proj(observations)
        if times is not None:
            cond = cond + self.time_embed(times)
        if dt is not None:
            cond = cond + self.dt_embed(dt)

        T = self.action_chunk_size
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, None, :, :]

        for layer in self.transformer_layers:
            x = layer(x, cond, mask=causal_mask, deterministic=deterministic)

        v = self.final_layer(x, cond)
        v = v.reshape(-1, self.action_chunk_size * self.action_dim)

        # ── KEY CHANGE: only compute denoiser when needed ──
        if self.use_denoiser and return_denoiser:
            z = self.denoiser_final_layer(x, cond)
            z = z.reshape(-1, self.action_chunk_size * self.action_dim)
            return v, z

        return v


# ──────────────────────────────────────────────────────────────────────
# Timing utility
# ──────────────────────────────────────────────────────────────────────
def bench(fn, *args, n_warmup=N_WARMUP, n_iter=N_BENCH, label=""):
    """Time a JIT-compiled function. Returns (result, mean_ms)."""
    # Warmup (includes compilation)
    for _ in range(n_warmup):
        result = fn(*args)
        jax.block_until_ready(result)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn(*args)
        jax.block_until_ready(result)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    if label:
        print(f"  {label}: {mean_ms:.2f} ± {std_ms:.2f} ms")
    return result, mean_ms


# ──────────────────────────────────────────────────────────────────────
# TEST 1: Batched log-prob correctness + speed
# ──────────────────────────────────────────────────────────────────────
def test_batched_log_prob():
    print("\n" + "=" * 70)
    print("TEST 1: Batched compute_flow_log_prob")
    print("=" * 70)

    rng = jax.random.PRNGKey(SEED)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    # --- Create TF actor ---
    actor_def = make_tf_actor(use_denoiser=False)
    params = init_actor(actor_def, init_rng, use_denoiser=False)

    def actor_fn(obs, actions, t, params=None, is_encoded=False):
        return actor_def.apply({'params': params}, observations=obs, actions=actions, times=t)

    # --- Generate fake chains ---
    chains = jax.random.normal(data_rng, (BATCH_SIZE, FLOW_STEPS + 1, FULL_ACT_DIM))
    observations = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, OBS_DIM))

    common_kwargs = dict(
        actor_fn=actor_fn,
        noise_fn=None,
        observations=observations,
        chains=chains,
        flow_steps=FLOW_STEPS,
        use_constant_noise=True,
        constant_noise_std=0.01,
        is_encoded=False,
        params=params,
        get_entropy=True,
        clip_intermediate=True,
        clip_value=1.0,
        ft_flow_steps=FLOW_STEPS,
    )

    # JIT both versions
    prod_fn = jax.jit(partial(prod_log_prob, **common_kwargs))
    batched_fn = jax.jit(partial(batched_compute_flow_log_prob, **common_kwargs))

    # --- Correctness ---
    prod_result = prod_fn()
    batched_result = batched_fn()

    logprob_prod, ent_prod, info_prod = prod_result
    logprob_batched, ent_batched, info_batched = batched_result

    logprob_diff = jnp.abs(logprob_prod - logprob_batched).max()
    ent_diff = jnp.abs(ent_prod - ent_batched).max()
    vel_diff = jnp.abs(info_prod['vel_mean'] - info_batched['vel_mean'])

    print(f"\n  Correctness (max abs diff):")
    print(f"    logprob: {logprob_diff:.2e}")
    print(f"    entropy: {ent_diff:.2e}")
    print(f"    vel_mean: {vel_diff:.2e}")

    assert logprob_diff < 1e-4, f"Log prob mismatch: {logprob_diff}"
    assert ent_diff < 1e-4, f"Entropy mismatch: {ent_diff}"
    print("    ✓ Numerical match confirmed")

    # --- Speed ---
    print(f"\n  Speed (B={BATCH_SIZE}, S={FLOW_STEPS}, hidden={HIDDEN_DIM}, layers={NUM_LAYERS}):")
    _, prod_ms = bench(prod_fn, label="Sequential (production)")
    _, batched_ms = bench(batched_fn, label="Batched (proposed)")
    speedup = prod_ms / batched_ms
    print(f"  Speedup: {speedup:.2f}x")

    # --- Also test with MLP actor for comparison ---
    print(f"\n  MLP baseline (same test for reference):")
    mlp_def = make_mlp_actor(use_denoiser=False)
    mlp_params = mlp_def.init(init_rng, observations=jnp.ones((1, OBS_DIM)),
                               actions=jnp.ones((1, FULL_ACT_DIM)),
                               times=jnp.ones((1, 1)))['params']

    def mlp_actor_fn(obs, actions, t, params=None, is_encoded=False):
        return mlp_def.apply({'params': params}, observations=obs, actions=actions, times=t)

    mlp_kwargs = {**common_kwargs, 'actor_fn': mlp_actor_fn, 'params': mlp_params}
    mlp_prod_fn = jax.jit(partial(prod_log_prob, **mlp_kwargs))
    mlp_batched_fn = jax.jit(partial(batched_compute_flow_log_prob, **mlp_kwargs))

    _, mlp_prod_ms = bench(mlp_prod_fn, label="MLP Sequential")
    _, mlp_batched_ms = bench(mlp_batched_fn, label="MLP Batched")
    print(f"  MLP Speedup: {mlp_prod_ms / mlp_batched_ms:.2f}x")

    return speedup


# ──────────────────────────────────────────────────────────────────────
# TEST 2: jax.lax.scan SDE correctness + speed
# ──────────────────────────────────────────────────────────────────────
def test_scan_sde():
    print("\n" + "=" * 70)
    print("TEST 2: jax.lax.scan SDE sampling")
    print("=" * 70)

    rng = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(rng)

    actor_def = make_tf_actor(use_denoiser=False)
    params = init_actor(actor_def, init_rng, use_denoiser=False)

    def actor_fn(obs, actions, t, params=None, is_encoded=False):
        return actor_def.apply({'params': params}, observations=obs, actions=actions, times=t)

    observations = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, OBS_DIM))

    common_kwargs = dict(
        actor_fn=actor_fn,
        noise_fn=None,
        observations=observations,
        flow_steps=FLOW_STEPS,
        act_dim=FULL_ACT_DIM,
        use_constant_noise=True,
        constant_noise_std=0.01,
        randn_clip_value=3.0,
        act_min=-1.0,
        act_max=1.0,
        clip_intermediate=True,
        clip_value=1.0,
        is_encoded=False,
        params=params,
        ft_flow_steps=FLOW_STEPS,
    )

    sample_rng = jax.random.PRNGKey(99)

    prod_fn = jax.jit(partial(prod_sde, rng=sample_rng, **common_kwargs))
    scan_fn = jax.jit(partial(scan_sample_flow_actions_sde, rng=sample_rng, **common_kwargs))

    # --- Correctness ---
    prod_actions, prod_chains, prod_logprob, prod_sigmas = prod_fn()
    scan_actions, scan_chains, scan_logprob, scan_sigmas = scan_fn()

    action_diff = jnp.abs(prod_actions - scan_actions).max()
    chain_diff = jnp.abs(prod_chains - scan_chains).max()
    logprob_diff = jnp.abs(prod_logprob - scan_logprob).max()
    sigma_diff = jnp.abs(prod_sigmas - scan_sigmas).max()

    print(f"\n  Correctness (max abs diff):")
    print(f"    actions:  {action_diff:.2e}")
    print(f"    chains:   {chain_diff:.2e}")
    print(f"    logprob:  {logprob_diff:.2e}")
    print(f"    sigmas:   {sigma_diff:.2e}")

    assert action_diff < 1e-4, f"Action mismatch: {action_diff}"
    assert logprob_diff < 1e-4, f"Log prob mismatch: {logprob_diff}"
    print("    ✓ Numerical match confirmed")

    # --- Speed ---
    print(f"\n  Speed (B={BATCH_SIZE}, S={FLOW_STEPS}):")
    _, prod_ms = bench(prod_fn, label="Python for-loop (production)")
    _, scan_ms = bench(scan_fn, label="jax.lax.scan (proposed)")
    speedup = prod_ms / scan_ms
    print(f"  Speedup: {speedup:.2f}x")

    return speedup


# ──────────────────────────────────────────────────────────────────────
# TEST 3: fori_loop ODE correctness + speed
# ──────────────────────────────────────────────────────────────────────
def test_fori_ode():
    print("\n" + "=" * 70)
    print("TEST 3: jax.lax.fori_loop ODE sampling")
    print("=" * 70)

    rng = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(rng)

    actor_def = make_tf_actor(use_denoiser=False)
    params = init_actor(actor_def, init_rng, use_denoiser=False)

    def actor_fn(obs, actions, t, is_encoded=False):
        return actor_def.apply({'params': params}, observations=obs, actions=actions, times=t)

    observations = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, OBS_DIM))

    common_kwargs = dict(
        actor_fn=actor_fn,
        observations=observations,
        flow_steps=FLOW_STEPS,
        act_dim=FULL_ACT_DIM,
        act_min=-1.0,
        act_max=1.0,
        clip_intermediate=True,
        clip_value=1.0,
        is_encoded=False,
    )

    sample_rng = jax.random.PRNGKey(99)
    noises = jax.random.normal(sample_rng, (BATCH_SIZE, FULL_ACT_DIM))

    prod_fn = jax.jit(partial(prod_ode, rng=sample_rng, noises=noises, **common_kwargs))
    fori_fn = jax.jit(partial(fori_sample_flow_actions_ode, rng=sample_rng, noises=noises, **common_kwargs))

    # --- Correctness ---
    prod_actions = prod_fn()
    fori_actions = fori_fn()

    action_diff = jnp.abs(prod_actions - fori_actions).max()
    print(f"\n  Correctness (max abs diff):")
    print(f"    actions: {action_diff:.2e}")

    assert action_diff < 1e-4, f"Action mismatch: {action_diff}"
    print("    ✓ Numerical match confirmed")

    # --- Speed ---
    print(f"\n  Speed (B={BATCH_SIZE}, S={FLOW_STEPS}):")
    _, prod_ms = bench(prod_fn, label="Python for-loop (production)")
    _, fori_ms = bench(fori_fn, label="jax.lax.fori_loop (proposed)")
    speedup = prod_ms / fori_ms
    print(f"  Speedup: {speedup:.2f}x")

    return speedup


# ──────────────────────────────────────────────────────────────────────
# TEST 4: Conditional denoiser correctness + speed
# ──────────────────────────────────────────────────────────────────────
def test_conditional_denoiser():
    print("\n" + "=" * 70)
    print("TEST 4: Conditional denoiser (skip when return_denoiser=False)")
    print("=" * 70)

    rng = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(rng)

    # Create both actor variants
    prod_def = make_tf_actor(use_denoiser=True)
    opt_def = ActorVectorFieldTF_OptDenoiser(
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        action_chunk_size=HORIZON,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout_rate=0.0,
        use_denoiser=True,
    )

    # Init with denoiser=True to create all params
    obs = jnp.ones((1, OBS_DIM))
    act = jnp.ones((1, FULL_ACT_DIM))
    t = jnp.ones((1, 1))
    prod_params = prod_def.init(init_rng, observations=obs, actions=act, times=t,
                                 return_denoiser=True, deterministic=True)['params']
    opt_params = opt_def.init(init_rng, observations=obs, actions=act, times=t,
                               return_denoiser=True, deterministic=True)['params']

    # Test data
    observations = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, OBS_DIM))
    actions = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, FULL_ACT_DIM))
    times = jax.random.uniform(jax.random.PRNGKey(2), (BATCH_SIZE, 1))

    # --- Correctness: velocity output should match when return_denoiser=False ---
    prod_fn_no_den = jax.jit(lambda: prod_def.apply(
        {'params': prod_params}, observations=observations, actions=actions, times=times,
        return_denoiser=False, deterministic=True))
    opt_fn_no_den = jax.jit(lambda: opt_def.apply(
        {'params': opt_params}, observations=observations, actions=actions, times=times,
        return_denoiser=False, deterministic=True))

    prod_v = prod_fn_no_den()
    opt_v = opt_fn_no_den()

    vel_diff = jnp.abs(prod_v - opt_v).max()
    print(f"\n  Correctness (return_denoiser=False, max abs diff):")
    print(f"    velocity: {vel_diff:.2e}")
    assert vel_diff < 1e-5, f"Velocity mismatch: {vel_diff}"

    # --- Correctness: both outputs should match when return_denoiser=True ---
    prod_fn_den = jax.jit(lambda: prod_def.apply(
        {'params': prod_params}, observations=observations, actions=actions, times=times,
        return_denoiser=True, deterministic=True))
    opt_fn_den = jax.jit(lambda: opt_def.apply(
        {'params': opt_params}, observations=observations, actions=actions, times=times,
        return_denoiser=True, deterministic=True))

    prod_v2, prod_z = prod_fn_den()
    opt_v2, opt_z = opt_fn_den()

    vel_diff2 = jnp.abs(prod_v2 - opt_v2).max()
    z_diff = jnp.abs(prod_z - opt_z).max()
    print(f"    velocity (denoiser=True): {vel_diff2:.2e}")
    print(f"    denoiser output: {z_diff:.2e}")
    assert vel_diff2 < 1e-5, f"Velocity mismatch with denoiser: {vel_diff2}"
    assert z_diff < 1e-5, f"Denoiser mismatch: {z_diff}"
    print("    ✓ Numerical match confirmed")

    # --- Speed: return_denoiser=False (this is the hot path) ---
    print(f"\n  Speed (B={BATCH_SIZE}, return_denoiser=False):")
    _, prod_ms = bench(prod_fn_no_den, label="Production (always computes denoiser)")
    _, opt_ms = bench(opt_fn_no_den, label="Optimized (skips denoiser)")
    speedup = prod_ms / opt_ms
    print(f"  Speedup: {speedup:.2f}x")

    # --- Speed: single forward pass (10 calls simulating flow steps) ---
    print(f"\n  Speed: 10 sequential calls (simulating flow_steps=10):")
    def prod_10_calls():
        v = prod_v
        for _ in range(10):
            v = prod_def.apply(
                {'params': prod_params}, observations=observations, actions=actions,
                times=times, return_denoiser=False, deterministic=True)
        return v

    def opt_10_calls():
        v = opt_v
        for _ in range(10):
            v = opt_def.apply(
                {'params': opt_params}, observations=observations, actions=actions,
                times=times, return_denoiser=False, deterministic=True)
        return v

    prod_10_fn = jax.jit(prod_10_calls)
    opt_10_fn = jax.jit(opt_10_calls)

    _, prod_10_ms = bench(prod_10_fn, label="Production 10 calls")
    _, opt_10_ms = bench(opt_10_fn, label="Optimized 10 calls")
    print(f"  Speedup (10 calls): {prod_10_ms / opt_10_ms:.2f}x")

    return speedup


# ──────────────────────────────────────────────────────────────────────
# TEST 5: Full actor update comparison (TF vs MLP)
# ──────────────────────────────────────────────────────────────────────
def test_tf_vs_mlp_forward_pass():
    """Compare raw forward pass times of TF vs MLP to establish baseline."""
    print("\n" + "=" * 70)
    print("TEST 5: TF vs MLP raw forward pass comparison")
    print("=" * 70)

    rng = jax.random.PRNGKey(SEED)
    rng, tf_rng, mlp_rng = jax.random.split(rng, 3)

    # TF actor
    tf_def = make_tf_actor(use_denoiser=False)
    tf_params = init_actor(tf_def, tf_rng, use_denoiser=False)

    # MLP actor
    mlp_def = make_mlp_actor(use_denoiser=False)
    mlp_params = mlp_def.init(mlp_rng, observations=jnp.ones((1, OBS_DIM)),
                               actions=jnp.ones((1, FULL_ACT_DIM)),
                               times=jnp.ones((1, 1)))['params']

    observations = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, OBS_DIM))
    actions = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, FULL_ACT_DIM))
    times = jax.random.uniform(jax.random.PRNGKey(2), (BATCH_SIZE, 1))

    tf_fn = jax.jit(lambda: tf_def.apply(
        {'params': tf_params}, observations=observations, actions=actions, times=times))
    mlp_fn = jax.jit(lambda: mlp_def.apply(
        {'params': mlp_params}, observations=observations, actions=actions, times=times))

    print(f"\n  Single forward pass (B={BATCH_SIZE}):")
    _, tf_ms = bench(tf_fn, label=f"TF ({NUM_LAYERS}L, {HIDDEN_DIM}D, {NUM_HEADS}H)")
    _, mlp_ms = bench(mlp_fn, label="MLP (4x512)")
    print(f"  TF/MLP ratio: {tf_ms / mlp_ms:.2f}x slower")

    # 10 sequential forward passes (simulating flow_steps=10)
    def tf_10():
        v = jnp.zeros((BATCH_SIZE, FULL_ACT_DIM))
        for i in range(10):
            t = jnp.full((BATCH_SIZE, 1), i / 10.0)
            v = tf_def.apply({'params': tf_params}, observations=observations,
                             actions=actions, times=t)
        return v

    def mlp_10():
        v = jnp.zeros((BATCH_SIZE, FULL_ACT_DIM))
        for i in range(10):
            t = jnp.full((BATCH_SIZE, 1), i / 10.0)
            v = mlp_def.apply({'params': mlp_params}, observations=observations,
                              actions=actions, times=t)
        return v

    tf_10_fn = jax.jit(tf_10)
    mlp_10_fn = jax.jit(mlp_10)

    print(f"\n  10 sequential forward passes (simulating flow_steps=10):")
    _, tf_10_ms = bench(tf_10_fn, label=f"TF 10 calls")
    _, mlp_10_ms = bench(mlp_10_fn, label="MLP 10 calls")
    print(f"  TF/MLP ratio: {tf_10_ms / mlp_10_ms:.2f}x slower")

    # Batched: 10 steps merged into one call with 10x batch
    big_obs = jnp.repeat(observations, 10, axis=0)
    big_act = jnp.repeat(actions, 10, axis=0)
    big_t = jnp.tile(jnp.arange(10, dtype=jnp.float32) / 10.0, (BATCH_SIZE,))[:, None]

    tf_batched_fn = jax.jit(lambda: tf_def.apply(
        {'params': tf_params}, observations=big_obs, actions=big_act, times=big_t))
    mlp_batched_fn = jax.jit(lambda: mlp_def.apply(
        {'params': mlp_params}, observations=big_obs, actions=big_act, times=big_t))

    print(f"\n  1 batched forward pass (B={BATCH_SIZE}*10 = {BATCH_SIZE*10}):")
    _, tf_bat_ms = bench(tf_batched_fn, label="TF batched")
    _, mlp_bat_ms = bench(mlp_batched_fn, label="MLP batched")
    print(f"  TF batched vs TF 10-seq: {tf_10_ms / tf_bat_ms:.2f}x speedup")
    print(f"  MLP batched vs MLP 10-seq: {mlp_10_ms / mlp_bat_ms:.2f}x speedup")


# ──────────────────────────────────────────────────────────────────────
# TEST 6: Compilation time comparison
# ──────────────────────────────────────────────────────────────────────
def test_compilation_time():
    """Compare JIT compilation times (scan/fori vs Python for-loop)."""
    print("\n" + "=" * 70)
    print("TEST 6: JIT compilation time comparison")
    print("=" * 70)

    rng = jax.random.PRNGKey(SEED)
    rng, init_rng = jax.random.split(rng)

    actor_def = make_tf_actor(use_denoiser=False)
    params = init_actor(actor_def, init_rng, use_denoiser=False)

    def actor_fn(obs, actions, t, params=None, is_encoded=False):
        return actor_def.apply({'params': params}, observations=obs, actions=actions, times=t)

    observations = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, OBS_DIM))
    sample_rng = jax.random.PRNGKey(99)

    # ODE
    ode_kwargs = dict(
        actor_fn=actor_fn, observations=observations,
        flow_steps=FLOW_STEPS, act_dim=FULL_ACT_DIM,
        is_encoded=False,
    )

    print(f"\n  ODE compilation (B={BATCH_SIZE}, S={FLOW_STEPS}):")
    noises = jax.random.normal(sample_rng, (BATCH_SIZE, FULL_ACT_DIM))

    t0 = time.perf_counter()
    prod_ode_jit = jax.jit(partial(prod_ode, rng=sample_rng, noises=noises, **ode_kwargs))
    _ = prod_ode_jit()
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    print(f"    Python for-loop: {(t1-t0)*1000:.0f} ms")

    t0 = time.perf_counter()
    fori_ode_jit = jax.jit(partial(fori_sample_flow_actions_ode, rng=sample_rng, noises=noises, **ode_kwargs))
    _ = fori_ode_jit()
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    print(f"    fori_loop:       {(t1-t0)*1000:.0f} ms")

    # SDE
    sde_kwargs = dict(
        actor_fn=actor_fn, noise_fn=None, observations=observations,
        flow_steps=FLOW_STEPS, act_dim=FULL_ACT_DIM,
        use_constant_noise=True, constant_noise_std=0.01,
        is_encoded=False, params=params,
    )

    print(f"\n  SDE compilation:")
    t0 = time.perf_counter()
    prod_sde_jit = jax.jit(partial(prod_sde, rng=sample_rng, **sde_kwargs))
    _ = prod_sde_jit()
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    print(f"    Python for-loop: {(t1-t0)*1000:.0f} ms")

    t0 = time.perf_counter()
    scan_sde_jit = jax.jit(partial(scan_sample_flow_actions_sde, rng=sample_rng, **sde_kwargs))
    _ = scan_sde_jit()
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    print(f"    jax.lax.scan:    {(t1-t0)*1000:.0f} ms")

    # Log prob
    chains = jax.random.normal(jax.random.PRNGKey(7), (BATCH_SIZE, FLOW_STEPS + 1, FULL_ACT_DIM))
    lp_kwargs = dict(
        actor_fn=actor_fn, noise_fn=None, observations=observations, chains=chains,
        flow_steps=FLOW_STEPS, use_constant_noise=True, constant_noise_std=0.01,
        is_encoded=False, params=params, get_entropy=True,
        clip_intermediate=True, clip_value=1.0, ft_flow_steps=FLOW_STEPS,
    )

    print(f"\n  Log-prob compilation:")
    t0 = time.perf_counter()
    prod_lp_jit = jax.jit(partial(prod_log_prob, **lp_kwargs))
    _ = prod_lp_jit()
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    print(f"    Sequential:  {(t1-t0)*1000:.0f} ms")

    t0 = time.perf_counter()
    batch_lp_jit = jax.jit(partial(batched_compute_flow_log_prob, **lp_kwargs))
    _ = batch_lp_jit()
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    print(f"    Batched:     {(t1-t0)*1000:.0f} ms")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"JAX devices: {jax.devices()}")
    print(f"Config: B={BATCH_SIZE}, obs={OBS_DIM}, act={ACTION_DIM}, "
          f"H={HORIZON}, S={FLOW_STEPS}, hidden={HIDDEN_DIM}, "
          f"layers={NUM_LAYERS}, heads={NUM_HEADS}")

    test_tf_vs_mlp_forward_pass()
    test_batched_log_prob()
    test_scan_sde()
    test_fori_ode()
    test_conditional_denoiser()
    test_compilation_time()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
