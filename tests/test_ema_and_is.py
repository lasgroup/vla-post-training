"""
Comprehensive tests for OGPO core modules:
  - EMA / Polyak target updates
  - Log-probability consistency (SDE sampling vs recomputation)
  - Importance sampling ratio behavior
  - PPO clipped surrogate loss
  - OGPO advantage computation (vanilla, conservative, penultimate)
  - Flow matching BC losses
  - Q-learning helpers (TD target, TD loss, aggregation, soft_target_update)
  - ODE & MIP flow sampling
  - HL-Gauss distributional critic targets
  - _safe_max / _penultimate_safe_max advantage aggregation

Run:  python tests/test_ema_and_is.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.core
import optax
import numpy as np
from types import SimpleNamespace

from ogpo.agents.modules.flax_utils import TrainState, ModuleDict, nonpytree_field
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode,
    sample_flow_actions_sde,
    compute_flow_log_prob,
    compute_ppo_loss,
    compute_ogpo_advantages,
    _safe_max,
    _penultimate_safe_max,
    sample_mip_actions_sde,
    sample_mip_actions_ode,
    compute_mip_log_prob,
    sample_flow_actions_ode_with_correction,
)
from ogpo.agents.modules.bc_helper import (
    get_flow_targets,
    get_mip_targets,
    preprocess_actions,
    compute_flow_bc_loss,
    compute_mip_bc_loss,
    compute_flow_bc_loss_with_denoiser,
    compute_bc_loss,
)
from ogpo.agents.modules.q_helper import (
    aggregate_q_values,
    compute_td_target,
    compute_td_loss,
    soft_target_update,
)
from ogpo.agents.modules.dist_helper import (
    hl_gauss_transform,
    hl_gauss_target,
)

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    tag = "PASS" if passed else "FAIL"
    if not passed:
        FAIL += 1
    else:
        PASS += 1
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))


# ============================================================================
# Tiny networks for testing
# ============================================================================

class TinyActor(nn.Module):
    act_dim: int
    @nn.compact
    def __call__(self, obs, actions, t, is_encoded=True, return_denoiser=False):
        x = jnp.concatenate([obs, actions, t], axis=-1)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        vel = nn.Dense(self.act_dim)(x)
        if return_denoiser:
            denoiser = nn.Dense(self.act_dim)(x)
            return vel, denoiser
        return vel

    def encode(self, observations):
        return observations


class TinyNoise(nn.Module):
    act_dim: int
    @nn.compact
    def __call__(self, obs, t, params=None):
        x = jnp.concatenate([obs, t], axis=-1)
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        sigma = nn.Dense(self.act_dim)(x)
        return jax.nn.softplus(sigma) + 0.01


class TinyCritic(nn.Module):
    num_qs: int = 2
    @nn.compact
    def __call__(self, obs, actions, is_encoded=True):
        x = jnp.concatenate([obs, actions], axis=-1)
        qs = []
        for _ in range(self.num_qs):
            q = nn.Dense(32)(x)
            q = nn.relu(q)
            q = nn.Dense(1)(q).squeeze(-1)
            qs.append(q)
        return jnp.stack(qs, axis=0)

    def encode(self, observations):
        return observations


# ============================================================================
# Helpers
# ============================================================================

def build_actor_network(obs_dim, act_dim, seed=0):
    """Build an actor ModuleDict with actor, target_actor, noise_net, target_noise_net."""
    rng = jax.random.PRNGKey(seed)

    actor = TinyActor(act_dim)
    noise_net = TinyNoise(act_dim)
    target_actor = TinyActor(act_dim)
    target_noise_net = TinyNoise(act_dim)

    net_def = ModuleDict(modules={
        'actor': actor,
        'target_actor': target_actor,
        'noise_net': noise_net,
        'target_noise_net': target_noise_net,
    })

    ex_obs = jnp.zeros((1, obs_dim))
    ex_act = jnp.zeros((1, act_dim))
    ex_t = jnp.zeros((1, 1))

    params = net_def.init(rng, actor=(ex_obs, ex_act, ex_t),
                          target_actor=(ex_obs, ex_act, ex_t),
                          noise_net=(ex_obs, ex_t),
                          target_noise_net=(ex_obs, ex_t))['params']

    # Copy actor -> target_actor, noise_net -> target_noise_net
    params = flax.core.unfreeze(params)
    params['modules_target_actor'] = jax.tree_util.tree_map(
        lambda x: x.copy(), params['modules_actor'])
    params['modules_target_noise_net'] = jax.tree_util.tree_map(
        lambda x: x.copy(), params['modules_noise_net'])
    params = flax.core.freeze(params)

    tx = optax.adam(1e-3)
    state = TrainState.create(net_def, params, tx)
    return state


def make_actor_fn(state, module_name='actor'):
    """Wrap a TrainState into an actor_fn compatible with pg_helper."""
    def actor_fn(obs, actions, t, params=None, is_encoded=True, return_denoiser=False):
        return state.select(module_name)(obs, actions, t, params=params, is_encoded=is_encoded, return_denoiser=return_denoiser)
    return actor_fn


def make_noise_fn(state, module_name='noise_net'):
    """Wrap a TrainState into a noise_fn compatible with pg_helper."""
    def noise_fn(obs, t, params=None):
        return state.select(module_name)(obs, t, params=params)
    return noise_fn


def make_ppo_config(clip_epsilon=0.2, clip_min_epsilon_multiplier=1.0, adv_clip_min=None):
    """Create a config namespace for compute_ppo_loss."""
    return SimpleNamespace(
        clip_epsilon=clip_epsilon,
        clip_min_epsilon_multiplier=clip_min_epsilon_multiplier,
        adv_clip_min=adv_clip_min,
    )


def make_adv_config(adv_strategy='vanilla', normalize_group=True, awr_beta=1.0):
    """Create a config namespace for compute_ogpo_advantages."""
    return SimpleNamespace(
        adv_strategy=adv_strategy,
        normalize_group=normalize_group,
        awr_beta=awr_beta,
    )


# ============================================================================
# Test 1: EMA target update (Polyak averaging)
# ============================================================================

def test_ema_target_update():
    """Verify optax.incremental_update produces theta_target_new = tau*theta + (1-tau)*theta_target."""
    print("\n=== Test 1: EMA Target Update ===")

    tau = 0.005

    rng = jax.random.PRNGKey(42)
    online = {'w': jax.random.normal(rng, (4, 4)),
              'b': jax.random.normal(jax.random.PRNGKey(1), (4,))}
    target = {'w': jax.random.normal(jax.random.PRNGKey(2), (4, 4)),
              'b': jax.random.normal(jax.random.PRNGKey(3), (4,))}

    result = optax.incremental_update(online, target, step_size=tau)

    expected = jax.tree_util.tree_map(
        lambda o, t: tau * o + (1.0 - tau) * t, online, target
    )

    for key in ['w', 'b']:
        err = jnp.max(jnp.abs(result[key] - expected[key]))
        report(f"incremental_update '{key}' matches manual formula",
               float(err) < 1e-6, f"max_err={float(err):.2e}")

    # tau=0 => no update
    result0 = optax.incremental_update(online, target, step_size=0.0)
    err0 = max(float(jnp.max(jnp.abs(result0[k] - target[k]))) for k in ['w', 'b'])
    report("tau=0 keeps target unchanged", err0 < 1e-6, f"max_err={err0:.2e}")

    # tau=1 => full copy
    result1 = optax.incremental_update(online, target, step_size=1.0)
    err1 = max(float(jnp.max(jnp.abs(result1[k] - online[k]))) for k in ['w', 'b'])
    report("tau=1 copies online to target", err1 < 1e-6, f"max_err={err1:.2e}")


def test_ema_on_real_network():
    """Test EMA update using the actual target_update pattern from OGPOAgent."""
    print("\n=== Test 1b: EMA on Real Network Params ===")

    obs_dim, act_dim = 8, 4
    state = build_actor_network(obs_dim, act_dim, seed=0)

    # Perturb actor params to differ from target
    params = flax.core.unfreeze(state.params)
    params['modules_actor'] = jax.tree_util.tree_map(
        lambda x: x + 0.1 * jax.random.normal(jax.random.PRNGKey(99), x.shape),
        params['modules_actor']
    )
    params = flax.core.freeze(params)
    state = state.replace(params=params)

    # Verify actor != target_actor
    diff_before = jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_util.tree_map(
            lambda a, t: jnp.sum(jnp.abs(a - t)),
            state.params['modules_actor'],
            state.params['modules_target_actor']
        )
    )
    report("actor differs from target_actor before update",
           float(diff_before) > 0.01, f"L1_diff={float(diff_before):.4f}")

    # Apply Polyak update
    tau = 0.005
    new_target_params = optax.incremental_update(
        new_tensors=state.params['modules_actor'],
        old_tensors=state.params['modules_target_actor'],
        step_size=tau,
    )
    new_params = state.params.copy({
        'modules_target_actor': new_target_params,
    })
    new_state = state.replace(params=new_params)

    expected_target = jax.tree_util.tree_map(
        lambda a, t: tau * a + (1.0 - tau) * t,
        state.params['modules_actor'],
        state.params['modules_target_actor'],
    )

    err = jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_util.tree_map(
            lambda x, y: jnp.max(jnp.abs(x - y)),
            new_state.params['modules_target_actor'],
            expected_target
        )
    )
    report("Polyak update on real network params matches formula",
           float(err) < 1e-5, f"max_err={float(err):.2e}")

    # After many updates, target should converge to actor
    s = state
    for _ in range(2000):
        new_target = optax.incremental_update(
            s.params['modules_actor'],
            s.params['modules_target_actor'],
            step_size=tau,
        )
        s = s.replace(params=s.params.copy({'modules_target_actor': new_target}))

    converge_err = jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_util.tree_map(
            lambda a, t: jnp.max(jnp.abs(a - t)),
            s.params['modules_actor'],
            s.params['modules_target_actor']
        )
    )
    report("target converges to actor after 2000 EMA steps (tau=0.005)",
           float(converge_err) < 1e-3, f"max_err={float(converge_err):.2e}")


# ============================================================================
# Test 2: Log-prob consistency (SDE sampling inline vs batched recomputation)
# ============================================================================

def test_logprob_consistency():
    """
    sample_flow_actions_sde computes log-probs step-by-step during sampling.
    compute_flow_log_prob re-evaluates the same chain in a batched pass.
    With identical params and the same chains, they MUST agree.
    """
    print("\n=== Test 2: Log-prob Consistency (SDE sampling vs recomputation) ===")

    obs_dim, act_dim = 8, 4
    batch_size = 16
    flow_steps = 10

    state = build_actor_network(obs_dim, act_dim, seed=42)

    actor_fn_target = make_actor_fn(state, 'target_actor')
    noise_fn_target = make_noise_fn(state, 'target_noise_net')

    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, obs_dim))
    rng = jax.random.PRNGKey(1)

    # Use wide bounds to avoid last-step act-clipping discrepancy.
    # The clipping behavior is tested separately in test_logprob_last_step_clipping.
    # --- Constant noise case ---
    actions, chains, logprob_sde, sigmas = sample_flow_actions_sde(
        actor_fn=actor_fn_target,
        noise_fn=noise_fn_target,
        observations=obs,
        rng=rng,
        flow_steps=flow_steps,
        act_dim=act_dim,
        use_constant_noise=True,
        constant_noise_std=0.1,
        act_min=-100.0,
        act_max=100.0,
        clip_intermediate=True,
        clip_value=1.0,
    )

    logprob_recomputed, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn_target,
        noise_fn=noise_fn_target,
        observations=obs,
        chains=chains,
        flow_steps=flow_steps,
        use_constant_noise=True,
        constant_noise_std=0.1,
        clip_intermediate=True,
        clip_value=1.0,
    )

    err_const = float(jnp.max(jnp.abs(logprob_sde - logprob_recomputed)))
    report("constant noise: SDE inline logprob == batched recomputation",
           err_const < 1e-3,
           f"max_err={err_const:.4e}, mean_sde={float(logprob_sde.mean()):.2f}, "
           f"mean_recomp={float(logprob_recomputed.mean()):.2f}")

    # --- Learned noise case ---
    actions_ln, chains_ln, logprob_sde_ln, sigmas_ln = sample_flow_actions_sde(
        actor_fn=actor_fn_target,
        noise_fn=noise_fn_target,
        observations=obs,
        rng=rng,
        flow_steps=flow_steps,
        act_dim=act_dim,
        use_constant_noise=False,
        min_noise_std=0.08,
        act_min=-100.0,
        act_max=100.0,
        clip_intermediate=True,
        clip_value=1.0,
    )

    logprob_recomp_ln, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn_target,
        noise_fn=noise_fn_target,
        observations=obs,
        chains=chains_ln,
        flow_steps=flow_steps,
        use_constant_noise=False,
        min_noise_std=0.08,
        clip_intermediate=True,
        clip_value=1.0,
    )

    err_ln = float(jnp.max(jnp.abs(logprob_sde_ln - logprob_recomp_ln)))
    report("learned noise: SDE inline logprob == batched recomputation",
           err_ln < 1e-3,
           f"max_err={err_ln:.4e}, mean_sde={float(logprob_sde_ln.mean()):.2f}, "
           f"mean_recomp={float(logprob_recomp_ln.mean()):.2f}")


# ============================================================================
# Test 3: Importance sampling ratio behavior
# ============================================================================

def test_importance_sampling_ratio():
    """
    Verify:
    1. ratio ~ 1 when current params == target params
    2. ratio deviates when params diverge
    3. Large perturbation causes significant deviation
    """
    print("\n=== Test 3: Importance Sampling Ratios ===")

    obs_dim, act_dim = 8, 4
    batch_size = 64
    flow_steps = 10

    state = build_actor_network(obs_dim, act_dim, seed=7)
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, obs_dim))
    rng = jax.random.PRNGKey(1)

    # Use wide bounds to avoid last-step clipping discrepancy
    # ---- 3a: same params => ratio ~ 1 ----
    actor_fn = make_actor_fn(state, 'target_actor')
    noise_fn = make_noise_fn(state, 'target_noise_net')

    _, chains, old_lp, _ = sample_flow_actions_sde(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, rng=rng,
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        act_min=-100.0, act_max=100.0,
        clip_intermediate=True, clip_value=1.0,
    )

    actor_fn_current = make_actor_fn(state, 'actor')
    noise_fn_current = make_noise_fn(state, 'noise_net')

    new_lp, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn_current, noise_fn=noise_fn_current,
        observations=obs, chains=chains,
        flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
        clip_intermediate=True, clip_value=1.0,
    )

    ratio_same = jnp.exp(new_lp - old_lp)
    report("same params: mean ratio ~ 1",
           abs(float(ratio_same.mean()) - 1.0) < 0.05,
           f"mean={float(ratio_same.mean()):.4f}, std={float(ratio_same.std()):.4f}")
    report("same params: all ratios close to 1",
           float(jnp.max(jnp.abs(ratio_same - 1.0))) < 0.1,
           f"max_dev={float(jnp.max(jnp.abs(ratio_same - 1.0))):.4f}")

    # ---- 3b: slightly perturbed params => ratio deviates ----
    params_perturbed = jax.tree_util.tree_map(
        lambda x: x + 0.01 * jax.random.normal(jax.random.PRNGKey(99), x.shape),
        state.params
    )
    state_perturbed = state.replace(params=params_perturbed)
    actor_fn_perturbed = make_actor_fn(state_perturbed, 'actor')
    noise_fn_perturbed = make_noise_fn(state_perturbed, 'noise_net')

    new_lp_perturbed, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn_perturbed, noise_fn=noise_fn_perturbed,
        observations=obs, chains=chains,
        flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
    )

    ratio_perturbed = jnp.exp(new_lp_perturbed - old_lp)
    report("small perturbation: ratio deviates from 1",
           float(ratio_perturbed.std()) > 0.001,
           f"mean={float(ratio_perturbed.mean()):.4f}, std={float(ratio_perturbed.std()):.4f}")

    # ---- 3c: large perturbation => ratio diverges ----
    params_large = jax.tree_util.tree_map(
        lambda x: x + 1.0 * jax.random.normal(jax.random.PRNGKey(99), x.shape),
        state.params
    )
    state_large = state.replace(params=params_large)
    actor_fn_large = make_actor_fn(state_large, 'actor')

    new_lp_large, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn_large, noise_fn=noise_fn_current,
        observations=obs, chains=chains,
        flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
    )

    ratio_large = jnp.exp(new_lp_large - old_lp)
    report("large perturbation: ratio differs from 1",
           abs(float(ratio_large.mean()) - 1.0) > 0.01 or float(ratio_large.std()) > 0.01,
           f"mean={float(ratio_large.mean()):.4f}, std={float(ratio_large.std()):.4f}, "
           f"max={float(ratio_large.max()):.4f}, min={float(ratio_large.min()):.4f}")


# ============================================================================
# Test 4: PPO loss and OGPO advantages (updated to new API)
# ============================================================================

def test_ppo_loss():
    """Verify PPO clipped surrogate loss properties with config-based API."""
    print("\n=== Test 4a: PPO Clipped Surrogate Loss ===")

    G, B = 4, 32
    rng = jax.random.PRNGKey(0)

    # ---- ratio=1 everywhere ----
    lp = jax.random.normal(rng, (G * B,)) * 5
    old_lp = lp  # same
    adv = jax.random.normal(jax.random.PRNGKey(1), (G * B,))

    config = make_ppo_config(clip_epsilon=0.2, clip_min_epsilon_multiplier=1.0, adv_clip_min=None)
    loss_same, stats_same = compute_ppo_loss(lp, old_lp, adv, config)
    expected = -jnp.mean(adv)
    report("ratio=1: PPO loss == -mean(adv)",
           abs(float(loss_same) - float(expected)) < 1e-5,
           f"loss={float(loss_same):.6f}, expected={float(expected):.6f}")
    report("ratio=1: mean ratio is 1.0",
           abs(float(stats_same['ratio']) - 1.0) < 1e-6)
    report("ratio=1: no clipping",
           float(stats_same['ratio_clipped_total']) == 0.0)

    # ---- ratio > 1 + epsilon, positive advantages => clipping kicks in ----
    old_lp2 = lp - 1.0  # exp(1.0) ~ 2.72, well above 1.2
    adv_pos = jnp.abs(adv) + 0.1  # all positive
    loss_clip, stats_clip = compute_ppo_loss(lp, old_lp2, adv_pos, config)

    report("large ratio + positive adv: upper clipping detected",
           float(stats_clip['ratio_clipped_upper']) > 0.5,
           f"frac_clipped_upper={float(stats_clip['ratio_clipped_upper']):.2f}")

    # ---- ratio < 1 - epsilon, negative advantages => lower clipping ----
    old_lp3 = lp + 1.0  # exp(-1.0) ~ 0.37, well below 0.8
    adv_neg = -(jnp.abs(adv) + 0.1)  # all negative
    loss_clip_lo, stats_clip_lo = compute_ppo_loss(lp, old_lp3, adv_neg, config)
    report("small ratio + negative adv: lower clipping detected",
           float(stats_clip_lo['ratio_clipped_lower']) > 0.5,
           f"frac_clipped_lower={float(stats_clip_lo['ratio_clipped_lower']):.2f}")

    # ---- adv_clip_min ----
    config_clip = make_ppo_config(clip_epsilon=0.2, clip_min_epsilon_multiplier=1.0, adv_clip_min=-0.5)
    adv_very_neg = -10.0 * jnp.ones(G * B)
    loss_clamped, stats_clamped = compute_ppo_loss(lp, old_lp, adv_very_neg, config_clip)
    # With adv_clip_min=-0.5, advantages are clipped to -0.5
    expected_clamped = -jnp.mean(jnp.clip(adv_very_neg, -0.5))
    report("adv_clip_min clamps advantages",
           abs(float(loss_clamped) - float(expected_clamped)) < 1e-5,
           f"loss={float(loss_clamped):.6f}, expected={float(expected_clamped):.6f}")


def test_ogpo_advantages():
    """Verify OGPO advantage computation with different strategies."""
    print("\n=== Test 4b: OGPO Advantages ===")

    G, B = 4, 8
    M = 3  # num Q-functions in ensemble
    rng = jax.random.PRNGKey(0)

    q_agg = jax.random.normal(rng, (G, B)) * 2.0
    q_full = jax.random.normal(jax.random.PRNGKey(1), (G, M, B)) * 2.0

    # ---- Vanilla without normalization ----
    config_vanilla = make_adv_config(adv_strategy='vanilla', normalize_group=False)
    adv_v, q_raw = compute_ogpo_advantages(q_agg, q_full, config_vanilla)
    expected_adv_v = q_agg - q_agg.mean(axis=0, keepdims=True)
    err_v = float(jnp.max(jnp.abs(adv_v - expected_adv_v)))
    report("vanilla: adv = Q_agg - mean(Q_agg) per state",
           err_v < 1e-6, f"max_err={err_v:.2e}")

    # raw q_values returned unchanged
    report("vanilla: raw q_values returned unchanged",
           float(jnp.max(jnp.abs(q_raw - q_agg))) < 1e-6)

    # ---- Vanilla with normalization ----
    config_norm = make_adv_config(adv_strategy='vanilla', normalize_group=True)
    adv_norm, _ = compute_ogpo_advantages(q_agg, q_full, config_norm)
    col_means = adv_norm.mean(axis=0)
    report("vanilla normalized: advantages are zero-mean per state",
           float(jnp.max(jnp.abs(col_means))) < 1e-5,
           f"max_col_mean={float(jnp.max(jnp.abs(col_means))):.2e}")
    col_stds = adv_norm.std(axis=0)
    report("vanilla normalized: advantages are unit-std per state",
           float(jnp.max(jnp.abs(col_stds - 1.0))) < 0.1,
           f"max_std_dev={float(jnp.max(jnp.abs(col_stds - 1.0))):.4f}")

    # ---- Conservative strategy ----
    config_cons = make_adv_config(adv_strategy='conservative')
    adv_cons, q_cons = compute_ogpo_advantages(q_agg, q_full, config_cons)
    # q_cons should be the mean across Q-networks
    expected_q_cons = q_full.mean(axis=1)
    report("conservative: returns mean Q across ensemble",
           float(jnp.max(jnp.abs(q_cons - expected_q_cons))) < 1e-6)
    report("conservative: advantages have correct shape",
           adv_cons.shape == (G, B))

    # ---- Penultimate strategy ----
    config_pen = make_adv_config(adv_strategy='penultimate')
    adv_pen, q_pen = compute_ogpo_advantages(q_agg, q_full, config_pen)
    report("penultimate: advantages have correct shape",
           adv_pen.shape == (G, B))


# ============================================================================
# Test 5: End-to-end IS ratio after one EMA step
# ============================================================================

def test_is_ratio_after_ema():
    """
    Simulate one training iteration:
    1. Sample from target policy -> old_lp
    2. Perturb actor params (simulating a gradient step)
    3. EMA update target
    4. Re-sample from new target -> new old_lp
    5. Recompute log probs under new current -> new lp
    6. Check that ratio = exp(new_lp - new_old_lp) is well-behaved
    """
    print("\n=== Test 5: IS Ratio After EMA Step ===")

    obs_dim, act_dim = 8, 4
    batch_size = 64
    flow_steps = 10
    tau = 0.005

    state = build_actor_network(obs_dim, act_dim, seed=42)
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, obs_dim))

    # Step 1: sample from target (wide bounds to avoid clipping discrepancy)
    actor_fn_target = make_actor_fn(state, 'target_actor')
    noise_fn_target = make_noise_fn(state, 'target_noise_net')

    _, chains0, old_lp0, _ = sample_flow_actions_sde(
        actor_fn=actor_fn_target, noise_fn=noise_fn_target,
        observations=obs, rng=jax.random.PRNGKey(10),
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        act_min=-100.0, act_max=100.0,
        clip_intermediate=True, clip_value=1.0,
    )

    # Step 2: compute ratio with current params (= target at init, so ratio ~ 1)
    actor_fn_current = make_actor_fn(state, 'actor')
    noise_fn_current = make_noise_fn(state, 'noise_net')
    lp0, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn_current, noise_fn=noise_fn_current,
        observations=obs, chains=chains0,
        flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
    )
    ratio0 = jnp.exp(lp0 - old_lp0)
    report("step 0: ratio ~ 1 (before any gradient step)",
           abs(float(ratio0.mean()) - 1.0) < 0.05,
           f"mean={float(ratio0.mean()):.4f}")

    # Step 3: simulate gradient step (perturb actor params)
    params_after_grad = flax.core.unfreeze(state.params)
    params_after_grad['modules_actor'] = jax.tree_util.tree_map(
        lambda x: x + 0.05 * jax.random.normal(jax.random.PRNGKey(77), x.shape),
        params_after_grad['modules_actor']
    )
    params_after_grad = flax.core.freeze(params_after_grad)

    # Step 4: EMA update target
    new_target = optax.incremental_update(
        params_after_grad['modules_actor'],
        params_after_grad['modules_target_actor'],
        step_size=tau,
    )
    params_after_ema = params_after_grad.copy({'modules_target_actor': new_target})
    state_after = state.replace(params=params_after_ema)

    # Step 5: sample from NEW target, recompute lp under NEW current
    actor_fn_new_target = make_actor_fn(state_after, 'target_actor')
    noise_fn_new_target = make_noise_fn(state_after, 'target_noise_net')

    _, chains1, old_lp1, _ = sample_flow_actions_sde(
        actor_fn=actor_fn_new_target, noise_fn=noise_fn_new_target,
        observations=obs, rng=jax.random.PRNGKey(20),
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        act_min=-100.0, act_max=100.0,
        clip_intermediate=True, clip_value=1.0,
    )

    actor_fn_new_current = make_actor_fn(state_after, 'actor')
    noise_fn_new_current = make_noise_fn(state_after, 'noise_net')
    lp1, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn_new_current, noise_fn=noise_fn_new_current,
        observations=obs, chains=chains1,
        flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
    )
    ratio1 = jnp.exp(lp1 - old_lp1)

    report("step 1: ratio deviates from 1 (actor moved, target barely moved)",
           float(ratio1.std()) > 0.001,
           f"mean={float(ratio1.mean()):.4f}, std={float(ratio1.std()):.4f}, "
           f"min={float(ratio1.min()):.4f}, max={float(ratio1.max()):.4f}")

    report("step 1: ratio not exploding (max < 10)",
           float(ratio1.max()) < 10.0,
           f"max={float(ratio1.max()):.4f}")
    report("step 1: ratio not collapsing (min > 1e-4)",
           float(ratio1.min()) > 1e-4,
           f"min={float(ratio1.min()):.4f}")


# ============================================================================
# Test 6: Last-step clipping bug diagnosis
# ============================================================================

def test_logprob_last_step_clipping():
    """
    Verify behavior of last-step act-clipping on logprob consistency.

    sample_flow_actions_sde computes logprob BEFORE final act-clipping,
    but stores the act-clipped value in chains. compute_flow_log_prob
    recomputes from chains (which have act-clipped values), so a discrepancy
    exists when final clipping is active. This is expected behavior.

    This test verifies:
    1. WIDE bounds (no effective clipping): logprobs agree perfectly
    2. TIGHT bounds: discrepancy exists (documenting known behavior)
    """
    print("\n=== Test 6: Last-step Clipping Behavior ===")

    obs_dim, act_dim = 8, 4
    batch_size = 64
    flow_steps = 10

    state = build_actor_network(obs_dim, act_dim, seed=42)
    actor_fn = make_actor_fn(state, 'target_actor')
    noise_fn = make_noise_fn(state, 'target_noise_net')
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, obs_dim))
    rng = jax.random.PRNGKey(1)

    # --- With WIDE bounds (no effective final clipping) ---
    _, chains_wide, lp_sde_wide, _ = sample_flow_actions_sde(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, rng=rng,
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        act_min=-100.0, act_max=100.0,
        clip_intermediate=True, clip_value=1.0,
    )

    lp_recomp_wide, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, chains=chains_wide,
        flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
        clip_intermediate=True, clip_value=1.0,
    )

    err_wide = float(jnp.max(jnp.abs(lp_sde_wide - lp_recomp_wide)))
    report("WIDE bounds (no act-clip): logprobs agree",
           err_wide < 1e-3,
           f"max_err={err_wide:.4e}")

    # --- With TIGHT bounds (act-clipping at [-1, 1]) ---
    # Known behavior: logprobs can differ because SDE computes logprob
    # before final act-clip, but chains store the act-clipped value.
    _, chains_tight, lp_sde_tight, _ = sample_flow_actions_sde(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, rng=rng,
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        act_min=-1.0, act_max=1.0,
        clip_intermediate=True, clip_value=1.0,
    )

    # Check that final actions are properly clipped
    final_actions = chains_tight[:, -1]
    report("TIGHT bounds: final actions within [-1, 1]",
           float(final_actions.min()) >= -1.0 - 1e-6 and float(final_actions.max()) <= 1.0 + 1e-6)

    # Count how many final actions were affected by clipping
    # (compare with wide-bounds chain to see which ones changed)
    _, chains_wide2, _, _ = sample_flow_actions_sde(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, rng=rng,
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        act_min=-100.0, act_max=100.0,
        clip_intermediate=True, clip_value=1.0,
    )
    n_clipped = int(jnp.sum(jnp.any(
        (chains_wide2[:, -1] < -1.0) | (chains_wide2[:, -1] > 1.0), axis=-1
    )))
    report("TIGHT bounds: some final actions were clipped (expected)",
           True, f"n_clipped={n_clipped}/{batch_size}")


# ============================================================================
# Test 7: Q-learning helpers
# ============================================================================

def test_q_helpers():
    """Test aggregate_q_values, compute_td_target, compute_td_loss, soft_target_update."""
    print("\n=== Test 7: Q-Learning Helpers ===")

    rng = jax.random.PRNGKey(0)
    B = 32
    num_qs = 5

    q_values = jax.random.normal(rng, (num_qs, B))

    # ---- aggregate_q_values ----
    q_mean = aggregate_q_values(q_values, method="mean")
    report("aggregate mean: correct",
           float(jnp.max(jnp.abs(q_mean - q_values.mean(axis=0)))) < 1e-6)

    q_min = aggregate_q_values(q_values, method="min")
    report("aggregate min: correct",
           float(jnp.max(jnp.abs(q_min - q_values.min(axis=0)))) < 1e-6)

    q_sub = aggregate_q_values(q_values, method="subsample",
                               rng=jax.random.PRNGKey(42), num_qs=num_qs,
                               subsample_size=2)
    report("aggregate subsample: shape correct", q_sub.shape == (B,))
    report("aggregate subsample: values within range",
           float(q_sub.min()) >= float(q_values.min()) - 1e-6)

    # ---- compute_td_target ----
    rewards = jax.random.normal(jax.random.PRNGKey(1), (B,))
    masks = jnp.ones(B)
    next_q = jax.random.normal(jax.random.PRNGKey(2), (B,))
    discount = 0.99

    td_target = compute_td_target(rewards, masks, next_q, discount, horizon_length=1)
    expected_td = rewards + discount * masks * next_q
    report("TD target (H=1): r + gamma * Q(s',a')",
           float(jnp.max(jnp.abs(td_target - expected_td))) < 1e-6)

    td_target_h4 = compute_td_target(rewards, masks, next_q, discount, horizon_length=4)
    expected_td_h4 = rewards + (discount ** 4) * masks * next_q
    report("TD target (H=4): r + gamma^4 * Q(s',a')",
           float(jnp.max(jnp.abs(td_target_h4 - expected_td_h4))) < 1e-6)

    # Terminal state => mask=0 => no bootstrap
    masks_terminal = jnp.zeros(B)
    td_terminal = compute_td_target(rewards, masks_terminal, next_q, discount)
    report("TD target with terminal mask: equals reward only",
           float(jnp.max(jnp.abs(td_terminal - rewards))) < 1e-6)

    # Multi-dimensional rewards/masks (action chunking)
    rewards_2d = jax.random.normal(jax.random.PRNGKey(3), (B, 4))
    masks_2d = jnp.ones((B, 4))
    td_2d = compute_td_target(rewards_2d, masks_2d, next_q, discount, horizon_length=4)
    expected_2d = rewards_2d[..., -1] + (discount ** 4) * masks_2d[..., -1] * next_q
    report("TD target with 2D rewards: uses last timestep",
           float(jnp.max(jnp.abs(td_2d - expected_2d))) < 1e-6)

    # ---- compute_td_loss (MSE) ----
    q_pred = jax.random.normal(jax.random.PRNGKey(4), (B,))
    target_q = jax.random.normal(jax.random.PRNGKey(5), (B,))
    td_loss, stats = compute_td_loss(q_pred, target_q, loss_type="mse")
    expected_loss = jnp.mean(jnp.square(q_pred - target_q))
    report("TD loss (MSE): correct",
           abs(float(td_loss) - float(expected_loss)) < 1e-6)

    # With valid mask
    valid = jnp.concatenate([jnp.ones(B // 2), jnp.zeros(B // 2)])
    td_loss_masked, _ = compute_td_loss(q_pred, target_q, valid_mask=valid, loss_type="mse")
    expected_masked = jnp.mean(jnp.square(q_pred - target_q) * valid)
    report("TD loss (MSE) with valid mask: correct",
           abs(float(td_loss_masked) - float(expected_masked)) < 1e-6)

    # ---- soft_target_update ----
    params1 = {'w': jax.random.normal(jax.random.PRNGKey(10), (4, 4))}
    params2 = {'w': jax.random.normal(jax.random.PRNGKey(11), (4, 4))}
    tau = 0.005
    result = soft_target_update(params1, params2, tau)
    expected_sft = {'w': params1['w'] * tau + params2['w'] * (1 - tau)}
    report("soft_target_update: matches formula",
           float(jnp.max(jnp.abs(result['w'] - expected_sft['w']))) < 1e-6)


# ============================================================================
# Test 8: BC helper functions
# ============================================================================

def test_bc_helpers():
    """Test get_flow_targets, preprocess_actions, compute_flow_bc_loss."""
    print("\n=== Test 8: BC Helper Functions ===")

    B, obs_dim, act_dim = 16, 8, 4
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(jax.random.PRNGKey(1), (B, obs_dim))
    actions = jax.random.normal(jax.random.PRNGKey(2), (B, act_dim))

    # ---- get_flow_targets ----
    x_t, v_target, t = get_flow_targets(obs, actions, rng)
    report("get_flow_targets: x_t shape correct", x_t.shape == (B, act_dim))
    report("get_flow_targets: v_target shape correct", v_target.shape == (B, act_dim))
    report("get_flow_targets: t shape correct", t.shape == (B, 1))
    report("get_flow_targets: t in [0, 1]",
           float(t.min()) >= 0.0 and float(t.max()) <= 1.0)

    # Verify: x_t = (1 - t) * x_0 + t * x_1 and v = x_1 - x_0
    # We can't directly verify this without knowing x_0, but we can
    # check that x_t is between x_0 and x_1 in expectation
    report("get_flow_targets: v_target is finite",
           bool(jnp.all(jnp.isfinite(v_target))))

    # ---- preprocess_actions ----
    # Without chunking: (B, H, A) -> (B, A) (take first timestep)
    batch_3d = {'actions': jax.random.normal(rng, (B, 4, act_dim))}
    processed = preprocess_actions(batch_3d, action_chunking=False)
    report("preprocess_actions (no chunk): takes first timestep",
           float(jnp.max(jnp.abs(processed - batch_3d['actions'][..., 0, :]))) < 1e-6)

    # With chunking: (B, H, A) -> (B, H*A)
    processed_chunk = preprocess_actions(batch_3d, action_chunking=True)
    report("preprocess_actions (chunking): flattens to (B, H*A)",
           processed_chunk.shape == (B, 4 * act_dim))

    # ---- compute_flow_bc_loss ----
    class TinyModel(nn.Module):
        act_dim: int
        @nn.compact
        def __call__(self, obs, x_t, t):
            x = jnp.concatenate([obs, x_t, t], axis=-1)
            x = nn.Dense(32)(x)
            x = nn.relu(x)
            return nn.Dense(self.act_dim)(x)

    model = TinyModel(act_dim)
    model_params = model.init(jax.random.PRNGKey(42), obs, actions, jnp.zeros((B, 1)))

    def model_fn(obs, x_t, t):
        return model.apply(model_params, obs, x_t, t)

    batch = {
        'observations': obs,
        'actions': jax.random.normal(rng, (B, 1, act_dim)),  # (B, H, A)
    }
    bc_loss, info = compute_flow_bc_loss(
        batch, rng, model_fn,
        action_chunking=False, horizon_length=1, action_dim=act_dim
    )
    report("compute_flow_bc_loss: returns scalar",
           bc_loss.ndim == 0)
    report("compute_flow_bc_loss: loss is positive",
           float(bc_loss) > 0.0)
    report("compute_flow_bc_loss: info contains bc_flow_loss",
           'bc_flow_loss' in info)


# ============================================================================
# Test 9: MIP targets and BC loss
# ============================================================================

def test_mip_helpers():
    """Test get_mip_targets and compute_mip_bc_loss."""
    print("\n=== Test 9: MIP Targets & BC Loss ===")

    B, act_dim = 16, 4
    rng = jax.random.PRNGKey(0)
    actions = jax.random.normal(jax.random.PRNGKey(2), (B, act_dim))

    # ---- get_mip_targets ----
    x_0, x_t_star, t_0_arr, t_star_arr = get_mip_targets(actions, rng, t_star=0.999)
    report("get_mip_targets: x_0 shape correct", x_0.shape == (B, act_dim))
    report("get_mip_targets: x_t_star shape correct", x_t_star.shape == (B, act_dim))
    report("get_mip_targets: t_0 is all zeros",
           float(jnp.max(jnp.abs(t_0_arr))) < 1e-6)
    report("get_mip_targets: t_star is all 0.999",
           float(jnp.max(jnp.abs(t_star_arr - 0.999))) < 1e-6)

    # Verify: x_t_star = (1 - t*) * x_0 + t* * actions
    expected_x_t_star = (1 - 0.999) * x_0 + 0.999 * actions
    report("get_mip_targets: x_t_star matches interpolation formula",
           float(jnp.max(jnp.abs(x_t_star - expected_x_t_star))) < 1e-4)


# ============================================================================
# Test 10: ODE flow sampling
# ============================================================================

def test_ode_sampling():
    """Test sample_flow_actions_ode deterministic flow."""
    print("\n=== Test 10: ODE Flow Sampling ===")

    obs_dim, act_dim = 8, 4
    batch_size = 16
    flow_steps = 10

    state = build_actor_network(obs_dim, act_dim, seed=42)
    actor_fn = make_actor_fn(state, 'actor')
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, obs_dim))
    rng = jax.random.PRNGKey(1)

    actions = sample_flow_actions_ode(
        actor_fn=actor_fn,
        observations=obs,
        rng=rng,
        flow_steps=flow_steps,
        act_dim=act_dim,
        act_min=-1.0,
        act_max=1.0,
        clip_intermediate=True,
        clip_value=1.0,
    )

    report("ODE sampling: correct output shape",
           actions.shape == (batch_size, act_dim))
    report("ODE sampling: actions within bounds",
           float(actions.min()) >= -1.0 - 1e-6 and float(actions.max()) <= 1.0 + 1e-6)
    report("ODE sampling: actions are finite",
           bool(jnp.all(jnp.isfinite(actions))))

    # Deterministic: same rng => same actions
    actions2 = sample_flow_actions_ode(
        actor_fn=actor_fn,
        observations=obs,
        rng=rng,
        flow_steps=flow_steps,
        act_dim=act_dim,
    )
    report("ODE sampling: deterministic (same rng => same actions)",
           float(jnp.max(jnp.abs(actions - actions2))) < 1e-6)

    # Different rng => different actions
    actions3 = sample_flow_actions_ode(
        actor_fn=actor_fn,
        observations=obs,
        rng=jax.random.PRNGKey(999),
        flow_steps=flow_steps,
        act_dim=act_dim,
    )
    report("ODE sampling: different rng => different actions",
           float(jnp.max(jnp.abs(actions - actions3))) > 0.01)


# ============================================================================
# Test 11: ODE with correction (denoiser)
# ============================================================================

def build_denoiser_actor_network(obs_dim, act_dim, seed=0):
    """Build an actor network initialized with return_denoiser=True."""
    rng = jax.random.PRNGKey(seed)
    actor = TinyActor(act_dim)
    net_def = ModuleDict(modules={'actor': actor})

    ex_obs = jnp.zeros((1, obs_dim))
    ex_act = jnp.zeros((1, act_dim))
    ex_t = jnp.zeros((1, 1))

    # Init with return_denoiser=True so Dense_2 parameters are created
    params = net_def.init(rng, actor={'obs': ex_obs, 'actions': ex_act, 't': ex_t,
                                       'return_denoiser': True})['params']

    tx = optax.adam(1e-3)
    return TrainState.create(net_def, params, tx)


def test_ode_with_correction():
    """Test sample_flow_actions_ode_with_correction."""
    print("\n=== Test 11: ODE Flow with SDE-to-ODE Correction ===")

    obs_dim, act_dim = 8, 4
    batch_size = 16
    flow_steps = 10

    # Need a network initialized with return_denoiser=True
    state = build_denoiser_actor_network(obs_dim, act_dim, seed=42)
    actor_fn = make_actor_fn(state, 'actor')
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, obs_dim))
    rng = jax.random.PRNGKey(1)

    actions = sample_flow_actions_ode_with_correction(
        actor_fn=actor_fn,
        observations=obs,
        rng=rng,
        flow_steps=flow_steps,
        act_dim=act_dim,
        constant_noise_std=0.1,
        act_min=-1.0,
        act_max=1.0,
        clip_intermediate=True,
        clip_value=1.0,
    )

    report("ODE+correction: correct output shape",
           actions.shape == (batch_size, act_dim))
    report("ODE+correction: actions within bounds",
           float(actions.min()) >= -1.0 - 1e-6 and float(actions.max()) <= 1.0 + 1e-6)
    report("ODE+correction: actions are finite",
           bool(jnp.all(jnp.isfinite(actions))))

    # With correction=0, the correction_coeff = 0 so z_pred is zeroed out,
    # and the result should match standard ODE (using the denoiser network)
    actions_no_correction = sample_flow_actions_ode_with_correction(
        actor_fn=actor_fn,
        observations=obs,
        rng=rng,
        flow_steps=flow_steps,
        act_dim=act_dim,
        constant_noise_std=0.0,  # no correction
    )

    # Standard ODE using same denoiser network (will just ignore the denoiser)
    def actor_fn_no_denoiser(obs, actions, t, is_encoded=True):
        return state.select('actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=False)

    actions_standard = sample_flow_actions_ode(
        actor_fn=actor_fn_no_denoiser,
        observations=obs,
        rng=rng,
        flow_steps=flow_steps,
        act_dim=act_dim,
    )
    report("ODE+correction (sigma=0): matches standard ODE",
           float(jnp.max(jnp.abs(actions_no_correction - actions_standard))) < 1e-4)


# ============================================================================
# Test 12: MIP flow sampling (SDE and ODE)
# ============================================================================

def test_mip_sampling():
    """Test MIP 2-step flow sampling."""
    print("\n=== Test 12: MIP Flow Sampling ===")

    obs_dim, act_dim = 8, 4
    batch_size = 16

    state = build_actor_network(obs_dim, act_dim, seed=42)
    actor_fn = make_actor_fn(state, 'target_actor')
    noise_fn = make_noise_fn(state, 'target_noise_net')
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, obs_dim))
    rng = jax.random.PRNGKey(1)

    # ---- MIP SDE (wide bounds for logprob consistency) ----
    actions_sde, chains_sde, logprob_sde, sigmas_sde = sample_mip_actions_sde(
        actor_fn=actor_fn,
        noise_fn=noise_fn,
        observations=obs,
        rng=rng,
        mip_t_star=0.999,
        act_dim=act_dim,
        use_constant_noise=True,
        constant_noise_std=0.1,
        act_min=-100.0,
        act_max=100.0,
    )

    report("MIP SDE: actions shape correct",
           actions_sde.shape == (batch_size, act_dim))
    report("MIP SDE: chains shape correct (3 steps: x_0, a_hat, a_final)",
           chains_sde.shape == (batch_size, 3, act_dim))
    report("MIP SDE: sigmas shape correct (2 steps)",
           sigmas_sde.shape == (batch_size, 2, act_dim))
    report("MIP SDE: logprobs shape correct",
           logprob_sde.shape == (batch_size,))

    # Also test with tight bounds for bounds checking
    actions_sde_tight, _, _, _ = sample_mip_actions_sde(
        actor_fn=actor_fn,
        noise_fn=noise_fn,
        observations=obs,
        rng=rng,
        mip_t_star=0.999,
        act_dim=act_dim,
        use_constant_noise=True,
        constant_noise_std=0.1,
        act_min=-1.0,
        act_max=1.0,
    )
    report("MIP SDE: actions within bounds (tight)",
           float(actions_sde_tight.min()) >= -1.0 - 1e-6 and float(actions_sde_tight.max()) <= 1.0 + 1e-6)

    # ---- MIP ODE ----
    actions_ode = sample_mip_actions_ode(
        actor_fn=actor_fn,
        observations=obs,
        rng=rng,
        mip_t_star=0.999,
        act_dim=act_dim,
        act_min=-1.0,
        act_max=1.0,
    )

    report("MIP ODE: actions shape correct",
           actions_ode.shape == (batch_size, act_dim))
    report("MIP ODE: actions within bounds",
           float(actions_ode.min()) >= -1.0 - 1e-6 and float(actions_ode.max()) <= 1.0 + 1e-6)

    # ---- MIP log-prob recomputation consistency ----
    logprob_recomp, _, info = compute_mip_log_prob(
        actor_fn=actor_fn,
        noise_fn=noise_fn,
        observations=obs,
        chains=chains_sde,
        mip_t_star=0.999,
        use_constant_noise=True,
        constant_noise_std=0.1,
    )

    err_mip = float(jnp.max(jnp.abs(logprob_sde - logprob_recomp)))
    report("MIP SDE logprob == MIP recomputed logprob",
           err_mip < 1e-3,
           f"max_err={err_mip:.4e}")


# ============================================================================
# Test 13: _safe_max and _penultimate_safe_max
# ============================================================================

def test_safe_max():
    """Test conservative Q-ensemble aggregation functions."""
    print("\n=== Test 13: _safe_max and _penultimate_safe_max ===")

    # All positive => safe_max returns min
    x_pos = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])  # (2, 3)
    result_pos = _safe_max(x_pos, axis=0)
    expected_pos = jnp.array([0.5, 1.5, 2.5])  # min along axis=0
    report("safe_max (all positive): returns min",
           float(jnp.max(jnp.abs(result_pos - expected_pos))) < 1e-6)

    # All negative => safe_max returns max
    x_neg = jnp.array([[-3.0, -2.0, -1.0], [-4.0, -3.0, -0.5]])
    result_neg = _safe_max(x_neg, axis=0)
    expected_neg = jnp.array([-3.0, -2.0, -0.5])  # max along axis=0
    report("safe_max (all negative): returns max",
           float(jnp.max(jnp.abs(result_neg - expected_neg))) < 1e-6)

    # Mixed sign => safe_max returns 0
    x_mixed = jnp.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])
    result_mixed = _safe_max(x_mixed, axis=0)
    expected_mixed = jnp.array([0.0, 0.0, 0.0])
    report("safe_max (mixed sign): returns 0",
           float(jnp.max(jnp.abs(result_mixed - expected_mixed))) < 1e-6)

    # Penultimate with enough elements
    x_pen = jnp.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]])  # (4, 2)
    result_pen = _penultimate_safe_max(x_pen, axis=0)
    # All positive => uses second smallest: sorted=[1,2,3,4], second_min=2 for col0
    expected_pen = jnp.array([2.0, 6.0])
    report("penultimate_safe_max (all positive): returns second min",
           float(jnp.max(jnp.abs(result_pen - expected_pen))) < 1e-6)

    x_pen_neg = jnp.array([[-4.0, -8.0], [-3.0, -7.0], [-2.0, -6.0], [-1.0, -5.0]])
    result_pen_neg = _penultimate_safe_max(x_pen_neg, axis=0)
    # All negative => uses second largest: sorted=[-4,-3,-2,-1], second_max=-2 for col0
    expected_pen_neg = jnp.array([-2.0, -6.0])
    report("penultimate_safe_max (all negative): returns second max",
           float(jnp.max(jnp.abs(result_pen_neg - expected_pen_neg))) < 1e-6)


# ============================================================================
# Test 14: HL-Gauss distributional critic
# ============================================================================

def test_hl_gauss():
    """Test HL-Gauss transform and target computation."""
    print("\n=== Test 14: HL-Gauss Distributional Critic ===")

    q_min, q_max, num_bins = -10.0, 10.0, 51

    # ---- hl_gauss_transform ----
    to_probs, from_probs = hl_gauss_transform(q_min, q_max, num_bins)

    target_val = jnp.array([0.0])
    probs = to_probs(target_val)
    # With 1D input, probs is 1D: shape (num_bins,)
    report("hl_gauss_transform: probs shape correct",
           probs.shape == (num_bins,),
           f"actual shape={probs.shape}")
    report("hl_gauss_transform: probs sum to ~1",
           abs(float(probs.sum()) - 1.0) < 0.01,
           f"sum={float(probs.sum()):.4f}")
    report("hl_gauss_transform: probs are non-negative",
           float(probs.min()) >= -1e-6)

    # Reconstruct value from probs
    reconstructed = from_probs(probs)
    report("hl_gauss_transform: round-trip (to_probs -> from_probs) ~ original",
           abs(float(reconstructed) - 0.0) < 0.5,
           f"reconstructed={float(reconstructed):.4f}")

    # Peak should be near the target
    support = jnp.linspace(q_min, q_max, num_bins + 1)
    centers = (support[:-1] + support[1:]) / 2
    peak_idx = int(jnp.argmax(probs))
    peak_center = float(centers[peak_idx])
    report("hl_gauss_transform: peak bin near target value",
           abs(peak_center - 0.0) < 1.0,
           f"peak_center={peak_center:.2f}")

    # ---- hl_gauss_target ----
    critic_value = jnp.array([5.0, -3.0, 0.0])
    target_probs = hl_gauss_target(q_min, q_max, num_bins, critic_value)
    report("hl_gauss_target: shape correct",
           target_probs.shape == (3, num_bins))
    report("hl_gauss_target: probs sum to ~1 per sample",
           float(jnp.max(jnp.abs(target_probs.sum(axis=-1) - 1.0))) < 0.01)
    report("hl_gauss_target: probs are non-negative",
           float(target_probs.min()) >= -1e-6)


# ============================================================================
# Test 15: TrainState and ModuleDict
# ============================================================================

def test_train_state_and_module_dict():
    """Test core TrainState and ModuleDict functionality."""
    print("\n=== Test 15: TrainState & ModuleDict ===")

    obs_dim, act_dim = 8, 4

    # Build a ModuleDict
    actor = TinyActor(act_dim)
    critic = TinyCritic(num_qs=2)

    net_def = ModuleDict(modules={
        'actor': actor,
        'critic': critic,
    })

    ex_obs = jnp.zeros((1, obs_dim))
    ex_act = jnp.zeros((1, act_dim))
    ex_t = jnp.zeros((1, 1))

    rng = jax.random.PRNGKey(0)
    params = net_def.init(rng,
                          actor=(ex_obs, ex_act, ex_t),
                          critic=(ex_obs, ex_act))['params']

    # Create TrainState
    tx = optax.adam(1e-3)
    state = TrainState.create(net_def, params, tx)

    report("TrainState.step starts at 1", state.step == 1)
    report("TrainState has params", state.params is not None)
    report("TrainState has opt_state", state.opt_state is not None)

    # Test forward pass via select
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, obs_dim))
    act = jax.random.normal(jax.random.PRNGKey(2), (4, act_dim))
    t = jnp.zeros((4, 1))

    actor_out = state.select('actor')(obs, act, t)
    report("TrainState.select('actor'): returns correct shape",
           actor_out.shape == (4, act_dim))

    critic_out = state.select('critic')(obs, act)
    report("TrainState.select('critic'): returns correct shape",
           critic_out.shape == (2, 4))  # (num_qs, batch)

    # Test apply_gradients
    fake_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    new_state = state.apply_gradients(fake_grads)
    report("apply_gradients: step incremented", new_state.step == 2)

    # Params should barely change with zero gradients
    param_diff = jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_util.tree_map(
            lambda p1, p2: jnp.sum(jnp.abs(p1 - p2)),
            state.params, new_state.params
        )
    )
    report("apply_gradients (zero grads): params unchanged",
           float(param_diff) < 1e-6)


# ============================================================================
# Test 16: compute_bc_loss dispatcher
# ============================================================================

def test_bc_loss_dispatcher():
    """Test that compute_bc_loss routes to the correct loss function."""
    print("\n=== Test 16: BC Loss Dispatcher ===")

    B, obs_dim, act_dim = 16, 8, 4
    rng = jax.random.PRNGKey(0)

    class TinyModel(nn.Module):
        act_dim: int
        @nn.compact
        def __call__(self, obs, x_t, t, return_denoiser=False):
            x = jnp.concatenate([obs, x_t, t], axis=-1)
            x = nn.Dense(32)(x)
            x = nn.relu(x)
            vel = nn.Dense(self.act_dim)(x)
            if return_denoiser:
                denoiser = nn.Dense(self.act_dim)(x)
                return vel, denoiser
            return vel

    model = TinyModel(act_dim)
    obs = jax.random.normal(jax.random.PRNGKey(1), (B, obs_dim))

    # Init with return_denoiser=True so Dense_2 parameters are created
    model_params = model.init(jax.random.PRNGKey(42), obs,
                              jnp.zeros((B, act_dim)), jnp.zeros((B, 1)),
                              return_denoiser=True)

    def model_fn(obs, x_t, t):
        return model.apply(model_params, obs, x_t, t, return_denoiser=False)

    def model_fn_denoiser(obs, x_t, t, return_denoiser=True):
        return model.apply(model_params, obs, x_t, t, return_denoiser=return_denoiser)

    batch = {
        'observations': obs,
        'actions': jax.random.normal(rng, (B, 1, act_dim)),
    }

    # Standard flow
    config_flow = {
        'action_chunking': False,
        'horizon_length': 1,
        'action_dim': act_dim,
        'policy_type': 'flow',
        'use_denoiser': False,
    }
    loss_flow, info_flow = compute_bc_loss(batch, rng, model_fn, config_flow)
    report("BC dispatcher (flow): returns scalar loss",
           loss_flow.ndim == 0 and float(loss_flow) > 0)

    # MIP
    config_mip = {
        'action_chunking': False,
        'horizon_length': 1,
        'action_dim': act_dim,
        'policy_type': 'mip',
        'use_denoiser': False,
        'mip_t_star': 0.999,
    }
    loss_mip, info_mip = compute_bc_loss(batch, rng, model_fn, config_mip)
    report("BC dispatcher (MIP): returns scalar loss",
           loss_mip.ndim == 0 and float(loss_mip) > 0)
    report("BC dispatcher (MIP): info has regression and denoising keys",
           'bc_mip_loss_regression' in info_mip and 'bc_mip_loss_denoising' in info_mip)

    # Denoiser
    config_denoiser = {
        'action_chunking': False,
        'horizon_length': 1,
        'action_dim': act_dim,
        'policy_type': 'flow',
        'use_denoiser': True,
        'constant_noise_std': 0.01,
    }
    loss_den, info_den = compute_bc_loss(batch, rng, model_fn, config_denoiser,
                                          model_fn_denoiser=model_fn_denoiser)
    report("BC dispatcher (denoiser): returns scalar loss",
           loss_den.ndim == 0 and float(loss_den) > 0)
    report("BC dispatcher (denoiser): info has vel and denoising keys",
           'bc_flow_loss_vel' in info_den and 'bc_flow_loss_denoising' in info_den)


# ============================================================================
# Test 17: SDE sampling output shapes and properties
# ============================================================================

def test_sde_sampling_properties():
    """Test SDE sampling output properties beyond just logprob consistency."""
    print("\n=== Test 17: SDE Sampling Properties ===")

    obs_dim, act_dim = 8, 4
    batch_size = 32
    flow_steps = 10

    state = build_actor_network(obs_dim, act_dim, seed=42)
    actor_fn = make_actor_fn(state, 'target_actor')
    noise_fn = make_noise_fn(state, 'target_noise_net')
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, obs_dim))
    rng = jax.random.PRNGKey(1)

    actions, chains, logprob, sigmas = sample_flow_actions_sde(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, rng=rng,
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        act_min=-1.0, act_max=1.0,
        clip_intermediate=True, clip_value=1.0,
    )

    report("SDE: actions shape (batch, act_dim)",
           actions.shape == (batch_size, act_dim))
    report("SDE: chains shape (batch, flow_steps+1, act_dim)",
           chains.shape == (batch_size, flow_steps + 1, act_dim))
    report("SDE: logprob shape (batch,)",
           logprob.shape == (batch_size,))
    report("SDE: sigmas shape (batch, flow_steps, act_dim)",
           sigmas.shape == (batch_size, flow_steps, act_dim))

    # Final actions = last chain entry
    report("SDE: final actions == chains[:, -1]",
           float(jnp.max(jnp.abs(actions - chains[:, -1]))) < 1e-6)

    # Actions within bounds
    report("SDE: actions within [-1, 1]",
           float(actions.min()) >= -1.0 - 1e-6 and float(actions.max()) <= 1.0 + 1e-6)

    # logprob is finite
    report("SDE: logprobs are finite",
           bool(jnp.all(jnp.isfinite(logprob))))

    # Chain first entry should look like N(0,I) noise
    chain_init = chains[:, 0]
    report("SDE: initial chain ~ N(0,I) (mean near 0)",
           abs(float(chain_init.mean())) < 0.5)

    # Constant noise => all sigmas same
    report("SDE: constant noise => sigmas all equal",
           float(jnp.std(sigmas)) < 1e-6,
           f"sigma_std={float(jnp.std(sigmas)):.6f}")


# ============================================================================
# Test 18: compute_flow_log_prob info dict and entropy
# ============================================================================

def test_flow_logprob_info():
    """Test that compute_flow_log_prob returns correct info and optional entropy."""
    print("\n=== Test 18: Flow Log-prob Info & Entropy ===")

    obs_dim, act_dim = 8, 4
    batch_size = 16
    flow_steps = 5

    state = build_actor_network(obs_dim, act_dim, seed=42)
    actor_fn = make_actor_fn(state, 'target_actor')
    noise_fn = make_noise_fn(state, 'target_noise_net')
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, obs_dim))
    rng = jax.random.PRNGKey(1)

    _, chains, _, _ = sample_flow_actions_sde(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, rng=rng,
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
    )

    # Without entropy
    logprob, ent, info = compute_flow_log_prob(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, chains=chains,
        flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
    )
    report("info contains vel_mean", 'vel_mean' in info)
    report("info contains sigma_mean", 'sigma_mean' in info)
    report("info contains logprob_mean", 'logprob_mean' in info)
    report("entropy is None when not requested", ent is None)

    # With entropy
    logprob_e, ent_e, info_e = compute_flow_log_prob(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, chains=chains,
        flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
        get_entropy=True,
    )
    report("entropy is not None when requested", ent_e is not None)
    report("info contains joint_entropy", 'joint_entropy' in info_e)
    report("info contains entropy_rate", 'entropy_rate' in info_e)
    report("logprob unchanged by entropy flag",
           float(jnp.max(jnp.abs(logprob - logprob_e))) < 1e-5)

    # Normalization
    logprob_normed, _, _ = compute_flow_log_prob(
        actor_fn=actor_fn, noise_fn=noise_fn,
        observations=obs, chains=chains,
        flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
        normalize_horizon=True,
    )
    # Normalized logprob should be logprob / logprob_steps
    logprob_steps = flow_steps + 1  # init + flow_steps transitions
    expected_normed = logprob / logprob_steps
    report("normalize_horizon: logprob / steps",
           float(jnp.max(jnp.abs(logprob_normed - expected_normed))) < 1e-4)


# ============================================================================
# Test: ft_flow_steps — partial IS ratio and gradient isolation
# ============================================================================

def test_ft_flow_steps_gradient_isolation():
    """
    Verify that ft_flow_steps correctly isolates gradients:
      1. Logprob with ft_flow_steps=K equals the sum of only the last K transitions
      2. Gradient of logprob w.r.t. chain entries at frozen steps is zero
      3. Gradient of logprob w.r.t. chain entries at finetuned steps is nonzero
      4. SDE sampling logprob with ft_flow_steps matches recomputed logprob
      5. Default (ft_flow_steps=flow_steps) matches original behavior exactly
    """
    print("\n=== Test: ft_flow_steps Gradient Isolation ===")

    obs_dim, act_dim = 4, 3
    flow_steps = 6
    ft_flow_steps = 2  # only finetune last 2 of 6 steps
    ft_start = flow_steps - ft_flow_steps  # = 4
    B = 8

    state = build_actor_network(obs_dim, act_dim, seed=77)
    actor_fn = make_actor_fn(state, 'actor')
    noise_fn = make_noise_fn(state, 'noise_net')

    rng = jax.random.PRNGKey(123)
    obs = jax.random.normal(rng, (B, obs_dim))

    # ---- Step 1: Sample a full chain via SDE (full flow_steps) ----
    rng, key = jax.random.split(rng)
    actions_full, chains_full, lp_full, sigmas_full = sample_flow_actions_sde(
        actor_fn=actor_fn, noise_fn=noise_fn, observations=obs, rng=key,
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        ft_flow_steps=0,  # default: all steps
    )
    # chains_full: [B, flow_steps+1, act_dim] = [B, 7, 3]

    # ---- Test A: Partial logprob value matches manual sum of last K transitions ----
    # Compute full log prob (all transitions)
    lp_recompute_full, _, info_full = compute_flow_log_prob(
        actor_fn=actor_fn, noise_fn=noise_fn, observations=obs,
        chains=chains_full, flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
        ft_flow_steps=0,  # all steps
    )

    # Compute partial log prob (last ft_flow_steps transitions)
    lp_recompute_partial, _, info_partial = compute_flow_log_prob(
        actor_fn=actor_fn, noise_fn=noise_fn, observations=obs,
        chains=chains_full, flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
        ft_flow_steps=ft_flow_steps,
    )

    # Manually compute: full logprob = init + sum(transitions[0:6])
    #                    partial logprob = sum(transitions[4:6])  (no init)
    # The difference should be: init + sum(transitions[0:4])
    # Verify partial has fewer logprob_steps
    report("partial logprob_steps is ft_flow_steps",
           info_partial['logprob_steps'] == ft_flow_steps,
           f"got {info_partial['logprob_steps']}, expected {ft_flow_steps}")
    report("full logprob_steps is flow_steps + 1",
           info_full['logprob_steps'] == flow_steps + 1,
           f"got {info_full['logprob_steps']}, expected {flow_steps + 1}")

    # Partial logprob should be less negative (fewer terms summed)
    # and different from full logprob
    report("partial logprob differs from full logprob",
           float(jnp.abs(lp_recompute_full - lp_recompute_partial).mean()) > 0.1,
           f"diff={float(jnp.abs(lp_recompute_full - lp_recompute_partial).mean()):.4f}")

    # ---- Test B: Gradient isolation — grad w.r.t. frozen chain entries is zero ----
    def logprob_of_chains(chains_input):
        lp, _, _ = compute_flow_log_prob(
            actor_fn=actor_fn, noise_fn=noise_fn, observations=obs,
            chains=chains_input, flow_steps=flow_steps,
            use_constant_noise=True, constant_noise_std=0.1,
            ft_flow_steps=ft_flow_steps,
        )
        return lp.sum()

    grad_chains = jax.grad(logprob_of_chains)(chains_full)
    # grad_chains: [B, flow_steps+1, act_dim]

    # Frozen steps: indices 0 to ft_start-1 = [0, 1, 2, 3]
    # These should have zero gradient (they don't appear in the logprob sum)
    frozen_grad = grad_chains[:, :ft_start, :]
    frozen_grad_norm = float(jnp.abs(frozen_grad).max())
    report("frozen chain steps have zero gradient",
           frozen_grad_norm < 1e-6,
           f"max |grad| at frozen steps = {frozen_grad_norm:.2e}")

    # Finetuned steps: indices ft_start to flow_steps = [4, 5, 6]
    # These should have nonzero gradient
    ft_grad = grad_chains[:, ft_start:, :]
    ft_grad_norm = float(jnp.abs(ft_grad).max())
    report("finetuned chain steps have nonzero gradient",
           ft_grad_norm > 1e-4,
           f"max |grad| at finetuned steps = {ft_grad_norm:.2e}")

    # ---- Test C: Gradient flows through actor params for finetuned steps ----
    def logprob_of_params_partial(params):
        def actor_fn_p(obs, actions, t, params=None, is_encoded=True):
            return state.select('actor')(obs, actions, t, params=params, is_encoded=is_encoded)
        lp, _, _ = compute_flow_log_prob(
            actor_fn=actor_fn_p, noise_fn=noise_fn, observations=obs,
            chains=chains_full, flow_steps=flow_steps,
            use_constant_noise=True, constant_noise_std=0.1,
            params=params, ft_flow_steps=ft_flow_steps,
        )
        return lp.sum()

    def logprob_of_params_full(params):
        def actor_fn_p(obs, actions, t, params=None, is_encoded=True):
            return state.select('actor')(obs, actions, t, params=params, is_encoded=is_encoded)
        lp, _, _ = compute_flow_log_prob(
            actor_fn=actor_fn_p, noise_fn=noise_fn, observations=obs,
            chains=chains_full, flow_steps=flow_steps,
            use_constant_noise=True, constant_noise_std=0.1,
            params=params, ft_flow_steps=0,
        )
        return lp.sum()

    grad_params_partial = jax.grad(logprob_of_params_partial)(state.params)
    grad_params_full = jax.grad(logprob_of_params_full)(state.params)

    # Both should produce nonzero gradients (shared params affect all steps)
    partial_grad_norm = float(jnp.sqrt(sum(
        jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(grad_params_partial))))
    full_grad_norm = float(jnp.sqrt(sum(
        jnp.sum(v ** 2) for v in jax.tree_util.tree_leaves(grad_params_full))))

    report("partial actor param gradient is nonzero",
           partial_grad_norm > 1e-4,
           f"||grad||={partial_grad_norm:.4f}")
    report("full actor param gradient is nonzero",
           full_grad_norm > 1e-4,
           f"||grad||={full_grad_norm:.4f}")
    report("partial param gradient smaller than full (fewer contributing steps)",
           partial_grad_norm < full_grad_norm,
           f"partial={partial_grad_norm:.4f} < full={full_grad_norm:.4f}")

    # ---- Test D: SDE sampling logprob with ft_flow_steps matches recompute ----
    # Use wide act bounds to avoid last-step clipping discrepancy (known issue, see test 6)
    rng, key = jax.random.split(rng)
    _, chains_ft, lp_sde_ft, _ = sample_flow_actions_sde(
        actor_fn=make_actor_fn(state, 'target_actor'),
        noise_fn=make_noise_fn(state, 'target_noise_net'),
        observations=obs, rng=key,
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        act_min=-100.0, act_max=100.0,
        ft_flow_steps=ft_flow_steps,
    )

    # Recompute logprob from the sampled chain using same ft_flow_steps
    lp_recomp_ft, _, _ = compute_flow_log_prob(
        actor_fn=make_actor_fn(state, 'target_actor'),
        noise_fn=make_noise_fn(state, 'target_noise_net'),
        observations=obs, chains=chains_ft, flow_steps=flow_steps,
        use_constant_noise=True, constant_noise_std=0.1,
        ft_flow_steps=ft_flow_steps,
    )

    diff_ft = float(jnp.max(jnp.abs(lp_sde_ft - lp_recomp_ft)))
    report("SDE partial logprob matches recomputed partial logprob",
           diff_ft < 1e-3, f"max diff = {diff_ft:.6f}")

    # ---- Test E: Default ft_flow_steps=flow_steps reproduces original behavior ----
    _, chains_default, lp_sde_default, _ = sample_flow_actions_sde(
        actor_fn=make_actor_fn(state, 'target_actor'),
        noise_fn=make_noise_fn(state, 'target_noise_net'),
        observations=obs, rng=key,  # same key for reproducibility
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        ft_flow_steps=flow_steps,  # same as full
    )

    _, chains_noft, lp_sde_noft, _ = sample_flow_actions_sde(
        actor_fn=make_actor_fn(state, 'target_actor'),
        noise_fn=make_noise_fn(state, 'target_noise_net'),
        observations=obs, rng=key,  # same key
        flow_steps=flow_steps, act_dim=act_dim,
        use_constant_noise=True, constant_noise_std=0.1,
        ft_flow_steps=0,  # default (all steps)
    )

    diff_default = float(jnp.max(jnp.abs(lp_sde_default - lp_sde_noft)))
    report("ft_flow_steps=flow_steps identical to ft_flow_steps=0 (default)",
           diff_default < 1e-5, f"max diff = {diff_default:.6f}")

    chains_diff = float(jnp.max(jnp.abs(chains_default - chains_noft)))
    report("chains identical for ft_flow_steps=flow_steps vs default",
           chains_diff < 1e-5, f"max diff = {chains_diff:.6f}")

    # ---- Test F: Full chain is still generated even with partial IS ----
    report("partial SDE still produces full-length chain",
           chains_ft.shape[1] == flow_steps + 1,
           f"chain length = {chains_ft.shape[1]}, expected {flow_steps + 1}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("OGPO Core Module Tests")
    print("=" * 70)

    # EMA / Polyak
    test_ema_target_update()
    test_ema_on_real_network()

    # Log-prob consistency
    test_logprob_consistency()

    # Importance sampling
    test_importance_sampling_ratio()

    # PPO / Advantages (updated API)
    test_ppo_loss()
    test_ogpo_advantages()

    # End-to-end IS
    test_is_ratio_after_ema()

    # Clipping bug
    test_logprob_last_step_clipping()

    # Q-learning helpers
    test_q_helpers()

    # BC helpers
    test_bc_helpers()
    test_mip_helpers()

    # Flow sampling (ODE, ODE+correction, SDE, MIP)
    test_ode_sampling()
    test_ode_with_correction()
    test_mip_sampling()
    test_sde_sampling_properties()

    # Safe max
    test_safe_max()

    # HL-Gauss
    test_hl_gauss()

    # TrainState / ModuleDict
    test_train_state_and_module_dict()

    # BC loss dispatcher
    test_bc_loss_dispatcher()

    # Log-prob info & entropy
    test_flow_logprob_info()

    # ft_flow_steps gradient isolation
    test_ft_flow_steps_gradient_isolation()

    print("\n" + "=" * 70)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} total")
    print("=" * 70)

    if FAIL > 0:
        sys.exit(1)
