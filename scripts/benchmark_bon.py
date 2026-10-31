"""Benchmark: Current (scan-based) vs Proposed (vectorized) BON for OGPO.

Usage:
    python scripts/benchmark_bon.py [--batch_size=32] [--best_of_n=8] [--num_qs=10]
"""
import sys
import os
import time
import argparse

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

# ── Config & Agent Setup ──────────────────────────────────────────────────
from ogpo.utils.config_loader import load_config, config_to_agent_dict
from ogpo.agents.ogpo import OGPOAgent


def create_test_agent(obs_dim=20, action_dim=4, horizon_length=4, best_of_n=8, num_qs=10, subsample_bon=True):
    """Create an OGPO agent with the given settings for benchmarking."""
    config = load_config('ogpo', {})
    # Override relevant settings
    config['agent']['best_of_n'] = best_of_n
    config['agent']['num_qs'] = num_qs
    config['agent']['subsample_bon'] = subsample_bon
    config['agent']['horizon_length'] = horizon_length
    config['agent']['action_chunking'] = True
    config['agent']['encoder'] = None
    config['agent']['actor_backbone'] = 'mlp'
    config['agent']['critic_backbone'] = 'mlp'
    config['agent']['useTF'] = False
    config['agent']['useSimBa'] = False
    config['agent']['useFiLM'] = False
    config['agent']['policy_type'] = 'flow'
    config['agent']['flow_steps'] = 10
    config['agent']['use_denoiser'] = True
    config['agent']['use_constant_noise'] = True
    config['agent']['constant_noise_std'] = 0.005
    config['agent']['min_noise_std'] = 0.005
    config['agent']['max_noise_std'] = 0.005

    agent_config = config_to_agent_dict(config)
    agent_config['ob_dims'] = (obs_dim,)
    agent_config['action_dim'] = action_dim

    ex_obs = jnp.zeros(obs_dim)
    ex_actions = jnp.zeros(action_dim)

    agent = OGPOAgent.create(
        seed=42,
        ex_observations=ex_obs,
        ex_actions=ex_actions,
        config=agent_config,
        adv_clip_min=None,
    )
    return agent


# ── Proposed Vectorized BON ──────────────────────────────────────────────
@partial(jax.jit, static_argnames=())
def sample_actions_BON_vectorized(agent, observations, rng):
    """Fully vectorized Best-of-N (proposed)."""
    expected_single_ndim = 3 if agent.config.encoder else 1
    is_single = observations.ndim == expected_single_ndim
    if is_single:
        observations = observations[None]

    batch_size = observations.shape[0]
    num_samples = agent.config.best_of_n

    # 1.  Generate all B*N candidate actions in one forward pass
    rng, action_key = jax.random.split(rng)
    obs_expanded = jnp.repeat(observations, num_samples, axis=0)  # [B*N, obs_dim]
    sampled_actions, _, _, _ = agent.sample_actions_with_noise(
        obs_expanded, rng=action_key, use_target=True, is_encoded=False
    )

    # 2.  Evaluate Q-ensemble: (num_qs, B*N) or (num_qs, B*N, 1)
    q_ensemble = agent.critic_network.select('target_critic')(
        obs_expanded, actions=sampled_actions, is_encoded=False
    )
    # Normalize to (num_qs, B*N), then reshape → (num_qs, B, N)
    q_flat = q_ensemble.reshape(agent.config.num_qs, -1)       # (num_qs, B*N)
    q_values = q_flat.reshape(agent.config.num_qs, batch_size, num_samples)

    # 3.  Aggregate Q-ensemble per candidate (Python if — static at trace time)
    if agent.config.subsample_bon:
        rng, sub_key = jax.random.split(rng)
        idxs = jax.random.randint(sub_key, (2, batch_size, num_samples), 0, agent.config.num_qs)
        qs_sub = jnp.take_along_axis(q_values, idxs, axis=0)  # (2, B, N)
        min_qs = jnp.min(qs_sub, axis=0)                       # (B, N)
    else:
        min_qs = jnp.min(q_values, axis=0)                     # (B, N)

    # 4.  Select best among N for each batch element
    best_idx = jnp.argmax(min_qs, axis=1)                      # (B,)
    actions_reshaped = sampled_actions.reshape(batch_size, num_samples, -1)
    best_actions = jax.vmap(lambda a, i: a[i])(actions_reshaped, best_idx)  # (B, act_dim)

    return best_actions


# ── Memory helpers ────────────────────────────────────────────────────────
def get_gpu_memory_bytes():
    """Return current GPU memory usage in bytes (returns 0 on CPU)."""
    try:
        dev = jax.local_devices()[0]
        stats = dev.memory_stats()
        if stats:
            return stats.get('bytes_in_use', 0)
    except Exception:
        pass
    return 0


def benchmark_fn(fn, *args, warmup=3, repeat=20, label=""):
    """Benchmark a JAX function: warmup, then measure time + peak memory."""
    # Warmup (includes JIT compilation)
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    mem_before = get_gpu_memory_bytes()

    times = []
    peak_mem = 0
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        mem_now = get_gpu_memory_bytes()
        peak_mem = max(peak_mem, mem_now - mem_before)

    times = np.array(times)
    print(f"[{label}]")
    print(f"  mean:   {times.mean()*1e3:8.2f} ms")
    print(f"  median: {np.median(times)*1e3:8.2f} ms")
    print(f"  std:    {times.std()*1e3:8.2f} ms")
    print(f"  min:    {times.min()*1e3:8.2f} ms")
    print(f"  max:    {times.max()*1e3:8.2f} ms")
    if peak_mem > 0:
        print(f"  peak Δmem: {peak_mem / 1024**2:.1f} MB")
    print()
    return times, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sizes', type=str, default='1,8,32,64',
                        help='Comma-separated batch sizes to test')
    parser.add_argument('--best_of_n', type=int, default=8)
    parser.add_argument('--num_qs', type=int, default=10)
    parser.add_argument('--obs_dim', type=int, default=20)
    parser.add_argument('--action_dim', type=int, default=4)
    parser.add_argument('--horizon_length', type=int, default=4)
    parser.add_argument('--subsample_bon', action='store_true', default=True)
    parser.add_argument('--no_subsample_bon', dest='subsample_bon', action='store_false')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=20)
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    print("=" * 60)
    print("BON Benchmark: Current (scan) vs Proposed (vectorized)")
    print("=" * 60)
    print(f"  obs_dim={args.obs_dim}, action_dim={args.action_dim}, "
          f"horizon_length={args.horizon_length}")
    print(f"  best_of_n={args.best_of_n}, num_qs={args.num_qs}, "
          f"subsample_bon={args.subsample_bon}")
    print(f"  warmup={args.warmup}, repeat={args.repeat}")
    print(f"  batch_sizes={batch_sizes}")
    print(f"  device: {jax.local_devices()[0]}")
    print()

    # Create agent once
    print("Creating agent...", flush=True)
    agent = create_test_agent(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        horizon_length=args.horizon_length,
        best_of_n=args.best_of_n,
        num_qs=args.num_qs,
        subsample_bon=args.subsample_bon,
    )
    print("Agent created.\n")

    rng = jax.random.PRNGKey(0)

    for bs in batch_sizes:
        print("-" * 60)
        print(f"Batch size = {bs}   (total fwd passes per BON call = {bs * args.best_of_n})")
        print("-" * 60)
        obs = jnp.zeros((bs, args.obs_dim)) if bs > 1 else jnp.zeros(args.obs_dim)

        # ── Current (scan + vmap) ──
        def run_current():
            return agent.sample_actions_BON(observations=obs, rng=rng)

        t_current, out_current = benchmark_fn(
            run_current, warmup=args.warmup, repeat=args.repeat,
            label=f"Current (scan+vmap)  bs={bs}"
        )

        # ── Proposed (vectorized) ──
        def run_proposed():
            return sample_actions_BON_vectorized(agent, obs, rng)

        t_proposed, out_proposed = benchmark_fn(
            run_proposed, warmup=args.warmup, repeat=args.repeat,
            label=f"Proposed (vectorized) bs={bs}"
        )

        # Verify outputs have same shape
        shape_c = np.array(out_current).shape
        shape_p = np.array(out_proposed).shape
        print(f"  Output shapes: current={shape_c}, proposed={shape_p}")

        speedup = np.median(t_current) / np.median(t_proposed)
        print(f"  Speedup (median): {speedup:.2f}x")
        print()


if __name__ == '__main__':
    main()
