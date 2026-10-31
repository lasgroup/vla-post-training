"""Measure relative compute/memory for image-based (PaliGemma frozen encoder) runs.

PaliGemma pre-encodes images to 2048-dim embeddings before training.
The agent sees obs_dim = 2048 (SigLIP) + proprio_dim as flat state.
This profiles the training loop compute, NOT the one-time pre-encoding.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python tests/profile_compute_image.py
    CUDA_VISIBLE_DEVICES=0 uv run python tests/profile_compute_image.py --algos=ogpo,expo
    CUDA_VISIBLE_DEVICES=0 uv run python tests/profile_compute_image.py --proprio_dim=9 --img_embed_dim=2048

The default parameters match square_image_paligemma.sh:
    obs_dim = 2048 (SigLIP) + 9 (proprio) = 2057
    action_dim = 7, horizon = 4
    5-layer 512-dim MLPs, actor_layer_norm=true
"""
import argparse
import gc
import os
import subprocess
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

HAS_MEM_STATS = False

def _init_mem():
    global HAS_MEM_STATS
    try:
        jax.devices()[0].memory_stats()
        HAS_MEM_STATS = True
    except Exception:
        HAS_MEM_STATS = False

def mem_in_use_gb():
    if HAS_MEM_STATS:
        return max(d.memory_stats()["bytes_in_use"] for d in jax.devices()) / 1e9
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return max(float(x) for x in out.strip().split("\n")) / 1024
    except Exception:
        return -1

def peak_mem_gb():
    if HAS_MEM_STATS:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices()) / 1e9
    return mem_in_use_gb()

def reset_peak():
    if HAS_MEM_STATS:
        for d in jax.devices():
            try: d.clear_memory_stats()
            except: pass


def make_config(algo, obs_dim, action_dim, batch_size, utd, horizon, num_qs,
                best_of_n, flow_steps, group_num_samples, hidden_dims):
    from ogpo.utils.config_loader import load_config, config_to_agent_dict
    base = load_config(algo, {})
    c = config_to_agent_dict(base)

    c["batch_size"] = batch_size
    c["horizon_length"] = horizon
    c["action_chunking"] = True
    c["num_qs"] = num_qs
    c["best_of_n"] = best_of_n
    c["flow_steps"] = flow_steps
    c["ft_flow_steps"] = flow_steps
    c["discount"] = 0.99
    c["tau"] = 0.05
    c["actor_tau"] = 0.005
    c["lr"] = 3e-4
    c["actor_lr"] = 3e-4
    c["critic_lr"] = 3e-4
    c["clip_grad_norm"] = 1000.0
    c["time_embedding"] = "sinusoidal"
    c["time_embedding_dim"] = 32
    c["use_constant_noise"] = True
    c["constant_noise_std"] = 0.01
    c["clip_intermediate_actions"] = True
    c["actor_hidden_dims"] = hidden_dims
    c["value_hidden_dims"] = hidden_dims
    c["actor_layer_norm"] = True
    c["layer_norm"] = True

    if algo == "ogpo":
        c["use_denoiser"] = True
        c["use_bc_regularization"] = True
        c["error_correct_ode_to_sde"] = True  # legacy FPO+denoiser correction (renamed from error_correct_sde_to_ode)
        c["adv_strategy"] = "vanilla"
        c["ppo_lr"] = 4.5e-5
        c["group_num_samples"] = group_num_samples
        c["ppo_batch_size"] = batch_size
        c["q_agg"] = "subsample"
        c["subsample_bon"] = True
        c["awr_beta"] = 10
        c["bc_coeff"] = 1.0
        c["q_variance_reduction"] = False
        c["mc_regression"] = False
    elif algo == "expo":
        c["n_action_samples"] = best_of_n
        c["n_edit_samples"] = best_of_n
        c["edit_policy_bound"] = 0.05
        c["edit_lr"] = 3e-4
        c["alpha_lr"] = 3e-4
        c["use_bc_regularization"] = False
    elif algo == "qc":
        c["alpha"] = 0
        c["actor_num_samples"] = best_of_n
        c["q_agg"] = "mean"
    elif algo == "dsrl":
        pass

    return c


def make_batches(obs_dim, action_dim, batch_size, utd, horizon, flat_actions=False):
    rng = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    if flat_actions:
        act_shape_flat = (batch_size, horizon * action_dim)
        act_shape_utd = (utd, batch_size, horizon * action_dim)
    else:
        act_shape_flat = (batch_size, horizon, action_dim)
        act_shape_utd = (utd, batch_size, horizon, action_dim)

    flat = {
        "observations": jax.random.normal(k1, (batch_size, obs_dim)),
        "actions": jax.random.normal(k2, act_shape_flat),
        "rewards": jax.random.normal(k3, (batch_size,)),
        "masks": jnp.ones((batch_size,)),
        "next_observations": jax.random.normal(k4, (batch_size, obs_dim)),
    }
    utd_b = {
        "observations": jax.random.normal(k1, (utd, batch_size, obs_dim)),
        "actions": jax.random.normal(k2, act_shape_utd),
        "rewards": jax.random.normal(k3, (utd, batch_size)),
        "masks": jnp.ones((utd, batch_size)),
        "next_observations": jax.random.normal(k4, (utd, batch_size, obs_dim)),
    }
    return flat, utd_b


def create_agent(algo, obs_dim, action_dim, config):
    ex_obs = jnp.zeros((obs_dim,))
    ex_act = jnp.zeros((action_dim,))

    if algo == "ogpo":
        from ogpo.agents.ogpo import OGPOAgent
        return OGPOAgent.create(0, ex_obs, ex_act, config, adv_clip_min=None)
    elif algo == "expo":
        from ogpo.agents.expo import EXPOAgent
        return EXPOAgent.create(0, ex_obs, ex_act, config)
    elif algo == "qc":
        from ogpo.agents.qc import ACFQLAgent
        return ACFQLAgent.create(0, ex_obs, ex_act, config)
    elif algo == "dsrl":
        from ogpo.agents.dsrl import DSRLBestOfNAgent
        return DSRLBestOfNAgent.create(0, ex_obs, ex_act, config)
    else:
        raise ValueError(f"Unknown algo: {algo}")


def time_fn(fn, warmup=1, repeats=3):
    def _wait(r):
        leaves = jax.tree.leaves(r) if not isinstance(r, tuple) else sum([jax.tree.leaves(x) for x in r], [])
        jax.block_until_ready(leaves)
    for _ in range(warmup):
        result = fn()
        _wait(result)
    times = []
    peak_mems = []
    for _ in range(repeats):
        reset_peak()
        baseline = mem_in_use_gb()
        t0 = time.time()
        result = fn()
        _wait(result)
        times.append(time.time() - t0)
        peak = peak_mem_gb()
        peak_mems.append(peak - baseline if baseline > 0 and peak > baseline else peak)
    return np.median(times), np.median(peak_mems), result


def profile_algo(algo, obs_dim, action_dim, batch_size, utd, horizon,
                 num_qs, best_of_n, flow_steps, group_num_samples, hidden_dims):
    print(f"\n{'='*60}")
    print(f"  {algo.upper()}  (image-based, obs_dim={obs_dim})")
    print(f"{'='*60}")

    config = make_config(algo, obs_dim, action_dim, batch_size, utd, horizon,
                         num_qs, best_of_n, flow_steps, group_num_samples, hidden_dims)

    flat_batch, utd_batch = make_batches(
        obs_dim, action_dim, batch_size, utd, horizon,
        flat_actions=(algo == "dsrl"),
    )

    results = {"algo": algo}

    try:
        t0 = time.time()
        agent = create_agent(algo, obs_dim, action_dim, config)
        jax.block_until_ready(jax.tree.leaves(agent))
        results["create_s"] = time.time() - t0
        print(f"  Created in {results['create_s']:.1f}s")
    except Exception:
        print(f"  FAILED to create agent:")
        traceback.print_exc()
        results["error"] = "create"
        return results

    leaves = jax.tree.leaves(agent)
    param_count = sum(x.size for x in leaves if hasattr(x, 'size'))
    results["params_M"] = param_count / 1e6
    print(f"  Parameters: {results['params_M']:.1f}M")

    # bc_update
    try:
        def bc_fn():
            return agent.bc_update(flat_batch)
        t, mem, (agent_new, _) = time_fn(bc_fn, warmup=1, repeats=3)
        agent = agent_new
        results["bc_ms"] = t * 1000
        results["bc_mem"] = mem
        print(f"  bc_update:      {results['bc_ms']:8.1f} ms  |  delta {mem:.2f} GB")
    except Exception:
        print(f"  bc_update FAILED:")
        traceback.print_exc()
        results["bc_ms"] = -1

    # batch_update
    try:
        if algo == "dsrl":
            def update_fn():
                return agent.batch_update(utd_batch)
        else:
            def update_fn():
                return agent.batch_update(utd_batch, utd_batch, True)
        t, mem, (agent_new, _) = time_fn(update_fn, warmup=1, repeats=3)
        agent = agent_new
        results["update_ms"] = t * 1000
        results["update_mem"] = mem
        print(f"  batch_update:   {results['update_ms']:8.1f} ms  |  delta {mem:.2f} GB")
    except Exception:
        print(f"  batch_update FAILED:")
        traceback.print_exc()
        results["update_ms"] = -1

    # sample_actions
    try:
        eval_obs = jnp.zeros((4, obs_dim))
        def sample_fn():
            return agent.sample_actions(eval_obs, jax.random.PRNGKey(0))
        t, mem, _ = time_fn(sample_fn, warmup=1, repeats=5)
        results["sample_ms"] = t * 1000
        print(f"  sample_actions: {results['sample_ms']:8.1f} ms  (batch=4)")
    except Exception:
        print(f"  sample_actions FAILED:")
        traceback.print_exc()
        results["sample_ms"] = -1

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Profile compute for image-based (PaliGemma frozen) runs")
    parser.add_argument("--img_embed_dim", type=int, default=2048,
                        help="Image embedding dim (2048 for PaliGemma SigLIP)")
    parser.add_argument("--proprio_dim", type=int, default=9,
                        help="Proprioceptive state dim (9 for square: eef_pos+quat+gripper)")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--utd", type=int, default=1,
                        help="UTD ratio (paligemma runs typically use utd=1)")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--num_qs", type=int, default=10)
    parser.add_argument("--best_of_n", type=int, default=8)
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--group_num_samples", type=int, default=32)
    parser.add_argument("--hidden_dims", type=str, default="512,512,512,512,512",
                        help="MLP hidden dims (comma-separated)")
    parser.add_argument("--algos", type=str, default="ogpo,expo,qc,dsrl")
    parser.add_argument("--compare_state", action="store_true",
                        help="Also profile state-based (obs_dim=23) for comparison")
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",")]
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    obs_dim_image = args.img_embed_dim + args.proprio_dim
    obs_dim_state = 23  # square low-dim

    print("=" * 70)
    print("  Image-Based Compute Profiler (PaliGemma Frozen Encoder)")
    print("=" * 70)
    print(f"  Image embedding: {args.img_embed_dim}-dim (SigLIP)")
    print(f"  Proprio: {args.proprio_dim}-dim → total obs_dim = {obs_dim_image}")
    print(f"  action_dim={args.action_dim}  horizon={args.horizon}  "
          f"batch={args.batch_size}  utd={args.utd}")
    print(f"  num_qs={args.num_qs}  best_of_n={args.best_of_n}  "
          f"flow_steps={args.flow_steps}")
    print(f"  hidden_dims={hidden_dims}")
    print(f"  algos: {algos}")
    if args.compare_state:
        print(f"  Also comparing: state-based (obs_dim={obs_dim_state})")

    _init_mem()
    devices = jax.devices()
    print(f"\n  JAX: {jax.__version__}, {len(devices)}x {devices[0].device_kind}")
    print(f"  memory_stats: {'yes' if HAS_MEM_STATS else 'no (nvidia-smi fallback)'}")

    # --- State-based baseline (optional) ---
    state_results = {}
    if args.compare_state:
        print(f"\n{'*'*70}")
        print(f"  STATE-BASED BASELINE (obs_dim={obs_dim_state})")
        print(f"{'*'*70}")
        for algo in algos:
            gc.collect(); jax.clear_caches()
            try:
                r = profile_algo(algo, obs_dim_state, args.action_dim,
                                 args.batch_size, args.utd, args.horizon,
                                 args.num_qs, args.best_of_n, args.flow_steps,
                                 args.group_num_samples, hidden_dims)
                state_results[algo] = r
            except Exception:
                traceback.print_exc()
                state_results[algo] = {"algo": algo, "error": "crash"}

    # --- Image-based profiling ---
    print(f"\n{'*'*70}")
    print(f"  IMAGE-BASED (obs_dim={obs_dim_image})")
    print(f"{'*'*70}")
    image_results = {}
    for algo in algos:
        gc.collect(); jax.clear_caches()
        try:
            r = profile_algo(algo, obs_dim_image, args.action_dim,
                             args.batch_size, args.utd, args.horizon,
                             args.num_qs, args.best_of_n, args.flow_steps,
                             args.group_num_samples, hidden_dims)
            image_results[algo] = r
        except Exception:
            traceback.print_exc()
            image_results[algo] = {"algo": algo, "error": "crash"}

    # --- Summary ---
    print(f"\n{'='*80}")
    print("  SUMMARY — Image-Based (PaliGemma frozen encoder)")
    print(f"{'='*80}")
    print(f"  {'Algo':<8} {'Params':>8} {'bc_update':>12} {'batch_update':>14} "
          f"{'sample':>10} {'Δ Mem':>10} {'vs OGPO':>10}")
    print(f"  {'-'*74}")

    base_time = image_results.get("ogpo", {}).get("update_ms", None)
    if base_time is None:
        base_time = next((r.get("update_ms") for r in image_results.values()
                         if r.get("update_ms", -1) > 0), 1)

    for algo in algos:
        r = image_results.get(algo, {})
        if "error" in r:
            print(f"  {algo:<8} {'FAILED':>8}")
            continue
        params = f"{r.get('params_M', 0):.1f}M"
        bc = f"{r['bc_ms']:.0f}ms" if r.get("bc_ms", -1) > 0 else "---"
        upd = f"{r['update_ms']:.0f}ms" if r.get("update_ms", -1) > 0 else "---"
        smp = f"{r['sample_ms']:.1f}ms" if r.get("sample_ms", -1) > 0 else "---"
        mem = f"{r.get('update_mem', -1):.2f}GB" if r.get("update_mem", -1) > 0 else "---"
        rel = f"{r['update_ms']/base_time:.2f}x" if r.get("update_ms", -1) > 0 and base_time > 0 else "---"
        print(f"  {algo:<8} {params:>8} {bc:>12} {upd:>14} {smp:>10} {mem:>10} {rel:>10}")

    # --- State vs Image comparison ---
    if args.compare_state and state_results:
        print(f"\n{'='*80}")
        print(f"  STATE vs IMAGE — batch_update slowdown from obs_dim {obs_dim_state} → {obs_dim_image}")
        print(f"{'='*80}")
        print(f"  {'Algo':<8} {'State ms':>10} {'Image ms':>10} {'Slowdown':>10} "
              f"{'State Params':>13} {'Image Params':>13}")
        print(f"  {'-'*68}")
        for algo in algos:
            s = state_results.get(algo, {})
            i = image_results.get(algo, {})
            s_ms = s.get("update_ms", -1)
            i_ms = i.get("update_ms", -1)
            s_p = f"{s.get('params_M', 0):.1f}M"
            i_p = f"{i.get('params_M', 0):.1f}M"
            if s_ms > 0 and i_ms > 0:
                ratio = f"{i_ms/s_ms:.2f}x"
                print(f"  {algo:<8} {s_ms:>9.0f}ms {i_ms:>9.0f}ms {ratio:>10} {s_p:>13} {i_p:>13}")
            else:
                print(f"  {algo:<8} {'---':>10} {'---':>10} {'---':>10} {s_p:>13} {i_p:>13}")

    # --- jax_frac recommendations ---
    print(f"\n  --- jax_frac recommendations (image-based) ---")
    try:
        if HAS_MEM_STATS:
            limit_gb = jax.devices()[0].memory_stats()["bytes_limit"] / 1e9
        else:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
            limit_gb = max(float(x) for x in out.strip().split("\n")) / 1024
    except Exception:
        limit_gb = 0

    if limit_gb > 0:
        print(f"  GPU total: {limit_gb:.0f}GB")
        for algo in algos:
            r = image_results.get(algo, {})
            params_gb = r.get("params_M", 0) * 4 * 3 / 1000
            delta = r.get("update_mem", 0)
            total = params_gb + max(delta, 0) + 0.5
            frac = min(total / limit_gb, 0.95)
            print(f"  {algo:<8} params≈{params_gb:.1f}GB + Δmem≈{max(delta,0):.1f}GB "
                  f"→ ~{total:.1f}GB → jax_frac={frac:.2f}")


if __name__ == "__main__":
    main()
