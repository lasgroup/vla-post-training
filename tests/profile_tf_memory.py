"""Profile max transformer size for OGPO on multi-GPU.

Usage:
    # Quick: test a single config (default D=512, L=4)
    CUDA_VISIBLE_DEVICES=0,1 uv run python tests/profile_tf_memory.py

    # Specific single config
    CUDA_VISIBLE_DEVICES=0,1 uv run python tests/profile_tf_memory.py --dim=768 --layers=6

    # Sweep to find max config
    CUDA_VISIBLE_DEVICES=0,1 uv run python tests/profile_tf_memory.py --sweep

    # Custom sweep
    CUDA_VISIBLE_DEVICES=0,1 uv run python tests/profile_tf_memory.py --sweep --dims=512,768,1024 --layer_range=4,6,8
"""
import argparse
import gc
import os
import subprocess
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

HAS_MEMORY_STATS = False


def _check_memory_stats():
    global HAS_MEMORY_STATS
    try:
        jax.devices()[0].memory_stats()
        HAS_MEMORY_STATS = True
    except Exception:
        HAS_MEMORY_STATS = False
    return HAS_MEMORY_STATS


def get_memory_nvidia_smi():
    """Fallback: parse nvidia-smi for per-GPU memory usage."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        results = {}
        for line in out.strip().split("\n"):
            idx, used, total = [x.strip() for x in line.split(",")]
            results[int(idx)] = {
                "in_use_gb": float(used) / 1024,
                "peak_gb": float(used) / 1024,
                "limit_gb": float(total) / 1024,
            }
        return results
    except Exception:
        return {}


def get_memory_gb(device):
    if HAS_MEMORY_STATS:
        stats = device.memory_stats()
        return {
            "in_use_gb": stats["bytes_in_use"] / 1e9,
            "peak_gb": stats["peak_bytes_in_use"] / 1e9,
            "limit_gb": stats["bytes_limit"] / 1e9,
        }
    else:
        nv = get_memory_nvidia_smi()
        return nv.get(device.id, {"in_use_gb": -1, "peak_gb": -1, "limit_gb": -1})


def reset_peak(devices):
    if HAS_MEMORY_STATS:
        for d in devices:
            try:
                d.clear_memory_stats()
            except Exception:
                pass


def report_memory(devices, label=""):
    for d in devices:
        m = get_memory_gb(d)
        print(f"  [{label}] GPU {d.id}: {m['in_use_gb']:.2f} / {m['peak_gb']:.2f} / {m['limit_gb']:.2f} GB "
              f"(in_use / peak / limit)")


def make_agent_config(
    embed_dim=512,
    num_layers=4,
    num_heads=8,
    num_qs=10,
    batch_size=256,
    horizon_length=4,
    flow_steps=10,
    group_num_samples=16,
    ppo_batch_size=256,
):
    from ogpo.utils.config_loader import load_config, config_to_agent_dict

    base = load_config("ogpo", {})
    agent = config_to_agent_dict(base)

    agent["actor_backbone"] = "tf"
    agent["critic_backbone"] = "tf"
    agent["tf_pi_layers"] = num_layers
    agent["tf_pi_embed_dim"] = embed_dim
    agent["tf_pi_heads"] = num_heads
    agent["tf_pi_conditioning"] = "adaln"
    agent["tf_q_layers"] = num_layers
    agent["tf_q_embed_dim"] = embed_dim
    agent["tf_q_heads"] = num_heads
    agent["tf_q_conditioning"] = "adaln"
    agent["num_qs"] = num_qs
    agent["batch_size"] = batch_size
    agent["horizon_length"] = horizon_length
    agent["flow_steps"] = flow_steps
    agent["ft_flow_steps"] = flow_steps
    agent["time_embedding"] = "scalar"
    agent["time_embedding_dim"] = 32
    agent["use_denoiser"] = True
    agent["use_constant_noise"] = True
    agent["use_bc_regularization"] = True
    agent["error_correct_ode_to_sde"] = True  # legacy FPO+denoiser correction (renamed from error_correct_sde_to_ode)
    agent["adv_strategy"] = "conservative"
    agent["ppo_lr"] = 1e-4
    agent["clip_grad_norm"] = 1.0
    agent["group_num_samples"] = group_num_samples
    agent["ppo_batch_size"] = ppo_batch_size
    agent["awr_beta"] = 10
    agent["bc_coeff"] = 1.0
    agent["clip_epsilon"] = 0.01
    agent["entropy_coeff"] = 0.0
    agent["best_of_n"] = 8
    agent["q_agg"] = "subsample"
    agent["subsample_bon"] = True
    agent["discount"] = 0.99
    agent["tau"] = 0.05
    agent["actor_tau"] = 0.005
    agent["actor_lr"] = 3e-4
    agent["critic_lr"] = 3e-4
    agent["lr"] = 3e-4
    agent["use_constant_scheduler_for_bc"] = False
    agent["actor_scheduler"] = "cosine"
    agent["actor_warmup_steps"] = 1000
    agent["actor_decay_steps"] = 10000
    agent["actor_end_value"] = 1e-6
    agent["actor_weight_decay"] = 1e-2
    agent["critic_scheduler"] = "cosine"
    agent["critic_warmup_steps"] = 1000
    agent["critic_decay_steps"] = 10000
    agent["critic_end_value"] = 1e-6
    agent["critic_weight_decay"] = 1e-2
    agent["q_variance_reduction"] = False
    agent["mc_regression"] = False

    return agent


def make_synthetic_data(obs_dim, action_dim, batch_size, utd, rng_key):
    k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 6)

    flat_batch = {
        "observations": jax.random.normal(k1, (batch_size, obs_dim)),
        "actions": jax.random.normal(k2, (batch_size, action_dim)),
        "rewards": jax.random.normal(k3, (batch_size,)),
        "masks": jnp.ones((batch_size,)),
        "next_observations": jax.random.normal(k4, (batch_size, obs_dim)),
    }

    utd_batch = {
        "observations": jax.random.normal(k1, (utd, batch_size, obs_dim)),
        "actions": jax.random.normal(k2, (utd, batch_size, action_dim)),
        "rewards": jax.random.normal(k3, (utd, batch_size)),
        "masks": jnp.ones((utd, batch_size)),
        "next_observations": jax.random.normal(k4, (utd, batch_size, obs_dim)),
    }

    return flat_batch, utd_batch


def run_step(name, fn, devices):
    """Run a profiling step with timing and memory reporting. Returns (result, peak_gb, elapsed)."""
    reset_peak(devices)
    t0 = time.time()
    result = fn()
    jax.block_until_ready(jax.tree.leaves(result) if not isinstance(result, tuple) else
                          sum([jax.tree.leaves(r) for r in result], []))
    elapsed = time.time() - t0
    peak = max(get_memory_gb(d)["peak_gb"] for d in devices)
    in_use = max(get_memory_gb(d)["in_use_gb"] for d in devices)
    print(f"  {name}: peak={peak:.2f}GB, in_use={in_use:.2f}GB, time={elapsed:.1f}s")
    return result, peak, elapsed


def profile_config(embed_dim, num_layers, num_heads, num_qs, batch_size, utd,
                   mesh, devices, obs_dim=11, action_dim=4):
    from ogpo.agents.ogpo import OGPOAgent
    from ogpo.utils.sharding import shard_agent, shard_batch, shard_batch_flat

    config_str = f"D={embed_dim} L={num_layers} H={num_heads} Qs={num_qs} B={batch_size} UTD={utd}"
    print(f"\n{'='*70}")
    print(f"Testing: {config_str}")
    print(f"{'='*70}")

    results = {"config": config_str, "embed_dim": embed_dim, "num_layers": num_layers,
               "num_heads": num_heads, "num_qs": num_qs, "batch_size": batch_size}

    rng = jax.random.PRNGKey(42)
    ex_obs = jnp.zeros((obs_dim,))
    ex_act = jnp.zeros((action_dim,))

    agent_config = make_agent_config(
        embed_dim=embed_dim, num_layers=num_layers, num_heads=num_heads,
        num_qs=num_qs, batch_size=batch_size,
    )

    gc.collect()
    jax.clear_caches()
    reset_peak(devices)

    def _shard_flat(b):
        return shard_batch_flat(b, mesh) if mesh is not None else b

    def _shard_utd(b):
        return shard_batch(b, mesh) if mesh is not None else b

    full_act_dim = action_dim * agent_config["horizon_length"]
    steps = [
        "create", "shard", "bc_update", "bc_update_compiled",
        "q_warmup", "batch_update", "batch_update_compiled",
        "sample_actions", "ppo_update",
    ]

    agent = None
    flat_batch = utd_batch = None

    for step in steps:
        try:
            if step == "create":
                def fn():
                    return OGPOAgent.create(42, ex_obs, ex_act, agent_config, adv_clip_min=None)
                agent, peak, elapsed = run_step(step, fn, devices)
                results["create_gb"] = peak

            elif step == "shard":
                if mesh is None:
                    continue
                def fn():
                    return shard_agent(agent, mesh)
                agent, peak, elapsed = run_step(step, fn, devices)
                results["shard_gb"] = peak

            elif step == "bc_update":
                if flat_batch is None:
                    flat_batch, utd_batch = make_synthetic_data(
                        obs_dim, full_act_dim, batch_size, utd, rng)
                def fn():
                    return agent.bc_update(_shard_flat(flat_batch))
                (agent, _info), peak, elapsed = run_step(step, fn, devices)
                results["bc_peak_gb"] = peak

            elif step == "bc_update_compiled":
                def fn():
                    return agent.bc_update(_shard_flat(flat_batch))
                (agent, _info), peak, elapsed = run_step(step, fn, devices)
                results["bc_peak_gb_compiled"] = peak

            elif step == "q_warmup":
                def fn():
                    return agent.batch_q_warmup_update(_shard_utd(utd_batch))
                (agent, _info), peak, elapsed = run_step(step, fn, devices)
                results["q_warmup_peak_gb"] = peak

            elif step == "batch_update":
                def fn():
                    return agent.batch_update(
                        _shard_utd(utd_batch), _shard_utd(utd_batch), True)
                (agent, _info), peak, elapsed = run_step(step, fn, devices)
                results["batch_update_peak_gb"] = peak

            elif step == "batch_update_compiled":
                def fn():
                    return agent.batch_update(
                        _shard_utd(utd_batch), _shard_utd(utd_batch), True)
                (agent, _info), peak, elapsed = run_step(step, fn, devices)
                results["batch_update_peak_gb_compiled"] = peak

            elif step == "sample_actions":
                single_obs = jnp.zeros((1, obs_dim))
                def fn():
                    return agent.sample_actions(single_obs, jax.random.PRNGKey(0))
                _, peak, elapsed = run_step(step, fn, devices)
                results["sample_peak_gb"] = peak

            elif step == "ppo_update":
                k1, k2 = jax.random.split(jax.random.PRNGKey(99))
                ppo_bs = agent_config["ppo_batch_size"]
                ppo_obs = jax.random.normal(k1, (ppo_bs, obs_dim))
                ppo_act = jax.random.normal(k2, (ppo_bs, full_act_dim))
                ppo_adv = jnp.ones((ppo_bs,))
                ppo_old_logp = jnp.zeros((ppo_bs,))
                def fn():
                    return agent.ppo_update(ppo_obs, ppo_act, ppo_adv, ppo_old_logp)
                (agent, _info), peak, elapsed = run_step(step, fn, devices)
                results["ppo_peak_gb"] = peak

        except Exception as e:
            print(f"\n  FAILED at {step}:")
            traceback.print_exc()
            results["error"] = f"{step}: {e}"
            results["success"] = False
            return results

    results["success"] = True
    report_memory(devices, "final")
    return results


def count_params(embed_dim, num_layers, num_heads, num_qs, action_dim=4, horizon=4, obs_dim=11):
    D = embed_dim
    full_act = action_dim * horizon
    actor_per_layer = 3 * D * D + D * D + 6 * D * D + D * 4 * D + 4 * D * D
    actor_total = obs_dim * D + full_act * D + num_layers * actor_per_layer + 2 * D * D + D * action_dim
    actor_with_target = actor_total * 2
    critic_per_layer = actor_per_layer
    critic_single = obs_dim * D + full_act * D + num_layers * critic_per_layer + D * 1
    critic_total = critic_single * num_qs * 2
    total_params = actor_with_target + critic_total
    total_memory_bytes = total_params * 4 * 3
    return {
        "actor_params": actor_with_target,
        "critic_params": critic_total,
        "total_params": total_params,
        "param_memory_gb": total_memory_bytes / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile max TF size for OGPO on multi-GPU")
    parser.add_argument("--dim", type=int, default=None, help="Single embed_dim to test")
    parser.add_argument("--layers", type=int, default=None, help="Single num_layers to test")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_qs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--utd", type=int, default=4)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--dims", type=str, default="256,384,512,640,768,896,1024")
    parser.add_argument("--layer_range", type=str, default="2,4,6,8")
    parser.add_argument("--obs_dim", type=int, default=11)
    parser.add_argument("--action_dim", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("OGPO Transformer Memory Profiler")
    print("=" * 60)

    devices = jax.devices()
    num_gpus = len(devices)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {num_gpus} × {devices[0].device_kind}")

    has_stats = _check_memory_stats()
    print(f"memory_stats() available: {has_stats}")
    if not has_stats:
        print("  (falling back to nvidia-smi for memory reporting)")

    for d in devices:
        m = get_memory_gb(d)
        print(f"  GPU {d.id}: {m['limit_gb']:.1f} GB total, {m['in_use_gb']:.2f} GB in use")

    from ogpo.utils.sharding import create_mesh
    if num_gpus > 1:
        mesh = create_mesh("multi")
        print(f"Multi-GPU mesh: {num_gpus} devices")
    else:
        mesh = None
        print("Single GPU mode")

    # Sanity check: can we even create a minimal agent?
    print("\n--- Sanity check: creating minimal config ---")
    try:
        cfg = make_agent_config(embed_dim=128, num_layers=2, num_heads=8, num_qs=2, batch_size=32)
        print(f"  Config created OK, keys: {len(cfg)}")
        from ogpo.agents.ogpo import OGPOAgent
        ex_obs = jnp.zeros((args.obs_dim,))
        ex_act = jnp.zeros((args.action_dim,))
        agent = OGPOAgent.create(0, ex_obs, ex_act, cfg, adv_clip_min=None)
        print(f"  Minimal agent created OK")
        del agent
        gc.collect()
        jax.clear_caches()
    except Exception:
        print("  FAILED to create minimal agent!")
        traceback.print_exc()
        print("\nFix the above error before profiling larger configs.")
        sys.exit(1)

    if args.sweep:
        dims = [int(x) for x in args.dims.split(",")]
        layers = [int(x) for x in args.layer_range.split(",")]

        print(f"\nSweep: dims={dims}, layers={layers}, heads={args.heads}, "
              f"num_qs={args.num_qs}, batch={args.batch_size}, utd={args.utd}")

        print(f"\n{'Dim':>5} {'Layers':>6} {'Actor(M)':>9} {'Critic(M)':>10} {'Total(M)':>9} {'ParamMem':>9}")
        print("-" * 55)
        for D in dims:
            for L in layers:
                if D % args.heads != 0:
                    continue
                est = count_params(D, L, args.heads, args.num_qs, args.action_dim, 4, args.obs_dim)
                print(f"{D:>5} {L:>6} {est['actor_params']/1e6:>8.1f}M "
                      f"{est['critic_params']/1e6:>9.1f}M {est['total_params']/1e6:>8.1f}M "
                      f"{est['param_memory_gb']:>8.2f}GB")

        all_results = []
        for D in dims:
            for L in layers:
                if D % args.heads != 0:
                    print(f"\nSkipping D={D}, H={args.heads}: not divisible")
                    continue
                try:
                    r = profile_config(
                        D, L, args.heads, args.num_qs, args.batch_size, args.utd,
                        mesh, devices, args.obs_dim, args.action_dim,
                    )
                    all_results.append(r)
                except Exception as e:
                    print(f"\n  CRASHED (unrecoverable):")
                    traceback.print_exc()
                    all_results.append({
                        "config": f"D={D} L={L} H={args.heads} Qs={args.num_qs} B={args.batch_size} UTD={args.utd}",
                        "embed_dim": D, "num_layers": L,
                        "error": str(e), "success": False,
                    })

                gc.collect()
                jax.clear_caches()

        print(f"\n{'='*90}")
        print("SUMMARY")
        print(f"{'='*90}")
        print(f"{'Config':<40} {'BC Peak':>8} {'Update Peak':>12} {'PPO Peak':>9} {'Status'}")
        print("-" * 90)
        for r in all_results:
            bc = f"{r['bc_peak_gb_compiled']:.2f}GB" if r.get("bc_peak_gb_compiled") else "---"
            up = f"{r['batch_update_peak_gb_compiled']:.2f}GB" if r.get("batch_update_peak_gb_compiled") else "---"
            ppo = f"{r['ppo_peak_gb']:.2f}GB" if r.get("ppo_peak_gb") else "---"
            status = "OK" if r.get("success") else f"FAIL: {r.get('error', '?')}"
            print(f"{r['config']:<40} {bc:>8} {up:>12} {ppo:>9}  {status}")

        passing = [r for r in all_results if r.get("success")]
        if passing:
            best = max(passing, key=lambda r: r["embed_dim"] * r["num_layers"])
            print(f"\nLargest passing config: {best['config']}")
            peak = best.get('batch_update_peak_gb_compiled', best.get('batch_update_peak_gb', '?'))
            print(f"  batch_update peak: {peak} GB")
        else:
            print("\nNo configs passed all tests!")

    else:
        D = args.dim or 512
        L = args.layers or 4
        if D % args.heads != 0:
            print(f"ERROR: dim={D} must be divisible by heads={args.heads}")
            sys.exit(1)

        est = count_params(D, L, args.heads, args.num_qs, args.action_dim, 4, args.obs_dim)
        print(f"\nEstimated params: {est['total_params']/1e6:.1f}M "
              f"(actor={est['actor_params']/1e6:.1f}M, critic={est['critic_params']/1e6:.1f}M)")
        print(f"Estimated param memory (with Adam): {est['param_memory_gb']:.2f} GB")

        r = profile_config(
            D, L, args.heads, args.num_qs, args.batch_size, args.utd,
            mesh, devices, args.obs_dim, args.action_dim,
        )
        if r.get("success"):
            print(f"\nSUCCESS: {r['config']}")
        else:
            print(f"\nFAILED at: {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
