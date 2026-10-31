"""Measure relative compute/memory across OGPO variants, EXPO, QC, DSRL at full online-RL load.

Variants:
  ogpo      — baseline (q_variance_reduction=false, 1 a_next denoising for critic target)
  ogpo_qvr  — baseline with q_variance_reduction=true (N a_next samples; the expensive case V is designed to replace)
  ogpo_v    — Ablation A (V replaces a_next in critic target, V baseline in actor adv)

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python tests/profile_compute.py
    CUDA_VISIBLE_DEVICES=0 uv run python tests/profile_compute.py --obs_dim=23 --action_dim=7
    CUDA_VISIBLE_DEVICES=0 uv run python tests/profile_compute.py --algos=ogpo,ogpo_qvr,ogpo_v
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
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        return max(float(x) for x in out.strip().split("\n")) / 1024
    except Exception:
        return -1

def peak_mem_gb():
    if HAS_MEM_STATS:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices()) / 1e9
    return mem_in_use_gb()  # nvidia-smi only gives current, not peak

def reset_peak():
    if HAS_MEM_STATS:
        for d in jax.devices():
            try: d.clear_memory_stats()
            except: pass


def make_config(algo, obs_dim, action_dim, batch_size, utd, horizon, num_qs,
                best_of_n, flow_steps, group_num_samples):
    from ogpo.utils.config_loader import load_config, config_to_agent_dict
    # ogpo_v and ogpo_qvr reuse the ogpo config and toggle flags below
    base_algo = "ogpo" if algo in ("ogpo_v", "ogpo_qvr") else algo
    base = load_config(base_algo, {})
    c = config_to_agent_dict(base)

    c["batch_size"] = batch_size
    c["horizon_length"] = horizon
    c["action_chunking"] = True
    c["num_qs"] = num_qs
    c["best_of_n"] = best_of_n
    c["flow_steps"] = flow_steps
    c["ft_flow_steps"] = flow_steps
    c["discount"] = 0.99
    c["tau"] = 0.005
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

    if algo in ("ogpo", "ogpo_v", "ogpo_qvr"):
        c["use_denoiser"] = True
        c["use_bc_regularization"] = True
        c["error_correct_ode_to_sde"] = True  # legacy FPO+denoiser correction (renamed from error_correct_sde_to_ode)
        c["adv_strategy"] = "conservative"
        c["ppo_lr"] = 1e-4
        c["group_num_samples"] = group_num_samples
        c["ppo_batch_size"] = batch_size
        c["q_agg"] = "subsample"
        c["subsample_bon"] = True
        c["awr_beta"] = 10
        c["bc_coeff"] = 1.0
        c["q_variance_reduction"] = False
        c["mc_regression"] = False
        if algo == "ogpo_qvr":
            # Expensive baseline: N a_next samples via flow denoising for the critic target
            c["q_variance_reduction"] = True
            c["q_vr_num_samples"] = best_of_n
            c["q_vr_reduction"] = "mean"
        elif algo == "ogpo_v":
            # Ablation A: V trained by off-policy TD, V(s') replaces next_q
            c["use_value_fn"] = True
            c["v_ensemble_size"] = num_qs
            c["v_agg"] = "mean"
            c["v_training_mode"] = "td_offpolicy"
            c["v_loss_coeff"] = 1.0
            c["adv_baseline"] = "v_fn"
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

    # Most agents expect (B, horizon, action_dim); DSRL expects flat (B, horizon*action_dim)
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

    if algo in ("ogpo", "ogpo_v", "ogpo_qvr"):
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
                 num_qs, best_of_n, flow_steps, group_num_samples):
    print(f"\n{'='*60}")
    print(f"  {algo.upper()}")
    print(f"{'='*60}")

    config = make_config(algo, obs_dim, action_dim, batch_size, utd, horizon,
                         num_qs, best_of_n, flow_steps, group_num_samples)

    flat_batch, utd_batch = make_batches(
        obs_dim, action_dim, batch_size, utd, horizon,
        flat_actions=(algo == "dsrl"),
    )

    results = {"algo": algo}

    # Create agent
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

    # Count parameters
    leaves = jax.tree.leaves(agent)
    param_count = sum(x.size for x in leaves if hasattr(x, 'size'))
    results["params_M"] = param_count / 1e6
    print(f"  Parameters: {results['params_M']:.1f}M")

    # --- bc_update (flat batch) ---
    try:
        def bc_fn():
            return agent.bc_update(flat_batch)
        t, mem, (agent_new, _) = time_fn(bc_fn, warmup=1, repeats=3)
        agent = agent_new
        results["bc_ms"] = t * 1000
        results["bc_mem"] = mem
        print(f"  bc_update:      {results['bc_ms']:8.1f} ms  |  peak {mem:.2f} GB")
    except Exception:
        print(f"  bc_update FAILED:")
        traceback.print_exc()
        results["bc_ms"] = -1

    # --- batch_update (UTD batch, full online RL load) ---
    try:
        if algo == "dsrl":
            # DSRL's _update_with_bc traces both cond branches even with
            # success_flag=False, hitting a shape bug on synthetic (B,H,A) actions.
            # Use the no-success-buffer path directly.
            def update_fn():
                return agent.batch_update(utd_batch)
        else:
            def update_fn():
                return agent.batch_update(utd_batch, utd_batch, True)
        t, mem, (agent_new, _) = time_fn(update_fn, warmup=1, repeats=3)
        agent = agent_new
        results["update_ms"] = t * 1000
        results["update_mem"] = mem
        print(f"  batch_update:   {results['update_ms']:8.1f} ms  |  peak {mem:.2f} GB")
    except Exception:
        print(f"  batch_update FAILED:")
        traceback.print_exc()
        results["update_ms"] = -1

    # --- Per-phase split for OGPO variants (diagnostic) ---
    if algo.startswith("ogpo"):
        try:
            def critic_only_fn():
                return agent.batch_q_warmup_update(utd_batch)
            t_c, mem_c, _ = time_fn(critic_only_fn, warmup=1, repeats=3)
            results["critic_only_ms"] = t_c * 1000
            print(f"  critic_only:    {results['critic_only_ms']:8.1f} ms")
        except Exception:
            print(f"  critic_only FAILED:")
            traceback.print_exc()
            results["critic_only_ms"] = -1

        try:
            @jax.jit
            def actor_only_batch(a, b, bs):
                def one_actor_step(ag, bt):
                    bb, bbs = bt
                    new_rng, rng1 = jax.random.split(ag.rng, 2)
                    def actor_loss_fn(p):
                        return ag.actor_total_loss(bb, bbs, p, True, rng1)
                    new_actor_state, info = ag.actor_network.apply_loss_fn(actor_loss_fn)
                    return ag.replace(actor_network=new_actor_state, rng=new_rng), info
                return jax.lax.scan(one_actor_step, a, (b, bs))

            def actor_only_fn():
                return actor_only_batch(agent, utd_batch, utd_batch)
            t_a, mem_a, _ = time_fn(actor_only_fn, warmup=1, repeats=3)
            results["actor_only_ms"] = t_a * 1000
            print(f"  actor_only:     {results['actor_only_ms']:8.1f} ms")
        except Exception:
            print(f"  actor_only FAILED:")
            traceback.print_exc()
            results["actor_only_ms"] = -1

    # --- sample_actions (small batch, inference) ---
    try:
        # Use small batch (not single obs) to avoid BoN reshape edge cases
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


def flop_estimate(algo, obs_dim, action_dim, horizon, batch_size, utd,
                  num_qs, best_of_n, flow_steps, group_num_samples):
    """Rough relative FLOP estimate per online step (analytical)."""
    H = horizon
    D_obs = obs_dim
    D_act = action_dim * H
    B = batch_size
    N_q = num_qs
    N_bon = best_of_n
    F = flow_steps

    # Rough cost units (1 unit = one MLP forward pass through a 4-layer 512-dim network)
    mlp_fwd = 1.0
    mlp_bwd = 2.5  # backward ≈ 2-3x forward

    if algo == "ogpo":
        # Baseline (q_variance_reduction=false): single a_next sample per critic step.
        actor_bc = mlp_fwd + mlp_bwd
        actor_ode_for_critic = F * mlp_fwd           # 1 a_next × F flow steps
        critic_fwd = N_q * mlp_fwd                   # target_Q(s', a_next)
        critic_loss = N_q * (mlp_fwd + mlp_bwd)      # Q TD loss
        per_step = actor_bc + actor_ode_for_critic + critic_fwd + critic_loss
        # PPO/GRPO (once per env step, outside UTD scan):
        ppo = group_num_samples * F * mlp_fwd + group_num_samples * N_q * mlp_fwd + (mlp_fwd + mlp_bwd)
        total = per_step * utd + ppo

    elif algo == "ogpo_qvr":
        # Expensive baseline (q_variance_reduction=true): N a_next samples per critic step.
        # This is the case V-fn is designed to replace.
        actor_bc = mlp_fwd + mlp_bwd
        actor_ode_for_critic = N_bon * F * mlp_fwd   # N a_next × F flow steps
        critic_fwd = N_bon * N_q * mlp_fwd           # target_Q on each of N samples
        critic_loss = N_q * (mlp_fwd + mlp_bwd)
        per_step = actor_bc + actor_ode_for_critic + critic_fwd + critic_loss
        ppo = group_num_samples * F * mlp_fwd + group_num_samples * N_q * mlp_fwd + (mlp_fwd + mlp_bwd)
        total = per_step * utd + ppo

    elif algo == "ogpo_v":
        # Ablation A: V(s') replaces the a_next denoising for the critic target.
        actor_bc = mlp_fwd + mlp_bwd
        v_next_fwd = N_q * mlp_fwd                   # target_V(s') for Q bootstrap
        critic_loss = N_q * (mlp_fwd + mlp_bwd)      # Q TD loss
        v_loss = N_q * (mlp_fwd + mlp_bwd)           # V TD loss
        per_step = actor_bc + v_next_fwd + critic_loss + v_loss
        # PPO/GRPO: G flow samples + Q eval + target_V(s) baseline + 1 fwd+bwd
        v_adv_fwd = N_q * mlp_fwd
        ppo = group_num_samples * F * mlp_fwd + group_num_samples * N_q * mlp_fwd + v_adv_fwd + (mlp_fwd + mlp_bwd)
        total = per_step * utd + ppo

    elif algo == "expo":
        # Per scan step:
        #   Critic: OTF next actions (N_bon*F fwd actor + N_bon fwd edit + (N_bon+M)*N_q fwd critic) + TD
        #   Edit: F fwd actor (base actions) + 1 fwd edit + N_q fwd critic + bwd edit
        #   Temp: trivial
        #   Base BC: 1 fwd+bwd actor
        N_edit = N_bon
        otf_cost = N_bon * F * mlp_fwd + N_edit * mlp_fwd + (N_bon + N_edit) * N_q * mlp_fwd
        critic_loss = N_q * (mlp_fwd + mlp_bwd)
        edit_loss = mlp_fwd + mlp_bwd + N_q * mlp_fwd  # edit fwd/bwd + critic fwd
        base_bc = mlp_fwd + mlp_bwd
        per_step = otf_cost + critic_loss + edit_loss + base_bc
        total = per_step * utd

    elif algo == "qc":
        # Per scan step (combined loss, single backward):
        #   Critic: BoN (N_bon * F * fwd actor + N_q * fwd critic for scoring) + TD (N_q fwd)
        #   Actor: BC flow (1 fwd) + BoN SFT (N_bon * F * fwd actor + 1 fwd) or distill
        #   Single backward through everything
        bon_sampling = N_bon * F * mlp_fwd + N_q * mlp_fwd  # for critic next actions
        critic_td = N_q * mlp_fwd
        actor_bc = mlp_fwd
        actor_bon_sft = N_bon * F * mlp_fwd + mlp_fwd  # if alpha > 0
        fwd_total = bon_sampling + critic_td + actor_bc + actor_bon_sft
        bwd_total = fwd_total * 2.5  # single backward through all
        per_step = fwd_total + bwd_total
        total = per_step * utd

    elif algo == "dsrl":
        # Per scan step:
        #   Actor: F fwd (sampling next actions) + 1 fwd+bwd
        #   Critic: N_q fwd+bwd
        #   Temperature: trivial
        actor_sampling = F * mlp_fwd
        actor_loss = mlp_fwd + mlp_bwd
        critic_loss = N_q * (mlp_fwd + mlp_bwd)
        per_step = actor_sampling + actor_loss + critic_loss
        total = per_step * utd

    else:
        total = 0

    return total


def main():
    parser = argparse.ArgumentParser(description="Profile compute across OGPO/EXPO/QC/DSRL")
    parser.add_argument("--obs_dim", type=int, default=23)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--utd", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--num_qs", type=int, default=10)
    parser.add_argument("--best_of_n", type=int, default=8)
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--group_num_samples", type=int, default=16)
    parser.add_argument("--algos", type=str, default="ogpo,ogpo_qvr,ogpo_v,expo,qc,dsrl")
    parser.add_argument("--analytical_only", action="store_true",
                        help="Skip actual profiling, just print FLOP estimates")
    args = parser.parse_args()

    algos = [a.strip() for a in args.algos.split(",")]

    print("=" * 65)
    print("  OGPO Compute Profiler — Online RL Full Load")
    print("=" * 65)
    print(f"  obs_dim={args.obs_dim}  action_dim={args.action_dim}  "
          f"horizon={args.horizon}  batch={args.batch_size}  utd={args.utd}")
    print(f"  num_qs={args.num_qs}  best_of_n={args.best_of_n}  "
          f"flow_steps={args.flow_steps}  group_samples={args.group_num_samples}")
    print(f"  algos: {algos}")

    # --- Analytical FLOP estimates ---
    print(f"\n{'='*65}")
    print("  ANALYTICAL ESTIMATES (relative cost units per env step)")
    print(f"{'='*65}")
    estimates = {}
    for algo in algos:
        est = flop_estimate(algo, args.obs_dim, args.action_dim, args.horizon,
                            args.batch_size, args.utd, args.num_qs, args.best_of_n,
                            args.flow_steps, args.group_num_samples)
        estimates[algo] = est

    base = estimates.get("ogpo", estimates.get(algos[0], 1))
    print(f"\n  {'Algo':<8} {'Cost':>10} {'Relative':>10}")
    print(f"  {'-'*30}")
    for algo in algos:
        rel = estimates[algo] / base if base > 0 else 0
        print(f"  {algo:<8} {estimates[algo]:>10.1f} {rel:>9.2f}x")

    if args.analytical_only:
        return

    # --- Actual profiling ---
    _init_mem()
    devices = jax.devices()
    print(f"\n  JAX: {jax.__version__}, {len(devices)}x {devices[0].device_kind}")
    print(f"  memory_stats: {'yes' if HAS_MEM_STATS else 'no (using nvidia-smi)'}")

    all_results = {}
    for algo in algos:
        gc.collect()
        jax.clear_caches()
        try:
            r = profile_algo(algo, args.obs_dim, args.action_dim, args.batch_size,
                             args.utd, args.horizon, args.num_qs, args.best_of_n,
                             args.flow_steps, args.group_num_samples)
            all_results[algo] = r
        except Exception:
            print(f"\n  {algo.upper()} CRASHED:")
            traceback.print_exc()
            all_results[algo] = {"algo": algo, "error": "crash"}

    # --- Summary ---
    print(f"\n{'='*75}")
    print("  SUMMARY — batch_update (full online RL step)")
    print(f"{'='*75}")
    print(f"  {'Algo':<10} {'Params':>8} {'bc':>8} {'actor':>8} {'critic':>8} {'batch':>8} {'sample':>8} {'Mem':>8} {'Rel':>8}")
    print(f"  {'-'*82}")

    base_time = all_results.get("ogpo", {}).get("update_ms", None)
    if base_time is None:
        base_time = next((r.get("update_ms") for r in all_results.values() if r.get("update_ms", -1) > 0), 1)

    for algo in algos:
        r = all_results.get(algo, {})
        if "error" in r:
            print(f"  {algo:<10} {'FAILED':>8}")
            continue
        params = f"{r.get('params_M', 0):.1f}M"
        bc = f"{r['bc_ms']:.0f}ms" if r.get("bc_ms", -1) > 0 else "---"
        ac = f"{r['actor_only_ms']:.0f}ms" if r.get("actor_only_ms", -1) > 0 else "---"
        cr = f"{r['critic_only_ms']:.0f}ms" if r.get("critic_only_ms", -1) > 0 else "---"
        upd = f"{r['update_ms']:.0f}ms" if r.get("update_ms", -1) > 0 else "---"
        smp = f"{r['sample_ms']:.1f}ms" if r.get("sample_ms", -1) > 0 else "---"
        mem = f"{r.get('update_mem', -1):.2f}GB" if r.get("update_mem", -1) > 0 else "---"
        rel = f"{r['update_ms']/base_time:.2f}x" if r.get("update_ms", -1) > 0 and base_time > 0 else "---"
        print(f"  {algo:<10} {params:>8} {bc:>8} {ac:>8} {cr:>8} {upd:>8} {smp:>8} {mem:>8} {rel:>8}")

    # jax_frac recommendations
    print(f"\n  --- jax_frac recommendations ---")
    if HAS_MEM_STATS:
        limit_gb = jax.devices()[0].memory_stats()["bytes_limit"] / 1e9
    else:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
            limit_gb = max(float(x) for x in out.strip().split("\n")) / 1024
        except Exception:
            limit_gb = 0
    if limit_gb > 0:
        print(f"  GPU total: {limit_gb:.0f}GB")
        print(f"  Note: 'delta mem' is peak-during-op minus baseline (activation/JIT overhead)")
        for algo in algos:
            r = all_results.get(algo, {})
            params_gb = r.get("params_M", 0) * 4 * 3 / 1000  # params + adam m,v in GB
            delta = r.get("update_mem", 0)
            total = params_gb + max(delta, 0) + 0.5  # params + activations + headroom
            frac = min(total / limit_gb, 0.95)
            print(f"  {algo:<8} params≈{params_gb:.1f}GB + activations≈{max(delta,0):.1f}GB → ~{total:.1f}GB → jax_frac={frac:.2f}")
    else:
        print("  Could not determine GPU memory limit")


if __name__ == "__main__":
    main()
