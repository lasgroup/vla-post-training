#!/usr/bin/env python
"""Noise-level sweep: does a higher sampling noise make Best-of-N's signal real?

READ-ONLY -- trains nothing, launches nothing, touches no simulator.

Tier B (scripts/probe_policy_candidate_spread.py) measured where real pi0.5
candidates sit on the Tier-A eps axis when collection samples them the way it
does today: create_trained_policy(...) with no `sample_kwargs`
(filtered_sft_learner.py:368) => model.sample_actions defaults num_steps=10,
noise_level=0.0, i.e. the DETERMINISTIC ODE. All candidate diversity comes from
the per-row initial Gaussian draw. Measured eps ~= 0.074 (PG arms), which puts
the within-state Q spread at ~16% of the critic's own MC error.

This script re-runs the Tier-B battery across a ladder of noise_level values to
answer: is there a noise level at which the spread the critic ranks by clears
its own error bar, and what does that cost?

TWO THINGS TO KEEP IN MIND WHEN READING THE OUTPUT
1. The pi0 SDE is marginal-preserving by construction (pi0.py:147-166: the drift
   carries the sigma^2/2 * score term), so in the continuous-time limit raising
   noise_level does NOT change the terminal action distribution at all. Any extra
   spread it produces at num_steps=10 is discretisation error. That is precisely
   why this has to be measured rather than reasoned about.
2. `q_candidates_mean` as a function of noise level is a CIRCULAR quality proxy --
   it is the same critic whose ranking is under test. `dist_from_ode_action` is
   the non-circular half: how far the sampler has been pushed off the policy's own
   mode. Neither is a substitute for an eval or for Tier C.

REFERENCE CALIBRATION. OGPO_public/scripts/ogpo/*.sh (all 15, incl. the three
PaliGemma recipes) use use_tapered_noise=true with constant_noise_std=0.05 and
flow_steps=10 -- note min/max_noise_std=0.01 are inert there, they only feed the
unused NoiseInjectionNetwork. Its per-step injected std is sigma_i = 0.05*sqrt(1-i/10)
applied directly (pg_helper.py: `distrax.Normal(mean_next, sigma)`, NO sqrt(dt)),
whereas ours is noise_level*sqrt(t/(1-t))*sqrt(|dt|) (pi0.py:160-166). Matching
total injected std over the chain gives the equivalence printed at startup.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys

import h5py
import jax
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import flax.nnx as nnx  # noqa: E402
import openpi.models.model as _model  # noqa: E402
import src.training.config as _config  # noqa: E402
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME  # noqa: E402

# Reuse Tier A/B rather than cloning (duplication-by-copy is this repo's
# documented signature bug class -- CLAUDE.md, Decisions log OQ-2).
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tier_a = _load("tier_a", "scripts/probe_critic_action_sensitivity.py")
tier_b = _load("tier_b", "scripts/probe_policy_candidate_spread.py")

REAL_ACT_DIM = tier_a.REAL_ACT_DIM
MODEL_ACT_DIM = tier_a.MODEL_ACT_DIM
ACTION_HORIZON = tier_a.ACTION_HORIZON

DEFAULT_LADDER = [0.0, 0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5]


def ours_total_injected_std(noise_level: float, num_steps: int = 10) -> float:
    """Total std injected over our SDE chain (pi0.py:158-166), per action dim.

    time runs 1 -> 0 with dt = -1/num_steps and is clipped to <= 1-|dt|, so the
    first two steps share t = 1-|dt|. Increments are independent, so the totals
    add in variance.
    """
    dt = 1.0 / num_steps
    var = 0.0
    for i in range(num_steps):
        t = min(1.0 - i * dt, 1.0 - dt)
        sigma_t = noise_level * math.sqrt(t / (1.0 - t))
        var += (sigma_t * math.sqrt(dt)) ** 2
    return math.sqrt(var)


def reference_total_injected_std(sigma_base: float = 0.05, flow_steps: int = 10) -> float:
    """Total std injected over the reference's tapered SDE chain, per action dim.

    OGPO_public/ogpo/agents/modules/pg_helper.py: sigma_i = sigma_base*sqrt(1-t_i)
    with t_i = i/flow_steps, used directly as the Normal scale (no sqrt(dt)).
    """
    return math.sqrt(sum((sigma_base * math.sqrt(1.0 - i / flow_steps)) ** 2
                         for i in range(flow_steps)))


def reference_equivalent_noise_level(sigma_base: float = 0.05, steps: int = 10) -> float:
    """Our noise_level with the same total injected std as the reference recipe."""
    return reference_total_injected_std(sigma_base, steps) / ours_total_injected_std(1.0, steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--config-name", default="pi05_libero_online_ogpo_ref")
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--num-states", type=int, default=128)
    ap.add_argument("--group", type=int, default=8, help="candidates per state (BON_N)")
    ap.add_argument("--chunk", type=int, default=4, help="states per policy forward")
    ap.add_argument("--noise-levels", nargs="+", type=float, default=DEFAULT_LADDER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_noise_level_sweep.json")
    args = ap.parse_args()

    ref_std = reference_total_injected_std()
    ref_nl = reference_equivalent_noise_level()
    print(f"[sweep] jax devices: {jax.devices()}", flush=True)
    print(f"[sweep] reference (OGPO_public, tapered sigma_base=0.05, 10 steps): "
          f"total injected std {ref_std:.4f} per dim", flush=True)
    print(f"[sweep] our equivalent noise_level = {ref_nl:.4f} "
          f"(recipes currently pass --rl.noise_level 0.02, and COLLECTION uses 0.0)",
          flush=True)
    for nl in args.noise_levels:
        print(f"          noise_level {nl:<5} -> total injected std "
              f"{ours_total_injected_std(nl):.4f}", flush=True)

    cfg = _config.get_config(args.config_name)
    root = pathlib.Path(args.ckpt_root)
    G = args.group
    real = tier_a.act_real_index()
    results = []
    structure_log: list[str] = []

    for run in args.runs:
        rd = root / run
        shard = rd / "runtime_state" / "replay_shards" / f"step_{args.step:08d}.h5"
        params_dir = rd / str(args.step) / "params"
        print(f"\n[sweep] ===== {run} =====\n  shard  {shard}\n  params {params_dir}", flush=True)

        with h5py.File(shard, "r") as f:
            obs_idx = np.asarray(f["links"]["obs_index"][()], dtype=np.int64)
            buf_act = np.asarray(f["transitions"]["actions"][()], dtype=np.float32)
            mc = np.asarray(f["transitions"]["mc_return"][()], dtype=np.float32)
            is_succ = np.asarray(f["transitions"]["is_success"][()], dtype=np.float32)

        rng_np = np.random.default_rng(args.seed)
        n_take = min(args.num_states, obs_idx.shape[0])
        n_take -= n_take % args.chunk          # keep the jit shape fixed; no ragged tail
        sel = np.sort(rng_np.choice(obs_idx.shape[0], size=n_take, replace=False))
        rows = obs_idx[sel]
        uniq, back = np.unique(rows, return_inverse=True)
        obs_u = tier_b.load_obs_rows(shard, uniq)
        obs = jax.tree.map(lambda v: np.asarray(v)[back],
                           {k: v for k, v in obs_u.items() if k != "_obs_keys"})
        S = sel.size
        print(f"  states={S} group={G} chunk={args.chunk}", flush=True)

        model = cfg.model.load(_model.restore_params(params_dir))
        critic = tier_a.load_critic(rd / "rl_state" / str(args.step), "ema_params", structure_log)

        @nnx.jit(static_argnames=("num_steps", "noise_level"))
        def sample_jit(m, rng, observation, noise, num_steps=10, noise_level=0.0):
            return m.sample_actions(rng, observation, num_steps=num_steps,
                                    noise=noise, noise_level=noise_level)

        pf = np.repeat(obs[PREFIX_EMBEDDING_NAME], G, axis=0)
        st = np.repeat(obs["state"], G, axis=0)
        q_buf = tier_a.batched_q(critic, obs[PREFIX_EMBEDDING_NAME], obs["state"],
                                 buf_act[sel].reshape(S, -1)).mean(axis=0)
        mc_rmse = float(np.sqrt(np.mean((q_buf - mc[sel]) ** 2)))
        succ = is_succ[sel] > 0.5
        print(f"  critic MC rmse {mc_rmse:.4g}   n_success {int(succ.sum())}/{S}", flush=True)

        run_res = {
            "run": run, "n_states": int(S), "group": int(G),
            "q_mc_rmse": mc_rmse,
            "reference_equivalent_noise_level": ref_nl,
            "buffer_action_per_dim_std_over_dataset": float(
                buf_act.reshape(buf_act.shape[0], -1)[:, real].std(axis=0).mean()),
            "levels": [],
        }
        ode_flat = None

        for nl in args.noise_levels:
            # Every state uses the SAME initial-noise draw across levels, so the
            # only thing changing along the ladder is the SDE injection itself.
            key = jax.random.key(args.seed)
            cands = np.zeros((S, G, ACTION_HORIZON, MODEL_ACT_DIM), np.float32)
            for lo in range(0, S, args.chunk):
                hi = lo + args.chunk
                sub = jax.tree.map(lambda v: v[lo:hi], obs)
                tiled = tier_b.tile_obs(sub, G)
                n = (hi - lo) * G
                key, nk, sk = jax.random.split(key, 3)
                noise = jax.random.normal(nk, (n, ACTION_HORIZON, MODEL_ACT_DIM))
                o = _model.Observation.from_dict(tiled)
                a = np.asarray(sample_jit(model, sk, o, noise, num_steps=10,
                                          noise_level=float(nl)), np.float32)
                cands[lo:hi] = a.reshape(hi - lo, G, ACTION_HORIZON, MODEL_ACT_DIM)
            cands[..., REAL_ACT_DIM:] = 0.0
            flat = cands.reshape(S, G, -1)
            if nl == 0.0 and ode_flat is None:
                ode_flat = flat.copy()

            per_dim_std = flat[:, :, real].std(axis=1, ddof=1)
            q = tier_a.batched_q(critic, pf, st, flat.reshape(S * G, -1)).reshape(-1, S, G)
            qm = q.mean(axis=0)
            summ = tier_a.summarize(f"nl={nl}", q)
            lvl = {
                "noise_level": float(nl),
                "total_injected_std": ours_total_injected_std(float(nl)),
                "eps_equivalent": float(per_dim_std.mean()),
                "eps_equivalent_median": float(np.median(per_dim_std)),
                "sigma_within": summ["sigma_within_mean"],
                "sigma_across": summ["sigma_across"],
                "rho_noise": summ["sigma_within_mean"] / mc_rmse,
                "per_head_sigma_within": summ["per_head_sigma_within_mean"],
                "between_head_corr": summ["between_head_centred_corr"],
                "cons_zero_frac": summ["cons_zero_frac"],
                "q_argmax_minus_mean": float((qm.max(axis=1) - qm.mean(axis=1)).mean()),
                "gain_over_mc_rmse": float((qm.max(axis=1) - qm.mean(axis=1)).mean() / mc_rmse),
                # Quality proxies. q_candidates_mean is CIRCULAR (same critic);
                # dist_from_ode_action is not -- it is displacement off the mode.
                "q_candidates_mean": float(qm.mean()),
                "dist_from_ode_action": (
                    None if ode_flat is None else
                    float(np.sqrt(((flat[:, :, real] - ode_flat[:, :, real]) ** 2)
                                  .sum(-1)).mean())
                ),
                "sigma_within_success": (float(qm[succ].std(axis=1, ddof=1).mean())
                                         if succ.any() else None),
                "sigma_within_failure": (float(qm[~succ].std(axis=1, ddof=1).mean())
                                         if (~succ).any() else None),
            }
            run_res["levels"].append(lvl)
            print(f"  nl={nl:<5} eps={lvl['eps_equivalent']:.4f}  "
                  f"sigma_within={lvl['sigma_within']:.4g}  rho_noise={lvl['rho_noise']:.4f}  "
                  f"gain/rmse={lvl['gain_over_mc_rmse']:.4f}  "
                  f"Qmean={lvl['q_candidates_mean']:.4g}  "
                  f"d_ode={lvl['dist_from_ode_action']}", flush=True)
            # Flush after every level: this runs on `preempt`.
            pathlib.Path(args.out).write_text(
                json.dumps(results + [run_res], indent=2))

        results.append(run_res)
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\n[sweep] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
