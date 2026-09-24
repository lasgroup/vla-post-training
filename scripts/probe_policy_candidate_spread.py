#!/usr/bin/env python
"""Tier-B probe: how far apart are the action chunks Best-of-N actually chooses between?

READ-ONLY -- trains nothing, launches nothing, touches no simulator.

Tier A (scripts/probe_critic_action_sensitivity.py) measured the critic's action
RESPONSE SURFACE with synthetic Gaussian perturbations of size eps. It left one
question open: where on that eps axis do real pi0.5 candidates sit? That number
converts the Tier-A ladder into a verdict on Best-of-N, so this script measures it.

Method. The replay shard stores the fully-transformed model observation (uint8
images, tokenized prompt, normalised+padded state), so the policy can be replayed
on exactly the states the critic was trained on without a simulator. Collection
builds its policy via create_trained_policy(...) with no `sample_kwargs`
(filtered_sft_learner.py:368), i.e. model.sample_actions defaults:
num_steps=10, noise_level=0.0. Candidate diversity therefore comes ENTIRELY from
the per-row initial Gaussian noise draw (filtered_sft_learner.py:534-536), which
is reproduced here exactly.

sample_actions returns actions in the normalised model space the critic trained
on, so no unnormalise/renormalise round-trip is needed -- only the zeroing of
dims 7..31, which the collection path gets for free by slicing to 7 dims for the
env (libero_policy.py:104) and re-padding for the critic.
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
import jax.numpy as jnp
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import flax.nnx as nnx  # noqa: E402
import openpi.models.model as _model  # noqa: E402
import src.training.config as _config  # noqa: E402
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME  # noqa: E402

# Reuse the Tier-A machinery rather than cloning it (duplication-by-copy is this
# repo's documented signature bug class -- CLAUDE.md, Decisions log OQ-2).
_spec = importlib.util.spec_from_file_location(
    "tier_a", _ROOT / "scripts" / "probe_critic_action_sensitivity.py"
)
tier_a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tier_a)

REAL_ACT_DIM = tier_a.REAL_ACT_DIM
MODEL_ACT_DIM = tier_a.MODEL_ACT_DIM
ACTION_HORIZON = tier_a.ACTION_HORIZON


def load_obs_rows(shard: pathlib.Path, rows: np.ndarray) -> dict:
    """Reads the model observation for `rows` (sorted) plus the critic's prefix."""
    out: dict = {}
    with h5py.File(shard, "r") as f:
        o = f["observations"]
        keys = sorted(o.keys())
        out["_obs_keys"] = keys
        out["image"] = {k: np.asarray(o["image"][k][rows]) for k in sorted(o["image"].keys())}
        out["image_mask"] = {
            k: np.asarray(o["image_mask"][k][rows]) for k in sorted(o["image_mask"].keys())
        }
        out["state"] = np.asarray(o["state"][rows], dtype=np.float32)
        for k in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
            if k in o:
                out[k] = np.asarray(o[k][rows])
        out[PREFIX_EMBEDDING_NAME] = np.asarray(o[PREFIX_EMBEDDING_NAME][rows], dtype=np.float32)
    return out


def tile_obs(obs: dict, g: int) -> dict:
    """Repeats every leaf g times along the batch axis (matches collection tiling)."""
    def rep(v):
        return np.repeat(np.asarray(v), g, axis=0)
    out = {
        "image": {k: rep(v) for k, v in obs["image"].items()},
        "image_mask": {k: rep(v) for k, v in obs["image_mask"].items()},
        "state": rep(obs["state"]),
    }
    for k in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
        if k in obs:
            out[k] = rep(obs[k])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--config-name", default="pi05_libero_online_ogpo_ref")
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--num-states", type=int, default=256)
    ap.add_argument("--group", type=int, default=8, help="candidates per state (BON_N)")
    ap.add_argument("--chunk", type=int, default=64, help="policy rows per forward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_policy_candidate_spread.json")
    args = ap.parse_args()

    print(f"[tierB] jax devices: {jax.devices()}", flush=True)
    cfg = _config.get_config(args.config_name)
    print(f"[tierB] model: {type(cfg.model).__name__} "
          f"action_horizon={cfg.model.action_horizon} action_dim={cfg.model.action_dim}", flush=True)

    root = pathlib.Path(args.ckpt_root)
    G = args.group
    real = tier_a.act_real_index()
    results = []
    structure_log: list[str] = []

    for run in args.runs:
        rd = root / run
        shard = rd / "runtime_state" / "replay_shards" / f"step_{args.step:08d}.h5"
        params_dir = rd / str(args.step) / "params"
        print(f"\n[tierB] ===== {run} =====\n  shard  {shard}\n  params {params_dir}", flush=True)

        with h5py.File(shard, "r") as f:
            obs_idx = np.asarray(f["links"]["obs_index"][()], dtype=np.int64)
            buf_act = np.asarray(f["transitions"]["actions"][()], dtype=np.float32)
            mc = np.asarray(f["transitions"]["mc_return"][()], dtype=np.float32)
            is_succ = np.asarray(f["transitions"]["is_success"][()], dtype=np.float32)
        rng_np = np.random.default_rng(args.seed)
        sel = np.sort(rng_np.choice(obs_idx.shape[0], size=min(args.num_states, obs_idx.shape[0]),
                                    replace=False))
        rows = obs_idx[sel]
        # h5py fancy indexing requires strictly increasing indices, and two
        # transitions can reference the same observation row -- read the unique
        # rows once, then scatter back to the per-transition order.
        uniq, back = np.unique(rows, return_inverse=True)
        obs_u = load_obs_rows(shard, uniq)
        obs = jax.tree.map(lambda v: np.asarray(v)[back],
                           {k: v for k, v in obs_u.items() if k != "_obs_keys"})
        S = sel.size
        print(f"  states={S} group={G}  image keys={sorted(obs['image'])} "
              f"shape={next(iter(obs['image'].values())).shape}", flush=True)

        params = _model.restore_params(params_dir)
        model = cfg.model.load(params)

        @nnx.jit(static_argnames=("num_steps",))
        def sample_jit(m, rng, observation, noise, num_steps=10):
            return m.sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        print("  policy loaded", flush=True)

        # ---- sample G candidates per state, collection settings exactly -------
        key = jax.random.key(args.seed)
        cands = np.zeros((S, G, ACTION_HORIZON, MODEL_ACT_DIM), np.float32)
        for lo in range(0, S, args.chunk):
            hi = min(lo + args.chunk, S)
            if hi - lo != args.chunk:
                # Ragged tail would force a jit retrace of a 3B model; drop it and say so.
                print(f"    dropping ragged tail of {hi - lo} states (jit shape stability)", flush=True)
                cands = cands[:lo]; S = lo; sel = sel[:lo]; break
            sub = jax.tree.map(lambda v: v[lo:hi], obs)
            tiled = tile_obs(sub, G)
            n = (hi - lo) * G
            key, nk, sk = jax.random.split(key, 3)
            noise = jax.random.normal(nk, (n, ACTION_HORIZON, MODEL_ACT_DIM))
            o = _model.Observation.from_dict(tiled)
            a = np.asarray(sample_jit(model, sk, o, noise, num_steps=10), np.float32)
            cands[lo:hi] = a.reshape(hi - lo, G, ACTION_HORIZON, MODEL_ACT_DIM)
            print(f"    sampled {hi}/{S}", flush=True)

        obs = jax.tree.map(lambda v: v[:S], obs)
        # The env only ever sees dims 0..6; the buffer re-pads the rest with zeros.
        cands[..., REAL_ACT_DIM:] = 0.0
        flat = cands.reshape(S, G, -1)                       # (S, G, 320)

        # ---- how far apart are the candidates, in the Tier-A eps unit? --------
        per_dim_std = flat[:, :, real].std(axis=1, ddof=1)   # (S, 70)
        eps_equiv = float(per_dim_std.mean())
        res = {
            "run": run,
            "n_states": int(S),
            "group": int(G),
            "eps_equivalent": eps_equiv,
            "eps_equivalent_median": float(np.median(per_dim_std)),
            "candidate_pairwise_rms_dist": float(
                np.sqrt(2.0) * per_dim_std.mean() * math.sqrt(real.size)
            ),
            "buffer_action_per_dim_std_over_dataset": float(
                buf_act.reshape(buf_act.shape[0], -1)[:, real].std(axis=0).mean()
            ),
        }

        # ---- score them -------------------------------------------------------
        critic = tier_a.load_critic(rd / "rl_state" / str(args.step), "ema_params", structure_log)
        pf = np.repeat(obs[PREFIX_EMBEDDING_NAME], G, axis=0)
        st = np.repeat(obs["state"], G, axis=0)
        q = tier_a.batched_q(critic, pf, st, flat.reshape(S * G, -1)).reshape(-1, S, G)
        res["policy_candidates"] = tier_a.summarize("policy_candidates", q)

        qm = q.mean(axis=0)                                   # (S, G) ensemble mean
        q_buf = tier_a.batched_q(critic, obs[PREFIX_EMBEDDING_NAME], obs["state"],
                                 buf_act[sel].reshape(S, -1)).mean(axis=0)
        mc_rmse = float(np.sqrt(np.mean((q_buf - mc[sel]) ** 2)))
        res["q_mc_rmse"] = mc_rmse
        res["policy_candidates"]["rho_noise"] = res["policy_candidates"]["sigma_within_mean"] / mc_rmse
        res["bon"] = {
            # What Best-of-N actually gains, in Q units, over picking at random.
            "q_argmax_minus_mean": float((qm.max(axis=1) - qm.mean(axis=1)).mean()),
            "q_argmax_minus_min": float((qm.max(axis=1) - qm.min(axis=1)).mean()),
            "gain_over_mc_rmse": float((qm.max(axis=1) - qm.mean(axis=1)).mean() / mc_rmse),
            "q_candidates_mean": float(qm.mean()),
            "q_buffer_action_mean": float(q_buf.mean()),
        }
        succ = is_succ[sel] > 0.5
        res["sigma_within_by_success"] = {
            "success": {"n": int(succ.sum()),
                        "sigma": float(qm[succ].std(axis=1, ddof=1).mean()) if succ.any() else None},
            "failure": {"n": int((~succ).sum()),
                        "sigma": float(qm[~succ].std(axis=1, ddof=1).mean()) if (~succ).any() else None},
        }
        results.append(res)
        print(json.dumps(res, indent=2), flush=True)
        # Flush after every arm: this probe runs on `preempt`, so a kill between
        # arms must not lose the ones already finished.
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[tierB] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
