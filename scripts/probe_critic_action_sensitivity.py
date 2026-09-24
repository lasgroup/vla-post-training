#!/usr/bin/env python
"""Tier-A critic action-sensitivity probe. READ-ONLY -- trains nothing, launches nothing.

Measures how much Q varies ACROSS ACTIONS at a fixed state: the quantity that
Best-of-N selection and the group-centred policy gradient both consume, and the
one `q_mc_corr` (an across-state statistic) does not measure.

Needs no simulator and no pi0.5 forward pass. `--collect.store_prefix_rep` puts
the critic's complete input in the replay shard, so the shard plus rl_state/<step>
is everything:

    x = [ prefix_embedding 2048 | state 32 | action 10*32=320 ]   -> 2400

Within that 2400, only 2048 + 8 + 70 = 2126 coordinates carry signal. The other
274 (state[8:32] and action dims 7..31 of each of the 10 timesteps) are
identically zero for every row ever seen, so their columns of `proj.kernel` have
received exactly zero gradient for the whole run (dL/dW_ij = delta_j * x_i). They
are a free, perfectly-matched null sample for "how far did this column move from
init" -- used below as the baseline for the weight-norm comparison.

Usage (see scripts/probe_critic_action_sensitivity.sbatch):
    uv run scripts/probe_critic_action_sensitivity.py \
        --ckpt-root /path/to/checkpoints/ogpo_multitask_4task/pi05_libero_online_ogpo_ref \
        --runs mt4_ref_s0 mt4_ref_nc_s0 mt4_ref_nopg_s0 --step 100000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import flax.nnx as nnx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.rl.networks.bronet_critic import BroNetStateActionCritic  # noqa: E402
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME  # noqa: E402

# Architecture, from scripts/ogpo_multitask_4task.sh:247-248 and CriticTrainingConfig.
HIDDEN_DIM = 1024
DEPTH = 2
NUM_QS = 10
ACTION_HORIZON = 10
MODEL_ACT_DIM = 32
REAL_ACT_DIM = 7   # LIBERO: src/envs/libero.py:79-80
REAL_STATE_DIM = 8  # LIBERO: src/envs/wrappers.py:92-99


# --------------------------------------------------------------------------- io


def load_shard(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Reads only the datasets the critic consumes; never touches the image groups."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        obs = f["observations"]
        if PREFIX_EMBEDDING_NAME not in obs:
            raise KeyError(
                f"{path} has no observations/{PREFIX_EMBEDDING_NAME}. The run was "
                "launched without --collect.store_prefix_rep, so the critic's input "
                "cannot be reconstructed without a pi0.5 forward pass. Re-run the "
                "probe against a shard from a store_prefix_rep run."
            )
        out["prefix"] = np.asarray(obs[PREFIX_EMBEDDING_NAME][()], dtype=np.float32)
        out["state"] = np.asarray(obs["state"][()], dtype=np.float32)
        txn = f["transitions"]
        for k in ("actions", "reward", "discount", "mc_return", "is_success"):
            if k in txn:
                out[k] = np.asarray(txn[k][()], dtype=np.float32)
        if "task_index" in txn:
            out["task_index"] = np.asarray(txn["task_index"][()], dtype=np.int32)
        out["obs_index"] = np.asarray(f["links"]["obs_index"][()], dtype=np.int64)
    return out


def dump_tree(prefix: str, node, lines: list[str], depth: int = 0) -> None:
    if isinstance(node, dict):
        for k in sorted(node, key=str):
            dump_tree(f"{prefix}/{k}", node[k], lines, depth + 1)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            dump_tree(f"{prefix}/{i}", v, lines, depth + 1)
    else:
        shape = getattr(node, "shape", None)
        lines.append(f"  {prefix}  shape={shape} dtype={getattr(node, 'dtype', type(node).__name__)}")


def normalize_restored(node):
    """Adapts an orbax-restored tree back to the shape nnx.update expects.

    Two structural mismatches are possible and both are mechanical, not errors:
    orbax serialises the integer keys of `nets` (a Python list in the module) as
    decimal strings, and an nnx VariableState may round-trip as a single-key
    {"value": array} wrapper.
    """
    if isinstance(node, dict):
        if set(node) in ({"value"}, {"raw_value"}):
            return normalize_restored(next(iter(node.values())))
        out = {}
        for k, v in node.items():
            key = int(k) if isinstance(k, str) and k.isdigit() else k
            out[key] = normalize_restored(v)
        return out
    if isinstance(node, (list, tuple)):
        return {i: normalize_restored(v) for i, v in enumerate(node)}
    return node


def build_critic(rngs_seed: int) -> BroNetStateActionCritic:
    dummy_obs = {
        PREFIX_EMBEDDING_NAME: jnp.zeros((1, 2048), jnp.float32),
        "state": jnp.zeros((1, MODEL_ACT_DIM), jnp.float32),
    }
    dummy_act = jnp.zeros((1, ACTION_HORIZON * MODEL_ACT_DIM), jnp.float32)
    return BroNetStateActionCritic(
        observation=dummy_obs, action=dummy_act, hidden_dim=HIDDEN_DIM,
        depth=DEPTH, num_qs=NUM_QS, rngs=nnx.Rngs(rngs_seed),
    )


def load_critic(ckpt_path: pathlib.Path, which: str, structure_log: list[str]):
    """Restores the Q-ensemble. `which` in {"ema_params", "params"}."""
    raw = ocp.StandardCheckpointer().restore(ckpt_path)
    if not structure_log:
        dump_tree("", raw, structure_log)
    if "state_action_critic_state" not in raw:
        raise KeyError(
            f"{ckpt_path} top-level keys are {sorted(raw)}; expected "
            "'state_action_critic_state' (see _rl_checkpoint_state, "
            "advantage_weighted_sft_learner.py:228-232)."
        )
    sub = raw["state_action_critic_state"]
    if which not in sub or sub[which] is None:
        raise KeyError(f"{ckpt_path}: state_action_critic_state has no '{which}' (keys: {sorted(sub)}).")
    critic = build_critic(rngs_seed=0)
    nnx.update(critic, normalize_restored(sub[which]))
    return critic


# ------------------------------------------------------------------- geometry


def slice_masks(input_dim: int) -> dict[str, np.ndarray]:
    """Column masks over the 2400-d critic input."""
    m: dict[str, np.ndarray] = {}
    prefix = np.zeros(input_dim, bool); prefix[:2048] = True
    st_real = np.zeros(input_dim, bool); st_real[2048 : 2048 + REAL_STATE_DIM] = True
    st_pad = np.zeros(input_dim, bool); st_pad[2048 + REAL_STATE_DIM : 2048 + MODEL_ACT_DIM] = True
    act_real = np.zeros(input_dim, bool)
    act_pad = np.zeros(input_dim, bool)
    base = 2048 + MODEL_ACT_DIM
    for t in range(ACTION_HORIZON):
        s = base + t * MODEL_ACT_DIM
        act_real[s : s + REAL_ACT_DIM] = True
        act_pad[s + REAL_ACT_DIM : s + MODEL_ACT_DIM] = True
    m["prefix"] = prefix
    m["state_real"] = st_real
    m["state_pad"] = st_pad
    m["action_real"] = act_real
    m["action_pad"] = act_pad
    m["dead_null"] = st_pad | act_pad  # the zero-gradient baseline
    return m


def act_real_index() -> np.ndarray:
    """Indices of the 70 real dims inside the flat 320-d action vector."""
    idx = []
    for t in range(ACTION_HORIZON):
        idx.extend(range(t * MODEL_ACT_DIM, t * MODEL_ACT_DIM + REAL_ACT_DIM))
    return np.asarray(idx, dtype=np.int64)


# ------------------------------------------------------------------ the probe


def q_all_heads(critic, prefix, state, action) -> np.ndarray:
    """-> (num_qs, B)."""
    obs = {PREFIX_EMBEDDING_NAME: jnp.asarray(prefix), "state": jnp.asarray(state)}
    return np.asarray(critic(obs, jnp.asarray(action)))


def batched_q(critic, prefix, state, action, chunk: int = 4096) -> np.ndarray:
    outs = [
        q_all_heads(critic, prefix[i : i + chunk], state[i : i + chunk], action[i : i + chunk])
        for i in range(0, prefix.shape[0], chunk)
    ]
    return np.concatenate(outs, axis=1)


def summarize(name: str, q_sets: np.ndarray) -> dict:
    """q_sets: (num_qs, S, G) -- G candidate actions per state."""
    q_mean_head = q_sets.mean(axis=0)                      # (S, G) reduction="mean"
    within = q_mean_head.std(axis=1, ddof=1)               # (S,)
    across = q_mean_head.mean(axis=1).std(ddof=1)          # scalar
    per_head_within = q_sets.std(axis=2, ddof=1)           # (num_qs, S)
    # Between-head agreement on the *centred* candidate ordering.
    c = q_sets - q_sets.mean(axis=2, keepdims=True)        # (num_qs, S, G)
    nrm = np.linalg.norm(c, axis=2)                        # (num_qs, S)
    ok = nrm > 1e-12
    corrs = []
    for i in range(q_sets.shape[0]):
        for j in range(i + 1, q_sets.shape[0]):
            v = ok[i] & ok[j]
            if v.any():
                corrs.append(((c[i][v] * c[j][v]).sum(1) / (nrm[i][v] * nrm[j][v])).mean())
    # Sign-unanimity, as grpo_conservative computes it (update_actor.py:64-86).
    lo, hi = c.min(axis=0), c.max(axis=0)                  # (S, G)
    gated = np.maximum(lo, 0.0) + np.minimum(hi, 0.0)
    return {
        "set": name,
        "sigma_within_mean": float(within.mean()),
        "sigma_within_median": float(np.median(within)),
        "sigma_across": float(across),
        "rho_scale": float(within.mean() / across) if across > 0 else float("nan"),
        "per_head_sigma_within_mean": float(per_head_within.mean()),
        "between_head_centred_corr": float(np.mean(corrs)) if corrs else float("nan"),
        "cons_zero_frac": float((gated == 0.0).mean()),
        "q_mean": float(q_sets.mean()),
    }


def run_one(tag: str, critic, d: dict, rng: np.random.Generator, n_states: int,
            n_swap: int, mc_residual_rms: float | None) -> dict:
    res: dict = {"run": tag}
    n = d["prefix"].shape[0]
    obs_idx = d["obs_index"]
    n_txn = obs_idx.shape[0]
    sel = rng.choice(n_txn, size=min(n_states, n_txn), replace=False)
    sel.sort()

    prefix = d["prefix"][obs_idx[sel]]
    state = d["state"][obs_idx[sel]]
    act = d["actions"][sel].reshape(len(sel), -1)          # (S, 320)
    S = len(sel)
    real = act_real_index()

    # ---- 0. sanity: reproduce the logged across-state statistic ------------
    q_buf = batched_q(critic, prefix, state, act)          # (num_qs, S)
    q_buf_m = q_buf.mean(axis=0)
    if "mc_return" in d:
        mc = d["mc_return"][sel]
        res["sanity_q_vs_mc_corr"] = float(np.corrcoef(q_buf_m, mc)[0, 1])
        res["sanity_q_mc_rmse"] = float(np.sqrt(np.mean((q_buf_m - mc) ** 2)))
        res["mc_return_mean"] = float(mc.mean())
    res["q_buf_mean"] = float(q_buf_m.mean())
    res["q_buf_std_across_states"] = float(q_buf_m.std(ddof=1))
    sigma_across = res["q_buf_std_across_states"]

    sets: list[dict] = []

    # ---- 1. swap control: does Q condition on the state-action PAIRING? ----
    m = min(n_swap, S)
    pr_m, st_m, ac_m = prefix[:m], state[:m], act[:m]
    grid_p = np.repeat(pr_m, m, axis=0)
    grid_s = np.repeat(st_m, m, axis=0)
    grid_a = np.tile(ac_m, (m, 1))
    q_grid = batched_q(critic, grid_p, grid_s, grid_a).reshape(NUM_QS, m, m)
    qg = q_grid.mean(axis=0)                                # (state i, action j)
    diag = np.diag(qg)
    off = qg[~np.eye(m, dtype=bool)].reshape(m, m - 1)
    res["swap"] = {
        "n": int(m),
        "own_action_q_mean": float(diag.mean()),
        "other_action_q_mean": float(off.mean()),
        "own_minus_other": float((diag - off.mean(axis=1)).mean()),
        "own_minus_other_in_sigma_across": float(
            (diag - off.mean(axis=1)).mean() / sigma_across) if sigma_across > 0 else float("nan"),
        # Fraction of states where the true action outranks all foreign ones.
        "own_is_argmax_frac": float((qg.argmax(axis=1) == np.arange(m)).mean()),
        "row_sigma_over_actions": float(qg.std(axis=1, ddof=1).mean()),
        "col_sigma_over_states": float(qg.std(axis=0, ddof=1).mean()),
    }
    sets.append(summarize("swap_foreign_actions", q_grid.transpose(0, 1, 2)))

    # ---- 2. random actions in the empirical per-dim range ------------------
    lo = act[:, real].min(axis=0)
    hi = act[:, real].max(axis=0)
    G = 8
    a_rand = np.tile(act[:, None, :], (1, G, 1))
    a_rand[:, :, real] = rng.uniform(lo, hi, size=(S, G, real.size)).astype(np.float32)
    q_rand = batched_q(
        critic, np.repeat(prefix, G, 0), np.repeat(state, G, 0), a_rand.reshape(S * G, -1)
    ).reshape(NUM_QS, S, G)
    sets.append(summarize("uniform_random", q_rand))
    res["random_vs_buffer_q_gap"] = float(q_buf_m.mean() - q_rand.mean(axis=0).mean())

    # ---- 3. perturbation ladder around the buffer action -------------------
    ladder = {}
    for eps in (0.01, 0.03, 0.1, 0.3, 1.0):
        a_p = np.tile(act[:, None, :], (1, G, 1))
        a_p[:, :, real] += (eps * rng.standard_normal((S, G, real.size))).astype(np.float32)
        q_p = batched_q(
            critic, np.repeat(prefix, G, 0), np.repeat(state, G, 0), a_p.reshape(S * G, -1)
        ).reshape(NUM_QS, S, G)
        s = summarize(f"perturb_eps{eps}", q_p)
        if mc_residual_rms:
            s["rho_noise"] = s["sigma_within_mean"] / mc_residual_rms
        ladder[str(eps)] = s
        sets.append(s)
    res["perturbation_ladder"] = ladder

    # ---- 4. per-timestep ablation: which chunk step does Q listen to? ------
    eps = 0.1
    per_t = []
    for t in range(ACTION_HORIZON):
        cols = np.arange(t * MODEL_ACT_DIM, t * MODEL_ACT_DIM + REAL_ACT_DIM)
        a_t = act.copy()
        a_t[:, cols] += (eps * rng.standard_normal((S, cols.size))).astype(np.float32)
        q_t = batched_q(critic, prefix, state, a_t).mean(axis=0)
        per_t.append(float(np.abs(q_t - q_buf_m).mean()))
    res["per_timestep_abs_dq"] = per_t

    # ---- 5. zeroed action ---------------------------------------------------
    a_zero = act.copy(); a_zero[:, real] = 0.0
    res["zero_action_q_mean"] = float(batched_q(critic, prefix, state, a_zero).mean())

    res["action_sets"] = sets

    # ---- 7b. stratify the eps=0.1 within-state spread ----------------------
    # A value function that represents decision points should show a small
    # action gap when nothing is at stake and a large one near the grasp. A
    # spread that is FLAT across episode phase is itself the finding.
    eps = 0.1
    a_p = np.tile(act[:, None, :], (1, G, 1))
    a_p[:, :, real] += (eps * rng.standard_normal((S, G, real.size))).astype(np.float32)
    q_p = batched_q(
        critic, np.repeat(prefix, G, 0), np.repeat(state, G, 0), a_p.reshape(S * G, -1)
    ).reshape(NUM_QS, S, G)
    within_s = q_p.mean(axis=0).std(axis=1, ddof=1)          # (S,)

    strat: dict = {}
    if "discount" in d:
        ends = np.flatnonzero(d["discount"] == 0.0)
        strat["n_episode_ends_in_shard"] = int(ends.size)
        if ends.size == 0:
            strat["phase"] = "unavailable: no discount==0 rows in this shard"
        else:
            # steps-to-episode-end for every transition, then for our sample.
            nxt = np.searchsorted(ends, np.arange(n_txn), side="left")
            nxt = np.clip(nxt, 0, ends.size - 1)
            to_end = ends[nxt] - np.arange(n_txn)
            te = to_end[sel].astype(np.float64)
            q1, q2 = np.quantile(te, [1 / 3, 2 / 3])
            bins = {"late(near end)": te <= q1, "mid": (te > q1) & (te <= q2), "early": te > q2}
            strat["phase"] = {
                k: {"n": int(v.sum()),
                    "sigma_within_mean": float(within_s[v].mean()) if v.any() else None,
                    "q_mean": float(q_buf_m[v].mean()) if v.any() else None}
                for k, v in bins.items()
            }
    if "is_success" in d:
        isc = d["is_success"][sel] > 0.5
        strat["by_is_success"] = {
            k: {"n": int(v.sum()),
                "sigma_within_mean": float(within_s[v].mean()) if v.any() else None,
                "q_mean": float(q_buf_m[v].mean()) if v.any() else None}
            for k, v in {"success": isc, "failure": ~isc}.items()
        }
    if "task_index" in d:
        ti = d["task_index"][sel]
        strat["by_task"] = {
            int(t): {"n": int((ti == t).sum()),
                     "sigma_within_mean": float(within_s[ti == t].mean()),
                     "q_mean": float(q_buf_m[ti == t].mean())}
            for t in np.unique(ti)
        }
    res["stratified_sigma_within_eps0.1"] = strat

    # ---- 6. input-gradient sensitivity -------------------------------------
    def q_sum(pf, stt, ac):
        obs = {PREFIX_EMBEDDING_NAME: pf, "state": stt}
        return critic(obs, ac).mean()

    gp, gs, ga = jax.grad(q_sum, argnums=(0, 1, 2))(
        jnp.asarray(prefix[:256]), jnp.asarray(state[:256]), jnp.asarray(act[:256])
    )
    gp, gs, ga = np.asarray(gp), np.asarray(gs), np.asarray(ga)
    res["input_grad_rms_per_dim"] = {
        "prefix": float(np.sqrt((gp ** 2).mean())),
        "state_real": float(np.sqrt((gs[:, :REAL_STATE_DIM] ** 2).mean())),
        "state_pad": float(np.sqrt((gs[:, REAL_STATE_DIM:] ** 2).mean())),
        "action_real": float(np.sqrt((ga[:, real] ** 2).mean())),
        "action_pad": float(np.sqrt((np.delete(ga, real, axis=1) ** 2).mean())),
    }

    # ---- 7. proj.kernel column norms, against the dead-column null ---------
    masks = slice_masks(2048 + MODEL_ACT_DIM + ACTION_HORIZON * MODEL_ACT_DIM)
    st = nnx.state(critic)
    kernels = []
    for i in range(NUM_QS):
        kernels.append(np.asarray(jax.tree.leaves(st["nets"][i]["proj"]["kernel"])[0]))
    colnorm = np.stack([np.linalg.norm(k, axis=1) for k in kernels]).mean(axis=0)  # (2400,)
    res["proj_col_norm_mean"] = {k: float(colnorm[v].mean()) for k, v in masks.items()}
    null = colnorm[masks["dead_null"]].mean()
    res["proj_col_norm_vs_dead_null"] = {
        k: float(colnorm[v].mean() / null) for k, v in masks.items() if k != "dead_null"
    }
    return res


# ------------------------------------------------------------------------ cli


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=100000)
    ap.add_argument("--num-states", type=int, default=1024)
    ap.add_argument("--num-swap", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="ema_params", choices=["ema_params", "params"])
    ap.add_argument("--out", default="probe_critic_action_sensitivity.json")
    args = ap.parse_args()

    print(f"[probe] jax devices: {jax.devices()}", flush=True)
    root = pathlib.Path(args.ckpt_root)
    structure_log: list[str] = []
    results = []

    for run in args.runs:
        rd = root / run
        ckpt = rd / "rl_state" / str(args.step)
        shard = rd / "runtime_state" / "replay_shards" / f"step_{args.step:08d}.h5"
        print(f"\n[probe] ===== {run} =====\n  ckpt  {ckpt}\n  shard {shard}", flush=True)
        d = load_shard(shard)
        print(f"  shard: {d['obs_index'].shape[0]} transitions, "
              f"{d['prefix'].shape[0]} obs, prefix dim {d['prefix'].shape[-1]}, "
              f"state dim {d['state'].shape[-1]}, actions {d['actions'].shape}", flush=True)
        critic = load_critic(ckpt, args.params, structure_log)
        if structure_log:
            print("\n[probe] restored checkpoint structure:")
            print("\n".join(structure_log[:400]), flush=True)
            structure_log.append("__printed__")
        mc_rms = None
        rng = np.random.default_rng(args.seed)
        r = run_one(run, critic, d, rng, args.num_states, args.num_swap, mc_rms)
        # rho_noise needs the critic's own MC residual, now that we have it.
        if "sanity_q_mc_rmse" in r:
            for s in r["action_sets"]:
                s["rho_noise"] = s["sigma_within_mean"] / r["sanity_q_mc_rmse"]
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    # Untrained baseline: identical battery on a freshly initialised critic.
    print("\n[probe] ===== init (untrained) =====", flush=True)
    d0 = load_shard(
        root / args.runs[0] / "runtime_state" / "replay_shards" / f"step_{args.step:08d}.h5"
    )
    r0 = run_one("init_untrained", build_critic(rngs_seed=12345), d0,
                 np.random.default_rng(args.seed), args.num_states, args.num_swap, None)
    if "sanity_q_mc_rmse" in r0:
        for s in r0["action_sets"]:
            s["rho_noise"] = s["sigma_within_mean"] / r0["sanity_q_mc_rmse"]
    results.append(r0)
    print(json.dumps(r0, indent=2), flush=True)

    pathlib.Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[probe] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
