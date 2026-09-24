#!/usr/bin/env python
"""Does V distinguish the next-states the Best-of-N candidates lead to?

WHY. With td_weight = 0.95 the critic learns Q almost entirely from the TD
target r + gamma * V(s'). In LIBERO s' is essentially deterministic given
(s, a), so that target carries almost no sampling noise -- unlike the MC
target, which drags the full ~26-point continuation noise with it. So the
action signal Q actually learns from flows through V(s'), and Q can rank the
N candidates only as well as V separates the N next-states they produce.

Tier A/B measured sigma_within(Q) = 5.97 against a critic error of ~16. This
probe measures the corresponding spread of the TD target across the same
candidates. Two outcomes, and they point at different fixes:

  sigma_TD ~= sigma_Q   -> Q faithfully learned its target; the ceiling is V,
                           and no ranking loss on the Q head can lift it.
  sigma_TD >> sigma_Q   -> the signal is in the target and Q is not learning
                           it; a reweighted/ranking loss on Q is worth it.

METHOD. Drive G envs to one shared state by seeded reset plus a random walk-in
(same construction as the Tier C calibrate phase). Read the G candidate chunks
and their critic scores off the production Best-of-N path via the opt-in
`_bon_record` hook. Execute candidate k in env k for exactly ONE chunk. Then
read each env's own (state, prefix) off the hook again and evaluate the V head
on it. Compare spread(Q over candidates) with spread(TD target over candidates).

READ-ONLY: restores the checkpoint, takes no train step, never saves.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import pathlib
import sys

import flax.nnx as nnx
import jax
import numpy as np

mp.set_start_method("spawn", force=True)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.envs import make_env  # noqa: E402
from src.rl.advantage_weighted_sft.update_critic import summarize_critic_values  # noqa: E402
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env  # noqa: E402
from src.rl.networks.per_task_critic import TASK_INDEX_NAME  # noqa: E402
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner  # noqa: E402
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME  # noqa: E402
import src.training.config as _config  # noqa: E402
from src.training.utils import init_logging  # noqa: E402


def parse_probe_args() -> argparse.Namespace:
    """Strips --probe.* out of argv so the remainder is exp.py's exact CLI."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--probe.states", dest="states", type=int, default=64)
    ap.add_argument("--probe.max-walk-in", dest="max_walk_in", type=int, default=12,
                    help="probe states sit 1..max_walk_in chunks into an episode")
    ap.add_argument("--probe.seed", dest="seed", type=int, default=12345)
    ap.add_argument("--probe.out", dest="out", default="probe_vnext.json")
    known, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    return known


def reset_wave(env, task_id: str, seed: int):
    """Same task and same seed in every env => identical initial states."""
    env.seed([int(seed)] * env.env_num)
    obs, info = env.reset(options={"task_id": [task_id] * env.env_num})
    return obs, info


def bon_step(agent, obs, info, task_ids):
    """One Best-of-N selection, with the production hook capturing its inputs.

    Returns (chosen_actions, record). The wave is single-task by construction,
    so the hook must yield exactly one group; anything else means the caller
    broke that invariant and the index bookkeeping below would silently
    mis-assign candidates to envs.
    """
    agent._bon_record = []
    out = agent.sample_actions(
        obs, task_description=info["task_description"], task_id=list(task_ids)
    )
    rec = agent._bon_record
    agent._bon_record = None
    if len(rec) != 1:
        raise RuntimeError(
            f"expected a single-task Best-of-N group, got {len(rec)}; the wave "
            "must be driven with one task_id for every env."
        )
    act = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)
    return act, rec[0]


def unpermute(rec, key, G):
    """Hook rows are in task-group order; map them back to env slots."""
    rows = np.asarray(rec[key])
    out = np.zeros((G, *rows.shape[1:]), dtype=rows.dtype)
    for row, env_index in enumerate(rec["indices"]):
        out[int(env_index)] = rows[row]
    return out


def eval_v(v_model, cfg, state, prefix, task_index):
    obs = {"state": state, PREFIX_EMBEDDING_NAME: prefix}
    if task_index is not None:
        obs[TASK_INDEX_NAME] = task_index
    logits = v_model(obs)
    v = np.asarray(summarize_critic_values(
        logits, cfg, critic_reduction=cfg.rl.critic.reduction))
    per_head = np.asarray(logits)
    return v, per_head


def main() -> int:
    pa = parse_probe_args()
    cfg = _config.cli()
    init_logging()

    # Same hard guard as Tier C: initialize_checkpoint_dir is called with
    # overwrite = not resume, and overwrite=True rmtree's the run directory.
    if not cfg.resume:
        raise ValueError(
            "This probe must run with --resume. Without it the learner builds its "
            f"checkpoint manager with overwrite=True, which wipes {cfg.checkpoint_dir} "
            "on startup. Add --resume (the sbatch does)."
        )

    gamma = float(cfg.rl.discount)
    G = int(cfg.rl.n_samples)
    if G < 2:
        raise ValueError(
            f"needs rl.n_samples >= 2 (best-of-N candidates); got {G}. "
            "Run against a best-of-N config (BON_N=8)."
        )
    if cfg.collect.env_num != G:
        raise ValueError(
            f"start_data_collection sizes _episode_storage by collect.env_num="
            f"{cfg.collect.env_num}, but this probe drives {G} envs (rl.n_samples). "
            "Set --collect.env_num to match rl.n_samples."
        )

    agent = OGPOAgentLearner(cfg)
    steps_present = [int(x) for x in agent._checkpoint_manager.all_steps()]
    logging.info(f"[vnext] restored at step {agent.training_steps}; "
                 f"checkpoint steps present: {steps_present}")

    v_params = (
        agent._value_state.ema_params
        if agent._value_state.ema_params is not None
        else agent._value_state.params
    )
    v_model = nnx.merge(agent._value_state.model_def, v_params)
    v_model.eval()

    env = filtered_sft_wrap_env(
        env_fn=make_env(cfg, cfg.collect.tasks, num_devices=1), config=cfg, env_num=G
    )
    # sample_actions composes the policy from _train_state.ema_params, which the
    # learner attaches ONLY inside a collection window (filtered_sft_learner.py
    # :329 detaches, :845-851 reattaches, :874-876 detaches again). Nothing below
    # calls add_data/save_episode, so the replay buffer stays untouched.
    agent.start_data_collection(step=None)

    tasks = list(cfg.collect.tasks)
    out: dict = {
        "step": int(agent.training_steps), "gamma": gamma, "group": G,
        "tasks": tasks, "checkpoint_steps_present": steps_present,
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "td_weight_note": "TD target measured here is what the critic is trained on",
    }
    outp = pathlib.Path(pa.out)

    def flush():
        outp.write_text(json.dumps(out, indent=2, default=float))

    rng = np.random.default_rng(pa.seed)
    rows = []

    for i in range(pa.states):
        task = tasks[i % len(tasks)]
        seed = int(rng.integers(0, 2**31 - 1))
        obs, info = reset_wave(env, task, seed)
        tids = [task] * G

        # Walk in so probe states are not all at t=0. Every env takes the same
        # Best-of-N action, so the wave stays at one shared state.
        aborted = False
        for _ in range(int(rng.integers(1, pa.max_walk_in))):
            act, _rec = bon_step(agent, obs, info, tids)
            common = np.repeat(act[:1], G, axis=0)
            obs, rew, term, trunc, _ = env.step(common)
            if np.logical_or(np.asarray(term), np.asarray(trunc)).any():
                aborted = True
                break
        if aborted:
            logging.info(f"[vnext] state {i+1}/{pa.states} {task}: episode ended "
                         "during walk-in, skipped")
            continue

        # The candidates Best-of-N is actually choosing between, at this state.
        _act, rec = bon_step(agent, obs, info, tids)
        cands = np.asarray(rec["candidates"])          # (G, n_samples, H, act)
        scores = np.asarray(rec["scores"])             # (G, n_samples)
        env0 = int(rec["indices"][0])
        q = scores[0].astype(np.float64)               # all envs share the state
        if cands.shape[1] != G:
            raise RuntimeError(
                f"n_samples={cands.shape[1]} != G={G}; this probe assigns candidate "
                "k to env k and needs them equal.")
        chosen = np.stack([cands[0, k] for k in range(G)], axis=0)

        # Execute candidate k in env k for exactly one chunk.
        nobs, rew, term, trunc, _ = env.step(chosen)
        rew = np.asarray(rew, dtype=np.float64)
        term = np.asarray(term, dtype=bool)
        trunc = np.asarray(trunc, dtype=bool)
        chunk_ret = np.zeros(G)
        resid = np.ones(G)
        ended = np.zeros(G, bool)
        succ = np.zeros(G, bool)
        for j in range(rew.shape[1]):
            for e in range(G):
                if ended[e]:
                    continue
                chunk_ret[e] += resid[e] * rew[e, j]
                resid[e] *= gamma
                if term[e, j] or trunc[e, j]:
                    ended[e] = True
                    succ[e] = bool(term[e, j])

        # Read each env's own next-state critic inputs off the hook. Envs that
        # ended are excluded: the vec env's post-termination observation is not
        # the terminal state, so V would be evaluated on the wrong thing.
        _act2, rec2 = bon_step(agent, nobs, info, tids)
        st2 = unpermute(rec2, "state", G)
        pf2 = unpermute(rec2, "prefix", G)
        ti2 = unpermute(rec2, "task_index", G) if "task_index" in rec2 else None
        v_next, v_heads = eval_v(v_model, cfg, st2, pf2, ti2)

        alive = ~ended
        # Bootstrap only where the episode continues, exactly as the TD target does.
        td = chunk_ret + np.where(ended, 0.0, resid * v_next)

        n_alive = int(alive.sum())
        row = {
            "state_index": i, "task": task, "seed": seed, "n_alive": n_alive,
            "q": q.tolist(), "v_next": v_next.tolist(), "td_target": td.tolist(),
            "chunk_return": chunk_ret.tolist(), "ended": ended.tolist(),
            "success": succ.tolist(),
            "sigma_q": float(q.std(ddof=1)),
            "sigma_v_next": float(v_next.std(ddof=1)),
            "sigma_td": float(td.std(ddof=1)),
            "per_head_sigma_v_next": float(np.asarray(v_heads).std(axis=-1, ddof=1).mean())
            if np.asarray(v_heads).ndim > 1 else None,
        }
        if n_alive >= 4:
            qa, tda, va = q[alive], td[alive], v_next[alive]
            row["sigma_q_alive"] = float(qa.std(ddof=1))
            row["sigma_td_alive"] = float(tda.std(ddof=1))
            row["sigma_v_next_alive"] = float(va.std(ddof=1))
            # Does Q order the candidates the way its own TD target does?
            row["corr_q_td"] = float(np.corrcoef(qa, tda)[0, 1]) if tda.std() > 0 else None
            row["argmax_agree"] = bool(int(np.argmax(qa)) == int(np.argmax(tda)))
        rows.append(row)
        logging.info(
            f"[vnext] {i+1}/{pa.states} {task} alive={n_alive}/{G} "
            f"sigma_Q={row['sigma_q']:.4g} sigma_V'={row['sigma_v_next']:.4g} "
            f"sigma_TD={row['sigma_td']:.4g}")
        out["states"] = rows
        flush()

    # ------------------------- verdict --------------------------------------
    usable = [r for r in rows if r.get("sigma_td_alive") is not None]

    def m(key):
        vals = [r[key] for r in usable if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    sq, std_, sv = m("sigma_q_alive"), m("sigma_td_alive"), m("sigma_v_next_alive")
    corrs = [r["corr_q_td"] for r in usable if r.get("corr_q_td") is not None]
    agrees = [r["argmax_agree"] for r in usable if "argmax_agree" in r]
    out["verdict"] = {
        "n_states_total": len(rows),
        "n_states_usable": len(usable),
        "sigma_q_mean": sq,
        "sigma_v_next_mean": sv,
        "sigma_td_mean": std_,
        # The decisive ratio. ~1 => Q learned its target and V is the ceiling.
        # >>1 => the target carries signal Q is failing to pick up.
        "sigma_td_over_sigma_q": (std_ / sq) if (sq and std_) else None,
        "corr_q_td_mean": float(np.mean(corrs)) if corrs else None,
        "corr_q_td_frac_positive": float(np.mean([c > 0 for c in corrs])) if corrs else None,
        "argmax_agree_frac": float(np.mean(agrees)) if agrees else None,
    }
    logging.info(f"[vnext] verdict: {json.dumps(out['verdict'], default=float)}")
    flush()

    agent.end_data_collection()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
