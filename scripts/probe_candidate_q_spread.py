#!/usr/bin/env python
"""Read-only probe: variance of the critic's Q over M sampled candidate chunks, along rollouts.

For a restored checkpoint, rolls out N episodes of each task in
cfg.collect.eval_tasks. At EVERY policy query the production best-of-N path
(AdvantageWeightedSFTLearner.sample_actions with rl.n_samples = M,
src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py:412-657)
samples M candidate action chunks from M initial-noise draws through the
deterministic ODE and scores each with the critic, reduced over heads by
rl.critic.reduction. The opt-in `_bon_record` hook (:621-646) hands the M
candidates and their M reduced Q-scores back here. The probe records the
VARIANCE of those M scores -- how diverse the policy's candidates look to the
critic at that state -- and executes either

  QSPREAD_BON=1  the argmax-Q candidate (production best-of-N collection), or
  QSPREAD_BON=0  candidate 0 -- one iid policy draw, i.e. what single-sample
                 collection would have executed; the plain-policy control.

Outputs, under QSPREAD_OUT_DIR:
  <task>/ep<NN>_seed<seed>_<succ|fail>.png   chunk index vs. Q-variance, per rollout
  <task>/mean_trace.png                       mean over the rollouts alive at each
                                              chunk index, alive count beneath it
  q_spread_results.json                       every raw (chunks, M) score matrix,
                                              executed candidate index per chunk,
                                              success / env steps / seed per episode

"Chunk index" is the k-th policy query within the episode, from 0. No critic
scoring is reimplemented here: a second copy of the scoring block could drift
from the normalize->pad order the critic was trained on (CLAUDE.md sharp
edges), so the production block is reused through the hook.

Restores via the same tyro CLI as scripts/exp.py (ENTRY-swap contract:
ENTRY=scripts/probe_candidate_q_spread.py through
scripts/ogpo_multitask_4task_ref.sh, see scripts/probe_candidate_q_spread.sbatch),
so the checkpoint (ARM/SEED/CONFIG_NAME), the task list (TASKS), M (BON_N ->
--rl.n_samples) and the episode length (EP_MULT) come from the recipe and
there is no second copy of the flag block to drift.

READ-ONLY wrt the checkpoint: requires --resume, never calls agent.update(),
add_data, save_episode or any save path.

`reset_wave` / `policy_chunk` / `rollout` are adapted from
scripts/probe_rollout_gifs.py (frame capture dropped, per-query score capture
and candidate choice added) -- a third divergent copy, by this repo's
duplication-by-copy convention (CLAUDE.md Decisions log OQ-2).

Knobs (env vars):
  QSPREAD_OUT_DIR   output directory (required)
  QSPREAD_EPISODES  rollouts N per task (default 8); exactly N are recorded,
                    ceil(N / collect.env_num) waves are run
  QSPREAD_SEED      base seed (default 0)
  QSPREAD_BON       1 = execute argmax-Q, 0 = execute candidate 0 (default 1)
  QSPREAD_PASSES    sample_actions calls per query (default 1). M = QSPREAD_PASSES x
                    rl.n_samples: the observation is tiled rl.n_samples times
                    BEFORE the VLM prefix pass (AWR:471), so a large M in one
                    call is a large VLM batch; splitting it over passes keeps
                    the peak memory at the collection-validated batch and is
                    the same selection (iid candidates, deterministic scores).
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import sys

import jax
import numpy as np

mp.set_start_method("spawn", force=True)  # allows using subprocenvs (scripts/exp.py:46-48)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# The env / learner imports live inside main(): they pull in LIBERO, MuJoCo
# and torch, which tests/ogpo/test_candidate_q_spread.py must not need to
# exercise the pure helpers below on a CPU login node.

SERIES_BLUE = "#2a78d6"     # single-series hue (docs: dataviz reference palette, slot 1)
INK_SECONDARY = "#52514e"   # recessive text / secondary panel


# --------------------------------------------------------------------------- #
# pure helpers (tested)
# --------------------------------------------------------------------------- #
def candidate_q_variance(scores) -> np.ndarray:
    """(n_env, M) reduced Q-scores -> (n_env,) variance over the M candidates.

    Population variance (ddof=0): the M candidates ARE the whole set the
    selector chose between, and the number wanted is that set's spread, not an
    estimate of a wider population's.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] < 2:
        raise ValueError(
            f"candidate_q_variance expects (n_env, M >= 2) scores, got shape {s.shape}"
        )
    return s.var(axis=1)


def gather_records(records: list[dict], env_num: int):
    """Per-prompt-group `_bon_record` dicts -> per-env arrays in env order.

    sample_actions groups envs by prompt and appends one dict per group whose
    `indices` are that group's env slots (AWR:449-451, :621-646). Returns
    (candidates (env_num, M, H, act), scores (env_num, M), best_idx (env_num,)).
    Raises unless the groups' indices partition range(env_num) exactly -- a
    silent misalignment would make the BoN=0 arm execute another env's
    candidate with no error.
    """
    if not records:
        raise RuntimeError(
            "sample_actions recorded no best-of-N groups: it took the single-sample "
            "path. Needs rl.n_samples >= 2 (BON_N) and training_steps >= "
            "rl.critic.inference_start_step."
        )
    seen = np.concatenate([np.asarray(r["indices"], dtype=np.int64) for r in records])
    if sorted(seen.tolist()) != list(range(env_num)):
        raise RuntimeError(
            f"best-of-N group indices {sorted(seen.tolist())} do not partition "
            f"range({env_num}); candidate/env alignment is broken."
        )
    m = int(np.asarray(records[0]["scores"]).shape[1])
    cand_shape = tuple(np.asarray(records[0]["candidates"]).shape[2:])
    candidates = np.zeros((env_num, m, *cand_shape), dtype=np.float32)
    scores = np.zeros((env_num, m), dtype=np.float32)
    best_idx = np.zeros(env_num, dtype=np.int64)
    for r in records:
        idx = np.asarray(r["indices"], dtype=np.int64)
        c = np.asarray(r["candidates"], dtype=np.float32)
        s = np.asarray(r["scores"], dtype=np.float32)
        b = np.asarray(r["best_idx"], dtype=np.int64)
        if c.shape != (len(idx), m, *cand_shape) or s.shape != (len(idx), m) or b.shape != (len(idx),):
            raise RuntimeError(
                f"best-of-N record shapes disagree within a wave: candidates {c.shape}, "
                f"scores {s.shape}, best_idx {b.shape} for {len(idx)} envs, M={m}, "
                f"chunk shape {cand_shape}."
            )
        candidates[idx] = c
        scores[idx] = s
        best_idx[idx] = b
    return candidates, scores, best_idx


def mean_over_alive(traces: list) -> tuple[np.ndarray, np.ndarray]:
    """Ragged per-episode traces -> (mean at each chunk index over the episodes
    that reached it, alive count). Length = the longest trace, so every index
    has at least one contributor.
    """
    if not traces:
        raise ValueError("mean_over_alive needs at least one trace")
    length = max(len(t) for t in traces)
    if length == 0:
        raise ValueError("mean_over_alive: every trace is empty")
    total = np.zeros(length, dtype=np.float64)
    alive = np.zeros(length, dtype=np.int64)
    for t in traces:
        t = np.asarray(t, dtype=np.float64)
        total[: len(t)] += t
        alive[: len(t)] += 1
    return total / alive, alive


# --------------------------------------------------------------------------- #
# env driving
# --------------------------------------------------------------------------- #
def reset_wave(env, task_id: str, seeds: list[int]):
    """Resets every env to `task_id`, each from its OWN seed. LiberoWrapper.reset
    draws its init state from a seeded rng (src/envs/libero.py:28,48), so
    distinct per-env seeds give a wave G different initial conditions."""
    env.seed([int(s) for s in seeds])
    obs, info = env.reset(options={"task_id": [task_id] * env.env_num})
    return obs, info


def policy_chunk(agent, obs, info, task_ids):
    out = agent.sample_actions(
        obs, task_description=info["task_description"], task_id=list(task_ids)
    )
    return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)


def rollout(env, agent, obs, info, task_ids, max_chunks: int, use_bon: bool, passes: int = 1):
    """Drives one wave to termination.

    Every policy query runs through the production best-of-N path with the
    `_bon_record` hook armed, `passes` times; each call yields rl.n_samples
    candidates and scores per env via gather_records, concatenated to
    M = passes x rl.n_samples. Executes the argmax-Q chunk over the union
    (use_bon) or candidate 0 of the first pass. Splitting is exact: candidates
    are iid initial-noise draws and the critic's score is deterministic per
    (state, candidate), so the union argmax is the single-pass selection.
    Returns (success, env steps, per-env (chunks, M) score matrices, per-env
    executed candidate indices). Envs that finish early stop being stepped and
    stop being recorded.
    """
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")
    G = env.env_num
    steps = np.zeros(G, int)
    done = np.zeros(G, bool)
    succ = np.zeros(G, bool)
    scores_per_env: list[list[np.ndarray]] = [[] for _ in range(G)]
    exec_per_env: list[list[int]] = [[] for _ in range(G)]
    n = 0
    while not done.all() and n < max_chunks:
        cand_parts, score_parts, best_parts = [], [], []
        for p in range(passes):
            if agent._bon_record is not None:
                raise RuntimeError("_bon_record is already armed; the hook must be per-query")
            agent._bon_record = []
            try:
                best = policy_chunk(agent, obs, info, task_ids)
                rec = agent._bon_record
            finally:
                # Disarm even when the policy/critic forward raises: a hook left
                # armed makes the NEXT call in this process die on the guard above
                # instead of on the real cause, and keeps appending to a stale list.
                # Nothing is swallowed -- the exception propagates.
                agent._bon_record = None
            cands_p, scores_p, best_idx_p = gather_records(rec, G)
            # A non-finite chunk is a policy failure, not an alignment failure; say
            # which, because array_equal is False on any NaN and would otherwise
            # blame the record below.
            if not np.isfinite(best).all():
                raise RuntimeError(
                    f"sample_actions returned a non-finite chunk at query {n} pass {p} "
                    f"(rows {np.where(~np.isfinite(best).reshape(G, -1).all(axis=1))[0].tolist()})."
                )
            # The chunk sample_actions returned IS candidates[best_idx] by construction
            # (AWR:619 indexes, :650 stores the same float32 buffer). Checking it pins
            # the indices->env alignment that the BoN=0 arm relies on: a misaligned
            # gather would execute another env's candidate with no error.
            if not np.array_equal(cands_p[np.arange(G), best_idx_p], best):
                raise RuntimeError(
                    "gathered argmax candidates differ from the chunk sample_actions "
                    "returned; the per-group record is misaligned with the env order."
                )
            cand_parts.append(cands_p)
            score_parts.append(scores_p)
            best_parts.append(best_idx_p)
        cands = np.concatenate(cand_parts, axis=1)    # (G, M, H, act)
        scores = np.concatenate(score_parts, axis=1)  # (G, M)
        if use_bon:
            exec_idx = scores.argmax(axis=1)
            # Single pass: the union argmax must reproduce production's own
            # pick (same float32 scores, same np.argmax, first-index ties).
            if passes == 1 and not np.array_equal(exec_idx, best_parts[0]):
                raise RuntimeError(
                    "argmax over the recorded scores differs from sample_actions' "
                    "best_idx; the recorded scores are not the ones it selected on."
                )
        else:
            exec_idx = np.zeros(G, dtype=np.int64)
        act = cands[np.arange(G), exec_idx]
        live = np.where(~done)[0]
        nobs, _rew, term, trunc, _info = env.step(act[live], id=live.tolist())
        term = np.asarray(term, dtype=bool)
        trunc = np.asarray(trunc, dtype=bool)
        tt = np.logical_or(term, trunc)
        for row, e in enumerate(live):
            scores_per_env[e].append(scores[e])
            exec_per_env[e].append(int(exec_idx[e]))
            for j in range(term.shape[1]):
                if done[e]:
                    break
                steps[e] += 1
                if tt[row, j]:
                    done[e] = True
                    succ[e] = bool(term[row, j])

        def _scatter(prev, new):
            prev = np.array(prev, copy=True)
            prev[live] = new
            return prev

        obs = jax.tree.map(_scatter, obs, nobs)
        n += 1
    score_mats = [np.stack(s, axis=0) for s in scores_per_env]
    return succ, steps, score_mats, exec_per_env


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")  # headless batch job
    import matplotlib.pyplot as plt

    return plt


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_episode(path: pathlib.Path, variance: np.ndarray, title: str, m: int):
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=130)
    ax.plot(np.arange(len(variance)), variance, color=SERIES_BLUE, linewidth=2)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("chunk index (policy query within the episode)")
    ax.set_ylabel(f"variance of the {m} candidates' Q")
    ax.set_xlim(left=0)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mean(path: pathlib.Path, mean: np.ndarray, alive: np.ndarray, title: str, m: int):
    """Two panels, one axis each: the mean trace, and how many rollouts were
    still alive at each chunk index (the mean's support). Never a second
    y-axis on the same panel."""
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 5.2), dpi=130, gridspec_kw={"height_ratios": [3, 1]}
    )
    x = np.arange(len(mean))
    ax0.plot(x, mean, color=SERIES_BLUE, linewidth=2)
    ax0.set_title(title, fontsize=10, loc="left")
    ax0.set_ylabel(f"mean variance of the {m} candidates' Q")
    _style(ax0)
    ax1.step(x, alive, where="post", color=INK_SECONDARY, linewidth=2)
    ax1.set_ylabel("rollouts alive")
    ax1.set_xlabel("chunk index (policy query within the episode)")
    ax1.set_ylim(0, int(alive.max()) + 1)
    ax1.set_xlim(left=0)
    _style(ax1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    import src.training.config as _config
    from src.training.utils import init_logging

    cfg = _config.cli()
    init_logging()

    # HARD GUARD, same reasoning as the gif probe / Tier C: initialize_checkpoint_dir
    # is called with overwrite=not resume, and overwrite=True does
    # checkpoint_dir.rmtree(). A missing --resume would DELETE the checkpoint
    # before this probe read a single number from it.
    if not cfg.resume:
        raise ValueError(
            "This probe must run with --resume: the learner otherwise constructs "
            f"with overwrite=True and WIPES {cfg.checkpoint_dir}. The recipes "
            "default to --resume; do not pass FRESH=1."
        )

    out_dir_s = os.environ.get("QSPREAD_OUT_DIR")
    if not out_dir_s:
        raise ValueError("set QSPREAD_OUT_DIR=<directory> for the plots and JSON")
    out_dir = pathlib.Path(out_dir_s)
    episodes = int(os.environ.get("QSPREAD_EPISODES", "8"))
    if episodes < 1:
        raise ValueError(f"QSPREAD_EPISODES must be >= 1, got {episodes}")
    base_seed = int(os.environ.get("QSPREAD_SEED", "0"))
    bon_s = os.environ.get("QSPREAD_BON", "1")
    if bon_s not in ("0", "1"):
        raise ValueError(f"QSPREAD_BON must be 0 or 1, got {bon_s!r}")
    use_bon = bon_s == "1"
    passes = int(os.environ.get("QSPREAD_PASSES", "1"))
    if passes < 1:
        raise ValueError(f"QSPREAD_PASSES must be >= 1, got {passes}")

    m_pass = int(cfg.rl.n_samples)
    if m_pass < 2:
        raise ValueError(
            f"rl.n_samples={m_pass}: sample_actions would take the single-sample "
            "path with no record. Set BON_N=<n> (--rl.n_samples) >= 2; the total "
            "candidate count is QSPREAD_PASSES x rl.n_samples."
        )
    m = m_pass * passes

    from src.envs import make_env
    from src.envs.libero import get_max_steps_libero
    from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env
    from src.rl.ogpo.ogpo_learner import OGPOAgentLearner

    agent = OGPOAgentLearner(cfg)
    # openpi silently downgrades --resume to a fresh start when the checkpoint
    # dir is absent or holds no checkpoints (openpi/training/checkpoints.py:56-61);
    # this probe would then score candidates with UNTRAINED critics and base
    # policy weights under the trained run's name. Refuse instead.
    if not agent._resuming:
        raise ValueError(
            f"--resume was passed but no checkpoint was found under {cfg.checkpoint_dir} "
            "-- openpi downgraded to a fresh start. Check ARM/SEED/CONFIG_NAME/"
            "checkpoint_base_dir against the run that produced it."
        )
    if agent.training_steps < cfg.rl.critic.inference_start_step:
        raise ValueError(
            f"restored step {agent.training_steps} < rl.critic.inference_start_step="
            f"{cfg.rl.critic.inference_start_step}: sample_actions would take the "
            "single-sample path (AWR:420) and score nothing. Lower "
            "--rl.critic.inference_start_step for this probe."
        )
    steps_present = [int(x) for x in agent._checkpoint_manager.all_steps()]
    logging.info(
        f"[qspread] agent restored at step {agent.training_steps} from {cfg.checkpoint_dir}; "
        f"checkpoint steps present: {steps_present}; M={m} ({passes} pass(es) x "
        f"{m_pass}) bon={use_bon} reduction={cfg.rl.critic.reduction}"
    )

    # env_num = collect.env_num by convention with the sibling probes. Not a
    # hard constraint here: start_data_collection sizes _episode_storage from
    # collect.env_num (filtered_sft_learner.py:987), but nothing below calls
    # save_episode/add_data, so that list is never indexed.
    G = cfg.collect.env_num
    env = filtered_sft_wrap_env(
        env_fn=make_env(cfg, cfg.collect.eval_tasks, num_devices=1), config=cfg, env_num=G
    )
    # sample_actions composes the policy from _train_state.ema_params, which is
    # attached only inside a collection window (filtered_sft_learner.py:607-612
    # composes; :989-991 attaches). A probe driving the agent directly has to
    # open it itself.
    # Nothing below calls add_data/save_episode, so the replay buffer is untouched.
    agent.start_data_collection(step=None)

    tasks = list(dict.fromkeys(cfg.collect.eval_tasks))  # de-dup, keep order
    waves = math.ceil(episodes / G)
    out = {
        "step": int(agent.training_steps),
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "checkpoint_steps_present": steps_present,
        "n_samples": m,
        "n_samples_per_pass": m_pass,
        "passes": passes,
        "critic_reduction": str(cfg.rl.critic.reduction),
        "bon": use_bon,
        "episode_steps_multiplier": cfg.collect.episode_steps_multiplier,
        "replan_steps": int(cfg.collect.replan_steps),
        "episodes_per_task": episodes,
        "base_seed": base_seed,
        "tasks": {},
    }
    summary_path = out_dir / "q_spread_results.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    def flush():
        # tmp + os.replace: a preemption inside a truncating write would lose
        # every previously flushed wave, not just the current one.
        tmp = summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2, default=float))
        os.replace(tmp, summary_path)

    arm = "BoN=argmax-Q" if use_bon else "BoN=off (candidate 0)"
    for task in tasks:
        # Suite-specific TimeLimit -- make_env_libero derives it from the task
        # itself (src/envs/libero.py:88-102), so this must too.
        suite_name = "_".join(task.split("_")[:-1])
        max_steps = get_max_steps_libero(suite_name) * cfg.collect.episode_steps_multiplier
        max_chunks = max_steps // cfg.collect.replan_steps + 2
        cell = out["tasks"].setdefault(task, {"max_steps": int(max_steps), "episodes": []})
        traces: list[np.ndarray] = []
        for w in range(waves):
            seeds = [base_seed * 7919 + w * G + i for i in range(G)]
            obs, info = reset_wave(env, task, seeds)
            tids = [task] * G
            succ, steps, score_mats, exec_idx = rollout(
                env, agent, obs, info, tids, max_chunks, use_bon, passes
            )
            for e in range(G):
                ep_idx = w * G + e
                if ep_idx >= episodes:
                    break  # surplus envs of the last wave: stepped, not recorded
                var = candidate_q_variance(score_mats[e])
                tag = "succ" if succ[e] else "fail"
                png = out_dir / task / f"ep{ep_idx:02d}_seed{seeds[e]}_{tag}.png"
                plot_episode(
                    png,
                    var,
                    f"{task}  ep{ep_idx:02d}  seed {seeds[e]}  "
                    f"{'SUCCESS' if succ[e] else 'FAILURE'} after {int(steps[e])} env steps  "
                    f"[{arm}, M={m}]",
                    m,
                )
                cell["episodes"].append(
                    {
                        "episode": ep_idx,
                        "seed": seeds[e],
                        "success": bool(succ[e]),
                        "steps": int(steps[e]),
                        "chunks": int(score_mats[e].shape[0]),
                        "executed_idx": exec_idx[e],
                        "scores": score_mats[e].tolist(),
                        "q_variance": var.tolist(),
                        "plot": str(png),
                    }
                )
                traces.append(var)
            n_rec = min(G, episodes - w * G)  # recorded envs of this wave
            sr = float(np.mean([ep["success"] for ep in cell["episodes"]]))
            wave_var = np.concatenate([candidate_q_variance(s) for s in score_mats[:n_rec]])
            logging.info(
                f"[qspread] {task} wave={w} recorded={n_rec}/{G} "
                f"wave_SR={float(np.mean(succ[:n_rec])):.3f} cum_SR={sr:.3f} "
                f"steps={steps[:n_rec].tolist()} q_var[min/median/max]="
                f"{wave_var.min():.4g}/{np.median(wave_var):.4g}/{wave_var.max():.4g}"
            )
            flush()
        mean, alive = mean_over_alive(traces)
        mean_png = out_dir / task / "mean_trace.png"
        plot_mean(
            mean_png,
            mean,
            alive,
            f"{task}  mean over {len(traces)} rollouts alive at each chunk  "
            f"SR={float(np.mean([ep['success'] for ep in cell['episodes']])):.2f}  [{arm}, M={m}]",
            m,
        )
        cell["mean_trace"] = {
            "mean_q_variance": mean.tolist(),
            "alive": alive.tolist(),
            "plot": str(mean_png),
        }
        flush()

    logging.info("[qspread] === per-task success rate ===")
    for task, cell in out["tasks"].items():
        sr = float(np.mean([e["success"] for e in cell["episodes"]]))
        logging.info(f"[qspread] {task:16s} SR={sr:.3f} n={len(cell['episodes'])}")

    agent.end_data_collection()
    env.close()
    logging.info(f"[qspread] results -> {summary_path}; plots under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
