#!/usr/bin/env python
"""Read-only rollout-GIF probe: qualitative behavior check for a trained checkpoint.

Restores the deployed policy for a given ARM/SEED via the same tyro CLI
scripts/exp.py uses (ENTRY-swap contract -- ENTRY=scripts/probe_rollout_gifs.py
through scripts/ogpo_multitask_4task_ref.sh, see scripts/probe_rollout_gifs.sbatch
-- so the recipe's flag block is not cloned a second time to drift; the knobs
that matter for restore -- CONFIG_NAME, ARM, SEED, TASKS, EP_MULT -- render
identically to the training run's, though a few inert-at-eval ones don't, see
docs/changes/2026-09-05-rollout-gif-probe/VERIFICATION.md finding 3) and rolls
it out on cfg.collect.eval_tasks, saving one GIF per episode plus a JSON
summary of success/step counts.

READ-ONLY wrt the checkpoint: requires --resume, never calls agent.update() or
any save path, so no train step is ever taken.

`rollout`/`policy_chunk`/`reset_wave` are adapted from
scripts/probe_counterfactual_rollouts.py (return/candidate tracking stripped,
frame capture added); `write_gif` is copied from
experiments/language_grounding/stage4_trained_instructions.py. Reused by copy,
not import: neither script exposes these as a shared utility, and each has
already diverged from the other (this repo's duplication-by-copy convention,
CLAUDE.md Decisions log OQ-2).

Knobs (env vars):
  ROLLOUT_OUT_DIR     output directory (required)
  ROLLOUT_EPISODES    episodes per task, rounded down to waves*env_num (default 8)
  ROLLOUT_SEED        base seed (default 0)
  ROLLOUT_GIF_STRIDE  keep every k-th frame in the GIFs (default 2)
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import pathlib
import sys

import jax
import numpy as np

mp.set_start_method("spawn", force=True)  # allows using subprocenvs (scripts/exp.py:46-48)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.envs import make_env  # noqa: E402
from src.envs.libero import get_max_steps_libero  # noqa: E402
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env  # noqa: E402
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner  # noqa: E402
import src.training.config as _config  # noqa: E402
from src.training.utils import init_logging  # noqa: E402


def reset_wave(env, task_id: str, seeds: list[int]):
    """Resets every env to `task_id`, each from its OWN seed. LiberoWrapper.reset
    draws its init state from a seeded rng (src/envs/libero.py), so distinct
    per-env seeds are what give a wave G DIFFERENT initial conditions instead
    of G copies of one -- a single shared seed (as Tier C deliberately uses,
    for a controlled comparison) would make this probe's episode count purely
    cosmetic."""
    env.seed([int(s) for s in seeds])
    obs, info = env.reset(options={"task_id": [task_id] * env.env_num})
    return obs, info


def policy_chunk(agent, obs, info, task_ids):
    out = agent.sample_actions(
        obs, task_description=info["task_description"], task_id=list(task_ids)
    )
    return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)


def rollout(env, agent, obs, info, task_ids, max_chunks: int):
    """Drives one wave to termination, capturing the agentview frame every
    step. Returns (success, steps, per-env frame lists)."""
    G = env.env_num
    steps = np.zeros(G, int)
    done = np.zeros(G, bool)
    succ = np.zeros(G, bool)
    frames = [[np.asarray(obs["observation/image"][e][-1])] for e in range(G)]
    n = 0
    while not done.all() and n < max_chunks:
        act = policy_chunk(agent, obs, info, task_ids)
        live = np.where(~done)[0]
        nobs, _rew, term, trunc, _info = env.step(act[live], id=live.tolist())
        term = np.asarray(term, dtype=bool)
        trunc = np.asarray(trunc, dtype=bool)
        tt = np.logical_or(term, trunc)
        imgs = np.asarray(nobs["observation/image"])  # (live, Q, H, W, 3)
        for row, e in enumerate(live):
            for j in range(term.shape[1]):
                if done[e]:
                    break
                steps[e] += 1
                frames[e].append(imgs[row, j])
                if tt[row, j]:
                    done[e] = True
                    succ[e] = bool(term[row, j])

        def _scatter(prev, new):
            prev = np.array(prev, copy=True)
            prev[live] = new
            return prev

        obs = jax.tree.map(_scatter, obs, nobs)
        n += 1
    return succ, steps, frames


def write_gif(path: pathlib.Path, frames, stride: int):
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(path), [np.asarray(f) for f in frames[::stride]], duration=int(100 * stride), loop=0
    )  # ms


def main() -> int:
    cfg = _config.cli()
    init_logging()

    # HARD GUARD, same reasoning as Tier C / stage 4: initialize_checkpoint_dir
    # is called with overwrite=not resume, and overwrite=True does
    # checkpoint_dir.rmtree(). A missing --resume would DELETE the checkpoint
    # before this probe read a single frame from it.
    if not cfg.resume:
        raise ValueError(
            "This probe must run with --resume: the learner otherwise constructs "
            f"with overwrite=True and WIPES {cfg.checkpoint_dir}. The recipes "
            "default to --resume; do not pass FRESH=1."
        )

    out_dir = pathlib.Path(os.environ["ROLLOUT_OUT_DIR"])
    episodes = int(os.environ.get("ROLLOUT_EPISODES", "8"))
    base_seed = int(os.environ.get("ROLLOUT_SEED", "0"))
    gif_stride = int(os.environ.get("ROLLOUT_GIF_STRIDE", "2"))

    agent = OGPOAgentLearner(cfg)
    # openpi silently downgrades --resume to a fresh start when the checkpoint
    # dir is absent or holds no checkpoints (openpi/training/checkpoints.py:56-61):
    # agent._resuming would then be False and training_steps 0, and this probe
    # would quietly roll out the UNTRAINED base pi05_libero weights under the
    # trained run's name. Refuse instead -- this repo's failures are quiet by
    # default (CLAUDE.md non-negotiables), and the whole point of this probe is
    # to look at a specific trained checkpoint's behavior.
    if not agent._resuming:
        raise ValueError(
            f"--resume was passed but no checkpoint was found under {cfg.checkpoint_dir} "
            "-- openpi downgraded to a fresh start, so this would roll out the "
            "UNTRAINED base weights, not the trained checkpoint. Check ARM/SEED/"
            "CONFIG_NAME/checkpoint_base_dir against the run that produced it."
        )
    steps_present = [int(x) for x in agent._checkpoint_manager.all_steps()]
    logging.info(
        f"[rollout] agent restored at step {agent.training_steps} from {cfg.checkpoint_dir}; "
        f"checkpoint steps present: {steps_present}"
    )

    G = cfg.collect.env_num
    env = filtered_sft_wrap_env(
        env_fn=make_env(cfg, cfg.collect.eval_tasks, num_devices=1), config=cfg, env_num=G
    )
    # sample_actions composes the policy from _train_state.ema_params, which is
    # attached only inside a collection window (filtered_sft_learner.py:329,
    # :845-851). A probe driving the agent directly has to open it itself.
    agent.start_data_collection(step=None)

    tasks = list(dict.fromkeys(cfg.collect.eval_tasks))  # de-dup, keep order
    waves = max(1, episodes // G)
    out = {
        "step": int(agent.training_steps),
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "checkpoint_steps_present": steps_present,
        "episode_steps_multiplier": cfg.collect.episode_steps_multiplier,
        "episodes_per_task": waves * G,
        "tasks": {},
    }
    summary_path = out_dir / "rollout_results.json"

    def flush():
        summary_path.write_text(json.dumps(out, indent=2, default=float))

    for task in tasks:
        # Suite-specific TimeLimit -- make_env_libero derives it from the task
        # itself (src/envs/libero.py:88-102), so this must too: TASKS is an
        # exposed override, and a non-libero_90 task would otherwise get the
        # wrong bound and be truncated mid-episode without any error.
        suite_name = "_".join(task.split("_")[:-1])
        max_steps = get_max_steps_libero(suite_name) * cfg.collect.episode_steps_multiplier
        max_chunks = max_steps // cfg.collect.replan_steps + 2
        cell = out["tasks"].setdefault(task, {"max_steps": int(max_steps), "episodes": []})
        for w in range(waves):
            # Distinct per-env seeds -- a single shared seed would give every
            # env in the wave the SAME initial state (see reset_wave), which
            # would make the episode count purely cosmetic.
            seeds = [base_seed * 7919 + w * G + i for i in range(G)]
            obs, info = reset_wave(env, task, seeds)
            tids = [task] * G
            succ, steps, frames = rollout(env, agent, obs, info, tids, max_chunks)
            for e in range(G):
                ep_idx = w * G + e
                gif = out_dir / "rollouts" / task / (
                    f"ep{ep_idx:02d}_seed{seeds[e]}_{'succ' if succ[e] else 'fail'}.gif"
                )
                write_gif(gif, frames[e], gif_stride)
                cell["episodes"].append(
                    {
                        "episode": ep_idx,
                        "seed": seeds[e],
                        "success": bool(succ[e]),
                        "steps": int(steps[e]),
                        "gif": str(gif),
                    }
                )
            sr = float(np.mean([ep["success"] for ep in cell["episodes"]]))
            logging.info(
                f"[rollout] {task} wave={w} wave_SR={float(np.mean(succ)):.3f} cum_SR={sr:.3f} "
                f"steps={steps.tolist()}"
            )
            flush()

    logging.info("[rollout] === per-task success rate ===")
    for task, cell in out["tasks"].items():
        sr = float(np.mean([e["success"] for e in cell["episodes"]]))
        logging.info(f"[rollout] {task:16s} SR={sr:.3f} n={len(cell['episodes'])}")

    agent.end_data_collection()
    env.close()
    logging.info(f"[rollout] results -> {summary_path}; gifs under {out_dir / 'rollouts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
