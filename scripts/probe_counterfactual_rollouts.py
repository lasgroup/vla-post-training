#!/usr/bin/env python
"""Tier-C probe: are the critic's Best-of-N rankings CORRECT, not merely small?

Tier A measured the critic's action response surface; Tier B measured where real
pi0.5 candidates sit on it and found the within-state Q spread is ~16% of the
critic's own MC error. Both are statements about MAGNITUDE. Neither can say
whether the ordering is right -- a small signal pointing the right way is still
worth having. Only counterfactual rollouts answer that.

METHOD
  For a probe state s reached by the deployed policy:
    - take the N candidate chunks Best-of-N actually chose between, and their
      critic scores (recorded from the production selection path via the opt-in
      `_bon_record` hook -- NOT a reimplementation of the scoring block);
    - execute candidate k in env k, then let every env continue under pi to
      termination;
    - record the realised discounted return G_k.
  Then compare the critic's ordering of {Q_k} against the realised {G_k}.

GETTING BACK TO s. LIBERO exposes set_init_state/get_sim_state, and the round
trip is exact (sim.set_state_from_flattened + sim.forward, no settle steps).
It is NOT used here: those methods sit on the raw LIBERO env, so calling them
through the vector env bypasses Pi0ObservationWrapper and QueryFrequencyWrapper
entirely and returns a raw obs the policy cannot consume, with the TimeLimit
counter unrestored. Instead every env in a wave is driven to s by DETERMINISTIC
REPLAY: identical seeds (LiberoWrapper.reset draws its init state from a seeded
rng) plus the trunk's own recorded action chunks, through the real step path.
Phase 0 asserts that determinism rather than assuming it.

WHAT THIS COSTS, AND WHY `--probe.phase calibrate` EXISTS FIRST
  The decisive statistic is Delta = G(critic's argmax) - mean_k G_k, paired per
  state. Its per-state noise is ~1.06 * sigma_cont, where sigma_cont is the
  spread of returns from executing the SAME action and continuing under pi.
  Detecting a Tier-B-sized effect needs N ~ (1.06 * sigma_cont / target_SE)^2
  states, and at ~1 min per state that is the difference between a 1-hour job
  and a 17-hour one. `calibrate` measures sigma_cont (same candidate in all G
  envs, different continuation seeds) and prints the required N. Run it, read
  the number, then decide on `full`.

Config comes through the same tyro CLI as scripts/exp.py, so the recipe supplies
it verbatim (ENTRY=scripts/probe_counterfactual_rollouts.py bash
scripts/ogpo_multitask_4task_ref.sh ... --resume) and there is no second copy of
the flag block to drift.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import pathlib
import sys

import jax
import numpy as np

mp.set_start_method("spawn", force=True)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.envs import make_env  # noqa: E402
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env  # noqa: E402
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner  # noqa: E402
import src.training.config as _config  # noqa: E402
from src.training.utils import init_logging  # noqa: E402


def parse_probe_args() -> argparse.Namespace:
    """Strips --probe.* out of argv so the remainder is exp.py's exact CLI."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--probe.phase", dest="phase",
                    choices=["determinism", "calibrate", "full"], default="calibrate")
    ap.add_argument("--probe.trunks", dest="trunks", type=int, default=8,
                    help="trunk episodes to roll out (one wave of env_num at a time)")
    ap.add_argument("--probe.per-episode", dest="per_episode", type=int, default=3,
                    help="probe points sampled per trunk episode")
    ap.add_argument("--probe.max-probes", dest="max_probes", type=int, default=48)
    ap.add_argument("--probe.repeats", dest="repeats", type=int, default=1,
                    help="continuation repeats per candidate (full phase)")
    ap.add_argument("--probe.calib-states", dest="calib_states", type=int, default=10)
    ap.add_argument("--probe.target-se", dest="target_se", type=float, default=2.0,
                    help="return-unit SE the calibrate phase sizes N against")
    ap.add_argument("--probe.seed", dest="seed", type=int, default=12345)
    ap.add_argument("--probe.out", dest="out", default="probe_tierc.json")
    known, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    return known


# --------------------------------------------------------------------------- #
# env driving
# --------------------------------------------------------------------------- #
def reset_wave(env, task_id: str, seed: int):
    """Resets every env to the SAME task and SAME seed => identical initial states.

    LiberoWrapper.reset picks its init state with self.rng (seeded by seed()), so
    a per-worker identical seed list is what makes the wave a controlled comparison.
    """
    env.seed([int(seed)] * env.env_num)
    obs, info = env.reset(options={"task_id": [task_id] * env.env_num})
    return obs, info


def policy_chunk(agent, obs, info, task_ids):
    out = agent.sample_actions(
        obs, task_description=info["task_description"], task_id=list(task_ids)
    )
    return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)


def rollout(env, agent, obs, info, task_ids, gamma, first_action=None,
            record_chunks=False, max_chunks=200):
    """Runs a wave to termination.

    Returns (per-env discounted return, env steps, success, recorded chunks,
    final obs). Envs that finish early stop being stepped; their return is
    frozen at the terminating sub-step.
    """
    G = env.env_num
    ret = np.zeros(G); disc = np.ones(G); steps = np.zeros(G, int)
    done = np.zeros(G, bool); succ = np.zeros(G, bool)
    chunks: list[np.ndarray] = []
    n = 0
    while not done.all() and n < max_chunks:
        if first_action is not None:
            act = first_action
            first_action = None
        else:
            act = policy_chunk(agent, obs, info, task_ids)
        if record_chunks:
            chunks.append(act.copy())
        live_ids = np.where(~done)[0]
        nobs, rew, term, trunc, _ = env.step(act[live_ids], id=live_ids.tolist())
        rew = np.asarray(rew, dtype=np.float64)
        term = np.asarray(term, dtype=bool); trunc = np.asarray(trunc, dtype=bool)
        tt = np.logical_or(term, trunc)
        for j in range(rew.shape[1]):
            for row, e in enumerate(live_ids):
                if done[e]:
                    continue
                ret[e] += disc[e] * rew[row, j]
                disc[e] *= gamma
                steps[e] += 1
                if tt[row, j]:
                    done[e] = True
                    succ[e] = bool(term[row, j])
        # Scatter the stepped envs' observations back into the full-width obs.
        def _scatter(prev, new):
            prev = np.array(prev, copy=True)
            prev[live_ids] = new
            return prev
        obs = jax.tree.map(_scatter, obs, nobs)
        n += 1
    return ret, steps, succ, chunks, obs


# --------------------------------------------------------------------------- #
def main() -> int:
    pa = parse_probe_args()
    cfg = _config.cli()
    init_logging()

    # HARD GUARD. initialize_checkpoint_dir is called with overwrite=not resume
    # (filtered_sft_learner.py:256), and overwrite=True does checkpoint_dir.rmtree().
    # A missing --resume would therefore DELETE a finished 100k-step run before
    # this probe read a single number from it. Refuse rather than risk it.
    if not cfg.resume:
        raise ValueError(
            "Tier C must run with --resume. Without it the learner constructs its "
            "checkpoint manager with overwrite=True, which wipes "
            f"{cfg.checkpoint_dir} on startup. Add --resume (the sbatch does)."
        )

    gamma = float(cfg.rl.discount)
    G = int(cfg.rl.n_samples)
    if G < 2:
        raise ValueError(
            f"Tier C needs rl.n_samples >= 2 (best-of-N candidates); got {G}. "
            "Run the probe against a best-of-N config (BON_N=8)."
        )
    logging.info(f"[tierC] phase={pa.phase} G={G} gamma={gamma} tasks={cfg.collect.tasks}")

    agent = OGPOAgentLearner(cfg)
    steps_present = [int(x) for x in agent._checkpoint_manager.all_steps()]
    logging.info(f"[tierC] agent restored at step {agent.training_steps}; "
                 f"checkpoint steps present: {steps_present}")
    env = filtered_sft_wrap_env(
        env_fn=make_env(cfg, cfg.collect.tasks, num_devices=1), config=cfg, env_num=G
    )
    # sample_actions composes the policy from _train_state.ema_params, and the
    # learner attaches that ONLY inside a collection window: __init__ slices the
    # EMA out to self._ema and sets ema_params=None (filtered_sft_learner.py:329),
    # start_data_collection puts it back (:845-851), end_data_collection removes
    # it again (:874-876). Calling sample_actions outside that window raises
    # AttributeError in compose_full_params. collect.py:123 opens the window; a
    # probe that drives the agent directly has to do the same.
    if cfg.collect.env_num != G:
        raise ValueError(
            f"start_data_collection sizes _episode_storage by collect.env_num="
            f"{cfg.collect.env_num}, but this probe drives {G} envs (rl.n_samples). "
            "Set --collect.env_num to match rl.n_samples."
        )
    agent.start_data_collection(step=None)
    # Nothing below calls add_data/save_episode, so the replay buffer is untouched;
    # end_data_collection(step=None) takes the eval path and skips the registry guard.
    tasks = list(cfg.collect.tasks)
    out: dict = {"phase": pa.phase, "step": int(agent.training_steps),
                 "gamma": gamma, "group": G, "tasks": tasks,
                 "checkpoint_steps_present": steps_present,
                 "checkpoint_dir": str(cfg.checkpoint_dir)}
    outp = pathlib.Path(pa.out)

    def flush():
        outp.write_text(json.dumps(out, indent=2, default=float))

    # ---------------- phase 0: is the wave actually a controlled comparison? --
    # Same task, same seed, same actions in every env => byte-identical rewards.
    # If this fails, replay cannot recover a probe state and Tier C is off.
    task0 = tasks[0]
    obs, info = reset_wave(env, task0, pa.seed)
    tids = [task0] * G
    act = policy_chunk(agent, obs, info, tids)
    common = np.repeat(act[:1], G, axis=0)          # every env gets env 0's chunk
    r0, s0, _u0, _c, _o = rollout(env, agent, obs, info, tids, gamma,
                                  first_action=common, max_chunks=3)
    spread = float(np.max(r0) - np.min(r0))
    out["determinism"] = {
        "returns": r0.tolist(), "steps": s0.tolist(), "max_minus_min": spread,
        "identical": spread == 0.0,
    }
    logging.info(f"[tierC] determinism check: return spread over {G} "
                 f"identically-driven envs = {spread:.6g}")
    flush()
    if pa.phase == "determinism":
        agent.end_data_collection(); env.close(); return 0
    if spread != 0.0:
        # Not fatal for `calibrate` (which measures continuation noise anyway),
        # but it means replay-to-s is approximate; say so loudly and record it.
        logging.warning("[tierC] envs are NOT bit-identical under identical driving; "
                        "replay-to-probe-state is approximate. Recorded in JSON.")

    rng = np.random.default_rng(pa.seed)

    # ---------------- phase: calibrate ---------------------------------------
    # sigma_cont = spread of realised return when the SAME candidate is executed
    # and the continuation is left to its own randomness. This is the noise floor
    # every Tier-C comparison sits on.
    if pa.phase == "calibrate":
        rows = []
        for i in range(pa.calib_states):
            task = tasks[i % len(tasks)]
            seed = int(rng.integers(0, 2**31 - 1))
            obs, info = reset_wave(env, task, seed)
            tids = [task] * G
            # Walk a random distance in so probe states are not all at t=0.
            for _ in range(int(rng.integers(1, 12))):
                a = policy_chunk(agent, obs, info, tids)
                _r, _s, _u, _c, obs = rollout(env, agent, obs, info, tids, gamma,
                                              first_action=a, max_chunks=1)
            a = policy_chunk(agent, obs, info, tids)
            common = np.repeat(a[:1], G, axis=0)
            r, st, sc, _c, _o = rollout(env, agent, obs, info, tids, gamma,
                                        first_action=common)
            rows.append({"task": task, "seed": seed, "returns": r.tolist(),
                         "steps": st.tolist(), "success": sc.tolist(),
                         "sigma": float(r.std(ddof=1)),
                         "success_rate": float(sc.mean())})
            logging.info(f"[tierC] calib {i+1}/{pa.calib_states} {task} "
                         f"sigma_cont={rows[-1]['sigma']:.4g} "
                         f"SR={rows[-1]['success_rate']:.2f} "
                         f"ret={r.mean():.4g}")
            out["calibration"] = rows
            flush()
        sig = float(np.mean([r["sigma"] for r in rows]))
        need = (1.06 * sig / pa.target_se) ** 2
        out["sigma_cont_mean"] = sig
        out["states_needed_for_target_se"] = {"target_se": pa.target_se,
                                              "n_states": need}
        logging.info(f"[tierC] sigma_cont={sig:.4g} -> {need:.0f} states needed "
                     f"for SE={pa.target_se} on Delta (at repeats=1)")
        flush()
        agent.end_data_collection(); env.close(); return 0

    # ---------------- phase: full --------------------------------------------
    # Trunk pass: roll the deployed policy and bank (task, seed, chunk prefix).
    probes = []
    for w in range(max(1, pa.trunks // G)):
        task = tasks[w % len(tasks)]
        seed = int(rng.integers(0, 2**31 - 1))
        obs, info = reset_wave(env, task, seed)
        tids = [task] * G
        r, st, sc, chunks, _o = rollout(env, agent, obs, info, tids, gamma,
                                        record_chunks=True)
        n_chunks = len(chunks)
        if n_chunks < 4:
            continue
        # Stratify probe points across the episode rather than clustering at t=0.
        for frac in np.linspace(0.2, 0.8, pa.per_episode):
            d = int(frac * n_chunks)
            if d < 1:
                continue
            probes.append({"task": task, "seed": seed, "d": d,
                           "prefix": [c[0].copy() for c in chunks[:d]],
                           "trunk_return": float(r[0]), "trunk_success": bool(sc[0])})
        logging.info(f"[tierC] trunk {w+1}: {task} len={n_chunks} chunks, "
                     f"SR={sc.mean():.2f}, banked {len(probes)} probes")
    rng.shuffle(probes)
    probes = probes[:pa.max_probes]
    # Task order matters: LiberoWrapper.reset rebuilds the whole env whenever the
    # bddl file changes, so group probes by task instead of interleaving.
    probes.sort(key=lambda p: p["task"])
    out["n_probes"] = len(probes)
    flush()

    records = []
    for pi, p in enumerate(probes):
        for rep in range(pa.repeats):
            obs, info = reset_wave(env, p["task"], p["seed"])
            tids = [p["task"]] * G
            # Replay the trunk prefix identically in every env.
            for c in p["prefix"]:
                a = np.repeat(np.asarray(c)[None], G, axis=0)
                _r, _s, _u, _c2, obs = rollout(env, agent, obs, info, tids, gamma,
                                               first_action=a, max_chunks=1)
            # Candidates + scores straight out of the production selection path.
            agent._bon_record = []
            _ = policy_chunk(agent, obs, info, tids)
            rec = agent._bon_record
            agent._bon_record = None
            if len(rec) != 1:
                raise RuntimeError(
                    f"expected one best-of-N task group (all {G} envs share a task) "
                    f"but sample_actions recorded {len(rec)}; the wave is not "
                    "single-task, which breaks the candidate/env alignment."
                )
            cands = rec[0]["candidates"]          # (G, n_samples, H, act)
            scores = rec[0]["scores"]             # (G, n_samples)
            # All G envs are in the same state, so env 0's candidate set is THE
            # candidate set; assign candidate k to env k.
            q = scores[0]
            first = np.stack([cands[0, k] for k in range(G)], axis=0)
            r, st, sc, _c3, _o = rollout(env, agent, obs, info, tids, gamma,
                                         first_action=first)
            records.append({
                "probe": pi, "rep": rep, "task": p["task"], "seed": p["seed"],
                "d": p["d"], "q": q.tolist(), "return": r.tolist(),
                "steps": st.tolist(), "success": sc.tolist(),
            })
            logging.info(
                f"[tierC] probe {pi+1}/{len(probes)} rep{rep} {p['task']} d={p['d']} "
                f"Q[{q.min():.4g},{q.max():.4g}] G[{r.min():.4g},{r.max():.4g}] "
                f"SR={sc.mean():.2f}")
            out["records"] = records
            flush()

    # ---------------- the verdict -------------------------------------------
    from scipy.stats import spearmanr
    by_probe: dict[int, list] = {}
    for rec in records:
        by_probe.setdefault(rec["probe"], []).append(rec)
    rhos, deltas, regrets, sr_pick, sr_rand = [], [], [], [], []
    for _, reps in by_probe.items():
        q = np.asarray(reps[0]["q"], dtype=np.float64)
        g = np.mean([r["return"] for r in reps], axis=0)
        s = np.mean([r["success"] for r in reps], axis=0)
        if np.ptp(q) == 0 or np.ptp(g) == 0:
            continue
        rhos.append(float(spearmanr(q, g).statistic))
        k = int(np.argmax(q))
        deltas.append(float(g[k] - g.mean()))
        regrets.append(float(g.max() - g[k]))
        sr_pick.append(float(s[k])); sr_rand.append(float(s.mean()))
    n = len(rhos)
    out["verdict"] = {
        "n_states": n,
        "spearman_mean": float(np.mean(rhos)) if n else None,
        "spearman_frac_positive": float(np.mean(np.asarray(rhos) > 0)) if n else None,
        # The money number: what Best-of-N buys in REALISED return over a random pick.
        "delta_argmax_minus_mean": float(np.mean(deltas)) if n else None,
        "delta_se": float(np.std(deltas, ddof=1) / np.sqrt(n)) if n > 1 else None,
        "regret_vs_oracle": float(np.mean(regrets)) if n else None,
        "success_rate_critic_pick": float(np.mean(sr_pick)) if n else None,
        "success_rate_random_pick": float(np.mean(sr_rand)) if n else None,
    }
    logging.info(f"[tierC] VERDICT {json.dumps(out['verdict'], indent=2, default=float)}")
    flush()
    agent.end_data_collection()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
