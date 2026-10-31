"""Critic ranking diagnostic: best-of-N rollouts with the pick swapped, logging every candidate's Q.

Same rollouts as the BoN eval (`evaluate_pi05_vector`, mode bon: N target-actor candidates per replan, target critic,
reset seeds step + episode index, rng PRNGKey(step)), but the chunk executed at each replan is picked by `select`:
best (the usual argmax of the `pi05_bon_agg` scores), worst (argmin of the same scores) or random (uniform). Run the
three rules on the same seeds: a critic that ranks actions well gives best > random > worst.

Per episode it saves the per-step rewards and, at each replan, the env step, all N candidates' Q of every head and the
picked index, so Q of the executed chunk can be compared with its Monte Carlo return-to-go.
"""
import os
import pickle
import time

import jax
import numpy as np

from ogpo.runners.pi05_online_runner import _Sharder, _VecEncoder, _pad, _vec_reset, _vec_step


def run_bon_select_diag(agent, vt, step, n_per_task, select, out_path, mesh=None):
    """`n_per_task` episodes of every task of the `VectorTasks` `vt` with BoN pick `select` -> list of episode dicts,
    also pickled to `out_path`. Episode keys: task, task_index, seed, select, success, length, terminated, rewards (T,),
    replan_t (R,) env step of each replan, q (R, num_qs, N) every candidate's Q, idx (R,) the picked candidate."""
    sharder = _Sharder(mesh)
    n, m = vt.num_slots, vt.m
    slot_task = np.repeat(np.arange(len(vt.names)), m)
    slot_j = np.tile(np.arange(m), len(vt.names))
    slot_rows = vt.slot_rows()
    chunk = sharder.chunk(min(vt.inference_batch, n))
    enc = _VecEncoder(agent, slot_rows, chunk, sharder)
    rng = jax.random.PRNGKey(step)
    clip = agent.clip_env_actions
    episodes, t0, w = [], time.time(), 0
    while w * m < n_per_task:
        k_w = min(m, n_per_task - w * m)
        active = slot_j < k_w
        seeds = [step + w * m + int(slot_j[s]) for s in range(n)]  # = step + episode index, as the evals
        obs = _vec_reset(vt.vec, active, seeds)
        rec = [dict(t=[], q=[], idx=[], rewards=[]) for _ in range(n)]
        steps = np.zeros(n, np.int32)
        chunks, h = None, 0
        while active.any():
            if chunks is None or h == chunks.shape[1]:
                imgs, feats = enc.batch(obs)
                outs = []
                for i in range(0, n, chunk):
                    k = len(imgs[i:i + chunk])
                    rng, key = jax.random.split(rng)
                    si, sf, sr = sharder(_pad(imgs[i:i + chunk], chunk), _pad(feats[i:i + chunk], chunk),
                                         _pad(slot_rows[i:i + chunk], chunk))
                    a, q, idx = agent.sample_actions_BON_select(si, sf, key, sr, select=select)
                    outs.append((np.asarray(a)[:k], np.asarray(q)[:, :k], np.asarray(idx)[:k]))
                chunks = np.concatenate([o[0] for o in outs])
                q_all = np.concatenate([o[1] for o in outs], axis=1)  # (num_qs, n, N)
                idx_all = np.concatenate([o[2] for o in outs])
                for s in np.flatnonzero(active):
                    rec[s]['t'].append(int(steps[s]))
                    rec[s]['q'].append(q_all[:, s].astype(np.float32))
                    rec[s]['idx'].append(int(idx_all[s]))
                h = 0
            actions = np.where(active[:, None], chunks[:, h], 0.0).astype(np.float32)
            h += 1
            obs, reward, terminated, truncated, info = _vec_step(vt.vec, np.clip(actions, -1, 1) if clip else actions)
            for s in np.flatnonzero(active):
                rec[s]['rewards'].append(float(reward[s]))
                steps[s] += 1
                if terminated[s] or truncated[s]:
                    active[s] = False
                    ep_info = {key: v[s] for key, v in info.items()
                               if not key.startswith('_') and info.get('_' + key, np.ones(n, bool))[s]}
                    r = rec[s]
                    episodes.append(dict(
                        task=vt.names[slot_task[s]], task_index=int(slot_task[s]), seed=seeds[s], select=select,
                        success=float(ep_info.get('success', 0.0)), length=int(info['length'][s]),
                        terminated=bool(terminated[s]), rewards=np.asarray(r['rewards'], np.float32),
                        replan_t=np.asarray(r['t'], np.int32), q=np.stack(r['q']), idx=np.asarray(r['idx'], np.int32)))
        w += 1
        print(f"[DIAG {select}] wave {w}: {len(episodes)} episodes, {(time.time() - t0) / 60:.1f} min", flush=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path + ".tmp", "wb") as f:
        pickle.dump(dict(step=step, select=select, tasks=list(vt.names), episodes=episodes), f)
    os.replace(out_path + ".tmp", out_path)
    by_task = {name: np.mean([e['success'] for e in episodes if e['task'] == name]) for name in vt.names}
    print(f"[DIAG {select.upper()}] step={step} | success={np.mean([e['success'] for e in episodes]):.3f} | "
          + " ".join(f"{k}={v:.2f}" for k, v in by_task.items())
          + f" | {(time.time() - t0) / 60:.1f} min | {out_path}", flush=True)
    return episodes
