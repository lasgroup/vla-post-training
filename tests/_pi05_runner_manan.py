"""VERBATIM copy of ogpo/runners/pi05_online_runner.py at manan/libero-vla-fix 73a4169
(`git show MERGE_HEAD:ogpo/runners/pi05_online_runner.py`), the pre-merge reference for the
differential tests in tests/test_pi05_multitask.py / tests/test_pi05_schedule.py. Do not edit.

Intended deviations from 73a4169 (each mirrors a deliberate runner change, so the comparison stays exact):
  - _update_once keeps 'observations' in bc_batch (docs/changes/2026-10-14-molmo-pi05: window-start critic
    features = the delta reference for droid BC targets).
"""
# fmt: off
"""Online OGPO loop for the pi05 agent on `libero_pi05-*` envs.

Same collection / buffer / success-buffer semantics as `online_rl_runner.run_online_rl`
(per-step transitions flushed on episode end, BoN rollouts, start_training,
sequence batches of `horizon_length`), with pi05 observations:

  buffer['observations'] = critic features [pooled prefix | normalised state]
  buffer['images']       = (2, 224, 224, 3) uint8 base + wrist
  buffer['actions']      = raw 7-D env actions

The BC anchor needs 10-step windows, so it gets its own `bc_batch`
(success buffer when ready, replay buffer otherwise).

Two schedules: `run_pi05_online_rl` (one update per env step after start_training,
as OGPO's online_rl_runner) and `run_pi05_round_rl` (collection rounds of full
episodes with a frozen policy, then a block of updates, as vla-post-training).

Multi-task (round schedule only, as vla-post-training's collect.tasks): `env` / `eval_env`
may be a list of (task_name, env); every transition carries `task_ids` (row of the agent's
prompt table), each round collects `rollouts_per_round` episodes per task (interleaved), and
eval runs `eval_episodes` per task, logging the overall mean plus `success/<task>`. A single
env is the one-task case (task id 0 everywhere), unchanged.
"""

import gc
import os
import pickle
import time

import jax
import numpy as np
import tqdm

from ogpo.networks.pi05 import transforms as T
from ogpo.utils.datasets import ReplayBuffer
from ogpo.utils.sharding import shard_batch_flat


def _images(obs):
    return np.stack([T.resize_images(obs['image']), T.resize_images(obs['wrist_image'])])


class _Encoder:
    """Per-step obs -> (images, critic features). `task_id` selects the prompt (row of the agent's table)."""

    def __init__(self, agent, task_id=0):
        self.agent = agent
        self.task_id = task_id
        self.task_ids = np.array([task_id], np.int32)

    def __call__(self, obs):
        imgs = _images(obs)
        feats = self.agent.encode(imgs[None], obs['state'][None].astype(np.float32), self.task_ids)
        return imgs, np.asarray(feats[0])


def _as_task_list(env):
    """-> [(name, env, prompt_row)]. Single env: one task on row 0. Lists: (name, env) gets row = position,
    (name, env, row) passes through (eval-only tasks live past the training rows of the prompt table)."""
    if not isinstance(env, (list, tuple)):
        return [("task0", env, 0)]
    return [(t[0], t[1], t[2] if len(t) > 2 else k) for k, t in enumerate(env)]


def evaluate_pi05(agent, env, global_step, num_episodes, mode, task_id=0):
    """mode: 'ode' (online actor) or 'bon' (SDE best-of-N, target actor + target critic)."""
    rng = jax.random.PRNGKey(global_step)
    enc = _Encoder(agent, task_id)
    successes, returns, lengths = [], [], []
    for i in range(num_episodes):
        obs, _ = env.reset(seed=global_step + i)
        done, ret, queue, info = False, 0.0, [], {}
        while not done:
            if not queue:
                rng, key = jax.random.split(rng)
                imgs, feats = enc(obs)
                if mode == 'ode':
                    chunk = agent.sample_actions_ode(imgs[None], key, enc.task_ids)
                else:
                    chunk = agent.sample_actions_BON(imgs[None], feats[None], key, enc.task_ids)
                queue = list(np.asarray(chunk[0]))
            obs, r, term, trunc, info = env.step(np.clip(queue.pop(0), -1, 1))
            ret += r
            done = term or trunc
        successes.append(float(info.get('success', 0.0)))
        returns.append(ret)
        lengths.append(env.t)
    succ = np.array(successes) > 0.5
    return {
        'success': float(np.mean(successes)),
        'return': float(np.mean(returns)),
        'length': float(np.mean(np.array(lengths)[succ])) if succ.any() else 0.0,
    }


def _sample_success(rb, batch_size, seq_len, discount, per_task=None):
    if rb._traj_success_mask[:rb.size].sum() < batch_size + seq_len:
        return None
    return rb.sample_sequence(batch_size, sequence_length=seq_len, discount=discount, success_only=True,
                              per_task=per_task)


def _task_counts(rb, num_tasks):
    """Buffer composition per task: (transitions, successful transitions) for task ids 0..num_tasks-1."""
    n = rb.size
    if n == 0 or 'task_ids' not in rb._dict:
        return {}
    task = np.asarray(rb._dict['task_ids'][:n])
    succ = rb._traj_success_mask[:n]
    return {k: (int((task == k).sum()), int((succ & (task == k)).sum())) for k in range(num_tasks)}


def _make_buffer(feats, imgs, buffer_size, seed):
    rb = ReplayBuffer.create(dict(
        observations=feats, images=imgs, actions=np.zeros(T.ENV_ACTION_DIM, np.float32),
        rewards=np.array(0.0, np.float32), terminals=np.array(0.0, np.float32), masks=np.array(1.0, np.float32),
        next_observations=feats, next_images=imgs, mc_returns=np.array(0.0, np.float32),
        task_ids=np.array(0, np.int32),
    ), buffer_size, numpy_rng=np.random.default_rng(seed + 30000))
    print(f"[pi05 runner] replay buffer: {buffer_size} transitions, "
          f"{sum(v.nbytes for v in rb._dict.values()) / 1e9:.1f} GB")
    return rb


def _flush_episode(rb, trajectory, info, discount):
    """Tag a finished episode with success / MC return-to-go and add it. -> (is_success, return, mc_return_0)."""
    is_success = float(info.get('success', 0.0))
    ret = 0.0  # discounted return-to-go, no truncation bootstrap (diagnostics only)
    for t in reversed(trajectory):
        ret = float(t['rewards']) + discount * ret
        t['mc_returns'] = ret
    for t in trajectory:
        t['is_success'] = is_success
        t['score'] = is_success if is_success != 0 else 1e-9
        rb.add_transition(t)
    return is_success, float(sum(t['rewards'] for t in trajectory)), ret


def _update_once(agent, rb, shard, *, batch_size, success_buffer_batch_size, horizon_length, bc_horizon, discount,
                 bc_batch_size=None, task_balance='none', num_tasks=1, ppo_batch_size=None):
    """One OGPO update (actor + RB critic + SB critic). -> (agent, info, batch, bc_batch, success_flag, seconds).

    batch_size: TD critic batch (the PPO states are drawn from it); success_buffer_batch_size: success-buffer
    critic batch; bc_batch_size: BC anchor batch (default: success_buffer_batch_size, the original coupling).
    task_balance (multi-task): 'none' = uniform over transitions everywhere (original);
      'bc' = BC anchor stratified by task (equal windows per task);
      'bc_ppo' = also a separate task-stratified PPO state batch of ppo_batch_size (instead of a subsample of the
                 TD batch); the two critic batches stay transition-proportional;
      'all' = every batch stratified by task.
    """
    bc_batch_size = bc_batch_size or success_buffer_batch_size
    K = num_tasks if (task_balance != 'none' and num_tasks > 1) else None
    strat_bc = K if task_balance in ('bc', 'bc_ppo', 'all') else None
    strat_critic = K if task_balance == 'all' else None
    batch = rb.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount, per_task=strat_critic)
    batch_success = _sample_success(rb, success_buffer_batch_size, horizon_length, discount, per_task=strat_critic)
    success_flag = batch_success is not None
    if not success_flag:
        batch_success = batch
    bc_batch = _sample_success(rb, bc_batch_size, bc_horizon, discount, per_task=strat_bc)
    if bc_batch is None:
        bc_batch = rb.sample_sequence(bc_batch_size, sequence_length=bc_horizon, discount=discount, per_task=strat_bc)
    bc_batch = {k: bc_batch[k] for k in ('images', 'observations', 'actions', 'valid', 'task_ids') if k in bc_batch}
    ppo_batch = None
    if K is not None and task_balance in ('bc_ppo', 'all') and ppo_batch_size:
        pb = rb.sample_sequence(ppo_batch_size, sequence_length=horizon_length, discount=discount, per_task=K)
        ppo_batch = {k: pb[k] for k in ('images', 'observations', 'task_ids') if k in pb}
    batch, batch_success, bc_batch = shard(batch), shard(batch_success), shard(bc_batch)
    ppo_batch = shard(ppo_batch) if ppo_batch is not None else None

    t0 = time.time()
    agent, info = agent.update(batch, batch_success, bc_batch, success_flag, ppo_batch)
    jax.block_until_ready(agent.actor_params)
    return agent, info, batch, bc_batch, success_flag, time.time() - t0


class _LogWindow:
    """Accumulates update infos between log steps; logs window means (+ a few maxes and grad diagnostics)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.infos, self.t_update, self.n_update = {}, 0.0, 0

    def add(self, info, success_flag, seconds):
        for k, v in jax.device_get(info).items():
            self.infos.setdefault(k, []).append(float(v))
        self.infos.setdefault('success_flag', []).append(float(success_flag))
        self.t_update += seconds
        self.n_update += 1

    def flush(self, agent, rb, logger, step, config, batch, bc_batch, extra=None, num_tasks=1):
        if not self.infos:
            return
        # window means (nan-aware: MC splits are NaN on batches without that kind of transition)
        payload = {k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float('nan')
                   for k, v in self.infos.items()}
        for k in ('actor/ratio_clipped_total', 'actor/approx_kl', 'critic/td_loss', 'actor/adv_std'):
            if k in self.infos:
                payload[k + '_max'] = float(np.nanmax(self.infos[k]))
        if config.get('grad_diagnostics', True):
            diag = agent.grad_diagnostics(batch, bc_batch)
            payload.update({f'diag/{k}': float(v) for k, v in jax.device_get(diag).items()})
        payload['updates_in_window'] = self.n_update
        payload['sec_per_update'] = self.t_update / max(self.n_update, 1)
        payload['rb_size'] = rb.size
        payload['rb_success_frac'] = float(rb._traj_success_mask[:rb.size].mean()) if rb.size else 0.0
        if num_tasks > 1:  # buffer composition per task: share of transitions, share of the success buffer
            counts = _task_counts(rb, num_tasks)
            n_all = max(sum(c[0] for c in counts.values()), 1)
            n_succ = max(sum(c[1] for c in counts.values()), 1)
            for k, (n_k, s_k) in counts.items():
                payload[f'rb_share/task{k}'] = n_k / n_all
                payload[f'rb_success_share/task{k}'] = s_k / n_succ
                payload[f'rb_success_frac/task{k}'] = s_k / max(n_k, 1)
        payload.update(extra or {})
        logger.log(payload, "online_agent", step=step)
        self.reset()


def run_pi05_online_rl(
    agent, env, eval_env, config, logger, *,
    online_steps, start_training, batch_size, success_buffer_batch_size, horizon_length,
    bc_horizon, discount, buffer_size, eval_interval, eval_episodes, log_interval, seed, save_dir=None,
    mesh=None, bc_batch_size=None,
):
    def _shard(b):
        return shard_batch_flat(b, mesh) if mesh is not None else b

    assert not isinstance(env, (list, tuple)), "multi-task needs training.collection_mode=rounds"
    enc = _Encoder(agent)
    obs, _ = env.reset(seed=seed + 10000)
    imgs, feats = enc(obs)

    rb = _make_buffer(feats, imgs, buffer_size, seed)
    online_rng = jax.random.PRNGKey(seed + 40000)
    episode_counter, trajectory, queue = 0, [], []
    window = _LogWindow()
    update_kw = dict(batch_size=batch_size, success_buffer_batch_size=success_buffer_batch_size,
                     horizon_length=horizon_length, bc_horizon=bc_horizon, discount=discount,
                     bc_batch_size=bc_batch_size)

    if eval_interval > 0:
        _run_eval(agent, eval_env, 0, eval_episodes, logger, rb)

    for i in tqdm.tqdm(range(1, online_steps + 1), dynamic_ncols=True, desc="Online RL"):
        online_rng, key = jax.random.split(online_rng)
        if not queue:
            chunk = agent.sample_actions_BON(imgs[None], feats[None], key, enc.task_ids)
            queue = list(np.asarray(chunk[0]))
        action = queue.pop(0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_imgs, next_feats = enc(next_obs)
        done = terminated or truncated
        trajectory.append(dict(
            observations=feats, images=imgs, actions=action, rewards=reward,
            terminals=float(done), masks=1.0 - float(terminated),
            next_observations=next_feats, next_images=next_imgs, task_ids=np.int32(0),
        ))

        if done:
            is_success, ep_return, mc0 = _flush_episode(rb, trajectory, info, discount)
            logger.log({'success': is_success, 'length': env.t, 'episode': episode_counter,
                        'return': ep_return, 'mc_return_0': mc0}, "env", step=i)
            trajectory, queue = [], []
            episode_counter += 1
            obs, _ = env.reset(seed=seed + 10000 + episode_counter)
            imgs, feats = enc(obs)
        else:
            imgs, feats = next_imgs, next_feats

        if i >= start_training and rb.size >= batch_size * horizon_length:
            agent, info_u, batch, bc_batch, success_flag, dt = _update_once(agent, rb, _shard, **update_kw)
            window.add(info_u, success_flag, dt)

        if i % log_interval == 0 and window.infos:
            window.flush(agent, rb, logger, i, config, batch, bc_batch)

        # No evals while only collecting (policy unchanged since step 0), as online_rl_runner's eval_start.
        if eval_interval > 0 and i > start_training and (i % eval_interval == 0 or i == online_steps):
            _run_eval(agent, eval_env, i, eval_episodes, logger, rb)
            save_checkpoint(agent, save_dir, i)

    return agent, rb


def _collect_episode(agent, env, enc, rng, reset_seed):
    """One full BoN episode with the current (frozen) target actor / critic. -> (trajectory, info, rng)."""
    obs, _ = env.reset(seed=reset_seed)
    imgs, feats = enc(obs)
    trajectory, queue, done, info = [], [], False, {}
    while not done:
        if not queue:
            rng, key = jax.random.split(rng)
            queue = list(np.asarray(agent.sample_actions_BON(imgs[None], feats[None], key, enc.task_ids)[0]))
        action = queue.pop(0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_imgs, next_feats = enc(next_obs)
        done = terminated or truncated
        trajectory.append(dict(
            observations=feats, images=imgs, actions=action, rewards=reward,
            terminals=float(done), masks=1.0 - float(terminated),
            next_observations=next_feats, next_images=next_imgs, task_ids=np.int32(enc.task_id),
        ))
        imgs, feats = next_imgs, next_feats
    return trajectory, info, rng


def run_pi05_round_rl(
    agent, env, eval_env, config, logger, *,
    num_rounds, rollouts_per_round, updates_per_round, batch_size, success_buffer_batch_size,
    horizon_length, bc_horizon, discount, buffer_size, eval_interval, eval_episodes, log_interval, seed,
    save_dir=None, mesh=None, bc_batch_size=None, task_balance='none',
):
    """Round-based collection, as vla-post-training's exp.py (FSFT / BoN / AWR):

        repeat num_rounds:  collect rollouts_per_round full episodes PER TASK (policy frozen)
                            -> updates_per_round OGPO updates on the replay buffer

    The step axis is the update count. Episodes are logged per round under env/*
    (round means, plus env/success/<task> with several tasks) with cumulative
    env/episodes and env/env_steps. `env` / `eval_env`: one env, or [(task_name, env), ...].
    """
    def _shard(b):
        return shard_batch_flat(b, mesh) if mesh is not None else b

    tasks = _as_task_list(env)
    encs = [_Encoder(agent, row) for _, _, row in tasks]
    enc = encs[0]
    obs, _ = tasks[0][1].reset(seed=seed + 10000)
    imgs, feats = enc(obs)  # example transition for the buffer schema
    rb = _make_buffer(feats, imgs, buffer_size, seed)
    rng = jax.random.PRNGKey(seed + 40000)
    window = _LogWindow()
    num_tasks = len(tasks)
    update_kw = dict(batch_size=batch_size, success_buffer_batch_size=success_buffer_batch_size,
                     horizon_length=horizon_length, bc_horizon=bc_horizon, discount=discount,
                     bc_batch_size=bc_batch_size, task_balance=task_balance, num_tasks=num_tasks,
                     ppo_batch_size=config['ppo_batch_size'])
    if task_balance != 'none':
        print(f"[pi05 runner] task_balance={task_balance} over {num_tasks} tasks")
    total_updates = num_rounds * updates_per_round
    step, episodes, env_steps = 0, 0, 0

    if eval_interval > 0:
        _run_eval(agent, eval_env, 0, eval_episodes, logger, rb)

    pbar = tqdm.tqdm(total=total_updates, dynamic_ncols=True, desc="Round RL")
    for r in range(num_rounds):
        t0 = time.time()
        succ, lengths, returns = [], [], []
        succ_by_task = {name: [] for name, _, _ in tasks}
        for _ in range(rollouts_per_round):
            for k, (name, task_env, _row) in enumerate(tasks):  # interleaved, as vla-post-training's task queue
                trajectory, info, rng = _collect_episode(agent, task_env, encs[k], rng, seed + 10000 + episodes)
                is_success, ep_return, _ = _flush_episode(rb, trajectory, info, discount)
                succ.append(is_success)
                succ_by_task[name].append(is_success)
                lengths.append(task_env.t)
                returns.append(ep_return)
                episodes += 1
                env_steps += task_env.t
        succ_len = [l for s, l in zip(succ, lengths) if s > 0.5]
        payload = {'success': float(np.mean(succ)), 'length': float(np.mean(succ_len)) if succ_len else 0.0,
                   'return': float(np.mean(returns)), 'round': r, 'episodes': episodes, 'env_steps': env_steps,
                   'collect_minutes': (time.time() - t0) / 60}
        if len(tasks) > 1:
            payload.update({f'success/{name}': float(np.mean(v)) for name, v in succ_by_task.items()})
        logger.log(payload, "env", step=step)
        per_task = (" | " + " ".join(f"{n}={np.mean(v):.2f}" for n, v in succ_by_task.items())) if len(tasks) > 1 else ""
        print(f"[ROUND {r}] step={step} | collected {rollouts_per_round * len(tasks)} eps, success {np.mean(succ):.2f}"
              f"{per_task} | total {episodes} eps / {env_steps} env steps | rb={rb.size} | {(time.time() - t0) / 60:.1f} min")

        for _ in range(updates_per_round):
            if rb.size < batch_size * horizon_length:
                raise RuntimeError(f"replay buffer has {rb.size} transitions, need {batch_size * horizon_length}")
            agent, info_u, batch, bc_batch, success_flag, dt = _update_once(agent, rb, _shard, **update_kw)
            window.add(info_u, success_flag, dt)
            step += 1
            pbar.update(1)
            if step % log_interval == 0:
                window.flush(agent, rb, logger, step, config, batch, bc_batch,
                             extra={'episodes': episodes, 'env_steps': env_steps, 'round': r}, num_tasks=num_tasks)
            if eval_interval > 0 and (step % eval_interval == 0 or step == total_updates):
                _run_eval(agent, eval_env, step, eval_episodes, logger, rb)
                save_checkpoint(agent, save_dir, step)
    pbar.close()
    return agent, rb


def save_checkpoint(agent, save_dir, step):
    """Trainable state only (frozen PaliGemma comes from the pi05 checkpoint). Overwrites the previous one."""
    if save_dir is None:
        return
    state = jax.device_get(dict(
        step=step, actor_params=agent.actor_params, target_actor_params=agent.target_actor_params,
        actor_opt_state=agent.actor_opt_state, critic_params=agent.critic_network.params,
        critic_opt_state=agent.critic_network.opt_state,
    ))
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "checkpoint_latest.pkl")
    with open(path + ".tmp", "wb") as f:
        pickle.dump(state, f)
    os.replace(path + ".tmp", path)
    print(f"[pi05 runner] saved {path} (step {step})")


def _run_eval(agent, eval_env, step, eval_episodes, logger, rb):
    """`eval_episodes` episodes per task; several tasks -> overall mean plus success/<task>."""
    gc.collect()
    tasks = _as_task_list(eval_env)
    for mode, prefix in (('ode', 'eval'), ('bon', 'eval_sde')):
        per_task = {name: evaluate_pi05(agent, task_env, step, eval_episodes, mode, task_id=row)
                    for name, task_env, row in tasks}
        ev = {key: float(np.mean([v[key] for v in per_task.values()])) for key in ('success', 'return', 'length')}
        if len(tasks) > 1:
            ev.update({f'success/{name}': v['success'] for name, v in per_task.items()})
        logger.log(ev, prefix, step=step)
        detail = (" | " + " ".join(f"{n}={v['success']:.2f}" for n, v in per_task.items())) if len(tasks) > 1 else ""
        print(f"[EVAL {mode.upper()}] step={step} | success={ev['success']:.3f}{detail} | rb={rb.size} "
              f"({100 * rb._traj_success_mask[:rb.size].mean() if rb.size else 0:.1f}% success)")
