"""Online OGPO loop for the pi05 agent on its env families (`libero_pi05-*`, `molmo_pi05-*`).

Same collection / buffer / success-buffer semantics as `online_rl_runner.run_online_rl`
(per-step transitions flushed on episode end, BoN rollouts, start_training,
sequence batches of `horizon_length`), with pi05 observations:

  buffer['observations'] = critic features [pooled prefix | normalised state]
  buffer['images']       = (2, 224, 224, 3) uint8 base + wrist
  buffer['actions']      = raw env actions, `agent.env_action_dim` wide (libero 7, droid 8; absolute for droid,
                           the agent converts to deltas against the window-start state)

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

Parallel envs (round schedule only): a task's env slot may be a list of m envs. Collection and
eval then run m episodes of that task at once, in lockstep waves, with one batched inference call
over all m per replan (batch padded to m, so shapes never change). m = 1 reproduces the
one-env-per-task loop exactly (same reset seeds, RNG splits and task interleaving).

Vector envs (round schedule only, `env.vector_envs`): `env` / `eval_env` is a `VectorTasks`, T tasks x m
slots of one gymnasium vector env (worker processes). Waves then run every task at once, with one batched
inference pass over all T*m slots per replan (`env.inference_batch` rows per call). Collection reset seeds
match the in-process path; the rng is split per inference call instead of per task wave. With a device
`mesh` (gpu_mode=multi) the inference inputs are sharded over the devices (`_Sharder`): the agent is
replicated on the mesh, so unsharded inputs would make every device compute the whole batch; sharded, a
c-row call costs each device c / n_dev rows (the call width is rounded up to a multiple of n_dev).

Both schedules run a full actor+critic `update` on every `actor_update_interval`-th update and a
critic-only `critic_update` otherwise. `freeze_actor` (BoN-only baseline): `critic_update` only, no grad
diagnostics; the round schedule's `first_round_policy='ode'` collects round 0 with one unclipped ODE sample
(`sample_actions_plain`) instead of best-of-N, as vla-post-training's BoN before its critic is trained.
The SARSA-V critic (`use_value_fn`) lives entirely inside the agent's `update` / `critic_update`; the runner
only adds its value params to the checkpoint.

With `donate_agent`, an update frees the old agent's buffers, so nothing here may hold an agent
across an update: encoders and policies are rebuilt from the current agent.
"""

import dataclasses
import gc
import os
import pickle
import shutil
import time
from typing import Any

import jax
import numpy as np
import tqdm
from gymnasium.vector import AsyncVectorEnv

from ogpo.agents.modules.q_helper import compute_lambda_returns
from ogpo.networks.pi05 import transforms as T
from ogpo.utils.datasets import ReplayBuffer
from ogpo.utils.resume import buffer_dump_meta, load_buffer_dump, restore_trees, truncate_csvs
from ogpo.utils.sharding import shard_batch_flat


def _images(obs):
    return np.stack([T.resize_images(obs['image']), T.resize_images(obs['wrist_image'])])


def _pad(x, n):
    """Pad axis 0 to n rows by repeating the last row (padded outputs are discarded)."""
    return x if x.shape[0] == n else np.concatenate([x, np.repeat(x[-1:], n - x.shape[0], axis=0)])


class _Encoder:
    """Obs of one task -> (images, critic features). `task_id` selects the prompt (row of the agent's table).

    `__call__(obs)` encodes one obs; `batch(obs_list)` encodes k <= batch_size obs with the agent call
    padded to `batch_size` rows, so jit sees one shape.
    """

    def __init__(self, agent, task_id=0, batch_size=1):
        self.agent = agent
        self.task_id = task_id
        self.batch_size = batch_size
        self.task_ids = np.full(batch_size, task_id, np.int32)

    def batch(self, obs_list):
        imgs = np.stack([_images(o) for o in obs_list])
        states = np.stack([o['state'] for o in obs_list]).astype(np.float32)
        feats = self.agent.encode(_pad(imgs, self.batch_size), _pad(states, self.batch_size), self.task_ids)
        return imgs, np.asarray(feats)[:len(obs_list)]

    def __call__(self, obs):
        imgs, feats = self.batch([obs])
        return imgs[0], feats[0]


def _as_task_list(env):
    """-> [(name, [envs], prompt_row)]. Single env: one task on row 0. Lists: (name, env) gets row = position,
    (name, env, row) passes through (eval-only tasks live past the training rows of the prompt table).
    The env slot may be one env or a list of parallel envs of that task."""
    if not isinstance(env, (list, tuple)):
        return [("task0", [env], 0)]
    return [(t[0], list(t[1]) if isinstance(t[1], (list, tuple)) else [t[1]], t[2] if len(t) > 2 else k)
            for k, t in enumerate(env)]


def _run_waves(envs, n_episodes, seed_fn, enc, policy, rng, *, clip_actions, record, on_episode):
    """Run `n_episodes` of one task over `envs` (m slots) in lockstep waves.

    Each wave resets min(m, remaining) envs together, so all active slots replan on the same
    step; `policy(imgs, feats, task_ids, key) -> (m, H, A)` is called once per replan with the
    batch padded to m (`enc.batch_size` must be m). Slots that finish idle until the wave ends.
    `rng` is split once per replan. record=True encodes after every step and passes each episode's
    transitions to `on_episode`; otherwise obs are encoded only at replan. Episode i resets with
    `seed_fn(i)`. on_episode(dict(trajectory, info, ret, length, success)) runs in completion order.
    Returns the advanced rng.
    """
    m = len(envs)
    started = 0
    while started < n_episodes:
        k = min(m, n_episodes - started)
        obs = [envs[j].reset(seed=seed_fn(started + j))[0] for j in range(k)]
        started += k
        imgs = feats = None
        if record:
            imgs, feats = enc.batch(obs)
        active = list(range(k))
        queues = [[] for _ in range(k)]
        trajs = [[] for _ in range(k)]
        rets = [0.0] * k
        while active:
            if not queues[active[0]]:
                if not record:
                    imgs, feats = enc.batch(obs)
                rng, key = jax.random.split(rng)
                chunks = np.asarray(policy(_pad(imgs, m), _pad(feats, m), enc.task_ids, key))
                for j in active:
                    queues[j] = list(chunks[j])
            steps = {}
            for j in active:
                action = queues[j].pop(0)
                next_obs, reward, terminated, truncated, info = envs[j].step(
                    np.clip(action, -1, 1) if clip_actions else action)
                obs[j] = next_obs
                rets[j] += reward
                steps[j] = (action, reward, terminated, truncated, info)
            if record:
                next_imgs, next_feats = enc.batch(obs)
            for j in list(active):
                action, reward, terminated, truncated, info = steps[j]
                done = terminated or truncated
                if record:
                    trajs[j].append(dict(
                        observations=feats[j], images=imgs[j], actions=action, rewards=reward,
                        terminals=float(done), masks=1.0 - float(terminated),
                        next_observations=next_feats[j], next_images=next_imgs[j],
                        task_ids=np.int32(enc.task_id),
                    ))
                if done:
                    active.remove(j)
                    on_episode(dict(trajectory=trajs[j], info=info, ret=rets[j], length=envs[j].t,
                                    success=float(info.get('success', 0.0))))
            if record:
                imgs, feats = next_imgs, next_feats
    return rng


# A hung worker (e.g. LIBERO's ControlEnv.reset retries any exception forever) raises
# multiprocessing.TimeoutError instead of blocking the trainer. A LIBERO reset/step takes well under 10 s; a cold
# MolmoSpaces worker takes ~113 s to import and ~80-200 s for its first reset (smoke job 10805160), re-resets 0.7 s.
_WORKER_TIMEOUT_S = 600


@dataclasses.dataclass
class VectorTasks:
    """T tasks x m slots of one vector env. Slot s runs task s // m; its prompt-table row is rows[s // m].

    inference_batch: rows per encode / policy call (BoN memory grows linearly in rows).
    """
    vec: Any
    names: list
    rows: list
    m: int
    inference_batch: int

    @property
    def num_slots(self):
        return len(self.names) * self.m

    def slot_rows(self):
        return np.repeat(np.asarray(self.rows, np.int32), self.m)


# First prompt query loads every slot's sim in parallel (molmo: import + first scene reset per worker).
_WORKER_LOAD_TIMEOUT_S = 1800


def vector_prompts(vt):
    """{task name: prompt} read from the workers via each slot's `get_prompt()`, for env families whose prompt needs
    the loaded sim (molmo_pi05). Every slot of a task must report the same prompt."""
    vt.vec.call_async('get_prompt')
    per_slot = vt.vec.call_wait(timeout=_WORKER_LOAD_TIMEOUT_S)
    prompts = {}
    for s, prompt in enumerate(per_slot):
        name = vt.names[s // vt.m]
        if prompts.setdefault(name, prompt) != prompt:
            raise RuntimeError(f"{name}: slots disagree on the prompt ({prompts[name]!r} vs {prompt!r})")
    return prompts


def _vec_reset(vec, mask, seeds):
    """Reset the slots in `mask` with per-slot `seeds` -> obs dict of (n, ...) arrays (other rows: last obs)."""
    seeds = [int(sd) if on else None for sd, on in zip(seeds, mask)]
    options = {'reset_mask': np.asarray(mask, dtype=bool)}  # fresh dict: gymnasium pops reset_mask
    if isinstance(vec, AsyncVectorEnv):
        vec.reset_async(seed=seeds, options=options)
        return vec.reset_wait(timeout=_WORKER_TIMEOUT_S)[0]
    return vec.reset(seed=seeds, options=options)[0]


def _vec_step(vec, actions):
    if isinstance(vec, AsyncVectorEnv):
        vec.step_async(actions)
        return vec.step_wait(timeout=_WORKER_TIMEOUT_S)
    return vec.step(actions)


class _Sharder:
    """Data-parallel inference inputs over `mesh`: shards the padded (chunk, ...) arrays on axis 0, one slice
    per device. The agent is replicated on the mesh (`shard_agent`), so without this every device would
    compute the whole chunk; sharded, each device computes chunk / n_dev rows and the jitted agent call runs
    SPMD (per-row computation, no cross-row ops). `mesh=None` (single device / no mesh): identity.
    """

    def __init__(self, mesh=None):
        self.mesh = mesh
        self.n_dev = int(np.asarray(mesh.devices).size) if mesh is not None else 1

    def chunk(self, rows):
        """Rows per agent call for `rows` wanted: rounded up to a multiple of n_dev (padded rows are discarded)."""
        return -(-rows // self.n_dev) * self.n_dev

    def __call__(self, *arrays):
        if self.mesh is None:
            return arrays
        return tuple(shard_batch_flat(a, self.mesh) for a in arrays)


class _VecEncoder:
    """Vector obs (dict of (n, ...) arrays) -> (images (n, 2, 224, 224, 3), critic features (n, d)).

    One agent call per `chunk` rows, last chunk padded, so jit sees one shape. `slot_rows`: (n,) prompt rows.
    `sharder`: `_Sharder` applied to every call's inputs (default: identity).
    """

    def __init__(self, agent, slot_rows, chunk, sharder=None):
        self.agent = agent
        self.slot_rows = slot_rows
        self.chunk = chunk
        self.sharder = sharder or _Sharder()

    def batch(self, obs):
        imgs = np.stack([T.resize_images(obs['image']), T.resize_images(obs['wrist_image'])], axis=1)
        states = np.asarray(obs['state'], dtype=np.float32)
        c, feats = self.chunk, []
        for i in range(0, len(imgs), c):
            k = len(imgs[i:i + c])
            out = self.agent.encode(*self.sharder(_pad(imgs[i:i + c], c), _pad(states[i:i + c], c),
                                                  _pad(self.slot_rows[i:i + c], c)))
            feats.append(np.asarray(out)[:k])
        return imgs, np.concatenate(feats)


def _chunked_policy(policy, imgs, feats, slot_rows, rng, chunk, sharder=None):
    """policy over all rows in `chunk`-row calls (last padded), one rng split per call. -> (chunks (n, H, A), rng).

    `sharder`: `_Sharder` applied to every call's (imgs, feats, rows) (default: identity); the key is not sharded."""
    sharder = sharder or _Sharder()
    out = []
    for i in range(0, len(imgs), chunk):
        k = len(imgs[i:i + chunk])
        rng, key = jax.random.split(rng)
        out.append(np.asarray(policy(*sharder(_pad(imgs[i:i + chunk], chunk), _pad(feats[i:i + chunk], chunk),
                                              _pad(slot_rows[i:i + chunk], chunk)), key))[:k])
    return np.concatenate(out), rng


def _run_vector_waves(vt, n_per_task, seed_fn, agent, policy, rng, *, clip_actions, record, on_episode, sharder=None):
    """`_run_waves` over every task of `vt` at once: `n_per_task` episodes of each task in lockstep waves.

    Wave w resets k_w = min(m, n_per_task - w*m) slots of every task in one masked reset; episode j of
    task k in wave w resets with `seed_fn(k, w, j, k_w)`. All active slots replan together; a replan is
    one encode + policy pass over all T*m rows in `vt.inference_batch`-row calls (rounded up to a multiple
    of the device count with a `sharder`), `rng` split per call.
    Finished slots are frozen worker-side (`VectorSlotWrapper`), get zero actions and are ignored here.
    Episode length comes from the terminal `info['length']`. on_episode(dict(trajectory, info, ret,
    length, success, task)) runs in completion order (ties in slot order); `task` indexes `vt.names`.
    `sharder`: `_Sharder` over the device mesh (data-parallel inference); None = one device.
    Returns the advanced rng.
    """
    sharder = sharder or _Sharder()
    n, m = vt.num_slots, vt.m
    slot_task = np.repeat(np.arange(len(vt.names)), m)
    slot_j = np.tile(np.arange(m), len(vt.names))
    slot_rows = vt.slot_rows()
    chunk = sharder.chunk(min(vt.inference_batch, n))
    enc = _VecEncoder(agent, slot_rows, chunk, sharder)
    w = 0
    while w * m < n_per_task:
        k_w = min(m, n_per_task - w * m)
        active = slot_j < k_w
        obs = _vec_reset(vt.vec, active, [seed_fn(slot_task[s], w, slot_j[s], k_w) for s in range(n)])
        imgs = feats = None
        if record:
            imgs, feats = enc.batch(obs)
        trajs = [[] for _ in range(n)]
        rets = [0.0] * n
        chunks, h = None, 0
        while active.any():
            if chunks is None or h == chunks.shape[1]:
                if not record:
                    imgs, feats = enc.batch(obs)
                chunks, rng = _chunked_policy(policy, imgs, feats, slot_rows, rng, chunk, sharder)
                h = 0
            actions = np.where(active[:, None], chunks[:, h], 0.0).astype(np.float32)
            h += 1
            obs, reward, terminated, truncated, info = _vec_step(
                vt.vec, np.clip(actions, -1, 1) if clip_actions else actions)
            if record:
                next_imgs, next_feats = enc.batch(obs)
            for s in np.flatnonzero(active):
                done = bool(terminated[s] or truncated[s])
                rets[s] += float(reward[s])
                if record:
                    trajs[s].append(dict(
                        observations=feats[s], images=imgs[s], actions=actions[s], rewards=float(reward[s]),
                        terminals=float(done), masks=1.0 - float(terminated[s]),
                        next_observations=next_feats[s], next_images=next_imgs[s],
                        task_ids=np.int32(slot_rows[s]),
                    ))
                if done:
                    active[s] = False
                    ep_info = {key: v[s] for key, v in info.items()
                               if not key.startswith('_') and info.get('_' + key, np.ones(n, bool))[s]}
                    on_episode(dict(trajectory=trajs[s], info=ep_info, ret=rets[s], length=int(info['length'][s]),
                                    success=float(ep_info.get('success', 0.0)), task=int(slot_task[s])))
            if record:
                imgs, feats = next_imgs, next_feats
        w += 1
    return rng


def _episode_metrics(episodes):
    successes = np.array([e['success'] for e in episodes])
    lengths = np.array([e['length'] for e in episodes])
    succ = successes > 0.5
    return {
        'success': float(successes.mean()),
        'return': float(np.mean([e['ret'] for e in episodes])),
        'length': float(lengths[succ].mean()) if succ.any() else 0.0,
    }


def evaluate_pi05_vector(agent, vt, global_step, num_episodes, mode, sharder=None):
    """`evaluate_pi05` for every task of a `VectorTasks` at once -> {task name: metrics}.

    Reset seeds global_step + i for every task, as `evaluate_pi05`; one rng stream across tasks."""
    by_task = [[] for _ in vt.names]
    _run_vector_waves(vt, num_episodes, lambda k, w, j, k_w: global_step + w * vt.m + j, agent,
                      _policy(agent, mode), jax.random.PRNGKey(global_step),
                      clip_actions=agent.clip_env_actions, record=False, on_episode=lambda e: by_task[e['task']].append(e),
                      sharder=sharder)
    return {name: _episode_metrics(eps) for name, eps in zip(vt.names, by_task)}


def _policy(agent, mode):
    """-> policy(imgs, feats, task_ids, key) for `_run_waves`.

    mode: 'ode' (online actor, clipped; eval) | 'plain' (one unclipped ODE sample, target actor) | 'bon'
          | 'sample' (one unclipped ODE sample, online actor; AWR collection)."""
    # feats: the chunk-start critic features, the delta reference for droid actions (ignored for libero).
    if mode == 'ode':
        return lambda imgs, feats, ids, key: agent.sample_actions_ode(imgs, key, ids, feats)
    if mode == 'plain':
        return lambda imgs, feats, ids, key: agent.sample_actions_plain(imgs, key, ids, feats)
    if mode == 'sample':
        return lambda imgs, feats, ids, key: agent.sample_actions_explore(imgs, key, ids, feats)
    if mode != 'bon':
        raise ValueError(f"policy mode={mode!r}; expected ode | plain | bon | sample")
    return lambda imgs, feats, ids, key: agent.sample_actions_BON(imgs, feats, key, ids)


def evaluate_pi05(agent, env, global_step, num_episodes, mode, task_id=0):
    """mode: 'ode' (online actor) or 'bon' (best-of-N: target actor, target critic, pi05_bon_sampler / pi05_bon_agg).

    `env`: one env or a list of parallel envs of the task."""
    envs = list(env) if isinstance(env, (list, tuple)) else [env]
    episodes = []
    _run_waves(envs, num_episodes, lambda i: global_step + i, _Encoder(agent, task_id, len(envs)),
               _policy(agent, mode), jax.random.PRNGKey(global_step),
               clip_actions=agent.clip_env_actions, record=False, on_episode=episodes.append)
    return _episode_metrics(episodes)


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


def _make_buffer(feats, imgs, buffer_size, seed, action_dim, storage_dir=None, reuse_storage=False, lambda_returns=False):
    """storage_dir (`buffer.storage_dir`): `images` / `next_images` (~0.6 MB of a transition's ~0.62) become memmaps in
    that existing directory (node-local disk) instead of RAM; None keeps the whole buffer in RAM. reuse_storage: open the
    existing `<storage_dir>/{images,next_images}.npy` with their contents instead of creating zeroed files.
    lambda_returns=True (AWR): one more float per transition, the TD(λ) return `_recompute_lambda_returns` writes
    each round; other runs keep the original schema."""
    disk_keys = ('images', 'next_images') if storage_dir else ()
    schema = dict(
        observations=feats, images=imgs, actions=np.zeros(action_dim, np.float32),
        rewards=np.array(0.0, np.float32), terminals=np.array(0.0, np.float32), masks=np.array(1.0, np.float32),
        next_observations=feats, next_images=imgs, mc_returns=np.array(0.0, np.float32),
        task_ids=np.array(0, np.int32),
    )
    if lambda_returns:
        schema['lambda_returns'] = np.array(0.0, np.float32)
    rb = ReplayBuffer.create(schema, buffer_size, numpy_rng=np.random.default_rng(seed + 30000), disk_dir=storage_dir,
                             disk_keys=disk_keys, reuse_disk=reuse_storage)
    on_disk = sum(rb._dict[k].nbytes for k in disk_keys)
    print(f"[pi05 runner] replay buffer: {buffer_size} transitions, "
          f"{(sum(v.nbytes for v in rb._dict.values()) - on_disk) / 1e9:.1f} GB in RAM"
          + (f", {on_disk / 1e9:.1f} GB of {list(disk_keys)} memmapped in {storage_dir}" if disk_keys else ""))
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
        if 'lambda_returns' in rb:  # AWR: placeholder until the round's `_recompute_lambda_returns`
            t['lambda_returns'] = 0.0
        rb.add_transition(t)
    return is_success, float(sum(t['rewards'] for t in trajectory)), ret


def _update_once(agent, rb, shard, *, batch_size, success_buffer_batch_size, horizon_length, bc_horizon, discount,
                 bc_batch_size=None, task_balance='none', num_tasks=1, ppo_batch_size=None, is_actor=True):
    """One OGPO update (actor + RB critic + SB critic). -> (agent, info, batch, bc_batch, success_flag, seconds).

    is_actor=False: critic-only `critic_update` (RB + SB critic); no BC / PPO batches are sampled and
    bc_batch is None.

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
    if not is_actor:
        batch, batch_success = shard(batch), shard(batch_success)
        t0 = time.time()
        agent, info = agent.critic_update(batch, batch_success, success_flag)
        jax.block_until_ready(agent.critic_network.params)
        return agent, info, batch, None, success_flag, time.time() - t0
    bc_batch = _sample_success(rb, bc_batch_size, bc_horizon, discount, per_task=strat_bc)
    if bc_batch is None:
        bc_batch = rb.sample_sequence(bc_batch_size, sequence_length=bc_horizon, discount=discount, per_task=strat_bc)
    # observations: window-start critic features, the delta reference for droid's BC targets
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


def _recompute_lambda_returns(agent, rb, discount, lam, chunk=8192, n_new=0):
    """AWR (Peng et al. 2019, App. C): TD(λ) returns for every buffered transition, bootstrapped with the agent's
    current V (the previous round's fit), written into the buffer's `lambda_returns`. -> stats dict.

    The recursion runs in insertion order (FIFO across the wrap) and restarts at every episode end
    (`compute_lambda_returns`). V runs on the cached critic features (no VLM pass) in fixed-size chunks, the last
    one zero-padded, so one compile serves every call.

    Also scores that V against the observed return-to-go (`_v_mc_stats`): `vmc_*` over the whole buffer and
    `vmc_new_*` over the newest `n_new` transitions (this round's collection, which V has not been fitted on).
    """
    order = rb.insertion_order()
    n = order.shape[0]

    def values(key):
        arr, out = rb[key], np.empty(n, np.float32)
        for lo in range(0, n, chunk):
            idx = order[lo:lo + chunk]
            x = arr[idx]
            if x.shape[0] < chunk:
                x = np.concatenate([x, np.zeros((chunk - x.shape[0], *x.shape[1:]), x.dtype)])
            out[lo:lo + idx.shape[0]] = np.asarray(agent.state_values(x))[:idx.shape[0]]
        return out

    v, v_next = values('observations'), values('next_observations')
    returns = compute_lambda_returns(np.asarray(rb['rewards'][order]), np.asarray(rb['masks'][order]),
                                     np.asarray(rb['terminals'][order]), v_next, discount, lam)
    rb.write_field('lambda_returns', order, returns)
    adv = returns - v  # against the pre-fit V, for monitoring only
    stats = {'lambda_return_mean': float(returns.mean()), 'lambda_return_min': float(returns.min()),
             'lambda_return_max': float(returns.max()), 'v_pre_fit_mean': float(v.mean()),
             'adv_pre_fit_mean': float(adv.mean()), 'adv_pre_fit_std': float(adv.std())}
    if 'mc_returns' not in rb or n == 0:
        return stats
    mc = np.asarray(rb['mc_returns'][order], np.float64)
    succ = np.asarray(rb['is_success'][order]) > 0.5 if 'is_success' in rb else np.zeros(n, bool)
    tasks = np.asarray(rb['task_ids'][order]) if 'task_ids' in rb else None
    stats.update(_v_mc_stats(v, mc, succ, 'vmc_'))
    stats['target_mc_corr'] = _corr(returns, mc)  # how well the λ-target itself tracks the observed return
    stats['adv_success_mean'] = float(adv[succ].mean()) if succ.any() else float('nan')
    stats['adv_fail_mean'] = float(adv[~succ].mean()) if (~succ).any() else float('nan')
    k = min(int(n_new), n)
    if k > 1:
        new = slice(n - k, n)  # insertion order is oldest first
        stats.update(_v_mc_stats(v[new], mc[new], succ[new], 'vmc_new_'))
        if tasks is not None:
            for t in np.unique(tasks[new]):
                m = tasks[new] == t
                stats[f'vmc_new_corr/task{int(t)}'] = _corr(v[new][m], mc[new][m])
    return stats


def _corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape[0] < 2 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _v_mc_stats(v, mc, succ, prefix):
    """V(s) against the observed discounted return-to-go `mc` (`_flush_episode`). -> {prefix + name: float}.

    mc has no truncation bootstrap, so on a truncated failure it is the time-limited return, which V cannot see
    (no time feature); read the success / failure splits next to the correlations, not only the bias.
    """
    v, mc = np.asarray(v, np.float64), np.asarray(mc, np.float64)
    err = v - mc
    rank = lambda x: np.argsort(np.argsort(x, kind='stable'), kind='stable')
    split = lambda x, m: float(x[m].mean()) if m.any() else float('nan')
    out = {'corr': _corr(v, mc), 'spearman': _corr(rank(v), rank(mc)),
           'bias': float(err.mean()), 'rmse': float(np.sqrt(np.mean(err ** 2))),
           'explained_var': float(1.0 - err.var() / mc.var()) if mc.var() > 0 else float('nan'),
           'mc_mean': float(mc.mean()), 'v_mean': float(v.mean()),
           'v_success': split(v, succ), 'mc_success': split(mc, succ),
           'v_fail': split(v, ~succ), 'mc_fail': split(mc, ~succ), 'success_frac': float(succ.mean())}
    return {prefix + k: val for k, val in out.items()}


def _nanmean_dict(infos):
    return {k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float('nan') for k, v in infos.items()}


def _awr_value_fit(agent, rb, shard, *, batch_size, steps, discount):
    """AWR V fit (Alg. 1 line 5): `steps` V steps on uniformly sampled buffer states, regressing onto the stored
    R^λ (fixed for the round). -> (agent, window-mean info)."""
    infos = {}
    for _ in range(steps):
        b = rb.sample_sequence(batch_size, sequence_length=1, discount=discount)
        b = {k: b[k] for k in ('observations', 'lambda_returns', 'valid', 'task_ids') if k in b}
        agent, info = agent.value_update(shard(b))
        for k, val in jax.device_get(info).items():
            infos.setdefault(k, []).append(float(val))
    return agent, _nanmean_dict(infos)


def _awr_actor_once(agent, rb, shard, *, bc_batch_size, bc_horizon, discount, task_balance='none', num_tasks=1,
                    adv_stats=None):
    """One AWR policy step (Alg. 1 line 6) on replay-buffer windows: never the success buffer, which would make
    it filtered BC. -> (agent, info, bc_batch, seconds).

    adv_stats: (mean, std) of R^λ - V over the buffer (`awr_normalize_adv`), broadcast per window as
    `awr_adv_mean` / `awr_adv_std`; None leaves the batch unchanged."""
    strat = num_tasks if (task_balance in ('bc', 'bc_ppo', 'all') and num_tasks > 1) else None
    b = rb.sample_sequence(bc_batch_size, sequence_length=bc_horizon, discount=discount, per_task=strat)
    keys = ('images', 'observations', 'actions', 'valid', 'task_ids', 'lambda_returns')
    b = {k: b[k] for k in keys if k in b}
    if adv_stats is not None:
        n = b['lambda_returns'].shape[0]
        b['awr_adv_mean'] = np.full(n, adv_stats[0], np.float32)
        b['awr_adv_std'] = np.full(n, adv_stats[1], np.float32)
    bc_batch = shard(b)
    t0 = time.time()
    agent, info = agent.actor_update(bc_batch)
    jax.block_until_ready(agent.actor_params)
    return agent, info, bc_batch, time.time() - t0


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

    def flush(self, agent, rb, logger, step, config, batch, bc_batch, extra=None, num_tasks=1, diagnostics=True):
        """diagnostics=False (frozen actor): skip `grad_diagnostics` (needs an actor step's batches)."""
        if not self.infos:
            return
        # window means (nan-aware: MC splits are NaN on batches without that kind of transition)
        payload = {k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float('nan')
                   for k, v in self.infos.items()}
        for k in ('actor/ratio_clipped_total', 'actor/approx_kl', 'critic/td_loss', 'actor/adv_std'):
            if k in self.infos:
                payload[k + '_max'] = float(np.nanmax(self.infos[k]))
        if diagnostics and config.get('grad_diagnostics', True):
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
    mesh=None, bc_batch_size=None, actor_update_interval=1, freeze_actor=False, buffer_storage_dir=None,
):
    def _shard(b):
        return shard_batch_flat(b, mesh) if mesh is not None else b

    assert not isinstance(env, (list, tuple)), "multi-task needs training.collection_mode=rounds"
    enc = _Encoder(agent)
    obs, _ = env.reset(seed=seed + 10000)
    imgs, feats = enc(obs)

    rb = _make_buffer(feats, imgs, buffer_size, seed, agent.env_action_dim, storage_dir=buffer_storage_dir)
    online_rng = jax.random.PRNGKey(seed + 40000)
    episode_counter, trajectory, queue, n_updates = 0, [], [], 0
    batch = bc_batch = None
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
            is_actor = not freeze_actor and n_updates % actor_update_interval == 0
            agent, info_u, batch_u, bc_batch_u, success_flag, dt = _update_once(
                agent, rb, _shard, is_actor=is_actor, **update_kw)
            if is_actor:  # grad diagnostics use the last actor step's batches
                batch, bc_batch = batch_u, bc_batch_u
            enc.agent = agent  # the old agent is dead under donate_agent
            n_updates += 1
            window.add(info_u, success_flag, dt)

        if i % log_interval == 0 and window.infos:
            window.flush(agent, rb, logger, i, config, batch, bc_batch, diagnostics=not freeze_actor)

        # No evals while only collecting (policy unchanged since step 0), as online_rl_runner's eval_start.
        if eval_interval > 0 and i > start_training and (i % eval_interval == 0 or i == online_steps):
            _run_eval(agent, eval_env, i, eval_episodes, logger, rb)
            save_checkpoint(agent, save_dir, i)

    return agent, rb


def run_pi05_round_rl(
    agent, env, eval_env, config, logger, *,
    num_rounds, rollouts_per_round, updates_per_round, batch_size, success_buffer_batch_size,
    horizon_length, bc_horizon, discount, buffer_size, eval_interval, eval_episodes, log_interval, seed,
    save_dir=None, mesh=None, bc_batch_size=None, task_balance='none', actor_update_interval=1,
    freeze_actor=False, first_round_policy='bon', resume=None, buffer_storage_dir=None,
    resume_snapshot_dir=None, buffer_reuse_storage=False, awr_value_steps=0, awr_value_refreshes=1,
):
    """Round-based collection, as vla-post-training's exp.py (FSFT / BoN / AWR):

        repeat num_rounds:  collect rollouts_per_round full episodes PER TASK (policy frozen)
                            -> updates_per_round OGPO updates on the replay buffer

    The step axis is the update count. Episodes are logged per round under env/*
    (round means, plus env/success/<task> with several tasks) with cumulative
    env/episodes and env/env_steps. `env` / `eval_env`: one env, or [(task_name, env), ...]; the env slot may
    be a list of m parallel envs (episodes run in waves of m per task, tasks interleaved per wave), or a
    `VectorTasks` (all tasks' waves at once in worker processes).

    `resume`: a `ResumeStore`. Each round's rows and an agent checkpoint every `resume.ckpt_interval` updates are mirrored to
    its remote; if it already holds a manifest the run continues from it (see ogpo/utils/resume.py).

    `resume_snapshot_dir`: continue from a snapshot of a live run taken between two collections:
    `agent.pkl` (a `save_checkpoint` file written at the start of round R), `buffer/` (`save_buffer_dump` of the buffer after
    round R's collection) and `csv/` (the run's CSV logs). The images / next_images memmaps must already be in
    `buffer_storage_dir`, opened with `buffer_reuse_storage`. Continues with round R's updates. Exclusive with `resume`.

    AWR (`pi05_actor_objective: awr`, Peng et al. 2019 Alg. 1, as the reference `awr_agent._update`): every round
    collects with the online actor ('sample'), then `awr_value_refreshes` times recomputes the buffer's TD(λ) returns
    with the current V and fits V for `awr_value_steps` steps on those frozen targets, then recomputes R^λ once more
    with the fitted V (the actor's A = R^λ - V uses the fitted V on both sides) and runs `updates_per_round` AWR
    policy steps (the step axis counts policy steps only). `awr_normalize_adv` standardises A with that last pass's
    buffer-wide mean / std. A resume mid-round skips the V fit and runs only the last pass with the restored V.
    Eval is ODE only (there is no trained Q for best-of-N).
    """
    if resume is not None and resume_snapshot_dir:
        raise ValueError("run_pi05_round_rl: resume (ResumeStore) and resume_snapshot_dir are exclusive")
    if buffer_reuse_storage and not (resume_snapshot_dir and buffer_storage_dir):
        raise ValueError("run_pi05_round_rl: buffer_reuse_storage needs resume_snapshot_dir and buffer_storage_dir")
    if resume_snapshot_dir and buffer_storage_dir and not buffer_reuse_storage:
        raise ValueError("run_pi05_round_rl: resume_snapshot_dir with buffer_storage_dir needs buffer_reuse_storage "
                         "(creating the memmaps would zero the snapshot's images)")
    def _shard(b):
        return shard_batch_flat(b, mesh) if mesh is not None else b

    sharder = _Sharder(mesh)  # vector envs: data-parallel inference over the mesh
    vector = isinstance(env, VectorTasks)
    if vector:
        task_names, m = list(env.names), env.m
        probe = np.zeros(env.num_slots, dtype=bool)
        probe[0] = True
        obs = _vec_reset(env.vec, probe, [seed + 10000] + [None] * (env.num_slots - 1))
        imgs, feats = _VecEncoder(agent, env.slot_rows(), sharder.chunk(min(env.inference_batch, env.num_slots)),
                                  sharder).batch(obs)
        imgs, feats = imgs[0], feats[0]  # example transition for the buffer schema
    else:
        tasks = _as_task_list(env)
        task_names = [name for name, _, _ in tasks]
        m = len(tasks[0][1])
        if any(len(task_envs) != m for _, task_envs, _ in tasks):
            raise ValueError(f"every task needs the same number of parallel envs; got {[len(e) for _, e, _ in tasks]}")
        obs, _ = tasks[0][1][0].reset(seed=seed + 10000)
        imgs, feats = _Encoder(agent, tasks[0][2])(obs)  # example transition for the buffer schema
    awr = config.get('pi05_actor_objective', 'ppo') == 'awr'
    eval_modes = ('ode',) if awr else ('ode', 'bon')
    rb = _make_buffer(feats, imgs, buffer_size, seed, agent.env_action_dim, storage_dir=buffer_storage_dir,
                      reuse_storage=buffer_reuse_storage, lambda_returns=awr)
    rng = jax.random.PRNGKey(seed + 40000)
    window = _LogWindow()
    num_tasks = len(task_names)
    update_kw = dict(batch_size=batch_size, success_buffer_batch_size=success_buffer_batch_size,
                     horizon_length=horizon_length, bc_horizon=bc_horizon, discount=discount,
                     bc_batch_size=bc_batch_size, task_balance=task_balance, num_tasks=num_tasks,
                     ppo_batch_size=config['ppo_batch_size'])
    if task_balance != 'none':
        print(f"[pi05 runner] task_balance={task_balance} over {num_tasks} tasks")
    total_updates = num_rounds * updates_per_round
    step, episodes, env_steps = 0, 0, 0
    batch = bc_batch = None
    start_round, skip_collect = 0, False
    prefixes = ("eval", "eval_sde", "env", "online_agent") + (("awr",) if awr else ())
    csv_files = [os.path.join(save_dir, f"{p}.csv") for p in prefixes] if save_dir else []
    if resume is not None:
        if save_dir is None:
            raise ValueError("training.resume_s3 needs experiment.log=true (the CSV logs are part of the resumed state)")
        man = resume.manifest()
        if man is not None:
            off = resume.set_offset(man)
            total_updates = num_rounds * updates_per_round + off
            agent, step, episodes, env_steps = _resume_state(resume, man, agent, rb, save_dir, updates_per_round, seed)
            start_round = (step - off) // updates_per_round
            skip_collect = start_round == man['round']  # that round's rows are in the buffer; continue its updates
            rng = jax.random.fold_in(rng, start_round)
    if resume_snapshot_dir:
        if save_dir is None:
            raise ValueError("resume_snapshot_dir needs experiment.log=true (the CSV logs are part of the resumed state)")
        agent, step, episodes, env_steps = _resume_snapshot(resume_snapshot_dir, agent, rb, save_dir, updates_per_round, seed)
        start_round = step // updates_per_round
        skip_collect = True  # round R's rows are in the buffer; continue its updates
        rng = jax.random.fold_in(rng, start_round)

    if eval_interval > 0 and step == 0:
        _run_eval(agent, eval_env, 0, eval_episodes, logger, rb, sharder=sharder, modes=eval_modes)
    if resume is not None and step == 0:
        resume.save_agent(_agent_state(agent, 0), 0, csv_files)

    pbar = tqdm.tqdm(total=total_updates, initial=step, dynamic_ncols=True, desc="Round RL")
    for r in range(start_round, num_rounds):
        n_new = 0  # transitions collected this round (the λ pass scores V on them as held-out data)
        if not (skip_collect and r == start_round):
            t0 = time.time()
            pointer_before, total_before = rb.pointer, rb.total_added
            succ, lengths, returns = [], [], []
            succ_by_task = {name: [] for name in task_names}
            mode = 'sample' if awr else ('plain' if (r == 0 and first_round_policy == 'ode') else 'bon')
            collect_policy = _policy(agent, mode)

            def on_episode(ep, name):
                nonlocal episodes, env_steps
                is_success, ep_return, _ = _flush_episode(rb, ep['trajectory'], ep['info'], discount)
                succ.append(is_success)
                succ_by_task[name].append(is_success)
                lengths.append(ep['length'])
                returns.append(ep_return)
                episodes += 1
                env_steps += ep['length']

            if vector:
                # Same reset seeds as the in-process loop below: wave w, task k, slot j -> the
                # (w * T * m + k * k_w + j)-th episode of the round.
                base = episodes
                rng = _run_vector_waves(env, rollouts_per_round,
                                        lambda k, w, j, k_w: seed + 10000 + base + w * num_tasks * m + k * k_w + j,
                                        agent, collect_policy, rng, clip_actions=False, record=True,
                                        on_episode=lambda ep: on_episode(ep, task_names[ep['task']]), sharder=sharder)
            else:
                for w in range(0, rollouts_per_round, m):
                    for name, task_envs, row in tasks:  # interleaved, as vla-post-training's task queue
                        base = episodes
                        rng = _run_waves(task_envs, min(m, rollouts_per_round - w),
                                         lambda i, base=base: seed + 10000 + base + i,
                                         _Encoder(agent, row, m), collect_policy, rng, clip_actions=False, record=True,
                                         on_episode=lambda ep, name=name: on_episode(ep, name))
            succ_len = [l for s, l in zip(succ, lengths) if s > 0.5]
            payload = {'success': float(np.mean(succ)), 'length': float(np.mean(succ_len)) if succ_len else 0.0,
                       'return': float(np.mean(returns)), 'round': r, 'episodes': episodes, 'env_steps': env_steps,
                       'collect_minutes': (time.time() - t0) / 60}
            if num_tasks > 1:
                payload.update({f'success/{name}': float(np.mean(v)) for name, v in succ_by_task.items()})
            logger.log(payload, "env", step=step)
            per_task = (" | " + " ".join(f"{n}={np.mean(v):.2f}" for n, v in succ_by_task.items())) if num_tasks > 1 else ""
            print(f"[ROUND {r}] step={step} | collected {rollouts_per_round * num_tasks} eps ({mode}), success {np.mean(succ):.2f}"
                  f"{per_task} | total {episodes} eps / {env_steps} env steps | rb={rb.size} | {(time.time() - t0) / 60:.1f} min")
            n_new = rb.total_added - total_before
            if resume is not None:
                resume.save_chunk(rb, r, pointer_before, total_before, {'episodes': episodes, 'env_steps': env_steps})

        off = resume.step_offset if resume is not None else 0
        if awr:
            fresh = step - off == r * updates_per_round  # False when resuming mid-round: V_k is already fitted
            t0 = time.time()
            lam = config['awr_lambda']
            awr_info = {}
            if fresh:
                # reference `_update`: targets from the current V, frozen for the fit; K refreshes = K iterations
                for k in range(awr_value_refreshes):
                    pre = _recompute_lambda_returns(agent, rb, discount, lam, n_new=n_new if k == 0 else 0)
                    if k == 0:  # V of the previous round, scored on this round's (unseen) data
                        awr_info.update({f'awr/{key}': val for key, val in pre.items()})
                    agent, v_info = _awr_value_fit(agent, rb, _shard, batch_size=batch_size, steps=awr_value_steps,
                                                   discount=discount)
                    if k == awr_value_refreshes - 1:
                        awr_info.update(v_info)
            # reference: vals and new_vals recomputed with the fitted V before the actor update
            post = _recompute_lambda_returns(agent, rb, discount, lam, n_new=0 if fresh else n_new)
            if not fresh:  # mid-round resume: this is the only pass
                awr_info.update({f'awr/{key}': val for key, val in post.items()})
            awr_info.update({f'awr/post_{key.replace("_pre_fit", "")}': val for key, val in post.items()
                             if not key.startswith('vmc_new_')})
            adv_stats = ((post['adv_pre_fit_mean'], post['adv_pre_fit_std'])
                         if config.get('awr_normalize_adv', False) else None)
            awr_info.update({'round': r, 'value_fit_steps': awr_value_steps * awr_value_refreshes if fresh else 0,
                             'value_refreshes': awr_value_refreshes if fresh else 0,
                             'minutes': (time.time() - t0) / 60})
            logger.log(awr_info, "awr", step=step)
            print(f"[AWR] round {r} step={step} | R^λ mean {awr_info['awr/lambda_return_mean']:.3f} "
                  f"[{awr_info['awr/lambda_return_min']:.2f}, {awr_info['awr/lambda_return_max']:.2f}] | "
                  + (f"V fit {awr_value_refreshes} x {awr_value_steps} steps, last v_loss "
                     f"{awr_info.get('value/v_loss', float('nan')):.4f}"
                     if fresh else "mid-round resume: V fit skipped")
                  + f" | post-fit R^λ {post['lambda_return_mean']:.3f}, A mean {post['adv_pre_fit_mean']:.3f} "
                  f"std {post['adv_pre_fit_std']:.3f}" + (" (normalised for the actor)" if adv_stats else "")
                  + f" | {awr_info['minutes']:.1f} min")
            if 'awr/vmc_corr' in awr_info:
                g = lambda k: awr_info.get(f'awr/{k}', float('nan'))
                print(f"[AWR] round {r} V vs MC return (V before this round's fit) | buffer: corr {g('vmc_corr'):.3f} "
                      f"spearman {g('vmc_spearman'):.3f} EV {g('vmc_explained_var'):.3f} bias {g('vmc_bias'):.2f} | "
                      f"new data: corr {g('vmc_new_corr'):.3f} EV {g('vmc_new_explained_var'):.3f} "
                      f"bias {g('vmc_new_bias'):.2f} | λ-target corr {g('target_mc_corr'):.3f} | "
                      f"A succ/fail {g('adv_success_mean'):.2f}/{g('adv_fail_mean'):.2f}")
                print(f"[AWR] round {r} V vs MC return (fitted V, in-sample) | corr {g('post_vmc_corr'):.3f} "
                      f"spearman {g('post_vmc_spearman'):.3f} EV {g('post_vmc_explained_var'):.3f} "
                      f"bias {g('post_vmc_bias'):.2f} | A succ/fail {g('post_adv_success_mean'):.2f}/"
                      f"{g('post_adv_fail_mean'):.2f}")

        for _ in range(updates_per_round - (step - off - r * updates_per_round)):
            if rb.size < batch_size * horizon_length:
                raise RuntimeError(f"replay buffer has {rb.size} transitions, need {batch_size * horizon_length}")
            if awr:
                agent, info_u, bc_batch_u, dt = _awr_actor_once(
                    agent, rb, _shard, bc_batch_size=bc_batch_size or success_buffer_batch_size,
                    bc_horizon=bc_horizon, discount=discount, task_balance=task_balance, num_tasks=num_tasks,
                    adv_stats=adv_stats)
                batch, bc_batch, success_flag = None, bc_batch_u, False  # AWR's diagnostics need only the windows
            else:
                is_actor = not freeze_actor and step % actor_update_interval == 0
                agent, info_u, batch_u, bc_batch_u, success_flag, dt = _update_once(
                    agent, rb, _shard, is_actor=is_actor, **update_kw)
                if is_actor:  # grad diagnostics use the last actor step's batches
                    batch, bc_batch = batch_u, bc_batch_u
            window.add(info_u, success_flag, dt)
            step += 1
            pbar.update(1)
            if step % log_interval == 0:
                window.flush(agent, rb, logger, step, config, batch, bc_batch,
                             extra={'episodes': episodes, 'env_steps': env_steps, 'round': r}, num_tasks=num_tasks,
                             diagnostics=not freeze_actor)
            if eval_interval > 0 and (step % eval_interval == 0 or step == total_updates):
                _run_eval(agent, eval_env, step, eval_episodes, logger, rb, sharder=sharder, modes=eval_modes)
                save_checkpoint(agent, save_dir, step)
            if resume is not None and (step % resume.ckpt_interval == 0 or step == (r + 1) * updates_per_round + off):
                resume.save_agent(_agent_state(agent, step), step, csv_files)
    pbar.close()
    if resume is not None:
        resume.close()
    return agent, rb


def _agent_state(agent, step):
    """Trainable state on the host (frozen PaliGemma comes from the pi05 checkpoint)."""
    state = dict(
        step=step, actor_params=agent.actor_params, target_actor_params=agent.target_actor_params,
        actor_opt_state=agent.actor_opt_state, critic_params=agent.critic_network.params,
        critic_opt_state=agent.critic_network.opt_state, critic_step=agent.critic_network.step,
    )
    if agent.value_network is not None:  # SARSA-V critic (use_value_fn)
        state.update(value_params=agent.value_network.params, value_opt_state=agent.value_network.opt_state,
                     value_step=agent.value_network.step)
    return jax.device_get(state)


def _restore_agent(agent, state):
    """Put an `_agent_state` pytree (params, target params, optimiser states, steps) back into `agent`, in place on the device:
    the replaced buffers are deleted as the new ones are placed (`restore_trees`), so `agent` must not be used afterwards."""
    nets = [("critic", agent.critic_network, "critic")]
    if agent.value_network is not None:
        nets.append(("value", agent.value_network, "value"))
    pairs = [(agent.actor_params, state["actor_params"], "actor_params"),
             (agent.target_actor_params, state["target_actor_params"], "target_actor_params"),
             (agent.actor_opt_state, state["actor_opt_state"], "actor_opt_state")]
    for key, net, _ in nets:
        pairs += [(net.params, state[f"{key}_params"], f"{key}_params"), (net.opt_state, state[f"{key}_opt_state"], f"{key}_opt_state")]
        if isinstance(net.step, jax.Array):
            pairs.append((net.step, state[f"{key}_step"], f"{key}_step"))
    # a Python-int step is not restored by restore_trees; read it now so a bad state raises before any buffer is deleted
    int_steps = {key: int(state[f"{key}_step"]) for key, net, _ in nets if not isinstance(net.step, jax.Array)}
    # leaves that stay with the agent (frozen / init params, consts, rng) must survive the deletions
    kept = agent.replace(actor_params=None, target_actor_params=None, actor_opt_state=None, critic_network=None, value_network=None)
    done = iter(restore_trees(pairs, protect={id(x) for x in jax.tree.leaves(kept)}))
    replace = dict(actor_params=next(done), target_actor_params=next(done), actor_opt_state=next(done))
    for key, net, _ in nets:
        params, opt_state = next(done), next(done)
        step = next(done) if isinstance(net.step, jax.Array) else int_steps[key]
        replace["critic_network" if key == "critic" else "value_network"] = net.replace(params=params, opt_state=opt_state, step=step)
    return agent.replace(**replace)


def _resume_snapshot(snapshot_dir, agent, rb, save_dir, updates_per_round, seed):
    """Load `snapshot_dir` (see run_pi05_round_rl): agent + optimiser state, the buffer's RAM fields and counters, the CSV
    logs. -> (agent, update step, episodes, env_steps)."""
    with open(os.path.join(snapshot_dir, "agent.pkl"), "rb") as f:
        state = pickle.load(f)
    step = int(state["step"])
    dump_round = buffer_dump_meta(os.path.join(snapshot_dir, "buffer"))["round"]  # checked before the 35 GB dump is read
    if dump_round * updates_per_round != step:
        raise ValueError(f"{snapshot_dir}: agent.pkl is at update {step} but the buffer dump is after round {dump_round} "
                         f"(= update {dump_round * updates_per_round} with updates_per_round={updates_per_round}); "
                         f"they are not a consistent pair")
    agent = _restore_agent(agent, state)  # validates the agent trees before the buffer is loaded, as the ResumeStore path did
    meta = load_buffer_dump(rb, os.path.join(snapshot_dir, "buffer"))
    rb.numpy_rng = np.random.default_rng(seed + 30000 + step)
    truncate_csvs(os.path.join(snapshot_dir, "csv"), save_dir, step, keep_at_step=True)  # env.csv logs round R before its updates
    print(f"[resume] continuing at update {step} (round {step // updates_per_round}, buffer through that round: "
          f"{rb.size} transitions, {meta['episodes']} episodes) from {snapshot_dir}", flush=True)
    return agent, step, int(meta["episodes"]), int(meta["env_steps"])


def _resume_state(resume, man, agent, rb, save_dir, updates_per_round, seed):
    """Load the manifest's agent / buffer / CSV logs. -> (agent, update step, episodes, env_steps)."""
    state, csv_dir, chunk_dirs = resume.fetch(man)
    agent = _restore_agent(agent, state)  # validates the agent trees before the (possibly TB-scale) buffer is loaded
    meta = resume.load_buffer(rb, chunk_dirs)
    step = int(state["step"])
    rb.numpy_rng = np.random.default_rng(seed + 30000 + step)
    rnd = (step - resume.step_offset) // updates_per_round
    truncate_csvs(csv_dir, save_dir, step, keep_at_step=rnd == man["round"])
    shutil.rmtree(os.path.join(resume.local, "fetched"), ignore_errors=True)
    print(f"[resume] continuing at update {step} (round {rnd}, buffer through round {man['round']}: "
          f"{rb.size} transitions, {meta['episodes']} episodes)", flush=True)
    return agent, step, int(meta["episodes"]), int(meta["env_steps"])


def save_checkpoint(agent, save_dir, step):
    """Trainable state only (frozen PaliGemma comes from the pi05 checkpoint). Overwrites the previous one."""
    if save_dir is None:
        return
    state = _agent_state(agent, step)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "checkpoint_latest.pkl")
    with open(path + ".tmp", "wb") as f:
        pickle.dump(state, f)
    os.replace(path + ".tmp", path)
    print(f"[pi05 runner] saved {path} (step {step})")


def _run_eval(agent, eval_env, step, eval_episodes, logger, rb, sharder=None, modes=("ode", "bon")):
    """`eval_episodes` episodes per task; several tasks -> overall mean plus success/<task>.

    `sharder`: `_Sharder` for a `VectorTasks` eval env (data-parallel inference); None = one device."""
    gc.collect()
    for mode, prefix in ((m, p) for m, p in (('ode', 'eval'), ('bon', 'eval_sde')) if m in modes):
        t0 = time.time()
        if isinstance(eval_env, VectorTasks):
            per_task = evaluate_pi05_vector(agent, eval_env, step, eval_episodes, mode, sharder=sharder)
        else:
            per_task = {name: evaluate_pi05(agent, task_env, step, eval_episodes, mode, task_id=row)
                        for name, task_env, row in _as_task_list(eval_env)}
        ev = {key: float(np.mean([v[key] for v in per_task.values()])) for key in ('success', 'return', 'length')}
        if len(per_task) > 1:
            ev.update({f'success/{name}': v['success'] for name, v in per_task.items()})
        logger.log(ev, prefix, step=step)
        detail = (" | " + " ".join(f"{n}={v['success']:.2f}" for n, v in per_task.items())) if len(per_task) > 1 else ""
        buffer = (f" | rb={rb.size} ({100 * rb._traj_success_mask[:rb.size].mean() if rb.size else 0:.1f}% success)"
                  if rb is not None else "")
        print(f"[EVAL {mode.upper()}] step={step} | success={ev['success']:.3f}{detail}{buffer} | {(time.time() - t0) / 60:.1f} min")


def load_checkpoint(agent, path, keep_base_actor=False):
    """Restore what `save_checkpoint` wrote (actor, target actor, critic [+ SARSA-V value]) into `agent`; the optimiser
    states stay as freshly initialised. Call before sharding the agent. keep_base_actor: leave the (freshly built, i.e. base
    pi05) actor and target actor and restore only the critic [+ value]. -> (agent, step the checkpoint was saved at)."""
    with open(path, "rb") as f:
        state = pickle.load(f)

    def _like(name, cur, new):
        if jax.tree.structure(cur) != jax.tree.structure(new):
            raise ValueError(f"{path}: {name} tree does not match the agent (same agent flags as the training run?)")
        for c, n in zip(jax.tree.leaves(cur), jax.tree.leaves(new)):
            if tuple(c.shape) != tuple(n.shape):
                raise ValueError(f"{path}: {name} leaf shape {tuple(n.shape)} != the agent's {tuple(c.shape)}")
        return jax.tree.map(lambda c, n: jax.numpy.asarray(n, dtype=c.dtype), cur, new)

    replace = {} if keep_base_actor else dict(
        actor_params=_like("actor_params", agent.actor_params, state["actor_params"]),
        target_actor_params=_like("target_actor_params", agent.target_actor_params, state["target_actor_params"]))
    replace.update(
        critic_network=agent.critic_network.replace(
            params=_like("critic_params", agent.critic_network.params, state["critic_params"])),
    )
    if agent.value_network is not None:
        if "value_params" not in state:
            raise ValueError(f"{path} has no value_params but the agent has a SARSA-V value network")
        replace["value_network"] = agent.value_network.replace(
            params=_like("value_params", agent.value_network.params, state["value_params"]))
    print(f"[pi05 runner] restored {path} (step {state['step']})")
    return agent.replace(**replace), int(state["step"])


def run_pi05_eval_only(agent, eval_env, logger, *, step, eval_episodes, mesh=None, modes=("ode", "bon")):
    """One ODE eval (online actor) and one BoN eval (target actor + target critic) of a restored agent on `eval_env`,
    logged at `step` (reset seeds = step + episode index, as the training evals at that step)."""
    _run_eval(agent, eval_env, step, eval_episodes, logger, None, sharder=_Sharder(mesh), modes=modes)
