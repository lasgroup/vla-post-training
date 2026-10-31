"""
CPU tests for pi05 multi-task / parallel-env collection after the merge of manan/libero-vla-fix
(docs/changes/2026-10-06-merge-manan-libero-vla-fix; originally docs/changes/2026-09-28-pi05-multitask-collection).

  a. Dataset.sample_sequence: task_ids (Manan's contract, no next_task_ids), per_task quotas; unchanged output
     without the field (vs a verbatim copy of the HEAD 955ae22 method).
  b. OGPOPi05Agent._prefix gathers per-sample prompt rows at every call site (fake model), with the merged
     signatures (task_ids=None last); PPO subsample size under grad accumulation; explicit ppo_batch.
  c. run_pi05_round_rl with stub agent / envs:
       - parallel_envs m=1 vs a VERBATIM copy of Manan's runner (tests/_pi05_runner_manan.py): identical buffer,
         reset seeds, executed actions, agent call sequence (encode / BoN / ODE args and RNG keys, update batches,
         grad-diagnostics batches) and logged metrics; one task and three tasks (task_balance none / bc_ppo).
       - m>1 with 2 tasks: per-task episode counts, task_ids in the buffer, unique reset seeds, waves interleaved
         per task, every inference call padded to m, eval waves and per-task eval keys.
  d. main_pi05: validation before construction; env / prompt / runner wiring for parallel_envs and multi-task.

Run:  JAX_PLATFORMS=cpu uv run pytest tests/test_pi05_multitask.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ogpo.runners.pi05_online_runner as R
import tests._pi05_runner_manan as MAN
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent
from ogpo.networks.pi05.pi05 import Pi05
from ogpo.utils.datasets import ReplayBuffer
from tests._pi05_stubs import (H, Logger, StubAgent, TaskEnv, assert_buffers_equal, assert_call_logs_equal,
                               assert_logs_equal, ep_len, episodes_in_buffer, reg)


# ===========================================================================
# a. sample_sequence
# ===========================================================================

def _old_sample_sequence(self, batch_size, sequence_length, discount, ret_next_act=False, ret_mc=False, filter_fn=None, success_only=None, by_score=False):
    """Verbatim Dataset.sample_sequence at git HEAD 955ae22 (self -> first arg)."""
    B = batch_size
    T = sequence_length
    max_start = self.size - T  # FIXED: was self.size - T - 1
    if max_start < 0:
        raise ValueError(f"Sequence length too long for dataset size. Need at least {T} transitions, but have {self.size}")

    # Candidate start indices [0..max_start]
    starts = np.arange(max_start + 1, dtype=np.int64)

    # Base mask: all starts valid
    mask = np.ones(max_start + 1, dtype=bool)

    # Optional success-based filtering (per-trajectory)
    if success_only is not None:
        succ_mask = self._traj_success_mask[:max_start + 1]
        mask &= (succ_mask if success_only else ~succ_mask)

    # Optional extra filter (e.g., score-based thresholding)
    if filter_fn is not None:
        extra = filter_fn(self._dict, max_start + 1)
        if extra is None or len(extra) != max_start + 1:
            raise ValueError("filter_fn must return a boolean mask of length max_start + 1.")
        mask &= extra

    valid_indices = starts[mask]
    if len(valid_indices) == 0:
        raise ValueError("No transitions found matching filter")

    probs = None  # Default to uniform sampling
    if by_score:
        scores = self._dict.get('score', None)
        if scores is not None:
            w = np.asarray(scores[valid_indices], dtype=np.float64)
            total = w.sum()
            if np.isfinite(total) and total > 0:
                probs = w / total

    idxs = self.numpy_rng.choice(valid_indices, size=B, p=probs)

    offs = np.arange(T, dtype=np.int64)[None, :]          # (1, T)
    seq_idxs = idxs[:, None].astype(np.int64) + offs      # (B, T)

    obs_seq         = self['observations'][seq_idxs]      # (B, T, obs_dim)
    next_obs_seq    = self['next_observations'][seq_idxs] # (B, T, obs_dim)
    actions_seq     = self['actions'][seq_idxs]           # (B, T, act_dim)
    rewards_seq     = self['rewards'][seq_idxs]           # (B, T)
    masks_seq       = self['masks'][seq_idxs]             # (B, T)
    terminals_seq   = self['terminals'][seq_idxs]         # (B, T)

    masks_prefix     = np.minimum.accumulate(masks_seq, axis=1)
    terminals_prefix = np.maximum.accumulate(terminals_seq, axis=1)

    valid = np.ones_like(masks_seq, dtype=np.float32)
    valid[:, 1:] = 1.0 - terminals_prefix[:, :-1]

    rdtype = rewards_seq.dtype
    disc_pows = (discount ** np.arange(T)).astype(rdtype, copy=False)  # (T,)
    rewards_prefix = np.cumsum(rewards_seq * disc_pows[None, :], axis=1)

    first_obs = self['observations'][idxs]  # (B, obs_dim)
    last_obs = next_obs_seq[:, -1, ...]

    if self.p_aug is not None and first_obs.ndim > 3 and self.numpy_rng.random() < self.p_aug:
        tmp = {'observations': first_obs, 'next_observations': last_obs}
        self.augment(tmp, ['observations', 'next_observations'])
        first_obs = tmp['observations']
        last_obs = tmp['next_observations']

    result = dict(
        observations=first_obs.copy(),          # matches your return
        actions=actions_seq,
        masks=masks_prefix,
        rewards=rewards_prefix,
        terminals=terminals_prefix,
        valid=valid,
        next_observations=last_obs,
    )

    if 'images' in self._dict:
        result['images'] = self['images'][idxs]                 # (B, H, W, C)
        result['next_images'] = self['next_images'][seq_idxs[:, -1]]  # (B, H, W, C)
        if self.p_aug is not None and self.numpy_rng.random() < self.p_aug:
            self.augment(result, ['images', 'next_images'])

    if 'full_states' in self._dict:
        result['full_states'] = self['full_states'][idxs]
        result['next_full_states'] = self['next_full_states'][seq_idxs[:, -1]]

    if 'is_success' in self._dict:
        result['is_success'] = self['is_success'][seq_idxs]

    if ret_next_act:
        result['next_actions'] = self['actions'][np.minimum(seq_idxs + 1, self.size - 1)]

    if 'mc_returns' in self._dict:
        result['mc_returns'] = self['mc_returns'][idxs]
    return result


# (task, length, success) per episode; the 2nd/3rd boundaries switch task.
EPISODES = [(1, 4, 1.0), (0, 5, 0.0), (2, 3, 1.0), (0, 4, 0.0)]


def _seq_buffer(with_tasks, seed=5):
    ex = dict(observations=np.zeros(2, np.float32), actions=np.zeros(7, np.float32),
              rewards=np.float32(0), terminals=np.float32(0), masks=np.float32(1),
              next_observations=np.zeros(2, np.float32), images=np.zeros((2, 2, 3), np.uint8),
              next_images=np.zeros((2, 2, 3), np.uint8))
    if with_tasks:
        ex['task_ids'] = np.int32(0)
    rb = ReplayBuffer.create(ex, 100, numpy_rng=np.random.default_rng(seed))
    i = 0
    for task, n, succ in EPISODES:
        for t in range(n):
            tr = dict(observations=np.array([i, task], np.float32), actions=np.full(7, i, np.float32),
                      rewards=np.float32(-1), terminals=np.float32(t == n - 1),
                      masks=np.float32(1 - (t == n - 1) * succ),
                      next_observations=np.array([i + 1, task], np.float32),
                      images=np.full((2, 2, 3), i, np.uint8), next_images=np.full((2, 2, 3), i + 1, np.uint8),
                      is_success=succ)
            if with_tasks:
                tr['task_ids'] = np.int32(task)
            rb.add_transition(tr)
            i += 1
    return rb


def test_sample_sequence_task_ids():
    rb = _seq_buffer(True)
    tid = np.asarray(rb['task_ids'][:rb.size])
    T_ = 3
    b = rb.sample_sequence(256, sequence_length=T_, discount=0.9)
    idx = b['observations'][:, 0].astype(int)
    assert b['task_ids'].dtype == np.int32 and b['task_ids'].shape == (256,)
    np.testing.assert_array_equal(b['task_ids'], tid[idx])  # task of the first frame
    assert 'next_task_ids' not in b  # merged contract (Manan's): the critic bootstraps with task_ids
    # success-only sampling path also carries the field
    s = rb.sample_sequence(64, sequence_length=T_, discount=0.9, success_only=True)
    idx = s['observations'][:, 0].astype(int)
    np.testing.assert_array_equal(s['task_ids'], tid[idx])


def test_sample_sequence_per_task_quotas():
    rb = _seq_buffer(True)
    b = rb.sample_sequence(30, sequence_length=2, discount=0.9, per_task=3)
    counts = np.bincount(b['task_ids'], minlength=3)
    np.testing.assert_array_equal(counts, [10, 10, 10])
    np.testing.assert_array_equal(b['task_ids'], np.asarray(rb['task_ids'])[b['observations'][:, 0].astype(int)])


@pytest.mark.parametrize('kw', [dict(), dict(success_only=True), dict(ret_next_act=True, by_score=True)])
def test_sample_sequence_unchanged(kw):
    # without task_ids: identical keys + arrays to the pre-change method, same RNG stream
    new_rb, old_rb = _seq_buffer(False), _seq_buffer(False)
    for _ in range(3):
        new = new_rb.sample_sequence(32, sequence_length=4, discount=0.9, **kw)
        old = _old_sample_sequence(old_rb, 32, sequence_length=4, discount=0.9, **kw)
        assert new.keys() == old.keys()
        for k in old:
            assert new[k].dtype == old[k].dtype and np.array_equal(new[k], old[k]), k
    # with task_ids: only task_ids is added; everything else still identical
    new_rb, old_rb = _seq_buffer(True), _seq_buffer(True)
    new = new_rb.sample_sequence(32, sequence_length=4, discount=0.9, **kw)
    old = _old_sample_sequence(old_rb, 32, sequence_length=4, discount=0.9, **kw)
    assert new.keys() == old.keys() | {'task_ids'}
    for k in old:
        assert np.array_equal(new[k], old[k]), k


# ===========================================================================
# b. _prefix per-sample prompt rows
# ===========================================================================

class _Stop(Exception):
    pass


class _FakeModel:
    """Records what reaches Pi05.prefix, then aborts the caller."""

    def __init__(self):
        self.seen = []

    def apply(self, variables, *args, method=None):
        assert method is Pi05.prefix, method
        imgs, masks, tokens, token_mask = args
        marker = np.rint((np.asarray(imgs['base_0_rgb'][:, 0, 0, 0]) + 1.0) / 2.0 * 255.0).astype(int)
        self.seen.append((marker, np.asarray(tokens), np.asarray(token_mask)))
        raise _Stop()


N_TASKS, L, BATCH = 3, 6, 6


def _consts():
    tokens = np.stack([np.arange(L) + 100 * (t + 1) for t in range(N_TASKS)]).astype(np.int32)
    token_mask = np.stack([np.arange(L) < (2 + t) for t in range(N_TASKS)])
    return {'tokens': jnp.asarray(tokens), 'token_mask': jnp.asarray(token_mask),
            'action_q01': -jnp.ones(32), 'action_q99': jnp.ones(32),
            'state_q01': -jnp.ones(8), 'state_q99': jnp.ones(8)}


def _pi_agent(**cfg):
    config = dict(batch_size=BATCH, ppo_batch_size=BATCH, group_num_samples=2, best_of_n=4, num_qs=2,
                  q_variance_reduction=False, q_vr_num_samples=4, flow_steps=2, pi05_action_horizon=4,
                  horizon_length=2, pi05_critic_pooling='all')
    config.update(cfg)
    return OGPOPi05Agent(rng=jax.random.PRNGKey(0), actor_params={}, target_actor_params={}, actor_opt_state=None, init_actor_params={},
                         frozen_params={}, critic_network=None, consts=_consts(), config=config,
                         model=_FakeModel(), actor_tx=None)


def _images(offset=0):
    img = np.zeros((BATCH, 2, 2, 2, 3), np.uint8)
    img[:, 0] = (np.arange(BATCH) + offset)[:, None, None, None]
    return jnp.asarray(img)


TASK_IDS = np.array([2, 0, 1, 1, 2, 0], np.int32)


def _run(agent, fn):
    with jax.disable_jit(), pytest.raises(_Stop):
        fn()
    (marker, tokens, token_mask), = agent.model.seen
    return marker, tokens, token_mask


def _check(marker, tokens, token_mask, ids, offset=0):
    c = _consts()
    rows = ids[marker - offset]
    np.testing.assert_array_equal(tokens, np.asarray(c['tokens'])[rows])
    np.testing.assert_array_equal(token_mask, np.asarray(c['token_mask'])[rows])


def _batch():
    return {'images': _images(), 'next_images': _images(offset=50), 'task_ids': jnp.asarray(TASK_IDS),
            'observations': jnp.zeros((BATCH, 2056)),
            'actions': jnp.zeros((BATCH, 2, 7)), 'valid': jnp.ones((BATCH, 2))}


def test_prefix_acting_paths():
    key = jax.random.PRNGKey(1)
    for call in (lambda a: a.encode(_images(), jnp.zeros((BATCH, 8)), jnp.asarray(TASK_IDS)),
                 lambda a: a.sample_actions_BON(_images(), jnp.zeros((BATCH, 2056)), key, jnp.asarray(TASK_IDS)),
                 lambda a: a.sample_actions_ode(_images(), key, jnp.asarray(TASK_IDS))):
        agent = _pi_agent()
        marker, tok, tm = _run(agent, lambda: call(agent))
        np.testing.assert_array_equal(marker, np.arange(BATCH))
        _check(marker, tok, tm, TASK_IDS)


def test_prefix_single_task_default_row0():
    # task_ids=None (single task): every sample gets prompt row 0
    agent = _pi_agent()
    marker, tok, tm = _run(agent, lambda: agent.sample_actions_ode(_images(), jax.random.PRNGKey(1)))
    _check(marker, tok, tm, np.zeros(BATCH, np.int32))


@pytest.mark.parametrize('ppo_bs', [BATCH, 3])
def test_prefix_actor_loss(ppo_bs):
    agent = _pi_agent(ppo_batch_size=ppo_bs)
    marker, tok, tm = _run(agent, lambda: agent.actor_loss(_batch(), None, None, jax.random.PRNGKey(4)))
    assert marker.shape == (ppo_bs,) and len(set(marker.tolist())) == ppo_bs
    if ppo_bs < BATCH:
        assert not np.array_equal(marker, np.arange(ppo_bs))  # a real subsample, not the first rows
    _check(marker, tok, tm, TASK_IDS)


@pytest.mark.parametrize('K', [2, 3])
def test_prefix_actor_loss_accum_subsample(K):
    # grad accumulation: each micro-batch subsamples ppo_batch_size // K PPO states
    agent = _pi_agent(ppo_batch_size=3 if K == 3 else 4, grad_accum_steps=K)
    marker, tok, tm = _run(agent, lambda: agent.actor_loss(_batch(), None, None, jax.random.PRNGKey(4)))
    assert marker.shape == (agent.config['ppo_batch_size'] // K,)
    _check(marker, tok, tm, TASK_IDS)


def test_prefix_actor_loss_ppo_batch():
    # an explicit (task-stratified) ppo_batch replaces the TD-batch subsample; its task_ids pick the prompts
    agent = _pi_agent(ppo_batch_size=3)
    ppo_ids = np.array([1, 1, 0, 2], np.int32)
    ppo = {'images': _images(offset=20)[:4], 'observations': jnp.zeros((4, 2056)), 'task_ids': jnp.asarray(ppo_ids)}
    marker, tok, tm = _run(agent, lambda: agent.actor_loss(_batch(), None, None, jax.random.PRNGKey(4), ppo))
    np.testing.assert_array_equal(marker, np.arange(4) + 20)
    _check(marker, tok, tm, ppo_ids, offset=20)


@pytest.mark.parametrize('vr', [False, True])
def test_prefix_critic_loss_uses_task_ids(vr):
    # Merged (Manan's) rule: next_images are encoded with the chunk's task_ids. Rows whose chunk crosses into
    # another episode (the only place the two differ) have valid[:, -1] = 0, so they carry no TD loss.
    agent = _pi_agent(q_variance_reduction=vr)
    marker, tok, tm = _run(agent, lambda: agent.critic_loss(_batch(), None, jax.random.PRNGKey(2)))
    np.testing.assert_array_equal(marker, np.arange(BATCH) + 50)  # next_images
    _check(marker, tok, tm, TASK_IDS, offset=50)


def test_prefix_bc_loss():
    agent = _pi_agent()
    bc = {'images': _images(), 'actions': jnp.zeros((BATCH, 4, 7)), 'valid': jnp.ones((BATCH, 4)),
          'task_ids': jnp.asarray(TASK_IDS)}
    marker, tok, tm = _run(agent, lambda: agent.bc_loss(bc, None, jax.random.PRNGKey(3)))
    _check(marker, tok, tm, TASK_IDS)


def test_create_stacks_prompts_in_order(monkeypatch):
    import ogpo.agents.ogpo_pi05 as P
    monkeypatch.setattr(P, 'Pi05', lambda **kw: None)
    monkeypatch.setattr(P, 'load_pi05_params', lambda path: {})
    monkeypatch.setattr(P, 'split_params', lambda p: ({}, {'w': np.zeros(2)}))
    ns = {'actions': {'q01': np.zeros(32), 'q99': np.ones(32)}, 'state': {'q01': np.zeros(32), 'q99': np.ones(32)}}
    monkeypatch.setattr(P.T, 'load_norm_stats', lambda path, asset_id: ns)

    class _Tok:
        def __init__(self, *a):
            pass

        def tokenize(self, prompt):
            n = len(prompt)
            return np.full(5, n, np.int32), np.arange(5) < n

    monkeypatch.setattr(P.T, 'PromptTokenizer', _Tok)

    class _Seen(Exception):
        pass

    def _value(**kw):  # stop once consts are built
        raise _Seen()
    monkeypatch.setattr(P, 'Value', _value)
    cfg = dict(pi05_checkpoint='/x', pi05_robot='libero', pi05_tokenizer='x', pi05_action_horizon=4, horizon_length=2,
               pi05_clip_grad_norm=1.0, pi05_lr=1e-5, pi05_weight_decay=0.0, q_variance_reduction=False,
               q_vr_num_samples=4, q_vr_reduction='mean', pi05_critic_pooling='all',
               value_hidden_dims=(4,), layer_norm=True, num_qs=2, q_min=-1.0, q_max=0.0,
               pi05_critic_arch='mlp', pi05_bronet_hidden_dim=1024, pi05_bronet_depth=2,
               subsample_bon=True, pi05_bon_sampler='sde', pi05_bon_agg='subsample', use_value_fn=False)
    built = {}
    orig_stack = np.stack

    def spy_stack(x, *a, **k):
        out = orig_stack(x, *a, **k)
        built.setdefault('stacks', []).append(out)
        return out
    monkeypatch.setattr(P.np, 'stack', spy_stack)
    with pytest.raises(_Seen):
        OGPOPi05Agent.create(0, ['a', 'abc', 'ab'], cfg)
    tokens, mask = built['stacks'][:2]
    np.testing.assert_array_equal(tokens[:, 0], [1, 3, 2])
    np.testing.assert_array_equal(mask.sum(1), [1, 3, 2])


# ===========================================================================
# c. Round runner
# ===========================================================================

@pytest.fixture(autouse=True)
def tiny_images(monkeypatch):
    monkeypatch.setattr(R.T, 'resize_images', lambda x: x)  # same module object as MAN.T


KW = dict(batch_size=2, success_buffer_batch_size=2, horizon_length=3, bc_horizon=4, discount=0.99,
          buffer_size=2000, seed=3)
BASE = KW['seed'] + 10000


def _round(fn, env_spec, eval_spec, *, m=1, num_rounds=3, rollouts=2, updates=4, eval_interval=3, eval_episodes=2,
           log_interval=2, diag=True, ppo_batch_size=2, **extra):
    """env_spec / eval_spec: None (one task, one env) or a list of (name, prompt row) per task."""
    tr, ev, log = reg(), reg(), []
    agent, logger = StubAgent(log), Logger()

    def build(spec, r):
        if spec is None:  # one task: a bare env, or (as main_pi05 builds it) one (name, [m envs], 0) slot
            return TaskEnv(0, r) if m == 1 else [('task0', [TaskEnv(0, r) for _ in range(m)], 0)]
        out = []
        for name, row in spec:
            e = TaskEnv(row, r) if m == 1 else [TaskEnv(row, r) for _ in range(m)]
            out.append((name, e, row))
        return out
    _, rb = fn(agent, build(env_spec, tr), build(eval_spec, ev),
               {'grad_diagnostics': diag, 'ppo_batch_size': ppo_batch_size}, logger,
               num_rounds=num_rounds, rollouts_per_round=rollouts, updates_per_round=updates,
               eval_interval=eval_interval, eval_episodes=eval_episodes, log_interval=log_interval, **KW, **extra)
    return SimpleNamespace(rb=rb, log=log, logs=logger.logs, tr=tr, ev=ev)


def _assert_runs_equal(a, b):
    assert_buffers_equal(a.rb, b.rb)
    assert a.tr['resets'] == b.tr['resets'] and a.ev['resets'] == b.ev['resets']
    for x, y in ((a.tr, b.tr), (a.ev, b.ev)):
        assert len(x['actions']) == len(y['actions'])
        for p, q in zip(x['actions'], y['actions']):
            assert p[:3] == q[:3] and np.array_equal(p[3], q[3])
    assert_call_logs_equal(a.log, b.log)
    assert_logs_equal(a.logs, b.logs)


TASKS3 = [('a', 0), ('b', 1), ('c', 2)]


@pytest.mark.parametrize('case', ['single', 'multi', 'multi_bc_ppo'])
def test_round_m1_matches_manan(case):
    if case == 'single':
        spec, eval_spec, extra = None, None, {}
    else:
        spec = TASKS3
        eval_spec = TASKS3 + [('held_out', 3)]  # eval-only task on a row past the training rows
        extra = {'task_balance': 'bc_ppo' if case == 'multi_bc_ppo' else 'none'}
    old = _round(MAN.run_pi05_round_rl, spec, eval_spec, **extra)
    new = _round(R.run_pi05_round_rl, spec, eval_spec, actor_update_interval=1, **extra)
    _assert_runs_equal(old, new)
    kinds = [c[0] for c in new.log]
    assert kinds.count('update') == 3 * 4 and 'critic_update' not in kinds
    assert 'diag' in kinds and 'ode' in kinds and 'bon' in kinds
    if case == 'multi_bc_ppo':
        assert all(c[5] is not None for c in new.log if c[0] == 'update')  # stratified ppo_batch reached update
    # the reference has teeth: a different interleaving / seed would show up
    assert len({s for _, s in new.tr['resets']}) == len(new.tr['resets']) - 1  # only the schema reset repeats


def test_round_m1_default_interval_matches_manan():
    # actor_update_interval left at its default
    _assert_runs_equal(_round(MAN.run_pi05_round_rl, None, None), _round(R.run_pi05_round_rl, None, None))


@pytest.mark.parametrize('m,rollouts', [(3, 5), (2, 4), (4, 2)])
def test_round_parallel_two_tasks(m, rollouts):
    NR, EV = 2, 5
    spec = [('t0', 0), ('t1', 1)]
    run = _round(R.run_pi05_round_rl, spec, spec, m=m, num_rounds=NR, rollouts=rollouts, updates=3,
                 eval_interval=3, eval_episodes=EV, actor_update_interval=2)

    # every train env slot built m times per task
    assert [e.task for e in run.tr['built']] == [0] * m + [1] * m
    # buffer: episodes contiguous, right task / seed / step, actions planned from the right obs
    eps = episodes_in_buffer(run.rb)
    assert len(eps) == NR * 2 * rollouts
    assert sum(1 for t, _ in eps if t == 0) == sum(1 for t, _ in eps if t == 1) == NR * rollouts
    # every executed train action matches its obs (no stale feats across slots / waves)
    for task, s, t, a in run.tr['actions']:
        assert (a[0], a[1], a[2]) == (task, s, t)
    # reset seeds: the schema reset, then base + k for k = 0.. each used exactly once
    seeds = [s for _, s in run.tr['resets']]
    assert seeds[0] == BASE and sorted(seeds[1:]) == list(range(BASE, BASE + NR * 2 * rollouts))
    assert {s for _, s in eps} == set(seeds[1:])
    # waves: tasks interleaved per wave of min(m, remaining) episodes
    waves = []
    for r in range(NR):
        for w in range(0, rollouts, m):
            for task in (0, 1):
                waves += [task] * min(m, rollouts - w)
    resets_by_seed = sorted(run.tr['resets'][1:], key=lambda x: x[1])
    assert [t for t, _ in resets_by_seed] == waves
    assert sorted(t for t, _ in eps) == sorted(waves)
    # env logs per round: cumulative episodes, per-task success keys
    env_logs = [d for p, _, d in run.logs if p == 'env']
    assert [d['episodes'] for d in env_logs] == [2 * rollouts * (r + 1) for r in range(NR)]
    assert all({'success/t0', 'success/t1'} <= set(d) for d in env_logs)
    assert [d['env_steps'] for d in env_logs][-1] == run.rb.size

    # inference: every encode / policy call padded to m, one task per call, rows of that task's prompt
    for c in run.log:
        if c[0] == 'encode' and len(c[1]) != 1:  # the schema encode is a single obs
            assert len(c[1]) == m and len(c[3]) == m and len(set(c[3].tolist())) == 1
        if c[0] in ('bon', 'ode'):
            ids = c[-1]
            assert len(c[1]) == m and len(ids) == m and len(set(ids.tolist())) == 1
    assert sum(1 for c in run.log if c[0] == 'encode' and len(c[1]) == 1) == 1

    # eval: EV episodes per task and mode, in waves of m; seeds step + i; per-task keys
    evals = [(p, s, d) for p, s, d in run.logs if p in ('eval', 'eval_sde')]
    ev_steps = sorted({s for p, s, _ in evals})
    assert ev_steps == [0, 3, 6]
    # chronological: per eval, ODE then BoN, each task in turn, seeds step + i in wave order
    assert run.ev['resets'] == [(task, step + i) for step in ev_steps for _mode in range(2) for task in (0, 1)
                                for i in range(EV)]
    assert [e.task for e in run.ev['built']] == [0] * m + [1] * m  # persistent eval envs, m per task
    for p, _, d in evals:
        assert set(d) == {'success', 'return', 'length', 'success/t0', 'success/t1'}
        assert d['success'] == pytest.approx((d['success/t0'] + d['success/t1']) / 2)
    for (p, s, d) in evals:
        for name in ('t0', 't1'):
            assert d[f'success/{name}'] == pytest.approx(np.mean([(s + i) % 2 == 0 for i in range(EV)]))


def test_round_parallel_eval_returns_lengths():
    # eval aggregates over episodes of a wave exactly as the sequential loop would
    m, EV = 3, 4
    run_p = _round(R.run_pi05_round_rl, None, None, m=m, num_rounds=1, rollouts=3, updates=2, eval_interval=2,
                   eval_episodes=EV)
    for p, s, d in run_p.logs:
        if p in ('eval', 'eval_sde'):
            seeds = [s + i for i in range(EV)]
            succ = [x % 2 == 0 for x in seeds]
            assert d['success'] == pytest.approx(np.mean(succ))
            assert d['return'] == pytest.approx(np.mean([-ep_len(x) for x in seeds]))
            lens = [ep_len(x) for x, ok in zip(seeds, succ) if ok]
            assert d['length'] == pytest.approx(np.mean(lens) if lens else 0.0)


def test_round_rejects_unequal_parallel_envs():
    tr = reg()
    env = [('a', [TaskEnv(0, tr), TaskEnv(0, tr)], 0), ('b', [TaskEnv(1, tr)], 1)]
    with pytest.raises(ValueError, match='same number of parallel envs'):
        R.run_pi05_round_rl(StubAgent([]), env, env, {'ppo_batch_size': 2}, Logger(), num_rounds=1,
                            rollouts_per_round=1, updates_per_round=1, eval_interval=0, eval_episodes=1,
                            log_interval=1, **KW)


# ===========================================================================
# d. main_pi05
# ===========================================================================

GOOD = 'libero_pi05-libero_90-1'
GOOD2 = 'libero_pi05-libero_90-2'


class _Built(Exception):
    pass


def _boom(*a, **k):
    raise _Built()


def _patch_main(monkeypatch, M, *, built=None, devices=None):
    """Stub env / agent / mesh / logging. built: dict that collects what main constructs."""
    built = {} if built is None else built

    def make_env(name, seed, post_success_steps):
        e = SimpleNamespace(name=name, seed=seed, prompt=f"prompt:{name}")
        built.setdefault('envs', []).append(e)
        return e

    def create(cls, seed, prompt, config):
        built['prompt'], built['agent_config'] = prompt, config
        return SimpleNamespace(tag='agent')

    def capture(name):
        def run(agent, env, eval_env, config, logger, **kw):
            built['runner'], built['env'], built['eval_env'], built['kw'] = name, env, eval_env, kw
            raise _Built()
        return run
    monkeypatch.setattr(M, 'make_env', make_env)
    monkeypatch.setattr(M.OGPOPi05Agent, 'create', classmethod(create))
    monkeypatch.setattr(M, 'create_mesh', lambda mode: None if devices is None else SimpleNamespace(devices=np.zeros(devices)))
    monkeypatch.setattr(M, 'shard_pi05_agent', lambda agent, mesh, zero_actor_state=False: agent)
    monkeypatch.setattr(M, 'setup_experiment_logging', lambda **kw: (None, SimpleNamespace(close=lambda: None), None))
    monkeypatch.setattr(M, 'run_pi05_round_rl', capture('rounds'))
    monkeypatch.setattr(M, 'run_pi05_online_rl', capture('online'))
    return built


def _main(monkeypatch, *extra):
    import ogpo.main_pi05 as M
    # pi05_action_horizon is a script-only key (the recipes pass it); main reads it for bc_horizon
    monkeypatch.setattr(sys, 'argv', ['main_pi05.py', '--algo=ogpo', '--experiment.log=false',
                                      '--agent.pi05_action_horizon=10', *extra])
    return M


@pytest.mark.parametrize('extra,match', [
    ([f'--env.name={GOOD}', '--env.parallel_envs=0', '--training.collection_mode=rounds'], 'parallel_envs'),
    ([f'--env.name={GOOD}', '--env.parallel_envs=2'], 'collection_mode=rounds'),        # default schedule: online
    ([f'--env.name={GOOD}', '--env.parallel_envs=2', '--training.collection_mode=online'], 'collection_mode=rounds'),
    ([f'--env.name={GOOD},{GOOD2}'], 'rounds'),                                          # multi-task needs rounds
    ([f'--env.name={GOOD}', f'--env.eval_name={GOOD2}'], 'rounds'),                     # eval-only task = multi-task
    ([f'--env.name={GOOD}', '--training.actor_update_interval=0'], 'actor_update_interval'),
    ([f'--env.name={GOOD}', '--agent.grad_accum_steps=0'], 'grad_accum_steps'),
])
def test_main_pi05_validation(monkeypatch, extra, match):
    M = _main(monkeypatch, *extra)
    built = _patch_main(monkeypatch, M)
    with pytest.raises(ValueError, match=match):
        M.main()
    if 'actor_update_interval' in match or 'parallel_envs' in match:
        assert 'envs' not in built  # fires before any env is built


def test_main_pi05_rejects_non_pi05_env(monkeypatch):
    M = _main(monkeypatch, f'--env.name={GOOD},libero-libero_90-2', '--training.collection_mode=rounds')
    built = _patch_main(monkeypatch, M)
    with pytest.raises(AssertionError, match='libero-libero_90-2'):  # Manan's assert, kept
        M.main()
    assert 'envs' not in built


@pytest.mark.parametrize('extra,devices,match', [
    (['--agent.grad_accum_steps=3'], None, 'agent.batch_size=256 must be divisible by 1 devices x 3'),
    (['--agent.grad_accum_steps=4', '--agent.ppo_batch_size=6'], None, 'agent.ppo_batch_size=6'),
    (['--agent.grad_accum_steps=4', '--success_buffer.bc_batch_size=8'], 4, 'success_buffer.bc_batch_size=8'),
    (['--agent.grad_accum_steps=1', '--success_buffer.batch_size=6'], 4, 'success_buffer.batch_size=6'),
])
def test_main_pi05_batch_divisibility(monkeypatch, extra, devices, match):
    M = _main(monkeypatch, f'--env.name={GOOD}', '--agent.batch_size=256', '--agent.ppo_batch_size=64',
              '--success_buffer.batch_size=64', *extra)
    _patch_main(monkeypatch, M, devices=devices)
    with pytest.raises(ValueError, match=match):
        M.main()


def test_main_pi05_single_task_online_wiring(monkeypatch):
    M = _main(monkeypatch, f'--env.name={GOOD}', '--training.actor_update_interval=4')
    built = _patch_main(monkeypatch, M)
    with pytest.raises(_Built):
        M.main()
    assert built['runner'] == 'online'
    assert built['prompt'] == f'prompt:{GOOD}'                        # one prompt string, not a list
    assert built['env'].name == GOOD and built['env'].seed == 0      # one env, not a list
    assert built['eval_env'].name == GOOD and built['eval_env'].seed == 42
    assert built['kw']['actor_update_interval'] == 4
    cfg = built['agent_config']
    assert cfg['donate_agent'] is True and cfg['grad_accum_steps'] == 1


def test_main_pi05_parallel_multitask_wiring(monkeypatch):
    m = 3
    M = _main(monkeypatch, f'--env.name={GOOD},{GOOD2}', f'--env.eval_name={GOOD2},libero_pi05-libero_90-7',
              f'--env.parallel_envs={m}', '--training.collection_mode=rounds', '--training.num_rounds=2',
              '--training.rollouts_per_round=4', '--training.updates_per_round=5', '--training.task_balance=bc',
              '--agent.donate_agent=false', '--agent.grad_accum_steps=2')
    built = _patch_main(monkeypatch, M)
    with pytest.raises(_Built):
        M.main()
    assert built['runner'] == 'rounds'
    assert built['prompt'] == [f'prompt:{GOOD}', f'prompt:{GOOD2}', 'prompt:libero_pi05-libero_90-7']
    # train: (name, [m envs], row); m envs per task share the make_env seed (resets are seeded per episode)
    assert [(n, row) for n, _, row in built['env']] == [(GOOD, 0), (GOOD2, 1)]
    for (n, envs, row) in built['env']:
        assert isinstance(envs, list) and len(envs) == m and {e.name for e in envs} == {n}
        assert {e.seed for e in envs} == {row}
    # eval: same prompt rows as training, eval-only task on the next row; seeds 42 + row
    assert [(n, row) for n, _, row in built['eval_env']] == [(GOOD2, 1), ('libero_pi05-libero_90-7', 2)]
    for (n, envs, row) in built['eval_env']:
        assert len(envs) == m and {e.seed for e in envs} == {42 + row}
    kw = built['kw']
    assert (kw['num_rounds'], kw['rollouts_per_round'], kw['updates_per_round']) == (2, 4, 5)
    assert kw['task_balance'] == 'bc' and kw['actor_update_interval'] == 1
    cfg = built['agent_config']
    assert cfg['donate_agent'] is False and cfg['grad_accum_steps'] == 2


def test_main_pi05_single_task_parallel_uses_list_slots(monkeypatch):
    M = _main(monkeypatch, f'--env.name={GOOD}', '--env.parallel_envs=2', '--training.collection_mode=rounds',
              '--training.num_rounds=1', '--training.rollouts_per_round=2', '--training.updates_per_round=1')
    built = _patch_main(monkeypatch, M)
    with pytest.raises(_Built):
        M.main()
    assert built['prompt'] == [f'prompt:{GOOD}']
    assert [(n, len(e), row) for n, e, row in built['env']] == [(GOOD, 2, 0)]
    assert [(n, len(e), row) for n, e, row in built['eval_env']] == [(GOOD, 2, 0)]
