"""
Verifier tests for the pi05 BoN-only change (docs/changes/2026-10-09-pi05-bon-only/).

Agent (OGPOPi05Agent subclass with a fake `_prefix` / `_actor_fn`; real `_sample_sde`, `_sample_ode`, critic,
`_critic_actions`, `_to_env_actions`, and the real jitted acting methods):
  - default config (pi05_bon_sampler=sde, pi05_bon_agg=subsample) == VERBATIM 021fb48 sample_actions_BON
    (subsample_bon=true), bit-exact; pi05_bon_agg=min == 021fb48 with subsample_bon=false.
  - sample_actions_ode == VERBATIM 021fb48 sample_actions_ode, bit-exact.
  - pi05_bon_sampler=ode: candidates are a full (B*N, 10*32) N(0, I) draw from action_rng, Euler-integrated with
    TARGET actor params, never clipped, scored by the target critic with a mean over heads, argmax.
  - sample_actions_plain: one unclipped ODE sample from the target actor.
Runners (stub agent / envs, tests/_pi05_stubs.py):
  - freeze_actor: only critic_update, no grad diagnostics, evals unchanged; per-step runner no NameError.
  - first_round_policy='ode': round 0 collects with 'plain', every later round with 'bon', same RNG stream.
main_pi05: validation + forwarding of first_round_policy / freeze_actor. create(): the new ValueErrors.

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_pi05_bon_only.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import ogpo.agents.ogpo_pi05 as P
import ogpo.runners.pi05_online_runner as R
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState
from ogpo.agents.modules.pg_helper import sample_flow_actions_ode
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T
from ogpo.networks import Value
from tests._pi05_stubs import Logger, StubAgent, TaskEnv, reg

B, N, NUM_QS, H_PI, HL = 3, 8, 4, 10, 5
FLAT = H_PI * T.MODEL_ACTION_DIM          # 320: full 10 x 32 model chunk
L, S, K, HD = 2, 3, 1, 2
W_ONLINE, W_TARGET = 0.3, 1.7             # online / target actor "params" differ so a mixup is visible


class _FakePi05(OGPOPi05Agent):
    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        b_idx = images[:, 0, 0, 0, 0].astype(jnp.float32)
        n = images.shape[0]
        k = jnp.broadcast_to(b_idx[None, :, None, None, None], (L, n, S, K, HD))
        return None, (k, k + 100.0), jnp.ones((n, S), bool)

    def _actor_fn(self, params, kv_cache, prefix_mask):
        k, v = kv_cache
        assert k.shape[1] == prefix_mask.shape[0]
        b = k[0, :, 0, 0, 0]

        def actor_fn(obs, x, t, params_=None, is_encoded=True):
            # pushes samples well outside [-1, 1]; row-dependent through the (tiled) KV cache
            return params['w'] * x + 0.5 * b[:, None] + 2.0 * t

        return actor_fn


def _config(**kw):
    c = dict(pi05_action_horizon=H_PI, horizon_length=HL, num_qs=NUM_QS, best_of_n=N,
             flow_steps=10, use_constant_noise=True, constant_noise_std=0.2, min_noise_std=0.08,
             clip_intermediate_actions=True, denoised_clip_value=1.0, randn_clip_value=3.0, act_min=-1.0, act_max=1.0,
             normalize_denoising_horizon=True, normalize_act_space_dimension=True,
             subsample_bon=True, pi05_bon_sampler='sde', pi05_bon_agg='subsample')
    c.update(kw)
    return c


def _critic_state(seed=0):
    critic_def = Value(
        hidden_dims=(16, 16), layer_norm=True, num_ensembles=NUM_QS, critic_loss_type='mse',
        q_min=-100.0, q_max=0.0, obs_two_tier=True, two_tier_img_dim=PREFIX_DIM,
        two_tier_proprio_dim=STATE_DIM, two_tier_fused_dim=0)
    net = ModuleDict({'critic': critic_def, 'target_critic': copy.deepcopy(critic_def)})
    ex = dict(observations=jnp.zeros((1, PREFIX_DIM + STATE_DIM)), actions=jnp.zeros((1, HL * T.ENV_ACTION_DIM)))
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    params = net.init(k1, critic=ex, target_critic=ex)['params']
    params['modules_target_critic'] = net.init(k2, critic=ex, target_critic=ex)['params']['modules_critic']
    return TrainState.create(net, params, optax.adam(1e-3))


def _agent(**kw):
    dim_mask = np.zeros((H_PI, T.MODEL_ACTION_DIM), np.float32)
    dim_mask[:HL, :T.ENV_ACTION_DIM] = 1.0
    consts = {'tokens': jnp.zeros((1, 4), jnp.int32), 'dim_mask': jnp.asarray(dim_mask.reshape(-1)),
              'action_q01': -jnp.ones(T.MODEL_ACTION_DIM) * 0.8, 'action_q99': jnp.ones(T.MODEL_ACTION_DIM) * 1.3}
    return _FakePi05(rng=jax.random.PRNGKey(0), actor_params={'w': jnp.float32(W_ONLINE)},
                     target_actor_params={'w': jnp.float32(W_TARGET)}, actor_opt_state=None, init_actor_params={},
                     frozen_params={}, critic_network=_critic_state(), consts=consts, config=_config(**kw),
                     model=None, actor_tx=None)


def _inputs(seed=0):
    img = np.zeros((B, 2, 2, 2, 3), np.float32)
    img[:, 0] = np.arange(B)[:, None, None, None]
    obs = jax.random.normal(jax.random.PRNGKey(50 + seed), (B, PREFIX_DIM + STATE_DIM))
    return jnp.asarray(img), obs, jax.random.PRNGKey(7 + seed)


# --- verbatim copies of ogpo/agents/ogpo_pi05.py @ 021fb48 -------------------------------------------------------

def _head_sample_actions_BON(self, images, critic_obs, rng, task_ids=None):
    """Verbatim OGPOPi05Agent.sample_actions_BON at 021fb48."""
    B, N = images.shape[0], self.config['best_of_n']
    action_rng, sub_rng = jax.random.split(rng)
    _, kv, pmask = self._prefix(images, task_ids)
    # candidate index b * N + n, matching jnp.repeat in the reference
    kv_n = jax.tree.map(lambda x: jnp.repeat(x, N, axis=1), kv)
    actions, _, _, _ = self._sample_sde(self.target_actor_params, kv_n, jnp.repeat(pmask, N, axis=0), action_rng)
    q_values = self.critic_network.select('target_critic')(
        jnp.repeat(critic_obs, N, axis=0), self._critic_actions(actions), is_encoded=True,
    ).reshape(self.config['num_qs'], B, N)
    if self.config['subsample_bon']:
        idxs = jax.random.randint(sub_rng, (2, B, N), 0, self.config['num_qs'])
        min_qs = jnp.take_along_axis(q_values, idxs, axis=0).min(axis=0)
    else:
        min_qs = q_values.min(axis=0)
    best = jnp.argmax(min_qs, axis=1)
    best_actions = jax.vmap(lambda a, i: a[i])(actions.reshape(B, N, -1), best)
    return self._to_env_actions(best_actions)


def _head_sample_actions_ode(self, images, rng, task_ids=None):
    """Verbatim OGPOPi05Agent.sample_actions_ode at 021fb48."""
    _, kv, pmask = self._prefix(images, task_ids)
    actions = sample_flow_actions_ode(
        actor_fn=self._actor_fn(self.actor_params, kv, pmask), observations=jnp.zeros((images.shape[0], 1)),
        rng=rng, flow_steps=self.config['flow_steps'], act_dim=self._flat_dim,
        act_min=self.config['act_min'], act_max=self.config['act_max'],
        clip_intermediate=False, clip_value=self.config['denoised_clip_value'], is_encoded=True)
    return self._to_env_actions(actions)


_head_bon_jit = jax.jit(_head_sample_actions_BON)
_head_ode_jit = jax.jit(_head_sample_actions_ode)


# --- default paths: bit-identity vs 021fb48 ----------------------------------------------------------------------

@pytest.mark.parametrize('seed', [0, 1, 2])
def test_bon_default_bit_identical_to_head(seed):
    agent = _agent()  # pi05_bon_sampler=sde, pi05_bon_agg=subsample, subsample_bon=true (YAML defaults)
    images, obs, rng = _inputs(seed)
    new = agent.sample_actions_BON(images, obs, rng)
    old = _head_bon_jit(agent, images, obs, rng)
    assert new.shape == (B, HL, T.ENV_ACTION_DIM)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(old))
    # also both eager (jit vs eager differ by XLA-fusion ULPs, so compare like with like)
    with jax.disable_jit():
        np.testing.assert_array_equal(np.asarray(agent.sample_actions_BON(images, obs, rng)),
                                      np.asarray(_head_sample_actions_BON(agent, images, obs, rng)))


@pytest.mark.parametrize('seed', [0, 1])
def test_bon_min_bit_identical_to_head_no_subsample(seed):
    images, obs, rng = _inputs(seed)
    new = _agent(pi05_bon_agg='min').sample_actions_BON(images, obs, rng)
    old = _head_bon_jit(_agent(subsample_bon=False), images, obs, rng)
    np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


def test_bon_head_rules_differ_so_tests_have_teeth():
    # Over a few seeds, subsample and min must pick a different chunk at least once, and so must mean.
    diff_min = diff_mean = False
    for seed in range(6):
        images, obs, rng = _inputs(seed)
        sub = np.asarray(_agent().sample_actions_BON(images, obs, rng))
        diff_min |= not np.array_equal(sub, np.asarray(_agent(pi05_bon_agg='min').sample_actions_BON(images, obs, rng)))
        diff_mean |= not np.array_equal(sub, np.asarray(_agent(pi05_bon_agg='mean').sample_actions_BON(images, obs, rng)))
    assert diff_min and diff_mean


@pytest.mark.parametrize('seed', [0, 1])
def test_sample_actions_ode_bit_identical_to_head(seed):
    agent = _agent()
    images, _, rng = _inputs(seed)
    np.testing.assert_array_equal(np.asarray(agent.sample_actions_ode(images, rng)),
                                  np.asarray(_head_ode_jit(agent, images, rng)))


# --- new ODE paths ----------------------------------------------------------------------------------------------

def _euler(x, b_rows, w, steps=10):
    dt = 1.0 / steps
    for i in range(steps):
        t = jnp.full((x.shape[0], 1), i / steps)
        x = x + (w * x + 0.5 * b_rows[:, None] + 2.0 * t) * dt
    return x


def _expected_ode_bon(agent, images, obs, rng, w, agg='mean', clip=False):
    action_rng, _ = jax.random.split(rng)
    x0 = jax.random.normal(action_rng, (B * N, FLAT))                 # full 10 x 32 chunk noise, row b * N + n
    b_rows = jnp.repeat(jnp.arange(B, dtype=jnp.float32), N)
    x = _euler(x0, b_rows, w)
    if clip:
        x = jnp.clip(x, -1.0, 1.0)
    q = agent.critic_network.select('target_critic')(
        jnp.repeat(obs, N, axis=0), agent._critic_actions(x), is_encoded=True).reshape(NUM_QS, B, N)
    best = np.asarray((q.mean(0) if agg == 'mean' else q.min(0)).argmax(1))
    xb = np.asarray(x).reshape(B, N, -1)[np.arange(B), best]
    return np.asarray(agent._to_env_actions(jnp.asarray(xb))), xb


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_bon_ode_mean_matches_reference(seed):
    agent = _agent(pi05_bon_sampler='ode', pi05_bon_agg='mean')
    images, obs, rng = _inputs(seed)
    got = np.asarray(agent.sample_actions_BON(images, obs, rng))
    want, xb = _expected_ode_bon(agent, images, obs, rng, W_TARGET)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    assert np.abs(xb).max() > 1.0, "reference must leave [-1, 1] for the no-clip check to mean anything"
    # teeth: online params, clipped candidates, or min-agg each give a different answer (on some seed)
    if seed == 0:
        assert not np.allclose(got, _expected_ode_bon(agent, images, obs, rng, W_ONLINE)[0], atol=1e-3)
        assert not np.allclose(got, _expected_ode_bon(agent, images, obs, rng, W_TARGET, clip=True)[0], atol=1e-3)


def test_bon_ode_clip_not_applied_anywhere():
    # Executed chunk in model space = inverse of _to_env_actions; it must contain values beyond +-1.
    agent = _agent(pi05_bon_sampler='ode', pi05_bon_agg='mean')
    images, obs, rng = _inputs(0)
    env = np.asarray(agent.sample_actions_BON(images, obs, rng))
    q01, q99 = -0.8, 1.3
    model = (env - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    assert np.abs(model).max() > 1.0


def test_sample_actions_plain_is_unclipped_target_ode():
    agent = _agent()
    images, _, rng = _inputs(0)
    got = np.asarray(agent.sample_actions_plain(images, rng))
    x = _euler(jax.random.normal(rng, (B, FLAT)), jnp.arange(B, dtype=jnp.float32), W_TARGET)
    assert float(jnp.abs(x).max()) > 1.0
    np.testing.assert_allclose(got, np.asarray(agent._to_env_actions(x)), rtol=1e-5, atol=1e-5)
    # sample_actions_ode stays the clipped online-actor ODE
    x_ode = jnp.clip(_euler(jax.random.normal(rng, (B, FLAT)), jnp.arange(B, dtype=jnp.float32), W_ONLINE), -1, 1)
    np.testing.assert_allclose(np.asarray(agent.sample_actions_ode(images, rng)),
                               np.asarray(agent._to_env_actions(x_ode)), rtol=1e-5, atol=1e-5)


# --- create() validation ----------------------------------------------------------------------------------------

class _Stop(Exception):
    pass


@pytest.fixture
def fake_ckpt(monkeypatch):
    monkeypatch.setattr(P, 'Pi05', lambda **kw: None)
    monkeypatch.setattr(P, 'load_pi05_params', lambda path: {})
    monkeypatch.setattr(P, 'split_params', lambda p: ({}, {'w': np.zeros(2)}))
    ns = {'actions': {'q01': np.zeros(32), 'q99': np.ones(32)}, 'state': {'q01': np.zeros(32), 'q99': np.ones(32)}}
    monkeypatch.setattr(P.T, 'load_norm_stats', lambda path, asset_id: ns)

    class _Tok:
        def __init__(self, *a):
            pass

        def tokenize(self, prompt):
            return np.zeros(4, np.int32), np.ones(4, bool)

    monkeypatch.setattr(P.T, 'PromptTokenizer', _Tok)


class _StopCfg(dict):
    def __getitem__(self, k):
        if k == 'value_hidden_dims':
            raise _Stop()
        return super().__getitem__(k)


def _create_cfg(**kw):
    c = dict(pi05_checkpoint='/nonexistent', pi05_robot='libero', pi05_tokenizer='x', pi05_action_horizon=10, horizon_length=5,
             pi05_clip_grad_norm=1.0, pi05_lr=1e-5, pi05_weight_decay=0.0, q_variance_reduction=False,
             q_vr_num_samples=16, q_vr_reduction='mean', pi05_critic_pooling='all', pi05_critic_arch='mlp',
             subsample_bon=True, pi05_bon_sampler='sde', pi05_bon_agg='subsample', use_value_fn=False)
    c.update(kw)
    return _StopCfg(c)


@pytest.mark.parametrize('kw,match', [
    (dict(pi05_bon_sampler='SDE'), 'pi05_bon_sampler=.*sde \\| ode'),
    (dict(pi05_bon_sampler='euler'), 'pi05_bon_sampler'),
    (dict(pi05_bon_agg='max'), 'pi05_bon_agg=.*min \\| subsample \\| mean'),
    (dict(subsample_bon=False), 'subsample_bon=false'),
])
def test_create_rejects(fake_ckpt, kw, match):
    with pytest.raises(ValueError, match=match):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(**kw))


@pytest.mark.parametrize('kw', [dict(), dict(pi05_bon_sampler='ode', pi05_bon_agg='mean'),
                                dict(subsample_bon=False, pi05_bon_agg='min'),
                                dict(subsample_bon=False, pi05_bon_agg='mean')])
def test_create_accepts(fake_ckpt, kw):
    with pytest.raises(_Stop):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(**kw))


# --- runners ----------------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tiny_images(monkeypatch):
    monkeypatch.setattr(R.T, 'resize_images', lambda x: x)


class PlainStub(StubAgent):
    def _next(self):
        new = PlainStub(self.log, self.strict, self.gen + 1)
        if self.strict:
            self.dead = True
        return new

    def sample_actions_plain(self, imgs, key, task_ids=None, critic_obs=None):
        self._use('sample_actions_plain')
        # key at index 3, as in the 'bon' entry, so the round-0 keys can be compared
        self.log.append(('plain', np.asarray(imgs)[:, 0, 0, 0, 0].copy(), None, self._key(key),
                         None if task_ids is None else np.asarray(task_ids).copy()))
        return np.zeros((np.asarray(imgs).shape[0], 3, 7), np.float32)


COMMON = dict(batch_size=2, success_buffer_batch_size=2, horizon_length=3, bc_horizon=4, discount=0.99,
              buffer_size=2000, eval_episodes=2, seed=3)
PER_STEP = dict(online_steps=60, start_training=10, eval_interval=20, log_interval=5)
ROUNDS = dict(num_rounds=3, rollouts_per_round=2, updates_per_round=5, eval_interval=4, log_interval=3)


def _online(**extra):
    log = []
    agent, logger = PlainStub(log, strict=True), Logger()
    R.run_pi05_online_rl(agent, TaskEnv(0, reg()), TaskEnv(0, reg()), {'grad_diagnostics': True}, logger,
                         **COMMON, **PER_STEP, **extra)
    return SimpleNamespace(log=log, logs=logger.logs)


def _rounds(m=1, **extra):
    log = []
    agent, logger = PlainStub(log, strict=True), Logger()
    if m == 1:
        env, eval_env = TaskEnv(0, reg()), TaskEnv(0, reg())
    else:
        env = [('a', [TaskEnv(0, reg()) for _ in range(m)], 0), ('b', [TaskEnv(1, reg()) for _ in range(m)], 1)]
        eval_env = [('a', [TaskEnv(0, reg()) for _ in range(m)], 0), ('b', [TaskEnv(1, reg()) for _ in range(m)], 1)]
    R.run_pi05_round_rl(agent, env, eval_env, {'grad_diagnostics': True, 'ppo_batch_size': 2}, logger,
                        **COMMON, **{**ROUNDS, **extra})
    return SimpleNamespace(log=log, logs=logger.logs)


def _check_frozen(run):
    kinds = [c[0] for c in run.log]
    assert 'update' not in kinds and 'diag' not in kinds
    assert kinds.count('critic_update') > 5
    agent_logs = [d for p, _, d in run.logs if p == 'online_agent']
    assert agent_logs and all(not any(k.startswith('diag/') for k in d) for d in agent_logs)
    assert all('critic/td_loss' in d for d in agent_logs)
    assert {p for p, _, _ in run.logs} >= {'eval', 'eval_sde', 'env', 'online_agent'}


@pytest.mark.parametrize('interval', [1, 3])
def test_online_freeze_actor(interval):
    _check_frozen(_online(freeze_actor=True, actor_update_interval=interval))


@pytest.mark.parametrize('interval,m', [(1, 1), (2, 2)])
def test_rounds_freeze_actor(interval, m):
    run = _rounds(m=m, freeze_actor=True, actor_update_interval=interval)
    _check_frozen(run)
    assert [c[0] for c in run.log].count('critic_update') == ROUNDS['num_rounds'] * ROUNDS['updates_per_round']


def test_online_freeze_actor_false_keeps_actor_steps():
    kinds = [c[0] for c in _online(freeze_actor=False).log]
    assert 'critic_update' not in kinds and 'diag' in kinds and 'plain' not in kinds


@pytest.mark.parametrize('m', [1, 2])
def test_rounds_first_round_policy_ode(m):
    base = _rounds(m=m)
    run = _rounds(m=m, first_round_policy='ode')
    kinds = [c[0] for c in run.log]
    first_update = kinds.index('update')
    plain = [i for i, k in enumerate(kinds) if k == 'plain']
    assert plain and max(plain) < first_update                  # 'plain' only in round 0's collection
    # round 0 collection has no BoN call at all; eval at step 0 (before round 0) has ode + bon
    first_plain = plain[0]
    assert 'bon' not in kinds[first_plain:first_update]
    assert 'bon' in kinds[first_update:]                        # later rounds use BoN
    # identical schedule and RNG stream otherwise: replace the round-0 policy calls and compare to the default run
    assert [('bon' if k == 'plain' else k) for k in kinds] == [c[0] for c in base.log]
    for a, b in zip(run.log, base.log):
        if a[0] == 'plain':
            np.testing.assert_array_equal(a[3], b[3])           # same key as the BoN call it replaced
    rounds = [d for p, _, d in run.logs if p == 'env' and 'round' in d]
    assert len(rounds) == ROUNDS['num_rounds']


def test_rounds_default_never_plain():
    assert 'plain' not in [c[0] for c in _rounds().log]


def test_policy_bad_mode_raises():
    with pytest.raises(ValueError, match='ode \\| plain \\| bon'):
        R._policy(StubAgent([]), 'sde')


# --- main_pi05 ---------------------------------------------------------------------------------------------------

from tests.test_pi05_multitask import GOOD, _Built, _main, _patch_main  # noqa: E402


@pytest.mark.parametrize('extra,match', [
    ([f'--env.name={GOOD}', '--training.first_round_policy=plain', '--training.collection_mode=rounds'],
     'first_round_policy=.*bon \\| ode'),
    ([f'--env.name={GOOD}', '--training.first_round_policy=ode'], 'collection_mode=rounds'),
])
def test_main_pi05_new_validation(monkeypatch, extra, match):
    M = _main(monkeypatch, *extra)
    built = _patch_main(monkeypatch, M)
    with pytest.raises(ValueError, match=match):
        M.main()
    assert 'envs' not in built


_RND = ['--training.num_rounds=1', '--training.rollouts_per_round=1', '--training.updates_per_round=1']


@pytest.mark.parametrize('extra,runner,freeze,frp', [
    (['--training.collection_mode=rounds', *_RND], 'rounds', False, 'bon'),
    (['--training.collection_mode=rounds', *_RND, '--training.freeze_actor=true', '--training.first_round_policy=ode'],
     'rounds', True, 'ode'),
    (['--training.freeze_actor=true'], 'online', True, None),
    ([], 'online', False, None),
])
def test_main_pi05_forwards(monkeypatch, capsys, extra, runner, freeze, frp):
    M = _main(monkeypatch, f'--env.name={GOOD}', *extra)
    built = _patch_main(monkeypatch, M)
    with pytest.raises(_Built):
        M.main()
    assert built['runner'] == runner
    assert built['kw']['freeze_actor'] is freeze
    assert built['kw'].get('first_round_policy') == frp
    assert ('freeze_actor: critic-only' in capsys.readouterr().out) == freeze


def test_main_pi05_agent_keys_reach_agent_config(monkeypatch):
    M = _main(monkeypatch, f'--env.name={GOOD}', '--agent.pi05_bon_sampler=ode', '--agent.pi05_bon_agg=mean')
    built = _patch_main(monkeypatch, M)
    with pytest.raises(_Built):
        M.main()
    c = built['agent_config']
    assert c['pi05_bon_sampler'] == 'ode' and c['pi05_bon_agg'] == 'mean' and c['subsample_bon'] is True


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
