"""
Tests for the pi05 robot profile (docs/changes/2026-10-14-molmo-pi05/): libero vs droid action contract.

  - libero: `_critic_actions` / `_normalize_env_actions` / `_to_env_actions` / the BC target are bit-identical to
    VERBATIM copies of ogpo/agents/ogpo_pi05.py @ e546157 (pre-change).
  - droid (pi05_droid_jointpos): 8-D env actions; dims 0-6 are deltas to the window-start state read back from the
    critic features (`_window_state`), dim 7 (gripper) absolute; model <-> env round trip; the BC target is
    normalise-then-ZERO-pad (openpi order, not VPT FSFT's pad-then-normalise = -1 on padded dims).
  - create(): pi05_robot validated, droid norm stats loaded from assets/droid, dim_mask / critic width use 8 dims.

Run:  JAX_PLATFORMS=cpu python -m pytest tests/test_pi05_robot_profile.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ogpo.agents.ogpo_pi05 as P
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T

B, H_PI, HL = 4, 15, 5
FLAT = H_PI * T.MODEL_ACTION_DIM

# pi05_droid_jointpos assets/droid/norm_stats.json q01 / q99 (first 8 dims; the other 24 are 0 / 0), rounded.
DROID_ACT_Q01 = [-0.287, -0.459, -0.286, -0.475, -0.466, -0.442, -0.532, 0.0]
DROID_ACT_Q99 = [0.286, 0.527, 0.283, 0.47, 0.453, 0.5, 0.532, 1.0]
DROID_STATE_Q01 = [-0.828, -0.84, -0.843, -2.773, -1.843, 1.172, -2.047, 0.0]
DROID_STATE_Q99 = [0.9, 1.385, 0.692, -0.454, 1.732, 3.467, 2.198, 0.991]


def _pad32(v):
    return jnp.asarray(np.concatenate([np.asarray(v, np.float32), np.zeros(32 - len(v), np.float32)]))


class _BCProbe(OGPOPi05Agent):
    """`_actor_fn` returns the flow velocity of a known target chunk TARGET, so the BC loss is ~0 iff the agent's
    internal target equals TARGET."""
    TARGET = None

    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        n = images.shape[0]
        return None, None, jnp.ones((n, 1), bool)

    def _actor_fn(self, params, kv_cache, prefix_mask):
        a = type(self).TARGET

        def actor_fn(obs, x, t, params_=None, is_encoded=True):
            x0 = (x - t * a) / (1.0 - t)
            return a - x0

        return actor_fn


def _agent(robot, cls=OGPOPi05Agent):
    if robot == 'droid':
        consts = {'action_q01': _pad32(DROID_ACT_Q01), 'action_q99': _pad32(DROID_ACT_Q99),
                  'state_q01': jnp.asarray(DROID_STATE_Q01), 'state_q99': jnp.asarray(DROID_STATE_Q99)}
    else:
        consts = {'action_q01': -jnp.ones(32) * 0.8, 'action_q99': jnp.ones(32) * 1.3,
                  'state_q01': -jnp.ones(STATE_DIM) * 0.5, 'state_q99': jnp.ones(STATE_DIM) * 2.0}
    return cls(rng=jax.random.PRNGKey(0), actor_params={}, target_actor_params={}, actor_opt_state=None,
               init_actor_params={}, frozen_params={}, critic_network=None, consts=consts,
               config=dict(pi05_action_horizon=H_PI, horizon_length=HL), model=None, actor_tx=None, robot=robot)


def _features(agent, raw_state, seed=0):
    """[prefix | normalised state], exactly as `encode` builds them."""
    q01, q99 = agent.consts['state_q01'], agent.consts['state_q99']
    state = (raw_state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    prefix = jax.random.normal(jax.random.PRNGKey(100 + seed), (raw_state.shape[0], PREFIX_DIM))
    return jnp.concatenate([prefix, state], axis=-1)


def _raw_state(seed=0):
    r = np.random.RandomState(seed)
    lo, hi = np.asarray(DROID_STATE_Q01), np.asarray(DROID_STATE_Q99)
    return jnp.asarray(r.uniform(lo, hi, (B, STATE_DIM)), jnp.float32)


# --- verbatim copies of ogpo/agents/ogpo_pi05.py @ e546157 --------------------------------------------------------

def _head_critic_actions(self, flat_actions):
    h, d = self._chunk_shape
    x = flat_actions.reshape(-1, h, d)[:, :self.config['horizon_length'], :T.ENV_ACTION_DIM]
    return x.reshape(x.shape[0], -1)


def _head_normalize_env_actions(self, env_actions):
    q01, q99 = self.consts['action_q01'][:T.ENV_ACTION_DIM], self.consts['action_q99'][:T.ENV_ACTION_DIM]
    return (env_actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _head_to_env_actions(self, flat_actions):
    h, d = self._chunk_shape
    x = flat_actions.reshape(-1, h, d)[:, :self.config['horizon_length'], :T.ENV_ACTION_DIM]
    q01, q99 = self.consts['action_q01'][:T.ENV_ACTION_DIM], self.consts['action_q99'][:T.ENV_ACTION_DIM]
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _head_bc_target(self, bc_actions):
    env_actions = _head_normalize_env_actions(self, bc_actions)
    a = jnp.pad(env_actions, ((0, 0), (0, 0), (0, T.MODEL_ACTION_DIM - T.ENV_ACTION_DIM)))
    return a.reshape(a.shape[0], -1)


# --- libero: unchanged ---------------------------------------------------------------------------------------------

def test_libero_profile_bit_identical_to_head():
    a = _agent('libero')
    assert a.env_action_dim == 7 and a.clip_env_actions
    flat = jax.random.normal(jax.random.PRNGKey(1), (B, FLAT)) * 1.5
    obs = jax.random.normal(jax.random.PRNGKey(2), (B, PREFIX_DIM + STATE_DIM))
    env = jax.random.normal(jax.random.PRNGKey(3), (B, HL, 7))
    assert np.array_equal(a._critic_actions(flat), _head_critic_actions(a, flat))
    assert np.array_equal(a._to_env_actions(flat), _head_to_env_actions(a, flat))
    assert np.array_equal(a._to_env_actions(flat, obs), _head_to_env_actions(a, flat))  # obs ignored without deltas
    assert np.array_equal(a._normalize_env_actions(env), _head_normalize_env_actions(a, env))
    assert np.array_equal(a._normalize_env_actions(env, obs), _head_normalize_env_actions(a, env))


def test_libero_bc_target_bit_identical_to_head():
    a = _agent('libero', _BCProbe)
    bc_actions = jax.random.normal(jax.random.PRNGKey(4), (B, H_PI, 7))
    _BCProbe.TARGET = _head_bc_target(a, bc_actions)
    bc = {'images': jnp.zeros((B, 2, 2, 2, 3)), 'actions': bc_actions, 'valid': jnp.ones((B, H_PI))}
    loss = a._bc_loss_per_sample(bc, None, jax.random.PRNGKey(5))
    assert float(jnp.max(loss)) < 1e-6, loss


# --- droid: deltas against the window-start state --------------------------------------------------------------------

def test_droid_window_state_recovers_raw_state():
    a = _agent('droid')
    s = _raw_state()
    np.testing.assert_allclose(a._window_state(_features(a, s)), s, atol=1e-5)


def test_droid_to_env_adds_state_to_joints_only():
    a = _agent('droid')
    assert a.env_action_dim == 8 and not a.clip_env_actions
    s = _raw_state(1)
    flat = jax.random.normal(jax.random.PRNGKey(6), (B, FLAT))
    env = a._to_env_actions(flat, _features(a, s))
    assert env.shape == (B, HL, 8)
    x = flat.reshape(B, H_PI, 32)[:, :HL, :8]
    q01, q99 = a.consts['action_q01'][:8], a.consts['action_q99'][:8]
    unnorm = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    np.testing.assert_allclose(env[..., :7], unnorm[..., :7] + s[:, None, :7], atol=1e-5)
    np.testing.assert_allclose(env[..., 7], unnorm[..., 7], atol=1e-6)  # gripper absolute
    # A different chunk-start state shifts the joints by exactly the state difference, the gripper not at all.
    s2 = s + 0.3
    env2 = a._to_env_actions(flat, _features(a, s2))
    np.testing.assert_allclose(env2[..., :7] - env[..., :7], np.full((B, HL, 7), 0.3), atol=1e-4)
    np.testing.assert_allclose(env2[..., 7], env[..., 7], atol=1e-6)


def test_droid_round_trip_model_env_model():
    a = _agent('droid')
    feats = _features(a, _raw_state(2))
    flat = jax.random.normal(jax.random.PRNGKey(7), (B, FLAT))
    back = a._normalize_env_actions(a._to_env_actions(flat, feats), feats)
    np.testing.assert_allclose(back, flat.reshape(B, H_PI, 32)[:, :HL, :8], atol=1e-4)
    # critic sees the same model-space chunk the sampler produced
    np.testing.assert_allclose(back.reshape(B, -1), a._critic_actions(flat), atol=1e-4)
    assert a._critic_actions(flat).shape == (B, HL * 8)


def test_droid_requires_critic_obs():
    a = _agent('droid')
    with pytest.raises(AssertionError, match='delta actions'):
        a._to_env_actions(jnp.zeros((B, FLAT)))
    with pytest.raises(AssertionError, match='delta actions'):
        a._normalize_env_actions(jnp.zeros((B, HL, 8)))


def test_droid_bc_target_is_delta_then_zero_pad():
    a = _agent('droid', _BCProbe)
    s = _raw_state(3)
    feats = _features(a, s)
    bc_actions = jnp.asarray(np.random.RandomState(4).uniform(-0.3, 0.3, (B, H_PI, 8)), jnp.float32)
    bc_actions = bc_actions.at[..., :7].add(s[:, None, :7])  # absolute joint targets near the window-start state
    q01, q99 = a.consts['action_q01'][:8], a.consts['action_q99'][:8]
    delta = bc_actions.at[..., :7].add(-s[:, None, :7])
    want = (delta - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    want = jnp.concatenate([want, jnp.zeros((B, H_PI, 24))], axis=-1).reshape(B, -1)
    _BCProbe.TARGET = want
    bc = {'images': jnp.zeros((B, 2, 2, 2, 3)), 'observations': feats, 'actions': bc_actions,
          'valid': jnp.ones((B, H_PI))}
    loss = a._bc_loss_per_sample(bc, None, jax.random.PRNGKey(8))
    assert float(jnp.max(loss)) < 1e-4, loss
    # VPT FSFT's pad-then-normalise would put -1 on the 24 padded dims: clearly distinguishable.
    _BCProbe.TARGET = want.reshape(B, H_PI, 32).at[..., 8:].set(-1.0).reshape(B, -1)
    assert float(jnp.min(a._bc_loss_per_sample(bc, None, jax.random.PRNGKey(8)))) > 0.1


# --- molmo recipe: no action clipping (act_min/act_max = -inf/inf, clip_intermediate_actions = false) ----------------

class _Linear(OGPOPi05Agent):
    """Actor velocity that drives samples well past +-1, so any clip would show."""

    def _actor_fn(self, params, kv_cache, prefix_mask):
        def actor_fn(obs, x, t, params_=None, is_encoded=True):
            return params['w'] * x + 2.5

        return actor_fn


def _recipe_agent(**kw):
    """droid agent with the noise / clip settings the molmo recipes pass (scripts/ogpo_pi05/molmo_mt16.sh)."""
    from ogpo.utils.config_loader import config_to_agent_dict, load_config, parse_cli_args
    cfg = config_to_agent_dict(load_config(*parse_cli_args([
        '--algo=ogpo', '--agent.flow_steps=10', '--agent.ft_flow_steps=10', '--agent.use_tapered_noise=true',
        '--agent.use_constant_noise=false', '--agent.error_correct_sde_to_ode=true', '--agent.constant_noise_std=0.1',
        '--agent.act_min=-inf', '--agent.act_max=inf', '--agent.clip_intermediate_actions=false'])))
    cfg.update(pi05_action_horizon=H_PI, horizon_length=HL, **kw)
    a = _agent('droid', _Linear)
    dim_mask = np.zeros((H_PI, 32), np.float32)
    dim_mask[:HL, :8] = 1.0
    return a.replace(config=cfg, consts={**a.consts, 'dim_mask': jnp.asarray(dim_mask.reshape(-1))})


def test_recipe_no_clip_config_parses():
    c = _recipe_agent().config
    assert c['act_min'] == -np.inf and c['act_max'] == np.inf and c['clip_intermediate_actions'] is False


def test_recipe_no_clip_sde_unclipped_and_logprob_consistent():
    a = _recipe_agent()
    params, n = {'w': jnp.float32(0.4)}, 6
    pmask = jnp.ones((n, 1), bool)
    actions, chains, old_lp, _ = a._sample_sde(params, None, pmask, jax.random.PRNGKey(11))
    real = actions.reshape(n, H_PI, 32)[:, :HL, :8]
    assert float(jnp.max(jnp.abs(real))) > 1.5, "samples should exceed +-1 when nothing clips"
    lp, _, _ = a._log_prob(params, None, pmask, chains, get_entropy=False)
    np.testing.assert_allclose(lp, old_lp, rtol=1e-4, atol=1e-4)  # PPO ratio = 1 at the sampling params


def test_recipe_no_clip_ode_eval_unclipped():
    a = _recipe_agent()
    n = 3
    flat = a._sample_ode({'w': jnp.float32(0.4)}, None, jnp.ones((n, 1), bool), jax.random.PRNGKey(12), clip=True)
    assert float(jnp.max(jnp.abs(flat.reshape(n, H_PI, 32)[:, :HL, :8]))) > 1.5


# --- create() -----------------------------------------------------------------------------------------------------

class _Stop(Exception):
    pass


class _StopCfg(dict):
    def __getitem__(self, k):
        if k == 'pi05_clip_grad_norm':  # after norm stats / dim_mask / consts are built
            raise _Stop()
        return super().__getitem__(k)


@pytest.fixture
def fake_ckpt(monkeypatch):
    calls = []
    monkeypatch.setattr(P, 'Pi05', lambda **kw: None)
    monkeypatch.setattr(P, 'load_pi05_params', lambda path: {})
    monkeypatch.setattr(P, 'split_params', lambda p: ({}, {'w': np.zeros(2)}))
    ns = {'actions': {'q01': np.zeros(32), 'q99': np.ones(32)}, 'state': {'q01': np.zeros(32), 'q99': np.ones(32)}}
    monkeypatch.setattr(P.T, 'load_norm_stats', lambda path, asset_id: calls.append((path, asset_id)) or ns)

    class _Tok:
        def __init__(self, *a):
            pass

        def tokenize(self, prompt):
            return np.zeros(4, np.int32), np.ones(4, bool)

    monkeypatch.setattr(P.T, 'PromptTokenizer', _Tok)
    return calls


def _create_cfg(**kw):
    c = dict(pi05_checkpoint='/ckpt', pi05_robot='droid', pi05_tokenizer='x', pi05_action_horizon=H_PI,
             horizon_length=HL, q_variance_reduction=False, q_vr_num_samples=16, q_vr_reduction='mean',
             pi05_critic_pooling='all', pi05_critic_arch='mlp', subsample_bon=True, pi05_bon_sampler='sde',
             pi05_bon_agg='subsample', use_value_fn=False)
    c.update(kw)
    return _StopCfg(c)


def test_create_rejects_unknown_robot(fake_ckpt):
    with pytest.raises(ValueError, match="pi05_robot='franka'.*libero.*droid"):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_robot='franka'))


@pytest.mark.parametrize('robot,asset_id', [('droid', 'droid'), ('libero', 'physical-intelligence/libero')])
def test_create_loads_robot_norm_stats(fake_ckpt, robot, asset_id):
    with pytest.raises(_Stop):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_robot=robot))
    assert fake_ckpt == [('/ckpt/assets', asset_id)]
