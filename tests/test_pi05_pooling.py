"""
CPU tests for OGPOPi05Agent._pool_prefix / encode / create `pi05_critic_pooling`
(docs/changes/2026-09-28-pi05-critic-segmented-pooling).

The pi05 VLM is replaced by a fake `_prefix` that returns a crafted (B, S, D) prefix_out and
prefix_mask laid out as `Pi05.embed_prefix` does: 3 cameras x n_cam tokens (third camera
masked, as `_model_images` sets it), then n_lang language tokens (trailing padding masked).

Run:  JAX_PLATFORMS=cpu uv run pytest tests/test_pi05_pooling.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import ogpo.agents.ogpo_pi05 as P
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, POOLING_SEGMENTS, PREFIX_DIM, STATE_DIM, T

B = 3
# (n_cam, n_lang, n_valid_lang, D); the second is the real layout (256 SigLIP tokens, MAX_TOKEN_LEN=200).
SHAPES = [(4, 5, 3, 6), (256, T.MAX_TOKEN_LEN, 17, 4)]


class _FakePi05(OGPOPi05Agent):
    def _prefix(self, images, task_ids, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        assert task_ids.shape == (images.shape[0],)
        return self.consts['fake_out'], None, self.consts['fake_mask']


def _layout(n_cam, n_lang, n_valid):
    mask = np.concatenate([np.ones(n_cam), np.ones(n_cam), np.zeros(n_cam),
                           np.ones(n_valid), np.zeros(n_lang - n_valid)]).astype(bool)
    return np.broadcast_to(mask, (B, 3 * n_cam + n_lang)).copy()


def _agent(mode, prefix_out, mask, n_lang):
    consts = {'tokens': jnp.zeros((1, n_lang), jnp.int32),
              'state_q01': -jnp.ones(STATE_DIM), 'state_q99': jnp.ones(STATE_DIM) * 3.0,
              'fake_out': jnp.asarray(prefix_out), 'fake_mask': jnp.asarray(mask)}
    return _FakePi05(rng=jax.random.PRNGKey(0), actor_params={}, target_actor_params={}, actor_opt_state=None,
                     init_actor_params={},
                     frozen_params={}, critic_network=None, consts=consts,
                     config={'pi05_critic_pooling': mode}, model=None, actor_tx=None)


def _states(seed=3):
    return jnp.asarray(np.random.RandomState(seed).randn(B, STATE_DIM), jnp.float32)


def _encode(mode, x, mask, n_lang):
    return np.asarray(_agent(mode, x, mask, n_lang).encode(jnp.zeros((B, 2, 2, 2, 3), jnp.uint8), _states(),
                                                     jnp.zeros(B, jnp.int32)))


def _ref(mode, x, mask, n_cam, n_lang):
    x = np.asarray(x, np.float64); m = mask.astype(np.float64)
    S = x.shape[1]; n_img = S - n_lang
    spans = {'segmented': [(0, n_img), (n_img, S)],
             'per_camera': [(0, n_cam), (n_cam, 2 * n_cam), (n_img, S)]}[mode]
    feats = []
    for lo, hi in spans:
        idx = np.nonzero(m[0, lo:hi])[0] + lo           # same mask for every row
        seg = x[:, idx].mean(axis=1)
        feats.append(seg / np.linalg.norm(seg, axis=-1, keepdims=True))
    st = np.asarray(_states(), np.float64)
    state = (st + 1.0) / (4.0 + 1e-6) * 2.0 - 1.0
    return np.concatenate(feats + [state], axis=-1)


def _x(n_cam, n_lang, D, seed=0, dtype=np.float32):
    return np.random.RandomState(seed).randn(B, 3 * n_cam + n_lang, D).astype(dtype) + 0.5


# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_cam,n_lang,n_valid,D', SHAPES)
def test_all_bit_identical_to_head(n_cam, n_lang, n_valid, D):
    x = _x(n_cam, n_lang, D); mask = _layout(n_cam, n_lang, n_valid)
    out = _encode('all', x, mask, n_lang)
    # HEAD 955ae22 encode, verbatim expression
    q01, q99 = -jnp.ones(STATE_DIM), jnp.ones(STATE_DIM) * 3.0
    state = (_states() - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    old = jax.jit(lambda p, s: jnp.concatenate([p.mean(axis=1).astype(jnp.float32), s], axis=-1))(jnp.asarray(x), state)
    np.testing.assert_array_equal(out, np.asarray(old))
    assert out.shape == (B, D + STATE_DIM)


@pytest.mark.parametrize('n_cam,n_lang,n_valid,D', SHAPES)
@pytest.mark.parametrize('mode', ['segmented', 'per_camera'])
def test_matches_numpy_reference(mode, n_cam, n_lang, n_valid, D):
    x = _x(n_cam, n_lang, D); mask = _layout(n_cam, n_lang, n_valid)
    out = _encode(mode, x, mask, n_lang)
    assert out.shape == (B, POOLING_SEGMENTS[mode] * D + STATE_DIM)
    np.testing.assert_allclose(out, _ref(mode, x, mask, n_cam, n_lang), rtol=1e-5, atol=1e-5)
    for k in range(POOLING_SEGMENTS[mode]):
        np.testing.assert_allclose(np.linalg.norm(out[:, k * D:(k + 1) * D], axis=-1), 1.0, atol=1e-5)


@pytest.mark.parametrize('mode', ['segmented', 'per_camera'])
def test_bf16_prefix(mode):
    n_cam, n_lang, n_valid, D = SHAPES[0]
    x = _x(n_cam, n_lang, D)
    mask = _layout(n_cam, n_lang, n_valid)
    xb = jnp.asarray(x).astype(jnp.bfloat16)
    out = _encode(mode, xb, mask, n_lang)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, _ref(mode, np.asarray(xb.astype(jnp.float32)), mask, n_cam, n_lang),
                               rtol=1e-5, atol=1e-5)
    assert _encode('all', xb, mask, n_lang).dtype == np.float32


@pytest.mark.parametrize('n_cam,n_lang,n_valid,D', SHAPES)
@pytest.mark.parametrize('mode', ['segmented', 'per_camera'])
def test_masked_positions_ignored(mode, n_cam, n_lang, n_valid, D):
    x = _x(n_cam, n_lang, D); mask = _layout(n_cam, n_lang, n_valid)
    base = _encode(mode, x, mask, n_lang)
    y = x.copy()
    y[:, 2 * n_cam:3 * n_cam] += 50.0                   # dummy third camera
    y[:, 3 * n_cam + n_valid:] -= 70.0                  # prompt padding
    np.testing.assert_array_equal(_encode(mode, y, mask, n_lang), base)


@pytest.mark.parametrize('n_cam,n_lang,n_valid,D', SHAPES)
def test_per_camera_segment_isolation(n_cam, n_lang, n_valid, D):
    x = _x(n_cam, n_lang, D); mask = _layout(n_cam, n_lang, n_valid)
    base = _encode('per_camera', x, mask, n_lang)
    regions = [(0, 1), (n_cam, n_cam + 1), (3 * n_cam, 3 * n_cam + 1)]   # base, wrist, first lang token
    for k, (lo, hi) in enumerate(regions):
        y = x.copy()
        y[:, lo:hi] += np.linspace(1, 3, D)[None, None] * 5.0
        out = _encode('per_camera', y, mask, n_lang)
        for j in range(3):
            sl = slice(j * D, (j + 1) * D)
            changed = not np.array_equal(out[:, sl], base[:, sl])
            assert changed == (j == k), (k, j)
        np.testing.assert_array_equal(out[:, 3 * D:], base[:, 3 * D:])


@pytest.mark.parametrize('n_cam,n_lang,n_valid,D', SHAPES)
def test_segmented_unmasked_changes(n_cam, n_lang, n_valid, D):
    x = _x(n_cam, n_lang, D); mask = _layout(n_cam, n_lang, n_valid)
    base = _encode('segmented', x, mask, n_lang)
    for k, pos in enumerate([n_cam, 3 * n_cam + n_valid - 1]):   # wrist token -> img seg; last valid lang -> lang seg
        y = x.copy(); y[:, pos] += np.linspace(1, 3, D) * 5.0
        out = _encode('segmented', y, mask, n_lang)
        for j in range(2):
            sl = slice(j * D, (j + 1) * D)
            assert (not np.array_equal(out[:, sl], base[:, sl])) == (j == k)


@pytest.mark.parametrize('mode', ['segmented', 'per_camera'])
def test_zero_segment_finite(mode):
    n_cam, n_lang, n_valid, D = SHAPES[0]
    x = _x(n_cam, n_lang, D); x[:, :n_cam] = 0.0; x[:, 3 * n_cam:] = 0.0
    mask = _layout(n_cam, n_lang, n_valid)
    out = _encode(mode, x, mask, n_lang)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[:, (POOLING_SEGMENTS[mode] - 1) * D:POOLING_SEGMENTS[mode] * D], 0.0)
    # fully-masked span (all-zero mask) is also finite
    m0 = mask.copy(); m0[:, 3 * n_cam:] = False
    assert np.isfinite(_encode(mode, _x(n_cam, n_lang, D), m0, n_lang)).all()


@pytest.mark.parametrize('mode', ['segmented', 'per_camera'])
def test_two_tier_segment_share(mode):
    """TwoTierObsEncoder re-normalises the joint k*D slice: each unit segment enters with norm 1/sqrt(k)."""
    n_cam, n_lang, n_valid, D = SHAPES[0]
    k = POOLING_SEGMENTS[mode]
    out = _encode(mode, _x(n_cam, n_lang, D), _layout(n_cam, n_lang, n_valid), n_lang)
    img = jnp.asarray(out[:, :k * D])
    img_norm = np.asarray(img / (jnp.linalg.norm(img, axis=-1, keepdims=True) + 1e-6))   # two_tier.py:48
    for j in range(k):
        np.testing.assert_allclose(np.linalg.norm(img_norm[:, j * D:(j + 1) * D], axis=-1), 1 / np.sqrt(k), atol=1e-5)


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ckpt(monkeypatch):
    calls = []
    monkeypatch.setattr(P, 'Pi05', lambda **kw: None)
    monkeypatch.setattr(P, 'load_pi05_params', lambda path: calls.append(path) or {})
    monkeypatch.setattr(P, 'split_params', lambda p: ({}, {'w': np.zeros(2)}))
    ns = {'actions': {'q01': np.zeros(32), 'q99': np.ones(32)}, 'state': {'q01': np.zeros(32), 'q99': np.ones(32)}}
    monkeypatch.setattr(P.T, 'load_norm_stats', lambda path, asset_id: calls.append(path) or ns)

    class _Tok:
        def __init__(self, *a):
            calls.append('tok')

        def tokenize(self, prompt):
            return np.zeros(4, np.int32), np.ones(4, bool)

    monkeypatch.setattr(P.T, 'PromptTokenizer', _Tok)
    return calls


def _create_cfg(mode):
    return dict(pi05_checkpoint='/nonexistent', pi05_robot='libero', pi05_tokenizer='x', pi05_action_horizon=10, horizon_length=5,
                pi05_clip_grad_norm=1.0, pi05_lr=1e-5, pi05_weight_decay=0.0,
                q_variance_reduction=False, q_vr_num_samples=8, q_vr_reduction='mean',
                pi05_critic_pooling=mode, value_hidden_dims=(16, 16), layer_norm=True, num_qs=2,
                q_min=-100.0, q_max=0.0, critic_scheduler='constant', ppo_lr=1e-4, critic_warmup_steps=0,
                critic_decay_steps=1, critic_end_value=0.0, clip_grad_norm=1.0, critic_weight_decay=0.0,
                pi05_critic_arch='mlp', pi05_bronet_hidden_dim=1024, pi05_bronet_depth=2,
                subsample_bon=True, pi05_bon_sampler='sde', pi05_bon_agg='subsample', use_value_fn=False)


@pytest.mark.parametrize('mode', ['mean', 'ALL', 'percamera', None])
def test_create_rejects_unknown_mode_before_loading(fake_ckpt, mode):
    with pytest.raises(ValueError, match='pi05_critic_pooling'):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(mode))
    assert fake_ckpt == []


@pytest.mark.parametrize('mode', ['all', 'segmented', 'per_camera'])
def test_create_critic_width(fake_ckpt, mode):
    agent = OGPOPi05Agent.create(0, ['task'], _create_cfg(mode))
    width = POOLING_SEGMENTS[mode] * PREFIX_DIM
    k0 = agent.critic_network.params['modules_critic']
    shapes = {'/'.join(map(str, p)): v.shape for p, v in jax.tree_util.tree_flatten_with_path(k0)[0]}
    img_proj = [s for n, s in shapes.items() if 'img_proj' in n and 'kernel' in n]
    assert img_proj and all(s[-2] == width for s in img_proj), shapes
    obs = jnp.asarray(np.random.RandomState(0).randn(4, width + STATE_DIM), jnp.float32)
    act = jnp.zeros((4, 5 * T.ENV_ACTION_DIM))
    q = agent.critic_network.select('critic')(obs, act, is_encoded=True)
    assert q.shape == (2, 4) and np.isfinite(np.asarray(q)).all()
    # A narrower obs (e.g. 'all'-width features into a wider critic) fails loudly. A wider one is
    # silently sliced by TwoTierObsEncoder (pre-existing behaviour, not introduced by this change).
    if mode != 'all':
        with pytest.raises(Exception):
            agent.critic_network.select('critic')(jnp.zeros((4, PREFIX_DIM + STATE_DIM)), act, is_encoded=True)


def test_yaml_and_script():
    from ogpo.utils.config_loader import load_config
    assert load_config('ogpo', {})['agent']['pi05_critic_pooling'] == 'all'
    sh = open(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'ogpo_pi05', 'molmo_mt16.sh')).read()
    assert 'pi05_critic_pooling=all' in sh  # bronet_vla requires 'all'
    assert '--agent.pi05_critic_pooling=$pi05_critic_pooling' in sh
