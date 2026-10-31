"""
CPU tests for the pi05 BroNet critic (docs/changes/2026-10-06-pi05-bronet-critic).

  - BroNet forward == hand-written numpy forward on the same (perturbed) params, depth 1/2/3.
  - BroNet == a verbatim copy of vla-post-training's nnx BroNet (add_final_layer=True) on copied params.
  - param tree / count formula, zero biases, orthogonal(sqrt 2) kernels on every Dense.
  - ensemblized heads get independent init; Value(bronet) output (num_qs, B) / (B,).
  - Value(head_arch='mlp') param tree + outputs identical to the verbatim pre-change Value
    (tests/_value_pre_bronet.py), with / without obs_two_tier, num_ensembles 1 / 10.
  - ValueError: unknown head_arch, bronet + hlgauss / use_film / action_repeat, create() bad pi05_critic_arch.
  - pi05 path: critic built by OGPOPi05Agent.create (checkpoint loaders faked) with bronet, through the
    real critic_loss / critic_update / update (fake pi05 prefix + sampler + actor loss): finite, critic
    params change, target critic moves by tau; grad_accum_steps=2 == full batch for a deterministic sampler.

Run:  JAX_PLATFORMS=cpu uv run pytest tests/test_pi05_bronet.py -q
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ogpo.agents.ogpo_pi05 as P
from ogpo.agents.ogpo_pi05 import OGPOPi05Agent, PREFIX_DIM, STATE_DIM, T, POOLING_SEGMENTS
from ogpo.networks import Value
from ogpo.networks.critics import ensemblize
from ogpo.networks.modules import BroNet

import _value_pre_bronet as OLD

IN, HID, BATCH = 13, 32, 5


def _flat(tree):
    return {'/'.join(str(getattr(k, 'key', k)) for k in p): v
            for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


def _perturb(params, seed):
    """Random biases / LN scales so a wrong LN or bias order is visible in the forward check."""
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree.unflatten(treedef, [x + 0.3 * jax.random.normal(k, x.shape) for x, k in zip(leaves, keys)])


# ---------------------------------------------------------------------------
# BroNet module
# ---------------------------------------------------------------------------

def _np_ln(x, p, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * p['scale'] + p['bias']


def _np_bronet(p, x, depth):
    p = jax.tree.map(lambda a: np.asarray(a, np.float64), p)
    x = np.asarray(x, np.float64)
    relu = lambda z: np.maximum(z, 0.0)
    dense = lambda i, z: z @ p[f'Dense_{i}']['kernel'] + p[f'Dense_{i}']['bias']
    x = relu(_np_ln(dense(0, x), p['LayerNorm_0']))
    for b in range(depth):
        r = relu(_np_ln(dense(2 * b + 1, x), p[f'LayerNorm_{2 * b + 1}']))
        r = _np_ln(dense(2 * b + 2, r), p[f'LayerNorm_{2 * b + 2}'])
        x = r + x                                    # no activation after the skip-add
    return dense(2 * depth + 1, x)


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_bronet_matches_numpy(depth):
    net = BroNet(hidden_dim=HID, depth=depth, out_dim=3)
    x = jax.random.normal(jax.random.PRNGKey(1), (BATCH, IN))
    params = _perturb(net.init(jax.random.PRNGKey(0), x)['params'], seed=depth)
    out = np.asarray(net.apply({'params': params}, x))
    assert out.shape == (BATCH, 3)
    np.testing.assert_allclose(out, _np_bronet(params, x, depth), rtol=1e-4, atol=1e-4)


def test_bronet_numpy_reference_has_teeth():
    """Activation after the skip-add (a plausible porting error) must be caught by the reference."""
    depth = 2
    net = BroNet(hidden_dim=HID, depth=depth)
    x = jax.random.normal(jax.random.PRNGKey(1), (BATCH, IN))
    params = _perturb(net.init(jax.random.PRNGKey(0), x)['params'], seed=0)
    p = jax.tree.map(lambda a: np.asarray(a, np.float64), params)
    relu = lambda z: np.maximum(z, 0.0)
    dense = lambda i, z: z @ p[f'Dense_{i}']['kernel'] + p[f'Dense_{i}']['bias']
    h = relu(_np_ln(dense(0, np.asarray(x, np.float64)), p['LayerNorm_0']))
    for b in range(depth):
        r = relu(_np_ln(dense(2 * b + 1, h), p[f'LayerNorm_{2 * b + 1}']))
        h = relu(_np_ln(dense(2 * b + 2, r), p[f'LayerNorm_{2 * b + 2}']) + h)   # mutant
    mutant = dense(2 * depth + 1, h)
    assert not np.allclose(np.asarray(net.apply({'params': params}, x)), mutant, rtol=1e-3, atol=1e-3)


# Verbatim copy of vla-post-training/src/rl/networks/bronet_critic.py `_BroNetBlock` / `BroNet` (nnx), with
# constants.default_init inlined (orthogonal(sqrt 2)). Only change: `pytree=False` on the class, because this
# venv's flax 0.12 rejects a plain list of modules (VLA pins flax 0.10.6); it does not affect the math.
def _vla_classes():
    import flax.nnx as nnx
    from typing import Callable

    def default_init(scale: float = math.sqrt(2.0)):
        return jax.nn.initializers.orthogonal(scale)

    class _BroNetBlock(nnx.Module):
        def __init__(self, hidden_dim: int, *, rngs: nnx.Rngs):
            self.linear1 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=default_init(), rngs=rngs)
            self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
            self.linear2 = nnx.Linear(hidden_dim, hidden_dim, kernel_init=default_init(), rngs=rngs)
            self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

        def __call__(self, x, activation: Callable, training: bool = False):
            res = self.linear1(x)
            res = self.norm1(res)
            res = activation(res)
            res = self.linear2(res)
            res = self.norm2(res)
            return res + x

    class VLABroNet(nnx.Module, pytree=False):
        def __init__(self, input, hidden_dim, depth, output_nodes=1, add_final_layer=False,
                     activations=nnx.relu, *, rngs: nnx.Rngs):
            input_dim = input if isinstance(input, int) else int(input.shape[-1])
            self.proj = nnx.Linear(input_dim, hidden_dim, kernel_init=default_init(), rngs=rngs)
            self.proj_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
            self.blocks = [_BroNetBlock(hidden_dim, rngs=rngs) for _ in range(depth)]
            self.activations = activations
            self.add_final_layer = add_final_layer
            if add_final_layer:
                self.final = nnx.Linear(hidden_dim, output_nodes, kernel_init=default_init(), rngs=rngs)

        def __call__(self, x, training: bool = False):
            x = self.proj(x)
            x = self.proj_norm(x)
            x = self.activations(x)
            for block in self.blocks:
                x = block(x, self.activations, training=training)
            if self.add_final_layer:
                x = self.final(x)
            return x

    return nnx, VLABroNet


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_bronet_matches_vla_nnx(depth):
    nnx, VLABroNet = _vla_classes()
    vla = VLABroNet(IN, HID, depth, output_nodes=1, add_final_layer=True, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (BATCH, IN))
    # Randomise the VLA net's LN / biases, then copy every VLA param into the linen tree.
    vla.proj_norm.scale.value = 1.0 + 0.3 * jax.random.normal(jax.random.PRNGKey(2), (HID,))
    vla.proj.bias.value = 0.3 * jax.random.normal(jax.random.PRNGKey(3), (HID,))
    for i, blk in enumerate(vla.blocks):
        blk.norm2.bias.value = 0.3 * jax.random.normal(jax.random.PRNGKey(10 + i), (HID,))
        blk.linear1.bias.value = 0.3 * jax.random.normal(jax.random.PRNGKey(20 + i), (HID,))
    lin = lambda m: {'kernel': m.kernel.value, 'bias': m.bias.value}
    ln = lambda m: {'scale': m.scale.value, 'bias': m.bias.value}
    p = {'Dense_0': lin(vla.proj), 'LayerNorm_0': ln(vla.proj_norm)}
    for i, blk in enumerate(vla.blocks):
        p[f'Dense_{2 * i + 1}'], p[f'LayerNorm_{2 * i + 1}'] = lin(blk.linear1), ln(blk.norm1)
        p[f'Dense_{2 * i + 2}'], p[f'LayerNorm_{2 * i + 2}'] = lin(blk.linear2), ln(blk.norm2)
    p[f'Dense_{2 * depth + 1}'] = lin(vla.final)
    net = BroNet(hidden_dim=HID, depth=depth)
    ref_tree = net.init(jax.random.PRNGKey(0), x)['params']
    assert jax.tree.structure(ref_tree) == jax.tree.structure(p)
    for a, b in zip(jax.tree.leaves(ref_tree), jax.tree.leaves(p)):
        assert a.shape == b.shape
    np.testing.assert_allclose(np.asarray(net.apply({'params': p}, x)), np.asarray(vla(x)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('depth,out', [(1, 1), (2, 1), (3, 4)])
def test_bronet_structure_count_init(depth, out):
    net = BroNet(hidden_dim=HID, depth=depth, out_dim=out)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, IN)))['params']
    assert set(params) == {f'Dense_{i}' for i in range(2 * depth + 2)} | {f'LayerNorm_{i}' for i in range(2 * depth + 1)}
    shapes = {k: v['kernel'].shape for k, v in params.items() if k.startswith('Dense')}
    assert shapes['Dense_0'] == (IN, HID) and shapes[f'Dense_{2 * depth + 1}'] == (HID, out)
    assert all(shapes[f'Dense_{i}'] == (HID, HID) for i in range(1, 2 * depth + 1))
    n = sum(x.size for x in jax.tree.leaves(params))
    expected = IN * HID + HID + (1 + 2 * depth) * 2 * HID + 2 * depth * (HID * HID + HID) + HID * out + out
    assert n == expected
    for k, v in params.items():
        np.testing.assert_array_equal(np.asarray(v['bias']), 0.0)            # zero biases (Dense and LN)
        if k.startswith('LayerNorm'):
            np.testing.assert_array_equal(np.asarray(v['scale']), 1.0)
            continue
        W = np.asarray(v['kernel'], np.float64)
        G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T                  # orthogonal(sqrt 2): = 2 I
        np.testing.assert_allclose(G, 2.0 * np.eye(G.shape[0]), atol=1e-4)


def test_bronet_default_scale_is_sqrt2_float():
    """kernel_init default must equal orthogonal(math.sqrt(2)) (VLA uses a Python float)."""
    k = jax.random.PRNGKey(4)
    a = BroNet.kernel_init(k, (HID, HID))
    b = jax.nn.initializers.orthogonal(math.sqrt(2.0))(k, (HID, HID))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)


def test_ensemblized_bronet_heads_independent():
    E = 4
    net = ensemblize(BroNet, E)(hidden_dim=HID, depth=2)
    x = jax.random.normal(jax.random.PRNGKey(1), (BATCH, IN))
    params = net.init(jax.random.PRNGKey(0), x)['params']
    for name, leaf in _flat(params).items():
        assert leaf.shape[0] == E, name
        if name.endswith('kernel'):
            for i in range(E):
                for j in range(i + 1, E):
                    assert not np.allclose(np.asarray(leaf[i]), np.asarray(leaf[j])), name
                W = np.asarray(leaf[i], np.float64)
                G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
                np.testing.assert_allclose(G, 2.0 * np.eye(G.shape[0]), atol=1e-4)
    out = np.asarray(net.apply({'params': params}, x))
    assert out.shape == (E, BATCH, 1)
    assert np.abs(out[0] - out[1]).max() > 1e-3
    # head e == a standalone BroNet on that head's params
    single = BroNet(hidden_dim=HID, depth=2)
    for e in range(E):
        pe = jax.tree.map(lambda a: a[e], params)
        np.testing.assert_allclose(out[e], np.asarray(single.apply({'params': pe}, x)), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Value
# ---------------------------------------------------------------------------

IMG_F, PROP = 24, STATE_DIM
ACT = 6


def _value_inputs(two_tier, b=BATCH):
    obs_dim = IMG_F + PROP if two_tier else 11
    r = np.random.RandomState(0)
    return (jnp.asarray(r.randn(b, obs_dim), jnp.float32), jnp.asarray(r.randn(b, ACT), jnp.float32))


def _tt(two_tier):
    return dict(obs_two_tier=True, two_tier_img_dim=IMG_F, two_tier_proprio_dim=PROP) if two_tier else {}


@pytest.mark.parametrize('two_tier', [False, True])
@pytest.mark.parametrize('E', [1, 10])
@pytest.mark.parametrize('loss_type', ['mse', 'hlgauss'])
def test_value_mlp_identical_to_pre_change(two_tier, E, loss_type):
    kw = dict(hidden_dims=(16, 16), layer_norm=True, num_ensembles=E, critic_loss_type=loss_type,
              num_bins=11, q_min=-10.0, q_max=0.0, **_tt(two_tier))
    obs, act = _value_inputs(two_tier)
    key = jax.random.PRNGKey(3)
    for new_def in (Value(**kw), Value(**kw, head_arch='mlp')):
        old_def = OLD.Value(**kw)
        pn = new_def.init(key, obs, act)
        po = old_def.init(key, obs, act)
        assert jax.tree.structure(pn) == jax.tree.structure(po)
        for a, b in zip(jax.tree.leaves(pn), jax.tree.leaves(po)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        on, oo = new_def.apply(pn, obs, act), old_def.apply(po, obs, act)
        np.testing.assert_array_equal(np.asarray(on), np.asarray(oo))
        assert on.shape == ((E, BATCH) if E > 1 else (BATCH,))
        if loss_type == 'hlgauss':
            _, ln = new_def.apply(pn, obs, act, return_logits=True)
            _, lo = old_def.apply(po, obs, act, return_logits=True)
            np.testing.assert_array_equal(np.asarray(ln), np.asarray(lo))


@pytest.mark.parametrize('two_tier', [False, True])
@pytest.mark.parametrize('E', [1, 10])
def test_value_bronet_shapes_and_tree(two_tier, E):
    v = Value(hidden_dims=(16, 16), num_ensembles=E, head_arch='bronet', bronet_hidden_dim=HID, bronet_depth=2,
              **_tt(two_tier))
    obs, act = _value_inputs(two_tier)
    params = v.init(jax.random.PRNGKey(0), obs, act)['params']
    assert set(params) == ({'value_net', 'two_tier_encoder'} if two_tier else {'value_net'})
    assert set(params['value_net']) == {f'Dense_{i}' for i in range(6)} | {f'LayerNorm_{i}' for i in range(5)}
    in_dim = (16 if two_tier else 11) + ACT                       # fused_dim defaults to hidden_dims[0]
    k0 = params['value_net']['Dense_0']['kernel']
    assert k0.shape == ((E, in_dim, HID) if E > 1 else (in_dim, HID))
    q = v.apply({'params': params}, obs, act)
    assert q.shape == ((E, BATCH) if E > 1 else (BATCH,)) and np.isfinite(np.asarray(q)).all()
    if E > 1:
        assert float(jnp.std(q, axis=0).mean()) > 1e-4                         # heads differ
    if two_tier:   # two-tier encoder identical to the mlp-head Value's (same submodule, same init key path)
        pm = Value(hidden_dims=(16, 16), num_ensembles=E, **_tt(True)).init(jax.random.PRNGKey(0), obs, act)['params']
        assert jax.tree.structure(pm['two_tier_encoder']) == jax.tree.structure(params['two_tier_encoder'])


@pytest.mark.parametrize('kw,match', [
    (dict(head_arch='BroNet'), "head_arch='BroNet'"),
    (dict(head_arch='simba'), 'expected'),
    (dict(head_arch='bronet', critic_loss_type='hlgauss', q_min=-1.0, q_max=0.0), 'hlgauss'),
    (dict(head_arch='bronet', use_film=True), 'use_film=True'),
    (dict(head_arch='bronet', action_repeat=True), 'action_repeat=True'),
])
def test_value_rejects(kw, match):
    v = Value(hidden_dims=(16, 16), num_ensembles=2, **kw)
    obs, act = _value_inputs(False)
    with pytest.raises(ValueError, match=match):
        v.init(jax.random.PRNGKey(0), obs, act)


# ---------------------------------------------------------------------------
# pi05 create() + critic_loss / critic_update / update
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_ckpt(monkeypatch):
    calls = []
    monkeypatch.setattr(P, 'Pi05', lambda **kw: None)
    monkeypatch.setattr(P, 'load_pi05_params', lambda path: calls.append(path) or {})
    monkeypatch.setattr(P, 'split_params', lambda p: ({}, {'w': np.zeros(2, np.float32)}))
    ns = {'actions': {'q01': -np.ones(32) * 0.8, 'q99': np.ones(32) * 1.3},
          'state': {'q01': np.zeros(32), 'q99': np.ones(32)}}
    monkeypatch.setattr(P.T, 'load_norm_stats', lambda path, asset_id: calls.append(path) or ns)

    class _Tok:
        def __init__(self, *a):
            calls.append('tok')

        def tokenize(self, prompt):
            return np.zeros(4, np.int32), np.ones(4, bool)

    monkeypatch.setattr(P.T, 'PromptTokenizer', _Tok)
    return calls


NUM_QS, HL, H_PI, B = 4, 2, 10, 8


def _create_cfg(**kw):
    c = dict(pi05_checkpoint='/nonexistent', pi05_robot='libero', pi05_tokenizer='x', pi05_action_horizon=H_PI, horizon_length=HL,
             pi05_clip_grad_norm=1.0, pi05_lr=1e-3, pi05_weight_decay=0.0,
             q_variance_reduction=False, q_vr_num_samples=3, q_vr_reduction='mean',
             pi05_critic_pooling='all', value_hidden_dims=(16, 16), layer_norm=True, num_qs=NUM_QS,
             q_min=-100.0, q_max=0.0, critic_scheduler='constant', ppo_lr=1e-3, critic_warmup_steps=0,
             critic_decay_steps=1, critic_end_value=0.0, clip_grad_norm=1.0, critic_weight_decay=0.0,
             pi05_critic_arch='bronet', pi05_bronet_hidden_dim=HID, pi05_bronet_depth=2,
             q_agg='mean', discount=0.99, tau=0.1, actor_tau=0.1,
             subsample_bon=True, pi05_bon_sampler='sde', pi05_bon_agg='subsample', use_value_fn=False)
    c.update(kw)
    return c


@pytest.mark.parametrize('arch', ['BroNet', 'simba', None, ''])
def test_create_rejects_unknown_arch_before_loading(fake_ckpt, arch):
    with pytest.raises(ValueError, match='pi05_critic_arch'):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch=arch))
    assert fake_ckpt == []


@pytest.mark.parametrize('arch', ['mlp', 'bronet'])
def test_create_builds_selected_critic(fake_ckpt, arch, capsys):
    agent = OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch=arch))
    pc = agent.critic_network.params['modules_critic']
    vn = pc['value_net']
    if arch == 'bronet':
        assert set(vn) == {f'Dense_{i}' for i in range(6)} | {f'LayerNorm_{i}' for i in range(5)}
        assert vn['Dense_0']['kernel'].shape == (NUM_QS, 16 + HL * T.ENV_ACTION_DIM, HID)
    else:
        assert set(vn) == {'Dense_0', 'Dense_1', 'Dense_2', 'LayerNorm_0', 'LayerNorm_1'}
        assert vn['Dense_0']['kernel'].shape == (NUM_QS, 16 + HL * T.ENV_ACTION_DIM, 16)
    assert 'two_tier_encoder' in pc
    for a, b in zip(jax.tree.leaves(pc), jax.tree.leaves(agent.critic_network.params['modules_target_critic'])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    n = sum(x.size for x in jax.tree.leaves(pc))
    assert f"[pi05] critic {arch} x {NUM_QS} heads: {n / 1e6:.1f}M params" in capsys.readouterr().out


class _FakePi05(OGPOPi05Agent):
    """Real critic_loss / _critic_step / update / critic_update; fake pi05 prefix, sampler and actor loss.

    images[:, 0, 0, 0, 0] carries a per-row id, so the sampled next action depends on the row's data,
    not its position (grad accumulation slices rows)."""

    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        rid = images[:, 0, 0, 0, 0].astype(jnp.float32)
        kv = (jnp.broadcast_to(rid[None, :, None, None, None], (1, rid.shape[0], 1, 1, 1)),) * 2
        return None, kv, jnp.ones((rid.shape[0], 1), bool)

    def _sample_sde(self, params, kv_cache, prefix_mask, rng):
        rid = kv_cache[0][0, :, 0, 0, 0]
        base = jnp.sin(jnp.arange(self._flat_dim, dtype=jnp.float32)[None] * 0.37 + rid[:, None])
        noise = 0.0 if self.config.get('_det', False) else 0.2 * jax.random.normal(rng, base.shape)
        return 0.8 * base + noise + 0.01 * jnp.sum(params['w']), None, None, None

    def actor_loss(self, batch, bc_batch, grad_params, rng, ppo_batch=None):
        loss = jnp.sum((grad_params['w'] - 1.0) ** 2) + jnp.mean(bc_batch['actions']) * jnp.sum(grad_params['w'])
        return loss, {'pg_loss': loss}


def _fake_agent(fake_ckpt_unused=None, **kw):
    real = OGPOPi05Agent.create(0, ['task'], _create_cfg(**kw))
    return _FakePi05(**{f: getattr(real, f) for f in real.__dataclass_fields__})


def _pi_batch(seed=1):
    r = np.random.RandomState(seed)
    feat = POOLING_SEGMENTS['all'] * PREFIX_DIM
    img = np.zeros((B, 2, 2, 2, 3), np.float32)
    img[:, 0] = np.arange(B)[:, None, None, None]
    return {
        'next_images': jnp.asarray(img),
        'observations': jnp.asarray(r.randn(B, feat + STATE_DIM), jnp.float32),
        'next_observations': jnp.asarray(r.randn(B, feat + STATE_DIM), jnp.float32),
        'actions': jnp.asarray(r.uniform(-1, 1, (B, HL, T.ENV_ACTION_DIM)), jnp.float32),
        'rewards': jnp.asarray(-r.rand(B, HL), jnp.float32),
        'masks': jnp.asarray((r.rand(B, HL) > 0.3).astype(np.float32)),
        'valid': jnp.asarray(np.ones((B, HL), np.float32)),
    }


def _changed(a, b):
    return any(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


@pytest.mark.parametrize('vr', [False, True])
def test_pi05_bronet_critic_loss_finite_and_grads(fake_ckpt, vr):
    agent = _fake_agent(q_variance_reduction=vr)
    batch = _pi_batch()
    gp = agent.critic_network.params
    loss, info = agent.critic_loss(batch, gp, jax.random.PRNGKey(0))
    assert np.isfinite(float(loss)) and float(loss) > 0
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values())
    assert float(info['q_head_std']) > 0                                       # independent heads
    g = jax.grad(lambda p: agent.critic_loss(batch, p, jax.random.PRNGKey(0))[0])(gp)
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree.leaves(g['modules_target_critic']))
    gv = _flat(g['modules_critic']['value_net'])
    assert all(float(jnp.abs(x).sum()) > 0 for k, x in gv.items() if k.endswith('kernel')), \
        [k for k, x in gv.items() if float(jnp.abs(x).sum()) == 0]
    assert sum(float(jnp.abs(x).sum()) for x in jax.tree.leaves(g['modules_critic']['two_tier_encoder'])) > 0


def test_pi05_bronet_critic_update_and_update(fake_ckpt):
    agent = _fake_agent()
    batch, bs = _pi_batch(1), _pi_batch(2)
    bc = {'actions': jnp.ones((B, HL, T.ENV_ACTION_DIM))}
    flag = jnp.asarray(False)
    old = agent.critic_network.params
    crit, cinfo = agent.critic_update(batch, bs, flag)
    assert all(np.isfinite(np.asarray(v)).all() for v in cinfo.values())
    assert _changed(crit.critic_network.params['modules_critic'], old['modules_critic'])
    # target = tau * new + (1 - tau) * old target (old target == old online at create)
    tau = agent.config['tau']
    exp_t = jax.tree.map(lambda p, tp: p * tau + tp * (1 - tau), crit.critic_network.params['modules_critic'],
                         old['modules_target_critic'])
    for a, b in zip(jax.tree.leaves(crit.critic_network.params['modules_target_critic']), jax.tree.leaves(exp_t)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)
    full, finfo = agent.update(batch, bs, bc, flag)
    assert all(np.isfinite(np.asarray(v)).all() for v in finfo.values())
    for a, b in zip(jax.tree.leaves(full.critic_network.params), jax.tree.leaves(crit.critic_network.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))           # critic half == critic_update
    assert _changed(full.actor_params, agent.actor_params)
    # several steps reduce the TD loss on a fixed batch
    a, losses = agent, []
    for _ in range(15):
        a, inf = a.critic_update(batch, bs, flag)
        losses.append(float(inf['critic/td_loss']))
    assert losses[-1] < losses[0], losses


def test_pi05_bronet_grad_accum_matches_full_batch(fake_ckpt):
    a1 = _fake_agent(_det=True)
    a2 = _fake_agent(_det=True, grad_accum_steps=2)
    batch, bs = _pi_batch(1), _pi_batch(2)
    bc = {'actions': jnp.ones((B, HL, T.ENV_ACTION_DIM))}
    flag = jnp.asarray(False)
    n1, i1 = a1.update(batch, bs, bc, flag)
    n2, i2 = a2.update(batch, bs, bc, flag)
    assert all(np.isfinite(np.asarray(v)).all() for v in i2.values())
    np.testing.assert_allclose(float(i1['critic/td_loss']), float(i2['critic/td_loss']), rtol=1e-5)
    np.testing.assert_allclose(float(i1['critic/grad/norm']), float(i2['critic/grad/norm']), rtol=1e-4)
    for a, b in zip(jax.tree.leaves(n1.critic_network.params), jax.tree.leaves(n2.critic_network.params)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=2e-5)
    assert _changed(n2.critic_network.params['modules_critic'], a2.critic_network.params['modules_critic'])
    c2, _ = a2.critic_update(batch, bs, flag)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(c2.critic_network.params))


def test_yaml_and_script_keys():
    from ogpo.utils.config_loader import load_config
    a = load_config('ogpo', {})['agent']
    assert a['pi05_critic_arch'] == 'mlp' and a['pi05_bronet_hidden_dim'] == 1024 and a['pi05_bronet_depth'] == 2
    sh = open(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'ogpo_pi05', 'molmo_mt16.sh')).read()
    for k in ('pi05_critic_arch', 'pi05_bronet_hidden_dim', 'pi05_bronet_depth'):
        assert f'--agent.{k}=${k}' in sh
    assert 'pi05_critic_arch=bronet_vla' in sh  # the recipe overrides the YAML's mlp default


# ---------------------------------------------------------------------------
# pi05_critic_arch: bronet_vla (docs/changes/2026-10-08-pi05-bronet-vla-critic)
# ---------------------------------------------------------------------------

ACT_PI = HL * T.ENV_ACTION_DIM
FEAT_ALL = POOLING_SEGMENTS['all'] * PREFIX_DIM


# Verbatim copy of vla-post-training/src/rl/networks/bronet_critic.py `_obs_vector_keys`, `_obs_input_dim`,
# `BroNetStateActionCritic` (nnx), with PREFIX_EMBEDDING_NAME inlined ("prefix_embedding",
# src/rl/prefix_embedding.py:4). Only change: `pytree=False` on the class (flax 0.12 vs VLA's 0.10.6, see above).
def _vla_critic_class():
    nnx, BroNetVLA = _vla_classes()
    PREFIX_EMBEDDING_NAME = "prefix_embedding"
    BroNet = BroNetVLA

    def _obs_vector_keys(observation):
        keys = []
        if isinstance(observation, dict) and PREFIX_EMBEDDING_NAME in observation:
            keys.append(PREFIX_EMBEDDING_NAME)
        keys.append("state")
        return keys

    def _obs_input_dim(observation, keys):
        return sum(int(jnp.asarray(observation[k]).shape[-1]) for k in keys)

    class BroNetStateActionCritic(nnx.Module, pytree=False):
        def __init__(self, observation, action, hidden_dim: int, depth: int, num_qs: int, num_bins: int = 1,
                     *, rngs: nnx.Rngs):
            self.num_bins = num_bins
            out_dim = max(1, num_bins)
            self._obs_keys = _obs_vector_keys(observation)
            input_dim = (
                _obs_input_dim(observation, self._obs_keys)
                + int(jnp.asarray(action).shape[-1])
            )
            self.nets = [
                BroNet(input_dim, hidden_dim, depth, output_nodes=out_dim, add_final_layer=True, rngs=rngs)
                for _ in range(num_qs)
            ]

        def __call__(self, observation, action, training: bool = False):
            parts = [jnp.asarray(observation[k]) for k in self._obs_keys] + [action]
            x = jnp.concatenate(parts, axis=-1)
            outs = [net(x, training=training) for net in self.nets]  # each (B, out_dim)
            if self.num_bins <= 1:
                outs = [o.squeeze(-1) for o in outs]  # (B,)
            return jnp.stack(outs, axis=0)  # (num_qs, B) or (num_qs, B, num_bins)

    return nnx, BroNetStateActionCritic


def _vla_net_to_linen(net, depth):
    lin = lambda m: {'kernel': m.kernel.value, 'bias': m.bias.value}
    ln = lambda m: {'scale': m.scale.value, 'bias': m.bias.value}
    p = {'Dense_0': lin(net.proj), 'LayerNorm_0': ln(net.proj_norm)}
    for i, blk in enumerate(net.blocks):
        p[f'Dense_{2 * i + 1}'], p[f'LayerNorm_{2 * i + 1}'] = lin(blk.linear1), ln(blk.norm1)
        p[f'Dense_{2 * i + 2}'], p[f'LayerNorm_{2 * i + 2}'] = lin(blk.linear2), ln(blk.norm2)
    p[f'Dense_{2 * depth + 1}'] = lin(net.final)
    return p


class _FakePrefixPi05(OGPOPi05Agent):
    """`images` *is* the (B, S, 2048) prefix hidden state; mask zeroes the last 3 tokens (VLA's 'all' ignores it)."""

    def _prefix(self, images, task_ids=None, kv_only=False):  # kv_only: 2026-10-26 pi05_drop_empty_camera signature
        m = jnp.ones(images.shape[:2], bool).at[:, -3:].set(False)
        return images, None, m


def test_create_bronet_vla_tree(fake_ckpt, capsys):
    agent = OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch='bronet_vla', pi05_critic_pooling='all'))
    pc = agent.critic_network.params['modules_critic']
    assert set(pc) == {'value_net'}                                          # no two_tier_encoder
    assert set(agent.critic_network.params['modules_target_critic']) == {'value_net'}
    vn = pc['value_net']
    assert set(vn) == {f'Dense_{i}' for i in range(6)} | {f'LayerNorm_{i}' for i in range(5)}
    assert vn['Dense_0']['kernel'].shape == (NUM_QS, PREFIX_DIM + STATE_DIM + ACT_PI, HID)
    assert PREFIX_DIM + STATE_DIM + ACT_PI == 2048 + 8 + 14
    assert vn['Dense_5']['kernel'].shape == (NUM_QS, HID, 1)
    # heads independently initialised, each orthogonal(sqrt 2), zero biases
    for name, leaf in _flat(vn).items():
        assert leaf.shape[0] == NUM_QS, name
        if name.endswith('kernel'):
            for i in range(NUM_QS):
                for j in range(i + 1, NUM_QS):
                    assert not np.allclose(np.asarray(leaf[i]), np.asarray(leaf[j])), name
                W = np.asarray(leaf[i], np.float64)
                G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
                np.testing.assert_allclose(G, 2.0 * np.eye(G.shape[0]), atol=1e-4)
        elif name.endswith('bias'):
            np.testing.assert_array_equal(np.asarray(leaf), 0.0)
    for a, b in zip(jax.tree.leaves(pc), jax.tree.leaves(agent.critic_network.params['modules_target_critic'])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    n = sum(x.size for x in jax.tree.leaves(pc))
    d_in = PREFIX_DIM + STATE_DIM + ACT_PI
    assert n == NUM_QS * (d_in * HID + HID + 5 * 2 * HID + 4 * (HID * HID + HID) + HID + 1)
    assert f"[pi05] critic bronet_vla x {NUM_QS} heads: {n / 1e6:.1f}M params" in capsys.readouterr().out


def test_bronet_vla_ignores_mlp_and_two_tier_keys(fake_ckpt):
    a = OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch='bronet_vla'))
    b = OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch='bronet_vla', value_hidden_dims=(64, 64, 64),
                                                      layer_norm=False, two_tier_fused_dim=7))
    for x, y in zip(jax.tree.leaves(a.critic_network.params), jax.tree.leaves(b.critic_network.params)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize('depth', [1, 2])
def test_bronet_vla_matches_vla_state_action_critic(fake_ckpt, depth):
    """encode() ('all' pooling) + our bronet_vla critic == VLA prefix mean + verbatim BroNetStateActionCritic."""
    nnx, VLACritic = _vla_critic_class()
    real = OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch='bronet_vla', pi05_bronet_depth=depth))
    agent = _FakePrefixPi05(**{f: getattr(real, f) for f in real.__dataclass_fields__})
    r = np.random.RandomState(7)
    S = 11
    prefix_out = jnp.asarray(r.randn(B, S, PREFIX_DIM), jnp.float32)
    states = jnp.asarray(r.uniform(-0.5, 1.5, (B, STATE_DIM)), jnp.float32)
    act = jnp.asarray(r.uniform(-1, 1, (B, ACT_PI)), jnp.float32)
    critic_obs = agent.encode(prefix_out, states)
    assert critic_obs.shape == (B, PREFIX_DIM + STATE_DIM)
    # VLA prefix: unmasked mean over every prefix token (best_of_n_learner.py:233-234)
    vla_prefix = jnp.mean(prefix_out.reshape((B, -1, PREFIX_DIM)), axis=1)
    np.testing.assert_allclose(np.asarray(critic_obs[:, :PREFIX_DIM]), np.asarray(vla_prefix), rtol=1e-6, atol=1e-6)
    vla_obs = {'prefix_embedding': critic_obs[:, :PREFIX_DIM], 'state': critic_obs[:, PREFIX_DIM:]}

    vla = VLACritic({'prefix_embedding': jnp.zeros((1, PREFIX_DIM)), 'state': jnp.zeros((1, STATE_DIM))},
                    jnp.zeros((1, ACT_PI)), HID, depth, NUM_QS, rngs=nnx.Rngs(0))
    assert len(vla.nets) == NUM_QS
    # randomise biases / LN params so an ordering or LN mismatch is visible
    for h, net in enumerate(vla.nets):
        ks = iter(jax.random.split(jax.random.PRNGKey(100 + h), 64))
        for m in [net.proj, net.proj_norm, net.final] + [x for blk in net.blocks
                                                         for x in (blk.linear1, blk.norm1, blk.linear2, blk.norm2)]:
            m.bias.value = 0.3 * jax.random.normal(next(ks), m.bias.value.shape)
            if hasattr(m, 'scale'):
                m.scale.value = 1.0 + 0.3 * jax.random.normal(next(ks), m.scale.value.shape)
    heads = [_vla_net_to_linen(net, depth) for net in vla.nets]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *heads)
    ours = agent.critic_network.params['modules_critic']['value_net']
    assert jax.tree.structure(stacked) == jax.tree.structure(ours)
    for a, b in zip(jax.tree.leaves(stacked), jax.tree.leaves(ours)):
        assert a.shape == b.shape
    params = dict(agent.critic_network.params)
    params['modules_critic'] = {'value_net': stacked}
    q = agent.critic_network.select('critic')(critic_obs, act, params=params, is_encoded=True)
    q_vla = vla(vla_obs, act)
    assert q.shape == q_vla.shape == (NUM_QS, B)
    np.testing.assert_allclose(np.asarray(q), np.asarray(q_vla), rtol=1e-5, atol=1e-5)
    assert float(jnp.std(q, axis=0).mean()) > 1e-3                           # heads differ
    # teeth: swapping the concat order (action first) must change the output
    q_bad = vla({'prefix_embedding': act, 'state': critic_obs}, jnp.zeros((B, 0)))
    assert not np.allclose(np.asarray(q), np.asarray(q_bad), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('pooling', ['segmented', 'per_camera'])
def test_create_rejects_bronet_vla_non_all_pooling(fake_ckpt, pooling):
    with pytest.raises(ValueError, match='pi05_critic_pooling=all'):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch='bronet_vla', pi05_critic_pooling=pooling))
    assert fake_ckpt == []


@pytest.mark.parametrize('arch', ['bronet-vla', 'BRONET_VLA', 'vla'])
def test_create_rejects_unknown_vla_spellings(fake_ckpt, arch):
    with pytest.raises(ValueError, match='expected mlp \\| bronet \\| bronet_vla'):
        OGPOPi05Agent.create(0, ['task'], _create_cfg(pi05_critic_arch=arch))
    assert fake_ckpt == []


@pytest.mark.parametrize('vr', [False, True])
def test_pi05_bronet_vla_critic_loss_grads_every_head(fake_ckpt, vr):
    agent = _fake_agent(pi05_critic_arch='bronet_vla', q_variance_reduction=vr)
    batch = _pi_batch()
    gp = agent.critic_network.params
    loss, info = agent.critic_loss(batch, gp, jax.random.PRNGKey(0))
    assert np.isfinite(float(loss)) and float(loss) > 0
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values())
    assert float(info['q_head_std']) > 0
    g = jax.grad(lambda p: agent.critic_loss(batch, p, jax.random.PRNGKey(0))[0])(gp)
    assert set(g['modules_critic']) == {'value_net'}
    assert all(float(jnp.abs(x).max()) == 0.0 for x in jax.tree.leaves(g['modules_target_critic']))
    for k, x in _flat(g['modules_critic']['value_net']).items():
        assert np.isfinite(np.asarray(x)).all(), k
        if k.endswith('kernel'):
            for h in range(NUM_QS):
                assert float(jnp.abs(x[h]).sum()) > 0, (k, h)
    # prefix columns of the first kernel receive gradient (raw prefix reaches every head)
    g0 = g['modules_critic']['value_net']['Dense_0']['kernel']
    for h in range(NUM_QS):
        assert float(jnp.abs(g0[h, :PREFIX_DIM]).sum()) > 0
        assert float(jnp.abs(g0[h, PREFIX_DIM:PREFIX_DIM + STATE_DIM]).sum()) > 0
        assert float(jnp.abs(g0[h, PREFIX_DIM + STATE_DIM:]).sum()) > 0


def test_pi05_bronet_vla_critic_update(fake_ckpt):
    agent = _fake_agent(pi05_critic_arch='bronet_vla')
    batch, bs = _pi_batch(1), _pi_batch(2)
    flag = jnp.asarray(False)
    old = agent.critic_network.params
    crit, cinfo = agent.critic_update(batch, bs, flag)
    assert all(np.isfinite(np.asarray(v)).all() for v in cinfo.values())
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(crit.critic_network.params))
    assert _changed(crit.critic_network.params['modules_critic'], old['modules_critic'])
    tau = agent.config['tau']
    exp_t = jax.tree.map(lambda p, tp: p * tau + tp * (1 - tau), crit.critic_network.params['modules_critic'],
                         old['modules_target_critic'])
    for a, b in zip(jax.tree.leaves(crit.critic_network.params['modules_target_critic']), jax.tree.leaves(exp_t)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7)
    a, losses = agent, []
    for _ in range(15):
        a, inf = a.critic_update(batch, bs, flag)
        losses.append(float(inf['critic/td_loss']))
    assert all(np.isfinite(losses)) and losses[-1] < losses[0], losses


@pytest.mark.parametrize('arch', ['mlp', 'bronet'])
@pytest.mark.parametrize('pooling', ['all', 'per_camera'])
def test_create_mlp_bronet_unchanged_vs_pre_change(fake_ckpt, arch, pooling):
    """create()'s critic for mlp / bronet == the pre-change construction (verbatim Value kwargs, same rng split)."""
    import copy
    from ogpo.agents.modules.flax_utils import ModuleDict
    c = _create_cfg(pi05_critic_arch=arch, pi05_critic_pooling=pooling)
    agent = OGPOPi05Agent.create(0, ['task'], c)
    rng = jax.random.PRNGKey(0)
    rng, critic_rng = jax.random.split(rng)
    feat_dim = POOLING_SEGMENTS[c['pi05_critic_pooling']] * PREFIX_DIM
    critic_def = Value(   # pre-change ogpo_pi05.create, verbatim
        hidden_dims=c['value_hidden_dims'], layer_norm=c['layer_norm'], num_ensembles=c['num_qs'],
        critic_loss_type='mse', q_min=c['q_min'], q_max=c['q_max'],
        obs_two_tier=True, two_tier_img_dim=feat_dim, two_tier_proprio_dim=STATE_DIM,
        two_tier_fused_dim=c.get('two_tier_fused_dim', 0), head_arch=c['pi05_critic_arch'],
        bronet_hidden_dim=c['pi05_bronet_hidden_dim'], bronet_depth=c['pi05_bronet_depth'])
    critic_net_def = ModuleDict({'critic': critic_def, 'target_critic': copy.deepcopy(critic_def)})
    ex = dict(observations=jnp.zeros((1, feat_dim + STATE_DIM)), actions=jnp.zeros((1, ACT_PI)))
    old = critic_net_def.init(critic_rng, critic=ex, target_critic=ex)['params']
    new = agent.critic_network.params
    assert jax.tree.structure(old['modules_critic']) == jax.tree.structure(new['modules_critic'])
    for a, b in zip(jax.tree.leaves(old['modules_critic']), jax.tree.leaves(new['modules_critic'])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    obs = jnp.asarray(np.random.RandomState(3).randn(B, feat_dim + STATE_DIM), jnp.float32)
    act = jnp.asarray(np.random.RandomState(4).randn(B, ACT_PI), jnp.float32)
    q_old = critic_net_def.apply({'params': old}, obs, act, name='critic')
    q_new = agent.critic_network.select('critic')(obs, act, is_encoded=True)
    np.testing.assert_array_equal(np.asarray(q_old), np.asarray(q_new))
