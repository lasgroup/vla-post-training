"""
Differential test for `agent.pi05_drop_empty_camera` (docs/changes/2026-10-26-pi05-drop-empty-camera).

Flag off vs on on the REAL pi05_droid_jointpos checkpoint (identical params: the on agent is `off.replace(config=...)`):
  1. encode() pooled critic features: bit-identical.
  2. _prefix(kv_only=True): K / V at the valid prefix positions, prefix_mask.
  3. _actor_fn velocities (actor, target, perturbed params; several x, t).
  4. sample_actions_ode / _plain / _BON (sde, ode; argmax agreement), _sample_sde + _log_prob.
  5. _bc_loss_per_sample, critic_loss (q_variance_reduction on / off), actor_loss and its gradient.
  6. (VDC_SPEED=1) kv_only prefix pass at B=64 / 256 and one full agent.update, off vs on.
  7. Pi05.embed_prefix ValueErrors (CPU, no checkpoint).

Noise floor for (2)-(5): the flag-off computation re-run with
  floor_L: 256 extra masked (padding) language tokens appended to every prompt (same semantics, S changes by +256,
           the same size of change as dropping the 256 masked right-wrist tokens), and
  floor_B: the prefix pass run on a batch padded with 5 extra rows (sliced back).
A sensitivity control (right-wrist slot UNmasked) shows the comparison would see a leak.

Run (GPU, real checkpoint): python -m pytest -s -q tests/test_pi05_drop_empty_camera.py
CPU-only part:              JAX_PLATFORMS=cpu python -m pytest -q tests/test_pi05_drop_empty_camera.py -k embed_prefix
"""
import dataclasses
import json
import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ogpo.networks.pi05.pi05 import Pi05

ROOT = os.path.join(os.path.dirname(__file__), '..')
CKPT = '/data/user_data/pchellap/vla-post-training/cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos'
FLAGS = os.path.join(ROOT, 'logs/fit_mt144/flags_s1.txt')
PROMPTS = ['put the red mug on the shelf.', 'pick up the spatula and place it in the box.', 'close the drawer.']
N_CAM_TOK = 256
PAD_L = 256
PAD_B = 5
B = 16
SPEED = os.environ.get('VDC_SPEED', '0') == '1'
CORRECTNESS = os.environ.get('VDC_CORRECTNESS', '1') == '1'
OUT = os.environ.get('VDC_OUT', os.path.join(ROOT, 'logs/verify_drop_camera/results.json'))

RESULTS = {}


def _has_gpu():
    try:
        return any(d.platform == 'gpu' for d in jax.devices())
    except RuntimeError:
        return False


gpu = pytest.mark.skipif(not (_has_gpu() and os.path.isdir(CKPT)), reason='needs a GPU and the pi05_droid_jointpos checkpoint')
corr = pytest.mark.skipif(not CORRECTNESS, reason='VDC_CORRECTNESS=0')


# ---------------------------------------------------------------------------
# 7. embed_prefix validation (CPU)
# ---------------------------------------------------------------------------

def test_embed_prefix_rejects_unknown_or_missing_base():
    m = Pi05(action_horizon=15)
    x = jnp.zeros((1, 224, 224, 3))
    tok, tm = jnp.zeros((1, 4), jnp.int32), jnp.ones((1, 4), bool)
    bad = {'base_0_rgb': x, 'left_wrist_0_rgb': x, 'bogus_rgb': x}
    with pytest.raises(ValueError, match='bogus_rgb'):
        m.apply({'params': {}}, bad, {k: jnp.ones(1, bool) for k in bad}, tok, tm, method=Pi05.embed_prefix)
    nobase = {'left_wrist_0_rgb': x, 'right_wrist_0_rgb': x}
    with pytest.raises(ValueError, match='base_0_rgb'):
        m.apply({'params': {}}, nobase, {k: jnp.ones(1, bool) for k in nobase}, tok, tm, method=Pi05.embed_prefix)
    RESULTS['7_embed_prefix_valueerror'] = 'PASS (unknown key and missing base_0_rgb both raise ValueError)'


# ---------------------------------------------------------------------------
# Construction (as ogpo/main_pi05.py)
# ---------------------------------------------------------------------------

def _agent_config(extra):
    from ogpo.utils.config_loader import config_to_agent_dict, load_config, parse_cli_args
    skip = ('--algo', '--env.name', '--buffer.storage_dir', '--experiment.', '--agent.grad_accum_steps',
            '--agent.batch_size', '--agent.ppo_batch_size', '--agent.group_num_samples', '--agent.q_vr_num_samples',
            '--agent.best_of_n', '--success_buffer.batch_size', '--success_buffer.bc_batch_size')
    flags = [l.strip() for l in open(FLAGS) if l.strip().startswith('--') and not l.strip().startswith(skip)]
    algo, cli = parse_cli_args(['--algo=ogpo', *flags, '--experiment.log=false', '--agent.grad_accum_steps=1', *extra])
    config = load_config(algo, cli)
    ac = config_to_agent_dict(config)
    for k in ('value_hidden_dims', 'actor_hidden_dims'):
        ac[k] = tuple(ac[k])
    discount = ac['discount']
    ac['q_min'], ac['q_max'] = -1.0 / (1.0 - discount), 0.0
    ac.setdefault('adv_clip_min', None)
    ac['donate_agent'] = False  # off / on agents share buffers
    return config, ac


def _with_cfg(agent, **kw):
    c = agent.config.copy()
    c.update(kw)
    return agent.replace(config=c)


def _fields(agent):
    return {f.name: getattr(agent, f.name) for f in dataclasses.fields(agent)}


def _make_classes():
    from ogpo.agents.ogpo_pi05 import OGPOPi05Agent

    class BatchPadAgent(OGPOPi05Agent):
        """Flag-off agent whose prefix pass runs on PAD_B extra rows (sliced back): noise floor."""

        pad_rows = PAD_B  # plain class attribute (not a dataclass field)

        def _prefix(self, images, task_ids=None, kv_only=False):
            b = images.shape[0]
            extra = np.asarray([(b - 1 - i) % b for i in range(type(self).pad_rows)])
            imgs = jnp.concatenate([images, images[extra][..., ::-1, :, :]], axis=0)
            tids = None if task_ids is None else jnp.concatenate([task_ids, task_ids[extra]])
            out, kv, pm = super()._prefix(imgs, tids, kv_only=kv_only)
            return (None if out is None else out[:b]), jax.tree.map(lambda x: x[:, :b], kv), pm[:b]

    class UnmaskedAgent(OGPOPi05Agent):
        """Sensitivity control: the right-wrist slot (all -1) is NOT masked."""

        def _model_images(self, images, drop_empty=False):
            imgs, masks = super()._model_images(images, drop_empty=False)
            masks['right_wrist_0_rgb'] = jnp.ones(images.shape[0], bool)
            return imgs, masks

    return BatchPadAgent, UnmaskedAgent


def _floor_agents(off):
    """Extra flag-off variants with identical semantics (noise floor): language padding +64 / +512 tokens, batch padding
    +3 / +11 rows, reversed row order inside the prefix pass."""
    BatchPadAgent, _ = _make_classes()

    class Pad3(BatchPadAgent):
        pad_rows = 3

    class Pad11(BatchPadAgent):
        pad_rows = 11

    class PermAgent(type(off)):
        def _prefix(self, images, task_ids=None, kv_only=False):
            out, kv, pm = super()._prefix(images[::-1], None if task_ids is None else task_ids[::-1], kv_only=kv_only)
            return (None if out is None else out[::-1]), jax.tree.map(lambda x: x[:, ::-1], kv), pm[::-1]

    def padL(n):
        return off.replace(consts={**off.consts,
                                   'tokens': jnp.pad(off.consts['tokens'], ((0, 0), (0, n))),
                                   'token_mask': jnp.pad(off.consts['token_mask'], ((0, 0), (0, n)))})
    f = _fields(off)
    return {'floor_L64': padL(64), 'floor_L512': padL(512), 'floor_B3': Pad3(**f), 'floor_B11': Pad11(**f),
            'floor_perm': PermAgent(**f)}


def _images(n, seed):
    """(n, 2, 224, 224, 3) uint8: even rows random noise, odd rows smooth structured gradients."""
    rng = np.random.default_rng(seed)
    out = rng.integers(0, 256, (n, 2, 224, 224, 3), dtype=np.uint8)
    yy, xx = np.meshgrid(np.linspace(0, 1, 224), np.linspace(0, 1, 224), indexing='ij')
    for i in range(1, n, 2):
        for cam in range(2):
            ph = rng.uniform(0, 2 * np.pi, 3)
            fr = rng.uniform(1, 4, 3)
            r = 0.5 + 0.5 * np.sin(2 * np.pi * fr[0] * xx + ph[0])
            g = 0.5 + 0.5 * np.cos(2 * np.pi * fr[1] * yy + ph[1])
            b = 0.5 + 0.5 * np.sin(2 * np.pi * fr[2] * (xx + yy) / 2 + ph[2])
            out[i, cam] = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    return out


def _d(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    da = np.abs(a - b)
    scale = max(float(np.abs(a).max()), 1e-30)
    rel = da / np.maximum(np.abs(a), 1e-3 * scale)
    return {'max_abs': float(da.max()), 'max_rel_to_scale': float(da.max() / scale), 'max_rel_elem': float(rel.max()),
            'mean_abs': float(da.mean()), 'frac_exact': float((da == 0).mean()), 'scale': scale}


def _cmp(name, ref, on, floors, factor=10.0, tol_abs=0.0):
    """Record off-vs-on and the floors; pass iff off-vs-on max_abs <= factor * max(floor max_abs) + tol_abs."""
    r = {'off_vs_on': _d(ref, on)}
    for k, v in floors.items():
        r[f'off_vs_{k}'] = _d(ref, v)
    floor = max(r[f'off_vs_{k}']['max_abs'] for k in floors)
    chg = r['off_vs_on']['max_abs']
    r['ratio_change_over_floor'] = (chg / floor) if floor > 0 else (0.0 if chg == 0 else float('inf'))
    r['pass'] = bool(chg <= factor * floor + tol_abs)
    RESULTS[name] = r
    print(f"[vdc] {name}: off-on max_abs {chg:.3e} (rel {r['off_vs_on']['max_rel_to_scale']:.2e}, scale {r['off_vs_on']['scale']:.3e}) | "
          + ' | '.join(f"{k} {r[f'off_vs_{k}']['max_abs']:.3e}" for k in floors)
          + f" | ratio {r['ratio_change_over_floor']:.2f} | {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    return r


@pytest.fixture(scope='module')
def env():
    from ogpo.agents.ogpo_pi05 import OGPOPi05Agent
    BatchPadAgent, UnmaskedAgent = _make_classes()
    config, ac = _agent_config(['--agent.batch_size=32', '--agent.ppo_batch_size=8', '--agent.group_num_samples=8',
                                '--agent.q_vr_num_samples=4', '--agent.best_of_n=8', '--success_buffer.batch_size=32',
                                '--success_buffer.bc_batch_size=32'])
    assert ac['pi05_drop_empty_camera'] is False  # yaml default
    base = OGPOPi05Agent.create(0, PROMPTS, ac)
    off = _with_cfg(base, pi05_drop_empty_camera=False)
    on = _with_cfg(base, pi05_drop_empty_camera=True)
    padL = off.replace(consts={**off.consts,
                               'tokens': jnp.pad(off.consts['tokens'], ((0, 0), (0, PAD_L))),
                               'token_mask': jnp.pad(off.consts['token_mask'], ((0, 0), (0, PAD_L)))})
    padB = BatchPadAgent(**_fields(off))
    unm = UnmaskedAgent(**_fields(off))
    imgs = jnp.asarray(_images(B, 1))
    rng = np.random.default_rng(2)
    states = jnp.asarray(rng.normal(size=(B, 8)).astype(np.float32) * 0.5)
    tids = jnp.asarray(rng.integers(0, len(PROMPTS), B).astype(np.int32))
    L = int(off.consts['tokens'].shape[-1])
    print(f"[vdc] devices {jax.devices()} | B={B} | prompt len L={L} | valid tokens per prompt "
          f"{np.asarray(off.consts['token_mask']).sum(-1).tolist()} | task_ids {np.asarray(tids).tolist()} | "
          f"x64={jax.config.jax_enable_x64}", flush=True)
    yield dict(cls=OGPOPi05Agent, config=config, ac=ac, off=off, on=on, padL=padL, padB=padB, unm=unm,
               imgs=imgs, states=states, tids=tids, L=L)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(RESULTS, f, indent=1)
    print(f"[vdc] results -> {OUT}", flush=True)


@partial(jax.jit, static_argnames=('kv_only',))
def _prefix_fn(agent, imgs, tids, kv_only):
    return agent._prefix(imgs, tids, kv_only=kv_only)


# ---------------------------------------------------------------------------
# 1. encode
# ---------------------------------------------------------------------------

@corr
@gpu
def test_1_encode_bit_identical(env):
    cls, off, on = env['cls'], env['off'], env['on']
    cache = getattr(cls.encode, '_cache_size', lambda: -1)
    n0 = cache()
    f_off = np.asarray(off.encode(env['imgs'], env['states'], env['tids']))
    f_on = np.asarray(on.encode(env['imgs'], env['states'], env['tids']))
    n1 = cache()
    f_off2 = np.asarray(off.encode(env['imgs'], env['states'], env['tids']))
    # the on agent's encode is its own compile (config is static), so equality is not a cache artefact
    assert n0 < 0 or n1 - n0 == 2, (n0, n1)
    out_on, _, _ = _prefix_fn(on, env['imgs'], env['tids'], False)
    assert out_on.shape[1] == 3 * N_CAM_TOK + env['L'], out_on.shape  # encode path keeps the full prefix
    eq = bool(np.array_equal(f_off, f_on))
    RESULTS['1_encode'] = {'shape': list(f_off.shape), 'bit_identical': eq, 'off_rerun_bit_identical': bool(np.array_equal(f_off, f_off2)),
                           'max_abs': float(np.abs(f_off - f_on).max()), 'separate_compiles': int(n1 - n0), 'pass': eq}
    print(f"[vdc] 1_encode: {RESULTS['1_encode']}", flush=True)
    env['feats'] = jnp.asarray(f_off)
    assert eq


# ---------------------------------------------------------------------------
# 2. KV cache
# ---------------------------------------------------------------------------

@corr
@gpu
def test_2_kv(env):
    L, imgs, tids = env['L'], env['imgs'], env['tids']
    _, kv_off, pm_off = _prefix_fn(env['off'], imgs, tids, True)
    out_none, kv_on, pm_on = _prefix_fn(env['on'], imgs, tids, True)
    _, kv_L, pm_L = _prefix_fn(env['padL'], imgs, tids, True)
    _, kv_B, pm_B = _prefix_fn(env['padB'], imgs, tids, True)
    _, kv_off_full, _ = _prefix_fn(env['off'], imgs, tids, False)
    assert out_none is None
    shapes = jax.tree.map(lambda x: tuple(x.shape), (kv_off, kv_on))
    print(f"[vdc] KV leaves (off, on): {shapes}; layout: base [0,256) left_wrist [256,512) right_wrist [512,768) "
          f"lang [768,{768 + L}) in the off prefix; on drops [512,768)", flush=True)
    S_off, S_on = 3 * N_CAM_TOK + L, 2 * N_CAM_TOK + L
    pm_off, pm_on, pm_L, pm_B = (np.asarray(x) for x in (pm_off, pm_on, pm_L, pm_B))
    assert pm_off.shape == (B, S_off) and pm_on.shape == (B, S_on)
    assert not pm_off[:, 2 * N_CAM_TOK:3 * N_CAM_TOK].any(), 'right-wrist slot not fully masked in the off prefix'
    assert pm_off[:, :2 * N_CAM_TOK].all()
    idx = np.concatenate([np.arange(2 * N_CAM_TOK), np.arange(3 * N_CAM_TOK, S_off)])
    assert np.array_equal(pm_off[:, idx], pm_on), 'prefix_mask disagrees at kept positions'
    assert np.array_equal(pm_L[:, :S_off], pm_off) and not pm_L[:, S_off:].any() and np.array_equal(pm_B, pm_off)
    same_off = all(np.array_equal(np.asarray(a), np.asarray(b))
                   for a, b in zip(jax.tree.leaves(kv_off), jax.tree.leaves(kv_off_full)))
    RESULTS['2_prefix_mask'] = {'S_off': S_off, 'S_on': S_on, 'masked_slot_positions': [512, 768],
                                'pm_agree_at_kept_positions': True, 'flag_off_kv_only_equals_full_pass': same_off,
                                'kv_shapes_off_on': str(shapes)}
    valid = pm_off[:, idx].astype(bool)  # (B, S_on) valid positions in the on layout
    langpad = ~valid
    leaves = [jax.tree.leaves(x) for x in (kv_off, kv_on, kv_L, kv_B)]
    names = ['K', 'V'] if len(leaves[0]) == 2 else [f'leaf{i}' for i in range(len(leaves[0]))]
    res = []
    for name, a_off, a_on, a_L, a_B in zip(names, *leaves):
        a_off = np.asarray(a_off.astype(jnp.float32))[:, :, idx]
        a_on = np.asarray(a_on.astype(jnp.float32))
        a_L = np.asarray(a_L.astype(jnp.float32))[:, :, idx]
        a_B = np.asarray(a_B.astype(jnp.float32))[:, :, idx]
        sel = lambda x: x[:, valid]  # (layers, n_valid, K, H)
        res.append(_cmp(f'2_kv_{name}_valid', sel(a_off), sel(a_on), {'floor_L': sel(a_L), 'floor_B': sel(a_B)}))
        # masked language-padding rows may differ (their queries attend to nothing -> uniform softmax over all S keys);
        # they are masked out of every downstream attention, so never consumed
        selp = lambda x: x[:, langpad]
        RESULTS[f'2_kv_{name}_masked_langpad_info'] = {'off_vs_on': _d(selp(a_off), selp(a_on)),
                                                       'off_vs_floor_L': _d(selp(a_off), selp(a_L))}
        print(f"[vdc] 2_kv_{name}_masked_langpad (info, not consumed): off-on max_abs "
              f"{RESULTS[f'2_kv_{name}_masked_langpad_info']['off_vs_on']['max_abs']:.3e}", flush=True)
    env['kv'] = {'off': (kv_off, pm_off), 'on': (kv_on, pm_on), 'padL': (kv_L, pm_L), 'padB': (kv_B, pm_B)}
    assert all(r['pass'] for r in res)


# ---------------------------------------------------------------------------
# 3. velocities
# ---------------------------------------------------------------------------

@jax.jit
def _vel(agent, params, kv, pm, x, t):
    return agent._actor_fn(params, kv, pm)(None, x, t)


@corr
@gpu
def test_3_velocities(env):
    kvs = env['kv']
    off = env['off']
    h, d = off._chunk_shape
    _, kv_U, pm_U = _prefix_fn(env['unm'], env['imgs'], env['tids'], True)
    leaves, tdef = jax.tree.flatten(off.actor_params)
    keys = jax.random.split(jax.random.PRNGKey(5), len(leaves))
    perturbed = jax.tree.unflatten(tdef, [p * (1 + 0.01 * jax.random.normal(k, p.shape, p.dtype)) for p, k in zip(leaves, keys)])
    params = {'actor': off.actor_params, 'target': off.target_actor_params, 'perturbed': perturbed}
    agents = {'off': env['off'], 'on': env['on'], 'padL': env['padL'], 'padB': env['padB']}
    ok = True
    for pname, p in params.items():
        for j in range(3):
            k1, k2 = jax.random.split(jax.random.PRNGKey(100 + j))
            x = jax.random.normal(k1, (B, h * d))
            t = jax.random.uniform(k2, (B, 1), minval=0.02, maxval=0.98)
            v = {k: np.asarray(_vel(agents[k], p, *kvs[k], x, t)) for k in agents}
            ok &= _cmp(f'3_vel_{pname}_draw{j}', v['off'], v['on'], {'floor_L': v['padL'], 'floor_B': v['padB']})['pass']
            if pname == 'actor' and j == 0:
                vu = np.asarray(_vel(off, p, kv_U, pm_U, x, t))
                RESULTS['3_vel_sensitivity_unmasked_right_wrist'] = _d(v['off'], vu)
                print(f"[vdc] 3 sensitivity control (right wrist UNmasked) off-vs-control max_abs "
                      f"{RESULTS['3_vel_sensitivity_unmasked_right_wrist']['max_abs']:.3e}", flush=True)
    assert ok


# ---------------------------------------------------------------------------
# 4. samples
# ---------------------------------------------------------------------------

@jax.jit
def _sde_lp(agent, imgs, tids, rng):
    _, kv, pm = agent._prefix(imgs, tids, kv_only=True)
    actions, chains, logprob, sigmas = agent._sample_sde(agent.target_actor_params, kv, pm, rng)
    lp, ent, _ = agent._log_prob(agent.actor_params, kv, pm, chains, get_entropy=True)
    return actions, chains, logprob, lp, ent


@jax.jit
def _lp_fixed(agent, imgs, tids, chains):
    _, kv, pm = agent._prefix(imgs, tids, kv_only=True)
    return agent._log_prob(agent.actor_params, kv, pm, chains, get_entropy=False)[0]


@jax.jit
def _bon_idx(agent, imgs, feats, rng, tids):
    """Copy of sample_actions_BON returning the argmax index (B,) and the candidates' Q (num_qs, B, N)."""
    from ogpo.agents.modules.q_helper import select_best_of_n
    Bn, N = imgs.shape[0], agent.config['best_of_n']
    action_rng, sub_rng = jax.random.split(rng)
    _, kv, pmask = agent._prefix(imgs, tids, kv_only=True)
    kv_n = jax.tree.map(lambda x: jnp.repeat(x, N, axis=1), kv)
    pmask_n = jnp.repeat(pmask, N, axis=0)
    if agent.config['pi05_bon_sampler'] == 'ode':
        actions = agent._sample_ode(agent.target_actor_params, kv_n, pmask_n, action_rng, clip=False)
    else:
        actions, _, _, _ = agent._sample_sde(agent.target_actor_params, kv_n, pmask_n, action_rng)
    q = agent.critic_network.select('target_critic')(
        jnp.repeat(feats, N, axis=0), agent._critic_actions(actions), is_encoded=True).reshape(agent.config['num_qs'], Bn, N)
    return select_best_of_n(q, agent.config['pi05_bon_agg'], sub_rng), q


@corr
@gpu
def test_4_samples(env):
    imgs, tids, feats = env['imgs'], env['tids'], env['feats']
    names = ('off', 'on', 'padL', 'padB')
    ag = {k: env[k] for k in names}
    ok = True
    for j, seed in enumerate((11, 12)):
        rng = jax.random.PRNGKey(seed)
        o = {k: np.asarray(ag[k].sample_actions_ode(imgs, rng, tids, feats)) for k in names}
        ok &= _cmp(f'4_sample_ode_s{j}', o['off'], o['on'], {'floor_L': o['padL'], 'floor_B': o['padB']})['pass']
        o = {k: np.asarray(ag[k].sample_actions_plain(imgs, rng, tids, feats)) for k in names}
        ok &= _cmp(f'4_sample_plain_s{j}', o['off'], o['on'], {'floor_L': o['padL'], 'floor_B': o['padB']})['pass']
    for sampler in ('sde', 'ode'):
        a2 = {k: _with_cfg(ag[k], pi05_bon_sampler=sampler) for k in names}
        rng = jax.random.PRNGKey(21)
        o = {k: np.asarray(a2[k].sample_actions_BON(imgs, feats, rng, tids)) for k in names}
        ok &= _cmp(f'4_bon_{sampler}_actions', o['off'], o['on'], {'floor_L': o['padL'], 'floor_B': o['padB']})['pass']
        res = {k: _bon_idx(a2[k], imgs, feats, rng, tids) for k in names}
        ix = {k: np.asarray(res[k][0]) for k in names}
        q = {k: np.asarray(res[k][1]) for k in names}
        RESULTS[f'4_bon_{sampler}_argmax'] = {'agree_off_on': float((ix['off'] == ix['on']).mean()),
                                              'agree_off_floor_L': float((ix['off'] == ix['padL']).mean()),
                                              'agree_off_floor_B': float((ix['off'] == ix['padB']).mean())}
        print(f"[vdc] 4_bon_{sampler}_argmax {RESULTS[f'4_bon_{sampler}_argmax']}", flush=True)
        ok &= _cmp(f'4_bon_{sampler}_candidate_q', q['off'], q['on'], {'floor_L': q['padL'], 'floor_B': q['padB']})['pass']
    rng = jax.random.PRNGKey(31)
    out = {k: [np.asarray(x) for x in _sde_lp(ag[k], imgs, tids, rng)] for k in names}
    chains_off = jnp.asarray(out['off'][1])  # log-prob of the SAME chains under each prefix (no sampling confound)
    for k in names:
        out[k].append(np.asarray(_lp_fixed(ag[k], imgs, tids, chains_off)))
    labels = ['actions', 'chains', 'old_logprob', 'logprob', 'entropy', 'logprob_of_off_chains']
    for i, lab in enumerate(labels):
        ok &= _cmp(f'4_sde_{lab}', out['off'][i], out['on'][i], {'floor_L': out['padL'][i], 'floor_B': out['padB'][i]})['pass']
    assert ok


# ---------------------------------------------------------------------------
# 5. losses / gradient on a runner-schema batch
# ---------------------------------------------------------------------------

def _batches(env, batch_size, sb_size, bc_size):
    from ogpo.runners import pi05_online_runner as R
    off, ac, config = env['off'], env['ac'], env['config']
    pool = _images(24, 7)
    rng = np.random.default_rng(3)
    states = rng.normal(size=(24, 8)).astype(np.float32) * 0.5
    ptids = rng.integers(0, len(PROMPTS), 24).astype(np.int32)
    feats = np.concatenate([np.asarray(off.encode(jnp.asarray(pool[i:i + 8]), jnp.asarray(states[i:i + 8]),
                                                  jnp.asarray(ptids[i:i + 8]))) for i in range(0, 24, 8)])
    rb = R._make_buffer(feats[0], pool[0], 4000, 0, off.env_action_dim)
    n_ep = 0
    while rb.size < 4000 - 150:
        traj = []
        start = int(rng.integers(24))
        for t in range(150):
            i, j = (start + t) % 24, (start + t + 1) % 24
            traj.append(dict(observations=feats[i], images=pool[i],
                             actions=rng.normal(size=off.env_action_dim).astype(np.float32) * 0.1,
                             rewards=np.float32(-1.0), terminals=np.float32(t == 149), masks=np.float32(t < 149),
                             next_observations=feats[j], next_images=pool[j], task_ids=np.int32(ptids[start])))
        R._flush_episode(rb, traj, {'success': float(n_ep % 2 == 0)}, ac['discount'])
        n_ep += 1
    H, disc = ac['horizon_length'], ac['discount']
    batch = rb.sample_sequence(batch_size, sequence_length=H, discount=disc)
    batch_success = R._sample_success(rb, sb_size, H, disc)
    assert batch_success is not None
    bc_h = config['buffer'].get('bc_horizon') or ac['pi05_action_horizon']
    bc_batch = R._sample_success(rb, bc_size, bc_h, disc)
    bc_batch = {k: bc_batch[k] for k in ('images', 'observations', 'actions', 'valid', 'task_ids') if k in bc_batch}
    to = lambda b: jax.tree.map(jnp.asarray, b)
    return to(batch), to(batch_success), to(bc_batch)


@jax.jit
def _losses(agent, batch, bc_batch, rng):
    k1, k2, k3 = jax.random.split(rng, 3)
    bc = agent._bc_loss_per_sample(bc_batch, agent.actor_params, k1)
    td, cinfo = agent.critic_loss(batch, agent.critic_network.params, k2)
    (al, ainfo), g = jax.value_and_grad(lambda p: agent.actor_loss(batch, bc_batch, p, k3), has_aux=True)(agent.actor_params)
    return bc, td, cinfo, al, ainfo, g


@jax.jit
def _gstats(g_ref, g):
    import optax
    dot = sum(jnp.vdot(a.astype(jnp.float32), b.astype(jnp.float32)) for a, b in zip(jax.tree.leaves(g_ref), jax.tree.leaves(g)))
    n_ref, n = optax.global_norm(g_ref), optax.global_norm(g)
    diff = optax.global_norm(jax.tree.map(lambda a, b: a - b, g_ref, g))
    return n_ref, n, dot / (n_ref * n), diff


@corr
@gpu
def test_5_losses_and_grad(env):
    batch, batch_success, bc_batch = _batches(env, 32, 32, 32)
    print(f"[vdc] 5 batch shapes: {jax.tree.map(lambda x: tuple(x.shape), batch)} | bc {jax.tree.map(lambda x: tuple(x.shape), bc_batch)}",
          flush=True)
    names = ('off', 'on', 'padL', 'padB')
    ok = True
    for qvr in (True, False):
        ag = {k: _with_cfg(env[k], q_variance_reduction=qvr) for k in names}
        rng = jax.random.PRNGKey(41)
        out = {k: _losses(ag[k], batch, bc_batch, rng) for k in names}
        if qvr:
            ok &= _cmp('5_bc_loss_per_sample', np.asarray(out['off'][0]), np.asarray(out['on'][0]),
                       {'floor_L': np.asarray(out['padL'][0]), 'floor_B': np.asarray(out['padB'][0])})['pass']
            ok &= _cmp('5_actor_loss', np.asarray(out['off'][3]), np.asarray(out['on'][3]),
                       {'floor_L': np.asarray(out['padL'][3]), 'floor_B': np.asarray(out['padB'][3])})['pass']
            for key in ('pg_loss', 'bc_loss', 'lp', 'old_lp', 'adv', 'q_values', 'adv_zero_frac'):
                ok &= _cmp(f'5_actor_info_{key}', np.asarray(out['off'][4][key]), np.asarray(out['on'][4][key]),
                           {'floor_L': np.asarray(out['padL'][4][key]), 'floor_B': np.asarray(out['padB'][4][key])},
                           tol_abs=1e-7)['pass']
            g = {}
            for k in ('on', 'padL', 'padB'):
                n_ref, n, cos, diff = (float(x) for x in _gstats(out['off'][5], out[k][5]))
                g[k] = {'norm_off': n_ref, 'norm': n, 'rel_norm_diff': abs(n - n_ref) / n_ref, 'cosine': cos,
                        'rel_diff_norm': diff / n_ref}
            floor = max(g['padL']['rel_diff_norm'], g['padB']['rel_diff_norm'])
            g['ratio_change_over_floor'] = g['on']['rel_diff_norm'] / floor if floor > 0 else float('inf')
            g['pass'] = bool(g['on']['rel_diff_norm'] <= 10 * floor + 1e-7)
            RESULTS['5_actor_grad'] = g
            print(f"[vdc] 5_actor_grad {g}", flush=True)
            ok &= g['pass']
        tag = 'qvr_on' if qvr else 'qvr_off'
        ok &= _cmp(f'5_critic_td_loss_{tag}', np.asarray(out['off'][1]), np.asarray(out['on'][1]),
                   {'floor_L': np.asarray(out['padL'][1]), 'floor_B': np.asarray(out['padB'][1])})['pass']
        for key in ('next_q_mean', 'target_q_mean', 'next_actions_std'):
            ok &= _cmp(f'5_critic_{key}_{tag}', np.asarray(out['off'][2][key]), np.asarray(out['on'][2][key]),
                       {'floor_L': np.asarray(out['padL'][2][key]), 'floor_B': np.asarray(out['padB'][2][key])})['pass']
        del out
    assert ok


# ---------------------------------------------------------------------------
# 6. speed (VDC_SPEED=1)
# ---------------------------------------------------------------------------

def _time(fn, reps):
    fn()  # compile / warm
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [round(t, 4) for t in ts]


@pytest.mark.skipif(not SPEED, reason='VDC_SPEED=1 only')
@gpu
def test_6_speed(env):
    off, on = env['off'], env['on']
    sp = {}
    for nb in ((64, 256) if os.environ.get("VDC_SKIP_PREFIX", "0") != "1" else ()):
        imgs = jnp.asarray(_images(nb, 9))
        tids = jnp.asarray(np.arange(nb, dtype=np.int32) % len(PROMPTS))
        for k, a in (('off', off), ('on', on)):
            m, ts = _time(lambda: jax.block_until_ready(_prefix_fn(a, imgs, tids, True)[1]), 5)
            sp[f'prefix_kv_only_B{nb}_{k}'] = {'median_s': m, 'reps': ts}
        sp[f'prefix_kv_only_B{nb}_speedup'] = sp[f'prefix_kv_only_B{nb}_off']['median_s'] / sp[f'prefix_kv_only_B{nb}_on']['median_s']
        print(f"[vdc] prefix B={nb}: off {sp[f'prefix_kv_only_B{nb}_off']['median_s']:.4f} s, on "
              f"{sp[f'prefix_kv_only_B{nb}_on']['median_s']:.4f} s, speedup {sp[f'prefix_kv_only_B{nb}_speedup']:.3f}x", flush=True)
        del imgs
    RESULTS['6_speed'] = sp
    bs = int(os.environ.get('VDC_UPDATE_BATCH', '64'))
    batch, batch_success, bc_batch = _batches(env, bs, bs, bs)
    cfg = dict(batch_size=bs, ppo_batch_size=int(os.environ.get('VDC_PPO', '16')),
               group_num_samples=int(os.environ.get('VDC_G', '8')), q_vr_num_samples=int(os.environ.get('VDC_QVR', '8')))
    for k, a in (('off', off), ('on', on)):
        state = {'a': _with_cfg(a, **cfg)}

        def step():
            new, _ = state['a'].update(batch, batch_success, bc_batch, True)
            jax.block_until_ready(new.actor_params)
            state['a'] = new
        m, ts = _time(step, 4)
        sp[f'update_{k}'] = {'median_s': m, 'reps': ts, **cfg, 'sb_batch': bs, 'bc_batch': bs}
        print(f"[vdc] update {k}: {m:.3f} s {ts}", flush=True)
        del state
    sp['update_speedup'] = sp['update_off']['median_s'] / sp['update_on']['median_s']
    print(f"[vdc] update ({cfg}, sb/bc {bs}): off {sp['update_off']['median_s']:.3f} s, on {sp['update_on']['median_s']:.3f} s, "
          f"speedup {sp['update_speedup']:.3f}x", flush=True)
    RESULTS['6_speed'] = sp
