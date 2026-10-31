"""Verifier 2 (2026-10-27-pi05-snapshot-resume, late change): `_restore_agent` / `restore_trees` / `_resume_state` vs a VERBATIM
copy of the pre-change restore (HEAD's `restore_tree` + `_restore_net` + `_resume_state`, exec'd from `git show HEAD:...`).

Stand-in agent shaped like OGPOPi05Agent (rng, actor / target / opt state, init_actor_params, frozen_params, consts, a real
flax_utils.TrainState critic, optional value TrainState), bf16 + f32 + int32 leaves, nested dicts, a leaf object shared between a
replaced and a kept field, a leaf object appearing twice among the replaced fields, a numpy leaf in a replaced tree, Python-int and
jax-array (weak-typed) steps, replicated and axis-sharded leaves on a 2-device CPU mesh, and the single-device case.

Run in its own process (sets the CPU device count before jax loads):
  JAX_PLATFORMS=cpu pytest tests/verify_restore_trees_diff.py -q
"""
import os
import sys

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import ogpo.runners.pi05_online_runner as R
from ogpo.agents.modules.flax_utils import TrainState, nonpytree_field
from ogpo.utils import resume as RS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _head(path):
    return subprocess.run(["git", "-C", REPO, "show", f"HEAD:{path}"], capture_output=True, text=True, check=True).stdout.replace("\r", "")


def _old_fns():
    """HEAD's restore_tree, _restore_net and _resume_state, verbatim."""
    rsrc = _head("ogpo/utils/resume.py")
    ns = dict(jax=jax, np=np, os=os, shutil=shutil, truncate_csvs=RS.truncate_csvs)
    lo, hi = rsrc.index("def restore_tree("), rsrc.index("# ----------------------------------------------------------------------------- the store")
    exec(compile(rsrc[lo:hi], "HEAD:resume.py", "exec"), ns)
    src = _head("ogpo/runners/pi05_online_runner.py")
    lo, hi = src.index("def _restore_net("), src.index("def save_checkpoint(")
    exec(compile(src[lo:hi], "HEAD:pi05_online_runner.py", "exec"), ns)
    return ns


OLD = _old_fns()


def old_restore_agent(agent, state):
    """The replace-dict of HEAD's _resume_state, verbatim (lines between `state, csv_dir, ...` and `meta = ...`)."""
    restore_tree, _restore_net = OLD["restore_tree"], OLD["_restore_net"]
    replace = dict(
        actor_params=restore_tree(agent.actor_params, state["actor_params"], "actor_params"),
        target_actor_params=restore_tree(agent.target_actor_params, state["target_actor_params"], "target_actor_params"),
        actor_opt_state=restore_tree(agent.actor_opt_state, state["actor_opt_state"], "actor_opt_state"),
        critic_network=_restore_net(agent.critic_network, state["critic_params"], state["critic_opt_state"], state["critic_step"]),
    )
    if agent.value_network is not None:
        replace["value_network"] = _restore_net(agent.value_network, state["value_params"], state["value_opt_state"],
                                                state["value_step"])
    return agent.replace(**replace)


def test_head_restore_tree_is_unchanged():
    cur = RS.restore_tree.__code__.co_code
    assert cur == OLD["restore_tree"].__code__.co_code  # the live restore_tree (load paths, tests) is still HEAD's


class Agent(flax.struct.PyTreeNode):  # OGPOPi05Agent's pytree fields, same order
    rng: Any
    actor_params: Any
    target_actor_params: Any
    actor_opt_state: Any
    init_actor_params: Any
    frozen_params: Any
    critic_network: Any
    consts: Any
    config: Any = nonpytree_field()
    value_network: Any = None


CRITIC_TX = optax.adam(1e-3)


def build(mesh, with_value=True, critic_int_step=True):
    """Fresh agent; every call returns equal values in new objects. Sharing: actor['shared_k'] is init['shared_k'] (kept),
    actor['shared_f'] is frozen['f'] (kept), actor['enc']['b'] is target['enc']['b'] (both replaced)."""
    g = np.random.default_rng(0)
    if mesh is None:
        rep = shard = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    else:
        rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    put = lambda x, s=None: jax.device_put(x, s or rep)
    bf = lambda *s: put(jnp.asarray(g.normal(size=s), jnp.bfloat16))
    f32 = lambda *s, sh=None: put(jnp.asarray(g.normal(size=s), jnp.float32), sh)
    shared_k, shared_f, dup_b = f32(3), f32(2, 2), bf(6)
    actor = {"enc": {"w": bf(4, 6), "b": dup_b}, "head": {"k": f32(8, sh=shard), "cnt": put(jnp.arange(3, dtype=jnp.int32))},
             "shared_k": shared_k, "shared_f": shared_f}
    target = {"enc": {"w": bf(4, 6), "b": dup_b}, "head": {"k": f32(8, sh=shard), "cnt": put(jnp.arange(3, dtype=jnp.int32))},
              "shared_k": f32(3), "shared_f": f32(2, 2)}
    actor_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, weight_decay=1e-2))
    opt = jax.tree.map(lambda x: put(x, x.sharding) if isinstance(x, jax.Array) else x, actor_tx.init(actor))
    init = {"shared_k": shared_k, "other": f32(5)}
    frozen = {"f": shared_f, "big": {"a": bf(3, 3), "b": f32(7)}}
    consts = {"tokens": put(jnp.arange(6, dtype=jnp.int32).reshape(2, 3)), "np_const": np.arange(4, dtype=np.float32)}

    def net(step, extra_np):
        params = {"modules_critic": {"w": f32(5, 2), "ln": bf(2)}, "modules_target_critic": {"w": f32(5, 2), "ln": bf(2)}}
        tx = CRITIC_TX  # one object: TrainState.tx is a static field, part of the treedef
        o = jax.tree.map(lambda x: put(x) if isinstance(x, jax.Array) else x, tx.init(params))
        if extra_np:
            params = dict(params, np_leaf=np.arange(3, dtype=np.float32))  # a numpy leaf inside a replaced tree
        return TrainState(step=step, apply_fn=None, model_def=None, params=params, tx=tx, opt_state=o)

    critic = net(1 if critic_int_step else put(jnp.array(1, jnp.int32)), extra_np=True)
    value = net(put(1), extra_np=False) if with_value else None  # put(1): weak-typed int32, as device_put of TrainState.step=1
    return Agent(rng=put(jax.random.PRNGKey(3)), actor_params=actor, target_actor_params=target, actor_opt_state=opt,
                 init_actor_params=init, frozen_params=frozen, critic_network=critic, consts=consts, config={"x": 1},
                 value_network=value)


def make_state(agent):
    """An _agent_state-like host pytree: float leaves as float32 (bf16 casts on restore), int leaves as int64."""
    g = np.random.default_rng(1)

    def leaf(c):
        c = np.asarray(c)
        if np.issubdtype(c.dtype, np.integer):
            return g.integers(0, 100, c.shape).astype(np.int64)
        return g.normal(size=c.shape).astype(np.float32)

    st = dict(step=4000, actor_params=jax.tree.map(leaf, agent.actor_params),
              target_actor_params=jax.tree.map(leaf, agent.target_actor_params),
              actor_opt_state=jax.tree.map(leaf, agent.actor_opt_state),
              critic_params=jax.tree.map(leaf, agent.critic_network.params),
              critic_opt_state=jax.tree.map(leaf, agent.critic_network.opt_state), critic_step=np.int64(11))
    if agent.value_network is not None:
        st.update(value_params=jax.tree.map(leaf, agent.value_network.params),
                  value_opt_state=jax.tree.map(leaf, agent.value_network.opt_state), value_step=np.array(13))
    return st


def assert_same(a, b):
    la, ta = jax.tree_util.tree_flatten_with_path(a)
    lb, tb = jax.tree_util.tree_flatten_with_path(b)
    assert ta == tb
    for (pa, x), (pb, y) in zip(la, lb):
        name = jax.tree_util.keystr(pa)
        assert pa == pb
        assert isinstance(x, jax.Array) == isinstance(y, jax.Array), name
        assert type(x) is type(y) or (isinstance(x, jax.Array) and isinstance(y, jax.Array)), (name, type(x), type(y))
        if isinstance(x, (int, float)):
            assert x == y, name
            continue
        assert x.dtype == y.dtype and x.shape == y.shape, (name, x.dtype, y.dtype, x.shape, y.shape)
        if isinstance(x, jax.Array):
            assert not x.is_deleted() and not y.is_deleted(), name
            assert x.weak_type == y.weak_type, (name, x.weak_type, y.weak_type)
            assert x.sharding == y.sharding, (name, x.sharding, y.sharding)
            assert getattr(x, "_committed", None) == getattr(y, "_committed", None), name
        assert np.array_equal(np.asarray(x).reshape(-1).view(np.uint8), np.asarray(y).reshape(-1).view(np.uint8)), name
    assert a.config == b.config


MESHES = ["mesh2", "single"]


def _mesh(kind):
    if kind == "single":
        return None
    assert len(jax.devices()) >= 2, jax.devices()
    return Mesh(np.array(jax.devices()[:2]), ("d",))


@pytest.mark.parametrize("kind", MESHES)
@pytest.mark.parametrize("with_value", [False, True])
@pytest.mark.parametrize("critic_int_step", [True, False])
def test_restore_agent_equals_pre_change(kind, with_value, critic_int_step):
    mesh = _mesh(kind)
    a_old, a_new = build(mesh, with_value, critic_int_step), build(mesh, with_value, critic_int_step)
    state = make_state(a_old)
    st_copy = copy.deepcopy(state)
    ref = old_restore_agent(a_old, copy.deepcopy(state))
    kept_before = {k: getattr(a_new, k) for k in ("rng", "init_actor_params", "frozen_params", "consts")}
    kept_vals = jax.tree.map(lambda x: np.asarray(x).copy(), kept_before)
    out = R._restore_agent(a_new, state)
    assert_same(ref, out)
    # state is not mutated
    jax.tree.map(lambda x, y: np.testing.assert_array_equal(x, y), state, st_copy)
    # kept fields: the same objects, alive, original values
    for k, v in kept_before.items():
        assert getattr(out, k) is v
        for leaf, val in zip(jax.tree.leaves(v), jax.tree.leaves(kept_vals[k])):
            assert not (isinstance(leaf, jax.Array) and leaf.is_deleted())
            np.testing.assert_array_equal(np.asarray(leaf).reshape(-1).view(np.uint8), val.reshape(-1).view(np.uint8))
    # shared with a kept field: alive and untouched in the kept field, restored value in the replaced one
    assert out.init_actor_params["shared_k"] is kept_before["init_actor_params"]["shared_k"]
    np.testing.assert_array_equal(np.asarray(out.actor_params["shared_k"]), state["actor_params"]["shared_k"])
    np.testing.assert_array_equal(np.asarray(out.frozen_params["f"]), kept_vals["frozen_params"]["f"])
    # the same object twice among replaced fields: each place gets its own restored value
    np.testing.assert_array_equal(np.asarray(out.actor_params["enc"]["b"], np.float32),
                                  state["actor_params"]["enc"]["b"].astype(jnp.bfloat16).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(out.target_actor_params["enc"]["b"], np.float32),
                                  state["target_actor_params"]["enc"]["b"].astype(jnp.bfloat16).astype(np.float32))
    assert out.actor_params["enc"]["b"] is not out.target_actor_params["enc"]["b"]
    # the replaced, unshared buffers of the input agent are deleted; shared-with-kept ones are not
    assert a_new.actor_params["enc"]["w"].is_deleted() and a_new.actor_params["enc"]["b"].is_deleted()
    assert a_new.critic_network.params["modules_critic"]["w"].is_deleted()
    assert not a_new.actor_params["shared_k"].is_deleted() and not a_new.actor_params["shared_f"].is_deleted()
    assert isinstance(out.critic_network.params["np_leaf"], np.ndarray)  # numpy cur leaf stays numpy (as HEAD)
    if critic_int_step:
        assert out.critic_network.step == 11 and type(out.critic_network.step) is int


@pytest.mark.parametrize("bad", ["value_params_shape", "critic_opt_structure", "value_step_shape", "missing_value_step_array"])
def test_invalid_inputs_raise_before_any_deletion(bad):
    mesh = _mesh("mesh2")
    agent = build(mesh, with_value=True, critic_int_step=False)
    state = make_state(agent)
    if bad == "value_params_shape":  # the LAST tree: everything before it is validated and would have been placed
        state["value_params"]["modules_critic"]["w"] = np.zeros((5, 3), np.float32)
        exc = ValueError
    elif bad == "critic_opt_structure":
        state["critic_opt_state"] = {"mu": np.zeros(1)}
        exc = ValueError
    elif bad == "value_step_shape":
        state["value_step"] = np.zeros(2)
        exc = ValueError
    else:
        del state["value_step"]
        exc = KeyError
    leaves = [x for x in jax.tree.leaves(agent) if isinstance(x, jax.Array)]
    with pytest.raises(exc):
        R._restore_agent(agent, state)
    assert not any(x.is_deleted() for x in leaves)


# fixed after the first verifier run: _restore_agent reads the Python-int steps before restore_trees deletes anything
@pytest.mark.parametrize("bad", ["missing", "not_scalar"])
def test_int_step_errors_raise_before_any_deletion(bad):
    agent = build(_mesh("mesh2"), with_value=False, critic_int_step=True)
    state = make_state(agent)
    if bad == "missing":
        del state["critic_step"]
    else:
        state["critic_step"] = np.zeros(3)
    leaves = [x for x in jax.tree.leaves(agent) if isinstance(x, jax.Array)]
    with pytest.raises((KeyError, TypeError)):
        R._restore_agent(agent, state)
    assert not any(x.is_deleted() for x in leaves)


def test_restore_trees_duplicates_protect_and_numpy():
    mesh = _mesh("mesh2")
    s = NamedSharding(mesh, P())
    x, keep = jax.device_put(jnp.ones(3), s), jax.device_put(jnp.zeros(2), s)
    npl = np.ones(2, np.float32)
    out = RS.restore_trees([({"a": x, "k": keep}, {"a": np.full(3, 2.0), "k": np.full(2, 5.0)}, "t1"),
                            ([x, npl], [np.full(3, 4.0), np.full(2, 6.0)], "t2")], protect={id(keep)})
    assert x.is_deleted() and not keep.is_deleted() and np.asarray(keep).sum() == 0
    np.testing.assert_array_equal(out[0]["a"], np.full(3, 2.0, np.float32))
    np.testing.assert_array_equal(out[0]["k"], np.full(2, 5.0, np.float32))
    np.testing.assert_array_equal(out[1][0], np.full(3, 4.0, np.float32))
    assert isinstance(out[1][1], np.ndarray) and out[1][1].dtype == np.float32
    assert out[0]["a"].sharding == s and out[0]["a"].dtype == jnp.float32


@pytest.mark.parametrize("with_value", [False, True])
def test_resume_state_equals_pre_change(tmp_path, with_value):
    """The whole ResumeStore restore (`_resume_state`), new vs HEAD, with fake fetch / load_buffer."""
    mesh = _mesh("mesh2")
    a_old, a_new = build(mesh, with_value, False), build(mesh, with_value, False)
    state = make_state(a_old)
    csv = tmp_path / "csv"
    csv.mkdir()
    (csv / "env.csv").write_text("success,step\n0.1,0\n0.2,4000\n0.3,5000\n")
    (csv / "online_agent.csv").write_text("l,step\n1,3000\n2,4000\n3,4500\n")

    class Store:
        def __init__(self, local):
            self.local = str(local)
            os.makedirs(os.path.join(self.local, "fetched"))

        def fetch(self, man):
            return copy.deepcopy(state), str(csv), ["c0"]

        def load_buffer(self, rb, dirs):
            rb.size = 77
            return {"episodes": 5, "env_steps": 9}

    res = []
    for fn, agent, name in ((OLD["_resume_state"], a_old, "old"), (R._resume_state, a_new, "new")):
        rb = SimpleNamespace(size=0, numpy_rng=None)
        out = fn(Store(tmp_path / f"local_{name}"), {"round": 4}, agent, rb, str(tmp_path / f"save_{name}"), 1000, 3)
        res.append((out, rb, {n: (tmp_path / f"save_{name}" / n).read_text() for n in ("env.csv", "online_agent.csv")}))
    (o_old, rb_old, csv_old), (o_new, rb_new, csv_new) = res
    assert o_old[1:] == o_new[1:] == (4000, 5, 9)
    assert_same(o_old[0], o_new[0])
    assert rb_old.numpy_rng.bit_generator.state == rb_new.numpy_rng.bit_generator.state
    assert csv_old == csv_new
    assert not os.path.exists(tmp_path / "local_new" / "fetched")
