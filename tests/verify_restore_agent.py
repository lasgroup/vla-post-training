"""Verifier (2026-10-27-pi05-snapshot-resume): `_resume_state` / `_restore_agent` vs a VERBATIM copy of the pre-change
`_resume_state` (git show HEAD:ogpo/runners/pi05_online_runner.py), on a small flax.struct agent with optax states, sharded
leaves (4 forced CPU devices), jax-array and int critic steps, with and without a value network.

Run:  JAX_PLATFORMS=cpu pytest tests/verify_restore_agent.py -q   (own process: it sets the CPU device count before jax loads)
"""
import os
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import shutil
import subprocess
from types import SimpleNamespace

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import ogpo.runners.pi05_online_runner as R
from ogpo.utils.datasets import ReplayBuffer
from ogpo.utils.resume import restore_tree, save_buffer_dump, truncate_csvs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _old_resume_state():
    """The pre-change _restore_net + _resume_state, exec'd verbatim from HEAD."""
    src = subprocess.run(["git", "-C", REPO, "show", "HEAD:ogpo/runners/pi05_online_runner.py"], capture_output=True,
                         text=True, check=True).stdout.replace("\r", "")
    lo = src.index("def _restore_net(")
    hi = src.index("def save_checkpoint(")
    ns = dict(jax=jax, np=np, os=os, shutil=shutil, restore_tree=restore_tree, truncate_csvs=truncate_csvs)
    exec(compile(src[lo:hi], "HEAD:pi05_online_runner.py", "exec"), ns)
    return ns["_resume_state"]


@flax.struct.dataclass
class Net:
    params: dict
    opt_state: object
    step: object


@flax.struct.dataclass
class Agent:
    rng: jax.Array
    actor_params: dict
    target_actor_params: dict
    actor_opt_state: object
    critic_network: Net
    value_network: object
    untouched: jax.Array


def _agent(with_value, array_step):
    devs = np.array(jax.devices()[:4])
    assert len(devs) == 4, jax.devices()
    mesh = Mesh(devs, ("data",))
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("data"))
    k = jax.random.PRNGKey(0)
    actor = {"w": jax.device_put(jax.random.normal(k, (8, 4), jnp.float32), shard),
             "b": jax.device_put(jnp.zeros(4, jnp.bfloat16), rep)}
    critic = {"q": jax.device_put(jax.random.normal(k, (4, 3), jnp.float32), rep)}
    tx = optax.adamw(1e-3)

    def net(p):
        return Net(params=p, opt_state=tx.init(p), step=jnp.array(0, jnp.int32) if array_step else 0)

    return Agent(rng=k, actor_params=actor, target_actor_params=jax.tree.map(jnp.copy, actor), actor_opt_state=tx.init(actor),
                 critic_network=net(critic), value_network=net({"v": critic["q"] * 2}) if with_value else None,
                 untouched=jnp.arange(3))


def _state(agent, step):
    """What save_checkpoint / _agent_state would write after training: numpy leaves, perturbed, float64 where float."""
    st = R._agent_state(agent, step)

    def pert(x):
        x = np.asarray(x)
        return (x.astype(np.float64) + 1.5) if np.issubdtype(x.dtype, np.floating) else (x + 7)

    out = jax.tree.map(pert, {k: v for k, v in st.items() if k != "step"})
    out["step"] = step
    if not isinstance(agent.critic_network.step, jax.Array):
        out["critic_step"] = 123
        if agent.value_network is not None:
            out["value_step"] = 456
    return out


class FakeStore:
    def __init__(self, tmp, state):
        self.local = str(tmp / "local")
        os.makedirs(os.path.join(self.local, "fetched"), exist_ok=True)
        self.state, self.csv = state, tmp / "csvsrc"
        self.csv.mkdir(exist_ok=True)
        (self.csv / "env.csv").write_text("success,step\n0.1,0\n0.2,100\n0.3,200\n")
        (self.csv / "eval.csv").write_text("success,step\n0.1,0\n0.2,100\n0.3,150\n")

    def fetch(self, man):
        return self.state, str(self.csv), []

    def load_buffer(self, rb, dirs):
        return {"episodes": 11, "env_steps": 222}


def _cmp_trees(a, b):
    la, ta = jax.tree.flatten(a)
    lb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(la, lb):
        assert type(x) is type(y)
        if isinstance(x, jax.Array):
            assert x.dtype == y.dtype and x.shape == y.shape
            assert x.sharding == y.sharding, (x.sharding, y.sharding)
            assert np.array_equal(np.asarray(x, np.float64 if x.dtype == jnp.bfloat16 else x.dtype),
                                  np.asarray(y, np.float64 if y.dtype == jnp.bfloat16 else y.dtype))
        else:
            assert x == y


@pytest.mark.parametrize("with_value", [False, True])
@pytest.mark.parametrize("array_step", [False, True])
def test_resume_state_matches_pre_change(tmp_path, with_value, array_step):
    old = _old_resume_state()
    agent = _agent(with_value, array_step)
    state = _state(agent, 100)
    man = {"round": 1}
    before = np.asarray(agent.actor_params["w"]).copy()  # the new restore deletes `agent`'s buffers (restore_trees): read first
    before_v = np.asarray(agent.value_network.params["v"]).copy() if with_value else None
    res = []
    for tag, fn in (("old", old), ("new", R._resume_state)):
        rb = SimpleNamespace(size=5, numpy_rng=None)
        a, step, eps, envs = fn(FakeStore(tmp_path / tag, state), man, agent, rb, str(tmp_path / tag / "save"), 100, 3)
        csvs = {n: open(tmp_path / tag / "save" / n).read() for n in ("env.csv", "eval.csv")}
        res.append((a, step, eps, envs, rb.numpy_rng.integers(1 << 30, size=4).tolist(), csvs))
    (a0, *r0), (a1, *r1) = res
    assert r0 == r1
    _cmp_trees(a0, a1)
    # sanity: the restore actually changed the restored leaves and left the others alone
    assert not np.array_equal(np.asarray(a1.actor_params["w"]), before)
    assert a1.rng is agent.rng and a1.untouched is agent.untouched
    if with_value:
        assert not np.array_equal(np.asarray(a1.value_network.params["v"]), before_v)


@pytest.mark.parametrize("with_value", [False, True])
def test_snapshot_restore_matches_pre_change_restore(tmp_path, with_value):
    """_resume_snapshot's agent == the pre-change ResumeStore restore of the same state."""
    old = _old_resume_state()
    agent = _agent(with_value, True)
    upr = 50
    state = _state(agent, 2 * upr)
    a_old, *_ = old(FakeStore(tmp_path / "o", state), {"round": 2}, agent, SimpleNamespace(size=0, numpy_rng=None),
                    str(tmp_path / "o" / "save"), upr, 3)
    ex = dict(observations=np.zeros(3, np.float32), terminals=np.array(0.0, np.float32), task_ids=np.array(0, np.int32))
    rb = ReplayBuffer.create(ex, 20, numpy_rng=np.random.default_rng(0))
    for i in range(25):
        rb.add_transition(dict(observations=np.full(3, i, np.float32), terminals=np.array(float(i % 4 == 3), np.float32), task_ids=np.array(i, np.int32), is_success=1.0))
    snap = tmp_path / "snap"
    save_buffer_dump(rb, str(snap / "buffer"), dict(round=2, episodes=4, env_steps=40))
    with open(snap / "agent.pkl", "wb") as f:
        pickle.dump(state, f)
    (snap / "csv").mkdir()
    (snap / "csv" / "env.csv").write_text("success,step\n0.1,0\n0.2,50\n0.3,100\n")
    rb2 = ReplayBuffer.create(ex, 20, numpy_rng=np.random.default_rng(0))
    a_new, step, eps, envs = R._resume_snapshot(str(snap), agent, rb2, str(tmp_path / "save"), upr, 3)
    assert (step, eps, envs) == (100, 4, 40)
    _cmp_trees(a_old, a_new)
    assert (rb2.pointer, rb2.size, rb2.total_added) == (rb.pointer, rb.size, rb.total_added)
    assert rb2.numpy_rng.integers(1 << 30) == np.random.default_rng(3 + 30000 + 100).integers(1 << 30)
    assert open(tmp_path / "save" / "env.csv").read().splitlines() == ["success,step", "0.1,0", "0.2,50", "0.3,100"]
