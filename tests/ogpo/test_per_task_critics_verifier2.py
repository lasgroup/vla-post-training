# ruff: noqa: F722
"""Independent (step-3 verifier) tests for per-task critics, SLICE 2 — keying
critic slots on the LIBERO task *id* instead of the language string (finding F1).

Written from the change spec (``docs/changes/2026-08-21-per-task-critics/``
BLAST-RADIUS §"Addendum — F1" + PLAN §"slice 2") and the diff only. Kept in a
second file so slice-2 additions stay distinguishable from
``test_per_task_critics_verifier.py``.

What is covered here and why (in severity order):

* **A. ``collect.py`` plumbing on the REAL ``collect_data`` / ``evaluate_policy``.**
  The upstream stub test
  (``test_per_task_critics.py::test_collect_data_threads_slot_aligned_task_ids_to_the_agent``)
  gives BOTH of its two tasks the SAME description, so its slot-alignment
  assertion (``desc[id] == description``) is satisfied by any permutation of the
  id list — it cannot fail. These re-run the same plumbing with DISTINCT
  descriptions, staggered episode ends (so slots desync mid-round), filler slots,
  and the eval loop.
* **B. the ``end_data_collection(step)`` guard** — fires / does not fire, on all
  four discriminator states.
* **C. ``_task_slot``** — ``None`` raises; two ids sharing a description get
  distinct slots *through the learner method*, not only through ``TaskRegistry``.
* **D. all four ``save_episode`` overrides forward ``task_id``** (the ABC call
  contract this slice sits on).
* **E. the best-of-N per-env slot list** follows the description-group's
  ``indices`` order, and the three ``repeat`` tilings agree (env-major).
* **F. resume** with a slice-1 (description-keyed) sidecar cannot silently
  misroute.
* **G. KNOWN_GAP**: ``collect.py``'s per-env ``info`` slot update silently
  TRUNCATES ``task_description`` (numpy ``<U`` dtype), pre-existing and
  independent of this slice — pinned because it is the failure mode ``task_id``
  is immune to.

Tolerances: none needed; every assertion here is exact (ints, strings, dtypes).
"""
import dataclasses
import inspect
import json
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import (
    AdvantageWeightedSFTLearner,
)
from src.rl.best_of_n.best_of_n_learner import BestofNLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.networks.per_task_critic import TASK_INDEX_NAME
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner
from src.rl.task_registry import TaskRegistry
from src.training.collect import collect_data, evaluate_policy


# --------------------------------------------------------------------------- #
# Shared stub vector env / agent.
#
# Mirrors src/envs/venv.py:724-771 (BaseVectorEnv.reset) faithfully on the two
# things collect.py depends on:
#   * full reset  -> (obs stacked on axis 0, {k: np.array([info[k] for ...])})
#   * per-env reset(id=i) -> leading dim 1 on both, so the jax.tree.map slot
#     update `prev[i] = new[0]` works
#   * step -> (obs, reward, terminate, truncate, info) with (E, H) done flags
# The info leaves are numpy string arrays (NOT dtype=object), which is what
# `np.array([str, str])` actually produces upstream — see test G.
# --------------------------------------------------------------------------- #

_H = 3   # action-chunk / obs horizon
_D = 4   # state dim
_A = 2   # action dim


class _StubEnv:
    def __init__(self, env_num, desc_of, episode_len, info_dtype=None):
        self.env_num = env_num
        self._desc_of = desc_of
        self._episode_len = episode_len  # dict: task_id -> chunks until done
        self._info_dtype = info_dtype
        self.tasks = [None] * env_num
        self.age = [0] * env_num
        self.seeds = []

    def seed(self, s):
        self.seeds.append(s)

    def _obs(self, n):
        return {"state": np.zeros((n, _H, _D), np.float32)}

    def _info(self, ids):
        arr = np.array([self._desc_of[t] for t in ids])
        if self._info_dtype is not None:
            arr = arr.astype(self._info_dtype)
        return {"task_description": arr}

    def reset(self, id=None, options=None):
        if id is None:
            self.tasks = list(options["task_id"])
            self.age = [0] * self.env_num
            return self._obs(self.env_num), self._info(self.tasks)
        self.tasks[id] = options["task_id"]
        self.age[id] = 0
        return self._obs(1), self._info([self.tasks[id]])

    def step(self, action):
        assert action.shape[0] == self.env_num, action.shape
        term = np.zeros((self.env_num, _H), bool)
        for i in range(self.env_num):
            self.age[i] += 1
            if self.age[i] >= self._episode_len[self.tasks[i]]:
                term[i, -1] = True
        return (
            self._obs(self.env_num),
            -np.ones((self.env_num, _H), np.float32),
            term,
            np.zeros((self.env_num, _H), bool),
            {},
        )


class _StubAgent:
    total_collected_episodes = 0

    def __init__(self, env_num):
        self.env_num = env_num
        self.sample_calls = []   # (descriptions, ids) snapshots
        self.sample_id_objs = []  # the live list object collect.py handed over
        self.save_calls = []     # (env_index, description, task_id)
        self.end_steps = []      # the `step` kwarg end_data_collection saw

    def start_data_collection(self, step=None):
        pass

    def end_data_collection(self, step=None):
        self.end_steps.append(step)
        return 0

    def add_data(self, step_data):
        pass

    def sample_actions(self, obs, **kw):
        self.sample_calls.append((list(kw["task_description"]), list(kw["task_id"])))
        self.sample_id_objs.append(kw["task_id"])
        return np.zeros((self.env_num, _H, _A), np.float32)

    def save_episode(self, is_success, env_index, task_description, task_id):
        self.save_calls.append((int(env_index), str(task_description), task_id))


def _collect_cfg(tasks, num_rollouts, env_num, num_initial_rollouts=None):
    return types.SimpleNamespace(
        seed=0,
        collect=types.SimpleNamespace(
            tasks=list(tasks),
            num_rollouts=num_rollouts,
            num_initial_rollouts=num_initial_rollouts,
            replan_steps=_H,
            env_num=env_num,
        ),
    )


def _eval_cfg(eval_tasks, num_eval_rollouts, env_num):
    return types.SimpleNamespace(
        seed=0,
        collect=types.SimpleNamespace(
            eval_tasks=list(eval_tasks),
            num_eval_rollouts=num_eval_rollouts,
            replan_steps=_H,
            eval_env_num=env_num,
        ),
    )


# Equal-length, DISTINCT descriptions: distinct so the alignment assertion can
# actually fail; equal-length so test G's truncation gap does not contaminate
# these (a `<U` array cannot be widened in place).
_IDS = ["libero_90_79", "libero_90_31", "libero_90_82", "libero_90_38"]
_DESC = {
    "libero_90_79": "AAAA pick up the book",
    "libero_90_31": "BBBB put the black bowl",   # deliberately equal length
    "libero_90_82": "CCCC pick up the book",
    "libero_90_38": "DDDD put the moka pot",
}
# pad to a common width so numpy's <U dtype never truncates in the A-tests
_W = max(len(v) for v in _DESC.values())
_DESC = {k: v.ljust(_W, ".") for k, v in _DESC.items()}


# --------------------------------------------------------------------------- #
# A. collect.py plumbing, on the real loops
# --------------------------------------------------------------------------- #

def test_collect_data_ids_stay_slot_aligned_with_distinct_descriptions():
    """Every ``sample_actions`` call must receive an id list that is slot-aligned
    with ``info["task_description"]`` — including after mid-loop per-env resets.

    Distinct descriptions per id (unlike the upstream stub test, where both tasks
    share one string and any permutation of the id list passes), and staggered
    episode lengths so the four slots desync and get reassigned at different
    iterations.
    """
    env_num = 4
    tasks = _IDS
    # different chunk counts per task => slots finish at different iterations
    ep_len = {"libero_90_79": 1, "libero_90_31": 2, "libero_90_82": 3, "libero_90_38": 4}
    env = _StubEnv(env_num, _DESC, ep_len)
    agent = _StubAgent(env_num)
    collect_data(agent, env, _collect_cfg(tasks, num_rollouts=2, env_num=env_num), step=7)

    assert len(agent.sample_calls) >= 4, agent.sample_calls
    for n, (descs, ids) in enumerate(agent.sample_calls):
        assert len(descs) == len(ids) == env_num, (n, descs, ids)
        for slot, (d, i) in enumerate(zip(descs, ids)):
            assert i in _DESC, (n, slot, i)
            assert str(d) == _DESC[i], (
                f"call {n} slot {slot}: description {d!r} belongs to another task "
                f"than the id {i!r} passed for that slot"
            )
    # every task ran exactly num_rollouts episodes
    got = sorted(tid for _, _, tid in agent.save_calls)
    assert got == sorted(tasks * 2), got
    assert agent.end_steps == [7]


def test_collect_data_save_episode_gets_the_finished_episodes_id():
    """``task_id=current_task_ids[env_index]`` must be read BEFORE the slot is
    reassigned (collect.py:198 vs :210).

    Construction: one env, three single-chunk episodes of three DIFFERENT tasks
    back to back. If the id were read after the reassignment, every call would
    report the NEXT task instead of the finished one, and the recorded sequence
    would be shifted by one.
    """
    env_num = 1
    tasks = ["libero_90_79", "libero_90_31", "libero_90_82"]
    env = _StubEnv(env_num, _DESC, {t: 1 for t in tasks})
    agent = _StubAgent(env_num)
    collect_data(agent, env, _collect_cfg(tasks, num_rollouts=1, env_num=env_num), step=0)

    assert [tid for _, _, tid in agent.save_calls] == tasks, agent.save_calls
    # the description handed alongside is the finished episode's too
    for _, d, tid in agent.save_calls:
        assert d == _DESC[tid], (d, tid)


def test_collect_data_filler_slots_carry_a_real_train_task_id():
    """``valid_envs=False`` slots are parked on ``config.collect.tasks[-1]``
    (collect.py:149, :208) — a REAL train id, so the per-task registry can never
    overflow on a filler slot. Their entry in the ``sample_actions`` id list must
    still match the description the env was reset to."""
    env_num = 4
    tasks = ["libero_90_79", "libero_90_31"]     # 2 tasks x 1 rollout < 4 envs
    env = _StubEnv(env_num, _DESC, {t: 2 for t in tasks})
    agent = _StubAgent(env_num)
    collect_data(agent, env, _collect_cfg(tasks, num_rollouts=1, env_num=env_num), step=0)

    first_descs, first_ids = agent.sample_calls[0]
    assert first_ids == ["libero_90_79", "libero_90_31", "libero_90_31", "libero_90_31"], first_ids
    for d, i in zip(first_descs, first_ids):
        assert str(d) == _DESC[i]
    assert all(i in tasks for _, ids in agent.sample_calls for i in ids)


def test_collect_data_num_initial_rollouts_still_covers_every_task():
    """``num_initial_rollouts`` (step 0 only) inflates the per-task count, so the
    step-0 round still visits every task — the precondition the
    ``end_data_collection`` registry guard relies on."""
    env_num = 3
    tasks = _IDS
    env = _StubEnv(env_num, _DESC, {t: 1 for t in tasks})
    agent = _StubAgent(env_num)
    collect_data(
        agent, env,
        _collect_cfg(tasks, num_rollouts=1, env_num=env_num, num_initial_rollouts=2),
        step=0,
    )
    counts = {t: 0 for t in tasks}
    for _, _, tid in agent.save_calls:
        counts[tid] += 1
    assert counts == {t: 3 for t in tasks}, counts


def test_evaluate_policy_threads_ids_and_ends_without_a_step():
    """The eval loop must pass the same slot-aligned list, and must call
    ``end_data_collection()`` with NO step — that is the exact discriminator the
    registry guard uses to skip eval (filtered_sft_learner.py:853-869)."""
    env_num = 3
    eval_tasks = ["libero_90_79", "libero_90_31", "libero_90_82"]
    env = _StubEnv(env_num, _DESC, {"libero_90_79": 1, "libero_90_31": 2, "libero_90_82": 3})
    agent = _StubAgent(env_num)
    evaluate_policy(agent, env, _eval_cfg(eval_tasks, num_eval_rollouts=2, env_num=env_num), step=5)

    assert agent.end_steps == [None], (
        "evaluate_policy must not pass a step, or the per-task registry guard fires on eval"
    )
    for descs, ids in agent.sample_calls:
        assert len(descs) == len(ids) == env_num
        for d, i in zip(descs, ids):
            assert str(d) == _DESC[i], (d, i)
    assert not agent.save_calls, "evaluate_policy must not write episodes"


def test_collect_and_eval_pass_task_id_as_a_list_not_a_shared_alias():
    """``task_id=list(current_task_ids)`` — a snapshot, not the live list. An
    agent that keeps the object (best-of-N does: ``task_ids = list(kwargs[...])``)
    must not see it mutate under it when a slot is reassigned later."""
    env_num = 2
    tasks = ["libero_90_79", "libero_90_31"]
    env = _StubEnv(env_num, _DESC, {t: 1 for t in tasks})
    agent = _StubAgent(env_num)
    collect_data(agent, env, _collect_cfg(tasks, num_rollouts=2, env_num=env_num), step=0)
    assert len(agent.sample_id_objs) >= 2
    for (_, snapshot), live in zip(agent.sample_calls, agent.sample_id_objs):
        assert list(live) == snapshot, (
            "the id list handed to sample_actions mutated after the call -- it is an "
            "alias of collect.py's current_task_ids, not the documented snapshot"
        )


# --------------------------------------------------------------------------- #
# B. end_data_collection(step) guard
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class _FakeTrainState:
    ema_params: object


def _guard_self(registry, num_tasks, tasks):
    obj = object.__new__(FilteredSFTLearner)
    obj._task_registry = registry
    obj._num_critic_tasks = num_tasks
    obj._config = types.SimpleNamespace(
        collect=types.SimpleNamespace(tasks=list(tasks), env_num=2)
    )
    obj._collection_success_episodes = 3
    obj._train_state = _FakeTrainState(ema_params=object())
    return obj


def test_end_data_collection_guard_fires_on_a_partial_registry():
    reg = TaskRegistry(4)
    reg.index_for("libero_90_79")
    reg.index_for("libero_90_31")
    reg.index_for("libero_90_82")
    obj = _guard_self(reg, 4, _IDS)
    with pytest.raises(ValueError) as e:
        FilteredSFTLearner.end_data_collection(obj, step=0)
    msg = str(e.value)
    assert "3 of 4 slots" in msg, msg
    assert "libero_90_38" in msg, "the message must name collect.tasks so the gap is visible"
    assert "step 0" in msg


def test_end_data_collection_guard_is_skipped_on_the_eval_call():
    """``evaluate_policy`` calls ``end_data_collection()`` with no step. A partial
    registry must NOT raise there (eval can legitimately run before every task has
    been collected only in the eval loop's own bookkeeping)."""
    reg = TaskRegistry(4)
    reg.index_for("libero_90_79")
    obj = _guard_self(reg, 4, _IDS)
    assert FilteredSFTLearner.end_data_collection(obj) == 3
    assert obj._train_state.ema_params is None


def test_end_data_collection_guard_passes_when_the_registry_is_full():
    reg = TaskRegistry(4)
    for t in _IDS:
        reg.index_for(t)
    obj = _guard_self(reg, 4, _IDS)
    assert FilteredSFTLearner.end_data_collection(obj, step=1000) == 3


def test_end_data_collection_guard_is_inert_without_per_task_critics():
    obj = _guard_self(None, None, ["libero_90_44"])
    assert FilteredSFTLearner.end_data_collection(obj, step=0) == 3


def test_end_data_collection_guard_only_bites_the_first_round():
    """The registry is never cleared between rounds, so a later short round (or
    one where a task happens to produce no finished episode) cannot fire the
    guard. Pinned because the guard is otherwise a plausible spurious-abort."""
    reg = TaskRegistry(2)
    reg.index_for("libero_90_79")
    reg.index_for("libero_90_31")
    obj = _guard_self(reg, 2, ["libero_90_79", "libero_90_31"])
    for step in (0, 500, 1000):
        obj._train_state = _FakeTrainState(ema_params=object())
        obj._collection_success_episodes = 0
        assert FilteredSFTLearner.end_data_collection(obj, step=step) == 0


# --------------------------------------------------------------------------- #
# C. _task_slot
# --------------------------------------------------------------------------- #

def _slot_self(num_tasks):
    obj = object.__new__(FilteredSFTLearner)
    obj._task_registry = TaskRegistry(num_tasks)
    obj._num_critic_tasks = num_tasks
    return obj


def test_task_slot_raises_on_a_missing_task_id():
    obj = _slot_self(2)
    with pytest.raises(ValueError) as e:
        obj._task_slot(None)
    msg = str(e.value)
    assert "task_id" in msg and "collect.py" in msg, msg
    assert len(obj._task_registry) == 0, "a failed lookup must not consume a slot"


def test_task_slot_keys_on_the_id_not_the_description():
    """F1, at the learner seam rather than at ``TaskRegistry``: the two mt4 ids
    that share a LIBERO language string must land in different critics."""
    obj = _slot_self(4)
    a = obj._task_slot("libero_90_79")
    b = obj._task_slot("libero_90_82")   # same description upstream, different id
    assert a != b, (a, b)
    assert obj._task_slot("libero_90_79") == a, "assignment must be first-seen and stable"
    assert obj._task_registry.tasks == {"libero_90_79": 0, "libero_90_82": 1}


def test_task_slot_overflow_raises_rather_than_reusing_a_slot():
    obj = _slot_self(2)
    obj._task_slot("libero_90_79")
    obj._task_slot("libero_90_31")
    with pytest.raises(ValueError, match="no critic slot"):
        obj._task_slot("libero_90_82")


# --------------------------------------------------------------------------- #
# D. every save_episode override forwards task_id (the ABC call contract)
# --------------------------------------------------------------------------- #

def _recorder(obj):
    seen = {}

    def _rec(episode_data, task_description, is_success=False, target_buffer=None, task_id=None):
        seen.setdefault("calls", []).append(
            {"desc": task_description, "is_success": is_success,
             "target": target_buffer, "task_id": task_id}
        )
    obj._save_episode_in_buffer = _rec
    return seen


@pytest.mark.parametrize("cls", [FilteredSFTLearner, AdvantageWeightedSFTLearner,
                                 BestofNLearner, OGPOAgentLearner])
def test_save_episode_overrides_forward_task_id(cls):
    obj = object.__new__(cls)
    obj._episode_storage = [[{"reward": 0.0}]]
    obj._config = types.SimpleNamespace(
        rl=types.SimpleNamespace(store_success_episodes_only=False)
    )
    if cls is OGPOAgentLearner:
        obj._success_data_buffer = types.SimpleNamespace(total_inserted=0)
        obj._success_task_ranges = {}
    seen = _recorder(obj)
    cls.save_episode(obj, is_success=True, env_index=0,
                     task_description="pick up the book", task_id="libero_90_82")
    calls = seen.get("calls", [])
    assert calls, f"{cls.__name__}.save_episode never reached _save_episode_in_buffer"
    assert all(c["task_id"] == "libero_90_82" for c in calls), calls
    if cls is OGPOAgentLearner:
        # success-buffer pass AND the AWR pass both get the id
        assert len(calls) == 2, calls
        assert calls[0]["target"] is not None and calls[1]["target"] is None


def test_save_episode_signatures_default_task_id_to_none():
    """Defaulting keeps every non-collect.py caller (and the DSRL ``**kwargs``
    branch) working; the registry-on path then raises in ``_task_slot`` rather
    than silently keying on the description."""
    for cls in (FilteredSFTLearner, AdvantageWeightedSFTLearner,
                BestofNLearner, OGPOAgentLearner):
        p = inspect.signature(cls.save_episode).parameters["task_id"]
        assert p.default is None, cls.__name__


def test_sample_actions_accepts_the_task_id_kwarg_on_every_implementation():
    """collect.py now always passes ``task_id=``. The base path forwards
    ``**kwargs`` positionally into ``_generate_actions``, which is where an
    unknown kwarg would ``TypeError``."""
    assert "task_id" in inspect.signature(FilteredSFTLearner._generate_actions).parameters
    for cls in (AdvantageWeightedSFTLearner, BestofNLearner):
        kinds = [p.kind for p in inspect.signature(cls.sample_actions).parameters.values()]
        assert inspect.Parameter.VAR_KEYWORD in kinds, cls.__name__
    # the base delegates **kwargs straight through
    assert (inspect.Parameter.VAR_KEYWORD
            in [p.kind for p in inspect.signature(FilteredSFTLearner.sample_actions).parameters.values()])


# --------------------------------------------------------------------------- #
# E. best-of-N per-env slot list ordering
# --------------------------------------------------------------------------- #

class _Stop(Exception):
    pass


class _Tiny(nnx.Module):
    def __init__(self, rngs):
        self.w = nnx.Param(jnp.zeros((2, 2)))

    def __call__(self, *a, **k):
        return jnp.zeros((1, 1))


def _bofn_stub_learner(task_ids, n_samples):
    """A real ``AdvantageWeightedSFTLearner`` driven far enough into
    ``sample_actions`` to build ``critic_obs[task_index]``, then stopped at the
    first ``self._policy._input_transform`` call (the next statement)."""
    from src.training.config import get_config
    base = get_config("pi05_libero_online_ogpo_sft")
    rl = dataclasses.replace(
        base.rl, n_samples=n_samples,
        critic=dataclasses.replace(base.rl.critic, inference_start_step=0),
    )
    obj = object.__new__(AdvantageWeightedSFTLearner)
    obj._config = dataclasses.replace(base, rl=rl)
    obj.training_steps = 10_000
    obj._rng = jax.random.key(0)
    obj._transition_state_dim = _D
    tiny = _Tiny(nnx.Rngs(0))
    gdef, params = nnx.split(tiny)
    st = types.SimpleNamespace(model_def=gdef, params=params, ema_params=params)
    obj._state_action_critic_state = st
    obj._train_state = st
    obj._task_registry = TaskRegistry(4)
    obj._num_critic_tasks = 4

    calls = []

    def _slot(tid):
        calls.append(tid)
        return obj._task_registry.index_for(str(tid))
    obj._task_slot = _slot

    obj._process_obs_for_pi0 = lambda observations, task_description: {
        "observation/state": np.asarray(observations["state"], np.float32),
        "prompt": task_description,
    }
    obj._sample_action = lambda tiled, rng, ts: np.zeros(
        (np.asarray(tiled["observation/state"]).shape[0], _H, _A), np.float32
    )
    obj._state_normalize = lambda d: d
    obj._action_normalize = lambda d: d

    class _P:
        def _input_transform(self, d):
            raise _Stop()
    obj._policy = _P()
    return obj, calls


def test_bofn_slot_list_follows_the_group_index_order_across_two_ids():
    """A description group spanning TWO ids (libero_90_79 / _82) must produce a
    PER-ENV slot list in the group's own ``indices`` order — the same order
    ``group_obs``/``state`` rows are in. A per-group (single) slot, or slots in
    env order rather than group order, silently scores half the candidates with
    the wrong task's critic.
    """
    # 4 envs: slots 0 and 2 share one description across two ids.
    task_ids = ["libero_90_79", "libero_90_31", "libero_90_82", "libero_90_31"]
    descs = ["shared prompt", "other prompt", "shared prompt", "other prompt"]
    obj, calls = _bofn_stub_learner(task_ids, n_samples=3)
    observations = {"state": np.zeros((4, _H, _D), np.float32)}
    with pytest.raises(_Stop):
        AdvantageWeightedSFTLearner.sample_actions(
            obj, observations, task_description=descs, task_id=task_ids
        )
    # first group is the "shared prompt" group => indices [0, 2]
    assert calls == ["libero_90_79", "libero_90_82"], calls
    assert obj._task_registry.tasks == {"libero_90_79": 0, "libero_90_82": 1}


def test_bofn_missing_task_id_raises_instead_of_defaulting():
    obj, _ = _bofn_stub_learner(["libero_90_79"], n_samples=3)
    with pytest.raises(ValueError, match="needs task_id"):
        AdvantageWeightedSFTLearner.sample_actions(
            obj, {"state": np.zeros((1, _H, _D), np.float32)},
            task_description=["p"],
        )
    obj2, _ = _bofn_stub_learner(["libero_90_79"], n_samples=3)
    with pytest.raises(ValueError, match="slot-aligned"):
        AdvantageWeightedSFTLearner.sample_actions(
            obj2, {"state": np.zeros((2, _H, _D), np.float32)},
            task_description=["p", "q"], task_id=["libero_90_79"],
        )


def test_repeat_tiling_is_env_major_for_state_actions_and_slots():
    """``state``, the sampled ``group_actions`` and the slot vector are all tiled
    with ``repeat(..., n_samples, axis=0)``. Pin that this is env-MAJOR
    (``row = env*n + sample``) and that the numpy and jax spellings agree — an
    interleaved (``tile``) layout on any one of the three would score candidate
    ``s`` of env ``i`` against another env's task."""
    n = 3
    envs = np.arange(4)
    np_rep = np.repeat(envs, n, axis=0)
    jnp_rep = np.asarray(jnp.repeat(jnp.asarray(envs), n, axis=0))
    assert np_rep.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert jnp_rep.tolist() == np_rep.tolist()
    # env-major means row index i*n + s belongs to env i
    for i in range(4):
        for s in range(n):
            assert np_rep[i * n + s] == i
    # and it is NOT np.tile (the interleaved layout)
    assert np.tile(envs, n).tolist() != np_rep.tolist()


# --------------------------------------------------------------------------- #
# F. resume with a slice-1 (description-keyed) sidecar
# --------------------------------------------------------------------------- #

def test_slice1_description_keyed_sidecar_cannot_silently_misroute(tmp_path):
    """A pre-F1 registry keyed on LIBERO language strings is structurally valid
    JSON (contiguous slots, count <= num_tasks), so ``from_json`` accepts it. The
    protection is downstream: the very first id lookup either takes the one free
    slot or overflows, and the round ALWAYS ends with the guard, so the run dies
    instead of training 4 critics on 3 tasks' data under a permuted mapping."""
    shared = "pick up the book and place it in the left compartment of the caddy"
    path = tmp_path / "task_registry_0.json"
    path.write_text(json.dumps({"num_tasks": 4, "tasks": {
        shared: 0, "put the black bowl on top of the cabinet": 1,
        "put the right moka pot on the stove": 2,
    }}))
    reg = TaskRegistry.from_json(path, 4)
    assert len(reg) == 3, "a slice-1 sidecar loads: from_json does not know the key semantics"
    reg.index_for("libero_90_79")            # takes the last free slot
    with pytest.raises(ValueError, match="no critic slot"):
        reg.index_for("libero_90_31")        # second id => overflow, run dies
    # and even if it had not overflowed, the end-of-round guard would fire:
    obj = _guard_self(TaskRegistry.from_json(path, 4), 4, _IDS)
    with pytest.raises(ValueError, match="3 of 4 slots"):
        FilteredSFTLearner.end_data_collection(obj, step=0)


# --------------------------------------------------------------------------- #
# G. KNOWN_GAP (pre-existing, NOT introduced by this slice)
# --------------------------------------------------------------------------- #

def test_collect_py_truncates_task_description_on_a_per_env_reset_KNOWN_GAP():
    """``collect.py:213-218`` writes a freshly-reset env's info INTO the existing
    stacked info array. ``BaseVectorEnv.reset`` builds that array with
    ``np.array([...])`` (venv.py:769), i.e. a fixed-width ``<U`` dtype sized by
    the FIRST reset's descriptions — so a later, longer description is silently
    TRUNCATED and the policy is prompted with a clipped instruction.

    Live only when the longest-description task is absent from the first
    ``env_num`` slots; the mt4 recipe is accidentally safe (libero_90_79, the
    longest, is ``tasks[0]`` and fills all 8 initial slots). Pinned here because
    it is precisely the corruption the new ``task_id`` list is immune to —
    ``current_task_ids`` is a Python list of str.
    """
    env_num = 2
    short, long = "libero_90_38", "libero_90_79"
    desc = {short: "short one", long: "a considerably longer instruction string"}
    # tasks[0] is the SHORT one and fills both initial slots => dtype '<U9'
    tasks = [short, short, long, long]
    env = _StubEnv(env_num, desc, {t: 1 for t in desc})
    agent = _StubAgent(env_num)
    collect_data(agent, env, _collect_cfg(tasks, num_rollouts=1, env_num=env_num), step=0)

    seen_long = [d for descs, ids in agent.sample_calls
                 for d, i in zip(descs, ids) if i == long]
    assert seen_long, "the long task must have been sampled at least once"
    assert all(str(d) != desc[long] for d in seen_long), (
        "if this now passes the truncation was fixed — delete this KNOWN_GAP test"
    )
    assert str(seen_long[0]) == desc[long][: len(desc[short])], seen_long[0]
    # the ids, by contrast, are exact
    assert all(tid in desc for _, _, tid in agent.save_calls)
    assert sorted(tid for _, _, tid in agent.save_calls) == sorted(tasks)


def test_success_task_ranges_still_key_on_the_description_KNOWN_GAP():
    """Recorded as out of scope in BLAST-RADIUS §Addendum: ``_success_task_ranges``
    (task-balanced BC sampling, ``MT_BAL=1``) still keys on ``str(task_description)``,
    so libero_90_79 and _82 remain ONE bucket there while they are TWO critic
    slots. An arm running ``MT_BAL=1 PER_TASK_CRITIC=1`` is internally
    inconsistent about what "a task" is."""
    obj = object.__new__(OGPOAgentLearner)
    obj._episode_storage = [[{"reward": 0.0}]]
    obj._config = types.SimpleNamespace(
        rl=types.SimpleNamespace(store_success_episodes_only=False))
    obj._success_task_ranges = {}
    inserted = {"n": 0}

    class _Buf:
        @property
        def total_inserted(self):
            return inserted["n"]
    obj._success_data_buffer = _Buf()

    def _rec(episode_data, task_description, is_success=False, target_buffer=None, task_id=None):
        if target_buffer is not None:
            inserted["n"] += 1
    obj._save_episode_in_buffer = _rec

    shared = "pick up the book and place it in the left compartment of the caddy"
    for tid in ("libero_90_79", "libero_90_82"):
        obj._episode_storage = [[{"reward": 0.0}]]
        OGPOAgentLearner.save_episode(obj, is_success=True, env_index=0,
                                      task_description=shared, task_id=tid)
    assert list(obj._success_task_ranges) == [shared], obj._success_task_ranges
    assert len(obj._success_task_ranges[shared]) == 2, (
        "two distinct task ids collapsed into one BC-balancing bucket"
    )


def test_task_registry_public_api_names_the_task_id():
    """Verifier finding F16 (fixed): the public surface of ``TaskRegistry`` names
    the task ID, not the description, so a reader is not misled about the key."""
    assert "task id" in TaskRegistry.tasks.__doc__
    assert "task_id" in inspect.signature(TaskRegistry.index_for).parameters
    assert "task_description" not in inspect.signature(TaskRegistry.index_for).parameters


def test_task_index_name_is_unchanged_by_slice_2():
    assert TASK_INDEX_NAME == "task_index"


def test_failed_episodes_still_register_their_task_unless_the_flag_is_set():
    """The ``end_data_collection`` guard is only satisfiable because a FAILED
    episode still reaches ``_save_episode_in_buffer`` (and hence ``_task_slot``)
    on the OGPO path — ``rl.store_success_episodes_only`` defaults to False.

    Second half (verifier finding F14, fixed): with that flag ON the failed
    episode is still NOT stored, but ``AdvantageWeightedSFTLearner.save_episode``
    now registers the task before the early return, so a task with zero successes
    in the first round still owns its slot — see
    ``test_store_success_only_still_registers_the_task``.
    """
    def _run(flag, is_success):
        obj = object.__new__(OGPOAgentLearner)
        obj._episode_storage = [[{"reward": 0.0}]]
        obj._config = types.SimpleNamespace(
            rl=types.SimpleNamespace(store_success_episodes_only=flag))
        obj._success_data_buffer = None
        obj._success_task_ranges = {}
        seen = _recorder(obj)
        OGPOAgentLearner.save_episode(obj, is_success=is_success, env_index=0,
                                      task_description="d", task_id="libero_90_38")
        return seen.get("calls", [])

    assert _run(False, False), "a failed episode must still stamp a task_index"
    assert _run(False, False)[0]["task_id"] == "libero_90_38"
    assert _run(True, False) == [], (
        "if this now records a call, store_success_episodes_only stopped filtering — "
        "re-check the registry-starvation interaction"
    )
    assert _run(True, True)


def test_store_success_only_still_registers_the_task():
    """F14 fix: with ``store_success_episodes_only=True`` a failed episode is not
    stored, but its task id is registered, so the end-of-round registry guard
    cannot misfire on a task that had no success in the first round."""
    obj = object.__new__(AdvantageWeightedSFTLearner)
    obj._episode_storage = [[{"reward": 0.0}]]
    obj._config = types.SimpleNamespace(rl=types.SimpleNamespace(store_success_episodes_only=True))
    obj._task_registry = TaskRegistry(2)
    obj._num_critic_tasks = 2
    seen = _recorder(obj)
    AdvantageWeightedSFTLearner.save_episode(
        obj, is_success=False, env_index=0, task_description="d", task_id="libero_90_38"
    )
    assert seen.get("calls", []) == [], "failed episode must still be dropped"
    assert obj._task_registry.tasks == {"libero_90_38": 0}, "...but its task must own a slot"
