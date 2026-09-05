from types import SimpleNamespace
from typing import ClassVar

import gymnasium as gym
import numpy as np

import src.envs.libero as libero_module


class _FakeOffScreenEnv(gym.Env):
    metadata: ClassVar[dict] = {}
    instances: ClassVar[list["_FakeOffScreenEnv"]] = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.closed = False
        self.reset_calls = 0
        self.state = None
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        self.instances.append(self)

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.reset_calls += 1
        return np.zeros(1, dtype=np.float32), {}

    def set_init_state(self, state):
        self.state = np.asarray(state)
        return {"state": self.state.copy()}

    def step(self, action):
        del action
        return {}, 0.0, False, {}

    def close(self):
        self.closed = True


class _FakeSuite:
    def get_task(self, task_id):
        return SimpleNamespace(
            problem_folder=f"problem_{task_id}",
            bddl_file=f"task_{task_id}.bddl",
            language=f"task description {task_id}",
        )


def test_fixed_initial_state_is_reused_without_recreating_same_task_env(
    monkeypatch, tmp_path
):
    _FakeOffScreenEnv.instances = []
    suite = _FakeSuite()
    monkeypatch.setattr(libero_module, "OffScreenRenderEnv", _FakeOffScreenEnv)
    monkeypatch.setattr(
        libero_module.benchmark,
        "get_benchmark_dict",
        lambda: {"libero_90": lambda: suite},
    )
    monkeypatch.setattr(libero_module, "get_libero_path", lambda _: str(tmp_path))
    monkeypatch.setattr(
        libero_module,
        "get_task_init_states",
        lambda _suite, _task_id: np.asarray([[10.0], [20.0], [30.0]]),
    )

    env = libero_module.LiberoWrapper(bddl_file_name="bootstrap.bddl")
    _, info = env.reset(options={"task_id": "libero_90_7", "init_state_index": 1})
    assert info == {
        "task_description": "task description 7",
        "init_state_index": 1,
    }
    assert len(_FakeOffScreenEnv.instances) == 2
    assert _FakeOffScreenEnv.instances[0].closed
    assert _FakeOffScreenEnv.instances[1].state is not None
    assert _FakeOffScreenEnv.instances[1].state.tolist() == [20.0]

    _, info = env.reset(options={"task_id": "libero_90_7", "init_state_index": 2})
    assert info["init_state_index"] == 2
    assert len(_FakeOffScreenEnv.instances) == 2
    assert _FakeOffScreenEnv.instances[1].state is not None
    assert _FakeOffScreenEnv.instances[1].state.tolist() == [30.0]

    _, _, _, _, step_info = env.step(np.zeros(7, dtype=np.float32))
    assert step_info["init_state_index"] == 2


def test_fixed_initial_state_rejects_out_of_range_index(monkeypatch, tmp_path):
    _FakeOffScreenEnv.instances = []
    suite = _FakeSuite()
    monkeypatch.setattr(libero_module, "OffScreenRenderEnv", _FakeOffScreenEnv)
    monkeypatch.setattr(
        libero_module.benchmark,
        "get_benchmark_dict",
        lambda: {"libero_90": lambda: suite},
    )
    monkeypatch.setattr(libero_module, "get_libero_path", lambda _: str(tmp_path))
    monkeypatch.setattr(
        libero_module,
        "get_task_init_states",
        lambda _suite, _task_id: np.asarray([[10.0], [20.0]]),
    )

    env = libero_module.LiberoWrapper(bddl_file_name="bootstrap.bddl")
    try:
        env.reset(options={"task_id": "libero_90_7", "init_state_index": 2})
    except IndexError as error:
        assert "outside [0, 2)" in str(error)
    else:
        raise AssertionError("out-of-range initial-state index was accepted")
