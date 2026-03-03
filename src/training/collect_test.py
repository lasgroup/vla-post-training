import numpy as np
from unittest.mock import MagicMock
from pathlib import Path

# Ensure local `openpi` and `openpi_client` packages are importable when running tests from repo root.
_ROOT = Path(__file__).resolve().parents[2]

from collect import collect_data
from src.envs.venv import DummyVectorEnv
NUM_REPLAN_STEPS = 10

MAX_STEPS = {
    0: 80,
    1: 40,
}

## Mocking the necessary components
class MockEnv:
    def __init__(self, id: int = 0):
        # This is a dummy environment that stores transitions from all replan steps during step.
        self._step = 0
        self._max_steps = MAX_STEPS.get(id)

    def _get_obs(self):
        return {
            "observation": {
                "pi0/image": np.ones((NUM_REPLAN_STEPS, 224, 224, 3), dtype=np.uint8) * self._step,
                "pi0/state": np.ones((NUM_REPLAN_STEPS, 7)) * self._step
            },
            "action": np.zeros((NUM_REPLAN_STEPS, 7))
        }

    def reset(self, id=None):
        # Returns (obs, info)
        self._step = 0
        return self._get_obs(), {}

    def step(self, action):
        # Return next_obs, reward, terminate, truncate, info
        obs = self._get_obs()
        obs["action"] = action
        # Simulate success for the first env after one step

        terminate = np.array([False] * NUM_REPLAN_STEPS)
        truncate = np.array([False] * NUM_REPLAN_STEPS)
        self._step += NUM_REPLAN_STEPS
        if self._step >= self._max_steps:
            terminate[-1] = True
        return obs, np.zeros(NUM_REPLAN_STEPS), terminate, truncate, {}

    def close(self): pass


class MockDataSet:
    def __init__(self):
        self._num_episodes = 0
        self._frames = []

    @property
    def num_episodes(self):
        return self._num_episodes

    def add_frame(self, frame):
        self._frames.append(frame)

    def save_episode(self):
        self._num_episodes += 1

    @property
    def frames(self):
        return self._frames


def test_collect_data_integration():
    # 1. Setup Mock Config
    config = MagicMock()
    config.collect.num_rollouts = 3
    config.collect.resize_image = 0
    config.collect.replan_steps = NUM_REPLAN_STEPS

    # 2. Setup Mock Policy
    policy = MagicMock()
    # Mocking policy.infer to return an action chunk
    policy.infer.return_value = {"actions": np.zeros((2, NUM_REPLAN_STEPS, 7))}

    # 3. Setup Mock Dataset
    dataset = MockDataSet()

    # 4. Initialize Mock Env

    env = DummyVectorEnv([lambda i=i: MockEnv(id=i) for i in range(2)])
    sharding_spec = None  # Not needed for mock CPU run

    # 5. Run the collection
    metrics, num_episodes = collect_data(
        policy=policy,
        dataset=dataset,
        sharding_spec=sharding_spec,
        env=env,
        task_description="Test Task",
        config=config
    )

    # 6. Assertions
    # Environment 1 terminates after 40 steps and environment 0 after 80.
    expected_number_of_episodes = 3
    expected_number_of_transitions = 40 * 2 + 80
    assert "success_rate" in metrics
    assert metrics["success_rate"] > 0
    assert num_episodes == expected_number_of_episodes
    assert len(dataset.frames) == expected_number_of_transitions
    print("Test passed: Data collection logic executed successfully.")