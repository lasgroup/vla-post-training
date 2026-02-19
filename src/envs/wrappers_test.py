import pytest
import numpy as np
import gymnasium as gym
import sys
from pathlib import Path

# Ensure local `openpi` and `openpi_client` packages are importable when running tests from repo root.
_ROOT = Path(__file__).resolve().parents[2]

# --- FIX START ---
# Add the repo root to sys.path so 'jaxrl2' can be found
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.envs.wrappers import QueryFrequencyWrapper


# Assuming your class is in the namespace or imported here
# from your_module import QueryFrequencyWrapper

class MockEnv(gym.Env):
    """A simple env that returns predictable observations and rewards."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.count = 0

    def step(self, action):
        self.count += 1
        obs = np.array([float(self.count)], dtype=np.float32)
        reward = 1.0  # Constant reward for easy math
        terminated = self.count >= 5  # Terminate at step 5
        truncated = False
        info = {"step": self.count}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.count = 0
        return np.array([0.0], dtype=np.float32), {}


def test_full_transitions_mode():
    query_freq = 3
    env = MockEnv()
    wrapped = QueryFrequencyWrapper(env, query_frequency=query_freq, store_full_transitions=True)

    # Mock action: (3, 1) array
    action = np.zeros((query_freq, 1))
    obs_act, reward, term, trunc, info = wrapped.step(action)
    obs = obs_act["observation"]

    # Check shapes: they should all have the first dimension as query_freq
    assert obs.shape == (query_freq, 1)
    assert reward.shape == (query_freq,)
    assert len(info["step"]) == query_freq
    # Check values
    np.testing.assert_array_equal(obs.flatten(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(reward, [1.0, 1.0, 1.0])


def test_discounted_summary_mode():
    query_freq = 3
    gamma = 0.9
    env = MockEnv()
    wrapped = QueryFrequencyWrapper(env, query_frequency=query_freq,
                                    discount=gamma, store_full_transitions=False)

    action = np.zeros((query_freq, 1))
    obs_act, reward, term, trunc, info = wrapped.step(action)
    obs = obs_act["observation"]

    # 1. Check reward: 1.0 + (1.0 * 0.9) + (1.0 * 0.9^2) = 1 + 0.9 + 0.81 = 2.71
    expected_reward = 1.0 + 0.9 + 0.81
    assert reward == pytest.approx(expected_reward)

    # 2. Check observation: should only be the LAST one (3.0)
    assert obs.shape == (1,)
    assert obs[0] == 3.0

    # 3. Check info: should be the last step's info
    assert info["step"] == 3


def test_early_termination():
    """Verify the wrapper handles the environment ending mid-query."""
    query_freq = 10  # Long query
    env = MockEnv()  # MockEnv ends at step 5
    wrapped = QueryFrequencyWrapper(env, query_frequency=query_freq,
                                    discount=1.0, store_full_transitions=True)

    action = np.zeros((query_freq, 1))
    obs_act, reward, term, trunc, info = wrapped.step(action)
    obs = obs_act["observation"]

    # It should have broken early at step 5
    assert len(obs) == 5
    assert term[-1] == True
    assert reward.sum() == 5.0