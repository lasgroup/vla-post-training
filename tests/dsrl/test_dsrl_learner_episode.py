from types import SimpleNamespace

import numpy as np

from src.rl.dsrl.dsrl_learner import DSRLLearner


class _Buffer:
    def __init__(self):
        self.inserted = []

    def insert(self, data):
        self.inserted.append(data)



def _make_episode(action):
    return [
        {
            "observation": {"observation/state": np.array([0.1, 0.2], dtype=np.float32)},
            "next_observation": {"observation/state": np.array([0.3, 0.4], dtype=np.float32)},
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.array([0.0], dtype=np.float32),
            "terminate": np.array([False]),
            "truncate": np.array([False]),
        },
        {
            "observation": {"observation/state": np.array([0.5, 0.6], dtype=np.float32)},
            "next_observation": {"observation/state": np.array([0.7, 0.8], dtype=np.float32)},
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.array([0.0], dtype=np.float32),
            "terminate": np.array([False]),
            "truncate": np.array([False]),
        },
        {
            "observation": {"observation/state": np.array([0.9, 1.0], dtype=np.float32)},
            "next_observation": {"observation/state": np.array([1.1, 1.2], dtype=np.float32)},
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.array([1.0], dtype=np.float32),
            "terminate": np.array([True]),
            "truncate": np.array([False]),
        },
    ]



def _make_learner_stub(discount: float = 0.9, replan_steps: int = 2):
    learner = DSRLLearner.__new__(DSRLLearner)
    learner._sac_image_size = 0
    learner._action_dim = 3
    learner._expected_action_shape = (1, 3)
    learner._online_data_buffer = _Buffer()
    learner._episode_storage = [[]]
    learner._collection_success_episodes = 0
    learner._config = SimpleNamespace(
        rl=SimpleNamespace(discount=discount),
        collect=SimpleNamespace(replan_steps=replan_steps, env_max_reward=0.0),
    )
    return learner



def test_save_episode_preserves_sparse_reward_and_latent_actions():
    learner = _make_learner_stub(discount=0.9, replan_steps=2)
    action = np.array([[0.2, -0.3, 0.4]], dtype=np.float32)
    learner._episode_storage[0] = _make_episode(action)

    learner.save_episode(is_success=True, env_index=0)

    assert len(learner._online_data_buffer.inserted) == 3
    inserted = learner._online_data_buffer.inserted

    rewards = [float(x["reward"][0]) for x in inserted]
    discounts = [float(x["discount"][0]) for x in inserted]
    mc_returns = [float(x["mc_return"][0]) for x in inserted]

    # Sparse DSRL shaping should remain: -1, -1, 0 for successful episodes.
    assert rewards == [-1.0, -1.0, 0.0]
    # Discount = gamma^replan_steps except successful terminal transition.
    assert discounts == [0.81, 0.81, 0.0]
    # MC returns under sparse rewards and per-transition discounts.
    np.testing.assert_allclose(mc_returns, [-1.81, -1.0, 0.0], atol=1e-6)

    for transition in inserted:
        assert set(transition.keys()) == {
            "observation",
            "actions",
            "next_observation",
            "reward",
            "mc_return",
            "discount",
        }
        assert transition["actions"].shape == (1, 3)
        np.testing.assert_allclose(transition["actions"], np.array([[0.2, -0.3, 0.4]], dtype=np.float32))

    assert learner._collection_success_episodes == 1
