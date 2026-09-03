from types import SimpleNamespace

import numpy as np
import pytest

from src.training.collect import collect_data, evaluate_policy


ENV_NUM = 8
REPLAN_STEPS = 5
ACTION_DIM = 7


def _expected_policy_actions(policy_horizon):
    env_ids = np.arange(ENV_NUM, dtype=np.float32)[:, None, None]
    timesteps = np.arange(policy_horizon, dtype=np.float32)[None, :, None]
    action_dims = np.arange(ACTION_DIM, dtype=np.float32)[None, None, :]
    return env_ids * 1_000 + timesteps * 10 + action_dims


class _FakeVectorEnv:
    env_num = ENV_NUM

    def __init__(self):
        self.action_batches = []
        self.seed_value = None

    def seed(self, seed):
        self.seed_value = seed

    @staticmethod
    def _observation(env_num):
        return {
            "observation/state": np.zeros(
                (env_num, REPLAN_STEPS, ACTION_DIM), dtype=np.float32
            )
        }

    def reset(self, id=None, options=None):
        del options
        env_num = 1 if id is not None else self.env_num
        return self._observation(env_num), {
            "task_description": np.asarray(["test task"] * env_num)
        }

    def step(self, actions):
        actions = np.asarray(actions)
        self.action_batches.append(actions.copy())
        assert actions.shape == (ENV_NUM, REPLAN_STEPS, ACTION_DIM)
        next_obs = self._observation(self.env_num)
        reward = np.zeros((ENV_NUM, REPLAN_STEPS), dtype=np.float32)
        reward[:, -1] = 1.0
        terminate = np.zeros((ENV_NUM, REPLAN_STEPS), dtype=bool)
        terminate[:, -1] = True
        truncate = np.zeros_like(terminate)
        return next_obs, reward, terminate, truncate, {}


class _FakeAgent:
    def __init__(self, *, policy_horizon, return_prefix):
        self.policy_horizon = policy_horizon
        self.return_prefix = return_prefix
        self.sample_calls = 0
        self.added_action_shapes = []
        self.added_actions = []
        self.saved_episodes = []
        self.total_collected_episodes = 0

    def start_data_collection(self, step=None):
        assert step in (None, 0)

    def sample_actions(self, observations, *, task_description):
        self.sample_calls += 1
        assert observations["observation/state"].shape[0] == ENV_NUM
        assert len(task_description) == ENV_NUM
        actions = _expected_policy_actions(self.policy_horizon)
        if self.return_prefix:
            return actions, np.zeros((ENV_NUM, 4), dtype=np.float32)
        return actions

    def add_data(self, step_data):
        action = step_data["action"]
        if isinstance(action, tuple):
            action = action[0]
        self.added_action_shapes.append(np.asarray(action).shape)
        self.added_actions.append(np.asarray(action).copy())

    def save_episode(self, *, is_success, env_index, task_description):
        self.saved_episodes.append((is_success, env_index, task_description))

    def end_data_collection(self, step=None):
        assert step in (None, 0)
        return len(self.saved_episodes)


@pytest.mark.parametrize(
    ("policy_horizon", "return_prefix"),
    [(10, False), (10, True)],
    ids=["filtered-sft", "best-of-n"],
)
def test_eight_envs_share_one_policy_call_and_execute_five_step_chunks(
    policy_horizon, return_prefix
):
    env = _FakeVectorEnv()
    agent = _FakeAgent(
        policy_horizon=policy_horizon,
        return_prefix=return_prefix,
    )
    config = SimpleNamespace(
        seed=13,
        collect=SimpleNamespace(
            tasks=["libero_90_0"],
            num_rollouts=ENV_NUM,
            num_initial_rollouts=None,
            replan_steps=REPLAN_STEPS,
        ),
    )

    metrics, episodes = collect_data(agent=agent, env=env, config=config, step=0)

    assert env.seed_value == 13
    assert agent.sample_calls == 1
    assert len(env.action_batches) == 1
    assert env.action_batches[0].shape == (ENV_NUM, REPLAN_STEPS, ACTION_DIM)
    expected = _expected_policy_actions(policy_horizon)[:, :REPLAN_STEPS]
    np.testing.assert_array_equal(env.action_batches[0], expected)
    assert agent.added_action_shapes == [(ENV_NUM, REPLAN_STEPS, ACTION_DIM)]
    np.testing.assert_array_equal(agent.added_actions[0], expected)
    assert len(agent.saved_episodes) == ENV_NUM
    assert episodes == ENV_NUM
    assert metrics["success_rate"] == 1.0


def test_best_of_n_evaluation_executes_only_the_replan_window():
    env = _FakeVectorEnv()
    agent = _FakeAgent(policy_horizon=10, return_prefix=True)
    config = SimpleNamespace(
        collect=SimpleNamespace(
            eval_tasks=["libero_90_0"],
            num_eval_rollouts=ENV_NUM,
            replan_steps=REPLAN_STEPS,
        )
    )

    metrics = evaluate_policy(agent=agent, env=env, config=config, step=0)

    assert agent.sample_calls == 1
    assert len(env.action_batches) == 1
    assert env.action_batches[0].shape == (ENV_NUM, REPLAN_STEPS, ACTION_DIM)
    np.testing.assert_array_equal(
        env.action_batches[0],
        _expected_policy_actions(agent.policy_horizon)[:, :REPLAN_STEPS],
    )
    assert metrics["eval/success_rate"] == 1.0
