from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import src.rl.best_of_n.best_of_n_learner as best_of_n_module
import src.rl.filtered_sft_agent.filtered_sft_learner as filtered_sft_module
from src.rl.best_of_n.best_of_n_learner import BestofNLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.training.config import BestofNLearnerConfig, CriticTrainingConfig


ENV_NUM = 8
ACTION_HORIZON = 10
ACTION_DIM = 7


def test_best_of_n_default_uses_benchmarked_candidate_batch():
    assert BestofNLearnerConfig().inference_candidates_per_batch == 30


class _FakePolicy:
    action_horizon = ACTION_HORIZON
    action_dim = ACTION_DIM

    def __init__(self):
        self.infer_calls = []
        self._input_transform = lambda inputs: {
            "state": np.asarray(inputs["observation/state"]),
        }

    def infer_with_model(self, **kwargs):
        self.infer_calls.append(kwargs)
        batch_size = np.asarray(kwargs["obs"]["observation/state"]).shape[0]
        prompt_ids = np.asarray(
            [int(prompt.rsplit(" ", 1)[-1]) for prompt in kwargs["obs"]["prompt"]],
            dtype=np.float32,
        )
        return {
            "actions": np.broadcast_to(
                prompt_ids[:, None, None],
                (batch_size, ACTION_HORIZON, ACTION_DIM),
            ).copy()
        }


class _FakeModel:
    def eval(self):
        return None


class _FakeQModel(_FakeModel):
    def __init__(self):
        self.flat_action_shapes = []

    def __call__(self, critic_obs, flat_actions):
        del critic_obs
        self.flat_action_shapes.append(tuple(flat_actions.shape))
        candidate_scores = jnp.tile(
            jnp.arange(3, dtype=jnp.float32), flat_actions.shape[0] // 3
        )
        return candidate_scores[jnp.newaxis, :]


class _FakeValueDistribution:
    def __init__(self, values):
        self.values = values

    def mean(self):
        return self.values


def test_filtered_sft_batches_all_eight_environments_in_one_policy_call(monkeypatch):
    policy = _FakePolicy()
    learner = object.__new__(FilteredSFTLearner)
    learner._policy = policy
    learner._data_sharding = None
    learner._rng = jax.random.key(0)
    learner._train_state = SimpleNamespace(
        ema_params=None,
        params=object(),
        model_def=object(),
    )
    learner._config = SimpleNamespace(collect=SimpleNamespace(store_prefix_rep=False))
    learner._process_obs_for_pi0 = lambda observations, task_description: {
        "observation/state": np.asarray(observations["observation/state"])[:, -1],
        "prompt": task_description,
    }
    monkeypatch.setattr(filtered_sft_module.nnx, "merge", lambda *_args: _FakeModel())

    task_description = [f"test task {index % 4}" for index in range(ENV_NUM)]
    actions = learner.sample_actions(
        {"observation/state": np.zeros((ENV_NUM, 1, ACTION_DIM), dtype=np.float32)},
        task_description=task_description,
    )

    assert actions.shape == (ENV_NUM, ACTION_HORIZON, ACTION_DIM)
    assert len(policy.infer_calls) == 1
    assert policy.infer_calls[0]["obs"]["prompt"] == task_description
    np.testing.assert_array_equal(
        actions[:, 0, 0], np.arange(ENV_NUM, dtype=np.float32) % 4
    )
    assert policy.infer_calls[0]["noise"].shape == (
        ENV_NUM,
        ACTION_HORIZON,
        ACTION_DIM,
    )


def test_best_of_n_batches_candidates_for_eight_mixed_task_environments(monkeypatch):
    policy = _FakePolicy()
    q_model = _FakeQModel()
    learner = object.__new__(BestofNLearner)
    learner.training_steps = 0
    learner._rng = jax.random.key(0)
    learner._policy = policy
    learner._train_state = SimpleNamespace(
        ema_params=None,
        params=object(),
        model_def="policy",
    )
    learner._state_action_critic_state = SimpleNamespace(
        ema_params=None,
        params=object(),
        model_def="critic",
    )
    learner._config = SimpleNamespace(
        collect=SimpleNamespace(store_prefix_rep=True),
        model=SimpleNamespace(action_dim=ACTION_DIM),
        rl=BestofNLearnerConfig(
            n_samples=3,
            inference_candidates_per_batch=2,
            critic=CriticTrainingConfig(
                inference_start_step=0,
                num_value_bins=1,
            ),
        ),
    )
    learner._transition_state_dim = ACTION_DIM
    learner._process_obs_for_pi0 = lambda observations, task_description: {
        "observation/state": np.asarray(observations["observation/state"])[:, -1],
        "prompt": task_description,
    }
    learner._state_normalize = lambda values: values
    learner._action_normalize = lambda values: values
    sample_calls = []
    sampled_prompts = []
    next_candidate = 0

    def sample_candidates(observations, rng, train_state, *, return_prefix_rep=False):
        nonlocal next_candidate
        del rng, train_state
        state = np.asarray(observations["observation/state"])
        batch_size = state.shape[0]
        sample_calls.append(batch_size)
        sampled_prompts.append(list(observations["prompt"]))
        candidate_count = batch_size // ENV_NUM
        env_ids = state[:, 0].astype(np.int32)
        prompt_ids = np.asarray(
            [int(prompt.rsplit(" ", 1)[-1]) for prompt in observations["prompt"]],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(prompt_ids, env_ids % 4)
        candidate_ids = (
            env_ids * 3 + next_candidate + np.tile(np.arange(candidate_count), ENV_NUM)
        ).astype(np.float32)
        next_candidate += candidate_count
        actions = np.broadcast_to(
            candidate_ids[:, None, None],
            (batch_size, ACTION_HORIZON, ACTION_DIM),
        ).copy()
        if not return_prefix_rep:
            return actions
        prefix_ids = env_ids * 10 + prompt_ids
        prefix = np.broadcast_to(
            prefix_ids[:, None, None],
            (batch_size, 2, 4),
        ).copy()
        return actions, prefix

    learner._sample_action = sample_candidates

    def merge(model_def, _params):
        return q_model if model_def == "critic" else _FakeModel()

    monkeypatch.setattr(best_of_n_module.nnx, "merge", merge)
    monkeypatch.setattr(
        best_of_n_module._model.Observation,
        "from_dict",
        lambda inputs: inputs,
    )
    monkeypatch.setattr(best_of_n_module, "get_value_bounds", lambda _config: (-1, 1))
    monkeypatch.setattr(
        best_of_n_module,
        "make_value_distribution",
        lambda logits, *_args: _FakeValueDistribution(logits),
    )

    state = np.broadcast_to(
        np.arange(ENV_NUM, dtype=np.float32)[:, None, None],
        (ENV_NUM, 1, ACTION_DIM),
    ).copy()
    task_description = [f"test task {index % 4}" for index in range(ENV_NUM)]
    result = learner.sample_actions(
        {"observation/state": state},
        task_description=task_description,
    )
    assert isinstance(result, tuple)
    actions, prefix = result
    assert prefix is not None

    assert sample_calls == [ENV_NUM * 2, ENV_NUM]
    assert sampled_prompts == [
        [prompt for prompt in task_description for _ in range(2)],
        task_description,
    ]
    assert (
        q_model.flat_action_shapes
        == [(ENV_NUM // 4 * 3, ACTION_HORIZON * ACTION_DIM)] * 4
    )
    assert actions.shape == (ENV_NUM, ACTION_HORIZON, ACTION_DIM)
    np.testing.assert_array_equal(actions[:, 0, 0], np.arange(2, ENV_NUM * 3, 3))
    np.testing.assert_array_equal(
        prefix[:, 0], np.arange(ENV_NUM) * 10 + np.arange(ENV_NUM) % 4
    )
