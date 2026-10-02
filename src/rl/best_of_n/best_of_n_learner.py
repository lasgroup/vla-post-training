# ruff: noqa: F722
import functools
import logging
from typing import Any
import gc

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.transforms as _transforms
from src.rl.best_of_n.update_critic import (
    init_state_action_critic_train_state,
    init_state_value_train_state,
    train_q_step,
    train_value_step,
    _build_mlp_critic_defs,
)
from src.rl.value_distribution import get_value_bounds, make_value_distribution
from src.rl.networks.rl_networks import ObsType, PREFIX_EMBEDDING_NAME
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.training.config import BestofNLearnerConfig


def _inference_model(train_state: training_utils.TrainState):
    """Merge a train state into an eval-mode module, preferring its EMA parameters."""
    params = train_state.ema_params if train_state.ema_params is not None else train_state.params
    model = nnx.merge(train_state.model_def, params)
    model.eval()
    return model


def _group_envs_by_task(task_description: list) -> dict[str, list[int]]:
    """Env indices per task, in first-seen order, so each policy call gets one prompt."""
    task_to_indices: dict[str, list[int]] = {}
    for i, task in enumerate(task_description):
        task_to_indices.setdefault(str(task), []).append(i)
    return task_to_indices


def _is_droid_layout(processed_obs: dict) -> bool:
    """LIBERO observations carry a flat `observation/state`; DROID/Molmo split joints and gripper."""
    return "observation/state" not in processed_obs


def _raw_state(processed_obs: dict) -> np.ndarray:
    """Un-normalized proprioceptive state, assembled the way the policy input transforms do."""
    if not _is_droid_layout(processed_obs):
        return np.asarray(processed_obs["observation/state"])
    # Droid/molmo layout: assemble state the same way DroidInputs does.
    joint = np.asarray(processed_obs["observation/joint_position"])
    gripper = np.asarray(processed_obs["observation/gripper_position"])
    if gripper.ndim == joint.ndim - 1:
        gripper = gripper[..., np.newaxis]
    return np.concatenate([joint, gripper], axis=-1)


def _select_best(candidates: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Pick the highest-scoring candidate per env.

    candidates: `[g * n, horizon, dim]`, env-major (env 0's n candidates first); scores: `[g, n]`.
    Returns `[g, horizon, dim]`.
    """
    group_env_num, n_samples = scores.shape
    candidates = np.asarray(candidates)
    candidates = candidates.reshape(group_env_num, n_samples, *candidates.shape[1:])
    return candidates[np.arange(group_env_num), scores.argmax(axis=1)]


class BestofNLearner(FilteredSFTLearner):
    # Prefixes are cached in the buffer at collection time, so critic updates consume
    # only `state` + the cached prefix and never read the stored images (the policy is
    # frozen). Dropping images cuts the buffer footprint ~45x and makes shard
    # save/restore cheap.
    _store_prefix_rep = True
    _buffer_obs_drop_keys = ("image",)

    def __init__(self, config):

        assert isinstance(config.rl, BestofNLearnerConfig), (
            "Only BestofNLearnerConfig should be passed to the Best-of-N agent"
        )

        model = config.model.create(jax.random.key(config.seed))
        fake_obs = config.model.fake_obs(batch_size=1)
        prefix_rep = model.get_prefix_rep(fake_obs)[0]
        del model
        assert prefix_rep.ndim == 3, f"Expected prefix_rep to have shape (batch, seq_len, embed_dim), but got {prefix_rep.shape}"
        prefix_embedding_shape = tuple(prefix_rep.shape[2:])
        dummy_obs = {
            "state": fake_obs.state,
            PREFIX_EMBEDDING_NAME: jnp.zeros((1, *prefix_embedding_shape), dtype=jnp.float32)
        }
        dummy_act = config.model.fake_act(batch_size=1)
        if config.rl.critic.use_bronet:
            from src.rl.networks.bronet_critic import BroNetStateActionCritic, BroNetStateValue
            hidden_dim = config.rl.critic.bronet_hidden_dim
            depth = config.rl.critic.bronet_depth
            num_qs = config.rl.critic.num_qs
            num_vs = config.rl.critic.num_vs
            num_bins = config.rl.critic.num_value_bins
            if config.rl.critic.use_distributional_critic:
                assert num_bins > 1, (
                    "use_distributional_critic=True requires num_value_bins > 1."
                )
            def state_action_critic_def(observation, action, rngs):
                return BroNetStateActionCritic(observation=observation, action=action, hidden_dim=hidden_dim, depth=depth, num_qs=num_qs, num_bins=num_bins, rngs=rngs)
            def state_value_def(observation, rngs):
                return BroNetStateValue(observation=observation, hidden_dim=hidden_dim, depth=depth, num_vs=num_vs, num_bins=num_bins, rngs=rngs)
        else:
            state_action_critic_def, state_value_def = _build_mlp_critic_defs(config)

        self._prefix_embed_dim = prefix_embedding_shape[-1]

        super().__init__(config)

        data_config = self._data_loader.data_config()
        normalizer = _transforms.Normalize(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        )
        self._state_normalize = normalizer
        self._action_normalize = normalizer
        self._transition_state_dim = int(dummy_obs["state"].shape[-1])

        q_init_rng, v_init_rng, self._rng = jax.random.split(self._rng, 3)
        self._state_action_critic_state, self._state_action_critic_state_sharding = (
            init_state_action_critic_train_state(
                self._config,
                q_init_rng,
                self._mesh,
                critic_def=state_action_critic_def,
                dummy_obs=dummy_obs,
                dummy_act=dummy_act,
            )
        )

        self._value_state, self._value_state_sharding = init_state_value_train_state(
            self._config,
            v_init_rng,
            self._mesh,
            critic_def=state_value_def,
            dummy_obs=dummy_obs,
        )
        self._rl_state_checkpointer = ocp.StandardCheckpointer()
        if self._resuming:
            self._restore_rl_checkpoint(step=int(self.training_steps))
        jax.block_until_ready(self._state_action_critic_state)
        jax.block_until_ready(self._value_state)

        del self._train_step
        gc.collect()

        # 1. Un-JIT the inner steps (JAX will compile these as part of the outer methods)
        self._q_train_step = functools.partial(train_q_step, self._config)
        self._value_train_step = functools.partial(train_value_step, self._config)
        self._refresh_critic_update_function()

    def _refresh_critic_update_function(self):
        def _critics_wrapper(batch, q_state, value_state, rng):
            return self._update_critics(
                batch=batch,
                q_state=q_state,
                value_state=value_state,
                rng=rng,
            )

        self._update_critics_jitted = jax.jit(
            _critics_wrapper,
            in_shardings=(
                self._data_sharding,
                self._state_action_critic_state_sharding,
                self._value_state_sharding,
                self._replicated_sharding,
            ),
            out_shardings=(
                self._state_action_critic_state_sharding,
                self._value_state_sharding,
                self._replicated_sharding,
                self._replicated_sharding,
            ),
            donate_argnums=(1, 2),
        )

    def _rl_checkpoint_state(self) -> dict[str, training_utils.TrainState]:
        return {
            "state_action_critic_state": self._state_action_critic_state,
            "value_state": self._value_state,
        }

    def _rl_checkpoint_dir(self) -> epath.Path:
        return epath.Path(self._config.checkpoint_dir) / "rl_state"

    def _rl_checkpoint_path(self, step: int) -> epath.Path:
        return self._rl_checkpoint_dir() / str(int(step))

    def _restore_rl_checkpoint(self, *, step: int) -> None:
        path = self._rl_checkpoint_path(step)
        if not path.exists():
            logging.warning(
                "No RL critic checkpoint found at %s; starting critics from scratch.",
                path,
            )
            return
        restored = self._rl_state_checkpointer.restore(
            path,
            self._rl_checkpoint_state(),
        )
        self._state_action_critic_state = restored["state_action_critic_state"]
        self._value_state = restored["value_state"]

    def save_checkpoint(self, step: int | None = None):
        if step is None:
            step = self.training_steps
        super().save_checkpoint(step=step)
        path = self._rl_checkpoint_path(step)
        if path.exists():
            return
        self._rl_checkpoint_dir().mkdir(parents=True, exist_ok=True)
        self._rl_state_checkpointer.save(
            path,
            self._rl_checkpoint_state(),
        )
        self._rl_state_checkpointer.wait_until_finished()

    def _make_buffer_dummy_data(self) -> dict:
        dummy = super()._make_buffer_dummy_data()
        dummy["observations"][PREFIX_EMBEDDING_NAME] = np.zeros((1, self._prefix_embed_dim), dtype=np.float32)
        for k in self._buffer_obs_drop_keys:
            dummy["observations"].pop(k, None)
        return dummy

    @at.typecheck
    def _online_batch_to_critic_batch(
            self,
            online_batch: dict[str, Any],
    ) -> tuple[
        ObsType,
        _model.Actions,
        ObsType,
        at.Float[at.Array, " b"],
        at.Float[at.Array, " b"],
        at.Float[at.Array, " b"],
    ]:
        # Critics only see the (normalized) state and the cached prefix embedding.
        def critic_obs(obs):
            return {"state": obs["state"], PREFIX_EMBEDDING_NAME: obs[PREFIX_EMBEDDING_NAME]}

        return (
            critic_obs(online_batch["observation"]),
            online_batch["actions"],
            critic_obs(online_batch["next_observation"]),
            online_batch["reward"],
            online_batch["discount"],
            online_batch["mc_return"],
        )

    def save_episode(self, is_success: bool, env_index: int, task_description: str):
        assert env_index in range(len(self._episode_storage)), \
            f"env_index must be between 0 and {len(self._episode_storage) - 1}, but got {env_index}."
        # extract episode data from storage and empty it
        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        self._save_episode_in_buffer(episode_data, task_description, is_success=is_success)

    @staticmethod
    def _pad_last_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
        """Zero-pad the last axis up to ``target_dim`` (no-op if already >= target_dim)."""
        arr = np.asarray(arr)
        if arr.shape[-1] >= target_dim:
            return arr
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, target_dim - arr.shape[-1])
        return np.pad(arr, pad_width, mode="constant", constant_values=0.0)

    # ------------------------------------------------------------------ #
    # Best-of-N action selection
    # ------------------------------------------------------------------ #

    def sample_actions(self, observations, **kwargs):
        """Sample `n_samples` candidate chunks per env and return the one the Q-critic scores highest.

        Returns `(actions, prefix)`: the selected absolute actions `[env, horizon, dim]` and each
        env's mean-pooled prefix embedding `[env, embed]` (cached in the buffer for critic updates).
        Envs are processed in groups that share a task, since the policy takes one prompt per call.
        """
        if self.training_steps < self._config.rl.critic.inference_start_step:
            return super().sample_actions(observations, **kwargs)
        rng, self._rng = jax.random.split(self._rng)
        task_description = self._per_env_task_descriptions(
            observations, kwargs.get("task_description")
        )
        # Build the (EMA) models once; they are shared across task groups.
        q_model = _inference_model(self._state_action_critic_state)
        policy_model = _inference_model(self._train_state)

        env_num = len(task_description)
        all_best_actions = None
        all_best_prefix = None
        for task, indices in _group_envs_by_task(task_description).items():
            group_obs = jax.tree.map(lambda x: np.asarray(x)[indices], observations)
            processed_obs = self._process_obs_for_pi0(group_obs, task_description=task)
            best, prefix = self._best_of_n(processed_obs, rng, q_model, policy_model)

            if all_best_actions is None:
                all_best_actions = np.zeros((env_num, *best.shape[1:]), dtype=np.float32)
                all_best_prefix = np.zeros((env_num, prefix.shape[-1]), dtype=np.float32)
            all_best_actions[indices] = np.asarray(best, dtype=np.float32)
            all_best_prefix[indices] = np.asarray(prefix, dtype=np.float32)

        return all_best_actions, all_best_prefix

    def _best_of_n(self, processed_obs, rng, q_model, policy_model) -> tuple[np.ndarray, np.ndarray]:
        """Best-of-N for one group of envs that share a prompt: `[g, horizon, dim]`, `[g, embed]`."""
        n_samples = self._config.rl.n_samples
        droid_layout = _is_droid_layout(processed_obs)
        # 1. Sample all candidates in one pass: [g * n, horizon, dim], absolute robot-space actions.
        candidates = self._sample_candidates(processed_obs, rng, n_samples)
        # 2. Critic observation: normalized state + prefix embedding, repeated per candidate.
        raw_state = _raw_state(processed_obs)
        critic_obs, prefix = self._critic_observation(
            processed_obs, raw_state, policy_model, n_samples, droid_layout=droid_layout
        )
        # 3. Score every candidate with the Q-critic: [g, n].
        scores = self._score_candidates(
            q_model, critic_obs, candidates, raw_state, n_samples, droid_layout=droid_layout
        )
        # 4. Keep the highest-scoring candidate per env.
        return _select_best(candidates, scores), prefix

    @staticmethod
    def _per_env_task_descriptions(observations, task_description) -> list:
        """Broadcast a single (or missing) prompt to one entry per env."""
        if task_description is None or isinstance(task_description, str):
            env_num = next(np.asarray(v).shape[0] for v in observations.values())
            return [task_description] * env_num
        return list(task_description)

    def _sample_candidates(self, processed_obs: dict, rng, n_samples: int) -> np.ndarray:
        """Tile each env's observation `n_samples` times and sample all candidates at once."""
        tiled_obs = {
            k: (v if k == "prompt" else np.repeat(np.asarray(v), n_samples, axis=0))
            for k, v in processed_obs.items()
        }
        return self._sample_action(tiled_obs, rng, self._train_state)

    def _critic_observation(
        self, processed_obs: dict, raw_state: np.ndarray, policy_model, n_samples: int, *, droid_layout: bool
    ) -> tuple[dict, np.ndarray]:
        """Build the critic's input exactly as the replay buffer stores it, repeated per candidate.

        Returns the critic observation and the mean-pooled prefix `[g, embed]`.
        """
        state = self._normalize_and_pad_state(raw_state, droid_layout=droid_layout)
        prefix = self._mean_prefix_embedding(processed_obs, policy_model)
        critic_obs = {
            "state": jnp.repeat(jnp.asarray(state, dtype=jnp.float32), n_samples, axis=0),
            PREFIX_EMBEDDING_NAME: jnp.repeat(jnp.asarray(prefix), n_samples, axis=0),
        }
        return critic_obs, prefix

    def _normalize_and_pad_state(self, raw_state: np.ndarray, *, droid_layout: bool) -> np.ndarray:
        """Match the buffer's state preprocessing order for the domain."""
        if droid_layout:
            # pad-then-normalize (droid buffer order)
            state = self._pad_last_dim(raw_state, self._transition_state_dim)
            return np.asarray(self._state_normalize({"state": state})["state"])
        # normalize-then-pad (libero buffer order)
        state = np.asarray(self._state_normalize({"state": raw_state})["state"])
        return self._pad_last_dim(state, self._transition_state_dim)

    def _mean_prefix_embedding(self, processed_obs: dict, policy_model) -> np.ndarray:
        """Policy prefix (VLM) embedding per env, mean-pooled over tokens: `[g, embed]`.

        Inputs are transformed per env before stacking: the policy input transforms treat a
        leading axis of size 3 as CHW.
        """
        group_env_num = next(np.asarray(v).shape[0] for k, v in processed_obs.items() if k != "prompt")
        per_env_inputs = [
            self._policy._input_transform(
                {k: (v if k == "prompt" else np.asarray(v)[i]) for k, v in processed_obs.items()}
            )
            for i in range(group_env_num)
        ]

        def _stack(*values):
            if values[0] is None:
                return None
            return jnp.stack([jnp.asarray(v) for v in values], axis=0)

        def _as_batched_array(value):
            if value is None:
                return None
            value = jnp.asarray(value)
            if value.ndim > 0 and value.shape[0] == group_env_num:
                return value
            return jnp.broadcast_to(value[jnp.newaxis, ...], (group_env_num,) + value.shape)

        inputs = jax.tree.map(_stack, *per_env_inputs)
        inputs = {
            k: (
                jax.tree.map(lambda x: None if x is None else jnp.asarray(x), v)
                if k in ("image", "state")
                else jax.tree.map(_as_batched_array, v)
            )
            for k, v in inputs.items()
        }
        prefix = self._get_prefix_rep_with_model(
            m=policy_model, observation=_model.Observation.from_dict(inputs)
        )
        prefix = np.asarray(prefix)
        if prefix.ndim == 3:
            prefix = prefix.reshape(prefix.shape[0], -1, prefix.shape[-1]).mean(axis=1)
        return prefix

    def _score_candidates(
        self,
        q_model,
        critic_obs: dict,
        candidates: np.ndarray,
        raw_state: np.ndarray,
        n_samples: int,
        *,
        droid_layout: bool,
    ) -> np.ndarray:
        """Q-value of every candidate, reduced over the critic ensemble: `[g, n]`."""
        actions = self._candidates_to_critic_actions(
            candidates, raw_state, n_samples, droid_layout=droid_layout
        )
        flat_actions = jnp.asarray(actions.reshape(actions.shape[0], -1))
        q_logits = q_model(critic_obs, flat_actions)  # [num_qs, batch] for Gaussian or [num_qs, batch, K] for Categorical
        lower, upper = get_value_bounds(self._config)
        q_dist = make_value_distribution(q_logits, self._config.rl.critic.num_value_bins, lower, upper)
        scores = np.asarray(q_dist.mean())  # [num_qs, batch] or [batch]
        if scores.ndim > 1:
            scores = scores.min(axis=0)  # pessimistic ensemble reduction
        return scores.reshape(-1, n_samples)

    def _candidates_to_critic_actions(
        self, candidates: np.ndarray, raw_state: np.ndarray, n_samples: int, *, droid_layout: bool
    ) -> np.ndarray:
        """Map absolute robot-space candidates into the action space the critic was trained on.

        The candidates come out of the policy's output transform (Unnormalize -> AbsoluteActions),
        while the critic saw the buffer's actions, so the domain's buffer preprocessing order is
        replicated here (see FilteredSFTLearner._get_policy_transforms):
          - libero: normalize -> pad (no delta)
          - droid/molmo: delta (abs - state, first 7 dims) -> pad -> normalize
        """
        model_act_dim = self._config.model.action_dim
        if droid_layout:
            # Absolute -> delta to match DeltaActions(make_bool_mask(7, -1)): subtract the raw state
            # from the first 7 dims, leaving the gripper absolute.
            state_tiled = np.repeat(np.asarray(raw_state), n_samples, axis=0)
            actions = np.array(candidates, copy=True)
            actions[..., :7] -= state_tiled[:, np.newaxis, :7]
            actions = self._pad_last_dim(actions, model_act_dim)
            return np.asarray(self._action_normalize({"actions": actions})["actions"])
        actions = np.asarray(self._action_normalize({"actions": np.asarray(candidates)})["actions"])
        return self._pad_last_dim(actions, model_act_dim)

    @at.typecheck
    def _update_critics(
            self,
            batch: dict[str, Any],
            q_state: training_utils.TrainState,
            value_state: training_utils.TrainState,
            rng: at.KeyArrayLike,
    ) -> tuple[
        training_utils.TrainState,
        training_utils.TrainState,
        dict[str, at.Array],
        dict[str, at.Array],
    ]:
        batch = self._online_batch_to_critic_batch(batch)
        q_rng, v_rng, rng = jax.random.split(rng, 3)
        # Update the state action critic state
        q_state, q_info = self._q_train_step(
            q_rng,
            q_state,
            value_state,
            batch,
        )
        # Update the value state
        value_state, value_info = self._value_train_step(
            v_rng,
            value_state,
            q_state,
            batch,
        )
        return q_state, value_state, q_info, value_info

    @at.typecheck
    def update(self) -> dict:
        self.training_steps += 1
        if self.training_steps % self._config.rl.critic.update_interval != 0:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }
        critic_batch_size = self._config.rl.critic.batch_size

        use_online = self._online_data_buffer.size >= critic_batch_size

        critic_info = {}
        if use_online:
            critic_online_batch = self._online_data_buffer.sample(batch_size=critic_batch_size)
            critic_rng, self._rng = jax.random.split(self._rng, 2)
            with sharding.set_mesh(self._mesh):
                q_state, value_state, q_info, value_info = (
                    self._update_critics_jitted(
                        critic_online_batch,
                        self._state_action_critic_state,
                        self._value_state,
                        critic_rng,
                    )
                )
            self._state_action_critic_state = q_state
            self._value_state = value_state

            critic_info = {
                f"critic/q_{key}": value for key, value in q_info.items()
            } | {f"critic/value_{key}": value for key, value in value_info.items()}
        info = (
                critic_info
                | {
                    "online_buffer_size": jnp.asarray(
                        float(self._online_data_buffer.size), dtype=jnp.float32
                    )
                }
        )
        info = jax.tree.map(np.asarray, info)
        return info
