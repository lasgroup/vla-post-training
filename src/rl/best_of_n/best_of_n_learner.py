# ruff: noqa: F722
import dataclasses
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
    _build_pi0_backbone_critic_defs,
)
from src.rl.value_distribution import get_value_bounds, make_value_distribution
from src.rl.networks.rl_networks import ObsType
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.config import BestofNLearnerConfig


class BestofNLearner(FilteredSFTLearner):
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
            state_action_critic_def, state_value_def = _build_pi0_backbone_critic_defs(config)
        
        self._prefix_embed_dim = None
        if config.collect.store_prefix_rep and PREFIX_EMBEDDING_NAME in dummy_obs:
            self._prefix_embed_dim = int(np.asarray(dummy_obs[PREFIX_EMBEDDING_NAME]).shape[-1])

        # When prefixes are cached (store_prefix_rep) and the critic is not retrained on
        # freshly resampled on-policy actions (train_on_policy_value_function), nothing
        # ever reads the stored images: critic updates consume only `state` + the cached
        # prefix, and the policy is frozen. Drop images from the online buffer entirely,
        # cutting its footprint ~45x and making shard save/restore cheap.
        self._buffer_obs_drop_keys: tuple[str, ...] = ()
        if config.collect.store_prefix_rep and not config.rl.train_on_policy_value_function:
            self._buffer_obs_drop_keys = ("image",)
            logging.info(
                "Best-of-N: prefixes are cached and critics are not trained on-policy; "
                "dropping images from the online replay buffer to save memory."
            )

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
        self._train_state_sharding = sharding.fsdp_sharding(
            self._train_state, self._mesh, log=False
        )

        def _critics_wrapper(batch, q_state, value_state, policy_state, rng):
            return self._update_critics(
                batch=batch,
                q_state=q_state,
                value_state=value_state,
                policy_state=policy_state,
                rng=rng,
            )

        self._update_critics_jitted = jax.jit(
            _critics_wrapper,
            in_shardings=(
                self._data_sharding,
                self._state_action_critic_state_sharding,
                self._value_state_sharding,
                self._train_state_sharding,
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

    def _make_buffer_dummy_data(self) -> dict:
        dummy = super()._make_buffer_dummy_data()
        if self._prefix_embed_dim is not None:
            zeros = np.zeros((1, self._prefix_embed_dim), dtype=np.float32)
            dummy["observations"][PREFIX_EMBEDDING_NAME] = zeros
        for k in self._buffer_obs_drop_keys:
            dummy["observations"].pop(k, None)
        return dummy

    def _recompute_prefix_embedding(
            self,
            *,
            observation: dict[str, Any],
            policy_state: training_utils.TrainState,
    ) -> at.Float[at.Array, "batch embed"] | None:
        model = self._get_policy_model(policy_state)
        # Both SFT-loader Observations and online-buffer dicts are already
        # fully transformed (repack, LiberoInputs, Normalize, tokenize, etc.)
        # by the data pipeline / _preprocess_insert
        obs = _model.Observation.from_dict(observation)
        prefix = self._policy._get_prefix_rep_with_model(model, observation=obs)
        prefix = prefix.reshape((prefix.shape[0], -1, prefix.shape[-1]))
        prefix = jnp.mean(prefix, axis=1)
        return prefix

    @staticmethod
    def _get_policy_model(policy_state: training_utils.TrainState) -> _model.BaseModel:
        """Merge policy params into a model. Call once per update() to avoid duplicates."""
        params = (
            policy_state.ema_params
            if policy_state.ema_params is not None
            else policy_state.params
        )
        model = nnx.merge(policy_state.model_def, params)
        model.eval()
        return model

    @at.typecheck
    def _online_batch_to_critic_batch(
            self,
            online_batch: dict[str, Any],
            policy_state: training_utils.TrainState,
    ) -> tuple[
        ObsType,
        _model.Actions,
        ObsType,
        at.Float[at.Array, " b"],
        at.Float[at.Array, " b"],
        at.Float[at.Array, " b"],
    ]:
        online_observation = online_batch["observation"]
        observation_dict: dict[str, Any] = {
            "state": online_observation["state"],
        }

        next_observation = online_batch["next_observation"]
        next_observation_dict: dict[str, Any] = {"state": next_observation["state"]}

        if PREFIX_EMBEDDING_NAME in online_observation and PREFIX_EMBEDDING_NAME in next_observation:
            observation_dict[PREFIX_EMBEDDING_NAME] = online_observation[PREFIX_EMBEDDING_NAME]
            next_observation_dict[PREFIX_EMBEDDING_NAME] = next_observation[PREFIX_EMBEDDING_NAME]
        else:
            observation_dict[PREFIX_EMBEDDING_NAME] = self._recompute_prefix_embedding(
                observation=online_observation, policy_state=policy_state,
            )
            next_observation_dict[PREFIX_EMBEDDING_NAME] = self._recompute_prefix_embedding(
                observation=next_observation, policy_state=policy_state,
            )

        return (
            observation_dict,
            online_batch["actions"],
            next_observation_dict,
            online_batch["reward"],
            online_batch["discount"],
            online_batch["mc_return"],
        )

    def _sft_batch_to_actor_batch(
            self,
            sft_batch: tuple[_model.Observation, _model.Actions],
            policy_state: training_utils.TrainState,
    ) -> tuple[_model.Observation, ObsType, _model.Actions]:
        policy_observation, actions = sft_batch
        policy_obs_dict = policy_observation.to_dict()

        critic_observation: dict[str, Any] = {
            "state": policy_obs_dict["state"],
        }

        prefix_embedding = self._recompute_prefix_embedding(
            observation=policy_obs_dict,
            policy_state=policy_state,
        )

        critic_observation[PREFIX_EMBEDDING_NAME] = prefix_embedding

        return policy_observation, critic_observation, actions

    def save_episode(self, is_success: bool, env_index: int, task_description: str, task_id: str | None = None):
        assert env_index in range(len(self._episode_storage)), \
            f"env_index must be between 0 and {len(self._episode_storage) - 1}, but got {env_index}."
        # extract episode data from storage and empty it
        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        self._save_episode_in_buffer(episode_data, task_description, is_success=is_success, task_id=task_id)

    @staticmethod
    def _pad_last_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
        """Zero-pad the last axis up to ``target_dim`` (no-op if already >= target_dim)."""
        arr = np.asarray(arr)
        if arr.shape[-1] >= target_dim:
            return arr
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-1] = (0, target_dim - arr.shape[-1])
        return np.pad(arr, pad_width, mode="constant", constant_values=0.0)

    def sample_actions(self, observations, **kwargs):
        if self.training_steps < self._config.rl.critic.inference_start_step:
            return super().sample_actions(observations, **kwargs)
        n_samples = self._config.rl.n_samples
        rng, self._rng = jax.random.split(self._rng)
        task_description = kwargs.get("task_description")

        # Group envs by task so each _sample_action call gets a single string prompt.
        if task_description is None or isinstance(task_description, str):
            task_description = [task_description] * next(
                np.asarray(v).shape[0] for v in observations.values()
            )
        task_to_indices: dict[str, list[int]] = {}
        for i, task in enumerate(task_description):
            task_to_indices.setdefault(str(task), []).append(i)

        env_num = len(task_description)
        return_prefix_rep = self._config.collect.store_prefix_rep
        all_best_actions = None
        all_best_prefix = None

        # Build q-model once, shared across task groups.
        q_params = (
            self._state_action_critic_state.ema_params
            if self._state_action_critic_state.ema_params is not None
            else self._state_action_critic_state.params
        )
        q_model = nnx.merge(self._state_action_critic_state.model_def, q_params)
        q_model.eval()

        # Build policy model once for prefix embedding.
        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        policy_model = nnx.merge(self._train_state.model_def, params)
        policy_model.eval()

        for task, indices in task_to_indices.items():
            group_obs = jax.tree.map(lambda x: np.asarray(x)[indices], observations)
            processed_obs = self._process_obs_for_pi0(group_obs, task_description=task)
            group_env_num = len(indices)

            # 1. Tile obs along batch dim and sample all candidates in one pass
            tiled_obs = {
                k: (v if k == "prompt" else np.repeat(np.asarray(v), n_samples, axis=0))
                for k, v in processed_obs.items()
            }
            group_actions = self._sample_action(tiled_obs, rng, self._train_state)
            # group_actions: [group_env_num * n_samples, horizon, dim]

            # 2. Build critic observation (normalize + pad state to match buffer preprocessing)
            if "observation/state" in processed_obs:
                raw_state = np.asarray(processed_obs["observation/state"])
                is_droid_layout = False
            else:
                # Droid/molmo layout: assemble state the same way DroidInputs does.
                joint = np.asarray(processed_obs["observation/joint_position"])
                gripper = np.asarray(processed_obs["observation/gripper_position"])
                if gripper.ndim == joint.ndim - 1:
                    gripper = gripper[..., np.newaxis]
                raw_state = np.concatenate([joint, gripper], axis=-1)
                is_droid_layout = True

            if is_droid_layout:
                # pad-then-normalize (droid buffer order)
                state = self._pad_last_dim(raw_state, self._transition_state_dim)
                state = np.asarray(self._state_normalize({"state": state})["state"])
            else:
                # normalize-then-pad (libero buffer order)
                state = np.asarray(self._state_normalize({"state": raw_state})["state"])
                state = self._pad_last_dim(state, self._transition_state_dim)
            state = jnp.repeat(jnp.asarray(state, dtype=jnp.float32), n_samples, axis=0)
            critic_obs: dict = {"state": state}

            # Compute prefix embedding on the non-tiled group, then repeat it
            # across candidates. This matches ralf/value_learning's expensive
            # model work. We transform per env before stacking to avoid LiberoInputs
            # misreading a 3-env HWC image batch as one CHW image.
            per_env_inputs = [
                self._policy._input_transform(
                    {
                        k: (v if k == "prompt" else np.asarray(v)[i])
                        for k, v in processed_obs.items()
                    }
                )
                for i in range(group_env_num)
            ]

            def _stack_prefix_inputs(*values):
                first = values[0]
                if first is None:
                    return None
                return jnp.stack([jnp.asarray(v) for v in values], axis=0)

            inputs = jax.tree.map(_stack_prefix_inputs, *per_env_inputs)
            batch_size = group_env_num

            def _as_batched_array(value):
                if value is None:
                    return None
                value = jnp.asarray(value)
                if value.ndim > 0 and value.shape[0] == batch_size:
                    return value
                return jnp.broadcast_to(
                    value[jnp.newaxis, ...], (batch_size,) + value.shape
                )

            inputs = {
                k: (
                    jax.tree.map(lambda x: None if x is None else jnp.asarray(x), v)
                    if k in ("image", "state")
                    else jax.tree.map(_as_batched_array, v)
                )
                for k, v in inputs.items()
            }
            obs_for_prefix = _model.Observation.from_dict(inputs)
            prefix = self._get_prefix_rep_with_model(
                m=policy_model, observation=obs_for_prefix
            )
            prefix = np.asarray(prefix)
            if prefix.ndim == 3:
                prefix = prefix.reshape(prefix.shape[0], -1, prefix.shape[-1]).mean(axis=1)
            critic_obs[PREFIX_EMBEDDING_NAME] = jnp.repeat(
                jnp.asarray(prefix), n_samples, axis=0
            )

            # 3. Score all candidates with Q-critic
            # The candidate actions are absolute, unnormalized, robot-space actions
            # (the policy _output_transform applies Unnormalize -> AbsoluteActions).
            # The critic was trained on whatever the buffer stores, so we must replicate
            # that domain's preprocessing order here:
            #   - libero: normalize -> pad (no delta)
            #   - droid/molmo: delta (abs - state, first 7 dims) -> pad -> normalize
            # See FilteredSFTLearner._get_policy_transforms for the buffer pipeline.
            model_act_dim = self._config.model.action_dim
            if is_droid_layout:
                # Convert absolute -> delta to match DeltaActions(make_bool_mask(7, -1)):
                # subtract the raw state from the first 7 dims, leaving the gripper absolute.
                # Operate on a copy so group_actions stays absolute for the env (see step 4).
                state_tiled = np.repeat(np.asarray(raw_state), n_samples, axis=0)
                candidate_actions = np.array(group_actions, copy=True)
                candidate_actions[..., :7] -= state_tiled[:, np.newaxis, :7]
                candidate_actions = self._pad_last_dim(candidate_actions, model_act_dim)
                actions_norm = np.asarray(
                    self._action_normalize({"actions": candidate_actions})["actions"]
                )
            else:
                actions_norm = np.asarray(
                    self._action_normalize({"actions": np.asarray(group_actions)})["actions"]
                )
                actions_norm = self._pad_last_dim(actions_norm, model_act_dim)
            flat_actions = jnp.asarray(
                actions_norm.reshape(group_env_num * n_samples, -1)
            )
            q_logits = q_model(critic_obs, flat_actions)
            # q_logits: [num_qs, batch] for Gaussian or [num_qs, batch, K] for Categorical

            # 4. Reduce ensemble, select best per env
            rl_config = self._config.rl
            _lower, _upper = get_value_bounds(self._config)
            q_dist = make_value_distribution(
                q_logits, rl_config.critic.num_value_bins, _lower, _upper
            )
            scores = np.asarray(q_dist.mean())  # [num_qs, batch] or [batch]
            if scores.ndim > 1:
                scores = scores.min(axis=0)
            scores = scores.reshape(group_env_num, n_samples)
            best_idx = scores.argmax(axis=1)

            group_actions = np.asarray(group_actions).reshape(
                group_env_num, n_samples, *np.asarray(group_actions).shape[1:]
            )
            best = group_actions[np.arange(group_env_num), best_idx]

            if all_best_actions is None:
                all_best_actions = np.zeros(
                    (env_num, *best.shape[1:]), dtype=np.float32
                )
            all_best_actions[indices] = np.asarray(best, dtype=np.float32)

            if return_prefix_rep:
                if all_best_prefix is None:
                    all_best_prefix = np.zeros((env_num, prefix.shape[-1]), dtype=np.float32)
                all_best_prefix[indices] = np.asarray(prefix, dtype=np.float32)

        return (all_best_actions, all_best_prefix) if return_prefix_rep else all_best_actions

    @at.typecheck
    def _get_on_policy_action(
            self,
            online_observation: _model.Observation,
            policy_state: training_utils.TrainState,
            rng: at.KeyArrayLike,
    ) -> _model.Actions:
        model = self._get_policy_model(policy_state)
        sampled_actions = model.sample_actions(
            observation=online_observation,
            rng=rng,
            return_info_dict=False,
            return_prefix_rep=False,
        )
        return sampled_actions
        
    @at.typecheck
    def _update_critics(
            self,
            batch: dict[str, Any],
            q_state: training_utils.TrainState,
            value_state: training_utils.TrainState,
            policy_state: training_utils.TrainState,
            rng: at.KeyArrayLike,
    ) -> tuple[
        training_utils.TrainState,
        training_utils.TrainState,
        dict[str, at.Array],
        dict[str, at.Array],
    ]:
        if self._config.rl.train_on_policy_value_function:
            # We replace the action from the batch with the on policy action
            # This ensures that we train an on policy critic.
            policy_sample_rng, rng = jax.random.split(rng, 2)
            value_actions = self._get_on_policy_action(
                online_observation=_model.Observation.from_dict(batch["observation"]),
                policy_state=policy_state,
                rng=policy_sample_rng,
            )
        else:
            value_actions = batch["actions"]
        # Add prefix representation to the batch for the critic
        batch = self._online_batch_to_critic_batch(
            batch,
            policy_state,
        )
        
        # Update the state action critic state
        num_updates = max(self._config.rl.critic.num_updates_per_batch, 1)

        value_batch = (batch[0], value_actions, batch[2], batch[3], batch[4], batch[5])

        for _ in range(num_updates):
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
                value_batch,
            )

        return q_state, value_state, q_info, value_info

    @at.typecheck
    def update(self) -> dict:

        if self._config.rl.critic.pre_training_steps == self.training_steps:
            # Reset optimizer state of the value and q function
            q_opt_state = self._state_action_critic_state.tx.init(
                nnx.filter_state(self._state_action_critic_state.params, nnx.Param)
            )
            new_ema_state_action_critic_params = jax.tree.map(jnp.copy, self._state_action_critic_state.params)
            self._state_action_critic_state = dataclasses.replace(
                self._state_action_critic_state,
                opt_state=q_opt_state,
                ema_params=new_ema_state_action_critic_params,
            )
            del new_ema_state_action_critic_params, q_opt_state

            v_opt_state = self._value_state.tx.init(
                nnx.filter_state(self._value_state.params, nnx.Param)
            )
            new_ema_value_params = jax.tree.map(jnp.copy, self._value_state.params)
            self._value_state = dataclasses.replace(
                self._value_state,
                opt_state=v_opt_state,
                ema_params=new_ema_value_params,
            )
            del new_ema_value_params, v_opt_state

        self.training_steps += 1
        update_critic = (
                self.training_steps >= self._config.rl.critic.training_start_step
                and self.training_steps % self._config.rl.critic.update_interval == 0
        )

        if not update_critic:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }
        if self._config.rl.critic.batch_size:
            critic_batch_size = self._config.rl.critic.batch_size
        else:

            critic_batch_size = int(self._config.batch_size * min(1.0, self._config.rl.online_ratio))

        use_online = self._online_data_buffer.size >= critic_batch_size

        critic_info = {}
        if use_online:
            critic_online_batch = self._online_data_buffer.sample(batch_size=critic_batch_size)
            if update_critic:
                critic_rng, self._rng = jax.random.split(self._rng, 2)
                with sharding.set_mesh(self._mesh):
                    q_state, value_state, q_info, value_info = (
                        self._update_critics_jitted(
                            critic_online_batch,
                            self._state_action_critic_state,
                            self._value_state,
                            self._train_state,
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
