# ruff: noqa: F722
import functools
import logging
from typing import Any, Dict
import gc

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
from src.rl.advantage_weighted_regression.update_actor import (
    train_step as train_actor_step,
)
from src.rl.advantage_weighted_regression.update_critic import (
    init_state_action_critic_train_state,
    init_state_value_train_state,
    train_q_step,
    train_value_step,
    StateActionCriticDef,
    StateValueDef,
    CriticBatch,
)
from src.rl.networks.rl_networks import ObsType, ActionType
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.config import OnlineTrainConfig


def _pytree_size_mb(tree) -> float:
    """Return total size of all arrays in a pytree, in megabytes."""
    leaves = jax.tree.leaves(tree)
    total_bytes = sum(
        leaf.size * leaf.dtype.itemsize for leaf in leaves if hasattr(leaf, "size")
    )
    return total_bytes / (1024 * 1024)


def _pytree_size_gb(tree) -> float:
    return _pytree_size_mb(tree) / 1024


def _pytree_per_device_size_gb(tree) -> float:
    """Return total per-device size of all arrays in a pytree, in GiB."""
    leaves = jax.tree.leaves(tree)
    total_bytes = 0
    for leaf in leaves:
        if not hasattr(leaf, "size"):
            continue
        if hasattr(leaf, "addressable_shards") and leaf.addressable_shards:
            shard = leaf.addressable_shards[0]
            total_bytes += shard.data.size * shard.data.dtype.itemsize
        else:
            total_bytes += leaf.size * leaf.dtype.itemsize
    return total_bytes / (1024**3)


def _log_device_memory(tag: str) -> None:
    """Log live GPU memory for device 0 and count of live arrays."""
    jax.effects_barrier()  # wait for async dispatch to finish
    stats = jax.local_devices()[0].memory_stats()
    if stats is None:
        logging.info(f"[MEM {tag}] memory_stats unavailable")
        return
    live_gb = stats.get("bytes_in_use", 0) / (1024**3)
    peak_gb = stats.get("peak_bytes_in_use", 0) / (1024**3)
    limit_gb = stats.get("bytes_limit", 0) / (1024**3)
    num_live = len(jax.live_arrays())
    logging.info(
        f"[MEM {tag}] live={live_gb:.2f} GiB, peak={peak_gb:.2f} GiB, "
        f"limit={limit_gb:.2f} GiB, num_live_arrays={num_live}"
    )


class AdvantageWeightedFilteredSFTLearner(FilteredSFTLearner):
    def __init__(
        self,
        config: OnlineTrainConfig,
        dummy_obs: ObsType,
        dummy_act: ActionType,
        state_action_critic_def: StateActionCriticDef,
        state_value_def: StateValueDef,
        task_description: str,
    ):
        self.task_description = task_description

        super().__init__(config)
        self._critic_update_frequency = self._get_critic_update_frequency()
        self._policy_update_frequency = self._get_policy_update_frequency()

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
        jax.block_until_ready(self._state_action_critic_state)
        jax.block_until_ready(self._value_state)

        logging.info(
            f"[OOM-DEBUG init] "
            f"train_state: {_pytree_size_mb(self._train_state):.2f} MB, "
            f"state_action_critic_state: {_pytree_size_mb(self._state_action_critic_state):.2f} MB, "
            f"value_state: {_pytree_size_mb(self._value_state):.2f} MB"
        )

        self._q_train_step = jax.jit(
            functools.partial(train_q_step, self._config),
            in_shardings=(
                self._replicated_sharding,
                self._state_action_critic_state_sharding,
                self._value_state_sharding,
                self._data_sharding,
            ),
            out_shardings=(
                self._state_action_critic_state_sharding,
                self._replicated_sharding,
            ),
            donate_argnums=(1,),
        )
        self._value_train_step = jax.jit(
            functools.partial(train_value_step, self._config),
            in_shardings=(
                self._replicated_sharding,
                self._value_state_sharding,
                self._state_action_critic_state_sharding,
                self._data_sharding,
            ),
            out_shardings=(self._value_state_sharding, self._replicated_sharding),
            donate_argnums=(1,),
        )
        del self._train_step
        gc.collect()
        self._train_step = jax.jit(
            functools.partial(train_actor_step, self._config),
            in_shardings=(
                self._replicated_sharding,
                self._train_state_sharding,
                self._state_action_critic_state_sharding,
                self._value_state_sharding,
                self._data_sharding,
            ),
            out_shardings=(
                self._train_state_sharding,
                self._replicated_sharding,
            ),
            donate_argnums=(1,),
        )

    def _get_critic_update_frequency(self) -> int:
        rl_config = getattr(self._config, "rl", None)
        updates = int(getattr(rl_config, "critic_update_frequency", 1))
        return max(1, updates)

    def _get_policy_update_frequency(self) -> int:
        rl_config = getattr(self._config, "rl", None)
        updates = int(getattr(rl_config, "policy_update_frequency", 1))
        return max(1, updates)

    def _recompute_prefix_embedding(
        self,
        *,
        observation: dict[str, Any] | _model.Observation | None,
    ) -> at.Float[at.Array, "batch embed"] | None:
        if observation is None:
            return None
        model = self._get_policy_model()
        # Both SFT-loader Observations and online-buffer dicts are already
        # fully transformed (repack, LiberoInputs, Normalize, tokenize, etc.)
        # by the data pipeline / _preprocess_insert, so we must NOT re-apply
        # _input_transform.  Convert to Observation if needed and go straight
        # to the model.
        if isinstance(observation, _model.Observation):
            obs = observation
        else:
            obs = _model.Observation.from_dict(observation)
        prefix = self._policy._get_prefix_rep_with_model(model, observation=obs)
        prefix = prefix.reshape((prefix.shape[0], -1, prefix.shape[-1]))
        prefix = jnp.mean(prefix, axis=1)
        jax.block_until_ready(prefix)
        return prefix

    def _get_policy_model(self) -> _model.BaseModel:
        """Merge policy params into a model. Call once per update() to avoid duplicates."""
        params = (
            self._train_state.ema_params
            if self._train_state.ema_params is not None
            else self._train_state.params
        )
        model = nnx.merge(self._train_state.model_def, params)
        model.eval()
        return model

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
    ]:
        online_observation = online_batch["observation"]
        observation_dict: dict[str, Any] = {
            "state": online_observation["state"],
        }
        print(f"Type of online observation: {type(online_observation)}")
        curr_prefix_embedding = self._recompute_prefix_embedding(
            observation=online_observation,
        )
        if curr_prefix_embedding is not None:
            observation_dict[PREFIX_EMBEDDING_NAME] = curr_prefix_embedding

        next_observation = online_batch["next_observation"]
        next_observation_dict: dict[str, Any] = {"state": next_observation["state"]}
        print(f"Type of next observation: {type(next_observation)}")
        next_prefix_embedding = self._recompute_prefix_embedding(
            observation=(
                next_observation if isinstance(next_observation, dict) else None
            ),
        )
        if next_prefix_embedding is not None:
            next_observation_dict[PREFIX_EMBEDDING_NAME] = next_prefix_embedding

        return (
            observation_dict,
            online_batch["actions"],
            next_observation_dict,
            online_batch["reward"],
            online_batch["discount"],
        )

    def _sft_batch_to_actor_batch(
        self,
        sft_batch: tuple[_model.Observation, _model.Actions],
    ) -> tuple[_model.Observation, ObsType, _model.Actions]:
        policy_observation, actions = sft_batch
        if isinstance(policy_observation, _model.Observation):
            pass
        elif hasattr(policy_observation, "to_dict"):
            policy_observation = _model.Observation.from_dict(
                dict(policy_observation.to_dict())
            )
        elif isinstance(policy_observation, dict):
            policy_observation = _model.Observation.from_dict(dict(policy_observation))
        else:
            raise TypeError(
                "Unsupported observation type for actor update: "
                f"{type(policy_observation)}."
            )
        policy_obs_dict = policy_observation.to_dict()

        critic_observation: dict[str, Any] = {
            "state": policy_obs_dict["state"],
        }
        print(f"Type of policy observation: {type(policy_observation)}")
        prefix_embedding = self._recompute_prefix_embedding(
            observation=policy_observation,
        )
        if prefix_embedding is not None:
            critic_observation[PREFIX_EMBEDDING_NAME] = prefix_embedding

        return policy_observation, critic_observation, actions

    def save_episode(self, is_success=False, env_index=0, **kwargs):
        return super().save_episode(True, env_index, **kwargs)

    @at.typecheck
    def _update_critics(self, batch: CriticBatch) -> dict[str, at.Array]:
        q_rng, v_rng, self._rng = jax.random.split(self._rng, 3)
        with sharding.set_mesh(self._mesh):
            q_state, q_info = self._q_train_step(
                q_rng,
                self._state_action_critic_state,
                self._value_state,
                batch,
            )
            jax.block_until_ready((q_state, q_info))
            value_state, value_info = self._value_train_step(
                v_rng,
                self._value_state,
                q_state,
                batch,
            )
            jax.block_until_ready((value_state, value_info))
        self._state_action_critic_state = q_state
        self._value_state = value_state
        current_info = {f"critic/q_{key}": value for key, value in q_info.items()} | {
            f"critic/value_{key}": value for key, value in value_info.items()
        }
        return current_info

    def _update_policy(self, batch: tuple[_model.Observation, ObsType, _model.Actions]):
        train_rng, self._rng = jax.random.split(self._rng)
        with sharding.set_mesh(self._mesh):
            train_state, actor_info = self._train_step(
                train_rng,
                self._train_state,
                self._state_action_critic_state,
                self._value_state,
                batch,
            )
        self._train_state = train_state
        info = {f"actor/{key}": value for key, value in actor_info.items()}
        return info

    @at.typecheck
    def update(self) -> dict:
        live_arrays = jax.live_arrays()
        # 3. Calculate total size
        total_bytes = sum(arr.nbytes for arr in live_arrays)
        total_gb = total_bytes / (1024**3)

        print(f"Step {self.training_steps + 1} Memory Check")
        print(f"Total live arrays on device: {len(live_arrays)}")
        print(f"Total tracked memory: {total_gb:.2f} GB")

        self.training_steps += 1
        if self.training_steps % 100 == 1:
            q_size = _pytree_size_mb(self._state_action_critic_state)
            v_size = _pytree_size_mb(self._value_state)
            logging.info(
                f"[OOM-DEBUG step={self.training_steps}] "
                f"state_action_critic_state: {q_size:.2f} MB, "
                f"value_state: {v_size:.2f} MB"
            )
        update_critic = self.training_steps % self._critic_update_frequency == 0
        update_policy = self.training_steps % self._policy_update_frequency == 0
        if not update_critic and not update_policy:
            return {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }

        batch = next(self._data_iter)
        use_online = (
            self._online_data_buffer.size >= self._online_data_buffer.batch_size
        )
        first_online = use_online and self.training_steps <= 2
        if first_online:
            _log_device_memory("before_get_policy_model")
            logging.info(
                f"[SIZE-DEBUG] train_state total (global): "
                f"params={_pytree_size_gb(self._train_state.params):.2f} GiB, "
                f"ema_params={_pytree_size_gb(self._train_state.ema_params):.2f} GiB, "
                f"opt_state={_pytree_size_gb(self._train_state.opt_state):.2f} GiB"
            )
            logging.info(
                f"[SIZE-DEBUG] train_state per-device: "
                f"params={_pytree_per_device_size_gb(self._train_state.params):.2f} GiB, "
                f"ema_params={_pytree_per_device_size_gb(self._train_state.ema_params):.2f} GiB, "
                f"opt_state={_pytree_per_device_size_gb(self._train_state.opt_state):.2f} GiB"
            )
            logging.info(
                f"[SIZE-DEBUG] SFT batch: global={_pytree_size_gb(batch):.2f} GiB, "
                f"per-device={_pytree_per_device_size_gb(batch):.2f} GiB"
            )

        # Create the policy model at most once per update() call.
        # needs_model = (use_online and update_critic) or update_policy

        critic_info, actor_info = {}, {}
        if use_online:
            online_batch_raw = self._online_data_buffer.sample()
            if first_online:
                logging.info(
                    f"[SIZE-DEBUG] online_batch_raw: global={_pytree_size_gb(online_batch_raw):.2f} GiB, "
                    f"per-device={_pytree_per_device_size_gb(online_batch_raw):.2f} GiB"
                )
            if update_critic:
                if first_online:
                    _log_device_memory("before_critic_batch")
                critic_batch = self._online_batch_to_critic_batch(
                    online_batch_raw,
                )
                if first_online:
                    _log_device_memory("after_critic_batch")
                critic_info = self._update_critics(critic_batch)
                if first_online:
                    _log_device_memory("after_update_critics")
                # del critic_batch
            online_batch = self._online_batch_to_sft_batch(online_batch_raw)
            # del online_batch_raw
            online_ratio = float(getattr(self._config.collect, "online_ratio", 0.5))
            if online_ratio >= 1.0:
                batch = online_batch
            elif online_ratio > 0:
                # Mix online and offline into a fixed-size batch instead of
                # concatenating (which would double the batch and OOM).
                first_leaf = jax.tree.leaves(batch)[0]
                batch_size = first_leaf.shape[0]
                n_online = min(
                    int(batch_size * online_ratio),
                    jax.tree.leaves(online_batch)[0].shape[0],
                )
                n_offline = batch_size - n_online
                batch = jax.tree.map(
                    lambda x, y: jnp.concatenate([x[:n_offline], y[:n_online]], axis=0),
                    batch,
                    online_batch,
                )
                # The online batch may be replicated (PartitionSpec()) while
                # the SFT batch is sharded. Re-shard the mixed result to
                # match the data sharding expected by _train_step.
                batch = jax.device_put(batch, self._data_sharding)
                # online_batch is replicated (~11 GiB/device of images); free it now.
                # del online_batch
        if update_policy:
            if first_online:
                _log_device_memory("before_actor_batch")
            actor_batch = self._sft_batch_to_actor_batch(
                batch,
            )
            if first_online:
                logging.info(
                    f"[SIZE-DEBUG] actor_batch (global): "
                    f"policy_obs={_pytree_size_gb(actor_batch[0]):.2f} GiB, "
                    f"critic_obs={_pytree_size_gb(actor_batch[1]):.2f} GiB, "
                    f"actions={_pytree_size_gb(actor_batch[2]):.2f} GiB"
                )
                logging.info(
                    f"[SIZE-DEBUG] actor_batch (per-device): "
                    f"policy_obs={_pytree_per_device_size_gb(actor_batch[0]):.2f} GiB, "
                    f"critic_obs={_pytree_per_device_size_gb(actor_batch[1]):.2f} GiB, "
                    f"actions={_pytree_per_device_size_gb(actor_batch[2]):.2f} GiB"
                )
        # Free batch and model copy BEFORE the heavy jitted train step so JAX
        # can reclaim memory and donate the train_state buffers.
        if update_policy:
            if first_online:
                _log_device_memory(
                    "before_update_policy (after del batch+policy_model)"
                )
            actor_info = self._update_policy(actor_batch)
            jax.block_until_ready((self._train_state, actor_info))
        info = (
            actor_info
            | critic_info
            | {
                "online_buffer_size": jnp.asarray(
                    float(self._online_data_buffer.size), dtype=jnp.float32
                )
            }
        )
        info = jax.tree.map(np.asarray, info)
        print(f"Tree size: {_pytree_size_mb(info):.2f} MB")
        return info
