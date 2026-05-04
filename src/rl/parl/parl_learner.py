import dataclasses
from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import AdvantageWeightedSFTLearner
from src.rl.filtered_sft_agent.update import train_step as train_sft_step
from src.rl.filtered_sft_agent.filtered_sft_learner import _copy_nnx_state
from src.rl.replay_buffer import ShardedReplayBuffer
import openpi.models.model as _model
import openpi.training.sharding as sharding
import jax
import functools
import numpy as np
import jax.numpy as jnp
from typing import Any, Dict


class PARLAgentWrapper(object):
    def __init__(self, base_agent: AdvantageWeightedSFTLearner):
        self._base_agent = base_agent
        self._config = self._base_agent._config
        self._parl_config = self._config.parl
        self._sft_buffer = self._setup_sft_buffer()
        # prepare train_step
        self._refresh_train_sft_step()

    def __getattr__(self, name):
        """
        Catches ANY attribute or method not defined in this wrapper
        (e.g., `data_sharding`, `_mesh`, `_train_state`, `batch_stats`)
        and forwards the request to the base agent.
        """
        # Prevent infinite recursion for magic methods
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return getattr(self.base_agent, name)

    def _setup_sft_buffer(
        self,
    ) -> ShardedReplayBuffer:
        # prepare dummy data for initializing the replay buffer
        obs_spec, act_spec = self._config.model.inputs_spec(batch_size=1)
        obs_spec_dict = obs_spec.to_dict()
        dummy_obs_dict = jax.tree.map(
            lambda spec: np.zeros(spec.shape, dtype=spec.dtype), obs_spec_dict
        )
        dummy_obs_dict = {k: v for k, v in dummy_obs_dict.items() if v is not None}
        if "image" in dummy_obs_dict:
            dummy_obs_dict["image"] = jax.tree.map(
                lambda v: v.astype(np.uint8), dummy_obs_dict["image"]
            )
        dummy_data = {
            "observation": dummy_obs_dict,
            "actions": np.zeros(act_spec.shape, dtype=act_spec.dtype),
        }

        return ShardedReplayBuffer(
            dummy_data=dummy_data,
            max_capacity=self._config.rl.buffer_capacity,
            data_sharding=self._base_agent._data_sharding,
            seed=self._config.seed,
            preprocess_fn=None,
            postprocess_fn=None,
            freeze_dict=False,
        )

    def _refresh_train_sft_step(self):
        self._train_sft_step = jax.jit(
            functools.partial(train_sft_step, self._config),
            in_shardings=(
                self._base_agent._replicated_sharding,
                self._base_agent._train_state_sharding,
                self._base_agent._data_sharding,
            ),
            out_shardings=(self._base_agent._train_state_sharding, self._base_agent._replicated_sharding),
            donate_argnums=(1,),
        )

    def save_episode(self, is_success: bool, env_index: int, task_description: str):

        assert env_index in range(
            len(self._base_agent._episode_storage)
        ), f"env_index must be between 0 and {len(self._base_agent._episode_storage) - 1}, but got {env_index}."
        # extract episode data from storage and empty it
        episode_data = self._base_agent._episode_storage[env_index]
        if self._parl_config.store_success_data_only:
            if is_success:
                self._save_episode_in_sft_buffer(episode_data, task_description)
        else:
            self._save_episode_in_sft_buffer(episode_data, task_description)
        self._base_agent.save_episode(is_success=is_success, env_index=env_index, task_description=task_description)

    def _save_episode_in_sft_buffer(self, episode_data, task_description):
        # concatenate all chunks
        episode_data = jax.tree_util.tree_map(
            lambda *xs: np.concatenate(xs, axis=0), *episode_data
        )
        done = np.logical_or(episode_data["terminate"], episode_data["truncate"])
        n_steps = np.where(done)[0][0] + 1
        act_h = int(self._config.model.action_horizon)
        n_windows = n_steps - act_h + 1
        if n_windows <= 0:
            return

        # process elements to account for action chunks
        _obs = {self._base_agent.obs_key_process_fn(k): v[:n_windows] for k, v in episode_data["observation"].items()}
        _actions = np.stack([episode_data["action"][start: start + act_h] for start in range(n_windows)])
        _actions = self._base_agent.post_step_action_filter(_actions)

        def transform(input):
            obs = self._base_agent._policy_transforms(input)
            actions = obs.pop("actions")
            return obs, actions

        # process observations and actions according to pi0 preprocessing
        _obs, _actions = transform({**_obs, "actions": np.array(_actions, copy=True),
                                    "prompt": str(task_description)})

        self._sft_buffer.insert(
            {
                "observation": _obs,
                "actions": _actions.astype(np.float32),
            }
        )

    def _online_batch_to_sft_batch(
        self, online_batch: Dict[str, Any]
    ) -> tuple[_model.Observation, _model.Actions]:
        return (
            _model.Observation.from_dict(online_batch["observation"]),
            online_batch["actions"],
        )

    def update(self):
        self._base_agent.training_steps += 1
        filtered_sft_update = self._base_agent.training_steps % self._parl_config.sft_update_every == 0
        if not filtered_sft_update:
            return self._base_agent.update()
        else:
            update_policy = (
                    self._base_agent.training_steps >= self._parl_config.policy_training_start_step
                    and self._base_agent.training_steps % self._parl_config.policy_update_interval == 0
            )
            if not update_policy:
                return {"online_buffer_size": self._base_agent._online_data_buffer.size,
                        "sft_buffer_size": self._sft_buffer.size}
            if self._sft_buffer.size == 0:
                return {}
            online_ratio = self._parl_config.online_ratio
            if online_ratio >= 1.0:
                batch = self._sft_buffer.sample(
                    batch_size=self._config.batch_size,
                )
                batch = self._online_batch_to_sft_batch(batch)
            else:
                batch = next(self._base_agent._data_iter)
                if online_ratio > 0.0:
                    first_leaf = jax.tree.leaves(batch)[0]
                    batch_size = first_leaf.shape[0]
                    n_online = int(self._config.batch_size * min(1.0, online_ratio))
                    n_offline = batch_size - n_online
                    online_batch_raw = self._sft_buffer.sample(
                        batch_size=n_online
                    )
                    online_batch = self._online_batch_to_sft_batch(online_batch_raw)
                    batch = jax.tree.map(
                        lambda x, y: jnp.concatenate([x[:n_offline], y[:n_online]], axis=0),
                        batch,
                        online_batch,
                    )

            train_rng, self._base_agent._rng = jax.random.split(self._base_agent._rng)
            with sharding.set_mesh(self._base_agent._mesh):
                policy_state, info = self._train_sft_step(train_rng, self._base_agent._train_state, batch)
            info = {f"parl_actor/{key}": value for key, value in info.items()}
            self._base_agent._train_state = policy_state
            if self._base_agent._resume_restore_ema:
                self._base_agent._train_state = dataclasses.replace(
                    self._base_agent._train_state,
                    ema_decay=self._base_agent._resume_ema_decay,
                    ema_params=_copy_nnx_state(self._base_agent_train_state.params),
                )
                self._base_agent._resume_restore_ema = False
                self._base_agent._resume_ema_decay = None
                self._base_agent._refresh_train_step()
                self._refresh_train_sft_step()
            info = info | {
                "online_buffer_size": jnp.asarray(
                    float(self._base_agent._online_data_buffer.size), dtype=jnp.float32
                ),
                "sft_buffer_size": jnp.asarray(
                    float(self._sft_buffer.size), dtype=jnp.float32
                ),
            }
            return info


