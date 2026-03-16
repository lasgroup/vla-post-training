from __future__ import annotations

import orbax.checkpoint as ocp
import functools
import logging
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.experimental import mesh_utils

import etils.epath as epath
import openpi.training.checkpoints as _checkpoints
from src.rl.agent import Agent
from src.rl.dsrl.update_actor import (
    init_policy_state,
    train_actor_step,
    PolicyDef,
)
from src.rl.dsrl.update_alpha import (
    alpha_autotune_enabled,
    alpha_value,
    init_alpha_state,
    resolve_target_entropy,
    train_alpha_step,
)
from src.rl.dsrl.update_critic import (
    StateActionCriticDef,
    init_state_action_critic_train_state,
    train_q_step,
)
from src.rl.dsrl.chunk_obs import (
    expected_chunk_action_shape,
    normalize_action_batch_shape,
    normalize_observation_for_model,
)
from src.rl.dsrl.dsrl_utils import (
    _resize_image_np,
    _copy_with_batch_dim,
    _extract_replay_observation,
    _finalize_replay_observation,
    _build_replay_observation_template,
)
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.networks.rl_networks import ActionType, ObsType
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig

logger = logging.getLogger(__name__)


class DSRLLearner(Agent):

    # Subclasses override to change the checkpoint subdirectory name.
    _ckpt_subdir: str = "dsrl_state"

    def __init__(
        self,
        config: OnlineTrainConfig,
        dummy_obs: ObsType,
        dummy_act: ActionType,
        state_action_critic_def: StateActionCriticDef,
        policy_def: PolicyDef,
        task_description: str,
    ):
        self._config = config
        self._rng = jax.random.key(config.seed)
        devices = mesh_utils.create_device_mesh((jax.device_count(),))
        self._mesh = jax.sharding.Mesh(devices, axis_names=("batch",))
        self._checkpoint_manager, self._resuming = (
            _checkpoints.initialize_checkpoint_dir(
                self._config.checkpoint_dir,
                keep_period=self._config.keep_period,
                overwrite=self._config.overwrite,
                resume=self._config.resume,
            )
        )
        self._sac_image_size = int(getattr(config.rl, "sac_image_size", 64))
        self._crop_padding = int(getattr(config.rl, "random_crop_padding", 4))

        raw_dummy_obs = jax.tree.map(lambda x: np.asarray(x), dummy_obs)
        replay_dummy_obs = _build_replay_observation_template(
            raw_dummy_obs,
            obs_prefix="pi0",
            sac_image_size=self._sac_image_size,
        )

        dummy_obs_normalized = self._normalize_obs(raw_dummy_obs)
        if self._sac_image_size > 0 and isinstance(dummy_obs_normalized, dict):
            for img_key in ("image", "wrist_image"):
                if img_key in dummy_obs_normalized:
                    dummy_obs_normalized[img_key] = jnp.asarray(
                        _resize_image_np(
                            np.asarray(dummy_obs_normalized[img_key]), self._sac_image_size
                        )
                    )
        self._dummy_obs = dummy_obs_normalized
        self._dummy_act = dummy_act
        self._expected_action_shape = expected_chunk_action_shape(np.asarray(dummy_act))
        self._action_dim = int(np.prod(np.asarray(dummy_act).shape[1:]))

        replay_dummy_obs = jax.tree_util.tree_map(lambda x: np.asarray(x), replay_dummy_obs)
        rl_cfg = getattr(self._config, "rl", None)
        self._online_data_buffer = ShardedReplayBuffer(
            dummy_data={
                "observation": replay_dummy_obs,
                "actions": np.zeros((1, self._action_dim), dtype=np.float32),
                "next_observation": replay_dummy_obs,
                "reward": np.zeros((1,), dtype=np.float32),
                "mc_return": np.zeros((1,), dtype=np.float32),
                "discount": np.zeros((1,), dtype=np.float32),
            },
            max_capacity=int(getattr(rl_cfg, "buffer_capacity", 100_000)),
            batch_size=int(getattr(self._config, "batch_size", 128)),
            data_sharding=None,
            seed=int(getattr(self._config, "seed", 0)),
            preprocess_fn=None,
            postprocess_fn=None,
            freeze_dict=False,
            load_paths=list(getattr(rl_cfg, "buffer_load_paths", ())),
            save_path=getattr(rl_cfg, "buffer_save_path", None),
        )
        self.replay = self._online_data_buffer
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

        self._use_random_noise = bool(getattr(rl_cfg, "warmup_gaussian_noise", True))

        q_init_rng, policy_init_rng, alpha_init_rng, self._rng = jax.random.split(self._rng, 4)
        self._state_action_critic_state, self._state_action_critic_state_sharding = (
            init_state_action_critic_train_state(
                self._config,
                q_init_rng,
                self._mesh,
                critic_def=state_action_critic_def,
                dummy_obs=dummy_obs_normalized,
                dummy_act=dummy_act,
                use_sharding=False,
            )
        )

        self._policy_state, self._policy_state_sharding = init_policy_state(
            self._config,
            policy_init_rng,
            self._mesh,
            policy_def=policy_def,
            dummy_obs=dummy_obs_normalized,
            dummy_act=dummy_act,
            use_sharding=False,
        )
        self._alpha_state, self._alpha_state_sharding = init_alpha_state(
            self._config,
            alpha_init_rng,
            self._mesh,
            use_sharding=False,
        )
        self._target_entropy = resolve_target_entropy(self._config, self._action_dim)
        self._autotune_alpha = alpha_autotune_enabled(self._config)

        self.training_steps = 0

        ckpt_dir = epath.Path(self._config.checkpoint_dir) / self._ckpt_subdir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = ckpt_dir
        self._checkpointer = ocp.StandardCheckpointer()

        if self._resuming:
            self._restore_checkpoint()

        def _sample_policy_actions(params, obs, rng):
            policy = nnx.merge(self._policy_state.model_def, params)
            policy.eval()
            dist = policy(obs)
            return dist.sample(seed=rng)

        def _eval_policy_actions(params, obs):
            policy = nnx.merge(self._policy_state.model_def, params)
            policy.eval()
            dist = policy(obs)
            if hasattr(dist, "mode"):
                return dist.mode()
            if hasattr(dist, "mean"):
                return dist.mean()
            return dist.sample(seed=jax.random.PRNGKey(0))

        self._sample_policy_actions_jit = jax.jit(_sample_policy_actions)
        self._eval_policy_actions_jit = jax.jit(_eval_policy_actions)
        warmup_obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), dummy_obs_normalized)
        warmup_rng = jax.random.fold_in(self._rng, 0)
        _ = jax.block_until_ready(
            self._sample_policy_actions_jit(self._policy_state.params, warmup_obs, warmup_rng)
        )
        _ = jax.block_until_ready(
            self._eval_policy_actions_jit(self._policy_state.params, warmup_obs)
        )
        self._train_critic_step = jax.jit(functools.partial(train_q_step, self._config))
        self._train_actor_step = jax.jit(functools.partial(train_actor_step, self._config))
        self._train_alpha_step = jax.jit(functools.partial(train_alpha_step, self._config))

        logger.info(
            "%s init: action_dim=%d, sac_image_size=%d, crop_padding=%d, "
            "target_entropy=%.3f, warmup_noise=%s",
            type(self).__name__, self._action_dim, self._sac_image_size, self._crop_padding,
            self._target_entropy, self._use_random_noise,
        )

    # ------------------------------------------------------------------
    # Observation normalization (override in subclasses)
    # ------------------------------------------------------------------

    def _normalize_obs(self, obs: Any) -> Any:
        """Normalize raw observations for the SAC actor/critic networks."""
        return normalize_observation_for_model(obs)

    def _normalize_replay_obs(self, obs_dict: Dict[str, Any]) -> Any:
        """Normalize a replay-buffer observation dict for the SAC model."""
        return normalize_observation_for_model(obs_dict)

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def _sample_action(
        self,
        observations: Dict[str, Any] | np.ndarray,
        rng: jax.random.PRNGKey,
        *,
        deterministic: bool = False,
        batch_actions: bool = True,
    ) -> np.ndarray:
        processed_obs = self._normalize_obs(observations)
        if self._sac_image_size > 0 and isinstance(processed_obs, dict):
            for img_key in ("image", "wrist_image"):
                if img_key in processed_obs:
                    processed_obs[img_key] = _resize_image_np(
                        np.asarray(processed_obs[img_key]), self._sac_image_size
                    )
        obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), processed_obs)

        if deterministic:
            sampled_actions = self._eval_policy_actions_jit(self._policy_state.params, obs)
        else:
            sampled_actions = self._sample_policy_actions_jit(self._policy_state.params, obs, rng)

        actions = np.asarray(sampled_actions, dtype=np.float32)
        actions = self._post_process_actions(actions)

        if not batch_actions:
            return actions

        if actions.ndim == 1:
            actions = actions[None, ...]
        actions = normalize_action_batch_shape(actions, self._expected_action_shape)
        return actions

    def _post_process_actions(self, actions: np.ndarray) -> np.ndarray:
        """Hook for subclasses to scale/transform sampled actions."""
        return actions

    def _generate_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        deterministic = kwargs.get("deterministic")
        batch_actions = kwargs.get("batch_actions")
        if deterministic is None:
            deterministic = False
        if batch_actions is None:
            batch_actions = True
        rng, self._rng = jax.random.split(self._rng)

        # Gaussian noise for first collection round
        if self._use_random_noise and not deterministic:
            obs_leaves = jax.tree_util.tree_leaves(observations)
            if obs_leaves:
                batch_dim = np.asarray(obs_leaves[0]).shape[0]
            else:
                batch_dim = 1
            noise = np.asarray(
                jax.random.normal(rng, (batch_dim, self._action_dim)),
                dtype=np.float32,
            )
            if batch_actions:
                noise = normalize_action_batch_shape(noise, self._expected_action_shape)
            return noise

        return np.asarray(
            self._sample_action(
                observations=observations,
                rng=rng,
                deterministic=bool(deterministic),
                batch_actions=bool(batch_actions),
            ),
            dtype=np.float32,
        )

    def eval_actions(self, observations, **kwargs):
        return self._generate_actions(observations, deterministic=True, **kwargs)

    def sample_actions(self, observations, **kwargs):
        return self._generate_actions(observations, deterministic=False, **kwargs)

    def add_data(self, step_data: StepData):
        def get_env_value(vec, env_id):
            return jax.tree.map(lambda x: x[env_id], vec)

        for i in range(self._config.collect.env_num):
            self._episode_storage[i].append(get_env_value(step_data, i))

    def _current_alpha(self) -> jax.Array:
        return alpha_value(self._alpha_state)

    def _update_alpha(self, entropy: jax.Array) -> dict[str, jax.Array]:
        entropy = jnp.asarray(entropy, dtype=jnp.float32)
        if not self._autotune_alpha:
            alpha = self._current_alpha()
            return {
                "alpha": alpha,
                "alpha_loss": jnp.asarray(0.0, dtype=jnp.float32),
                "entropy": entropy,
                "log_prob_mean": -entropy,
                "target_entropy": jnp.asarray(self._target_entropy, dtype=jnp.float32),
            }

        train_rng, self._rng = jax.random.split(self._rng)
        self._alpha_state, alpha_info = self._train_alpha_step(
            train_rng,
            self._alpha_state,
            entropy,
            jnp.asarray(self._target_entropy, dtype=jnp.float32),
        )
        return alpha_info

    # ------------------------------------------------------------------
    # Image augmentation
    # ------------------------------------------------------------------

    def _random_crop_single(self, rng: jax.Array, img: jax.Array, padding: int = 4) -> jax.Array:
        crop_from = jax.random.randint(rng, (2,), 0, 2 * padding + 1)
        crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
        padded = jnp.pad(
            img,
            ((padding, padding), (padding, padding), (0, 0)),
            mode="edge",
        )
        return jax.lax.dynamic_slice(padded, crop_from, img.shape)

    def _batched_random_crop(self, rng: jax.Array, imgs: jax.Array, padding: int = 4) -> jax.Array:
        keys = jax.random.split(rng, imgs.shape[0])
        return jax.vmap(lambda k, i: self._random_crop_single(k, i, padding))(keys, imgs)

    def _augment_images(self, obs_dict: Dict[str, jax.Array], rng: jax.Array) -> tuple[Dict[str, jax.Array], jax.Array]:
        if self._crop_padding <= 0:
            return obs_dict, rng
        augmented = dict(obs_dict)
        for img_key in ("image", "wrist_image"):
            if img_key not in augmented:
                continue
            img = augmented[img_key]
            if img.ndim == 4 and img.shape[1] > 1 and img.shape[2] > 1:
                rng, crop_rng = jax.random.split(rng)
                augmented[img_key] = self._batched_random_crop(
                    crop_rng, img, padding=self._crop_padding
                )
        return augmented, rng

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self):
        self.training_steps += 1

        batch_size = int(getattr(self._config, "batch_size", 128))
        if self._online_data_buffer.size < batch_size:
            return {}

        if self._use_random_noise:
            self._use_random_noise = False
            logger.info(
                "Disabling warmup Gaussian noise (buffer size=%d >= batch_size=%d).",
                self._online_data_buffer.size, batch_size,
            )

        info = {}
        latest_actor_observation = None

        if self.training_steps % int(getattr(self._config.rl, "critic_update_frequency", 1)) == 0:
            if batch_size != int(self._online_data_buffer.batch_size):
                raise ValueError(
                    "Configured train batch_size does not match replay batch_size: "
                    f"{batch_size} vs {self._online_data_buffer.batch_size}."
                )
            batch = self._online_data_buffer.sample()
            batch_observation = self._normalize_replay_obs(batch["observation"])
            batch_next_observation = self._normalize_replay_obs(batch["next_observation"])

            observation = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), batch_observation)
            next_observation = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), batch_next_observation)

            aug_rng, self._rng = jax.random.split(self._rng)
            observation, aug_rng = self._augment_images(observation, aug_rng)
            next_observation, aug_rng = self._augment_images(next_observation, aug_rng)

            actions = jnp.asarray(batch["actions"], dtype=jnp.float32)
            reward = jnp.asarray(batch["reward"], dtype=jnp.float32)
            discount = jnp.asarray(batch["discount"], dtype=jnp.float32)

            critic_batch = (observation, actions, next_observation, reward, discount)

            train_rng, self._rng = jax.random.split(self._rng)
            self._state_action_critic_state, critic_info = self._train_critic_step(
                train_rng,
                self._state_action_critic_state,
                self._policy_state,
                critic_batch,
                self._current_alpha(),
            )
            latest_actor_observation = observation
            info.update({f"critic/{k}": v for k, v in critic_info.items()})

        if self.training_steps % int(getattr(self._config.rl, "actor_update_frequency", 1)) == 0:
            if latest_actor_observation is None:
                batch = self._online_data_buffer.sample()
                batch_observation = self._normalize_replay_obs(batch["observation"])
                observation = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), batch_observation)
                aug_rng, self._rng = jax.random.split(self._rng)
                observation, _ = self._augment_images(observation, aug_rng)
            else:
                observation = latest_actor_observation

            train_rng, self._rng = jax.random.split(self._rng)
            self._policy_state, actor_info = self._train_actor_step(
                train_rng,
                self._policy_state,
                self._state_action_critic_state,
                observation,
                self._current_alpha(),
            )
            info.update({f"actor/{k}": v for k, v in actor_info.items()})
            if "entropy" in actor_info:
                alpha_info = self._update_alpha(actor_info["entropy"])
                info.update({f"alpha/{k}": v for k, v in alpha_info.items()})
            elif "log_prob_mean" in actor_info:
                alpha_info = self._update_alpha(-actor_info["log_prob_mean"])
                info.update({f"alpha/{k}": v for k, v in alpha_info.items()})

        info.setdefault("alpha/value", self._current_alpha())
        return info

    # ------------------------------------------------------------------
    # Episode saving with sparse -1/0 reward
    # ------------------------------------------------------------------

    def save_episode(self, is_success: bool = False, env_index: int = 0, **kwargs):
        """Process a completed episode and insert transitions into the replay buffer.

        Uses the DSRL sparse reward scheme:
        - Every step gets reward = -1
        - Last step of a *successful* episode gets reward = 0
        - Discount = gamma^query_freq for non-terminal steps
        - Discount = 0 for the terminal step of a successful episode
        """
        episode = self._episode_storage[env_index]
        self._episode_storage[env_index] = []

        if not episode:
            return

        obs_prefix = "pi0"
        query_freq = int(getattr(self._config.collect, "replan_steps", 1))
        bootstrap_discount = float(self._config.rl.discount) ** query_freq
        episode_len = len(episode)

        env_max_reward = float(getattr(self._config.collect, "env_max_reward", 0.0))
        if env_max_reward > 0 and episode:
            final_ep = episode[-1]
            final_reward = np.asarray(
                final_ep.get("reward", 0.0), dtype=np.float32
            ).reshape(-1)
            is_success = bool(np.max(final_reward) >= env_max_reward)

        rewards = np.full((episode_len,), -1.0, dtype=np.float32)
        discounts = np.full((episode_len,), bootstrap_discount, dtype=np.float32)
        if is_success and episode_len > 0:
            rewards[-1] = 0.0
            discounts[-1] = 0.0

        mc_returns = np.zeros((episode_len,), dtype=np.float32)
        running_return = np.float32(0.0)
        for idx in reversed(range(episode_len)):
            running_return = rewards[idx] + discounts[idx] * running_return
            mc_returns[idx] = running_return

        for idx, ep in enumerate(episode):
            obs = _extract_replay_observation(
                ep["observation"],
                obs_prefix=obs_prefix,
                sac_image_size=self._sac_image_size,
            )
            next_obs_raw = ep.get("next_observation", ep["observation"])
            try:
                next_obs_extracted = _extract_replay_observation(
                    next_obs_raw,
                    obs_prefix=obs_prefix,
                    sac_image_size=self._sac_image_size,
                )
            except (TypeError, KeyError):
                next_obs_extracted = obs
            next_obs = _finalize_replay_observation(obs, next_obs_extracted)

            act = ep.get("action", ep.get("actions"))
            if act is None:
                raise KeyError("Episode transition is missing `action` / `actions`.")
            policy_actions = np.asarray(act, dtype=np.float32).reshape(1, -1)

            self._online_data_buffer.insert(
                {
                    "observation": jax.tree_util.tree_map(_copy_with_batch_dim, obs),
                    "actions": policy_actions,
                    "next_observation": jax.tree_util.tree_map(_copy_with_batch_dim, next_obs),
                    "reward": np.asarray([rewards[idx]], dtype=np.float32),
                    "mc_return": np.asarray([mc_returns[idx]], dtype=np.float32),
                    "discount": np.asarray([discounts[idx]], dtype=np.float32),
                }
            )

        self._collection_success_episodes += int(is_success)

    def start_data_collection(self, step: int | None = None):
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

    def end_data_collection(self, step: int | None = None) -> int:
        collected_episodes = int(self._collection_success_episodes)
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0
        return collected_episodes

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            "training_steps": np.asarray(self.training_steps, dtype=np.int32),
            "rng": self._rng,
            "policy_state": self._policy_state,
            "state_action_critic_state": self._state_action_critic_state,
            "alpha_state": self._alpha_state,
            "use_random_noise": np.asarray(self._use_random_noise, dtype=np.bool_),
        }

    def save_checkpoint(self, step: int | None = None):
        if step is None:
            step = int(self.training_steps)

        path = self._ckpt_dir / str(step)
        state = self._checkpoint_state()

        if path.exists():
            return

        self._checkpointer.save(path, state)

    def _restore_checkpoint(self):
        if not self._ckpt_dir.exists():
            logger.info("No %s checkpoint directory found, starting from scratch.", self._ckpt_subdir)
            return

        steps = []
        for p in self._ckpt_dir.iterdir():
            if p.is_dir():
                try:
                    steps.append(int(p.name))
                except ValueError:
                    pass

        if not steps:
            logger.info("No %s checkpoints found, starting from scratch.", self._ckpt_subdir)
            return

        step = max(steps)
        path = self._ckpt_dir / str(step)

        restored = self._checkpointer.restore(
            path,
            self._checkpoint_state(),
        )

        self.training_steps = int(restored["training_steps"])
        self._rng = restored["rng"]
        self._policy_state = restored["policy_state"]
        self._state_action_critic_state = restored["state_action_critic_state"]
        self._alpha_state = restored["alpha_state"]
        self._use_random_noise = bool(restored["use_random_noise"])

        logger.info("Restored %s checkpoint from step %d", self._ckpt_subdir, step)
