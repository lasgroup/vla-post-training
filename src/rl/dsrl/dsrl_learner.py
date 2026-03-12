from __future__ import annotations

import functools
import logging
from typing import Any, Dict

import numpy as np
import PIL.Image
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.experimental import mesh_utils

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
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.networks.rl_networks import ActionType, ObsType
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_image_np(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a single HxWxC uint8 image to target_size x target_size."""
    if img.ndim == 3:
        h, w = img.shape[0], img.shape[1]
        if h == target_size and w == target_size:
            return img
        return np.array(
            PIL.Image.fromarray(img).resize((target_size, target_size)),
            dtype=img.dtype,
        )
    # Handle arbitrary leading dims (batch, temporal, etc.) by recursing.
    if img.ndim >= 4:
        return np.stack([_resize_image_np(img[i], target_size) for i in range(img.shape[0])])
    return img


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------

def _extract_replay_observation(
    observation: Any,
    *,
    obs_prefix: str,
    sac_image_size: int = 0,
) -> Dict[str, np.ndarray]:
    """Convert env observation payloads into replay observations.

    Returns a dict with:
    - ``state`` (required)
    - optional ``image``, ``wrist_image`` (resized to *sac_image_size* when > 0)

    - optional `prefix_embedding`
    """
    if not isinstance(observation, dict):
        raise TypeError(
            f"Expected dict observation for replay extraction, got {type(observation)}."
        )

    obs_dict = (
        observation["observation"]
        if isinstance(observation.get("observation"), dict)
        else observation
    )

    def _get_from_obs(*keys: str) -> Any | None:
        for key in keys:
            if key in obs_dict:
                return obs_dict[key]
        return None

    extracted: Dict[str, np.ndarray] = {}

    image = _get_from_obs(
        f"{obs_prefix}/image",
        "image",
        "observation/image",
        "observation/exterior_image_1_left",
        "exterior_image_1_left",
        "pixels",
    )
    if image is not None:
        img = np.asarray(image)
        if sac_image_size > 0:
            img = _resize_image_np(img, sac_image_size)
        extracted["image"] = np.asarray(img, dtype=np.uint8)

    wrist_image = _get_from_obs(
        f"{obs_prefix}/wrist_image",
        "wrist_image",
        "observation/wrist_image",
        "observation/wrist_image_left",
        "wrist_image_left",
    )
    if wrist_image is not None:
        wimg = np.asarray(wrist_image)
        if sac_image_size > 0:
            wimg = _resize_image_np(wimg, sac_image_size)
        extracted["wrist_image"] = np.asarray(wimg, dtype=np.uint8)

    state = _get_from_obs(
        f"{obs_prefix}/state",
        "state",
        "observation/state",
    )
    if state is None:
        joint_position = _get_from_obs(
            "observation/joint_position",
            "joint_position",
        )
        gripper_position = _get_from_obs(
            "observation/gripper_position",
            "gripper_position",
        )
        if joint_position is not None and gripper_position is not None:
            state = np.concatenate(
                [
                    np.asarray(joint_position, dtype=np.float32),
                    np.asarray(gripper_position, dtype=np.float32),
                ],
                axis=-1,
            )
    if state is None:
        raise KeyError(
            "Replay extraction requires a state vector. Expected one of "
            f"['{obs_prefix}/state', 'state', 'observation/state']."
        )
    extracted["state"] = np.asarray(state, dtype=np.float32)

    # Remove for testing
    # for src in (observation, obs_dict):
    #     if PREFIX_EMBEDDING_NAME in src:
    #         extracted[PREFIX_EMBEDDING_NAME] = np.asarray(
    #             src[PREFIX_EMBEDDING_NAME], dtype=np.float32
    #         )
    #         break
    #     if "prefix_rep" in src:
    #         extracted[PREFIX_EMBEDDING_NAME] = np.asarray(
    #             src["prefix_rep"], dtype=np.float32
    #         )
    #         break

    return extracted


def _finalize_replay_observation(
    current_obs: Dict[str, np.ndarray],
    next_obs: Dict[str, np.ndarray] | None,
) -> Dict[str, np.ndarray]:
    """Fill next-observation fields from current observation when missing."""
    if next_obs is None:
        next_obs = {}

    merged: Dict[str, np.ndarray] = {
        "state": np.asarray(next_obs.get("state", current_obs["state"]), dtype=np.float32)
    }
    if "image" in next_obs or "image" in current_obs:
        merged["image"] = np.asarray(
            next_obs.get("image", current_obs.get("image")), dtype=np.uint8
        )
    if "wrist_image" in next_obs or "wrist_image" in current_obs:
        merged["wrist_image"] = np.asarray(
            next_obs.get("wrist_image", current_obs.get("wrist_image")), dtype=np.uint8
        )
    # Remove for testing
    # prefix = next_obs.get(PREFIX_EMBEDDING_NAME, current_obs.get(PREFIX_EMBEDDING_NAME))
    # if prefix is not None:
    #     merged[PREFIX_EMBEDDING_NAME] = np.asarray(prefix, dtype=np.float32)
    return merged


def _build_replay_observation_template(
    observation: Any,
    *,
    obs_prefix: str,
    sac_image_size: int = 0,
) -> Dict[str, np.ndarray]:
    """Build fixed-shape replay template preserving image/state modalities."""
    extracted = _extract_replay_observation(observation, obs_prefix=obs_prefix, sac_image_size=sac_image_size,)
    template: Dict[str, np.ndarray] = {}
    if "image" in extracted:
        template["image"] = np.zeros_like(np.asarray(extracted["image"]), dtype=np.uint8)
    if "wrist_image" in extracted:
        template["wrist_image"] = np.zeros_like(np.asarray(extracted["wrist_image"]), dtype=np.uint8)
    template["state"] = np.zeros_like(np.asarray(extracted["state"]), dtype=np.float32)
    return template


def _copy_with_batch_dim(x: Any, *, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    return np.array(arr[None, ...], copy=True)


# ---------------------------------------------------------------------------
# DSRLLearner
# ---------------------------------------------------------------------------

class DSRLLearner(Agent):

    def __init__(self,
        config: OnlineTrainConfig,
        dummy_obs: ObsType,
        dummy_act: ActionType,
        state_action_critic_def: StateActionCriticDef,
        policy_def: PolicyDef,
        task_description: str,):
        self._config = config
        if bool(getattr(self._config.collect, "store_prefix_rep", False)):
            raise ValueError(
                "DSRL does not support collect.store_prefix_rep=True in this integration path."
            )
        self._rng = jax.random.key(config.seed)
        devices = mesh_utils.create_device_mesh((jax.device_count(),))
        self._mesh = jax.sharding.Mesh(devices, axis_names=("batch",))
        self._sac_image_size = int(getattr(cofig.rl, "sac_image_size", 64))
        self._crop_padding = int(getattr(config.rl, "random_crop_padding", 4))
        raw_dummy_obs = jax.tree.map(lambda x: np.asarray(x), dummy_obs)
        replay_dummy_obs = _build_replay_observation_template(
            raw_dummy_obs,
            obs_prefix="pi0", # str(self._config.collect.obs_prefix_key)
            sac_image_size=self._sac_image_size,
        )
        dummy_obs = normalize_observation_for_model(raw_dummy_obs)
        # Resize image keys to match SAC image size so that network init
        # (encoder → bottleneck) uses the same spatial dims as training data.
        if self._sac_image_size > 0 and isinstance(dummy_obs, dict):
            for img_key in ("image", "wrist_image"):
                if img_key in dummy_obs:
                    dummy_obs[img_key] = jnp.asarray(
                        _resize_image_np(
                            np.asarray(dummy_obs[img_key]), self._sac_image_size
                        )
                    )
        self._dummy_obs = dummy_obs
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
        # Keep attribute parity with existing DSRL collection code paths.
        self.replay = self._online_data_buffer
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0

        # Gaussian noise flag: use random noise instead of untrained policy
        # for the very first data collection round (matching reference i==0).
        rl_cfg = getattr(self._config, "rl", None)
        self._use_random_noise = bool(getattr(rl_cfg, "warmup_gaussian_noise", True))

        q_init_rng, policy_init_rng, alpha_init_rng, self._rng = jax.random.split(self._rng, 4)
        self._state_action_critic_state, self._state_action_critic_state_sharding = init_state_action_critic_train_state(
                self._config,
                q_init_rng,
                self._mesh,
                critic_def=state_action_critic_def,
                dummy_obs=dummy_obs,
                dummy_act=dummy_act,
                use_sharding=False
            )

        self._policy_state, self._policy_state_sharding = init_policy_state(
            self._config,
            policy_init_rng,
            self._mesh,
            policy_def=policy_def,
            dummy_obs=dummy_obs,
            dummy_act=dummy_act,
            use_sharding=False
        )
        self._alpha_state, self._alpha_state_sharding = init_alpha_state(
            self._config,
            alpha_init_rng,
            self._mesh,
            use_sharding=False,
        )
        self._target_entropy = resolve_target_entropy(self._config, self._action_dim)
        self._autotune_alpha = alpha_autotune_enabled(self._config)

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
        warmup_obs = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), dummy_obs)
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
            "DSRLLearner init: action_dim=%d, sac_image_size=%d, crop_padding=%d, "
            "target_entropy=%.3f, warmup_noise=%s",
            self._action_dim, self._sac_image_size, self._crop_padding,
            self._target_entropy, self._use_random_noise,
        )

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
        processed_obs = normalize_observation_for_model(observations)
        # Resize images to SAC resolution (e.g. 64×64) to match network init.
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
        if not batch_actions:
            return actions

        if actions.ndim == 1:
            actions = actions[None, ...]
        actions = normalize_action_batch_shape(actions, self._expected_action_shape)
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
            noise = np.asarray(jax.random.normal(rng, (batch_dim, self._action_dim)),dtype=np.float32,)
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

    # def sample_online_transitions(self) -> Dict[str, Any]:
    #     """Sample transitions from the shared online replay buffer interface."""
    #     if self._online_data_buffer.size == 0:
    #         raise ValueError("Cannot sample transitions from an empty online replay buffer.")
    #     batch = self._online_data_buffer.sample()

    #     return {
    #         "observation": batch["observation"],
    #         "actions": np.asarray(batch["actions"], dtype=np.float32),
    #         "next_observation": batch["next_observation"],
    #         "reward": np.asarray(batch["reward"], dtype=np.float32),
    #         "mc_return": np.asarray(batch["mc_return"], dtype=np.float32),
    #         "discount": np.asarray(batch["discount"], dtype=np.float32),
    #     }

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
    # Image augmentation applied to observation dicts at training time
    # ------------------------------------------------------------------
    def _random_crop_single(rng: jax.Array, img: jax.Array, padding: int = 4) -> jax.Array:
        """Random crop with edge-replicated padding for a single (H, W, C) image."""
        crop_from = jax.random.randint(rng, (2,), 0, 2 * padding + 1)
        crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
        padded = jnp.pad(
            img,
            ((padding, padding), (padding, padding), (0, 0)),
            mode="edge",
        )
        return jax.lax.dynamic_slice(padded, crop_from, img.shape)

    def _batched_random_crop(rng: jax.Array, imgs: jax.Array, padding: int = 4) -> jax.Array:
        """Random crop with edge padding for a (B, H, W, C) batch."""
        keys = jax.random.split(rng, imgs.shape[0])
        return jax.vmap(lambda k, i: _random_crop_single(k, i, padding))(keys, imgs)

    def _augment_images(self, obs_dict: Dict[str, jax.Array], rng: jax.Array) -> tuple[Dict[str, jax.Array], jax.Array]:
        """Apply random crop augmentation to image keys (matching reference)."""
        if self._crop_padding <= 0:
            return obs_dict, rng
        augmented = dict(obs_dict)
        for img_key in ("image", "wrist_image"):
            if img_key not in augmented:
                continue
            img = augmented[img_key]
            # Only augment spatial images (B, H, W, C) where H, W > 1.
            if img.ndim == 4 and img.shape[1] > 1 and img.shape[2] > 1:
                rng, crop_rng = jax.random.split(rng)
                augmented[img_key] = _batched_random_crop(
                    crop_rng, img, padding=self._crop_padding
                )
        return augmented, rng

    # ------------------------------------------------------------------
    # Training update
    # ------------------------------------------------------------------

    def update(self):
        self.training_steps += 1

        # After first real update, disable Gaussian noise exploration.
        batch_size = int(getattr(self._config, "batch_size", 128))
        if self._online_data_buffer.size < batch_size:
            return {}

        if self._use_random_noise:
            self._use_random_noise = False
            logger.info("Disabling warmup Gaussian noise (buffer size=%d >= batch_size=%d).", self._online_data_buffer.size, batch_size,)

        info = {}
        latest_actor_observation = None

        if self.training_steps % int(getattr(self._config.rl, "critic_update_frequency", 1)) == 0:
            if batch_size != int(self._online_data_buffer.batch_size):
                raise ValueError(
                    "Configured train batch_size does not match replay batch_size: "
                    f"{batch_size} vs {self._online_data_buffer.batch_size}."
                )
            batch = self._online_data_buffer.sample()
            batch_observation = normalize_observation_for_model(batch["observation"])
            batch_next_observation = normalize_observation_for_model(batch["next_observation"])

            # Convert to jax arrays.
            observation = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), batch_observation)
            next_observation = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), batch_next_observation)

            # Random crop augmentation
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
                batch_observation = normalize_observation_for_model(batch["observation"])
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
    # Episode saving with sparse -1/0 reward (matching reference)
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

        obs_prefix = "pi0"#str(self._config.collect.obs_prefix_key)
        query_freq = int(getattr(self._config.collect, "replan_steps", 1))
        bootstrap_discount = float(self._config.rl.discount) ** query_freq
        episode_len = len(episode)

        # Reward-threshold success detection
        # is_success = (reward == env_max_reward)
        # Override the terminated-based flag with a reward check when possible.
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
            # --- Extract observations (resized, no prefix embedding) ---
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
