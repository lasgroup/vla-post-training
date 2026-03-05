# ruff: noqa: E402
# suppress Numba FNV hashing warnings
import warnings

from src.rl.networks.mlp import MLP
from src.rl.networks.encoders.encoders import BaseEncoder, ImageEncoder, MLPEncoder
from src.rl.networks.encoders.cnn_encoder import CNNEncoder
from src.rl.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from src.rl.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from src.rl.networks.encoders.resnet_encoderv2 import (
    ResNetv2_18,
    ResNetv2_34,
    ResNetv2_Small,
)

warnings.filterwarnings("ignore", category=UserWarning, message=".*FNV hashing.*")

# suppress lerobot version warnings
import logging


class VersionWarningFilter(logging.Filter):
    def filter(self, record):
        # avoid lerobot warning
        return "is in 2.0 format" not in record.getMessage()


logging.getLogger().addFilter(VersionWarningFilter())

# disable datasets progress bars
from datasets import disable_progress_bars

disable_progress_bars()

# allows using subprocenvs
import multiprocessing as mp
import os

mp.set_start_method("spawn", force=True)

# Spawned env workers re-import this module. Keep them off GPU/JAX device init.
if mp.current_process().name != "MainProcess":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Avoid aggressive JAX GPU preallocation in the trainer process.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


import gc
import platform
from typing import Any

import flax.nnx as nnx
import gymnasium as gym
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

import openpi.training.utils as training_utils
from src.rl.dsrl_agent.dsrl_agent import DSRLLearner
from src.rl.dsrl_agent.chunk_ops import unwrap_dsrl_vector_observation
from src.rl.dsrl_agent.dsrl_vector_env import DSRLVectorEnv
from src.rl.dsrl_agent.update_critic import (
    StateActionCriticDef,
)
from src.rl.dsrl_agent.update_actor import (
    PolicyDef
)
from src.rl.networks.rl_networks import Policy
from src.rl.networks.decoders.values.state_action_value import StateActionEnsembleDecoder
from src.rl.networks.decoders.policies.learned_std_normal_policy import (
    LearnedStdNormalPolicyDecoder,
    LearnedStdTanhNormalPolicyDecoder,
)
from src.rl.networks.rl_networks import ObsType, ActionType, StateActionCritic

from src.envs.dmc_env import DMCEnv
from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.envs.libero import make_env_libero
from src.envs.wrappers import Pi0ObservationWrapper, QueryFrequencyWrapper

from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
import src.training.config as _config
from src.training.collect import collect_data
from src.training.utils import init_logging, init_wandb, log_images
import functools


def _get_rl_attr(config: _config.OnlineTrainConfig, name: str, default: Any) -> Any:
    rl = getattr(config, "rl", None)
    if rl is None:
        return default
    return getattr(rl, name, default)


def _build_actor_critic_defs(
    config: _config.OnlineTrainConfig,
    action_low: jax.Array,
    action_high: jax.Array,
    *,
    backend: str,
    policy_distribution: str,
) -> tuple[StateActionCriticDef, PolicyDef]:
    critic_encoder_hidden_dims = tuple(
        _get_rl_attr(config, "critic_encoder_hidden_dims", ())
    )
    critic_decoder_hidden_dims = tuple(
        _get_rl_attr(config, "critic_decoder_hidden_dims", (256, 256))
    )
    policy_decoder_hidden_dims = tuple(
        _get_rl_attr(config, "policy_decoder_hidden_dims", (256, 256))
    )
    critic_num_qs = int(_get_rl_attr(config, "critic_num_qs", 2))
    encoder_type = str(_get_rl_attr(config, "encoder_type", "resnet_34_v1")).lower()
    encoder_norm = str(_get_rl_attr(config, "encoder_norm", "group")).lower()
    use_spatial_softmax = bool(_get_rl_attr(config, "use_spatial_softmax", True))
    softmax_temperature = float(_get_rl_attr(config, "softmax_temperature", 1.0))
    image_latent_dim = int(_get_rl_attr(config, "image_latent_dim", 50))
    use_image_bottleneck = bool(_get_rl_attr(config, "use_image_bottleneck", True))
    use_state_branch = bool(_get_rl_attr(config, "use_state_branch", True))

    def _build_image_backbone(
        observation: ObsType,
        image_keys: tuple[str, ...],
        rngs: nnx.Rngs,
    ):
        if encoder_type == "small":
            return CNNEncoder(
                input_example=observation,
                features=(32, 32, 32, 32),
                strides=(2, 1, 1, 1),
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "impala":
            return ImpalaEncoder(
                input_example=observation,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "impala_small":
            return SmallerImpalaEncoder(
                input_example=observation,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_small":
            return ResNetSmall(
                input_example=observation,
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_18_v1":
            return ResNet18(
                input_example=observation,
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_34_v1":
            return ResNet34(
                input_example=observation,
                norm=encoder_norm,
                use_spatial_softmax=use_spatial_softmax,
                softmax_temperature=softmax_temperature,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_small_v2":
            v2_norm = "groupnorm" if encoder_norm == "group" else "batch"
            return ResNetv2_Small(
                input_example=observation,
                norm=v2_norm,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_18_v2":
            v2_norm = "groupnorm" if encoder_norm == "group" else "batch"
            return ResNetv2_18(
                input_example=observation,
                norm=v2_norm,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        if encoder_type == "resnet_34_v2":
            v2_norm = "groupnorm" if encoder_norm == "group" else "batch"
            return ResNetv2_34(
                input_example=observation,
                norm=v2_norm,
                image_keys=list(image_keys),
                rngs=rngs,
            )
        raise ValueError(
            f"Unsupported rl.encoder_type={encoder_type!r}. "
            "Expected one of: "
            "'small', 'impala', 'impala_small', "
            "'resnet_small', 'resnet_18_v1', 'resnet_34_v1', "
            "'resnet_small_v2', 'resnet_18_v2', 'resnet_34_v2'."
        )

    def encoder_def(observation: ObsType, rngs: nnx.Rngs):
        if not isinstance(observation, dict):
            return BaseEncoder(
                dummy_obs=observation,
                mlp_encoder_def=None,
                image_encoder_def=None,
                rngs=rngs,
            )

        state_vector_keys = ["state"]
        if PREFIX_EMBEDDING_NAME in observation:
            state_vector_keys = [PREFIX_EMBEDDING_NAME, "state"]

        use_pixel_encoder = backend == "libero"
        if use_pixel_encoder:
            image_keys = tuple(
                key for key in ("image", "wrist_image", "pixels") if key in observation
            )
            use_pixel_encoder = len(image_keys) > 0
        else:
            image_keys = ()

        mlp_encoder_def = None
        if use_state_branch or not use_pixel_encoder:
            network_def = lambda o, rg: MLP(
                input=o,
                hidden_dims=critic_encoder_hidden_dims,
                activate_final=True,
                rngs=rg,
            )
            mlp_encoder_def = lambda obs, rg: MLPEncoder(
                dummy_obs=obs,
                encoder_def=network_def,
                state_vector_keys=state_vector_keys,
                rngs=rg,
            )

        image_encoder_def = None
        if use_pixel_encoder:
            image_backbone_def = lambda obs, rg: _build_image_backbone(
                observation=obs,
                image_keys=image_keys,
                rngs=rg,
            )
            image_encoder_def = lambda obs, rg: ImageEncoder(
                dummy_obs=obs,
                encoder_def=image_backbone_def,
                latent_dim=image_latent_dim,
                use_bottleneck=use_image_bottleneck,
                rngs=rg,
            )

        return BaseEncoder(
            dummy_obs=observation,
            mlp_encoder_def=mlp_encoder_def,
            image_encoder_def=image_encoder_def,
            rngs=rngs,
        )

    def state_action_decoder_def(
        embedding: jax.Array, action: jax.Array, rngs: nnx.Rngs
    ) -> StateActionEnsembleDecoder:
        return StateActionEnsembleDecoder(
            observation=embedding,
            action=action,
            hidden_dims=critic_decoder_hidden_dims,
            num_qs=critic_num_qs,
            rngs=rngs,
        )

    def policy_decoder_def(
        embedding: jax.Array, action: jax.Array, rngs: nnx.Rngs
    ):
        if policy_distribution == "normal":
            return LearnedStdNormalPolicyDecoder(
                observation=embedding,
                action=action,
                hidden_dims=policy_decoder_hidden_dims,
                rngs=rngs,
            )
        if policy_distribution == "tanh_normal":
            return LearnedStdTanhNormalPolicyDecoder(
                observation=embedding,
                action=action,
                hidden_dims=policy_decoder_hidden_dims,
                low=action_low,
                high=action_high,
                rngs=rngs,
            )
        raise ValueError(
            f"Unsupported policy distribution {policy_distribution!r}. "
            "Expected 'normal' or 'tanh_normal'."
        )

    def state_action_critic_def(
        observation: ObsType, action: jax.Array, rngs: nnx.Rngs
    ) -> StateActionCritic:
        return StateActionCritic(
            observation=observation,
            action=action,
            encoder_def=encoder_def,
            decoder_def=state_action_decoder_def,
            rngs=rngs,
        )

    def policy_def(
        observation: ObsType, action: ActionType, rngs: nnx.Rngs
    ) -> Policy:
        return Policy(
            observation=observation, 
            action=action, 
            encoder_def=encoder_def,
            decoder_def=policy_decoder_def, 
            rngs=rngs)

    return state_action_critic_def, policy_def


def _wrap_dsrl_env_for_libero(env_fn, config, task_description: str):
    env_num = int(config.collect.env_num)
    add_states = bool(config.collect.add_states)
    obs_prefix_key = str(config.collect.obs_prefix_key)
    replan_steps = int(config.collect.replan_steps)
    discount = float(config.discount)
    add_per_step_data = bool(config.collect.add_per_step_data)

    class _DSRLVectorEnvInputCompatWrapper(gym.Wrapper):
        """Adapts env outputs to the schema expected by DSRLVectorEnv.

        DSRLVectorEnv expects observations shaped as:
          {"observation": <obs_dict>, "action": <action_chunk>}
        """

        def __init__(self, env: gym.Env):
            super().__init__(env)
            self._last_action_chunk: np.ndarray | None = None
            self._action_horizon = int(
                getattr(getattr(config, "model", None), "action_horizon", replan_steps)
            )
            self._action_dim = self._infer_action_dim()

        def _infer_action_dim(self) -> int:
            # Avoid QueryFrequencyWrapper.action_space (currently broken). Read
            # single-step action dims from the unwrapped/base env instead.
            default_dim = int(getattr(config.collect, "libero_action_dim", 7))
            base_env = getattr(self.env, "unwrapped", None)
            if base_env is not None:
                base_action_space = getattr(base_env, "action_space", None)
                if (
                    base_action_space is not None
                    and hasattr(base_action_space, "shape")
                    and base_action_space.shape is not None
                    and len(base_action_space.shape) > 0
                ):
                    return int(base_action_space.shape[-1])
                action_dim_attr = getattr(base_env, "action_dim", None)
                if action_dim_attr is not None:
                    return int(action_dim_attr)
            return default_dim

        def _wrap_obs(self, obs: dict[str, Any], action_chunk: np.ndarray) -> dict[str, Any]:
            return {
                "observation": obs,
                "action": np.asarray(action_chunk, dtype=np.float32),
            }

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            obs, info = self.env.reset(seed=seed, options=options)
            action_template = np.zeros(
                (self._action_horizon, self._action_dim), dtype=np.float32
            )
            self._last_action_chunk = action_template
            return self._wrap_obs(obs, action_template), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            action_chunk = np.asarray(action, dtype=np.float32)
            if action_chunk.ndim == 1:
                action_chunk = np.repeat(
                    action_chunk[None, ...], self._action_horizon, axis=0
                )
            self._last_action_chunk = action_chunk
            return self._wrap_obs(obs, action_chunk), reward, terminated, truncated, info

    env_factories = []
    for i in range(env_num):

        def _make_env(rank=i):
            base_env = env_fn(rank)
            base_env = Pi0ObservationWrapper(
                env=base_env,
                env_class="libero",
                task_description=task_description,
                add_states=add_states,
                pi0_obs_prefix=obs_prefix_key,
            )
            base_env = QueryFrequencyWrapper(
                env=base_env,
                query_frequency=replan_steps,
                discount=discount,
                store_full_transitions=add_per_step_data,
                pre_step_filter=lambda x: np.where(np.abs(x) < 0.0011, 0.0, x),
            )
            base_env = _DSRLVectorEnvInputCompatWrapper(base_env)
            return base_env

        env_factories.append(_make_env)

    env = (
        DSRLVectorEnv(env_factories)
        if env_num > 1
        else DummyVectorEnv(env_factories)
    )
    env.seed(int(config.seed))
    return env


def _build_training_env(config: _config.OnlineTrainConfig):
    backend = getattr(config.collect, "env_backend", "dmc")
    if backend == "libero":
        env_fn, task_description = make_env_libero(config)
        env = _wrap_dsrl_env_for_libero(
            env_fn=env_fn,
            config=config,
            task_description=task_description,
        )
        return env, task_description

    domain_name = str(getattr(config.collect, "dmc_domain_name", "walker"))
    task_name = str(getattr(config.collect, "dmc_task_name", "walk"))

    def make_env(seed):
        return DMCEnv(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": seed},
        )

    env_num = int(config.collect.env_num)
    env_fns = [functools.partial(make_env, seed=int(config.seed) + i) for i in range(env_num)]
    env = SubprocVectorEnv(env_fns) if env_num > 1 else DummyVectorEnv(env_fns)
    return env, ""


def _configure_dsrl_vector_env(
    env: Any,
    *,
    task_description: str | None = None,
) -> None:
    """Adjust runtime settings for DSRLVectorEnv in online collection.

    During collection we reset individual env ids; that produces batch size 1
    observations. A batch-sharded policy spec (PartitionSpec("batch")) fails
    for these partial resets when device count > 1. Use replicated sharding.
    """
    if not isinstance(env, DSRLVectorEnv):
        return
    sharding_spec = getattr(env, "_policy_sharding_spec", None)
    if sharding_spec is None:
        return
    env._policy_sharding_spec = jax.sharding.NamedSharding(
        sharding_spec.mesh,
        jax.sharding.PartitionSpec(),
    )
    if task_description is not None:
        env._task_description = str(task_description)

    # DSRLVectorEnv currently passes task_description="dummy" to its own
    # _process_obs_for_pi0(...) in reset()/step(). Patch the instance method
    # at call-site to use the real task language without editing the wrapper.
    if not getattr(env, "_task_description_patch_applied", False):
        original_process_obs_for_pi0 = env._process_obs_for_pi0

        def _process_obs_for_pi0_with_task(observations, task_description=None):
            resolved_task_description = task_description
            if resolved_task_description in (None, "", "dummy"):
                resolved_task_description = getattr(
                    env, "_task_description", task_description
                )
            processed_obs = original_process_obs_for_pi0(
                observations,
                task_description=resolved_task_description,
            )
            prompt = processed_obs.get("prompt")
            if prompt is not None and not isinstance(prompt, str):
                prompt_arr = np.asarray(prompt)
                if prompt_arr.size == 0:
                    processed_obs["prompt"] = str(resolved_task_description or "")
                else:
                    processed_obs["prompt"] = str(prompt_arr.reshape(-1)[0])
            return processed_obs

        env._process_obs_for_pi0 = _process_obs_for_pi0_with_task
        env._task_description_patch_applied = True

    logging.info(
        "Configured DSRLVectorEnv policy sharding/prompt handling for per-env resets."
    )

def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    if bool(getattr(config, "return_prefix_rep", False)):
        logging.info(
            "return_prefix_rep is enabled, but AWR critics recompute prefix embeddings "
            "from observations every update."
        )
    backend = getattr(config.collect, "env_backend", "dmc")
    env, task_description = _build_training_env(config)
    _configure_dsrl_vector_env(env, task_description=task_description)

    # Dummy observation and action
    reset_out = env.reset()
    if isinstance(reset_out, (tuple, list)) and len(reset_out) == 2:
        obs_batch = reset_out[0]
    else:
        obs_batch = reset_out
    model_obs_batch = unwrap_dsrl_vector_observation(obs_batch)
    # Keep exactly one env sample while preserving wrapper-provided dimensions.
    dummy_obs = jax.tree.map(
        lambda x: np.asarray(x)[0:1],
        model_obs_batch,
    )

    obs_action = None
    if isinstance(obs_batch, dict) and obs_batch.get("action") is not None:
        action_from_obs = np.asarray(obs_batch["action"], dtype=np.float32)
        if action_from_obs.ndim >= 2:
            obs_action = action_from_obs[0:1]

    if backend == "libero":
        # When using DSRLVectorEnv, the learner should emit policy noise
        # (H, action_dim) expected by env.step(...), not environment actions.
        if isinstance(env, DSRLVectorEnv):
            policy = getattr(env, "_policy", None)
            if policy is not None:
                action_horizon = int(getattr(policy, "action_horizon"))
                action_dim = int(getattr(policy, "action_dim"))
                dummy_act = jnp.zeros(
                    (1, action_horizon, action_dim), dtype=jnp.float32
                )
                logging.info(
                    "Using DSRLVectorEnv policy noise shape for DSRL init: "
                    "(horizon=%d, action_dim=%d).",
                    action_horizon,
                    action_dim,
                )
            else:
                action_horizon = int(getattr(config.model, "action_horizon", 10))
                action_dim = int(getattr(config.model, "action_dim", 32))
                dummy_act = jnp.zeros(
                    (1, action_horizon, action_dim), dtype=jnp.float32
                )
                logging.warning(
                    "DSRLVectorEnv policy not found; using model fallback noise shape "
                    "(horizon=%d, action_dim=%d).",
                    action_horizon,
                    action_dim,
                )
        # Otherwise (e.g. DummyVectorEnv path), use environment action chunks.
        elif obs_action is not None:
            dummy_act = jnp.asarray(obs_action, dtype=jnp.float32)
            logging.info(
                "Using action chunk from reset observation for DSRL init: shape=%s",
                tuple(np.asarray(dummy_act).shape),
            )
        else:
            action_dim = None
            env_action_dim = env.get_env_attr("action_dim", id=0)[0]
            if env_action_dim is not None:
                action_dim = int(env_action_dim)
                logging.info("Using env action_dim=%d for DSRL init.", action_dim)
            if action_dim is None:
                warmup_action = env.get_env_attr("_warm_up_action", id=0)[0]
                if warmup_action is not None:
                    action_dim = int(np.asarray(warmup_action).reshape(-1).shape[0])
                    logging.info(
                        "Using warm-up action length=%d for DSRL init.", action_dim
                    )
            if action_dim is None:
                action_dim = int(getattr(config.collect, "libero_action_dim", 7))
            action_horizon = int(config.collect.replan_steps)
            dummy_act = jnp.zeros((1, action_horizon, action_dim), dtype=jnp.float32)
            logging.warning(
                "Reset observation has no action chunk; using env/config fallback "
                "(horizon=%d, action_dim=%d).",
                action_horizon,
                action_dim,
            )
        # Use scalar bounds to avoid TFP broadcast issues with chunked action shapes.
        action_low = jnp.asarray(-1.0, dtype=jnp.float32)
        action_high = jnp.asarray(1.0, dtype=jnp.float32)
    else:
        action_space = env.action_space[0]
        dummy_act = jnp.asarray(action_space.sample(), dtype=jnp.float32)[None, ...]
        action_low = jnp.asarray(action_space.low, dtype=jnp.float32)
        action_high = jnp.asarray(action_space.high, dtype=jnp.float32)
    
    policy_distribution = "tanh_normal"
    logging.info("DSRL actor policy distribution: %s", policy_distribution)
    if backend == "libero":
        logging.info(
            "DSRL LIBERO encoder: type=%s norm=%s spatial_softmax=%s temp=%.3f "
            "image_latent_dim=%d bottleneck=%s state_branch=%s",
            str(_get_rl_attr(config, "encoder_type", "resnet_34_v1")),
            str(_get_rl_attr(config, "encoder_norm", "group")),
            bool(_get_rl_attr(config, "use_spatial_softmax", True)),
            float(_get_rl_attr(config, "softmax_temperature", 1.0)),
            int(_get_rl_attr(config, "image_latent_dim", 50)),
            bool(_get_rl_attr(config, "use_image_bottleneck", True)),
            bool(_get_rl_attr(config, "use_state_branch", True)),
        )
    state_action_critic_def, policy_def = _build_actor_critic_defs(
        config,
        action_low=action_low,
        action_high=action_high,
        backend=backend,
        policy_distribution=policy_distribution,
    )

    agent = DSRLLearner(config=config, 
                        dummy_obs=dummy_obs,
                        dummy_act=dummy_act,
                        state_action_critic_def=state_action_critic_def,
                        policy_def=policy_def,
                        task_description=task_description)
    
    init_wandb(config, resuming=False, enabled=config.wandb_enabled) #agent._resuming

    start_step = int(jax.device_get(agent._state_action_critic_state.step))
    agent.training_steps = start_step
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        info = agent.update()
        if info:
            infos.append(info)

        if step % config.log_interval == 0 and infos:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if step % config.collect.collect_interval == 0:
            #agent.save_checkpoint(step=step)
            collect_info, n_collected_episodes = collect_data(
                agent=agent,
                env=env,
                task_description=task_description, #"",
                config=config,
                step=step,
            )
            wandb.log(collect_info, step=step)
            collect_msg = (
                f"Collect step {step}: "
                f"episodes={config.collect.num_rollouts}, "
                f"success_rate={collect_info.get('success_rate', 0.0):.3f}, "
                f"step_reward_mean={collect_info.get('reward_step_mean', 0.0):.4f}"
            )
            if "episode_return_mean" in collect_info:
                collect_msg += (
                    f", ep_return_mean={collect_info['episode_return_mean']:.3f}, "
                    f"ep_return_max={collect_info['episode_return_max']:.3f}, "
                    f"ep_len_mean={collect_info['episode_length_mean']:.1f}"
                )
            pbar.write(collect_msg)
            if n_collected_episodes > 0:
                logging.info(
                    f"Collected {n_collected_episodes} successful episodes at step {step}."
                )

        #if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
        #    agent.save_checkpoint(step=step)

    #logging.info("Waiting for checkpoint manager to finish")
    #agent._checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())

# uv run /users/$USER/vla-post-training/scripts/train_online_with_dsrl_agent.py pi05_libero_online --exp-name=my_experiment 
