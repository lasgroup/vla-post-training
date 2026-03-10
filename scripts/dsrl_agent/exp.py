# ruff: noqa: E402
# suppress Numba FNV hashing warnings
import warnings
from collections import deque

from src.rl.networks.mlp import MLP
from src.rl.networks.encoders.encoders import BaseEncoder, ImageEncoder, MLPEncoder
from src.rl.networks.encoders.cnn_encoder import CNNEncoder
from src.rl.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from src.rl.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from src.rl.networks.encoders.resnet_encoderv2 import ResNetv2_18, ResNetv2_34, ResNetv2_Small

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


import platform
from typing import Any

import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

from src.rl.dsrl.dsrl_learner import DSRLLearner
from src.rl.dsrl.chunk_obs import unwrap_dsrl_vector_observation
from src.envs.dsrl_vector_env import DSRLVectorEnv
from src.rl.dsrl.update_critic import StateActionCriticDef
from src.rl.dsrl.update_actor import PolicyDef
from src.rl.networks.rl_networks import Policy
from src.rl.networks.decoders.values.state_action_value import StateActionEnsembleDecoder
from src.rl.networks.decoders.policies.learned_std_normal_policy import LearnedStdNormalPolicyDecoder, LearnedStdTanhNormalPolicyDecoder
from src.rl.networks.rl_networks import ObsType, ActionType, StateActionCritic
from src.envs import make_env
from src.envs.wrappers import (
    Pi0ObservationWrapper,
    QueryFrequencyWrapper,
    TimeToSuccessAsRewardWrapper,
)
import src.training.config as _config
from src.training.collect import collect_data
from src.training.utils import init_logging, init_wandb


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
    critic_encoder_hidden_dims = tuple(_get_rl_attr(config, "critic_encoder_hidden_dims", ()))
    critic_decoder_hidden_dims = tuple(_get_rl_attr(config, "critic_decoder_hidden_dims", (256, 256)))
    policy_decoder_hidden_dims = tuple(_get_rl_attr(config, "policy_decoder_hidden_dims", (256, 256)))
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
        # if PREFIX_EMBEDDING_NAME in observation:
        #     state_vector_keys = [PREFIX_EMBEDDING_NAME, "state"]

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


def _normalize_task_descriptions(
    task_description: list[str] | str,
    env_num: int,
) -> list[str]:
    if isinstance(task_description, str):
        return [task_description for _ in range(env_num)]
    if not task_description:
        return ["" for _ in range(env_num)]
    if len(task_description) == env_num:
        return [str(x) for x in task_description]
    if len(task_description) == 1:
        return [str(task_description[0]) for _ in range(env_num)]
    return [str(task_description[i % len(task_description)]) for i in range(env_num)]


def _get_pre_step_action_filter(domain: str):
    if domain == "libero":
        # Keep LIBERO near-zero clipping used by current DSRL rollouts.
        return lambda x: np.where(np.abs(x) < 0.0011, 0.0, x)
    return lambda x: x


def _wrap_dsrl_env(env_fn, config, task_description: list[str]):
    env_num = int(config.collect.env_num)
    replan_steps = int(config.collect.replan_steps)
    env_class = str(config.collect.domain)
    pre_step_action_filter = _get_pre_step_action_filter(env_class)

    env_factories = []
    for i in range(env_num):
        task_desc_i = task_description[i]

        def _make_env(rank=i, task_description_single=task_desc_i):
            base_env = env_fn(rank)
            if config.collect.use_time_to_success_as_reward:
                base_env = TimeToSuccessAsRewardWrapper(base_env)
            base_env = Pi0ObservationWrapper(
                env=base_env,
                env_class=env_class,
                task_description=task_description_single,
                molmo_config=getattr(config, "molmo", None),
            )
            base_env = QueryFrequencyWrapper(
                env=base_env,
                query_frequency=replan_steps,
                pre_step_filter=pre_step_action_filter,
            )
            return base_env

        env_factories.append(_make_env)

    env = DSRLVectorEnv(
        env_factories,
        config=config,
        task_description=task_description,
    )
    env.seed(int(config.seed))
    return env


def _build_training_env(config: _config.OnlineTrainConfig):
    env_fn, task_description = make_env(config)
    task_description = _normalize_task_descriptions(
        task_description,
        int(config.collect.env_num),
    )
    env = _wrap_dsrl_env(
        env_fn=env_fn,
        config=config,
        task_description=task_description,
    )
    return env, task_description

class DSRLCollectNoShiftAdapter:
    def __init__(self, inner):
        self._inner = inner
        self._pending_obs = deque()

    def sample_actions(self, observations, **kwargs):
        snap = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), observations)
        self._pending_obs.append(snap)
        return self._inner.sample_actions(observations, **kwargs)

    def add_data(self, step_data):
        if self._pending_obs:
            patched = dict(step_data)
            patched["observation"] = self._pending_obs.popleft()
            return self._inner.add_data(patched)
        return self._inner.add_data(step_data)

    def start_data_collection(self, *args, **kwargs):
        self._pending_obs.clear()
        return self._inner.start_data_collection(*args, **kwargs)

    def end_data_collection(self, *args, **kwargs):
        self._pending_obs.clear()
        return self._inner.end_data_collection(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)



def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    if bool(getattr(config.collect, "store_prefix_rep", False)):
        raise ValueError(
            "DSRL does not support collect.store_prefix_rep=True in this integration path."
        )
    backend = str(config.collect.domain)
    env, task_description = _build_training_env(config)

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
    action_dim = int(
        getattr(env, "policy_action_dim", int(getattr(config.model, "action_dim", 32)))
    )
    policy_action_horizon = int(
        getattr(env, "policy_action_horizon", int(getattr(config.model, "action_horizon", 10)))
    )
    action_horizon = 1
    dummy_act = jnp.zeros((1, action_horizon, action_dim), dtype=jnp.float32)
    logging.info(
        "Using compact DSRL latent-noise shape for init: "
        "(horizon=%d, action_dim=%d), policy_horizon=%d.",
        action_horizon,
        action_dim,
        policy_action_horizon,
    )
    # Use scalar bounds to avoid TFP broadcast issues with chunked action shapes.
    action_low = jnp.asarray(-1.0, dtype=jnp.float32)
    action_high = jnp.asarray(1.0, dtype=jnp.float32)
    
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
    agent = DSRLCollectNoShiftAdapter(agent)
    
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
