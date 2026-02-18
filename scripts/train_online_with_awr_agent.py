# ruff: noqa: E402
# suppress Numba FNV hashing warnings
import warnings

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

mp.set_start_method("spawn", force=True)

import platform
from typing import Any, Sequence

import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.training.utils as training_utils
from src.rl.advantage_weighted_regression import AdvantageWeightedFilteredSFTLearner
from src.rl.advantage_weighted_regression.update_critic import (
    StateActionCriticDef,
    StateValueDef,
)
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env
from src.rl.networks.decoders.values.state_action_value import StateActionEnsembleDecoder
from src.rl.networks.decoders.values.state_value import StateValueEnsembleDecoder
from src.rl.networks.encoders.resnet_encoderv1 import ResNet18
from src.rl.networks.rl_networks import StateActionCritic, StateValue
import src.training.config as _config
from src.training.collect import collect_data
from src.training.utils import init_logging, init_wandb, log_images


def _get_rl_attr(config: _config.OnlineTrainConfig, name: str, default: Any) -> Any:
    rl_config = getattr(config, "rl", None)
    return getattr(rl_config, name, default)


class ResNetObservationEncoder(nnx.Module):
    """Encodes OpenPI observations with a ResNet image backbone + low-dim state."""

    def __init__(
        self,
        observation: _model.Observation,
        *,
        image_keys: Sequence[str],
        num_filters: int,
        norm: str,
        use_spatial_softmax: bool,
        rngs: nnx.Rngs,
    ):
        self._image_keys = tuple(image_keys)
        self._image_shapes = {
            name: tuple(jnp.asarray(image).shape[1:])
            for name, image in observation.images.items()
        }
        init_inputs = self._to_resnet_inputs(observation)
        self._resnet = ResNet18(
            input_example=init_inputs,
            image_keys=list(self._image_keys),
            num_filters=num_filters,
            norm=norm,
            use_spatial_softmax=use_spatial_softmax,
            rngs=rngs,
        )

    def _to_resnet_inputs(
        self, observation: _model.Observation | jax.Array
    ) -> dict[str, dict[str, jax.Array] | jax.Array]:
        if isinstance(observation, _model.Observation):
            state = jnp.asarray(observation.state, dtype=jnp.float32)
            # Observation.from_dict stores images in [-1, 1] float; ResNetEncoder expects [0, 255].
            image_dict = {
                name: jnp.clip(
                    (jnp.asarray(image, dtype=jnp.float32) + 1.0) * 127.5,
                    0.0,
                    255.0,
                )
                for name, image in observation.images.items()
            }
        else:
            state = jnp.asarray(observation, dtype=jnp.float32)
            if state.ndim == 1:
                state = state[jnp.newaxis, :]
            batch_size = state.shape[0]
            image_dict = {
                name: jnp.zeros((batch_size, *shape), dtype=jnp.float32)
                for name, shape in self._image_shapes.items()
            }
        return {
            "image": image_dict,
            "state": state,
        }

    def __call__(
        self, observation: _model.Observation | jax.Array, training: bool = False
    ) -> jax.Array:
        obs_dict = self._to_resnet_inputs(observation)
        image_features = self._resnet(obs_dict, train=training)
        state = obs_dict["state"]
        if state.ndim > 2:
            state = state.reshape((state.shape[0], -1))
        return jnp.concatenate([image_features, state], axis=-1)


def _build_resnet_critic_defs(
    config: _config.OnlineTrainConfig,
    dummy_obs: _model.Observation,
) -> tuple[StateActionCriticDef, StateValueDef]:
    image_keys = [f"image|{name}" for name in sorted(dummy_obs.images.keys())]
    critic_hidden_dims = tuple(_get_rl_attr(config, "critic_hidden_dims", (1024, 512)))
    critic_num_qs = int(_get_rl_attr(config, "critic_num_qs", 2))
    critic_num_vs = int(_get_rl_attr(config, "critic_num_vs", 2))
    resnet_num_filters = int(_get_rl_attr(config, "critic_resnet_num_filters", 32))
    resnet_norm = str(_get_rl_attr(config, "critic_resnet_norm", "group"))
    use_spatial_softmax = bool(
        _get_rl_attr(config, "critic_resnet_use_spatial_softmax", True)
    )

    def encoder_def(observation: _model.Observation, rngs: nnx.Rngs):
        return ResNetObservationEncoder(
            observation=observation,
            image_keys=image_keys,
            num_filters=resnet_num_filters,
            norm=resnet_norm,
            use_spatial_softmax=use_spatial_softmax,
            rngs=rngs,
        )

    def state_action_decoder_def(
        embedding: jax.Array, action: jax.Array, rngs: nnx.Rngs
    ) -> StateActionEnsembleDecoder:
        return StateActionEnsembleDecoder(
            observation=embedding,
            action=action,
            hidden_dims=critic_hidden_dims,
            num_qs=critic_num_qs,
            rngs=rngs,
        )

    def state_value_decoder_def(
        embedding: jax.Array, rngs: nnx.Rngs
    ) -> StateValueEnsembleDecoder:
        return StateValueEnsembleDecoder(
            observation=embedding,
            hidden_dims=critic_hidden_dims,
            num_vs=critic_num_vs,
            rngs=rngs,
        )

    def state_action_critic_def(
        observation: _model.Observation, action: jax.Array, rngs: nnx.Rngs
    ) -> StateActionCritic:
        return StateActionCritic(
            observation=observation,
            action=action,
            encoder_def=encoder_def,
            decoder_def=state_action_decoder_def,
            rngs=rngs,
        )

    def state_value_def(observation: _model.Observation, rngs: nnx.Rngs) -> StateValue:
        return StateValue(
            observation=observation,
            encoder_def=encoder_def,
            decoder_def=state_value_decoder_def,
            rngs=rngs,
        )

    return state_action_critic_def, state_value_def


def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    from src.envs.libero import make_env_libero

    env_fn, task_description = make_env_libero(config)
    env = filtered_sft_wrap_env(
        env_fn=env_fn,
        config=config,
        task_description=task_description,
        env_class="libero",
    )

    dummy_obs = config.model.fake_obs(batch_size=1)
    dummy_act = config.model.fake_act(batch_size=1)
    state_action_critic_def, state_value_def = _build_resnet_critic_defs(
        config, dummy_obs
    )
    agent = AdvantageWeightedFilteredSFTLearner(
        config=config,
        dummy_obs=dummy_obs,
        dummy_act=dummy_act,
        state_action_critic_def=state_action_critic_def,
        state_value_def=state_value_def,
    )
    init_wandb(config, resuming=agent._resuming, enabled=config.wandb_enabled)

    batch = next(iter(agent._data_loader))
    logging.info(
        f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}"
    )
    log_images(batch)

    start_step = int(jax.device_get(agent._train_state.step))
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
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if step % config.collect.collect_interval == 0:
            agent.save_checkpoint(step=step)
            collect_info, n_collected_episodes = collect_data(
                agent=agent,
                env=env,
                task_description=task_description,
                config=config,
                step=step,
            )
            wandb.log(collect_info, step=step)
            if n_collected_episodes > 0:
                logging.info(
                    f"Collected {n_collected_episodes} successful episodes at step {step}."
                )

        if (
            step % config.save_interval == 0 and step > start_step
        ) or step == config.num_train_steps - 1:
            agent.save_checkpoint(step=step)

    logging.info("Waiting for checkpoint manager to finish")
    agent._checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
