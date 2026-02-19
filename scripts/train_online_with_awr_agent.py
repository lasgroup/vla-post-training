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
import os

mp.set_start_method("spawn", force=True)

# Spawned env workers re-import this module. Keep them off GPU/JAX device init.
if mp.current_process().name != "MainProcess":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Avoid aggressive JAX GPU preallocation in the trainer process.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import gc
import platform
from typing import Any

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
from src.rl.networks.decoders.values.state_action_value import (
    StateActionEnsembleDecoder,
)
from src.rl.networks.decoders.values.state_value import StateValueEnsembleDecoder
from src.rl.networks.rl_networks import ObsType, StateActionCritic, StateValue
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
import src.training.config as _config
from src.training.collect import collect_data
from src.training.utils import init_logging, init_wandb, log_images


def _get_rl_attr(config: _config.OnlineTrainConfig, name: str, default: Any) -> Any:
    rl_config = getattr(config, "rl", None)
    return getattr(rl_config, name, default)


def _infer_prefix_embedding_shape(
    config: _config.OnlineTrainConfig,
) -> tuple[int, ...] | None:
    model = None
    try:
        init_rng = jax.random.key(config.seed)
        model = config.model.create(init_rng)
        if not hasattr(model, "get_prefix_rep"):
            return None
        fake_obs = config.model.fake_obs(batch_size=1)
        prefix_rep = model.get_prefix_rep(fake_obs)
        if isinstance(prefix_rep, tuple):
            prefix_rep = prefix_rep[0]
        prefix_rep = jnp.asarray(prefix_rep, dtype=jnp.float32)
        return tuple(int(x) for x in prefix_rep.shape[1:])
    except Exception as exc:  # pragma: no cover - startup fallback path
        logging.warning("Failed to infer Pi0 prefix embedding shape: %s", exc)
        return None
    finally:
        del model
        gc.collect()


def _make_dummy_critic_observation(
    config: _config.OnlineTrainConfig,
    *,
    prefix_embedding_shape: tuple[int, ...] | None,
) -> dict[str, jax.Array]:
    fake_obs = config.model.fake_obs(batch_size=1)
    dummy_obs = {"state": jnp.asarray(fake_obs.state, dtype=jnp.float32)}
    if prefix_embedding_shape is not None:
        dummy_obs[PREFIX_EMBEDDING_NAME] = jnp.zeros(
            (1, *prefix_embedding_shape), dtype=jnp.float32
        )
    return dummy_obs


class Pi0BackboneObservationEncoder(nnx.Module):
    """Uses Pi0 prefix embeddings (if available) plus state as critic observations."""

    def __init__(
        self,
        observation: ObsType,
        *,
        prefix_embedding_shape: tuple[int, ...] | None,
        rngs: nnx.Rngs,
    ):
        del rngs
        self._prefix_embedding_shape = prefix_embedding_shape
        # Validate observation compatibility at init time.
        self._extract_state(observation)

    @staticmethod
    def _extract_state(observation: ObsType) -> jax.Array:
        if isinstance(observation, _model.Observation):
            state = observation.state
        elif isinstance(observation, dict):
            state = observation.get("state")
            if state is None:
                raise KeyError("Critic observation dict must include a 'state' field.")
        else:
            state = observation
        state = jnp.asarray(state, dtype=jnp.float32)
        if state.ndim == 1:
            state = state[jnp.newaxis, :]
        if state.ndim > 2:
            state = state.reshape((state.shape[0], -1))
        return state

    def _extract_prefix_embedding(
        self, observation: ObsType, *, batch_size: int
    ) -> jax.Array | None:
        prefix = None
        if isinstance(observation, dict):
            prefix = observation.get(PREFIX_EMBEDDING_NAME)

        if prefix is None:
            if self._prefix_embedding_shape is None:
                return None
            prefix = jnp.zeros(
                (batch_size, *self._prefix_embedding_shape), dtype=jnp.float32
            )
        else:
            prefix = jnp.asarray(prefix, dtype=jnp.float32)
            if prefix.ndim == 1:
                prefix = prefix[jnp.newaxis, :]
            if prefix.ndim == 2 and prefix.shape[0] != batch_size:
                if prefix.shape[0] == 1:
                    prefix = jnp.broadcast_to(prefix, (batch_size, prefix.shape[-1]))
                else:
                    raise ValueError(
                        "Prefix embedding batch mismatch: "
                        f"{prefix.shape[0]} vs {batch_size}."
                    )
            if prefix.ndim >= 3 and prefix.shape[0] != batch_size:
                if prefix.shape[0] == 1:
                    prefix = jnp.broadcast_to(prefix, (batch_size,) + prefix.shape[1:])
                else:
                    raise ValueError(
                        "Prefix embedding batch mismatch: "
                        f"{prefix.shape[0]} vs {batch_size}."
                    )

        if prefix.ndim == 2:
            return prefix

        # Prefix embeddings are expected as [B, S, E]. Pool sequence tokens to [B, E].
        prefix = prefix.reshape((prefix.shape[0], -1, prefix.shape[-1]))
        return jnp.mean(prefix, axis=1)

    def __call__(self, observation: ObsType, training: bool = False) -> jax.Array:
        del training
        state = self._extract_state(observation)
        prefix_embedding = self._extract_prefix_embedding(
            observation, batch_size=state.shape[0]
        )
        if prefix_embedding is None:
            return state
        return jnp.concatenate([state, prefix_embedding], axis=-1)


def _build_pi0_backbone_critic_defs(
    config: _config.OnlineTrainConfig,
    *,
    prefix_embedding_shape: tuple[int, ...] | None,
) -> tuple[StateActionCriticDef, StateValueDef]:
    critic_hidden_dims = tuple(_get_rl_attr(config, "critic_hidden_dims", (1024, 512)))
    critic_num_qs = int(_get_rl_attr(config, "critic_num_qs", 2))
    critic_num_vs = int(_get_rl_attr(config, "critic_num_vs", 2))

    def encoder_def(observation: ObsType, rngs: nnx.Rngs):
        return Pi0BackboneObservationEncoder(
            observation=observation,
            prefix_embedding_shape=prefix_embedding_shape,
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
        observation: ObsType, action: jax.Array, rngs: nnx.Rngs
    ) -> StateActionCritic:
        return StateActionCritic(
            observation=observation,
            action=action,
            encoder_def=encoder_def,
            decoder_def=state_action_decoder_def,
            rngs=rngs,
        )

    def state_value_def(observation: ObsType, rngs: nnx.Rngs) -> StateValue:
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
    if bool(getattr(config, "return_prefix_rep", False)):
        logging.info(
            "return_prefix_rep is enabled, but AWR critics recompute prefix embeddings "
            "from observations every update."
        )

    from src.envs.libero import make_env_libero

    env_fn, task_description = make_env_libero(config)
    env = filtered_sft_wrap_env(
        env_fn=env_fn,
        config=config,
        task_description=task_description,
        env_class="libero",
    )

    prefix_embedding_shape = _infer_prefix_embedding_shape(config)
    if prefix_embedding_shape is None:
        logging.warning(
            "Could not infer Pi0 prefix embedding shape; critic encoder will use state only."
        )
    else:
        logging.info(
            "Using Pi0 prefix embeddings for critic observations with shape %s.",
            prefix_embedding_shape,
        )
    dummy_obs = _make_dummy_critic_observation(
        config, prefix_embedding_shape=prefix_embedding_shape
    )
    dummy_act = config.model.fake_act(batch_size=1)
    state_action_critic_def, state_value_def = _build_pi0_backbone_critic_defs(
        config, prefix_embedding_shape=prefix_embedding_shape
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
