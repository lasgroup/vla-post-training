# ruff: noqa: E402
# suppress Numba FNV hashing warnings
import warnings

from src.rl.networks.mlp import MLP
from src.rl.networks.encoders.encoders import MLPEncoder

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

    def encoder_def(observation: ObsType, rngs: nnx.Rngs):
        network_def = lambda o, rg: MLP(
            input=o,
            hidden_dims=critic_encoder_hidden_dims,
            activate_final=True,
            rngs=rg,
        )
        state_vector_keys = ["state"]
        if isinstance(observation, dict) and PREFIX_EMBEDDING_NAME in observation:
            state_vector_keys = [PREFIX_EMBEDDING_NAME, "state"]
        return MLPEncoder(
            dummy_obs=observation,
            encoder_def=network_def,
            state_vector_keys=state_vector_keys,
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
    ) -> LearnedStdTanhNormalPolicyDecoder:
        return LearnedStdTanhNormalPolicyDecoder(
            observation=embedding,
            action=action,
            hidden_dims=policy_decoder_hidden_dims,
            low=action_low,
            high=action_high,
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


def _resolve_env_backend(config: _config.OnlineTrainConfig) -> str:
    backend = str(getattr(config.collect, "env_backend", "dmc")).strip().lower()
    if backend not in ("dmc", "libero"):
        raise ValueError(
            f"Unsupported collect.env_backend={backend!r}. Expected 'dmc' or 'libero'."
        )
    return backend


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

        def _wrap_obs(self, obs: dict[str, Any], action_chunk: np.ndarray) -> dict[str, Any]:
            return {
                "observation": obs,
                "action": np.asarray(action_chunk, dtype=np.float32),
            }

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            obs, info = self.env.reset(seed=seed, options=options)
            # QueryFrequencyWrapper expands action_space to (H, action_dim).
            action_template = np.asarray(self.env.action_space.sample(), dtype=np.float32)
            action_template[...] = 0.0
            self._last_action_chunk = action_template
            return self._wrap_obs(obs, action_template), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            action_chunk = np.asarray(action, dtype=np.float32)
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
    backend = _resolve_env_backend(config)
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

def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    if bool(getattr(config, "return_prefix_rep", False)):
        logging.info(
            "return_prefix_rep is enabled, but AWR critics recompute prefix embeddings "
            "from observations every update."
        )
    backend = _resolve_env_backend(config)
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
        lambda x: jnp.asarray(x, dtype=jnp.float32)[0:1],
        model_obs_batch,
    )

    obs_action = None
    if isinstance(obs_batch, dict) and obs_batch.get("action") is not None:
        action_from_obs = np.asarray(obs_batch["action"], dtype=np.float32)
        if action_from_obs.ndim >= 2:
            obs_action = action_from_obs[0:1]

    if backend == "libero":
        # DSRLVectorEnv-style wrappers expose previous action chunks in reset obs.
        # Prefer that shape when available, otherwise fall back to config/model specs.
        if obs_action is not None:
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
    
    state_action_critic_def, policy_def = _build_actor_critic_defs(config, action_low=action_low, action_high=action_high)

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
