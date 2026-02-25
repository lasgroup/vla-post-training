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
from flax.training import common_utils
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm
import wandb

import openpi.training.utils as training_utils
from src.rl.dsrl_agent.dsrl_agent import DSRLLearner
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
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env

from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
import src.training.config as _config
from src.training.collect import collect_data
from src.training.utils import init_logging, init_wandb, log_images
import functools

def _build_actor_critic_defs(
    config: _config.OnlineTrainConfig,
    action_low: jax.Array,
    action_high: jax.Array,
) -> tuple[StateActionCriticDef, PolicyDef]:
    critic_encoder_hidden_dims = (1024, 512) #tuple(_get_rl_attr(config, "critic_encoder_hidden_dims", (1024, 512)))
    critic_decoder_hidden_dims = () #tuple(_get_rl_attr(config, "critic_decoder_hidden_dims", ()))
    policy_decoder_hidden_dims = (256,)
    critic_num_qs = 2 #int(_get_rl_attr(config, "critic_num_qs", 2))

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

def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    if bool(getattr(config, "return_prefix_rep", False)):
        logging.info(
            "return_prefix_rep is enabled, but AWR critics recompute prefix embeddings "
            "from observations every update."
        )
    def make_env(seed): # cartpole, swingup
        return DMCEnv(domain_name="walker", task_name="walk", task_kwargs={"random": seed}) #, seed=seed

    env_num = int(config.collect.env_num)
    env_fns = [functools.partial(make_env, seed=int(config.seed) + i) for i in range(env_num)]
    env = SubprocVectorEnv(env_fns) if env_num > 1 else DummyVectorEnv(env_fns)
    task_description = ""
    
    dummy_obs = env.observation_space[0].sample()
    action_space = env.action_space[0]
    dummy_act = action_space.sample()
    action_low = jnp.asarray(action_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(action_space.high, dtype=jnp.float32)
    dummy_obs = jax.tree.map(
        lambda x: jnp.asarray(x, dtype=jnp.float32)[None, ...],
        dummy_obs,
    )
    dummy_act = jnp.asarray(dummy_act, dtype=jnp.float32)[None, ...]
    state_action_critic_def, policy_def = _build_actor_critic_defs(
        config, action_low=action_low, action_high=action_high
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
        infos.append(info)

        if step % config.log_interval == 0:
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
