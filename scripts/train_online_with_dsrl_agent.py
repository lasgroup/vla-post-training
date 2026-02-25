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
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
from src.rl.advantage_weighted_regression.update_critic import (
    StateActionCriticDef,
    StateValueDef,
)
from src.rl.networks.rl_networks import Policy
from src.rl.networks.decoders.values.state_action_value import StateActionEnsembleDecoder
from src.rl.networks.decoders.values.state_value import StateValueEnsembleDecoder
from src.rl.networks.decoders.policies.normal_policy import NormalPolicyDecoder
from src.rl.networks.rl_networks import ObsType, StateActionCritic

from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env

from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
import src.training.config as _config
from src.training.collect import collect_data
from src.training.utils import init_logging, init_wandb, log_images
from src.rl.dsrl_agent.dmc_env import DMCEnv
import functools

def _build_actor_critic_defs(
    config: _config.OnlineTrainConfig,
) -> tuple[StateActionCriticDef, StateValueDef]:
    critic_encoder_hidden_dims = (1024, 512) #tuple(_get_rl_attr(config, "critic_encoder_hidden_dims", (1024, 512)))
    critic_decoder_hidden_dims = () #tuple(_get_rl_attr(config, "critic_decoder_hidden_dims", ()))
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
    ) -> NormalPolicyDecoder:
        return NormalPolicyDecoder(
            observation=embedding,
            action=action,
            hidden_dims=critic_decoder_hidden_dims,
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

    def policy_def(observation: ObsType, rngs: nnx.Rngs) -> Policy:
        return Policy(
            observation=observation,
            encoder_def=encoder_def,
            decoder_def=policy_decoder_def,
            rngs=rngs,
        )

    return state_action_critic_def, policy_def

def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    if bool(getattr(config, "return_prefix_rep", False)):
        logging.info(
            "return_prefix_rep is enabled, but AWR critics recompute prefix embeddings "
            "from observations every update."
        )
    # Make Environment
    #from src.envs.libero import make_env_libero
    # env_fn, task_description = make_env_libero(config)
    # env = filtered_sft_wrap_env(
    #     env_fn=env_fn,
    #     config=config,
    #     task_description=task_description,
    #     env_class="libero",
    # )={"random": 42},
    def make_env(seed):
        return DMCEnv(domain_name="cartpole", task_name="swingup", task_kwargs={"random": seed}) #, seed=seed

    env_fns = [functools.partial(make_env, seed=42)]
    env = DummyVectorEnv(env_fns)
    task_description = ""
    
    dummy_obs = env.observation_space.sample()  #_make_dummy_critic_observation(config, prefix_embedding_shape=None)
    dummy_act = env.action_space.sample()       #config.model.fake_act(batch_size=1)
    state_action_critic_def, policy_def = _build_actor_critic_defs(config)

    # Create dummy observations and model definitions
    # prefix_embedding_shape = _infer_prefix_embedding_shape(config)
    # if prefix_embedding_shape is None:
    #     logging.warning(
    #         "Could not infer Pi0 prefix embedding shape; critic encoder will use state only."
    #     )
    # else:
    #     logging.info(
    #         "Using Pi0 prefix embeddings for critic observations with shape %s.",
    #         prefix_embedding_shape,
    #     )
    # dummy_obs = _make_dummy_critic_observation(
    #     config, prefix_embedding_shape=prefix_embedding_shape
    # )
    # dummy_act = config.model.fake_act(batch_size=1)
    # state_action_critic_def, state_value_def = _build_pi0_backbone_critic_defs(
    #     config, prefix_embedding_shape=prefix_embedding_shape
    # )
    # agent = AdvantageWeightedFilteredSFTLearner(
    #     config=config,
    #     dummy_obs=dummy_obs,
    #     dummy_act=dummy_act,
    #     state_action_critic_def=state_action_critic_def,
    #     state_value_def=state_value_def,
    #     task_description=task_description,
    # )
    agent = DSRLLearner(config=config, 
                        dummy_obs=dummy_obs,
                        dummy_act=dummy_act,
                        state_action_critic_def=state_action_critic_def,
                        policy_def=policy_def,
                        task_description=task_description)
    #init_wandb(config, resuming=False, enabled=config.wandb_enabled) #agent._resuming

    # batch = next(iter(agent._data_loader))
    # logging.info(
    #     f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}"
    # )
    # log_images(batch)

    #start_step = int(jax.device_get(agent._train_state.step))
    start_step = 0
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
        #infos.append(info)

        # if step % config.log_interval == 0:
        #     stacked_infos = common_utils.stack_forest(infos)
        #     reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
        #     info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
        #     pbar.write(f"Step {step}: {info_str}")
        #     wandb.log(reduced_info, step=step)
        #     infos = []

        if step % config.collect.collect_interval == 0:
            #agent.save_checkpoint(step=step)
            collect_info, n_collected_episodes = collect_data(
                agent=agent,
                env=env,
                task_description=task_description, #"",
                config=config,
                step=step,
            )

            # --- DEBUG: print one sampled batch from replay buffer ---
            # if hasattr(agent, "replay") and agent.replay.size > 0:
            #     batch = agent.replay.sample(batch_size=4)

            #     def _shape(x):
            #         x = jnp.asarray(x)
            #         return x.shape, x.dtype

            #     # observation might be a dict or array
            #     obs = batch.observation
            #     next_obs = batch.next_observation

            #     print("\n=== Replay buffer debug ===")
            #     print("replay.size:", agent.replay.size)
            #     print("action:", _shape(batch.action))
            #     print("reward:", _shape(batch.reward), "sample:", batch.reward[:4])
            #     print("done:", _shape(batch.done), "sample:", batch.done[:4])

            #     if isinstance(obs, dict):
            #         print("observation keys:", list(obs.keys()))
            #         # print a few leaf shapes
            #         for k in list(obs.keys())[:5]:
            #             print(f"obs[{k}]:", _shape(obs[k]))
            #     else:
            #         print("observation:", _shape(obs))

            #     if isinstance(next_obs, dict):
            #         print("next_observation keys:", list(next_obs.keys()))
            #         for k in list(next_obs.keys())[:5]:
            #             print(f"next_obs[{k}]:", _shape(next_obs[k]))
            #     else:
            #         print("next_observation:", _shape(next_obs))

            #     print("action[0]:", batch.action[0])
            #     print("================================\n")
            # --- end debug ---
            
                        # wandb.log(collect_info, step=step)
            # if n_collected_episodes > 0:
            #     logging.info(
            #         f"Collected {n_collected_episodes} successful episodes at step {step}."
            #     )

        #if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
        #    agent.save_checkpoint(step=step)

    #logging.info("Waiting for checkpoint manager to finish")
    #agent._checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())

# uv run /users/$USER/vla-post-training/scripts/train_online_with_dsrl_agent.py pi05_libero_online --exp-name=my_experiment --overwrite --checkpoint_base_dir /capstor/scratch/cscs/${USER}/checkpoints --weight-loader.params-path gs://openpi-assets/checkpoints/pi05_libero/params --num_train_steps 2000

