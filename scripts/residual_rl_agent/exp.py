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

import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import platform

from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

from src.rl.residual_rl.residual_rl_learner import ResidualRLLearner
from src.rl.dsrl.build_dsrl_networks import _build_actor_critic_defs
from src.rl.residual_rl.residual_rl_learner import _normalize_residual_observation
from src.rl.residual_rl.residual_rl_env import residual_rl_wrap_env

from src.envs import make_env
import src.training.config as _config
from src.training.collect import collect_data, evaluate_policy
from src.training.utils import init_logging, init_wandb


def _unwrap_residual_observation(observations):
    """Extract model observations from ResidualRLVectorEnv outputs.

    The residual env outputs flat dicts with keys like ``pi0/state``,
    ``pi0/image``, and ``base_action``. We pass them through as-is
    since ``_normalize_residual_observation`` handles the conversion.
    """
    if (
        isinstance(observations, dict)
        and "observation" in observations
        and isinstance(observations["observation"], dict)
    ):
        unpacked = dict(observations["observation"])
        # Preserve base_action from the top level.
        if "base_action" in observations:
            unpacked["base_action"] = observations["base_action"]
        return unpacked
    return observations


def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    env_fn, task_description = make_env(config, config.collect.tasks)
    env, task_description = residual_rl_wrap_env(env_fn, config, task_description)

    eval_env_fn, eval_task_description = make_env(config, config.collect.eval_tasks)
    eval_env, eval_task_description = residual_rl_wrap_env(
        eval_env_fn, config, eval_task_description, env_num=config.collect.eval_env_num
    )

    # Dummy observation and action
    reset_out = env.reset()
    model_obs_batch = _unwrap_residual_observation(reset_out[0])
    # Keep exactly one env sample while preserving wrapper-provided dimensions.
    dummy_obs = jax.tree.map(lambda x: np.asarray(x)[0:1], model_obs_batch)
    action_dim = int(
        getattr(env, "policy_action_dim",
                int(getattr(config.model, "action_dim", 32)))
    )
    action_horizon = 1
    dummy_act = jnp.zeros((1, action_horizon, action_dim), dtype=jnp.float32)
    # Use scalar bounds to avoid TFP broadcast issues with chunked action shapes.
    action_low = jnp.asarray(-1.0, dtype=jnp.float32)
    action_high = jnp.asarray(1.0, dtype=jnp.float32)

    policy_distribution = getattr(config.rl, "policy_distribution", "tanh_normal")
    state_action_critic_def, policy_def = _build_actor_critic_defs(
        config,
        action_low=action_low,
        action_high=action_high,
        policy_distribution=policy_distribution,
    )

    agent = ResidualRLLearner(
        config=config,
        dummy_obs=dummy_obs,
        dummy_act=dummy_act,
        state_action_critic_def=state_action_critic_def,
        policy_def=policy_def,
        task_description=task_description,
    )

    init_wandb(config, resuming=False, enabled=config.wandb_enabled)

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

        if step % config.collect.eval_interval == 0:
            eval_info = evaluate_policy(
                agent=agent,
                env=eval_env,
                task_description=eval_task_description,
                config=config,
                step=step,
            )
            wandb.log(eval_info, step=step)
            logging.info(
                f"Eval at step {step}: {', '.join(f'{k}={v:.4f}' for k, v in eval_info.items())}"
            )

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            agent.save_checkpoint(step=step)

    logging.info("Waiting for checkpoint manager to finish")
    agent._checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
