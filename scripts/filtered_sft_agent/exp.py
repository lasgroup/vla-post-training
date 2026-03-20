# ruff: noqa: E402

# uncomment to force determinism
# import os
# os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"

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

from flax.training import common_utils
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm
import wandb

from src.rl.filtered_sft_agent.filtered_sft_learner import (
    FilteredSFTLearner,
    filtered_sft_wrap_env,
)
from src.envs import make_env
import src.training.config as _config
from src.training.collect import collect_data, evaluate_policy
from src.training.utils import init_logging, init_wandb


def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    env_fn, task_description = make_env(config, config.collect.tasks)
    env = filtered_sft_wrap_env(
        env_fn=env_fn,
        config=config,
        task_description=task_description,
    )
    eval_env_fn, eval_task_description = make_env(config, config.collect.eval_tasks)
    eval_env = filtered_sft_wrap_env(
        env_fn=eval_env_fn,
        config=config,
        task_description=eval_task_description,
        env_num=config.collect.eval_env_num,
    )

    agent = FilteredSFTLearner(config)
    init_wandb(config, resuming=agent._resuming, enabled=config.wandb_enabled)

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
            # Infos may have different keys (actor-only, critic-only, or both),
            # so we normalize them before stacking.
            all_keys = set().union(*(d.keys() for d in infos))
            nan = jnp.array(float("nan"))
            normalized = [{k: d.get(k, nan) for k in sorted(all_keys)} for d in infos]
            stacked_infos = common_utils.stack_forest(normalized)
            reduced_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        if step % config.collect.collect_interval == 0:
            agent.save_checkpoint(step=step)
            collect_info, n_collected_episodes = collect_data(
                agent=agent,
                env=env,
                config=config,
                task_description=task_description,
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

        if (
            step % config.save_interval == 0 and step > start_step
        ) or step == config.num_train_steps - 1:
            agent.save_checkpoint(step=step)

    logging.info("Waiting for checkpoint manager to finish")
    agent._checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
