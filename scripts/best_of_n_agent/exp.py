# ruff: noqa: E402

# uncomment to force determinism
import os
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"

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
import wandb


from src.envs import make_env
from src.rl.best_of_n.best_of_n_learner import BestofNLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env
import src.training.config as _config
from src.training.collect import evaluate_policy
from src.training.utils import init_logging, init_wandb
from src.training.train_loop import train_loop


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

    agent = BestofNLearner(config=config)
    init_wandb(config, resuming=agent._resuming, enabled=config.wandb_enabled)

    if not agent._resuming:
        initial_eval_info = evaluate_policy(
            agent=agent,
            env=eval_env,
            task_description=eval_task_description,
            config=config,
            step=0,
        )
        wandb.log(initial_eval_info, step=0)
        logging.info(
            f"Initial eval (step 0): {', '.join(f'{k}={v:.4f}' for k, v in initial_eval_info.items())}"
        )

    train_loop(
        config=config,
        agent=agent,
        env=env,
        eval_env=eval_env,
        task_description=task_description,
        eval_task_description=eval_task_description,
    )


if __name__ == "__main__":
    main(_config.cli())
