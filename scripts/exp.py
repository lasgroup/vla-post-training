# ruff: noqa: E402
import time

_PROCESS_START_TIME = time.monotonic()

# uncomment to force determinism
# import os
# os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"

import logging
import multiprocessing as mp
import platform
import sys
import warnings

# this should only affect the hash algorithm for Numba caches
warnings.filterwarnings("ignore", category=UserWarning, message=".*FNV hashing.*")
# flax deprecation from openpi code
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.nnx.statelib")
# deprecation of type hints, partially from openpi
warnings.filterwarnings("ignore", message=".*deprecated by PEP 585.*")
# gymnasium does not like reading attributes/methods through wrappers
warnings.filterwarnings("ignore", message=".*env.seed to get variables.*")

# env workers are subprocesses; they must not fork an initialized JAX runtime
mp.set_start_method("spawn", force=True)

from flax.training import common_utils
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm

from src.envs import make_env
from src.rl.best_of_n.best_of_n_learner import BestofNLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner, filtered_sft_wrap_env
import src.training.config as _config
from src.training.collect import collect_data, evaluate_policy
from src.training.runtime_state import save_epoch_state, load_resume_state
from src.training.utils import init_logging, Logger

REQUEUE_EXIT_CODE = 42


def _make_vector_env(config: _config.OnlineTrainConfig, tasks: list[str], env_num: int | None = None):
    env_fn = make_env(config, tasks, num_devices=jax.device_count())
    return filtered_sft_wrap_env(env_fn, config=config, env_num=env_num)


def _collect(agent, config: _config.OnlineTrainConfig, logger: Logger, step: int) -> None:
    env = _make_vector_env(config, config.collect.tasks)
    collect_info, n_collected_episodes = collect_data(agent=agent, env=env, config=config, step=step)
    env.close()
    logger.log_metrics(collect_info, step=step)
    if n_collected_episodes > 0:
        logging.info(f"Collected {n_collected_episodes} successful episodes at step {step}.")


def _evaluate(agent, config: _config.OnlineTrainConfig, logger: Logger, step: int) -> None:
    if config.free_buffer_before_eval:
        # Persist the replay buffer and free its host memory for the duration of the eval.
        save_epoch_state(agent, config)
        del agent._online_data_buffer

    # Molmo envs can hold onto GPU render memory, so keep eval envs
    # short-lived instead of reserving that memory for the whole run.
    env = _make_vector_env(config, config.collect.eval_tasks, env_num=config.collect.eval_env_num)
    eval_info = evaluate_policy(agent=agent, env=env, config=config)
    env.close()
    logger.log_metrics(eval_info, step=step)
    logging.info(f"Eval at step {step}: {', '.join(f'{k}={v:.4f}' for k, v in eval_info.items())}")

    if config.free_buffer_before_eval:
        resume_state = load_resume_state(config)
        agent._online_data_buffer = agent._get_online_replay_buffer()
        agent._online_data_buffer.restore_shards(
            resume_state.replay_shard_dir, rng_state_json=resume_state.replay_rng_state_json
        )


def _reduce_infos(infos: list[dict]) -> dict:
    """Average update infos over steps; keys missing from a step count as NaN and are ignored."""
    all_keys = sorted(set().union(*(d.keys() for d in infos)))
    nan = jnp.array(float("nan"))
    stacked = common_utils.stack_forest([{k: d.get(k, nan) for k in all_keys} for d in infos])
    return jax.device_get(jax.tree.map(jnp.nanmean, stacked))


def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # BestofNLearnerConfig subclasses FilteredSFTLearnerConfig, so it must be checked first.
    if isinstance(config.rl, _config.BestofNLearnerConfig):
        agent = BestofNLearner(config)
    elif isinstance(config.rl, _config.FilteredSFTLearnerConfig):
        agent = FilteredSFTLearner(config)
    else:
        raise ValueError(f"Unsupported algorithm: {config.rl}")
    logger = Logger(config, resuming=agent._resuming)

    start_step = int(agent.training_steps)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    requeue = False
    infos = []
    for step in pbar:
        if step % config.collect.collect_interval == 0:
            _collect(agent, config, logger, step)

        if step > 0 and step % config.collect.eval_interval == 0:  # skip eval at step 0
            _evaluate(agent, config, logger, step)

        infos.append(agent.update())

        if step % config.log_interval == 0:
            reduced_info = _reduce_infos(infos)
            pbar.write(f"Step {step}: {', '.join(f'{k}={v:.4f}' for k, v in reduced_info.items())}")
            logger.log_metrics(reduced_info, step=step)
            infos = []

        about_to_collect = (step + 1) % config.collect.collect_interval == 0
        about_to_eval = (step + 1) % config.collect.eval_interval == 0
        out_of_time = (time.monotonic() - _PROCESS_START_TIME) >= config.max_runtime
        if about_to_collect or about_to_eval or out_of_time:
            save_epoch_state(agent, config)
            if out_of_time or (about_to_eval and config.requeue_before_eval):
                logging.info("Exiting at step %d for requeue.", step)
                requeue = True
                break

    agent._checkpoint_manager.close()
    if requeue:
        sys.exit(REQUEUE_EXIT_CODE)


if __name__ == "__main__":
    main(_config.cli())
