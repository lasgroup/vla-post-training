# ruff: noqa: E402
import time
_PROCESS_START_TIME = time.monotonic()
import sys
REQUEUE_EXIT_CODE = 42

# uncomment to force determinism
# import os
# os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"


import warnings


def _silence_known_warnings():
    # this should only affect the hash algorithm for Numba caches
    warnings.filterwarnings("ignore", category=UserWarning, message=".*FNV hashing.*")
    # flax deprecation from openpi code
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax.nnx.statelib")
    # deprecation of type hints, partially from openpi
    warnings.filterwarnings("ignore", message=".*deprecated by PEP 585.*")
    # gymnasium does not like reading attributes/methods through wrappers
    warnings.filterwarnings("ignore", message=".*env.seed to get variables.*")

_silence_known_warnings()


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

import os
import platform
import threading

from flax.training import common_utils
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm


from src.envs import make_env
from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import AdvantageWeightedSFTLearner
from src.rl.best_of_n.best_of_n_learner import BestofNLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner
import src.training.config as _config
from src.training.collect import collect_data, evaluate_policy, heartbeat, HEARTBEAT
from src.training.runtime_state import save_epoch_state, load_resume_state
from src.training import parallel_collect
from src.training.utils import init_logging, Logger


def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # OGPOSFTLearnerConfig subclasses AdvantageWeightedSFTLearnerConfig, so it
    # must be checked first.
    if isinstance(config.rl, _config.OGPOSFTLearnerConfig):
        algo_class = OGPOAgentLearner
    elif isinstance(config.rl, _config.AdvantageWeightedSFTLearnerConfig):
        algo_class = AdvantageWeightedSFTLearner
    elif isinstance(config.rl, _config.BestofNLearnerConfig):
        algo_class = BestofNLearner
    elif isinstance(config.rl, _config.FilteredSFTLearnerConfig):
        algo_class = FilteredSFTLearner
    else:
        raise ValueError(f"Unsupported algorithm: {config.rl}")
    agent = algo_class(config)
    logger = Logger(config, resuming=agent._resuming, enabled=config.wandb_enabled)

    num_devices = max(1, jax.device_count())
    start_step = int(agent.training_steps)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # A collection round paused by collect.max_runtime resumes at the same step.
    resume_state = getattr(agent, "_resume_state", None) if agent._resuming else None
    collection_progress = None if resume_state is None else resume_state.collection_progress
    if collection_progress is not None:
        assert collection_progress["step"] == start_step, (collection_progress, start_step)

    def collection_out_of_time():
        return (
            config.collect.max_runtime is not None
            and (time.monotonic() - _PROCESS_START_TIME) >= config.collect.max_runtime
        )

    # Collection pauses at collect.max_runtime (save, exit for requeue) and every
    # collect.checkpoint_interval (save, keep collecting), so a crash loses at most
    # one interval of rollouts.
    should_pause = None
    part_start_time = time.monotonic()
    if config.collect.max_runtime is not None or config.collect.checkpoint_interval is not None:
        def should_pause():
            return collection_out_of_time() or (
                config.collect.checkpoint_interval is not None
                and (time.monotonic() - part_start_time) >= config.collect.checkpoint_interval
            )

    # Parallel collection (collect.num_collectors): the training job hands each round
    # to collector jobs; a collector job (collect.collector_index) collects one slice
    # of the tasks and exits. See src/training/parallel_collect.py.
    is_collector = config.collect.collector_index is not None
    delegate_collection = bool(config.collect.num_collectors) and not is_collector
    if is_collector:
        parallel_collect.start_collector(agent, config)

    if config.collect.checkpoint_interval is not None:
        # Env creation/reset can hang without ever erroring (seen once: Molmo workers
        # stuck after "starting"). Exit with an error after 30 min without collection
        # progress so the job wrapper's crash-retry requeues it on a fresh node.
        def _stall_watchdog(limit=int(os.environ.get("COLLECT_STALL_LIMIT", 1800))):
            while True:
                time.sleep(60)
                t = HEARTBEAT["t"]
                if t is not None and time.monotonic() - t > limit:
                    logging.error("No collection progress for %d s; exiting so the job is retried.", limit)
                    logging.shutdown()
                    os._exit(1)

        threading.Thread(target=_stall_watchdog, daemon=True).start()

    infos = []
    for step in pbar:

        if step % config.collect.collect_interval == 0 and delegate_collection:
            merged = parallel_collect.run_round(agent, config, step, collection_progress)
            if merged is None:
                logging.info("Exiting at step %d to launch parallel collectors.", step)
                sys.exit(parallel_collect.PARALLEL_EXIT_CODE)
            collect_info, n_collected_episodes = merged
            collection_progress = None
            logger.log_metrics(collect_info, step=step)

        elif step % config.collect.collect_interval == 0:
            heartbeat()
            env_fn = make_env(config, config.collect.tasks, num_devices=num_devices)
            env = filtered_sft_wrap_env(
                env_fn=env_fn,
                config=config,
            )
            while True:
                part_start_time = time.monotonic()
                collect_info, n_collected_episodes, collection_progress = collect_data(
                    agent=agent,
                    env=env,
                    config=config,
                    step=step,
                    progress=collection_progress,
                    should_pause=should_pause,
                )
                if collection_progress is None:
                    break
                save_epoch_state(
                    agent, config, prepare_for_resume=True, collection_progress=collection_progress
                )
                if collection_out_of_time():
                    env.close()
                    logging.info("Exiting at step %d mid-collection for requeue.", step)
                    sys.exit(REQUEUE_EXIT_CODE)
                logging.info("Checkpointed collection at step %d; continuing.", step)
                # Fresh env workers for every part: Molmo workers grow by ~2-3 GB/h
                # each and would otherwise OOM the node late in an allocation.
                env.close()
                heartbeat()
                env = filtered_sft_wrap_env(
                    env_fn=make_env(config, config.collect.tasks, num_devices=num_devices),
                    config=config,
                )
                logging.info("Recreated collection env workers at step %d.", step)
            env.close()
            heartbeat(active=False)
            if is_collector:
                parallel_collect.finish_collector(agent, config, step, collect_info, n_collected_episodes)
                sys.exit(0)
            logger.log_metrics(collect_info, step=step)
            if n_collected_episodes > 0:
                logging.info(
                    f"Collected {n_collected_episodes} successful episodes at step {step}."
                )

        if (step > 0) and step % config.collect.eval_interval == 0:  # skip first eval
            if config.free_buffer_before_eval:
                save_epoch_state(agent, config, prepare_for_resume=True)
                del agent._online_data_buffer

            # Molmo envs can hold onto GPU render memory, so keep eval envs
            # short-lived instead of reserving that memory for the whole run.
            eval_env_fn = make_env(
                config,
                config.collect.eval_tasks,
                num_devices=num_devices,
            )
            eval_env = filtered_sft_wrap_env(
                eval_env_fn,
                config=config,
                env_num=config.collect.eval_env_num,
            )
            eval_info = evaluate_policy(
                agent=agent,
                env=eval_env,
                config=config,
                step=step,
            )
            eval_env.close()
            logger.log_metrics(eval_info, step=step)
            logging.info(
                f"Eval at step {step}: {', '.join(f'{k}={v:.4f}' for k, v in eval_info.items())}"
            )
            if config.free_buffer_before_eval:
                resume_state = load_resume_state(config)
                agent._online_data_buffer = agent._get_online_replay_buffer()
                agent._online_data_buffer.restore_shards(resume_state.replay_shard_dir, rng_state_json=resume_state.replay_rng_state_json)

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
            logger.log_metrics(reduced_info, step=step)
            infos = []

        about_to_collect = (step + 1) % config.collect.collect_interval == 0
        about_to_eval = (step + 1) % config.collect.eval_interval == 0
        out_of_time = (time.monotonic() - _PROCESS_START_TIME) >= config.max_runtime
        if about_to_collect or about_to_eval or out_of_time:
            save_epoch_state(agent, config, prepare_for_resume=True)
            if out_of_time or (about_to_eval and config.requeue_before_eval):
                logging.info("Exiting at step %d for requeue.", step)
                sys.exit(REQUEUE_EXIT_CODE)


if __name__ == "__main__":
    main(_config.cli())