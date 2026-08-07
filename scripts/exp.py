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

import platform

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
from src.training.collect import collect_data, evaluate_policy
from src.training.runtime_state import save_epoch_state, load_resume_state
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

    infos = []
    for step in pbar:

        if step % config.collect.collect_interval == 0:
            env_fn = make_env(config, config.collect.tasks, num_devices=num_devices)
            env = filtered_sft_wrap_env(
                env_fn=env_fn,
                config=config,
            )
            collect_info, n_collected_episodes = collect_data(
                agent=agent,
                env=env,
                config=config,
                step=step,
            )
            env.close()
            logger.log_metrics(collect_info, step=step)
            if n_collected_episodes > 0:
                logging.info(
                    f"Collected {n_collected_episodes} successful episodes at step {step}."
                )
            # Critic digestion burst: let the critic fit the just-collected
            # distribution before the actor's next update ranks fresh actions
            # with it. Step 0 is skipped — policy.training_start_step already
            # provides the initial actor-free head start.
            if (
                step > 0
                and getattr(config.rl, "post_collection_critic_steps", 0) > 0
                and hasattr(agent, "critic_digestion_burst")
            ):
                burst_info = agent.critic_digestion_burst()
                if burst_info:
                    logger.log_metrics(burst_info, step=step)
                    logging.info(
                        f"Critic digestion burst at step {step}: "
                        f"{int(burst_info['burst/steps'])} critic-only updates."
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
        runtime_exceeded = (time.monotonic() - _PROCESS_START_TIME) >= config.max_runtime
        out_of_time = runtime_exceeded and (about_to_collect or about_to_eval)
        if about_to_collect or about_to_eval or out_of_time:
            save_epoch_state(agent, config, prepare_for_resume=True)
            if out_of_time or (about_to_eval and config.requeue_before_eval):
                logging.info("Exiting at step %d for requeue.", step)
                sys.exit(REQUEUE_EXIT_CODE)


if __name__ == "__main__":
    main(_config.cli())