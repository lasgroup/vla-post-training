# ruff: noqa: E402
"""Dataset-only filtered SFT from preloaded rollout pickles.

This entrypoint intentionally never calls collect_data(...). It trains only from
episodes loaded via --rl.preload_episodes_from_path. Evaluation environments are
created only when PRELOADED_SFT_FINAL_EVAL=1 or PRELOADED_SFT_PERIODIC_EVAL=1.
"""

import json
import logging
import os
import platform
from pathlib import Path

from datasets import disable_progress_bars

# Keep env-worker/JAX import behavior consistent with the online entrypoint, even
# though this script does not spawn env workers.
disable_progress_bars()

from flax.training import common_utils
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm
import wandb

from src.envs import make_env
from src.rl.filtered_sft_agent.filtered_sft_learner import (
    FilteredSFTLearner,
    filtered_sft_wrap_env,
)
import src.training.config as _config
from src.training.collect import evaluate_policy
from src.training.runtime_state import save_epoch_state
from src.training.utils import init_logging, init_wandb


def _metrics_path(config: _config.OnlineTrainConfig) -> Path:
    return Path(config.checkpoint_dir) / "preloaded_sft_metrics.jsonl"


def _write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _to_float_dict(info: dict) -> dict[str, float]:
    out = {}
    for key, value in info.items():
        try:
            out[key] = float(jax.device_get(value))
        except Exception:
            try:
                out[key] = float(value)
            except Exception:
                out[key] = str(value)
    return out


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _make_eval_env(config: _config.OnlineTrainConfig):
    # MuJoCo/robosuite EGL device enumeration is not always the same as JAX's CUDA
    # device count inside CSCS EDF/Pyxis containers. Default to one render device so
    # vector env workers do not try invalid MUJOCO_EGL_DEVICE_ID values; override on
    # systems where multiple EGL render devices are known to work.
    num_render_devices = max(1, int(os.environ.get("MUJOCO_EGL_NUM_DEVICES", "1")))
    eval_tasks = config.collect.eval_tasks
    if isinstance(eval_tasks, str):
        eval_tasks = [eval_tasks]
    eval_env_fn = make_env(config, eval_tasks, num_devices=num_render_devices)
    return filtered_sft_wrap_env(
        eval_env_fn,
        config=config,
        env_num=config.collect.eval_env_num,
    )


def _run_eval(
    *,
    agent: FilteredSFTLearner,
    eval_env,
    config: _config.OnlineTrainConfig,
    step: int,
    metrics_path: Path,
    event: str,
) -> dict[str, float]:
    eval_metrics = _to_float_dict(
        evaluate_policy(
            agent=agent,
            env=eval_env,
            config=config,
            step=step,
        )
    )
    logging.info("%s metrics at step %d: %s", event, step, eval_metrics)
    wandb.log(eval_metrics if event == "eval" else {f"final/{k}": v for k, v in eval_metrics.items()}, step=step)
    _write_jsonl(metrics_path, {"event": event, "step": int(step), **eval_metrics})
    return eval_metrics


def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info("Running dataset-only preloaded SFT on: %s", platform.node())
    logging.info("Config: %s", config)

    rl_config = config.rl
    if not isinstance(rl_config, _config.FilteredSFTLearnerConfig):
        raise TypeError(f"preloaded SFT requires FilteredSFTLearnerConfig, got {type(rl_config)}")
    if rl_config.preload_episodes_from_path is None:
        raise ValueError("--rl.preload_episodes_from_path is required for preloaded SFT")
    if rl_config.online_ratio <= 0.0:
        raise ValueError("preloaded SFT requires --rl.online_ratio > 0; use 1.0 for dataset-only training")
    if config.num_train_steps <= 0:
        raise ValueError("--num_train_steps must be positive for preloaded SFT training")

    agent = FilteredSFTLearner(config)
    agent.preload_episodes()
    buffer_size = int(agent._online_data_buffer.size)
    if buffer_size <= 0:
        raise RuntimeError(
            "preload_episodes inserted zero transitions; check pickle path/schema and action horizon"
        )

    init_wandb(config, resuming=agent._resuming, enabled=config.wandb_enabled)

    metrics_path = _metrics_path(config)
    _write_jsonl(
        metrics_path,
        {
            "event": "preload_complete",
            "preload_episodes_from_path": str(rl_config.preload_episodes_from_path),
            "online_buffer_size": buffer_size,
            "checkpoint_dir": str(config.checkpoint_dir),
            "num_train_steps": int(config.num_train_steps),
            "batch_size": int(config.batch_size),
            "online_ratio": float(rl_config.online_ratio),
        },
    )

    start_step = int(agent.training_steps)
    periodic_eval_enabled = _env_flag("PRELOADED_SFT_PERIODIC_EVAL")
    final_eval_enabled = _env_flag("PRELOADED_SFT_FINAL_EVAL")
    eval_env = None
    if periodic_eval_enabled or final_eval_enabled:
        if config.collect.eval_interval <= 0 and periodic_eval_enabled:
            raise ValueError("periodic eval requires --collect.eval_interval > 0")
        logging.info(
            "Creating evaluation env: periodic=%s final=%s interval=%d rollouts=%d env_num=%d",
            periodic_eval_enabled,
            final_eval_enabled,
            int(config.collect.eval_interval),
            int(config.collect.num_eval_rollouts),
            int(config.collect.eval_env_num),
        )
        eval_env = _make_eval_env(config)

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    last_reduced_info = None
    final_eval_metrics = None
    try:
        for step in pbar:
            info = agent.update()
            if not info:
                raise RuntimeError(
                    f"agent.update returned empty info at step {step}; online buffer size={agent._online_data_buffer.size}"
                )
            infos.append(info)

            if step % config.log_interval == 0:
                all_keys = set().union(*(d.keys() for d in infos))
                nan = jnp.array(float("nan"))
                normalized = [{k: d.get(k, nan) for k in sorted(all_keys)} for d in infos]
                stacked_infos = common_utils.stack_forest(normalized)
                reduced_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_infos))
                reduced_info = _to_float_dict(reduced_info)
                last_reduced_info = reduced_info
                info_str = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in reduced_info.items()
                )
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
                _write_jsonl(metrics_path, {"event": "log", "step": int(step), **reduced_info})
                infos = []

            if periodic_eval_enabled and eval_env is not None and step % config.collect.eval_interval == 0:
                _run_eval(
                    agent=agent,
                    eval_env=eval_env,
                    config=config,
                    step=int(step),
                    metrics_path=metrics_path,
                    event="eval",
                )

            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                logging.info("Saving checkpoint at step %d", step)
                agent.save_checkpoint(step=step)
                # Runtime state is useful for requeue-safe launches, but keep this secondary to the model checkpoint.
                try:
                    save_epoch_state(agent, config)
                except Exception:
                    logging.exception("Failed to save runtime state; model checkpoint was still requested")

        agent._checkpoint_manager.wait_until_finished()

        if final_eval_enabled:
            if eval_env is None:
                eval_env = _make_eval_env(config)
            logging.info("Running final evaluation rollouts; this does not add data to the training buffer.")
            final_eval_metrics = _run_eval(
                agent=agent,
                eval_env=eval_env,
                config=config,
                step=int(config.num_train_steps - 1),
                metrics_path=metrics_path,
                event="final_eval",
            )
    finally:
        if eval_env is not None:
            eval_env.close()

    summary = {
        "event": "training_complete",
        "final_step": int(config.num_train_steps - 1),
        "online_buffer_size": int(agent._online_data_buffer.size),
        "checkpoint_dir": str(config.checkpoint_dir),
        "metrics_path": str(metrics_path),
        "last_metrics": last_reduced_info or {},
        "final_eval_metrics": final_eval_metrics or {},
    }
    _write_jsonl(metrics_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(_config.cli())
