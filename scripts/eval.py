# ruff: noqa: E402
"""Evaluate a (base or trained) policy on a set of tasks.

Runs `evaluate_policy` over `config.collect.eval_tasks` and reports per-task
success rates to the console and (optionally) to W&B.

Base policy (default): load the base pi0.5 LIBERO weights via the
config's weight loader, so no trained checkpoint is needed -- run with
`resume=False` (the default).

Trained checkpoint: point `checkpoint_base_dir` + `exp_name` at an existing run
and pass `--resume` to restore its (EMA) weights. NOTE: `--resume` always
restores the replay shards (see FilteredSFTLearner.__init__) even though eval
never samples from the buffer; there is no `--no-requeue` flag any more. To skip
that 12-25 GB load, point `checkpoint_base_dir` at a staging directory whose
`runtime_state/replay_shards/` is empty.

Which checkpoint is loaded is decided ENTIRELY by
`runtime_state/resume_state.json`'s "step" -- both the policy and the critic
(`rl_state/<step>`) are restored at that step. There is no --eval_step flag.

Set EVAL_METRIC_PREFIX to log under a namespace other than `eval/` (e.g.
`eval_newtasks`). Without it, re-evaluating a finished run overwrites its
original `eval/success_rate` at the same wandb step.

Examples:
    # Base pi0.5 on all 90 LIBERO-90 tasks (via the launcher / eval.yaml)
    ./scripts/launcher.py --config scripts/configs/eval.yaml

    # A trained checkpoint, run directly
    EVAL_METRIC_PREFIX=eval_newtasks uv run scripts/eval.py pi05_libero_online_aw_sft \\
        --checkpoint_base_dir /capstor/scratch/cscs/ralfroemer/checkpoints \\
        --exp_name <run_dir_name> \\
        --resume \\
        --collect.eval_tasks libero_90_44 libero_90_59 libero_90_5 \\
        --collect.num_eval_rollouts 5 --collect.eval_env_num 5 \\
        --rl.online_ratio 1.0
"""

# suppress Numba FNV hashing warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*FNV hashing.*")

# suppress lerobot version warnings
import logging


class VersionWarningFilter(logging.Filter):
    def filter(self, record):
        return "is in 2.0 format" not in record.getMessage()


logging.getLogger().addFilter(VersionWarningFilter())

# disable datasets progress bars
from datasets import disable_progress_bars

disable_progress_bars()

# allows using subprocenvs
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import json
import os
import platform

import jax
import wandb

from src.envs import make_env
from src.rl.advantage_weighted_sft.advantage_weighted_sft_learner import AdvantageWeightedSFTLearner
from src.rl.best_of_n.best_of_n_learner import BestofNLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env
from src.rl.ogpo.ogpo_learner import OGPOAgentLearner
import src.training.config as _config
from src.training.collect import evaluate_policy
from src.training.utils import init_logging, init_wandb


def _make_agent(config: _config.OnlineTrainConfig):
    # OGPOSFTLearnerConfig subclasses AdvantageWeightedSFTLearnerConfig, so it
    # must be checked first. Mirror the dispatch in scripts/exp.py.
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
    return algo_class(config)


def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    agent = _make_agent(config)
    init_wandb(config, resuming=agent._resuming, enabled=config.wandb_enabled)

    step = int(agent.training_steps)
    num_devices = max(1, jax.device_count())

    eval_env_fn = make_env(config, config.collect.eval_tasks, num_devices=num_devices)
    eval_env = filtered_sft_wrap_env(
        eval_env_fn,
        config=config,
        env_num=config.collect.eval_env_num,
    )
    try:
        metrics = evaluate_policy(agent=agent, env=eval_env, config=config, step=step)
    finally:
        eval_env.close()

    # Re-namespace the metrics when asked. `evaluate_policy` always emits
    # `eval/...`; logging that into a finished run at its original eval step
    # would overwrite the eval already recorded there (same wandb step, same
    # key). A distinct prefix keeps both side by side.
    prefix = (os.environ.get("EVAL_METRIC_PREFIX") or "eval").strip().strip("/") or "eval"
    if prefix != "eval":
        metrics = {
            (f"{prefix}/{k[len('eval/'):]}" if k.startswith("eval/") else k): v
            for k, v in metrics.items()
        }

    if config.wandb_enabled:
        wandb.log(metrics, step=step)

    # Console report: overall first, then per-task success rates sorted by task.
    logging.info(f"=== Eval at step {step} (metric prefix: {prefix}/) ===")
    overall = metrics.get(f"{prefix}/success_rate")
    if overall is not None:
        logging.info(f"overall success_rate: {overall:.4f}")
    per_task = {
        k.removeprefix(f"{prefix}/success_rate/"): v
        for k, v in metrics.items()
        if k.startswith(f"{prefix}/success_rate/")
    }
    for task in sorted(per_task):
        logging.info(f"  {task}: {per_task[task]:.4f}")

    # Persist a JSON summary next to the checkpoint dir for offline inspection.
    # The prefix is in the filename so a re-eval cannot clobber an earlier one.
    out_path = config.checkpoint_dir / f"eval_metrics_{prefix}_step{step}.json"
    try:
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Wrote eval metrics to {out_path}")
    except OSError as e:
        logging.warning(f"Could not write eval metrics to {out_path}: {e}")


if __name__ == "__main__":
    main(_config.cli())
