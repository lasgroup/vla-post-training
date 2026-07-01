#!/usr/bin/env python3
# ruff: noqa: E402
import json
import logging
import os
import platform
from pathlib import Path

import numpy as np
from datasets import disable_progress_bars

disable_progress_bars()

import jax
from openpi.policies import policy_config
from openpi_client import image_tools

from src.envs import make_env
from src.rl.agent import Agent
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env
import src.training.config as _config
from src.training.collect import evaluate_policy
from src.training.utils import init_logging


class DirectPretrainedPolicyAgent(Agent):
    def __init__(self, config):
        self._config = config
        params_path = str(config.weight_loader.params_path)
        checkpoint_dir = params_path[: -len("/params")] if params_path.endswith("/params") else params_path
        logging.info("Loading direct pretrained policy from %s", checkpoint_dir)
        self._policy = policy_config.create_trained_policy(config, checkpoint_dir)
        self.total_collected_episodes = 0

    def _process_obs_for_pi0(self, observations, task_description):
        obs = jax.tree_util.tree_map(lambda x: x[:, -1], observations)
        h = int(self._config.collect.resize_image_h)
        w = int(self._config.collect.resize_image_w)

        def resize_fn(x):
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(x, h, w))

        obs = {k: resize_fn(v) if "image" in k else v for k, v in obs.items()}
        obs["prompt"] = task_description
        return obs

    def sample_actions(self, observations, **kwargs):
        processed_obs = self._process_obs_for_pi0(
            observations,
            task_description=kwargs["task_description"],
        )
        first_obs = np.asarray(next(iter(processed_obs.values())))
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1
        outputs = self._policy.infer(processed_obs, sharding_spec=None)
        actions = outputs["actions"]
        if batch_size == 1 and actions.ndim == 2:
            actions = actions[np.newaxis, ...]
        return np.asarray(actions, dtype=np.float32)

    def eval_actions(self, observations, **kwargs):
        return self.sample_actions(observations, **kwargs)

    def save_checkpoint(self, step):
        return None

    def add_data(self, step_data):
        return None

    def save_episode(self, is_success=False, env_index=0, **kwargs):
        return None

    def start_data_collection(self, step=None):
        return None

    def end_data_collection(self, step=None):
        return 0

    def update(self):
        return {}


def _to_float_dict(info):
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


def main(config):
    init_logging()
    logging.info("Running DIRECT pretrained no-SFT eval on: %s", platform.node())
    logging.info("Config: %s", config)

    eval_tasks = config.collect.eval_tasks
    if isinstance(eval_tasks, str):
        eval_tasks = [eval_tasks]
    num_render_devices = max(1, int(os.environ.get("MUJOCO_EGL_NUM_DEVICES", "1")))
    eval_env_fn = make_env(config, eval_tasks, num_devices=num_render_devices)
    eval_env = filtered_sft_wrap_env(eval_env_fn, config=config, env_num=config.collect.eval_env_num)
    try:
        agent = DirectPretrainedPolicyAgent(config)
        metrics = _to_float_dict(evaluate_policy(agent=agent, env=eval_env, config=config, step=0))
    finally:
        eval_env.close()

    summary = {
        "event": "pretrained_direct_eval",
        "checkpoint": str(config.weight_loader.params_path),
        "task": list(eval_tasks),
        "eval_rollouts": int(config.collect.num_eval_rollouts),
        "eval_env_num": int(config.collect.eval_env_num),
        "metrics": metrics,
    }
    metrics_path = os.environ.get("PRETRAINED_EVAL_METRICS_PATH")
    if metrics_path:
        path = Path(metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(_config.cli())
