#!/usr/bin/env python3
# ruff: noqa: E402
import json
import logging
import os
import platform
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import disable_progress_bars

disable_progress_bars()

import jax
from openpi.policies import policy_config
from openpi_client import image_tools

from src.envs import make_env
from src.rl.agent import Agent
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner, filtered_sft_wrap_env
import src.training.config as _config
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
        processed_obs = self._process_obs_for_pi0(observations, task_description=kwargs["task_description"])
        first_obs = np.asarray(next(iter(processed_obs.values())))
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1
        outputs = self._policy.infer(processed_obs, sharding_spec=None)
        actions = outputs["actions"]
        if batch_size == 1 and actions.ndim == 2:
            actions = actions[np.newaxis, ...]
        return np.asarray(actions, dtype=np.float32)

    def eval_actions(self, observations, **kwargs):
        return self.sample_actions(observations, **kwargs)
    def save_checkpoint(self, step): return None
    def add_data(self, step_data): return None
    def save_episode(self, is_success=False, env_index=0, **kwargs): return None
    def start_data_collection(self, step=None): return None
    def end_data_collection(self, step=None): return 0
    def update(self): return {}


def _get_image(obs):
    if "observation/image" in obs:
        return obs["observation/image"]
    if "image" in obs:
        return obs["image"]
    raise KeyError(f"No image observation key found. Keys={list(obs.keys())}")


def _frame_from_obs(obs):
    image = _get_image(obs)
    arr = np.asarray(image)
    # Vector env + query wrapper: (env, history, H, W, C). Keep env 0 and allow caller to pass full chunks.
    if arr.ndim == 5:
        arr = arr[0, -1]
    elif arr.ndim == 4:
        # Could be (history,H,W,C) or (env,H,W,C). Use the last temporal/env slice; env_num=1 either way.
        arr = arr[-1]
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _chunk_frames_from_obs(obs):
    arr = np.asarray(_get_image(obs))
    if arr.ndim == 5:
        frames = [arr[0, i] for i in range(arr.shape[1])]
    elif arr.ndim == 4:
        frames = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 3:
        frames = [arr]
    else:
        raise ValueError(f"Unsupported image shape {arr.shape}")
    out = []
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        out.append(f)
    return out


def _save_gif(frames, path: Path, stride: int = 4, duration_ms: int = 90):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to save")
    selected = frames[::max(1, stride)]
    if not np.array_equal(selected[-1], frames[-1]):
        selected.append(frames[-1])
    images = [Image.fromarray(f) for f in selected]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True)


def main(config):
    init_logging()
    kind = os.environ.get("RENDER_POLICY_KIND", "sft")
    label = os.environ.get("RENDER_POLICY_LABEL", kind)
    out_dir = Path(os.environ["RENDER_OUTPUT_DIR"])
    episodes = int(os.environ.get("RENDER_EPISODES", "3"))
    frame_stride = int(os.environ.get("RENDER_FRAME_STRIDE", "4"))
    logging.info("Rendering GIFs on %s: kind=%s label=%s out=%s", platform.node(), kind, label, out_dir)

    if kind == "pretrained":
        agent = DirectPretrainedPolicyAgent(config)
    elif kind == "sft":
        agent = FilteredSFTLearner(config)
    else:
        raise ValueError(f"Unknown RENDER_POLICY_KIND={kind}")

    tasks = config.collect.eval_tasks
    if isinstance(tasks, str):
        tasks = [tasks]
    task_id = tasks[0]
    num_render_devices = max(1, int(os.environ.get("MUJOCO_EGL_NUM_DEVICES", "1")))
    env_fn = make_env(config, tasks, num_devices=num_render_devices)
    env = filtered_sft_wrap_env(env_fn, config=config, env_num=1)
    summary = []
    try:
        for ep in range(episodes):
            obs, info = env.reset(options={"task_id": [task_id]})
            frames = [_frame_from_obs(obs)]
            total_steps = 0
            success = False
            for _ in range(config.collect.max_episode_steps // max(1, config.collect.replan_steps) + 5):
                action_chunk = agent.sample_actions(obs, task_description=info["task_description"])
                env_action_chunk = action_chunk[0] if isinstance(action_chunk, tuple) else action_chunk
                next_obs, _, terminate, truncate, _ = env.step(env_action_chunk)
                frames.extend(_chunk_frames_from_obs(next_obs))
                done_per_step = np.logical_or(terminate, truncate)
                any_done = bool(done_per_step[0, -1])
                first_done_idx = int(np.argmax(done_per_step[0])) if any_done else config.collect.replan_steps - 1
                total_steps += first_done_idx + 1 if any_done else config.collect.replan_steps
                success = bool(terminate[0, -1])
                obs = next_obs
                if any_done:
                    break
            gif_path = out_dir / f"episode_{ep:02d}_success{int(success)}.gif"
            _save_gif(frames, gif_path, stride=frame_stride)
            summary.append({"episode": ep, "success": success, "steps": total_steps, "frames": len(frames), "gif": str(gif_path)})
            print(json.dumps(summary[-1], sort_keys=True), flush=True)
    finally:
        env.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "render_summary.json").write_text(json.dumps({"label": label, "kind": kind, "episodes": summary}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(_config.cli())
