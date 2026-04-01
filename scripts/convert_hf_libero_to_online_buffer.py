#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

import jax
import numpy as np
import openpi.transforms as _transforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from src.rl.replay_buffer import ShardedReplayBuffer
from src.training.config import get_config


def _canonicalize_image(image: Any) -> np.ndarray:
    img = image.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img * 255.0, 0.0, 255.0)
    img = img.astype(np.uint8)
    return np.ascontiguousarray(img)


def _extract_step(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation": {
            "observation/image": _canonicalize_image(frame['image']),
            "observation/wrist_image": _canonicalize_image(frame['wrist_image']),
            "observation/state": frame['state'].numpy(),
        },
        "action": frame["actions"].numpy(),
        "reward": -1.0,
        "done": False,
        "prompt": str(frame["task"]),
    }


def _episode_to_online_payload(
    steps: list[dict[str, Any]],
    *,
    action_horizon: int,
    discount: float,
    pre_token_transform,
    token_transform,
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any] | None:
    if not steps:
        return None

    observations = [s["observation"] for s in steps]
    # duplicate final observation
    next_observations = observations[1:] + [observations[-1]]
    actions = np.stack([s["action"] for s in steps], axis=0)
    rewards = np.asarray([s["reward"] for s in steps], dtype=np.float32)
    done = np.asarray([s["done"] for s in steps], dtype=bool)
    prompt = str(steps[0]["prompt"])
    # assume success
    done[-1] = True
    rewards[-1] = 0.0

    n_steps = int(np.where(done)[0][0] + 1)
    n_windows = n_steps - action_horizon + 1
    if n_windows <= 0:
        return None

    all_gammas = np.asarray([discount**i for i in range(n_steps)], dtype=np.float32)
    w_gammas = all_gammas[:action_horizon]
    last_gamma = float(discount**action_horizon)

    def strip_obs_prefix(k: str) -> str:
        return k.replace("observation/", "")

    stacked_obs = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *observations)
    stacked_next_obs = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *next_observations)

    _obs = {strip_obs_prefix(k): v[:n_windows] for k, v in stacked_obs.items()}
    _next_obs = {
        strip_obs_prefix(k): v[action_horizon - 1 : n_windows + action_horizon - 1]
        for k, v in stacked_next_obs.items()
    }

    _actions = np.stack([actions[start : start + action_horizon] for start in range(n_windows)], axis=0)
    _actions = np.where(np.abs(_actions) < 0.0011, 0.0, _actions)
    _reward = np.asarray(
        [(rewards[start : start + action_horizon] * w_gammas).sum() for start in range(n_windows)],
        dtype=np.float32,
    )
    _discount = np.asarray(
        [0.0 if np.any(done[start : start + action_horizon]) else last_gamma for start in range(n_windows)],
        dtype=np.float32,
    )
    _mc_return = ((all_gammas * rewards[:n_steps])[::-1].cumsum()[::-1] / all_gammas)[:n_windows].astype(np.float32)

    def transform(obs_dict: dict[str, Any], act: np.ndarray, text_prompt: str):
        obs_with_aux = dict(obs_dict)
        obs_with_aux.update({"actions": act, "prompt": text_prompt})
        proc = pre_token_transform(obs_with_aux)
        proc["image_mask"] = {k: np.full((n_windows,), bool(v)) for k, v in proc["image_mask"].items()}
        if isinstance(token_transform, _transforms.TokenizePrompt):
            if text_prompt not in token_cache:
                tok = token_transform({"prompt": text_prompt})
                token_cache[text_prompt] = (tok["tokenized_prompt"], tok["tokenized_prompt_mask"])
            tok_prompt, tok_mask = token_cache[text_prompt]
            proc["tokenized_prompt"] = np.broadcast_to(
                tok_prompt, (n_windows,) + tok_prompt.shape
            ).copy()
            proc["tokenized_prompt_mask"] = np.broadcast_to(
                tok_mask, (n_windows,) + tok_mask.shape
            ).copy()
        else:
            raise TypeError(f"Unsupported token transform: {type(token_transform)}")
        out_actions = proc.pop("actions")
        proc.pop("prompt")
        return proc, out_actions

    _next_obs, _ = transform(_next_obs, _actions, prompt)
    _obs, _actions = transform(_obs, _actions, prompt)

    return {
        "observation": _obs,
        "actions": _actions.astype(np.float32),
        "next_observation": _next_obs,
        "reward": _reward,
        "mc_return": _mc_return,
        "discount": _discount,
    }


def _build_transforms(config_name: str):
    train_cfg = get_config(config_name)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    tt_types = (_transforms.TokenizePrompt, _transforms.TokenizeFASTInputs)
    token_transforms = [t for t in data_cfg.model_transforms.inputs if isinstance(t, tt_types)]
    non_token_transforms = [t for t in data_cfg.model_transforms.inputs if not isinstance(t, tt_types)]
    if len(token_transforms) != 1:
        raise ValueError(f"Expected exactly one token transform, got {len(token_transforms)}")
    token_transform = token_transforms[0]
    pre_token_transform = _transforms.compose(
        [
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            _transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *non_token_transforms,
        ]
    )
    return train_cfg, data_cfg, pre_token_transform, token_transform


def run(args):

    train_cfg, _, pre_token_transform, token_transform = _build_transforms(args.config_name)
    action_horizon = int(train_cfg.model.action_horizon)
    discount = float(train_cfg.rl.discount)
    token_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    ds = LeRobotDataset("physical-intelligence/libero", root=None)
    print(f"Loaded dataset: frames={len(ds)}, episodes={getattr(ds, 'num_episodes', 'unknown')}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer: ShardedReplayBuffer | None = None
    processed_episodes = 0

    current_ep_id = 0
    current_steps: list[dict[str, Any]] = []

    def flush_episode(ep_steps: list[dict[str, Any]]):
        nonlocal buffer
        payload = _episode_to_online_payload(
            ep_steps,
            action_horizon=action_horizon,
            discount=discount,
            pre_token_transform=pre_token_transform,
            token_transform=token_transform,
            token_cache=token_cache,
        )
        if buffer is None:
            dummy_data = jax.tree_util.tree_map(lambda x: x[:1], payload)
            buffer = ShardedReplayBuffer(
                dummy_data=dummy_data,
                max_capacity=int(len(ds)),
                data_sharding=None,
                seed=0,
                preprocess_fn=None,
                postprocess_fn=None,
                freeze_dict=False,
                load_paths=None,
                save_path=str(output_dir),
            )
        buffer.insert(payload, save_episode=True)

    for idx in range(len(ds)):
        frame = ds[idx]
        ep_id = frame['episode_index'].numpy()
        if ep_id != current_ep_id:
            processed_episodes += 1
            print(f"Processing episode {ep_id}")
            flush_episode(current_steps)
            current_steps = []
            current_ep_id = ep_id

        current_steps.append(_extract_step(frame))

        if args.max_episodes is not None and processed_episodes >= args.max_episodes:
            break

    if current_steps and (args.max_episodes is None or processed_episodes < args.max_episodes):
        processed_episodes += 1
        flush_episode(current_steps)

    print("Finished conversion.")
    print(f"Processed episodes: {processed_episodes}")
    print(f"Saved episode files to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="pi05_libero_online_filtered_sft")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())