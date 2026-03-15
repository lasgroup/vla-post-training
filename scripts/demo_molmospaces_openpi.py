import collections
import dataclasses
import importlib
import logging
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
import tyro
import cv2

from molmo_spaces.policy.learned_policy.utils import PromptSampler
from molmo_spaces.utils.save_utils import save_frames_to_mp4
from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
import openpi.models.pi0_config as _pi0_config
import openpi.policies.droid_policy as _droid_policy
import openpi.transforms as _openpi_transforms
from openpi.training import config as _openpi_config
from src.envs.molmo import MolmoSpacesBenchmarkGymEnv
from src.envs.molmo import MolmoSpacesGymConfig
from src.envs.molmo_openpi import obs_to_openpi_input as _shared_obs_to_openpi_input
from openpi.policies import policy_config as _policy_config
from openpi.shared import download as _openpi_download


@dataclasses.dataclass
class Args:
    # OpenPI policy selection.
    checkpoint_dir: str | None = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
    default_prompt: str | None = None

    # MolmoSpaces benchmark/env setup.
    benchmark_path: str = (
        "/capstor/store/cscs/swissai/a143/molmospaces/assets/benchmarks/"
        "molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/"
        "FrankaPickDroidMiniBench_json_benchmark_20251231"
    )
    eval_config_cls: str = (
        "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig"
    )
    episode_sampling: Literal["sequential", "random"] = "sequential"
    seed: int = 0
    task_horizon_steps: int | None = None

    # Observation mapping from MolmoSpaces observation dict.
    exo_camera_key: str = "exo_camera_1"
    wrist_camera_key: str = "wrist_camera"

    # Action mapping from OpenPI output to MolmoSpaces action dict.
    execute_horizon: int = 8  # defaults to train_cfg.model.action_horizon
    grasping_type: str | None = None  # one of {"continuous", "binary"}; defaults to eval config
    gripper_threshold: float | None = None  # defaults to eval config
    gripper_scale: float = 255.0

    # Rollout control.
    num_episodes: int | None = None
    max_steps: int = 450

    # Video visualization.
    save_trajectory_video: bool = True
    video_dir: str = "outputs/demo_videos"
    video_text_scale: float = 0.6
    video_text_thickness: int = 1


class _RegisteredPolicyAdapter:
    """Minimal policy interface for MolmoSpaces policy-dependent sensors."""

    def __init__(
        self,
        inner_policy: Any,
        policy_name: str,
        prompt_sampler: PromptSampler | None,
    ) -> None:
        self._inner_policy = inner_policy
        self._policy_name = policy_name
        self._prompt_sampler = prompt_sampler
        self.target_poses = {"grasp": np.eye(4, dtype=np.float32)}
        self.task = None

    def reset(self) -> None:
        if self._prompt_sampler is not None:
            self._prompt_sampler.next()
        reset_fn = getattr(self._inner_policy, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def get_prompt(self, default_prompt: str | None) -> str:
        if self._prompt_sampler is not None and self.task is not None:
            return self._prompt_sampler.get_prompt(self.task).lower()
        return (default_prompt or "do the task").lower()

    def get_phase(self) -> str:
        return "inference"

    def get_all_phases(self) -> dict[str, int]:
        return {"inference": 0}

    def get_info(self) -> dict[str, Any]:
        return {"policy_name": self._policy_name}

def _resolve_eval_config(eval_config_cls: str):
    if ":" not in eval_config_cls:
        raise ValueError(
            f"Invalid eval_config_cls '{eval_config_cls}'. Expected 'module.path:ClassName'."
        )
    module_name, class_name = eval_config_cls.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Could not resolve class '{class_name}' in module '{module_name}'."
        ) from exc
    return cls()

def _action_overlay(action: np.ndarray, width: int, height: int) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    dim = max(1, int(min(action.shape[0], 8)))
    panel = np.full((height, width, 3), 20, dtype=np.uint8)
    center = height // 2
    panel[max(0, center - 1) : min(height, center + 1), :, :] = 90

    bar_w = max(6, width // (dim * 3))
    gap = max(2, (width - (bar_w * dim)) // (dim + 1))
    usable_h = max(2, int(height * 0.45))

    for i in range(dim):
        v = float(np.clip(action[i], -1.0, 1.0))
        x0 = min(width - 1, gap + i * (bar_w + gap))
        x1 = min(width, x0 + bar_w)
        if x1 <= x0:
            continue
        if v >= 0:
            y0 = max(0, center - int(v * usable_h))
            y1 = center
            panel[y0:y1, x0:x1, :] = np.array([80, 200, 120], dtype=np.uint8)
        else:
            y0 = center
            y1 = min(height, center + int(abs(v) * usable_h))
            panel[y0:y1, x0:x1, :] = np.array([220, 90, 90], dtype=np.uint8)
    return panel


def _wrap_text_to_width(
    text: str,
    max_width_px: int,
    font,
    font_scale: float,
    thickness: int,
) -> list[str]:
    if cv2 is None:
        return [text]

    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_w = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if candidate_w <= max_width_px:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _overlay_prompt_on_frame(
    frame: np.ndarray,
    prompt: str,
    *,
    font_scale: float,
    thickness: int,
) -> np.ndarray:
    if cv2 is None:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    margin = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_h = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] + 8
    max_width = max(100, w - 2 * margin)
    lines = _wrap_text_to_width(prompt, max_width, font, font_scale, thickness)

    box_h = line_h * len(lines) + 2 * margin
    box_h = min(box_h, h)

    # Translucent black background strip at the top for readability.
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), thickness=-1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    y = margin + line_h - 4
    for line in lines:
        cv2.putText(
            out,
            line,
            (margin, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        y += line_h
        if y >= h - margin:
            break

    return out


def _compose_rollout_frame(
    model_input: dict[str, Any],
    prompt: str,
    action: np.ndarray | None,
    args: Args,
) -> np.ndarray:
    exo = model_input["observation/exterior_image_1_left"]
    wrist = model_input["observation/wrist_image_left"]
    separator = np.full((max(exo.shape[0], wrist.shape[0]), 4, 3), 255, dtype=np.uint8)
    frame = np.concatenate([exo, separator, wrist], axis=1)

    # Always include the action panel so all video frames have identical shape.
    action_for_overlay = (
        np.asarray(action, dtype=np.float32)
        if action is not None
        else np.zeros(8, dtype=np.float32)
    )
    overlay_h = max(90, frame.shape[0] // 4)
    frame = np.concatenate(
        [frame, _action_overlay(action_for_overlay, frame.shape[1], overlay_h)],
        axis=0,
    )

    return _overlay_prompt_on_frame(
        frame,
        prompt,
        font_scale=args.video_text_scale,
        thickness=args.video_text_thickness,
    )


def _model_action_to_env_action(
    model_action: np.ndarray,
    grasping_type: str,
    gripper_threshold: float,
    gripper_scale: float,
) -> dict[str, np.ndarray]:
    model_action = np.asarray(model_action, dtype=np.float32)

    if grasping_type == "continuous":
        gripper = np.asarray([model_action[7] * gripper_scale], dtype=np.float32)
    elif grasping_type == "binary":
        gripper = np.asarray(
            [gripper_scale if model_action[7] > gripper_threshold else 0.0],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"Unsupported grasping_type='{grasping_type}'. "
            "Use 'continuous' or 'binary'."
        )

    return {"arm": model_action[:7], "gripper": gripper}


def _resolve_checkpoint_path(args: Args, eval_config: Any) -> str:
    checkpoint_path = args.checkpoint_dir or getattr(
        eval_config.policy_config, "checkpoint_path", None
    )
    if not checkpoint_path:
        raise ValueError(
            "No checkpoint path configured. Set eval_config.policy_config.checkpoint_path "
            "or pass --checkpoint-dir."
        )

    if urlparse(checkpoint_path).scheme:
        return checkpoint_path

    return str(Path(checkpoint_path).expanduser())


def _load_local_openpi_policy(
    checkpoint_path: str,
    *,
    default_prompt: str | None,
) -> tuple[Any, Any]:
    config_name = os.path.basename(checkpoint_path.rstrip("/"))
    checkpoint_path = str(_openpi_download.maybe_download(checkpoint_path))

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    if config_name != "pi05_droid_jointpos":
        raise ValueError(
            f"Unsupported checkpoint config '{config_name}'. "
            "This demo currently defines an explicit TrainConfig only for "
            "'pi05_droid_jointpos'."
        )
    # https://github.com/omarrayyann/openpi/blob/711487f019e5f03b254d427d4523b1f0805a4814/src/openpi/training/config.py#L682-L698
    train_cfg = _openpi_config.TrainConfig(
        name="pi05_droid_jointpos",
        model=_pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=_openpi_config.SimpleDataConfig(
            assets=_openpi_config.AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _openpi_transforms.Group(
                inputs=[_droid_policy.DroidInputs(model_type=_openpi_config.ModelType.PI05)],
                outputs=[
                    _openpi_transforms.AbsoluteActions(
                        _openpi_transforms.make_bool_mask(7, -1)
                    ),
                    _droid_policy.DroidOutputs(),
                ],
            ),
            base_config=_openpi_config.DataConfig(prompt_from_task=True),
        ),
    )
    policy = _policy_config.create_trained_policy(
        train_cfg,
        checkpoint_path,
        default_prompt=default_prompt,
    )
    return policy, train_cfg


def _resolve_execute_horizon(args: Args, train_cfg: Any, eval_config: Any) -> int:
    model_horizon = int(getattr(train_cfg.model, "action_horizon", 0) or 0)
    config_chunk_size = int(getattr(eval_config.policy_config, "chunk_size", 0) or 0)
    requested = args.execute_horizon
    if requested is None and config_chunk_size > 0:
        requested = config_chunk_size

    if requested is None:
        if model_horizon <= 0:
            raise ValueError(
                "Could not infer execute_horizon from model config. "
                "Please pass --execute-horizon explicitly."
            )
        return model_horizon

    if requested <= 0:
        raise ValueError(f"--execute-horizon must be positive, got {requested}.")

    if model_horizon > 0 and requested > model_horizon:
        logging.warning(
            "Requested execute_horizon=%d exceeds model action_horizon=%d; using %d.",
            requested,
            model_horizon,
            model_horizon,
        )
        return model_horizon

    return requested


def run(args: Args) -> None:
    if args.episode_sampling not in {"sequential", "random"}:
        raise ValueError("--episode-sampling must be one of {'sequential', 'random'}")

    benchmark_path = Path(args.benchmark_path).expanduser().resolve()
    benchmark_episodes = load_all_episodes(benchmark_path)
    if not benchmark_episodes:
        raise ValueError(
            f"No benchmark episodes found in {benchmark_path}. "
            "Expected benchmark.json or house_*/episode_*.json files."
        )

    eval_config = _resolve_eval_config(args.eval_config_cls)
    checkpoint_path = _resolve_checkpoint_path(args, eval_config)
    policy, train_cfg = _load_local_openpi_policy(
        checkpoint_path,
        default_prompt=args.default_prompt,
    )
    execute_horizon = _resolve_execute_horizon(args, train_cfg, eval_config)
    grasping_type = args.grasping_type or eval_config.policy_config.grasping_type
    gripper_threshold = (
        args.gripper_threshold
        if args.gripper_threshold is not None
        else eval_config.policy_config.grasping_threshold
    )
    prompt_sampler = PromptSampler(
        task_type=eval_config.task_type,
        prompt_templates=eval_config.policy_config.prompt_templates,
        prompt_object_word_num=eval_config.policy_config.prompt_object_word_num,
    )

    env_cfg = MolmoSpacesGymConfig(
        benchmark_dir=str(benchmark_path),
        eval_config_cls=args.eval_config_cls,
        episode_sampling=args.episode_sampling,
        seed=args.seed,
        task_horizon_steps=args.task_horizon_steps,
    )
    env = MolmoSpacesBenchmarkGymEnv(env_cfg)
    num_episodes = args.num_episodes if args.num_episodes is not None else len(benchmark_episodes)
    policy_name = getattr(train_cfg, "name", os.path.basename(checkpoint_path))
    registered_policy = _RegisteredPolicyAdapter(policy, policy_name, prompt_sampler)
    env.register_policy(registered_policy)
    video_dir = Path(args.video_dir).expanduser()
    video_fps = 1000.0 / float(eval_config.policy_dt_ms)
    logging.info(
        "OpenPI config=%s model_type=%s action_horizon=%s execute_horizon=%d checkpoint=%s",
        policy_name,
        getattr(train_cfg.model, "model_type", "unknown"),
        getattr(train_cfg.model, "action_horizon", "unknown"),
        execute_horizon,
        checkpoint_path,
    )

    total_success = 0
    total_reward = 0.0
    total_steps = 0
    for episode_idx in range(num_episodes):
        obs, info = env.reset(seed=args.seed + episode_idx)
        action_buffer: collections.deque[np.ndarray] = collections.deque()
        episode_prompt = registered_policy.get_prompt(args.default_prompt)
        video_frames: list[np.ndarray] = []

        success = False
        episode_reward = 0.0
        episode_steps = 0
        for step_idx in range(args.max_steps):
            model_input = _shared_obs_to_openpi_input(
                obs,
                exo_camera_key=args.exo_camera_key,
                wrist_camera_key=args.wrist_camera_key,
                gripper_obs_norm=0.824033,
                prompt=registered_policy.get_prompt(args.default_prompt),
            )

            if not action_buffer:
                action_chunk = np.asarray(
                    policy.infer(model_input, sharding_spec=None)["actions"]
                )
                chunk_len = int(action_chunk.shape[0]) if action_chunk.ndim > 0 else 0
                if chunk_len != execute_horizon:
                    logging.warning(
                        (
                            "Policy returned action chunk length %d while "
                            "execute_horizon=%d; %s."
                        ),
                        chunk_len,
                        execute_horizon,
                        "truncating to execute_horizon"
                        if chunk_len > execute_horizon
                        else "buffer may run short before the next inference call",
                    )
                action_buffer.extend(action_chunk[:execute_horizon])

            raw_action = action_buffer.popleft()
            if args.save_trajectory_video:
                video_frames.append(
                    _compose_rollout_frame(
                        model_input,
                        episode_prompt,
                        action=raw_action,
                        args=args,
                    )
                )

            env_action = _model_action_to_env_action(
                raw_action,
                grasping_type=grasping_type,
                gripper_threshold=gripper_threshold,
                gripper_scale=args.gripper_scale,
            )
            obs, reward, terminated, truncated, info = env.step(env_action)
            episode_reward += float(reward)
            episode_steps = step_idx + 1

            if info["success"]:
                success = True
                break

            if terminated or truncated:
                break

        total_success += int(success)
        total_reward += episode_reward
        total_steps += episode_steps
        logging.info(
            (
                "Episode %d/%d finished: success=%s, steps=%d, "
                "accumulated_reward=%.4f, success_so_far=%d"
            ),
            episode_idx + 1,
            num_episodes,
            success,
            episode_steps,
            episode_reward,
            total_success,
        )

        if args.save_trajectory_video and video_frames:
            video_path = video_dir / f"episode_{episode_idx:04d}.mp4"
            save_frames_to_mp4(np.asarray(video_frames, dtype=np.uint8), str(video_path), fps=video_fps)
            logging.info("Saved trajectory video with prompt overlay: %s", video_path)

    num_episodes = max(num_episodes, 1)
    logging.info(
        (
            "Done. Success rate: %.3f, total_steps=%d, "
            "total_accumulated_reward=%.4f, avg_steps=%.2f, avg_reward=%.4f"
        ),
        total_success / num_episodes,
        total_steps,
        total_reward,
        total_steps / num_episodes,
        total_reward / num_episodes,
    )
    env.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
