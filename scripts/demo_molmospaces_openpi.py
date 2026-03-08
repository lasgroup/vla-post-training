import collections
import dataclasses
import importlib
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro

from molmo_spaces.policy.learned_policy.utils import PromptSampler
from molmo_spaces.utils.save_utils import save_frames_to_mp4
from src.envs.molmo import MolmoSpacesBenchmarkGymEnv
from src.envs.molmo import MolmoSpacesGymConfig
from openpi.policies import policy_config as _policy_config
import src.training.config as _config

try:
    import cv2
except ImportError:
    cv2 = None


@dataclasses.dataclass
class Args:
    # OpenPI policy selection.
    config_name: str = "pi05_droid_jointpos"
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
    default_prompt: str | None = None

    # MolmoSpaces benchmark/env setup.
    benchmark_path: str = (
        "/capstor/store/cscs/swissai/a143/yardas/molmo-assets/benchmarks/"
        "molmospaces-bench-v1/ithor/FrankaPickHardBench/"
        "FrankaPickHardBench_20260206_json_benchmark"
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
    execute_horizon: int = 8
    grasping_type: str | None = None  # one of {"continuous", "binary"}; defaults to eval config
    gripper_threshold: float | None = None  # defaults to eval config
    gripper_scale: float = 255.0

    # Rollout control.
    num_episodes: int = 1
    max_steps: int = 300

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


def _as_uint8_hwc(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0 if image.max() <= 1.0 else image
    return np.clip(image, 0, 255).astype(np.uint8)


def _get_camera(obs: dict[str, Any], primary: str, fallback: tuple[str, ...]) -> np.ndarray:
    if primary in obs:
        return _as_uint8_hwc(obs[primary])
    for key in fallback:
        if key in obs:
            return _as_uint8_hwc(obs[key])
    raise KeyError(f"Missing camera key '{primary}'. Available keys: {list(obs.keys())}")


def _get_qpos(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    if "qpos" in obs:
        return obs["qpos"]
    if "robot_state" in obs and "qpos" in obs["robot_state"]:
        return obs["robot_state"]["qpos"]
    raise KeyError("Could not find qpos in observation. Expected 'qpos' or 'robot_state/qpos'.")


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


def _resolve_exo_camera_key(obs: dict[str, Any], args: Args) -> str:
    return (
        "droid_shoulder_light_randomization"
        if "droid_shoulder_light_randomization" in obs
        else args.exo_camera_key
    )


def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h == target_h and src_w == target_w:
        return image
    y_idx = np.linspace(0, src_h - 1, num=target_h, dtype=np.int32)
    x_idx = np.linspace(0, src_w - 1, num=target_w, dtype=np.int32)
    return image[y_idx][:, x_idx]


def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h == target_h:
        return image
    target_w = max(1, int(round((target_h / float(src_h)) * src_w)))
    return _resize_nearest(image, target_h, target_w)


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
    exo = _as_uint8_hwc(model_input["observation/exterior_image_1_left"])
    wrist = _as_uint8_hwc(model_input["observation/wrist_image_left"])

    target_h = max(exo.shape[0], wrist.shape[0])
    exo = _resize_to_height(exo, target_h)
    wrist = _resize_to_height(wrist, target_h)
    separator = np.full((target_h, 4, 3), 255, dtype=np.uint8)
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


def _obs_to_openpi_input(
    obs: dict[str, Any],
    args: Args,
    registered_policy: _RegisteredPolicyAdapter,
) -> dict[str, Any]:
    qpos = _get_qpos(obs)
    if "arm" not in qpos or "gripper" not in qpos:
        raise KeyError(f"Expected qpos to contain 'arm' and 'gripper'. Got: {list(qpos.keys())}")

    # Match PI_Policy camera key selection semantics exactly.
    exo_camera_key = _resolve_exo_camera_key(obs, args)
    wrist_camera_key = "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else args.wrist_camera_key
    exo = _get_camera(obs, exo_camera_key, ())
    wrist = _get_camera(obs, wrist_camera_key, ())

    # PI policy path uses normalized gripper input for droid-style OpenPI configs.
    gripper = np.clip(np.asarray(qpos["gripper"], dtype=np.float32)[0] / 0.824033, 0.0, 1.0)
    return {
        "observation/exterior_image_1_left": exo,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": np.asarray(qpos["arm"][:7], dtype=np.float32),
        "observation/gripper_position": np.asarray([gripper], dtype=np.float32),
        "prompt": registered_policy.get_prompt(args.default_prompt),
    }


def _model_action_to_env_action(
    model_action: np.ndarray,
    grasping_type: str,
    gripper_threshold: float,
    gripper_scale: float,
) -> dict[str, np.ndarray]:
    model_action = np.asarray(model_action, dtype=np.float32)
    if model_action.shape[0] < 8:
        raise ValueError(
            "Expected at least 8 action dims from OpenPI policy, got shape "
            f"{model_action.shape}."
        )

    arm = model_action[:7]
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

    return {"arm": arm, "gripper": gripper}


def run(args: Args) -> None:
    if args.episode_sampling not in {"sequential", "random"}:
        raise ValueError("--episode-sampling must be one of {'sequential', 'random'}")

    train_cfg = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    eval_config = _resolve_eval_config(args.eval_config_cls)
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
        benchmark_dir=args.benchmark_path,
        eval_config_cls=args.eval_config_cls,
        episode_sampling=args.episode_sampling,
        seed=args.seed,
        task_horizon_steps=args.task_horizon_steps,
    )
    env = MolmoSpacesBenchmarkGymEnv(env_cfg)
    registered_policy = _RegisteredPolicyAdapter(policy, args.config_name, prompt_sampler)
    env.register_policy(registered_policy)
    video_dir = Path(args.video_dir).expanduser()
    video_fps = 1000.0 / float(eval_config.policy_dt_ms)

    try:
        total_success = 0
        total_reward = 0.0
        total_steps = 0
        for episode_idx in range(args.num_episodes):
            obs, info = env.reset(seed=args.seed + episode_idx)
            action_buffer: collections.deque[np.ndarray] = collections.deque()
            episode_prompt = registered_policy.get_prompt(args.default_prompt)
            video_frames: list[np.ndarray] = []

            if args.save_trajectory_video:
                try:
                    initial_model_input = _obs_to_openpi_input(obs, args, registered_policy)
                    video_frames.append(
                        _compose_rollout_frame(
                            initial_model_input,
                            episode_prompt,
                            action=None,
                            args=args,
                        )
                    )
                except KeyError as exc:
                    logging.warning("Video frame capture skipped at reset: %s", exc)

            success = False
            episode_reward = 0.0
            episode_steps = 0
            for step_idx in range(args.max_steps):
                model_input = _obs_to_openpi_input(obs, args, registered_policy)

                if not action_buffer:
                    action_chunk = np.asarray(
                        policy.infer(model_input, sharding_spec=None)["actions"]
                    )
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk[None, :]
                    action_buffer.extend(action_chunk[: args.execute_horizon])

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

                if info.get("success") is True:
                    success = True

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
                args.num_episodes,
                success,
                episode_steps,
                episode_reward,
                total_success,
            )

            if args.save_trajectory_video and video_frames:
                video_path = video_dir / f"episode_{episode_idx:04d}.mp4"
                save_frames_to_mp4(np.asarray(video_frames, dtype=np.uint8), str(video_path), fps=video_fps)
                logging.info("Saved trajectory video with prompt overlay: %s", video_path)

        num_episodes = max(args.num_episodes, 1)
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
    finally:
        env.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
