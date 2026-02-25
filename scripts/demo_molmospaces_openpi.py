import collections
import dataclasses
import logging
import pathlib
from typing import Any, Literal

import imageio.v2 as imageio
import numpy as np
import tyro

from src.molmo.molmospaces_gym_env import MolmoSpacesBenchmarkGymEnv
from src.molmo.molmospaces_gym_env import MolmoSpacesGymConfig
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    # OpenPI policy selection.
    config_name: str = "pi05_droid"
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_droid"
    default_prompt: str | None = None

    # MolmoSpaces benchmark/env setup.
    benchmark_dir: str = "third_party/molmospaces/assets/benchmarks/path-to-benchmark"
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
    grasping_type: str = "continuous"  # one of {"continuous", "binary"}
    gripper_threshold: float = 0.5
    gripper_scale: float = 255.0

    # Rollout control.
    num_episodes: int = 1
    max_steps: int = 300

    # Video recording.
    record_video: bool = False
    video_dir: str = "data/molmospaces_videos"
    video_fps: int = 10


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


def _get_prompt(info: dict[str, Any], default_prompt: str | None) -> str:
    for key in ("task_description", "prompt", "instruction", "language"):
        if info.get(key):
            return str(info[key]).lower()
    return (default_prompt or "do the task").lower()


def _obs_to_openpi_input(
    obs: dict[str, Any],
    info: dict[str, Any],
    args: Args,
) -> dict[str, Any]:
    qpos = _get_qpos(obs)
    if "arm" not in qpos or "gripper" not in qpos:
        raise KeyError(f"Expected qpos to contain 'arm' and 'gripper'. Got: {list(qpos.keys())}")

    exo = _get_camera(obs, args.exo_camera_key, ("droid_shoulder_light_randomization",))
    wrist = _get_camera(obs, args.wrist_camera_key, ("wrist_camera_zed_mini",))

    # PI policy path uses normalized gripper input for droid-style OpenPI configs.
    gripper = np.clip(np.asarray(qpos["gripper"], dtype=np.float32)[0] / 0.824033, 0.0, 1.0)
    return {
        "observation/exterior_image_1_left": exo,
        "observation/wrist_image_left": wrist,
        "observation/joint_position": np.asarray(qpos["arm"][:7], dtype=np.float32),
        "observation/gripper_position": np.asarray([gripper], dtype=np.float32),
        "prompt": _get_prompt(info, args.default_prompt),
    }


def _model_action_to_env_action(model_action: np.ndarray, args: Args) -> dict[str, np.ndarray]:
    model_action = np.asarray(model_action, dtype=np.float32)
    if model_action.shape[0] < 8:
        raise ValueError(
            "Expected at least 8 action dims from OpenPI policy, got shape "
            f"{model_action.shape}."
        )

    arm = model_action[:7]
    if args.grasping_type == "continuous":
        gripper = np.asarray([model_action[7] * args.gripper_scale], dtype=np.float32)
    elif args.grasping_type == "binary":
        gripper = np.asarray(
            [args.gripper_scale if model_action[7] > args.gripper_threshold else 0.0],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"Unsupported grasping_type='{args.grasping_type}'. "
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

    env_cfg = MolmoSpacesGymConfig(
        benchmark_dir=args.benchmark_dir,
        eval_config_cls=args.eval_config_cls,
        episode_sampling=args.episode_sampling,
        seed=args.seed,
        task_horizon_steps=args.task_horizon_steps,
    )
    env = MolmoSpacesBenchmarkGymEnv(env_cfg)
    video_dir = pathlib.Path(args.video_dir)
    if args.record_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    try:
        total_success = 0
        total_reward = 0.0
        total_steps = 0
        for episode_idx in range(args.num_episodes):
            obs, info = env.reset(seed=args.seed + episode_idx)
            action_buffer: collections.deque[np.ndarray] = collections.deque()
            video_frames: list[np.ndarray] = []

            success = False
            episode_reward = 0.0
            episode_steps = 0
            if args.record_video:
                video_frames.append(
                    _get_camera(obs, args.exo_camera_key, ("droid_shoulder_light_randomization",))
                )
            for step_idx in range(args.max_steps):
                model_input = _obs_to_openpi_input(obs, info, args)

                if not action_buffer:
                    action_chunk = np.asarray(policy.infer(model_input)["actions"])
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk[None, :]
                    action_buffer.extend(action_chunk[: args.execute_horizon])

                env_action = _model_action_to_env_action(action_buffer.popleft(), args)
                obs, reward, terminated, truncated, info = env.step(env_action)
                episode_reward += float(reward)
                episode_steps = step_idx + 1
                if args.record_video:
                    video_frames.append(
                        _get_camera(
                            obs,
                            args.exo_camera_key,
                            ("droid_shoulder_light_randomization",),
                        )
                    )

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
            if args.record_video and video_frames:
                status = "success" if success else "failure"
                video_path = video_dir / f"episode_{episode_idx:04d}_{status}.mp4"
                imageio.mimwrite(
                    video_path,
                    video_frames,
                    fps=args.video_fps,
                )
                logging.info("Saved video: %s", video_path)

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
