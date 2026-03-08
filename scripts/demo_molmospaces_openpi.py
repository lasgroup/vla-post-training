import collections
import dataclasses
import importlib
import logging
from typing import Any, Literal

import numpy as np
import tyro

from molmo_spaces.policy.learned_policy.utils import PromptSampler
from src.envs.molmo import MolmoSpacesBenchmarkGymEnv
from src.envs.molmo import MolmoSpacesGymConfig
from openpi.policies import policy_config as _policy_config
import src.training.config as _config


@dataclasses.dataclass
class Args:
    # OpenPI policy selection.
    config_name: str = "pi05_droid"
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_droid"
    default_prompt: str | None = None

    # MolmoSpaces benchmark/env setup.
    benchmark_dir: str = "molmospaces/assets/benchmarks/path-to-benchmark"
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


def _obs_to_openpi_input(
    obs: dict[str, Any],
    args: Args,
    registered_policy: _RegisteredPolicyAdapter,
) -> dict[str, Any]:
    qpos = _get_qpos(obs)
    if "arm" not in qpos or "gripper" not in qpos:
        raise KeyError(f"Expected qpos to contain 'arm' and 'gripper'. Got: {list(qpos.keys())}")

    # Match PI_Policy camera key selection semantics exactly.
    exo_camera_key = (
        "droid_shoulder_light_randomization"
        if "droid_shoulder_light_randomization" in obs
        else args.exo_camera_key
    )
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
        benchmark_dir=args.benchmark_dir,
        eval_config_cls=args.eval_config_cls,
        episode_sampling=args.episode_sampling,
        seed=args.seed,
        task_horizon_steps=args.task_horizon_steps,
    )
    env = MolmoSpacesBenchmarkGymEnv(env_cfg)
    registered_policy = _RegisteredPolicyAdapter(policy, args.config_name, prompt_sampler)
    env.register_policy(registered_policy)

    try:
        total_success = 0
        total_reward = 0.0
        total_steps = 0
        for episode_idx in range(args.num_episodes):
            obs, info = env.reset(seed=args.seed + episode_idx)
            action_buffer: collections.deque[np.ndarray] = collections.deque()

            success = False
            episode_reward = 0.0
            episode_steps = 0
            for step_idx in range(args.max_steps):
                model_input = _obs_to_openpi_input(obs, args, registered_policy)

                if not action_buffer:
                    action_chunk = np.asarray(policy.infer(model_input)["actions"])
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk[None, :]
                    action_buffer.extend(action_chunk[: args.execute_horizon])

                env_action = _model_action_to_env_action(
                    action_buffer.popleft(),
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
