import os
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"


import collections
import logging
import os
import numpy as np

import openpi.models.pi0_config as _pi0_config
import openpi.policies.droid_policy as _droid_policy
import openpi.transforms as _openpi_transforms
from openpi.training import config as _openpi_config
from src.envs.molmo import MolmoSpacesBenchmarkGymEnv
from openpi.policies import policy_config as _policy_config
from openpi.shared import download as _openpi_download


def run() -> None:

    # create policy
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
    checkpoint_path = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
    checkpoint_path = str(_openpi_download.maybe_download(checkpoint_path))
    policy = _policy_config.create_trained_policy(train_cfg, checkpoint_path)

    # create env
    env = MolmoSpacesBenchmarkGymEnv()

    # logging
    logging.info(
        "OpenPI config=%s model_type=%s action_horizon=%s checkpoint=%s",
        os.path.basename(checkpoint_path),
        getattr(train_cfg.model, "model_type", "unknown"),
        getattr(train_cfg.model, "action_horizon", "unknown"),
        checkpoint_path,
    )

    # prompt_sampler
    from molmo_spaces.policy.learned_policy.utils import PromptSampler
    eval_config = env._make_eval_config()
    prompt_sampler = PromptSampler(
        task_type=eval_config.task_type,
        prompt_templates=eval_config.policy_config.prompt_templates,
        prompt_object_word_num=eval_config.policy_config.prompt_object_word_num,
    )

    # rollout
    total_success, total_reward, total_steps = 0, 0.0, 0
    # hardcode 6 episodes in order to see some successes
    for episode_idx in range(6):
        obs, info = env.reset(seed=0 + episode_idx)
        policy.reset()
        prompt_sampler.next()
        action_buffer: collections.deque[np.ndarray] = collections.deque()
        if episode_idx < 4:
            # hardcode 4 episodes to skip in order to see some successes
            continue

        success, episode_reward, episode_steps = False, 0.0, 0
        for step_idx in range(450):

            gripper_obs_norm: float = 0.824033
            qpos = obs["qpos"]
            gripper = np.asarray(qpos["gripper"], dtype=np.float32)
            grip = np.clip(float(gripper[0]) / gripper_obs_norm, 0.0, 1.0)

            model_input = {
                "observation/exterior_image_1_left": obs["exo_camera_1"],
                "observation/wrist_image_left": obs["wrist_camera"],
                "observation/joint_position": np.asarray(qpos["arm"][:7], dtype=np.float32),
                "observation/gripper_position": np.asarray([grip], dtype=np.float32),
                "prompt": prompt_sampler.get_prompt(env._task).lower(),
            }
            breakpoint()

            if not action_buffer:
                action_chunk = np.asarray(policy.infer(model_input, sharding_spec=None)["actions"])
                action_buffer.extend(action_chunk[:8])

            raw_action = np.asarray(action_buffer.popleft(), dtype=np.float32)
            env_action = {"arm": raw_action[:7], "gripper": np.asarray([255. if raw_action[7] > 0.5 else 0.0], dtype=np.float32)}

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
                "Episode %d finished: success=%s, steps=%d, "
                "accumulated_reward=%.4f, success_so_far=%d"
            ),
            episode_idx + 1,
            success,
            episode_steps,
            episode_reward,
            total_success,
        )

    env.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    run()


if __name__ == "__main__":
    main()
