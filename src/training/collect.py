
import collections
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import logging
import numpy as np
import shutil
import tqdm_loggable.auto as tqdm

import lerobot.datasets.lerobot_dataset as lerobot_dataset
from openpi.policies import policy_config

from src.envs.libero import make_env_libero, init_state_libero, get_action_chunk_libero, get_frame_libero, get_max_steps_libero


def collect_data(config, checkpoint_path, data_path):

    # load latest policy
    sharding_spec = NamedSharding(Mesh(mesh_utils.create_device_mesh((len(jax.devices()),)), axis_names=('batch',)), PartitionSpec('batch',))
    policy = policy_config.create_trained_policy(config, checkpoint_path)

    # prepare dataset to store collected data
    if data_path.exists():
        shutil.rmtree(data_path)
    allowed_keys = {"image", "wrist_image", "state", "actions"}
    collected_dataset = lerobot_dataset.LeRobotDataset.create(
        repo_id=config.data.repo_id,
        root=data_path,
        robot_type="panda",
        fps=10,
        features={k: v for k, v in lerobot_dataset.LeRobotDatasetMetadata(config.data.repo_id).features.items() if k in allowed_keys},
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # rollout parallel environments
    env, initial_states, task_description = make_env_libero(config.collect)
    total_episodes, total_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(config.collect.num_rollouts // config.collect.env_num + 1)):

        logging.info(f"\nTask: {task_description}")
        action_plan = collections.deque()
        obs = init_state_libero(env, initial_states, episode_idx, config)
        frames, solved = [], [np.array([False] * config.collect.env_num)]
        max_steps = get_max_steps_libero(config.collect.tasks[0])
        for _ in range(max_steps):

            if not action_plan:
                action_chunk = get_action_chunk_libero(obs, task_description, policy, config, sharding_spec)
                action_plan.extend([action_chunk[:, i, :] for i in range(config.collect.replan_steps)])
            action = action_plan.popleft()

            frames.append(get_frame_libero(obs, action, task_description))
            obs, _, done, _ = env.step(action.tolist())

            solved.append(np.logical_or(solved[-1], done))
            if np.all(solved[-1]):
                # break if all environments are solved
                break

        for i in range(config.collect.env_num):
            total_episodes += 1
            if not solved[-1][i]:
                # skip unsuccessful episodes
                continue
            for f, s in zip(frames, solved[1:]):
                # save successful ones until success
                f_single = {k: v[i] for k, v in f.items()}
                collected_dataset.add_frame(f_single)
                if s[i]:
                    break
            collected_dataset.save_episode()
            total_successes += 1

        logging.info(f"# Successes: {total_successes}/{total_episodes} ({total_successes / total_episodes * 100:.1f}%)")

    metrics = {"success_rate": float(total_successes) / float(total_episodes)}

    env.close()
    del policy
    return metrics, collected_dataset.num_episodes
