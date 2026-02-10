from typing import Dict
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import logging
import numpy as np
import shutil
import tqdm_loggable.auto as tqdm
from src.envs.venv import SubprocVectorEnv


def process_obs_for_pi0(observations: Dict, config, task_description: str, obs_prefix_key: str = "pi0/"):
    element = {}
    prompt_in_obs = False
    for key, val in observations.items():
        # Extract all observations relevant for the policy
        if obs_prefix_key in key:
            obs_key = key.split(obs_prefix_key)[-1]
            if obs_key == "prompt":
                prompt_in_obs = True
                element[obs_key] = val
            else:
                if 'image' in obs_key and config.collect.resize_image > 0:
                    from openpi_client import image_tools
                    # Rescale images
                    val = image_tools.convert_to_uint8(image_tools.resize_with_pad(val,
                                                                                   config.collect.resize_image,
                                                                                   config.collect.resize_image))
                obs_key = f'observation/{obs_key}'
                element[obs_key] = val
    # If prompt is not stored in obs, we add the default prompt here.
    if not prompt_in_obs:
        element["prompt"] = task_description
    return element


def get_action_chunk_from_policy(policy, obs, sharding_spec, config, task_description: str):
    if config.collect.add_per_step_data:
        current_obs = jax.tree_util.tree_map(lambda x: x[:, -1], obs["observation"])
    else:
        current_obs = obs["observation"]
    element = process_obs_for_pi0(current_obs, obs_prefix_key="pi0/", config=config, task_description=task_description)

    action_chunk = policy.infer(element, sharding_spec=sharding_spec)["actions"]
    return action_chunk


def add_frames_to_dataset(episode_data, config, dataset, task_description: str, obs_prefix: str = "pi0/"):
    """Adding individual frames to the dataset since lerobot add_frame only allows 1 insertion."""
    # TODO: This is a bit hacky. Perhaps we can just add the full episode all at once?
    def process_frame(ob):
        frame = {}
        # Extract actions and observations from total_obs
        obs, action = ob["observation"], ob["action"]
        for key, val in obs.items():
            if obs_prefix in key:
                obs_key = key.split(obs_prefix)[-1]
                frame[obs_key] = val
        frame["actions"] = action
        return frame
    if config.collect.add_per_step_data:
        # Add all the per time-step transitions one by one.
        for ep in episode_data:
            for step in range(config.collect.replan_steps):
                obs = jax.tree.map(lambda x: x[step], ep)
                dataset.add_frame(process_frame(obs), task=str(task_description))
    else:
        for ep in episode_data:
            dataset.add_frame(process_frame(ep))


def collect_data(
        policy,
        dataset,
        sharding_spec,
        env: SubprocVectorEnv,
        task_description,
        config):
    num_envs = len(env)
    total_episodes, total_successes = 0, 0
    logging.info(f"Collecting for task: {task_description}")
    total_episodes = 0
    num_rollouts = config.collect.num_rollouts

    def get_env_obs(obs, env_id):
        return jax.tree.map(lambda x: x[env_id], obs)

    with tqdm.tqdm(total=num_rollouts) as pbar:
        obs, info = env.reset()
        frames = [[] for _ in range(num_envs)]
        while total_episodes < num_rollouts:
            # Get action chunk from the policy
            action_chunk = get_action_chunk_from_policy(policy,
                                                        obs,
                                                        sharding_spec,
                                                        config,
                                                        task_description=task_description)
            # Apply full action chunk to the policy.
            # If config.collect.add_per_step_data is True, next_obs consists of all the transitions
            # obtained during the full action_chunk
            next_obs, _, terminate, truncate, _ = env.step(action_chunk)
            # Add data from each environment to its respective frame
            [frames[i].append(get_env_obs(next_obs, i)) for i in range(num_envs)]
            # Extract terminate or truncation flags
            if config.collect.add_per_step_data:
                assert terminate.shape[1] == config.collect.replan_steps
                current_terminate = jax.tree_util.tree_map(lambda x: x[:, -1], terminate)
                current_truncate = jax.tree_util.tree_map(lambda x: x[:, -1], truncate)
            else:
                current_terminate, current_truncate = terminate, truncate
            # Take the last step for terminate/truncation flag
            done = np.logical_or(current_terminate, current_truncate)
            # Check which environment is done
            done_indices = np.where(done)[0]
            total_episodes += len(done_indices)
            if len(done_indices) > 0:
                pbar.update(len(done_indices))
            # For the environments that are done, check if they terminated successfuly.
            for env_index in done_indices:
                # if the environment was done due to the success state being reached,
                # the agent may use this information for filtering data.
                success = current_terminate[env_index]
                if success:
                    episode_data = frames[env_index]
                    add_frames_to_dataset(episode_data, config, dataset, task_description=task_description)
                    dataset.save_episode()
                total_successes += success
                # Reset the environment
                reset_out = env.reset(id=env_index)
                assert isinstance(reset_out, (tuple, list))
                assert len(reset_out) == 2
                env_obs = reset_out[0]

                def update_state(prev_state, new_val_leaf):
                    prev_state[env_index] = new_val_leaf[0]
                    return prev_state
                next_obs = jax.tree.map(update_state, next_obs, env_obs)

                # Empty the episode buffer for this environment
                frames[env_index] = []
                pbar.set_postfix(SR=total_successes / total_episodes)
            obs = next_obs

    metrics = {"success_rate": float(total_successes) / float(total_episodes)}
    env.close()
    return metrics, dataset.num_episodes


def collect_data_lerobot_libero(
                 config,
                 checkpoint_path,
                 data_path):
    # load latest policy
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    from openpi.policies import policy_config
    from src.envs.libero import make_env_libero
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
    env, task_description = make_env_libero(config.collect)
    return collect_data(
        policy=policy,
        dataset=collected_dataset,
        sharding_spec=sharding_spec,
        env=env,
        task_description=task_description,
        config=config,
    )
