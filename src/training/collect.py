from typing import Dict
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import logging
import numpy as np
import shutil
import tqdm_loggable.auto as tqdm
from src.envs.venv import SubprocVectorEnv
from openpi_client import image_tools


def process_obs_for_pi0(observations: Dict, config, task_descriptions: list[str], obs_prefix_key: str = "pi0/"):
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
                    # Rescale images
                    val = image_tools.convert_to_uint8(image_tools.resize_with_pad(val,
                                                                                   config.collect.resize_image,
                                                                                   config.collect.resize_image))
                obs_key = f'observation/{obs_key}'
                element[obs_key] = val
    # If prompt is not stored in obs, we add the default prompt here.
    if not prompt_in_obs:
        element["prompt"] = np.asarray(task_descriptions) # TODO, WORKS ONLY WITH SINGLE TASK
    return element


def get_action_chunk_from_policy(policy, obs, sharding_spec, config, task_descriptions: list[str]):
    if config.collect.add_per_step_data:
        current_obs = jax.tree_util.tree_map(lambda x: x[:, -1], obs["observation"])
    else:
        current_obs = obs["observation"]
    element = process_obs_for_pi0(current_obs, obs_prefix_key="pi0/", config=config, task_descriptions=task_descriptions)
    action_chunk = policy.infer(element, sharding_spec=sharding_spec)["actions"]
    return action_chunk


def add_episode_to_dataset(episode_data, config, dataset, task_description: str, obs_prefix: str = "pi0/"):
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
        total_frames = len(episode_data)
        for n_frame, ep in enumerate(episode_data):
            ep_obs, terminate, truncate = ep["observation"],  ep["terminate"], ep["truncate"]
            total_chunks = config.collect.replan_steps
            # For the last frame where termination occurred check at which step this was observed.
            if n_frame == total_frames - 1:
                done = np.logical_or(terminate, truncate)
                done_indices = np.where(done)[0]
                if len(done_indices) > 0:
                    total_chunks = done_indices[0]
            for step in range(total_chunks):
                obs = jax.tree.map(lambda x: x[step], ep_obs)
                dataset.add_frame(process_frame(obs), task=str(task_description))
    else:
        for ep in episode_data:
            dataset.add_frame(process_frame(ep["observation"]))
    dataset.save_episode()


def shift_window(obs_act, next_obs_act):
    def move_obs(curr, nxt):
        # curr shape: (E, H, D) -> Take last step: (E, 1, D)
        last_step = curr[:, -1:]
        # nxt shape: (E, H, D) -> Take all but last step: (E, H-1, D)
        next_steps = nxt[:, :-1]
        # Concatenate along the horizon axis (axis 1)
        # Result shape: (E, H, D) containing steps H to 2H-1
        return np.concatenate([last_step, next_steps], axis=1)
    target_obs = jax.tree.map(move_obs, obs_act["observation"], next_obs_act["observation"])
    target_obs_act = {key: (target_obs if key == "observation" else val) for key, val in next_obs_act.items()}
    return target_obs_act


def collect_data(
        policy,
        dataset,
        sharding_spec,
        env: SubprocVectorEnv,
        task_descriptions: list[str],
        config):
    num_envs = len(env)
    total_episodes, total_successes = 0, 0
    episodes_per_env = [0 for _ in range(num_envs)]
    successes_per_env = [0 for _ in range(num_envs)]
    unique_tasks = set(task_descriptions)
    logging.info(f"Collecting for tasks: {unique_tasks}")
    total_episodes = 0
    num_rollouts = config.collect.num_rollouts

    def get_env_value(vec, env_id):
        return jax.tree.map(lambda x: x[env_id], vec)

    with tqdm.tqdm(total=num_rollouts) as pbar:
        obs, _ = env.reset()
        frames = [[] for _ in range(num_envs)]
        while total_episodes < num_rollouts:
            # Get action chunk from the policy
            action_chunk = get_action_chunk_from_policy(policy,
                                                        obs,
                                                        sharding_spec,
                                                        config,
                                                        task_descriptions=task_descriptions, 
                                                        )
            # Apply full action chunk to the policy.
            # If config.collect.add_per_step_data is True, next_obs consists of all the transitions
            # obtained during the full action_chunk
            next_obs, _, terminate, truncate, _ = env.step(action_chunk)
            # Add data from each environment to its respective frame
            # Apply the function to the PyTrees
            target_obs = shift_window(obs_act=obs, next_obs_act=next_obs)
            [frames[i].append(
                {
                    'observation': get_env_value(target_obs, i),
                    'terminate': get_env_value(terminate, i),
                    'truncate': get_env_value(truncate, i)
                }) for i in range(num_envs)]
            # Extract terminate or truncation flags
            if config.collect.add_per_step_data:
                assert terminate.shape[1] == config.collect.replan_steps
                terminate = terminate[:, -1]
                truncate = truncate[:, -1]
            # Take the last step for terminate/truncation flag
            done = np.logical_or(terminate, truncate)
            # Check which environment is done
            done_indices = np.where(done)[0]
            total_episodes += len(done_indices)
            if len(done_indices) > 0:
                pbar.update(len(done_indices))
            # For the environments that are done, check if they terminated successfuly.
            for env_index in done_indices:
                # if the environment was done due to the success state being reached,
                # the agent may use this information for filtering data.
                success = terminate[env_index]
                episode_data = frames[env_index]
                if success:
                    add_episode_to_dataset(episode_data, config, dataset, task_description=task_descriptions[env_index]) 
                total_successes += success
                episodes_per_env[env_index] += 1
                successes_per_env[env_index] += int(success)
                # Reset the environment
                env_obs, _ = env.reset(id=env_index)

                def update_state(prev_state, new_val_leaf):
                    prev_state[env_index] = new_val_leaf[0]
                    return prev_state
                next_obs = jax.tree.map(update_state, next_obs, env_obs)

                # Empty the episode buffer for this environment
                frames[env_index] = []
                pbar.set_postfix(SR=total_successes / total_episodes)
            obs = next_obs

    metrics = {"success_rate": float(total_successes) / float(total_episodes)}
    per_env_success_rates = [s / e if e > 0 else 0.0 for s, e in zip(successes_per_env, episodes_per_env)]
    for i in range(num_envs):
        metrics[f"success_rate_{config.collect.tasks[i]}"] = per_env_success_rates[i]
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
    env, task_descriptions = make_env_libero(config)
    return collect_data(
        policy=policy,
        dataset=collected_dataset,
        sharding_spec=sharding_spec,
        env=env,
        task_descriptions=task_descriptions,
        config=config,
    )
