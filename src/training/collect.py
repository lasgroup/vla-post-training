from typing import Any
import jax
import numpy as np
from src.envs.venv import BaseVectorEnv
from src.rl.agent import Agent
import tqdm_loggable.auto as tqdm


def _shift_window(
    observation: dict[str, Any], next_observation: dict[str, Any]
) -> dict[str, Any]:
    def move_obs(curr, nxt):
        # curr shape: (E, H, D) -> keep last step: (E, 1, D)
        last_step = curr[:, -1:]
        # nxt shape: (E, H, D) -> keep first H-1 steps: (E, H-1, D)
        next_steps = nxt[:, :-1]
        # Result shape: (E, H, D), aligned with action chunk rollout.
        return np.concatenate([last_step, next_steps], axis=1)

    return jax.tree.map(move_obs, observation, next_observation)


def evaluate_policy(
    agent: Agent, env: BaseVectorEnv, task_description: str, config, step: int
):
    num_rollouts = config.collect.num_eval_rollouts
    total_episodes = 0
    total_successes = 0
    tasks = set(task_description)
    episodes_per_task = {k: 0 for k in tasks}
    successes_per_task = {k: 0 for k in tasks}
    env_step_counts = np.zeros(env.env_num, dtype=np.int32)
    successful_episode_lengths = []

    with tqdm.tqdm(total=num_rollouts, desc="eval") as pbar:
        obs, _ = env.reset()

        while total_episodes < num_rollouts:
            action_chunk = agent.sample_actions(
                obs,
                task_description=task_description,
            )
            next_obs, _, terminate, truncate, _ = env.step(action_chunk)

            done_per_step = np.logical_or(terminate, truncate)
            any_done = done_per_step[:, -1]  # True if episode ended this chunk
            first_done_idx = np.argmax(done_per_step, axis=1)  # first True index
            steps_this_chunk = np.where(any_done, first_done_idx + 1, config.collect.replan_steps)
            env_step_counts += steps_this_chunk

            current_terminate = terminate[:, -1]
            current_truncate = truncate[:, -1]

            done = np.logical_or(current_terminate, current_truncate)
            done_indices = np.where(done)[0]

            if len(done_indices) > 0:
                total_episodes += len(done_indices)
                pbar.update(len(done_indices))

            for env_index in done_indices:
                success = bool(current_terminate[env_index])
                total_successes += int(success)
                successes_per_task[task_description[env_index]] += int(success)
                episodes_per_task[task_description[env_index]] += 1
                if success:
                    successful_episode_lengths.append(int(env_step_counts[env_index]))
                env_step_counts[env_index] = 0

                reset_out = env.reset(id=int(env_index))
                assert isinstance(reset_out, (tuple, list)) and len(reset_out) == 2
                env_obs = reset_out[0]

                def update_state(prev_state, new_val_leaf):
                    prev_state[env_index] = new_val_leaf[0]
                    return prev_state

                next_obs = jax.tree.map(update_state, next_obs, env_obs)

            if total_episodes > 0:
                pbar.set_postfix(SR=total_successes / total_episodes)

            obs = next_obs

    metrics = {"eval/success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0}
    if successful_episode_lengths:
        metrics["eval/mean_success_episode_length"] = np.mean(successful_episode_lengths)
    metrics["eval/total_collected_episodes"] = agent.total_collected_episodes
    metrics.update({f"eval/success_rate/{task}": successes_per_task[task] / episodes_per_task[task] if episodes_per_task[task] > 0 else 0.0 for task in tasks})
    return metrics


def collect_data(
    agent: Agent, env: BaseVectorEnv, task_description: list[str], config, step: int
):
    agent.start_data_collection(step=step)

    total_episodes = 0
    total_successes = 0
    tasks = set(task_description)
    episodes_per_task = {k: 0 for k in tasks}
    successes_per_task = {k: 0 for k in tasks}
    
    if step == 0 and config.collect.num_initial_rollouts is not None:
        num_rollouts = config.collect.num_initial_rollouts
    else:
        num_rollouts = config.collect.num_rollouts

    with tqdm.tqdm(total=num_rollouts) as pbar:
        obs, _ = env.reset()

        while total_episodes < num_rollouts:
            action_chunk = agent.sample_actions(
                obs,
                task_description=task_description,
            )
            next_obs, reward, terminate, truncate, _ = env.step(action_chunk)
            aligned_obs = _shift_window(observation=obs, next_observation=next_obs)

            step_data = {
                "observation": aligned_obs,
                "next_observation": next_obs,
                "action": action_chunk[:, :config.collect.replan_steps],
                "reward": reward,
                "terminate": terminate,
                "truncate": truncate,
            }
            agent.add_data(step_data)

            current_terminate = terminate[:, -1]
            current_truncate = truncate[:, -1]

            done = np.logical_or(current_terminate, current_truncate)
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                total_episodes += len(done_indices)
                pbar.update(len(done_indices))

            for env_index in done_indices:
                success = bool(current_terminate[env_index])
                total_successes += int(success)
                successes_per_task[task_description[env_index]] += int(success)
                episodes_per_task[task_description[env_index]] += 1

                agent.save_episode(
                    is_success=success,
                    env_index=int(env_index),
                    task_description=task_description[env_index],
                )

                reset_out = env.reset(id=int(env_index))
                assert isinstance(reset_out, (tuple, list)) and len(reset_out) == 2
                env_obs = reset_out[0]

                def update_state(prev_state, new_val_leaf):
                    prev_state[env_index] = new_val_leaf[0]
                    return prev_state

                next_obs = jax.tree.map(update_state, next_obs, env_obs)

            if total_episodes > 0:
                pbar.set_postfix(SR=total_successes / total_episodes)

            obs = next_obs

    collected_episodes = agent.end_data_collection(step=step)

    agent.total_collected_episodes += total_episodes
    metrics = {"success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0}
    metrics.update({f"success_rate/{task}": successes_per_task[task] / episodes_per_task[task] if episodes_per_task[task] > 0 else 0.0 for task in tasks})
    return metrics, collected_episodes
