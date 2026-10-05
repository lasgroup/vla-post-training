import logging
from typing import Any, Callable
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
    agent: Agent, env: BaseVectorEnv, config, step: int
):
    num_rollouts_per_task = config.collect.num_eval_rollouts
    total_episodes = 0
    total_successes = 0
    tasks = set(config.collect.eval_tasks)
    episodes_per_task = {k: 0 for k in tasks}
    successes_per_task = {k: 0 for k in tasks}
    env_step_counts = np.zeros(env.env_num, dtype=np.int32)
    successful_episode_lengths = []

    repeated_task_ids = []
    for t in config.collect.eval_tasks:
        repeated_task_ids.extend([t] * num_rollouts_per_task)
    num_rollouts = num_rollouts_per_task * len(config.collect.eval_tasks)
    agent.start_data_collection()

    with tqdm.tqdm(total=num_rollouts, desc="eval") as pbar:

        current_task_ids, valid_envs = [], []
        for _ in range(env.env_num):
            if repeated_task_ids:
                current_task_ids.append(repeated_task_ids.pop(0))
                valid_envs.append(True)
            else:
                current_task_ids.append(config.collect.eval_tasks[-1])
                valid_envs.append(False)

        obs, info = env.reset(options={"task_id": current_task_ids})

        while total_episodes < num_rollouts:
            action_chunk = agent.sample_actions(
                obs,
                task_description=info["task_description"],
            )
            env_action_chunk = action_chunk[0] if isinstance(action_chunk, tuple) else action_chunk
            next_obs, _, terminate, truncate, _ = env.step(env_action_chunk)

            done_per_step = np.logical_or(terminate, truncate)
            any_done = done_per_step[:, -1]  # True if episode ended this chunk
            first_done_idx = np.argmax(done_per_step, axis=1)  # first True index
            steps_this_chunk = np.where(any_done, first_done_idx + 1, config.collect.replan_steps)
            env_step_counts += steps_this_chunk

            current_terminate = terminate[:, -1]
            current_truncate = truncate[:, -1]

            done = np.logical_or(current_terminate, current_truncate)
            done = np.logical_and(done, valid_envs)
            done_indices = np.where(done)[0]

            if len(done_indices) > 0:
                total_episodes += len(done_indices)
                pbar.update(len(done_indices))

            for env_index in done_indices:
                success = bool(current_terminate[env_index])
                total_successes += int(success)
                successes_per_task[current_task_ids[env_index]] += int(success)
                episodes_per_task[current_task_ids[env_index]] += 1
                if success:
                    successful_episode_lengths.append(int(env_step_counts[env_index]))
                env_step_counts[env_index] = 0

                if repeated_task_ids:
                    task_id = repeated_task_ids.pop(0)
                else:
                    task_id = config.collect.eval_tasks[-1]
                    valid_envs[env_index] = False
                current_task_ids[env_index] = task_id
                env_obs, env_info = env.reset(id=int(env_index), options={"task_id": task_id})

                def update_state(prev_state, new_val_leaf):
                    prev_state[env_index] = new_val_leaf[0]
                    return prev_state

                next_obs = jax.tree.map(update_state, next_obs, env_obs)
                info = jax.tree.map(update_state, info, env_info)

            if total_episodes > 0:
                pbar.set_postfix(SR=total_successes / total_episodes)

            obs = next_obs

    agent.end_data_collection()
    metrics = {"eval/success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0}
    if successful_episode_lengths:
        metrics["eval/mean_success_episode_length"] = np.mean(successful_episode_lengths)
    metrics["eval/total_collected_episodes"] = agent.total_collected_episodes
    metrics.update({f"eval/success_rate/{task}": successes_per_task[task] / episodes_per_task[task] if episodes_per_task[task] > 0 else 0.0 for task in tasks})
    return metrics


def collect_data(
    agent: Agent,
    env: BaseVectorEnv,
    config,
    step: int,
    progress: dict | None = None,
    should_pause: Callable[[], bool] | None = None,
):
    """Collect one round of rollouts.

    Returns `(metrics, collected_episodes, progress)`. `progress` is None once the
    round is complete. If `should_pause` returns True, the round stops between env
    steps, drops the episodes still in flight and returns `metrics=None` plus the
    per-task counts so far; passing that `progress` back in (after a requeue)
    collects only the remaining rollouts.
    """
    part = 0 if progress is None else progress["part"]
    agent.start_data_collection(step=step)
    env.seed(config.seed + step + part)

    total_episodes = 0
    total_successes = 0
    tasks = set(config.collect.tasks)
    episodes_per_task = {k: 0 for k in tasks}
    successes_per_task = {k: 0 for k in tasks}
    if progress is not None:
        episodes_per_task.update(progress["episodes_per_task"])
        successes_per_task.update(progress["successes_per_task"])

    num_rollouts_per_task = config.collect.num_rollouts
    if step == 0 and config.collect.num_initial_rollouts is not None:
        num_rollouts_per_task += config.collect.num_initial_rollouts

    # Queue the rollouts still missing, in task order. Tasks may repeat in the list.
    already_done = dict(episodes_per_task)
    repeated_task_ids = []
    for t in config.collect.tasks:
        skip = min(already_done[t], num_rollouts_per_task)
        already_done[t] -= skip
        repeated_task_ids.extend([t] * (num_rollouts_per_task - skip))
    num_rollouts = len(repeated_task_ids)
    paused = False

    with tqdm.tqdm(total=num_rollouts) as pbar:

        current_task_ids, valid_envs = [], []
        for _ in range(env.env_num):
            if repeated_task_ids:
                current_task_ids.append(repeated_task_ids.pop(0))
                valid_envs.append(True)
            else:
                current_task_ids.append(config.collect.tasks[-1])
                valid_envs.append(False)

        obs, info = env.reset(options={"task_id": current_task_ids})

        while total_episodes < num_rollouts:
            # Every part finishes at least env_num episodes, so a run whose time
            # budget is already used up at startup still makes progress.
            if should_pause is not None and total_episodes >= env.env_num and should_pause():
                paused = True
                break
            action_chunk = agent.sample_actions(
                obs,
                task_description=info["task_description"],
            )
            env_action_chunk = action_chunk[0] if isinstance(action_chunk, tuple) else action_chunk
            next_obs, reward, terminate, truncate, _ = env.step(env_action_chunk)
            aligned_obs = _shift_window(observation=obs, next_observation=next_obs)

            if isinstance(action_chunk, tuple):
                action_payload = (env_action_chunk[:, :config.collect.replan_steps], action_chunk[1])
            else:
                action_payload = env_action_chunk[:, :config.collect.replan_steps]
            step_data = {
                "observation": aligned_obs,
                "next_observation": next_obs,
                "action": action_payload,
                "reward": reward,
                "terminate": terminate,
                "truncate": truncate,
            }
            agent.add_data(step_data)

            current_terminate = terminate[:, -1]
            current_truncate = truncate[:, -1]

            done = np.logical_or(current_terminate, current_truncate)
            done = np.logical_and(done, valid_envs)
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                total_episodes += len(done_indices)
                pbar.update(len(done_indices))

            for env_index in done_indices:
                success = bool(current_terminate[env_index])
                total_successes += int(success)
                successes_per_task[current_task_ids[env_index]] += int(success)
                episodes_per_task[current_task_ids[env_index]] += 1

                agent.save_episode(
                    is_success=success,
                    env_index=int(env_index),
                    task_description=info["task_description"][env_index],
                )

                if repeated_task_ids:
                    task_id = repeated_task_ids.pop(0)
                else:
                    task_id = config.collect.tasks[-1]
                    valid_envs[env_index] = False
                current_task_ids[env_index] = task_id
                env_obs, env_info = env.reset(id=int(env_index), options={"task_id": task_id})

                def update_state(prev_state, new_val_leaf):
                    prev_state[env_index] = new_val_leaf[0]
                    return prev_state

                next_obs = jax.tree.map(update_state, next_obs, env_obs)
                info = jax.tree.map(update_state, info, env_info)

            if total_episodes > 0:
                pbar.set_postfix(SR=total_successes / total_episodes)

            obs = next_obs

    # Drops the episodes still in flight if the round was paused.
    collected_episodes = agent.end_data_collection(step=step)

    agent.total_collected_episodes += total_episodes
    if paused:
        logging.info(
            "Paused collection at step %d after part %d: %d/%d rollouts of the round done.",
            step, part, sum(episodes_per_task.values()), num_rollouts_per_task * len(config.collect.tasks),
        )
        progress = {
            "step": step,
            "part": part + 1,
            "episodes_per_task": episodes_per_task,
            "successes_per_task": successes_per_task,
        }
        return None, collected_episodes, progress

    if part > 0:
        # Round finished across several parts: report it over all of them.
        total_episodes = sum(episodes_per_task.values())
        total_successes = sum(successes_per_task.values())
    metrics = {"success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0}
    metrics.update({f"success_rate/{task}": successes_per_task[task] / episodes_per_task[task] if episodes_per_task[task] > 0 else 0.0 for task in tasks})
    if part > 0:
        metrics["collect/num_parts"] = part + 1
    return metrics, collected_episodes, None
