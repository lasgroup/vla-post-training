from typing import Any
import jax
import logging
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
    agent.start_data_collection(evaluation=True)

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


def _tag_metrics(metrics: dict[str, Any], scale: float) -> dict[str, Any]:
    """Re-key `eval/<rest>` as `eval/cfg<scale>/<rest>` so scales don't collide."""
    tag = f"cfg{scale:g}"
    return {
        f"eval/{tag}/{k.removeprefix('eval/')}" if k.startswith("eval/") else f"{tag}/{k}": v
        for k, v in metrics.items()
    }


def evaluate_policy_sweep(
    agent: Agent, env: BaseVectorEnv, config, step: int
) -> dict[str, Any]:
    """Evaluate once per guidance scale, reusing the same envs.

    With a single scale (the default) this is `evaluate_policy` with untouched
    metric names. Agents without CFG support fall through to one plain eval.
    """
    scales = getattr(agent, "cfg_scales", [1.0])
    if len(scales) == 1:
        return evaluate_policy(agent=agent, env=env, config=config, step=step)

    metrics = {}
    for scale in scales:
        agent.set_cfg_scale(scale)
        logging.info("Evaluating at cfg_scale=%g (%d of %d)", scale, scales.index(scale) + 1, len(scales))
        metrics.update(_tag_metrics(evaluate_policy(agent=agent, env=env, config=config, step=step), scale))
    agent.set_cfg_scale(scales[0])
    return metrics


def collect_data(
    agent: Agent, env: BaseVectorEnv, config, step: int
):
    agent.start_data_collection(step=step)
    env.seed(config.seed + step)

    total_episodes = 0
    total_successes = 0
    tasks = set(config.collect.tasks)
    episodes_per_task = {k: 0 for k in tasks}
    successes_per_task = {k: 0 for k in tasks}
    
    num_rollouts_per_task = config.collect.num_rollouts
    if step == 0 and config.collect.num_initial_rollouts is not None:
        num_rollouts_per_task += config.collect.num_initial_rollouts
    num_rollouts = num_rollouts_per_task * len(config.collect.tasks)

    repeated_task_ids = []
    for t in config.collect.tasks:
        repeated_task_ids.extend([t] * num_rollouts_per_task)

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

    collected_episodes = agent.end_data_collection(step=step)

    agent.total_collected_episodes += total_episodes
    metrics = {"success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0}
    metrics.update({f"success_rate/{task}": successes_per_task[task] / episodes_per_task[task] if episodes_per_task[task] > 0 else 0.0 for task in tasks})
    return metrics, collected_episodes
