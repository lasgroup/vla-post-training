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


def collect_data(
    agent: Agent, env: BaseVectorEnv, task_descriptions: list[str], config, step: int
):
    agent.start_data_collection(step=step)
    env_num_multiask = len(config.collect.tasks)

    total_episodes = 0
    total_successes = 0
    episodes_per_env = [0] * env_num_multiask
    successes_per_env = [0] * env_num_multiask
    
    num_rollouts = config.collect.num_rollouts

    with tqdm.tqdm(total=num_rollouts) as pbar:
        obs, _ = env.reset()

        while total_episodes < num_rollouts:
            action_chunk = agent.sample_actions(
                obs,
                task_descriptions=task_descriptions,
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

            current_terminate = jax.tree.map(lambda x: x[:, -1], terminate)
            current_truncate = jax.tree.map(lambda x: x[:, -1], truncate)

            done = np.logical_or(current_terminate, current_truncate)
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                total_episodes += len(done_indices)
                pbar.update(len(done_indices))

            for env_index in done_indices:
                success = bool(current_terminate[env_index])
                total_successes += int(success)
                successes_per_env[env_index] += int(success)
                episodes_per_env[env_index] += 1

                agent.save_episode(
                    is_success=success,
                    env_index=int(env_index),
                    task_description=task_descriptions[env_index], 
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

    success_rate = (
        float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    )

    metrics = {"success_rate": success_rate}

    per_env_success_rates = [s / e if e > 0 else 0.0 for s, e in zip(successes_per_env, episodes_per_env)]
    success_rate_mean = np.mean(per_env_success_rates)
    metrics["success_rate_mean"] = success_rate_mean

    for i in range(env_num_multiask):
        metrics[f"Per Task/success_rate_{config.collect.task_names[i]}"] = per_env_success_rates[i]

    return metrics, collected_episodes