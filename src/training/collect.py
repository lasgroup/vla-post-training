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
    agent: Agent, env: BaseVectorEnv, task_description: str, config, step: int
):
    agent.start_data_collection(step=step)

    total_episodes = 0
    total_successes = 0
    num_rollouts = config.collect.num_rollouts

    with tqdm.tqdm(total=num_rollouts) as pbar:
        obs, _ = env.reset()

        while total_episodes < num_rollouts:
            action_chunk = agent.sample_actions(
                obs,
                task_description=task_description,
                batch_actions=True,
            )
            next_obs, reward, terminate, truncate, _ = env.step(action_chunk)

            if config.collect.add_per_step_data:
                aligned_obs = _shift_window(observation=obs, next_observation=next_obs)
            else:
                aligned_obs = next_obs

            step_data = {
                "observation": aligned_obs,
                "next_observation": next_obs,
                "action": action_chunk,
                "reward": reward,
                "terminate": terminate,
                "truncate": truncate,
            }
            agent.add_data(step_data)

            if config.collect.add_per_step_data:
                current_terminate = jax.tree.map(lambda x: x[:, -1], terminate)
                current_truncate = jax.tree.map(lambda x: x[:, -1], truncate)
            else:
                current_terminate, current_truncate = terminate, truncate

            done = np.logical_or(current_terminate, current_truncate)
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                total_episodes += len(done_indices)
                pbar.update(len(done_indices))

            for env_index in done_indices:
                success = bool(current_terminate[env_index])
                total_successes += int(success)
                agent.save_episode(
                    is_success=success,
                    env_index=int(env_index),
                    task_description=task_description,
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
    return {"success_rate": success_rate}, collected_episodes
