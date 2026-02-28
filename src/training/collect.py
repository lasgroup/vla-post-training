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


def _supports_shift_window(observation: dict[str, Any], next_observation: dict[str, Any]) -> bool:
    leaves_obs = jax.tree_util.tree_leaves(observation)
    leaves_next = jax.tree_util.tree_leaves(next_observation)
    if not leaves_obs or not leaves_next or len(leaves_obs) != len(leaves_next):
        return False
    return all(
        np.asarray(curr).ndim >= 3 and np.asarray(nxt).ndim >= 3
        for curr, nxt in zip(leaves_obs, leaves_next)
    )


def collect_data(
    agent: Agent, env: BaseVectorEnv, task_description: str, config, step: int
):
    agent.start_data_collection(step=step)

    total_episodes = 0
    total_successes = 0
    num_rollouts = config.collect.num_rollouts
    env_num = int(config.collect.env_num)
    running_episode_returns = np.zeros(env_num, dtype=np.float32)
    running_episode_lengths = np.zeros(env_num, dtype=np.int32)
    completed_episode_returns: list[float] = []
    completed_episode_lengths: list[int] = []
    reward_sum = 0.0
    reward_count = 0

    with tqdm.tqdm(total=num_rollouts) as pbar:
        obs, _ = env.reset()

        while total_episodes < num_rollouts:
            action_chunk = agent.sample_actions(
                obs,
                task_description=task_description,
                batch_actions=True,
            )
            next_obs, reward, terminate, truncate, _ = env.step(action_chunk)

            # DSRL stores one transition per action chunk and later reduces the
            # chunked reward/done signals in save_episode(). In that setup,
            # shift-window alignment makes s_t inconsistent with the sampled
            # chunk action. Keep raw obs for DSRL learners.
            use_shift_window = (
                config.collect.add_per_step_data
                and _supports_shift_window(obs, next_obs)
                and not hasattr(agent, "replay")
            )
            if use_shift_window:
                aligned_obs = _shift_window(observation=obs, next_observation=next_obs)
            else:
                aligned_obs = obs

            step_data = {
                "observation": aligned_obs,
                "next_observation": next_obs,
                "action": action_chunk,
                "reward": reward,
                "terminate": terminate,
                "truncate": truncate,
            }
            agent.add_data(step_data)

            if config.collect.add_per_step_data and np.asarray(terminate).ndim > 1:
                current_terminate = np.asarray(terminate)[:, -1]
                current_truncate = np.asarray(truncate)[:, -1]
            else:
                current_terminate, current_truncate = terminate, truncate

            if config.collect.add_per_step_data and np.asarray(reward).ndim > 1:
                current_reward = np.asarray(reward)[:, -1]
            else:
                current_reward = np.asarray(reward)
            current_reward = np.asarray(current_reward, dtype=np.float32).reshape(-1)
            if current_reward.shape[0] != env_num:
                # Single-env fallback if a scalar slips through.
                current_reward = np.broadcast_to(current_reward, (env_num,))

            reward_sum += float(np.sum(current_reward))
            reward_count += int(current_reward.size)
            running_episode_returns += current_reward
            running_episode_lengths += 1

            done = np.logical_or(current_terminate, current_truncate)
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                total_episodes += len(done_indices)
                pbar.update(len(done_indices))

            for env_index in done_indices:
                success = bool(current_terminate[env_index])
                total_successes += int(success)
                completed_episode_returns.append(float(running_episode_returns[env_index]))
                completed_episode_lengths.append(int(running_episode_lengths[env_index]))
                running_episode_returns[env_index] = 0.0
                running_episode_lengths[env_index] = 0
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

            # DSRLVectorEnv keeps an internal observation cache used at step time.
            # After per-env resets we reconstruct a full-batch `next_obs` above;
            # mirror that into the env cache to keep batch sizes aligned.
            if hasattr(env, "_last_obs"):
                setattr(env, "_last_obs", next_obs)

            if total_episodes > 0:
                pbar.set_postfix(SR=total_successes / total_episodes)

            obs = next_obs

    collected_episodes = agent.end_data_collection(step=step)

    success_rate = (
        float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    )
    collect_info = {
        "success_rate": success_rate,
        "reward_step_mean": (reward_sum / reward_count) if reward_count > 0 else 0.0,
        "reward_step_sum": reward_sum,
        "reward_steps": reward_count,
    }
    if completed_episode_returns:
        returns = np.asarray(completed_episode_returns, dtype=np.float32)
        lengths = np.asarray(completed_episode_lengths, dtype=np.float32)
        collect_info.update(
            {
                "episode_return_mean": float(np.mean(returns)),
                "episode_return_std": float(np.std(returns)),
                "episode_return_min": float(np.min(returns)),
                "episode_return_max": float(np.max(returns)),
                "episode_length_mean": float(np.mean(lengths)),
            }
        )
    return collect_info, collected_episodes
