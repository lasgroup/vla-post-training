import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import tqdm_loggable.auto as tqdm

from src.envs.venv import BaseVectorEnv
from src.rl.agent import Agent


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


def _evaluate_policy_legacy(agent: Agent, env: BaseVectorEnv, config, step: int):
    num_rollouts_per_task = config.collect.num_eval_rollouts
    total_episodes = 0
    total_successes = 0
    tasks = set(config.collect.tasks)
    episodes_per_task = {k: 0 for k in tasks}
    successes_per_task = {k: 0 for k in tasks}
    env_step_counts = np.zeros(env.env_num, dtype=np.int32)
    successful_episode_lengths = []

    repeated_task_ids = []
    for t in config.collect.tasks:
        repeated_task_ids.extend([t] * num_rollouts_per_task)
    num_rollouts = num_rollouts_per_task * len(config.collect.tasks)

    with tqdm.tqdm(total=num_rollouts, desc="eval") as pbar:
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
            env_action_chunk = (
                action_chunk[0] if isinstance(action_chunk, tuple) else action_chunk
            )
            next_obs, _, terminate, truncate, _ = env.step(env_action_chunk)

            done_per_step = np.logical_or(terminate, truncate)
            any_done = done_per_step[:, -1]  # True if episode ended this chunk
            first_done_idx = np.argmax(done_per_step, axis=1)  # first True index
            steps_this_chunk = np.where(
                any_done, first_done_idx + 1, config.collect.replan_steps
            )
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
                    task_id = config.collect.tasks[-1]
                    valid_envs[env_index] = False
                current_task_ids[env_index] = task_id
                env_obs, env_info = env.reset(
                    id=int(env_index), options={"task_id": task_id}
                )

                def update_state(prev_state, new_val_leaf):
                    prev_state[env_index] = new_val_leaf[0]
                    return prev_state

                next_obs = jax.tree.map(update_state, next_obs, env_obs)
                info = jax.tree.map(update_state, info, env_info)

            if total_episodes > 0:
                pbar.set_postfix(SR=total_successes / total_episodes)

            obs = next_obs

    metrics = {
        "eval/success_rate": float(total_successes) / float(total_episodes)
        if total_episodes > 0
        else 0.0
    }
    if successful_episode_lengths:
        metrics["eval/mean_success_episode_length"] = np.mean(
            successful_episode_lengths
        )
    metrics["eval/total_collected_episodes"] = agent.total_collected_episodes
    metrics.update(
        {
            f"eval/success_rate/{task}": successes_per_task[task]
            / episodes_per_task[task]
            if episodes_per_task[task] > 0
            else 0.0
            for task in tasks
        }
    )
    return metrics


def _load_fixed_eval_manifest(path: Path, config) -> tuple[list[dict[str, Any]], str]:
    manifest_bytes = path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest_bytes.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number} must be a JSON object")
        task = row.get("task")
        initial_state_index = row.get("initial_state_index")
        policy_seed = row.get("policy_seed")
        if not isinstance(task, str):
            raise TypeError(f"{path}:{line_number} task must be a string")
        if not isinstance(initial_state_index, int) or isinstance(
            initial_state_index, bool
        ):
            raise TypeError(
                f"{path}:{line_number} initial_state_index must be an integer"
            )
        if initial_state_index < 0:
            raise ValueError(
                f"{path}:{line_number} initial_state_index must be non-negative"
            )
        if not isinstance(policy_seed, int) or isinstance(policy_seed, bool):
            raise TypeError(f"{path}:{line_number} policy_seed must be an integer")
        if not 0 <= policy_seed < 2**32:
            raise ValueError(f"{path}:{line_number} policy_seed must fit uint32")
        manifest_index = row.get("manifest_index", len(rows))
        if (
            not isinstance(manifest_index, int)
            or isinstance(manifest_index, bool)
            or manifest_index != len(rows)
        ):
            raise ValueError(
                f"{path}:{line_number} manifest_index must equal {len(rows)}"
            )
        normalized = dict(row)
        normalized["manifest_index"] = manifest_index
        rows.append(normalized)

    expected_tasks = list(config.collect.tasks)
    expected_count = int(config.collect.num_eval_rollouts) * len(expected_tasks)
    if len(rows) != expected_count:
        raise ValueError(f"{path} has {len(rows)} rows; expected {expected_count}")
    manifest_tasks = sorted({row["task"] for row in rows})
    if manifest_tasks != sorted(expected_tasks):
        raise ValueError(
            f"{path} tasks do not match config: manifest={manifest_tasks} "
            f"config={sorted(expected_tasks)}"
        )
    counts = {task: 0 for task in expected_tasks}
    keys: set[tuple[str, int, int]] = set()
    for row in rows:
        counts[row["task"]] += 1
        key = (row["task"], row["initial_state_index"], row["policy_seed"])
        if key in keys:
            raise ValueError(f"duplicate fixed-evaluation tuple in {path}: {key}")
        keys.add(key)
    expected_per_task = int(config.collect.num_eval_rollouts)
    if any(count != expected_per_task for count in counts.values()):
        raise ValueError(
            f"{path} per-task counts do not match num_eval_rollouts={expected_per_task}: {counts}"
        )
    return rows, manifest_sha256


def _set_policy_seed(agent: Agent, policy_seed: int) -> None:
    # The SFT learner samples explicit action noise with agent._rng, while the
    # wrapped OpenPI policy maintains its own sampler key. Reset both so each
    # episode is independent of manifest order and earlier rollout lengths.
    if hasattr(agent, "_rng"):
        agent._rng = jax.random.key(policy_seed)
    policy = getattr(agent, "_policy", None)
    if policy is not None and hasattr(policy, "_rng"):
        policy._rng = jax.random.key(policy_seed ^ 0xA5A5A5A5)


def _load_fixed_eval_prefix(
    results_path: Path,
    manifest_rows: list[dict[str, Any]],
    manifest_sha256: str,
    checkpoint_id: str,
) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        results_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        index = len(results)
        if index >= len(manifest_rows):
            raise ValueError(f"{results_path}:{line_number} exceeds manifest length")
        spec = manifest_rows[index]
        expected = {
            "manifest_index": index,
            "task": spec["task"],
            "initial_state_index": spec["initial_state_index"],
            "policy_seed": spec["policy_seed"],
            "manifest_sha256": manifest_sha256,
            "checkpoint_id": checkpoint_id,
        }
        if not isinstance(row, dict):
            raise TypeError(f"{results_path}:{line_number} must be a JSON object")
        observed = {key: row.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"{results_path}:{line_number} is not the expected manifest prefix: "
                f"observed={observed} expected={expected}"
            )
        for bool_field in ("success", "truncated"):
            if not isinstance(row.get(bool_field), bool):
                raise TypeError(
                    f"{results_path}:{line_number} {bool_field} must be a boolean"
                )
        for int_field in ("episode_length", "action_chunks", "training_step"):
            value = row.get(int_field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(
                    f"{results_path}:{line_number} {int_field} must be a non-negative integer"
                )
        if row["episode_length"] == 0 or row["action_chunks"] == 0:
            raise ValueError(
                f"{results_path}:{line_number} completed episode counters must be positive"
            )
        results.append(row)
    return results


def _evaluate_policy_fixed_manifest(
    agent: Agent,
    env: BaseVectorEnv,
    config,
    step: int,
    manifest_path: Path,
    results_path: Path,
):
    if env.env_num != 1:
        raise ValueError(
            f"fixed-manifest evaluation requires env_num=1, got {env.env_num}"
        )
    manifest_rows, manifest_sha256 = _load_fixed_eval_manifest(manifest_path, config)
    checkpoint_id = os.environ.get("VLA_EVAL_CHECKPOINT_ID", "")
    if not checkpoint_id:
        raise ValueError(
            "VLA_EVAL_CHECKPOINT_ID is required for fixed-manifest evaluation"
        )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = _load_fixed_eval_prefix(
        results_path,
        manifest_rows,
        manifest_sha256,
        checkpoint_id,
    )

    with (
        tqdm.tqdm(
            total=len(manifest_rows),
            initial=len(results),
            desc="fixed_eval",
        ) as pbar,
        results_path.open("a", encoding="utf-8") as results_file,
    ):
        for spec in manifest_rows[len(results) :]:
            task = spec["task"]
            initial_state_index = int(spec["initial_state_index"])
            policy_seed = int(spec["policy_seed"])
            _set_policy_seed(agent, policy_seed)
            obs, info = cast(
                tuple[Any, dict[str, Any]],
                env.reset(
                    options={
                        "task_id": [task],
                        "init_state_index": [initial_state_index],
                    }
                ),
            )
            observed_state_index = int(
                np.asarray(info["init_state_index"]).reshape(-1)[0]
            )
            if observed_state_index != initial_state_index:
                raise RuntimeError(
                    f"reset selected state {observed_state_index}; expected {initial_state_index}"
                )

            episode_length = 0
            action_chunks = 0
            success = False
            truncated = False
            while True:
                action_chunk = agent.sample_actions(
                    obs,
                    task_description=info["task_description"],
                )
                env_action_chunk = (
                    action_chunk[0] if isinstance(action_chunk, tuple) else action_chunk
                )
                next_obs, _, terminate, truncate, _ = cast(
                    tuple[Any, Any, np.ndarray, np.ndarray, Any],
                    env.step(env_action_chunk),
                )
                action_chunks += 1

                done_per_step = np.logical_or(terminate, truncate)
                any_done = bool(done_per_step[0, -1])
                first_done_idx = int(np.argmax(done_per_step[0]))
                episode_length += (
                    first_done_idx + 1 if any_done else int(config.collect.replan_steps)
                )
                obs = next_obs
                if any_done:
                    success = bool(terminate[0, -1])
                    truncated = bool(truncate[0, -1])
                    break
                if action_chunks > 10_000:
                    raise RuntimeError(
                        "fixed evaluation exceeded 10,000 action chunks without termination"
                    )

            result = {
                "manifest_index": int(spec["manifest_index"]),
                "episode_id": spec.get(
                    "episode_id", f"episode_{spec['manifest_index']:03d}"
                ),
                "task": task,
                "initial_state_index": initial_state_index,
                "policy_seed": policy_seed,
                "manifest_sha256": manifest_sha256,
                "checkpoint_id": checkpoint_id,
                "training_step": int(step),
                "success": success,
                "truncated": truncated,
                "episode_length": episode_length,
                "action_chunks": action_chunks,
            }
            results_file.write(json.dumps(result, sort_keys=True) + "\n")
            results_file.flush()
            os.fsync(results_file.fileno())
            results.append(result)
            pbar.update(1)
            pbar.set_postfix(
                SR=sum(int(row["success"]) for row in results) / len(results)
            )

    successes = sum(int(row["success"]) for row in results)
    success_lengths = [int(row["episode_length"]) for row in results if row["success"]]
    tasks = list(config.collect.tasks)
    metrics: dict[str, Any] = {
        "eval/success_rate": successes / len(results) if results else 0.0,
        "eval/successes": successes,
        "eval/episodes": len(results),
        "eval/total_collected_episodes": getattr(agent, "total_collected_episodes", 0),
        "eval/manifest_sha256": manifest_sha256,
    }
    if success_lengths:
        metrics["eval/mean_success_episode_length"] = float(np.mean(success_lengths))
    for task in tasks:
        task_rows = [row for row in results if row["task"] == task]
        metrics[f"eval/success_rate/{task}"] = (
            sum(int(row["success"]) for row in task_rows) / len(task_rows)
            if task_rows
            else 0.0
        )
    return metrics


def evaluate_policy(agent: Agent, env: BaseVectorEnv, config, step: int):
    manifest = os.environ.get("VLA_FIXED_EVAL_MANIFEST")
    if manifest:
        results = os.environ.get("VLA_FIXED_EVAL_RESULTS")
        if not results:
            raise ValueError(
                "VLA_FIXED_EVAL_RESULTS is required with VLA_FIXED_EVAL_MANIFEST"
            )
        return _evaluate_policy_fixed_manifest(
            agent=agent,
            env=env,
            config=config,
            step=step,
            manifest_path=Path(manifest),
            results_path=Path(results),
        )
    return _evaluate_policy_legacy(agent=agent, env=env, config=config, step=step)


def collect_data(agent: Agent, env: BaseVectorEnv, config, step: int):
    agent.start_data_collection(step=step)

    total_episodes = 0
    total_successes = 0
    tasks = set(config.collect.tasks)
    episodes_per_task = {k: 0 for k in tasks}
    successes_per_task = {k: 0 for k in tasks}

    if step == 0 and config.collect.num_initial_rollouts is not None:
        num_rollouts_per_task = config.collect.num_initial_rollouts
    else:
        num_rollouts_per_task = config.collect.num_rollouts
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
            env_action_chunk = (
                action_chunk[0] if isinstance(action_chunk, tuple) else action_chunk
            )
            next_obs, reward, terminate, truncate, _ = env.step(env_action_chunk)
            aligned_obs = _shift_window(observation=obs, next_observation=next_obs)

            if isinstance(action_chunk, tuple):
                action_payload = (
                    env_action_chunk[:, : config.collect.replan_steps],
                    action_chunk[1],
                )
            else:
                action_payload = env_action_chunk[:, : config.collect.replan_steps]
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
                env_obs, env_info = env.reset(
                    id=int(env_index), options={"task_id": task_id}
                )

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
    metrics = {
        "success_rate": float(total_successes) / float(total_episodes)
        if total_episodes > 0
        else 0.0
    }
    metrics.update(
        {
            f"success_rate/{task}": successes_per_task[task] / episodes_per_task[task]
            if episodes_per_task[task] > 0
            else 0.0
            for task in tasks
        }
    )
    return metrics, collected_episodes
