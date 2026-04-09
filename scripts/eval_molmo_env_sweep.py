#!/usr/bin/env python3
"""Evaluate a checkpoint on a list of Molmo benchmark env IDs.

Typical use:
  - sweep molmo_0..molmo_29 for 10 episodes each
  - log per-env / per-episode progress
  - write machine-readable JSON/CSV outputs
  - resume from partial progress across multiple runs
  - optionally shard the env list across multiple workers / GPUs
"""

import os
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"

import collections
import csv
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro

from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec, load_all_episodes
from molmo_spaces.policy.learned_policy.utils import PromptSampler

from demo_molmospaces_openpi import (
    _load_local_openpi_policy,
    _model_action_to_env_action,
    _resolve_checkpoint_path,
    _resolve_eval_config,
    _resolve_execute_horizon,
)
from src.envs.molmo import MolmoSpacesBenchmarkGymEnv, MolmoSpacesGymConfig
from src.envs.molmo_openpi import obs_to_openpi_input as _shared_obs_to_openpi_input


@dataclasses.dataclass
class Args:
    checkpoint_dir: str | None = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
    default_prompt: str | None = None

    benchmark_path: str = (
        "/capstor/store/cscs/swissai/a143/molmospaces/assets/benchmarks/"
        "molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/"
        "FrankaPickDroidMiniBench_json_benchmark_20251231"
    )
    eval_config_cls: str = (
        "molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig"
    )

    env_range: str | None = "0-29"
    env_ids: list[str] = dataclasses.field(default_factory=list)
    episodes_per_env: int = 10
    max_steps: int = 450
    episode_sampling: Literal["sequential", "random"] = "sequential"
    seed: int = 0
    task_horizon_steps: int | None = None

    # Observation mapping.
    exo_camera_key: str = "exo_camera_1"
    wrist_camera_key: str = "wrist_camera"

    # Action mapping.
    execute_horizon: int = 8
    grasping_type: str | None = None
    gripper_threshold: float | None = None
    gripper_scale: float = 255.0

    # Sharding / progress.
    num_shards: int = 1
    shard_index: int = 0
    render_device: int = 0
    output_dir: str = "outputs/molmo_env_sweep"
    output_prefix: str = "results"
    progress_every: int = 1


def _parse_env_token(token: str) -> int:
    token = token.strip()
    if token.startswith("molmo_"):
        token = token.split("_", maxsplit=1)[1]
    return int(token)


def _expand_env_selection(args: Args) -> list[int]:
    env_ids: list[int] = []
    if args.env_range:
        start_str, end_str = args.env_range.split("-", maxsplit=1)
        start = _parse_env_token(start_str)
        end = _parse_env_token(end_str)
        if end < start:
            raise ValueError(f"Invalid env_range={args.env_range!r}: end < start")
        env_ids.extend(range(start, end + 1))
    env_ids.extend(_parse_env_token(token) for token in args.env_ids)
    if not env_ids:
        raise ValueError("No envs selected. Provide --env-range or --env-ids.")
    return list(dict.fromkeys(env_ids))


def _shard_env_ids(env_ids: list[int], num_shards: int, shard_index: int) -> list[int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    return [env_id for i, env_id in enumerate(env_ids) if i % num_shards == shard_index]


def _extract_task_description(episode: EpisodeSpec) -> str:
    language = episode.language
    if hasattr(language, "task_description"):
        return str(language.task_description)
    if isinstance(language, dict):
        return str(language.get("task_description", ""))
    return ""


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = ["env_id", "env_name", "successes", "episodes", "sr"]
    else:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_episode_history(progress_path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    history: dict[int, dict[int, dict[str, Any]]] = {}
    if not progress_path.exists():
        return history

    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "episode_complete":
                continue

            env_id = int(payload["env_id"])
            episode_in_env = int(payload["episode_in_env"])
            history.setdefault(env_id, {})[episode_in_env] = payload

    return history


def _load_existing_results(results_json_path: Path) -> dict[int, dict[str, Any]]:
    if not results_json_path.exists():
        return {}

    try:
        data = json.loads(results_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    return {int(row["env_id"]): row for row in data}


def _build_result_row(
    *,
    args: Args,
    checkpoint_path: str,
    episode_spec: EpisodeSpec,
    env_id: int,
    task_description: str,
    successes: int,
    rewards: list[float],
    steps_taken: list[int],
    policy_prompts: list[str],
    env_elapsed: float,
) -> dict[str, Any]:
    return {
        "env_id": env_id,
        "env_name": f"molmo_{env_id}",
        "house_index": episode_spec.house_index,
        "task_description": task_description,
        "task_cls": episode_spec.task.get("task_cls", ""),
        "successes": successes,
        "episodes": args.episodes_per_env,
        "sr": successes / max(args.episodes_per_env, 1),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_steps": float(np.mean(steps_taken)) if steps_taken else 0.0,
        "total_elapsed_sec": env_elapsed,
        "checkpoint": checkpoint_path,
        "policy_prompt_example": policy_prompts[0] if policy_prompts else "",
        "seed_base": args.seed,
        "render_device": args.render_device,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }


def _write_result_files(
    results: list[dict[str, Any]],
    results_json_path: Path,
    results_csv_path: Path,
    ranked_json_path: Path,
    ranked_csv_path: Path,
) -> None:
    results_sorted_by_env = sorted(results, key=lambda row: row["env_id"])
    results_sorted_by_rank = sorted(
        results,
        key=lambda row: (-row["sr"], -row["successes"], row["mean_steps"], row["env_id"]),
    )
    ranked_rows = [
        {"rank": rank, **row}
        for rank, row in enumerate(results_sorted_by_rank, start=1)
    ]

    results_json_path.write_text(
        json.dumps(results_sorted_by_env, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    ranked_json_path.write_text(
        json.dumps(ranked_rows, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_csv(results_csv_path, results_sorted_by_env)
    _write_csv(ranked_csv_path, ranked_rows)


def _prefix(args: Args) -> str:
    if args.num_shards > 1:
        return f"[shard {args.shard_index + 1}/{args.num_shards}]"
    return "[sweep]"


def _make_env_config(args: Args, benchmark_path: Path) -> MolmoSpacesGymConfig:
    config_kwargs: dict[str, Any] = {
        "benchmark_dir": str(benchmark_path),
        "eval_config_cls": args.eval_config_cls,
        "episode_sampling": args.episode_sampling,
        "seed": args.seed,
    }

    if "task_horizon_steps" in MolmoSpacesGymConfig.__dataclass_fields__:
        config_kwargs["task_horizon_steps"] = args.task_horizon_steps

    return MolmoSpacesGymConfig(**config_kwargs)


def run(args: Args) -> None:
    benchmark_path = Path(args.benchmark_path).expanduser().resolve()
    episodes = load_all_episodes(benchmark_path)
    if not episodes:
        raise ValueError(
            f"No benchmark episodes found in {benchmark_path}. "
            "Expected benchmark.json or house_*/episode_*.json files."
        )

    selected_env_ids = _expand_env_selection(args)
    for env_id in selected_env_ids:
        if not 0 <= env_id < len(episodes):
            raise ValueError(f"env_id {env_id} out of range [0, {len(episodes) - 1}]")
    shard_env_ids = _shard_env_ids(selected_env_ids, args.num_shards, args.shard_index)
    if not shard_env_ids:
        logging.warning("No envs assigned to this shard; exiting early.")
        return

    eval_config = _resolve_eval_config(args.eval_config_cls)
    checkpoint_path = _resolve_checkpoint_path(args, eval_config)
    policy, train_cfg = _load_local_openpi_policy(
        checkpoint_path,
        default_prompt=args.default_prompt,
    )
    execute_horizon = _resolve_execute_horizon(args, train_cfg, eval_config)
    grasping_type = args.grasping_type or eval_config.policy_config.grasping_type
    gripper_threshold = (
        args.gripper_threshold
        if args.gripper_threshold is not None
        else eval_config.policy_config.grasping_threshold
    )
    prompt_sampler = PromptSampler(
        task_type=eval_config.task_type,
        prompt_templates=eval_config.policy_config.prompt_templates,
        prompt_object_word_num=eval_config.policy_config.prompt_object_word_num,
    )
    env_cfg = _make_env_config(args, benchmark_path)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_tag = f"{args.output_prefix}_shard{args.shard_index:02d}-of-{args.num_shards:02d}"
    progress_path = output_dir / f"{shard_tag}.progress.jsonl"
    results_json_path = output_dir / f"{shard_tag}.json"
    results_csv_path = output_dir / f"{shard_tag}.csv"
    ranked_json_path = output_dir / f"{shard_tag}.ranked.json"
    ranked_csv_path = output_dir / f"{shard_tag}.ranked.csv"

    prefix = _prefix(args)
    total_envs = len(shard_env_ids)
    total_episodes = total_envs * args.episodes_per_env
    episode_history = _load_episode_history(progress_path)
    existing_results = _load_existing_results(results_json_path)
    completed_episodes = 0
    for env_id in shard_env_ids:
        if env_id in existing_results:
            completed_episodes += args.episodes_per_env
        else:
            completed_episodes += min(len(episode_history.get(env_id, {})), args.episodes_per_env)
    start_time = time.time()
    results: list[dict[str, Any]] = []

    logging.info(
        "%s starting sweep | checkpoint=%s | total_selected_envs=%d | shard_envs=%d | episodes_per_env=%d | resumed_episodes=%d",
        prefix,
        checkpoint_path,
        len(selected_env_ids),
        total_envs,
        args.episodes_per_env,
        completed_episodes,
    )

    for env_pos, env_id in enumerate(shard_env_ids, start=1):
        episode_spec = episodes[env_id]
        task_description = _extract_task_description(episode_spec)
        env_name = f"molmo_{env_id}"
        prior_episode_map = episode_history.get(env_id, {})
        prior_episode_payloads = [
            prior_episode_map[key]
            for key in sorted(prior_episode_map)
            if key <= args.episodes_per_env
        ]
        prior_completed = len(prior_episode_payloads)

        if env_id in existing_results:
            results.append(existing_results[env_id])
            logging.info(
                "%s skipping completed %s from prior results | sr=%.3f | successes=%s/%s",
                prefix,
                env_name,
                existing_results[env_id]["sr"],
                existing_results[env_id]["successes"],
                existing_results[env_id]["episodes"],
            )
            continue

        successes = sum(int(payload.get("success", False)) for payload in prior_episode_payloads)
        rewards = [float(payload.get("episode_reward", 0.0)) for payload in prior_episode_payloads]
        steps_taken = [int(payload.get("episode_steps", 0)) for payload in prior_episode_payloads]
        policy_prompts = [
            str(payload.get("policy_prompt", ""))
            for payload in prior_episode_payloads
            if payload.get("policy_prompt")
        ]
        env_elapsed_from_history = sum(
            float(payload.get("episode_elapsed_sec", 0.0))
            for payload in prior_episode_payloads
        )

        if prior_completed >= args.episodes_per_env:
            reconstructed = _build_result_row(
                args=args,
                checkpoint_path=checkpoint_path,
                episode_spec=episode_spec,
                env_id=env_id,
                task_description=task_description,
                successes=successes,
                rewards=rewards,
                steps_taken=steps_taken,
                policy_prompts=policy_prompts,
                env_elapsed=env_elapsed_from_history,
            )
            results.append(reconstructed)
            _write_result_files(
                results,
                results_json_path,
                results_csv_path,
                ranked_json_path,
                ranked_csv_path,
            )
            logging.info(
                "%s reconstructed completed %s from progress log | sr=%.3f | successes=%d/%d",
                prefix,
                env_name,
                reconstructed["sr"],
                successes,
                args.episodes_per_env,
            )
            continue

        env = MolmoSpacesBenchmarkGymEnv(
            episode_id=env_id,
            render_device=args.render_device,
            config=env_cfg,
        )
        env_start = time.time()

        if prior_completed > 0:
            logging.info(
                "%s resuming env %d/%d | %s | house=%s | task=%s | already_done=%d/%d | current_sr=%.3f",
                prefix,
                env_pos,
                total_envs,
                env_name,
                episode_spec.house_index,
                task_description,
                prior_completed,
                args.episodes_per_env,
                successes / prior_completed,
            )
        else:
            logging.info(
                "%s env %d/%d | %s | house=%s | task=%s",
                prefix,
                env_pos,
                total_envs,
                env_name,
                episode_spec.house_index,
                task_description,
            )

        for ep_idx in range(prior_completed, args.episodes_per_env):
            seed = args.seed + env_id * 1000 + ep_idx
            obs, _info = env.reset(seed=seed)
            policy.reset()
            prompt_sampler.next()
            policy_prompt = (
                prompt_sampler.get_prompt(env._task).lower()
                if getattr(env, "_task", None) is not None
                else (args.default_prompt or task_description or "do the task").lower()
            )
            policy_prompts.append(policy_prompt)

            action_buffer: collections.deque[np.ndarray] = collections.deque()
            episode_success = False
            episode_reward = 0.0
            episode_steps = 0

            ep_start = time.time()
            for step_idx in range(args.max_steps):
                model_input = _shared_obs_to_openpi_input(
                    obs,
                    exo_camera_key=args.exo_camera_key,
                    wrist_camera_key=args.wrist_camera_key,
                    gripper_obs_norm=0.824033,
                    prompt=policy_prompt,
                )

                if not action_buffer:
                    action_chunk = np.asarray(
                        policy.infer(model_input, sharding_spec=None)["actions"]
                    )
                    action_buffer.extend(action_chunk[:execute_horizon])

                raw_action = action_buffer.popleft()
                env_action = _model_action_to_env_action(
                    raw_action,
                    grasping_type=grasping_type,
                    gripper_threshold=gripper_threshold,
                    gripper_scale=args.gripper_scale,
                )
                obs, reward, terminated, truncated, info = env.step(env_action)
                episode_reward += float(reward)
                episode_steps = step_idx + 1

                if info["success"]:
                    episode_success = True
                    break
                if terminated or truncated:
                    break

            ep_elapsed = time.time() - ep_start
            successes += int(episode_success)
            rewards.append(episode_reward)
            steps_taken.append(episode_steps)
            completed_episodes += 1
            running_eps = ep_idx + 1
            running_sr = successes / running_eps
            total_progress = completed_episodes / total_episodes

            payload = {
                "event": "episode_complete",
                "shard_index": args.shard_index,
                "env_position": env_pos,
                "total_envs": total_envs,
                "env_id": env_id,
                "env_name": env_name,
                "episode_in_env": running_eps,
                "episodes_per_env": args.episodes_per_env,
                "success": episode_success,
                "running_successes": successes,
                "running_sr": running_sr,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "episode_elapsed_sec": ep_elapsed,
                "seed": seed,
                "task_description": task_description,
                "policy_prompt": policy_prompt,
                "completed_episodes": completed_episodes,
                "total_episodes": total_episodes,
                "total_progress": total_progress,
                "timestamp": time.time(),
            }
            _append_jsonl(progress_path, payload)

            if running_eps % args.progress_every == 0 or running_eps == args.episodes_per_env:
                elapsed = time.time() - start_time
                logging.info(
                    (
                        "%s env %d/%d | %s | episode %d/%d | success=%s | "
                        "env_sr=%.3f (%d/%d) | total_progress=%d/%d (%.1f%%) | %.1fs"
                    ),
                    prefix,
                    env_pos,
                    total_envs,
                    env_name,
                    running_eps,
                    args.episodes_per_env,
                    episode_success,
                    running_sr,
                    successes,
                    running_eps,
                    completed_episodes,
                    total_episodes,
                    100.0 * total_progress,
                    elapsed,
                )

        env.close()

        env_elapsed = env_elapsed_from_history + (time.time() - env_start)
        result = _build_result_row(
            args=args,
            checkpoint_path=checkpoint_path,
            episode_spec=episode_spec,
            env_id=env_id,
            task_description=task_description,
            successes=successes,
            rewards=rewards,
            steps_taken=steps_taken,
            policy_prompts=policy_prompts,
            env_elapsed=env_elapsed,
        )
        results.append(result)
        _write_result_files(
            results,
            results_json_path,
            results_csv_path,
            ranked_json_path,
            ranked_csv_path,
        )
        _append_jsonl(
            progress_path,
            {
                "event": "env_complete",
                "env_position": env_pos,
                "total_envs": total_envs,
                **result,
                "timestamp": time.time(),
            },
        )
        logging.info(
            "%s finished %s | sr=%.3f | successes=%d/%d | mean_steps=%.1f | %.1fs",
            prefix,
            env_name,
            result["sr"],
            successes,
            args.episodes_per_env,
            result["mean_steps"],
            env_elapsed,
        )

    _write_result_files(
        results,
        results_json_path,
        results_csv_path,
        ranked_json_path,
        ranked_csv_path,
    )

    logging.info("%s wrote %s", prefix, results_json_path)
    logging.info("%s wrote %s", prefix, ranked_json_path)
    logging.info("%s wrote %s", prefix, results_csv_path)
    logging.info("%s wrote %s", prefix, ranked_csv_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()
