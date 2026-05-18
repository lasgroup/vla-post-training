import jax
import jax.numpy as jnp
from flax.training import common_utils
import logging

import tqdm_loggable.auto as tqdm
import wandb
from src.rl.agent import Agent
from src.envs.venv import BaseVectorEnv
from src.training.collect import collect_data, evaluate_policy
import src.training.config as _config


def train_loop(config: _config.OnlineTrainConfig,
               agent: Agent,
               env: BaseVectorEnv,
               eval_env: BaseVectorEnv,
               task_description: list[str],
               eval_task_description: list[str]):
    # Warm up agent with offline data training before data collection
    warmup_start_steps = agent.warm_start_training_steps

    warmup_pbar = tqdm.tqdm(
        range(warmup_start_steps, config.rl.num_offline_pretraining_steps),
        initial=warmup_start_steps,
        total=config.rl.num_offline_pretraining_steps,
        dynamic_ncols=True,
    )

    infos = []
    pretraining_steps = 0
    for step in warmup_pbar:
        info = agent.pretrain_with_offline_data()
        infos.append(info)
        if step % config.log_interval == 0:
            # Infos may have different keys (actor-only, critic-only, or both),
            # so we normalize them before stacking.
            all_keys = set().union(*(d.keys() for d in infos))
            nan = jnp.array(float("nan"))
            normalized = [{k: d.get(k, nan) for k in sorted(all_keys)} for d in infos]
            stacked_infos = common_utils.stack_forest(normalized)
            reduced_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            warmup_pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        pretraining_steps = step
    # start_step = int(jax.device_get(agent._train_state.step))
    # agent.training_steps = start_step
    start_step = agent.training_steps
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        info = agent.update()
        infos.append(info)

        if step % config.log_interval == 0:
            # Infos may have different keys (actor-only, critic-only, or both),
            # so we normalize them before stacking.
            all_keys = set().union(*(d.keys() for d in infos))
            nan = jnp.array(float("nan"))
            normalized = [{k: d.get(k, nan) for k in sorted(all_keys)} for d in infos]
            stacked_infos = common_utils.stack_forest(normalized)
            reduced_info = jax.device_get(jax.tree.map(jnp.nanmean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step + pretraining_steps}: {info_str}")
            wandb.log(reduced_info, step=step + pretraining_steps)
            infos = []

        if step % config.collect.collect_interval == 0:
            agent.save_checkpoint(step=step)
            collect_info, n_collected_episodes = collect_data(
                agent=agent,
                env=env,
                task_description=task_description,
                config=config,
                step=step,
            )
            wandb.log(collect_info, step=step + pretraining_steps)
            if n_collected_episodes > 0:
                logging.info(
                    f"Collected {n_collected_episodes} episodes at step {step}."
                )

        if step > 0 and step % config.collect.eval_interval == 0:
            eval_info = evaluate_policy(
                agent=agent,
                env=eval_env,
                task_description=eval_task_description,
                config=config,
                step=step,
            )
            wandb.log(eval_info, step=step + pretraining_steps)
            logging.info(
                f"Eval at step {step}: {', '.join(f'{k}={v:.4f}' for k, v in eval_info.items())}"
            )

        if (
                step % config.save_interval == 0 and step > start_step
        ) or step == config.num_train_steps - 1:
            agent.save_checkpoint(step=step)
    logging.info("Waiting for checkpoint manager to finish")
    agent._checkpoint_manager.wait_until_finished()