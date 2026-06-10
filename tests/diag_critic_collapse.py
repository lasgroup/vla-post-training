"""Confirm the zero-loss/zero-grad collapse is caused by wrong value bounds.

Run:
    uv run python tests/diag_critic_collapse.py

Trains the real Q-critic for a few hundred steps on synthetic time-to-success
data, once with the BUGGY bounds [0, 1] and once with the CORRECT negative
bounds. With the buggy bounds every target falls in bin 0, the categorical
head collapses to a point mass there, and loss + grad_norm -> 0 (your wandb
symptom). With correct bounds the loss stays healthy and non-zero.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

import openpi.training.utils as training_utils
import openpi.training.optimizer as _optimizer

from src.training.config import get_config, StepSchedule
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.rl.best_of_n.update_critic import _build_pi0_backbone_critic_defs, train_q_step


def build_cfg(lower, upper):
    base = get_config("pi05_libero_online_best_of_n")
    crit = dataclasses.replace(
        base.rl.critic,
        use_distributional_critic=True,
        num_value_bins=51,
        distributional_target_reduction="mean",
        value_lower_bound=lower,
        value_upper_bound=upper,
        td_weight_schedule=StepSchedule(init_value=0.5, end_value=0.5, switch_step=500000),
    )
    rl = dataclasses.replace(base.rl, critic=crit, discount=0.995)
    collect = dataclasses.replace(base.collect, use_time_to_success_as_reward=True)
    return dataclasses.replace(base, rl=rl, collect=collect, batch_size=1024)


def make_state(critic, cfg):
    params = nnx.state(critic)
    tx = _optimizer.create_optimizer(cfg.rl.critic.optimizer, cfg.rl.critic.lr_schedule, weight_decay_mask=None)
    return training_utils.TrainState(
        step=0, params=params, model_def=nnx.graphdef(critic), tx=tx,
        opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
        ema_decay=cfg.rl.critic.ema_decay, ema_params=params,
    )


def run(lower, upper, label):
    cfg = build_cfg(lower, upper)
    b, sd, ed, fa = 16, 8, 16, 70
    obs = {"state": jax.random.normal(jax.random.key(1), (b, sd)),
           PREFIX_EMBEDDING_NAME: jax.random.normal(jax.random.key(2), (b, ed))}
    next_obs = {"state": jax.random.normal(jax.random.key(3), (b, sd)),
                PREFIX_EMBEDDING_NAME: jax.random.normal(jax.random.key(4), (b, ed))}
    actions = jax.random.normal(jax.random.key(5), (b, 10, 7))
    reward = -jnp.ones((b,))
    discount = jnp.full((b,), 0.995)
    mc_return = -jax.random.uniform(jax.random.key(6), (b,)) * 100.0

    sa_def, sv_def = _build_pi0_backbone_critic_defs(cfg)
    q_model = sa_def(obs, jnp.zeros((b, fa)), nnx.Rngs(0))
    v_model = sv_def(obs, nnx.Rngs(1))
    q_state = make_state(q_model, cfg)
    v_state = make_state(v_model, cfg)
    batch = (obs, actions, next_obs, reward, discount, mc_return)

    step_q = jax.jit(lambda qs: train_q_step(cfg, jax.random.key(0), qs, v_state, batch))
    print(f"\n=== {label}  bounds=({lower:.2f}, {upper:.2f}) ===")
    print(f"{'step':>5} {'loss':>12} {'mc_loss':>12} {'td_loss':>12} {'grad_norm':>12}")
    for i in range(601):
        q_state, info = step_q(q_state)
        if i % 100 == 0:
            print(f"{i:5d} {float(info['loss']):12.4e} {float(info['mc_loss']):12.4e} "
                  f"{float(info['td_loss']):12.4e} {float(info['grad_norm']):12.4e}")


def main():
    run(0.0, 1.0, "BUGGY (current behaviour)")
    run(-174.79, 1.73, "CORRECT (time-to-success range)")


if __name__ == "__main__":
    main()
