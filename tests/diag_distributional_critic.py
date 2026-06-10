"""Diagnostic for the zero value-loss / zero grad-norm bug in the distributional critic.

Run on a compute node:
    uv run python tests/diag_distributional_critic.py

It does two things:
  (A) Rebuilds the config the way the launcher + tyro would (start from the
      registered `pi05_libero_online_best_of_n` instance, then apply the YAML
      overrides via dataclasses.replace, which re-runs __post_init__), and prints
      the RESOLVED critic value bounds / num_bins / td_weight. This catches the
      config-resolution bug.
  (B) Builds tiny Q and V critics + a synthetic batch and calls the REAL
      train_q_step / train_value_step, printing the returned info dicts and a
      manual recomputation of the intermediate tensors (target_probs, logits,
      grad norm).
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
from src.rl.best_of_n.update_critic import (
    _build_pi0_backbone_critic_defs,
    train_q_step,
    train_value_step,
)
from src.rl.value_distribution import (
    get_value_bounds,
    make_value_distribution,
    make_bin_centers,
    categorical_project,
    reduce_ensemble_probs,
)


def build_config():
    base = get_config("pi05_libero_online_best_of_n")
    crit = dataclasses.replace(
        base.rl.critic,
        use_distributional_critic=True,
        num_value_bins=51,
        distributional_target_reduction="mean",
        inference_start_step=0,
        pre_training_steps=0,
        td_weight_schedule=StepSchedule(init_value=0.5, end_value=0.5, switch_step=500000),
    )
    rl = dataclasses.replace(base.rl, critic=crit, discount=0.995, buffer_capacity=250000)
    collect = dataclasses.replace(
        base.collect, use_time_to_success_as_reward=True, store_prefix_rep=True
    )
    cfg = dataclasses.replace(base, rl=rl, collect=collect, batch_size=1024)
    return cfg


def manual_build_state(critic, cfg):
    params = nnx.state(critic)
    tx = _optimizer.create_optimizer(
        cfg.rl.critic.optimizer, cfg.rl.critic.lr_schedule, weight_decay_mask=None
    )
    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(critic),
        tx=tx,
        opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
        ema_decay=cfg.rl.critic.ema_decay,
        ema_params=params,
    )


def main():
    cfg = build_config()

    print("=" * 70)
    print("(A) RESOLVED CONFIG")
    print("=" * 70)
    lo, hi = get_value_bounds(cfg)
    print(f"use_distributional_critic = {cfg.rl.critic.use_distributional_critic}")
    print(f"num_value_bins            = {cfg.rl.critic.num_value_bins}")
    print(f"distributional_reduction  = {cfg.rl.critic.distributional_target_reduction}")
    print(f"value_lower_bound         = {cfg.rl.critic.value_lower_bound}")
    print(f"value_upper_bound         = {cfg.rl.critic.value_upper_bound}")
    print(f"get_value_bounds()        = ({lo}, {hi})")
    print(f"use_time_to_success       = {cfg.collect.use_time_to_success_as_reward}")
    print(f"discount                  = {cfg.rl.discount}")
    print(f"td_weight @step0          = {float(cfg.rl.critic.td_weight_schedule.create()(0))}")
    bw = (hi - lo) / (cfg.rl.critic.num_value_bins - 1)
    print(f"bin width                 = {bw}")
    print(f"-> bin centers span [{lo}, {hi}]; rewards/returns are NEGATIVE "
          f"(time-to-success). If the span is [~0, ~1] the bounds are WRONG.")

    print()
    print("=" * 70)
    print("(B) REAL train_q_step / train_value_step ON SYNTHETIC DATA")
    print("=" * 70)

    b = 16
    state_dim = 8
    embed_dim = 16
    horizon, act_dim = 10, 7
    flat_act = horizon * act_dim

    rng = jax.random.key(0)
    obs = {
        "state": jax.random.normal(jax.random.key(1), (b, state_dim)),
        PREFIX_EMBEDDING_NAME: jax.random.normal(jax.random.key(2), (b, embed_dim)),
    }
    next_obs = {
        "state": jax.random.normal(jax.random.key(3), (b, state_dim)),
        PREFIX_EMBEDDING_NAME: jax.random.normal(jax.random.key(4), (b, embed_dim)),
    }
    actions = jax.random.normal(jax.random.key(5), (b, horizon, act_dim))
    # time-to-success style: negative per-step reward, mc_return strongly negative
    reward = -jnp.ones((b,))
    discount = jnp.full((b,), 0.995)
    mc_return = -jax.random.uniform(jax.random.key(6), (b,)) * 100.0

    sa_def, sv_def = _build_pi0_backbone_critic_defs(cfg)
    # Build critic with already-flat action so init dim matches the loss path.
    flat_actions_init = jnp.zeros((b, flat_act))
    q_model = sa_def(obs, flat_actions_init, nnx.Rngs(0))
    v_model = sv_def(obs, nnx.Rngs(1))

    q_state = manual_build_state(q_model, cfg)
    v_state = manual_build_state(v_model, cfg)

    batch = (obs, actions, next_obs, reward, discount, mc_return)

    new_q_state, q_info = train_q_step(cfg, rng, q_state, v_state, batch)
    new_v_state, v_info = train_value_step(cfg, rng, v_state, q_state, batch)

    print("\n-- q_info --")
    for k, v in q_info.items():
        print(f"  {k:12s} = {np.asarray(v)}")
    print("\n-- v_info --")
    for k, v in v_info.items():
        print(f"  {k:12s} = {np.asarray(v)}")

    # Manual recomputation of the q target to inspect target_probs.
    print("\n-- manual intermediates (q) --")
    centers = make_bin_centers(lo, hi, cfg.rl.critic.num_value_bins)
    q_logits = q_model(obs, flat_actions_init)
    print(f"  q_logits shape        = {q_logits.shape}")
    print(f"  q_logits std          = {float(jnp.std(q_logits)):.6e}")
    next_probs = reduce_ensemble_probs(
        v_model(next_obs), cfg.rl.critic.distributional_target_reduction, centers
    )
    target_probs = categorical_project(next_probs, reward, discount, centers)
    print(f"  next_probs sum (row0) = {float(next_probs[0].sum()):.6f}")
    print(f"  target_probs sum row0 = {float(target_probs[0].sum()):.6f}")
    print(f"  target_probs argmax   = {np.asarray(jnp.argmax(target_probs, -1))}")
    dist = make_value_distribution(q_logits, cfg.rl.critic.num_value_bins, lo, hi, "one_hot")
    print(f"  mc bin idx (discretized targets) = "
          f"{np.asarray(jnp.argmax(jax.nn.one_hot(jnp.clip(((mc_return - lo) * (cfg.rl.critic.num_value_bins-1) / (hi-lo)).round().astype(int), 0, cfg.rl.critic.num_value_bins-1), cfg.rl.critic.num_value_bins), -1))}")
    print(f"  q value_mean (pred)   = {np.asarray(dist.mean()).mean():.4f}")


if __name__ == "__main__":
    main()
