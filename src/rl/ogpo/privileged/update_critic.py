# ruff: noqa: F722
"""Critic train steps for the ``privileged_backup="next_action_q"`` ABLATION.

Not the default path. ``OGPOPrivilegedLearner`` runs the shared
``advantage_weighted_sft/update_critic`` steps unless this backup is selected,
so the default privileged arm differs from the baseline arm only in what the
critic sees, not in how it is trained.

When selected, these steps differ from the shared ones in exactly one place —
how the Q target bootstraps — and carry one extra batch element to make it
possible:

    baseline / default  Q(s, a) <- r + gamma * V(s')       (V is itself fit to
                                                            Q(s, a_buffer): the
                                                            SARSA-flavoured backup)
    next_action_q       Q(s, a) <- r + gamma * Q_ema(s', a')
                                                           (a' = the action the
                                                            policy actually took
                                                            at s', read out of the
                                                            stored trajectory at
                                                            collection — no policy
                                                            rollout at TD time)

Everything else — the TD/MC blend and its schedule, the value distribution, the
ensemble reduction, the optimizer/EMA update, the logged keys — is the shared
implementation, imported rather than copied, so the two stacks cannot drift.

``train_value_step`` still trains V (the non-GRPO advantage combinations and the
q/v logging read it); under this backup it simply no longer feeds the Q target.
"""
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.utils as training_utils
from src.rl.advantage_weighted_sft.update_critic import (
    _as_scalar_batch,
    _kernel_param_norm,
    _update_train_state,
    create_critic,
    critic_values_per_head,
    flatten_action_horizon,
    summarize_critic_values,
)
from src.rl.networks.rl_networks import ObsType
from src.rl.value_distribution import get_value_bounds, make_value_distribution
from src.training.config import OnlineTrainConfig, OGPOPrivilegedLearnerConfig


# (obs, actions, next_obs, NEXT actions, reward, discount, mc_return). The
# fourth element is the addition over the shared ``CriticBatch``.
PrivilegedCriticBatch = tuple[
    ObsType,
    _model.Actions,
    ObsType,
    _model.Actions,
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],
]


@at.typecheck
def train_q_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    q_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: PrivilegedCriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    rl = config.rl
    assert isinstance(rl, OGPOPrivilegedLearnerConfig)

    q_model = nnx.merge(q_state.model_def, q_state.params)
    q_model.train()

    step = q_state.step // rl.critic.num_updates_per_batch
    observation, actions, next_observation, next_actions, reward, discount, mc_return = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    mc_return = _as_scalar_batch(mc_return)
    actions = flatten_action_horizon(actions)

    backup_reduction = rl.critic.q_bootstrap_reduction or rl.critic.reduction
    if rl.privileged_backup == "next_action_q":
        # Target network: create_critic returns the EMA params when
        # critic.use_ema, so the bootstrap does not chase the live weights.
        target_q_model = create_critic(q_state, config)
        target_q_model.eval()
        bootstrap_target = summarize_critic_values(
            target_q_model(next_observation, flatten_action_horizon(next_actions)),
            config,
            critic_reduction=backup_reduction,
        )
    else:
        value_model = create_critic(value_state, config)
        value_model.eval()
        bootstrap_target = summarize_critic_values(
            value_model(next_observation), config, critic_reduction=backup_reduction
        )

    def loss_fn(q_model, observation, actions):
        td_weight = jnp.clip(rl.critic.td_weight_schedule.create()(step), 0.0, 1.0)
        q_logits = q_model(observation, actions)
        td_targets = reward + discount * jax.lax.stop_gradient(bootstrap_target)
        _lower, _upper = get_value_bounds(config)
        q_dist = make_value_distribution(
            q_logits, rl.critic.num_value_bins, _lower, _upper, rl.critic.value_target_type
        )
        td_loss = -jnp.mean(q_dist.log_prob(td_targets))
        mc_loss = -jnp.mean(q_dist.log_prob(mc_return))
        value_mean = jnp.mean(q_dist.mean())
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        # Ranking-quality proxy: Pearson corr between the (head-mean) Q
        # prediction and the observed MC return over this batch. This is THE
        # number the privileged critic is meant to move — the advantage only
        # ever reads the critic's ORDERING of actions.
        q_pred = jnp.mean(q_dist.mean(), axis=0) if q_dist.mean().ndim > 1 else q_dist.mean()
        qc = q_pred - jnp.mean(q_pred)
        mc = mc_return - jnp.mean(mc_return)
        q_mc_corr = jnp.sum(qc * mc) / jnp.maximum(
            jnp.linalg.norm(qc) * jnp.linalg.norm(mc), 1e-8
        )
        return loss, {
            "value_mean": value_mean,
            "td_loss": td_loss,
            "mc_loss": mc_loss,
            "td_weight": td_weight,
            "mc_corr": q_mc_corr,
            # Scale of the thing being bootstrapped through. Reading it next to
            # value_mean is how a collapsed backup (target pinned to the
            # never-succeeding fixed point) shows up.
            "bootstrap_mean": jnp.mean(bootstrap_target),
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(q_model, observation, actions)
    new_state = _update_train_state(q_state, q_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(q_model),
    } | aux_data
    return new_state, info


@at.typecheck
def train_value_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    value_state: training_utils.TrainState,
    q_state: training_utils.TrainState,
    batch: PrivilegedCriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """V(s) regressed onto Q(s, a_buffer) / the MC return — unchanged from the
    baseline apart from unpacking the 7-element privileged batch."""
    del rng
    rl = config.rl
    assert isinstance(rl, OGPOPrivilegedLearnerConfig)
    step = value_state.step // rl.critic.num_updates_per_batch
    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()

    q_model = create_critic(q_state, config)
    q_model.eval()

    observation, actions, _, _, _, _, mc_return = batch
    actions = flatten_action_horizon(actions)
    mc_return = _as_scalar_batch(mc_return)

    if rl.critic.per_critic_value_target:
        assert rl.critic.num_vs == rl.critic.num_qs, "per_critic_value_target needs num_vs == num_qs"
        bootstrap_target = critic_values_per_head(q_model(observation, actions), config)
    else:
        bootstrap_target = summarize_critic_values(
            q_model(observation, actions), config, critic_reduction=rl.critic.reduction
        )

    def loss_fn(value_model, observation):
        td_weight = jnp.clip(rl.critic.td_weight_schedule.create()(step), 0.0, 1.0)
        value_logits = value_model(observation)
        _lower, _upper = get_value_bounds(config)
        v_dist = make_value_distribution(
            value_logits, rl.critic.num_value_bins, _lower, _upper, rl.critic.value_target_type
        )
        mc_loss = -jnp.mean(v_dist.log_prob(mc_return))
        td_loss = -jnp.mean(v_dist.log_prob(jax.lax.stop_gradient(bootstrap_target)))
        value_mean = jnp.mean(v_dist.mean())
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        return loss, {
            "value_mean": value_mean,
            "td_loss": td_loss,
            "mc_loss": mc_loss,
            "td_weight": td_weight,
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(value_model, observation)
    new_state = _update_train_state(value_state, value_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(value_model),
    } | aux_data
    return new_state, info
