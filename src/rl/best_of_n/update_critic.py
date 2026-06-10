# ruff: noqa: F722
import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from src.training.config import OnlineTrainConfig, BestofNLearnerConfig
from src.rl.networks.rl_networks import (
    ObsType,
    ActionType,
    StateActionCritic,
    StateValue,
)
from src.rl.value_distribution import (
    get_value_bounds,
    make_value_distribution,
    make_bin_centers,
    categorical_project,
    reduce_ensemble_probs,
)
from src.rl.networks.bronet_critic import BroNetStateActionCritic, BroNetStateValue

# The critic may be either the MLP backbone (StateActionCritic / StateValue) or the
# BroNet backbone, depending on config.rl.critic.use_bronet. Both expose the same
# call signature and output convention, so the loss is architecture-agnostic; these
# unions just let @at.typecheck accept either backbone.
AnyStateActionCritic = StateActionCritic | BroNetStateActionCritic
AnyStateValue = StateValue | BroNetStateValue


CriticBatch = tuple[
    ObsType,
    _model.Actions,
    ObsType,
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],  # MC returns
]

StateActionCriticDef = Callable[[ObsType, ActionType, nnx.Rngs], StateActionCritic]
StateValueDef = Callable[[ObsType, nnx.Rngs], StateValue]


from src.rl.networks.encoders.encoders import MLPEncoder
from src.rl.networks.decoders.values.state_action_value import StateActionEnsembleDecoder
from src.rl.networks.decoders.values.state_value import StateValueEnsembleDecoder
from src.rl.networks.mlp import MLP
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME


def _build_pi0_backbone_critic_defs(config) -> tuple[StateActionCriticDef, StateValueDef]:
    # The distributional V-target is the Q distribution itself (and vice versa), which is
    # only valid when both critics share an identical categorical support.
    if config.rl.critic.use_distributional_critic:
        assert config.rl.critic.num_value_bins > 1, (
            "use_distributional_critic=True requires num_value_bins > 1."
        )
    critic_encoder_hidden_dims = config.rl.critic.encoder_hidden_dims
    critic_decoder_hidden_dims = config.rl.critic.decoder_hidden_dims
    critic_num_qs = config.rl.critic.num_qs
    critic_num_vs = config.rl.critic.num_vs

    def encoder_def(observation: ObsType, rngs: nnx.Rngs):
        network_def = lambda o, rg: MLP(
            input=o,
            hidden_dims=critic_encoder_hidden_dims,
            activate_final=True,
            rngs=rg,
        )
        state_vector_keys = ["state"]
        if isinstance(observation, dict) and PREFIX_EMBEDDING_NAME in observation:
            state_vector_keys = [PREFIX_EMBEDDING_NAME, "state"]
        return MLPEncoder(
            dummy_obs=observation,
            encoder_def=network_def,
            state_vector_keys=state_vector_keys,
            rngs=rngs,
        )

    def state_action_decoder_def(
        embedding: jax.Array, action: jax.Array, rngs: nnx.Rngs
    ) -> StateActionEnsembleDecoder:
        return StateActionEnsembleDecoder(
            observation=embedding,
            action=action,
            hidden_dims=critic_decoder_hidden_dims,
            num_qs=critic_num_qs,
            num_bins=config.rl.critic.num_value_bins,
            rngs=rngs,
        )

    def state_value_decoder_def(
        embedding: jax.Array, rngs: nnx.Rngs
    ) -> StateValueEnsembleDecoder:
        return StateValueEnsembleDecoder(
            observation=embedding,
            hidden_dims=critic_decoder_hidden_dims,
            num_vs=critic_num_vs,
            num_bins=config.rl.critic.num_value_bins,
            rngs=rngs,
        )

    def state_action_critic_def(
        observation: ObsType, action: jax.Array, rngs: nnx.Rngs
    ) -> StateActionCritic:
        return StateActionCritic(
            observation=observation,
            action=action,
            encoder_def=encoder_def,
            decoder_def=state_action_decoder_def,
            rngs=rngs,
        )

    def state_value_def(observation: ObsType, rngs: nnx.Rngs) -> StateValue:
        return StateValue(
            observation=observation,
            encoder_def=encoder_def,
            decoder_def=state_value_decoder_def,
            rngs=rngs,
        )

    return state_action_critic_def, state_value_def


def _use_ema_critic(config: OnlineTrainConfig) -> bool:
    return config.rl.critic.use_ema


def _critic_ema_decay(config: OnlineTrainConfig) -> float | None:
    return config.rl.critic.ema_decay


def create_critic(
    critic_state: training_utils.TrainState,
    config: OnlineTrainConfig,
) -> AnyStateActionCritic | AnyStateValue:
    critic_params = critic_state.params
    if critic_state.ema_params is not None and _use_ema_critic(config):
        critic_params = critic_state.ema_params
    return nnx.merge(critic_state.model_def, critic_params)


@at.typecheck
def _as_scalar_batch(values: at.ArrayLike) -> at.Float[at.Array, " b"]:
    values = jnp.asarray(values, dtype=jnp.float32)
    if values.ndim == 0:
        return values[jnp.newaxis]
    if values.ndim > 1:
        return values.reshape((values.shape[0], -1))[:, 0]
    return values


@at.typecheck
def summarize_critic_values(
    critic_logits: at.ArrayLike,
    config: OnlineTrainConfig,
    critic_reduction: str = "min",
) -> at.Float[at.Array, " b"]:
    assert isinstance(config.rl, BestofNLearnerConfig)
    lower, upper = get_value_bounds(config)
    dist = make_value_distribution(critic_logits, config.rl.critic.num_value_bins, lower, upper)
    expected_values = dist.mean()
    if expected_values.ndim > 1:
        # Take min across the ensemble members
        if critic_reduction == "min":
            expected_values = jnp.min(expected_values, axis=0)
        elif critic_reduction == "mean":
            expected_values = jnp.mean(expected_values, axis=0)
        else:
            raise NotImplementedError(
                f"Critic reduction {critic_reduction} is not implemented."
            )
    return _as_scalar_batch(expected_values)


@at.typecheck
def flatten_action_horizon(values: ActionType) -> at.Float[at.Array, "b a"]:
    return values.reshape((values.shape[0], -1))


def _ensure_rngs(rng: at.KeyArrayLike | nnx.Rngs) -> nnx.Rngs:
    if isinstance(rng, nnx.Rngs):
        return rng
    return nnx.Rngs(rng)


def init_state_action_critic_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    critic_def: StateActionCriticDef,
    dummy_obs: ObsType,
    dummy_act: ActionType,
) -> tuple[training_utils.TrainState, Any]:
    assert isinstance(config.rl, BestofNLearnerConfig)
    tx = _optimizer.create_optimizer(
        config.rl.critic.optimizer, config.rl.critic.lr_schedule, weight_decay_mask=None
    )
    ema_decay = _critic_ema_decay(config)
    # flatten the array across the array dim
    dummy_act = flatten_action_horizon(dummy_act)
    dummy_act = jax.tree.map(lambda x: x.reshape(*x.shape[:-1], -1), dummy_act)

    def init(obs, act, rng) -> training_utils.TrainState:
        critic = critic_def(obs, act, _ensure_rngs(rng))
        params = nnx.state(critic)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(critic),
            tx=tx,
            opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
            ema_decay=ema_decay,
            ema_params=None if ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, dummy_obs, dummy_act, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(dummy_obs, dummy_act, init_rng)
    return train_state, state_sharding


def init_state_value_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    critic_def: StateValueDef,
    dummy_obs: ObsType,
) -> tuple[training_utils.TrainState, Any]:
    assert isinstance(config.rl, BestofNLearnerConfig)
    tx = _optimizer.create_optimizer(
        config.rl.critic.optimizer, config.rl.critic.lr_schedule, weight_decay_mask=None
    )
    ema_decay = _critic_ema_decay(config)

    def init(obs, rng) -> training_utils.TrainState:
        critic = critic_def(obs, _ensure_rngs(rng))
        params = nnx.state(critic)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(critic),
            tx=tx,
            opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
            ema_decay=ema_decay,
            ema_params=None if ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, dummy_obs, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(dummy_obs, init_rng)
    return train_state, state_sharding


def _update_train_state(
    state: training_utils.TrainState,
    model: nnx.Module,
    grads: nnx.State,
) -> training_utils.TrainState:
    params = nnx.filter_state(state.params, nnx.Param)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    if state.ema_decay is not None and state.ema_params is not None:
        # Only EMA nnx.Param leaves. BatchStat (BatchNorm running stats) and RngState
        # (PRNG keys) are non-arithmetic — pass them through as-is from new_params.
        param_keys = set(nnx.filter_state(new_params, nnx.Param).flat_state())
        old_flat = dict(state.ema_params.flat_state())
        def _ema_or_copy(path, new_val):
            if path in param_keys:
                return state.ema_decay * old_flat[path] + (1 - state.ema_decay) * new_val
            return new_val
        new_state = dataclasses.replace(
            new_state,
            ema_params=new_params.map(_ema_or_copy),
        )
    return new_state


def _kernel_param_norm(model: nnx.Module) -> at.Float[at.Array, ""]:
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(
                nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")
            ),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return optax.global_norm(kernel_params)


@at.typecheck
def train_q_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    q_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    q_model = nnx.merge(q_state.model_def, q_state.params)
    q_model.train()
    value_model = create_critic(value_state, config)
    value_model.eval()
    assert isinstance(config.rl, BestofNLearnerConfig)

    step = q_state.step // config.rl.critic.num_updates_per_batch
    observation, actions, next_observation, reward, discount, mc_return = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    mc_return = _as_scalar_batch(mc_return)
    actions = flatten_action_horizon(actions)

    @at.typecheck
    def loss_fn(
        critic_model: AnyStateActionCritic,
        observation: ObsType,
        actions: _model.Actions,
        next_observation: ObsType,
        reward: at.Float[at.ArrayLike, " b"],
        discount: at.Float[at.ArrayLike, " b"],
        mc_return: at.Float[at.ArrayLike, " b"],
        target_value_model: AnyStateValue,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        # Architecture-agnostic: critic_model is either the MLP backbone or BroNet;
        # both emit scalar (n, b) logits when num_value_bins == 1 and categorical
        # (n, b, k) logits when num_value_bins > 1, so the loss only branches on
        # use_distributional_critic, never on the network type.
        q_logits = critic_model(observation, actions)
        _lower, _upper = get_value_bounds(config)
        num_bins = config.rl.critic.num_value_bins
        q_dist = make_value_distribution(q_logits, num_bins, _lower, _upper, config.rl.critic.value_target_type)
        td_weight = config.rl.critic.td_weight_schedule.create()(step)
        td_weight = jnp.clip(td_weight, 0.0, 1.0)

        if config.rl.critic.use_distributional_critic:
            # Distributional Bellman backup: project V(s')'s distribution under Tz = r + gamma * z.
            centers = make_bin_centers(_lower, _upper, num_bins)
            next_probs = reduce_ensemble_probs(
                target_value_model(next_observation),
                config.rl.critic.distributional_target_reduction,
                centers,
            )
            target_probs = jax.lax.stop_gradient(
                categorical_project(next_probs, reward, discount, centers)
            )
            q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
            td_loss = -jnp.mean(jnp.sum(target_probs[jnp.newaxis] * q_logprobs, axis=-1))
        else:
            bootstrapped_values = summarize_critic_values(
                target_value_model(next_observation),
                config,
                critic_reduction=config.rl.critic.reduction,
            )
            td_targets = reward + discount * jax.lax.stop_gradient(bootstrapped_values)
            td_loss = -jnp.mean(q_dist.log_prob(td_targets))

        mc_loss = -jnp.mean(q_dist.log_prob(mc_return))
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        value_mean = jnp.mean(q_dist.mean())
        return loss, {
            "value_mean": value_mean,
            "td_loss": td_loss,
            "mc_loss": mc_loss,
            "td_weight": td_weight,
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        q_model,
        observation,
        actions,
        next_observation,
        reward,
        discount,
        mc_return,
        value_model,
    )
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
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    assert isinstance(config.rl, BestofNLearnerConfig)
    step = value_state.step // config.rl.critic.num_updates_per_batch
    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()

    q_model = create_critic(q_state, config)
    q_model.eval()

    observation, actions, _, _, _, mc_return = batch
    actions = flatten_action_horizon(actions)
    mc_return = _as_scalar_batch(mc_return)

    @at.typecheck
    def loss_fn(
        critic_model: AnyStateValue,
        observation: ObsType,
        actions: _model.Actions,
        mc_return: at.Float[at.ArrayLike, " b"],
        target_q_model: AnyStateActionCritic,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        td_weight = config.rl.critic.td_weight_schedule.create()(step)
        td_weight = jnp.clip(td_weight, 0.0, 1.0)
        value_logits = critic_model(observation)
        _lower, _upper = get_value_bounds(config)
        num_bins = config.rl.critic.num_value_bins
        v_dist = make_value_distribution(value_logits, num_bins, _lower, _upper, config.rl.critic.value_target_type)

        if config.rl.critic.use_distributional_critic:
            # V regresses Q directly (identity Bellman map): target = reduced Q(s,a) distribution.
            centers = make_bin_centers(_lower, _upper, num_bins)
            target_probs = jax.lax.stop_gradient(
                reduce_ensemble_probs(
                    target_q_model(observation, actions),
                    config.rl.critic.distributional_target_reduction,
                    centers,
                )
            )
            v_logprobs = jax.nn.log_softmax(value_logits, axis=-1)
            td_loss = -jnp.mean(jnp.sum(target_probs[jnp.newaxis] * v_logprobs, axis=-1))
        else:
            q_values = summarize_critic_values(
                target_q_model(observation, actions),
                config,
                critic_reduction=config.rl.critic.reduction,
            )
            td_loss = -jnp.mean(v_dist.log_prob(jax.lax.stop_gradient(q_values)))

        mc_loss = -jnp.mean(v_dist.log_prob(mc_return))
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        value_mean = jnp.mean(v_dist.mean())
        return loss, {
            "value_mean": value_mean,
            "td_loss": td_loss,
            "mc_loss": mc_loss,
            "td_weight": td_weight,
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(
        value_model,
        observation,
        actions,
        mc_return,
        q_model,
    )
    new_state = _update_train_state(value_state, value_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(value_model),
    } | aux_data
    return new_state, info
