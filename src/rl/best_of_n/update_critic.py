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
from src.rl.value_distribution import get_value_bounds, make_value_distribution
from src.rl.networks.rl_networks import (
    ObsType,
    ActionType,
    StateActionCritic,
    StateValue,
)
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME

CriticBatch = tuple[
    ObsType,
    _model.Actions,
    ObsType,
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],
    at.Float[at.Array, " b"],       # MC returns
]

StateActionCriticDef = Callable[[ObsType, ActionType, nnx.Rngs], StateActionCritic]
StateValueDef = Callable[[ObsType, nnx.Rngs], StateValue]


def _use_ema_critic(config: OnlineTrainConfig) -> bool:
    assert isinstance(config.rl, BestofNLearnerConfig)
    return config.rl.use_ema_critic


def _critic_ema_decay(config: OnlineTrainConfig) -> float | None:
    assert isinstance(config.rl, BestofNLearnerConfig)
    return config.rl.critic_ema_decay


def create_critic(
    critic_state: training_utils.TrainState,
    config: OnlineTrainConfig,
) -> StateActionCritic | StateValue:
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
    lower, upper = get_value_bounds(config)
    dist = make_value_distribution(critic_logits, config.rl.num_value_bins, lower, upper)
    # expected_values: (num_heads, batch) for ensemble, (batch,) for single head.
    # For Gaussian: mean() = logits. For Categorical: mean() = E[return] via softmax.
    expected_values = dist.mean()
    if expected_values.ndim > 1:
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
        config.rl.critic_optimizer, config.rl.critic_lr_schedule, weight_decay_mask=None
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

    # Initialize eagerly on CPU so that _orthogonal_cpu (used by MLP's default_init)
    # actually runs on CPU and avoids the gpusolverDnCreate failure on GH200/Hopper.
    # jax.default_device(cpu) inside jax.jit has no effect on XLA device placement,
    # so the JIT path always sends linalg.qr to GPU, which fails on this cluster.
    cpu = jax.devices("cpu")[0]
    train_state = init(
        jax.device_put(dummy_obs, cpu),
        jax.device_put(dummy_act, cpu),
        jax.device_put(init_rng, cpu),
    )
    # Derive sharding from the exact initialized tree to keep GraphDef metadata
    # (including function-valued statics) aligned with the sharding pytree.
    state_sharding = sharding.fsdp_sharding(train_state, mesh, log=False)
    train_state = jax.device_put(train_state, state_sharding)
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
        config.rl.critic_optimizer, config.rl.critic_lr_schedule, weight_decay_mask=None
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

    cpu = jax.devices("cpu")[0]
    train_state = init(
        jax.device_put(dummy_obs, cpu),
        jax.device_put(init_rng, cpu),
    )
    # Derive sharding from the exact initialized tree to keep GraphDef metadata
    # (including function-valued statics) aligned with the sharding pytree.
    state_sharding = sharding.fsdp_sharding(train_state, mesh, log=False)
    train_state = jax.device_put(train_state, state_sharding)
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
    step = (q_state.step // config.rl.num_critic_updates_per_batch)
    observation, actions, next_observation, reward, discount, mc_return = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    mc_return = _as_scalar_batch(mc_return)
    actions = flatten_action_horizon(actions)
    assert isinstance(config.rl, BestofNLearnerConfig)

    @at.typecheck
    def loss_fn(
        critic_model: StateActionCritic,
        observation: ObsType,
        actions: _model.Actions,
        next_observation: ObsType,
        reward: at.Float[at.ArrayLike, " b"],
        discount: at.Float[at.ArrayLike, " b"],
        mc_return: at.Float[at.ArrayLike, " b"],
        target_value_model: StateValue,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        q_logits = critic_model(observation, actions)
        bootstrapped_values = summarize_critic_values(
            target_value_model(next_observation),
            config,
            critic_reduction=config.rl.critic_reduction,
        )
        td_targets = reward + discount * jax.lax.stop_gradient(bootstrapped_values)
        _lower, _upper = get_value_bounds(config)
        q_dist = make_value_distribution(q_logits, config.rl.num_value_bins, _lower, _upper, config.rl.value_target_type)
        td_weight = config.rl.td_weight_schedule.create()(step)
        td_weight = jnp.clip(td_weight, 0.0, 1.0)
        td_loss = -jnp.mean(q_dist.log_prob(td_targets))
        mc_loss = -jnp.mean(q_dist.log_prob(mc_return))
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        return loss, {
            "value_mean": jnp.mean(q_dist.mean()),
            "mc_loss": mc_loss,
            "td_loss": td_loss,
            # "td_target_mean": jnp.mean(td_targets),
            # "mc_error_mean": jnp.mean(mc_errors),
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
    step = (value_state.step // config.rl.num_critic_updates_per_batch)
    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()

    q_model = create_critic(q_state, config)
    q_model.eval()

    observation, actions, _, _, _, mc_return = batch

    actions = flatten_action_horizon(actions)
    mc_return = _as_scalar_batch(mc_return)

    @at.typecheck
    def loss_fn(
        critic_model: StateValue,
        observation: ObsType,
        actions: _model.Actions,
        mc_return: at.Float[at.ArrayLike, " b"],
        target_q_model: StateActionCritic,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        td_weight = config.rl.td_weight_schedule.create()(step)
        td_weight = jnp.clip(td_weight, 0.0, 1.0)
        value_logits = critic_model(observation)
        q_values = summarize_critic_values(
            target_q_model(observation, actions),
            config,
            critic_reduction=config.rl.critic_reduction,
        )
        _lower, _upper = get_value_bounds(config)
        v_dist = make_value_distribution(value_logits, config.rl.num_value_bins, _lower, _upper, config.rl.value_target_type)
        mc_loss = -jnp.mean(v_dist.log_prob(mc_return))
        td_loss = -jnp.mean(v_dist.log_prob(jax.lax.stop_gradient(q_values)))
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        return loss, {
            "value_mean": jnp.mean(v_dist.mean()),
            "mc_loss": mc_loss,
            "td_loss": td_loss,
            "td_weight": td_weight,
        }

    diff_state = nnx.DiffState(0, nnx.Param)
    (loss, aux_data), grads = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=diff_state
    )(value_model, observation, actions, mc_return, q_model)
    new_state = _update_train_state(value_state, value_model, grads)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": _kernel_param_norm(value_model),
    } | aux_data
    return new_state, info


# ---------------------------------------------------------------------------
# Variants that finetune the Pi0 prefix encoder jointly with the critics.
# When train_pi0_prefix_encoder=False these are never called.
# ---------------------------------------------------------------------------

# Raw-observation batch: observations are full buffer dicts (images + state +
# tokenized prompt), not pre-processed CriticBatch tuples.
RawCriticBatch = tuple[
    dict,                                    # raw observation
    _model.Actions,
    dict,                                    # raw next_observation
    at.Float[at.Array, " b"],               # reward
    at.Float[at.Array, " b"],               # discount
    at.Float[at.Array, " b"],               # mc_return
]


def _compute_prefix(
    pi0_model: _model.BaseModel,
    raw_obs: dict,
) -> jax.Array:
    """Run the Pi0 VLM prefix forward pass and mean-pool tokens. Differentiable."""
    obs_obj = _model.Observation.from_dict(raw_obs)
    prefix = pi0_model.get_prefix_rep(obs_obj)
    if isinstance(prefix, tuple):
        prefix = prefix[0]
    prefix = prefix.reshape(prefix.shape[0], -1, prefix.shape[-1])
    return jnp.mean(prefix, axis=1)


@at.typecheck
def train_q_step_with_encoder(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    q_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    pi0_encoder_state: training_utils.TrainState,
    batch: RawCriticBatch,
) -> tuple[training_utils.TrainState, training_utils.TrainState, dict[str, at.Array]]:
    """Q update that also backpropagates into the Pi0 prefix encoder."""
    del rng
    q_model = nnx.merge(q_state.model_def, q_state.params)
    q_model.train()
    pi0_model = nnx.merge(pi0_encoder_state.model_def, pi0_encoder_state.params)
    pi0_model.eval()
    value_model = create_critic(value_state, config)
    value_model.eval()
    assert isinstance(config.rl, BestofNLearnerConfig)
    step = q_state.step // config.rl.num_critic_updates_per_batch

    raw_obs, actions, raw_next_obs, reward, discount, mc_return = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    mc_return = _as_scalar_batch(mc_return)
    actions = flatten_action_horizon(actions)

    assert isinstance(config.rl, BestofNLearnerConfig)
    def loss_fn(
        critic_model: StateActionCritic,
        pi0_enc: _model.BaseModel,
        raw_obs: dict,
        raw_next_obs: dict,
        actions: _model.Actions,
        reward: at.Float[at.ArrayLike, " b"],
        discount: at.Float[at.ArrayLike, " b"],
        mc_return: at.Float[at.ArrayLike, " b"],
        target_value_model: StateValue,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        prefix = _compute_prefix(pi0_enc, raw_obs)
        next_prefix = _compute_prefix(pi0_enc, raw_next_obs)
        observation = {"state": raw_obs["state"], PREFIX_EMBEDDING_NAME: prefix}
        next_observation = {"state": raw_next_obs["state"], PREFIX_EMBEDDING_NAME: next_prefix}

        q_logits = critic_model(observation, actions)
        bootstrapped_values = summarize_critic_values(
            target_value_model(next_observation),
            config,
            critic_reduction=config.rl.critic_reduction,
        )
        td_targets = reward + discount * jax.lax.stop_gradient(bootstrapped_values)
        _lower, _upper = get_value_bounds(config)
        q_dist = make_value_distribution(
            q_logits, config.rl.num_value_bins, _lower, _upper, config.rl.value_target_type
        )
        td_weight = config.rl.td_weight_schedule.create()(step)
        td_weight = jnp.clip(td_weight, 0.0, 1.0)
        td_loss = -jnp.mean(q_dist.log_prob(td_targets))
        mc_loss = -jnp.mean(q_dist.log_prob(mc_return))
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        return loss, {
            "value_mean": jnp.mean(q_dist.mean()),
            "mc_loss": mc_loss,
            "td_loss": td_loss,
            "td_weight": td_weight,
        }

    diff_state_q = nnx.DiffState(0, nnx.Param)
    diff_state_pi0 = nnx.DiffState(1, nnx.Param)
    (loss, aux_data), (grads_q, grads_pi0) = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=(diff_state_q, diff_state_pi0)
    )(q_model, pi0_model, raw_obs, raw_next_obs, actions, reward, discount, mc_return, value_model)

    new_q_state = _update_train_state(q_state, q_model, grads_q)
    new_pi0_state = _update_train_state(pi0_encoder_state, pi0_model, grads_pi0)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads_q),
        "param_norm": _kernel_param_norm(q_model),
        "encoder_grad_norm": optax.global_norm(grads_pi0),
    } | aux_data
    return new_q_state, new_pi0_state, info


@at.typecheck
def train_value_step_with_encoder(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    value_state: training_utils.TrainState,
    q_state: training_utils.TrainState,
    pi0_encoder_state: training_utils.TrainState,
    batch: RawCriticBatch,
) -> tuple[training_utils.TrainState, training_utils.TrainState, dict[str, at.Array]]:
    """V update that also backpropagates into the Pi0 prefix encoder."""
    del rng
    assert isinstance(config.rl, BestofNLearnerConfig)
    step = value_state.step // config.rl.num_critic_updates_per_batch
    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()
    pi0_model = nnx.merge(pi0_encoder_state.model_def, pi0_encoder_state.params)
    pi0_model.eval()
    q_model = create_critic(q_state, config)
    q_model.eval()

    raw_obs, actions, _, _, _, mc_return = batch
    actions = flatten_action_horizon(actions)
    mc_return = _as_scalar_batch(mc_return)

    assert isinstance(config.rl, BestofNLearnerConfig)
    def loss_fn(
        critic_model: StateValue,
        pi0_enc: _model.BaseModel,
        raw_obs: dict,
        actions: _model.Actions,
        mc_return: at.Float[at.ArrayLike, " b"],
        target_q_model: StateActionCritic,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Array]]:
        prefix = _compute_prefix(pi0_enc, raw_obs)
        observation = {"state": raw_obs["state"], PREFIX_EMBEDDING_NAME: prefix}

        td_weight = config.rl.td_weight_schedule.create()(step)
        td_weight = jnp.clip(td_weight, 0.0, 1.0)
        value_logits = critic_model(observation)
        q_values = summarize_critic_values(
            target_q_model(observation, actions),
            config,
            critic_reduction=config.rl.critic_reduction,
        )
        _lower, _upper = get_value_bounds(config)
        v_dist = make_value_distribution(
            value_logits, config.rl.num_value_bins, _lower, _upper, config.rl.value_target_type
        )
        mc_loss = -jnp.mean(v_dist.log_prob(mc_return))
        td_loss = -jnp.mean(v_dist.log_prob(jax.lax.stop_gradient(q_values)))
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        return loss, {
            "value_mean": jnp.mean(v_dist.mean()),
            "mc_loss": mc_loss,
            "td_loss": td_loss,
            "td_weight": td_weight,
        }

    diff_state_v = nnx.DiffState(0, nnx.Param)
    diff_state_pi0 = nnx.DiffState(1, nnx.Param)
    (loss, aux_data), (grads_v, grads_pi0) = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=(diff_state_v, diff_state_pi0)
    )(value_model, pi0_model, raw_obs, actions, mc_return, q_model)

    new_value_state = _update_train_state(value_state, value_model, grads_v)
    new_pi0_state = _update_train_state(pi0_encoder_state, pi0_model, grads_pi0)
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads_v),
        "param_norm": _kernel_param_norm(value_model),
        "encoder_grad_norm": optax.global_norm(grads_pi0),
    } | aux_data
    return new_value_state, new_pi0_state, info
