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
from src.rl.networks.encoders.encoders import MLPEncoder
from src.rl.networks.encoders.resnet_encoderv1 import ResNetEncoder, ResNetBlock
from src.rl.networks.decoders.values.state_action_value import StateActionEnsembleDecoder
from src.rl.networks.decoders.values.state_value import StateValueEnsembleDecoder
from src.rl.networks.mlp import MLP
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


class ResNetStateEncoder(nnx.Module):
    """Encodes images with ResNet (spatial softmax), concatenates with state, then passes through a shared MLP."""

    def __init__(self, observation: ObsType, stage_sizes: tuple[int, ...], image_keys: list[str], hidden_dims: tuple[int, ...], *, rngs: nnx.Rngs):
        self.resnet = ResNetEncoder(
            input_example=observation,
            stage_sizes=stage_sizes,
            block_cls=ResNetBlock,
            image_keys=image_keys,
            use_spatial_softmax=True,
            rngs=rngs,
        )
        dummy_img_features = self.resnet(observation, train=False)
        dummy_state = jnp.asarray(observation["state"], dtype=jnp.float32)
        dummy_concat = jnp.concatenate([dummy_img_features, dummy_state], axis=-1)
        self.mlp = MLP(input=dummy_concat, hidden_dims=hidden_dims, activate_final=True, rngs=rngs)

    def __call__(self, observation: ObsType, training: bool = False) -> jax.Array:
        img_features = self.resnet(observation, train=training)
        state = observation["state"].astype(jnp.float32)
        x = jnp.concatenate([img_features, state], axis=-1)
        return self.mlp(x, training=training)


class Pi0PrefixResNetEncoder(nnx.Module):
    """Combines frozen pi0 prefix embedding with trainable ResNet image features.

    Concatenates [ResNet(images), prefix_embedding, state] and passes through a shared MLP.
    The ResNet is trainable; the prefix embedding is provided externally (computed from frozen Pi0).
    """

    def __init__(
        self,
        observation: ObsType,
        stage_sizes: tuple[int, ...],
        image_keys: list[str],
        prefix_embedding_key: str,
        hidden_dims: tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
        self.prefix_embedding_key = prefix_embedding_key
        self.resnet = ResNetEncoder(
            input_example=observation,
            stage_sizes=stage_sizes,
            block_cls=ResNetBlock,
            image_keys=image_keys,
            use_spatial_softmax=True,
            rngs=rngs,
        )
        dummy_img_features = self.resnet(observation, train=False)
        dummy_prefix = jnp.asarray(observation[prefix_embedding_key], dtype=jnp.float32)
        dummy_state = jnp.asarray(observation["state"], dtype=jnp.float32)
        dummy_concat = jnp.concatenate([dummy_img_features, dummy_prefix, dummy_state], axis=-1)
        self.mlp = MLP(input=dummy_concat, hidden_dims=hidden_dims, activate_final=True, rngs=rngs)

    def __call__(self, observation: ObsType, training: bool = False) -> jax.Array:
        img_features = self.resnet(observation, train=training)
        prefix = jnp.asarray(observation[self.prefix_embedding_key], dtype=jnp.float32)
        state = jnp.asarray(observation["state"], dtype=jnp.float32)
        x = jnp.concatenate([img_features, prefix, state], axis=-1)
        return self.mlp(x, training=training)


def _build_pi0_backbone_critic_defs(
    config: OnlineTrainConfig,
    *,
    prefix_embedding_shape: tuple[int, ...] | None,
) -> tuple[StateActionCriticDef, StateValueDef]:
    critic_encoder_hidden_dims = config.rl.critic_encoder_hidden_dims
    critic_decoder_hidden_dims = config.rl.critic_decoder_hidden_dims
    critic_num_qs = config.rl.critic_num_qs
    critic_num_vs = config.rl.critic_num_vs

    def encoder_def(observation: ObsType, rngs: nnx.Rngs):
        if config.rl.critic_encoder_type == "resnet":
            return ResNetStateEncoder(
                observation=observation,
                stage_sizes=(2, 2, 2, 2),  # ResNet-18
                image_keys=["image", "wrist_image"],
                hidden_dims=critic_encoder_hidden_dims,
                rngs=rngs,
            )
        if config.rl.critic_encoder_type == "pi0_prefix_resnet":
            return Pi0PrefixResNetEncoder(
                observation=observation,
                stage_sizes=(2, 2, 2, 2),  # ResNet-18
                image_keys=["image", "wrist_image"],
                prefix_embedding_key=PREFIX_EMBEDDING_NAME,
                hidden_dims=critic_encoder_hidden_dims,
                rngs=rngs,
            )
        # pi0_prefix: concatenate [prefix_embedding, state] through an MLP encoder
        state_vector_keys = [PREFIX_EMBEDDING_NAME, "state"] if prefix_embedding_shape is not None else ["state"]
        return MLPEncoder(
            dummy_obs=observation,
            encoder_def=lambda o, rg: MLP(
                input=o,
                hidden_dims=critic_encoder_hidden_dims,
                activate_final=True,
                rngs=rg,
            ),
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
            num_bins=config.rl.num_value_bins,
            rngs=rngs,
        )

    def state_value_decoder_def(
        embedding: jax.Array, rngs: nnx.Rngs
    ) -> StateValueEnsembleDecoder:
        return StateValueEnsembleDecoder(
            observation=embedding,
            hidden_dims=critic_decoder_hidden_dims,
            num_vs=critic_num_vs,
            num_bins=config.rl.num_value_bins,
            rngs=rngs,
        )

    action_compress_dim = config.rl.critic_action_compress_dim
    action_encoder_def: Callable | None = None
    if action_compress_dim is not None:
        def action_encoder_def(action: jax.Array, rngs: nnx.Rngs) -> nnx.Module:
            return MLP(
                input=action,
                hidden_dims=(128, action_compress_dim),
                activate_final=True,
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
            action_encoder_def=action_encoder_def,
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

    abstract_state = jax.eval_shape(init, dummy_obs, dummy_act, init_rng)
    state_sharding = sharding.fsdp_sharding(abstract_state, mesh, log=False)
    train_state = jax.jit(init, out_shardings=state_sharding)(dummy_obs, dummy_act, init_rng)
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

    abstract_state = jax.eval_shape(init, dummy_obs, init_rng)
    state_sharding = sharding.fsdp_sharding(abstract_state, mesh, log=False)
    train_state = jax.jit(init, out_shardings=state_sharding)(dummy_obs, init_rng)
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
