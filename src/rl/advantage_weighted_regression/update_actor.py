from src.training.config import OnlineTrainConfig
from src.rl.advantage_weighted_regression.update_critic import create_critic
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import dataclasses

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.utils as training_utils


def _awr_beta(config: OnlineTrainConfig) -> float:
    rl_config = getattr(config, "rl", None)
    beta = float(getattr(rl_config, "beta", 1.0))
    return max(beta, 1e-6)


@at.typecheck
def train_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    policy_state: training_utils.TrainState,
    state_action_critic_state: training_utils.TrainState,
    value_state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    policy = nnx.merge(policy_state.model_def, policy_state.params)
    policy.train()

    state_action_critic = create_critic(state_action_critic_state, config)
    value_critic = create_critic(value_state, config)

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        critic_model: nnx.Module,
        value_model: nnx.Module,
    ):
        # We up-weight terms that have high advantage
        value = value_model(observation)
        q_value = critic_model(observation, actions)
        advantage = q_value - value
        score = advantage / _awr_beta(config)
        # Normalize the weights across the batch axis for training stability and expand dim by one for the chunk loss.
        score = jax.nn.softmax(score, axis=0)[..., jnp.newaxis]
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        aux_data = {
            "advantage_weights": jnp.mean(score),
            "chunked_loss": jnp.mean(chunked_loss),
        }
        return jnp.mean(score * chunked_loss), aux_data


    train_rng = jax.random.fold_in(rng, policy_state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux_data), grads = nnx.value_and_grad(loss_fn, has_aux=True, argnums=diff_state)(
        policy,
        train_rng,
        observation,
        actions,
        state_action_critic,
        value_critic,
      )

    params = nnx.filter_state(policy_state.params, config.trainable_filter)
    updates, new_opt_state = policy_state.tx.update(grads, policy_state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(policy, new_params)
    new_params = nnx.state(policy)

    new_state = dataclasses.replace(policy_state, step=policy_state.step + 1, params=new_params,
                                    opt_state=new_opt_state)
    if policy_state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: policy_state.ema_decay * old + (1 - policy_state.ema_decay) * new,
                policy_state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        policy,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    } | aux_data
    return new_state, info
