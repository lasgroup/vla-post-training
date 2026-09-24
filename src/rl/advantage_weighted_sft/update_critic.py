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
from src.training.config import OnlineTrainConfig, AdvantageWeightedSFTLearnerConfig
from src.rl.networks.per_task_critic import (
    TASK_INDEX_NAME,
    PerTaskStateActionCritic,
    PerTaskStateValue,
)
from src.rl.networks.rl_networks import (
    ObsType,
    ActionType,
    StateActionCritic,
    StateValue,
)
from src.rl.value_distribution import get_value_bounds, make_value_distribution


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


def _use_ema_critic(config: OnlineTrainConfig) -> bool:
    return config.rl.critic.use_ema


def _critic_ema_decay(config: OnlineTrainConfig) -> float | None:
    return config.rl.critic.ema_decay


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
    dist = make_value_distribution(critic_logits, config.rl.critic.num_value_bins, lower, upper)
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


def critic_values_per_head(
    critic_logits: at.ArrayLike,
    config: OnlineTrainConfig,
) -> at.Float[at.Array, "n b"]:
    """Per-head expected values (no reduction across the ensemble)."""
    lower, upper = get_value_bounds(config)
    dist = make_value_distribution(critic_logits, config.rl.critic.num_value_bins, lower, upper)
    return dist.mean()  # (num_heads, b)


@at.typecheck
def flatten_action_horizon(values: ActionType) -> at.Float[at.Array, "b a"]:
    return values.reshape((values.shape[0], -1))


def _ensure_rngs(rng: at.KeyArrayLike | nnx.Rngs) -> nnx.Rngs:
    if isinstance(rng, nnx.Rngs):
        return rng
    return nnx.Rngs(rng)


def _num_critic_tasks(config: OnlineTrainConfig) -> int | None:
    return config.rl.critic.num_tasks


@at.typecheck
def per_task_mean(
    per_sample: at.Float[at.Array, "*heads b"],
    task_index: at.Int[at.Array, " b"],
    num_tasks: int,
) -> at.Float[at.Array, ""]:
    """Per-task mean of ``per_sample`` over the batch axis, averaged over the
    tasks PRESENT in the batch (decision D6).

    A task's loss is normalized by its own sample count, so an under-represented
    task still takes a full-size step; a task absent from the batch contributes
    exactly zero -- not NaN -- and does not shrink the others (``max(count, 1)``
    and ``max(n_present, 1)`` guards). Leading ensemble axes are averaged as the
    plain ``jnp.mean`` did. Pinned by tests/ogpo/test_per_task_critics.py.
    """
    onehot = jax.nn.one_hot(task_index, num_tasks, dtype=per_sample.dtype)  # (b, T)
    counts = jnp.sum(onehot, axis=0)  # (T,)
    sums = jnp.einsum("...b,bt->...t", per_sample, onehot)  # (*heads, T)
    task_means = sums / jnp.maximum(counts, 1.0)
    # mean over ensemble heads (all leading axes)
    task_means = task_means.reshape((-1, num_tasks)).mean(axis=0)  # (T,)
    present = (counts > 0).astype(per_sample.dtype)
    return jnp.sum(task_means * present) / jnp.maximum(jnp.sum(present), 1.0)


def _task_subtree_mask(num_tasks: int, task: int):
    """optax.masked predicate selecting task ``task``'s parameter subtree.

    The per-task wrappers hold their sub-critics in ``self.tasks[t]``, so the
    flattened nnx.State keypath of every such leaf starts
    ``(DictKey('tasks'), DictKey(t), ...)``. Returns a callable (evaluated by
    optax at init/update on the actual tree) so no template tree is needed at
    optimizer construction.
    """
    def _slot(keypath) -> int | None:
        if len(keypath) < 2:
            return None
        k0, k1 = keypath[0], keypath[1]
        if (
            isinstance(k0, jax.tree_util.DictKey) and k0.key == "tasks"
            and isinstance(k1, jax.tree_util.DictKey) and isinstance(k1.key, int)
            and 0 <= k1.key < num_tasks
        ):
            return k1.key
        return None

    def mask_fn(tree):
        # Fail CLOSED: every leaf must belong to some task slot. A leaf the
        # predicate cannot place (a renamed attribute, an nnx keypath change)
        # would otherwise make the masked clip a silent no-op for that leaf --
        # i.e. an unclipped critic with no error. Keypaths are static, so this
        # raises at tx.init / trace time, not on device.
        unplaced = [
            jax.tree_util.keystr(kp)
            for kp, _ in jax.tree_util.tree_leaves_with_path(tree)
            if _slot(kp) is None
        ]
        if unplaced:
            raise ValueError(
                f"per_task_clip_chain(num_tasks={num_tasks}): {len(unplaced)} param "
                f"leaves are not under tasks/<0..{num_tasks - 1}> (first: {unplaced[0]}); "
                "the per-task wrapper layout (PerTaskStateActionCritic.tasks) and "
                "_task_subtree_mask disagree -- fix the predicate, do not train unclipped."
            )
        return jax.tree_util.tree_map_with_path(lambda kp, _: _slot(kp) == task, tree)

    return mask_fn


def _critic_optimizer(config: OnlineTrainConfig) -> optax.GradientTransformation:
    """The critic optimizer. ``num_tasks is None`` is exactly today's path.

    With per-task critics the global ``clip_by_global_norm`` that
    ``_optimizer.AdamW.create`` bakes in (openpi/src/openpi/training/optimizer.py:81-85)
    would couple the tasks: one clip over T disjoint subtrees lets task A's
    gradient magnitude shrink task B's update. So the clip is applied once per
    task subtree (``optax.masked``), followed by the same AdamW core built here
    from the dataclass fields -- openpi is a submodule and is not edited
    (decision D5 / R1). Pinned by tests/ogpo/test_per_task_critics.py.
    """
    crit = config.rl.critic
    num_tasks = _num_critic_tasks(config)
    if num_tasks is None:
        return _optimizer.create_optimizer(crit.optimizer, crit.lr_schedule, weight_decay_mask=None)
    if not isinstance(crit.optimizer, _optimizer.AdamW):
        raise ValueError(
            "Per-task critics (rl.critic.num_tasks) mirror AdamW.create's "
            "clip+adamw chain per task; rl.critic.optimizer must be "
            f"openpi AdamW, got {type(crit.optimizer).__name__}."
        )
    opt = crit.optimizer
    adamw = optax.adamw(
        crit.lr_schedule.create(),
        b1=opt.b1, b2=opt.b2, eps=opt.eps, weight_decay=opt.weight_decay, mask=None,
    )
    return optax.chain(per_task_clip_chain(num_tasks, opt.clip_gradient_norm), adamw)


def per_task_clip_chain(num_tasks: int, max_norm: float) -> optax.GradientTransformation:
    """One ``clip_by_global_norm(max_norm)`` per task subtree, applied independently
    (``optax.masked`` computes the norm over the masked leaves only). Exposed so
    the clip can be tested without Adam's scale invariance hiding it."""
    return optax.chain(
        *[
            optax.masked(
                optax.clip_by_global_norm(max_norm), _task_subtree_mask(num_tasks, t)
            )
            for t in range(num_tasks)
        ]
    )


def init_state_action_critic_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    critic_def: StateActionCriticDef,
    dummy_obs: ObsType,
    dummy_act: ActionType,
) -> tuple[training_utils.TrainState, Any]:
    tx = _critic_optimizer(config)
    ema_decay = _critic_ema_decay(config)
    num_tasks = _num_critic_tasks(config)
    dummy_act = flatten_action_horizon(dummy_act)
    dummy_act = jax.tree.map(lambda x: x.reshape(*x.shape[:-1], -1), dummy_act)

    def init(obs, act, rng) -> training_utils.TrainState:
        if num_tasks is None:
            critic = critic_def(obs, act, _ensure_rngs(rng))
        else:
            # Per-task critics: T disjoint copies of the injected critic_def.
            critic = PerTaskStateActionCritic(
                critic_def, num_tasks, obs, act, rngs=_ensure_rngs(rng)
            )
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
    tx = _critic_optimizer(config)
    ema_decay = _critic_ema_decay(config)
    num_tasks = _num_critic_tasks(config)

    def init(obs, rng) -> training_utils.TrainState:
        if num_tasks is None:
            critic = critic_def(obs, _ensure_rngs(rng))
        else:
            critic = PerTaskStateValue(critic_def, num_tasks, obs, rngs=_ensure_rngs(rng))
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


def _per_task_aux(
    td_lp: at.Float[at.Array, "*heads b"],
    td_weight: at.Float[at.Array, ""],
    mc_lp: at.Float[at.Array, "*heads b"],
    task_index: at.Int[at.Array, " b"],
    num_tasks: int,
) -> dict[str, at.Array]:
    """Per-task ``loss_task_{t}`` / ``n_task_{t}`` (decision R5): the only way the
    residual coupling -- a task absent from a batch still takes an Adam step on
    zero gradient (R4) -- is observable. ``num_tasks`` is static, so the key set
    is fixed for a given config; exp.py NaN-fills keys missing in a window."""
    onehot = jax.nn.one_hot(task_index, num_tasks, dtype=td_lp.dtype)  # (b, T)
    counts = jnp.sum(onehot, axis=0)
    def _task_means(lp):
        sums = jnp.einsum("...b,bt->...t", lp, onehot)
        return (sums / jnp.maximum(counts, 1.0)).reshape((-1, num_tasks)).mean(axis=0)
    loss_t = -(td_weight * _task_means(td_lp) + (1 - td_weight) * _task_means(mc_lp))
    aux = {}
    for t in range(num_tasks):
        aux[f"loss_task_{t}"] = loss_t[t]
        aux[f"n_task_{t}"] = counts[t]
    return aux


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
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)

    step = q_state.step // config.rl.critic.num_updates_per_batch
    num_tasks = _num_critic_tasks(config)
    observation, actions, next_observation, reward, discount, mc_return = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    mc_return = _as_scalar_batch(mc_return)
    actions = flatten_action_horizon(actions)
    # Per-task critics: the loss is a per-task mean averaged over the tasks
    # present in the batch (per_task_mean); None keeps the plain batch mean
    # below verbatim (config-driven branch, resolved at trace time).
    task_index = observation[TASK_INDEX_NAME] if num_tasks is not None else None

    # (3) Q_i backs up on the reduced V ensemble; `mean` gives the mean target.
    q_backup_reduction = config.rl.critic.q_bootstrap_reduction or config.rl.critic.reduction
    bootstrap_target = summarize_critic_values(
        value_model(next_observation), config, critic_reduction=q_backup_reduction
    )

    def loss_fn(q_model, observation, actions):
        td_weight = jnp.clip(config.rl.critic.td_weight_schedule.create()(step), 0.0, 1.0)
        q_logits = q_model(observation, actions)
        td_targets = reward + discount * jax.lax.stop_gradient(bootstrap_target)
        _lower, _upper = get_value_bounds(config)
        q_dist = make_value_distribution(q_logits, config.rl.critic.num_value_bins, _lower, _upper, config.rl.critic.value_target_type)
        td_lp = q_dist.log_prob(td_targets)
        mc_lp = q_dist.log_prob(mc_return)
        if task_index is None:
            td_loss = -jnp.mean(td_lp)
            mc_loss = -jnp.mean(mc_lp)
        else:
            td_loss = -per_task_mean(td_lp, task_index, num_tasks)
            mc_loss = -per_task_mean(mc_lp, task_index, num_tasks)
        value_mean = jnp.mean(q_dist.mean())
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        # Ranking-quality proxy: Pearson corr between the (head-mean) Q
        # prediction and the observed MC return over this batch. Calibration
        # drift is harmless to a group-baseline actor, but a decaying corr
        # means the critic's ORDERING of actions is losing signal — the one
        # critic failure mode that actually reaches the advantage.
        q_pred = jnp.mean(q_dist.mean(), axis=0) if q_dist.mean().ndim > 1 else q_dist.mean()
        qc = q_pred - jnp.mean(q_pred)
        mc = mc_return - jnp.mean(mc_return)
        q_mc_corr = jnp.sum(qc * mc) / jnp.maximum(
            jnp.linalg.norm(qc) * jnp.linalg.norm(mc), 1e-8
        )
        aux = {
            "value_mean": value_mean,
            "td_loss": td_loss,
            "mc_loss": mc_loss,
            "td_weight": td_weight,
            "mc_corr": q_mc_corr,
        }
        if task_index is not None:
            aux |= _per_task_aux(td_lp, td_weight, mc_lp, task_index, num_tasks)
        return loss, aux

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
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    step = value_state.step // config.rl.critic.num_updates_per_batch
    num_tasks = _num_critic_tasks(config)
    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()

    q_model = create_critic(q_state, config)
    q_model.eval()

    observation, actions, _, _, _, mc_return = batch
    actions = flatten_action_horizon(actions)
    mc_return = _as_scalar_batch(mc_return)
    # Per-task critics: see train_q_step.
    task_index = observation[TASK_INDEX_NAME] if num_tasks is not None else None

    # (2) per-critic target pairs V_i with Q_i; else all V_i share the reduced Q target.
    if config.rl.critic.per_critic_value_target:
        assert config.rl.critic.num_vs == config.rl.critic.num_qs, "per_critic_value_target needs num_vs == num_qs"
        bootstrap_target = critic_values_per_head(q_model(observation, actions), config)  # (n, b)
    else:
        bootstrap_target = summarize_critic_values(
            q_model(observation, actions), config, critic_reduction=config.rl.critic.reduction
        )

    def loss_fn(value_model, observation):
        td_weight = jnp.clip(config.rl.critic.td_weight_schedule.create()(step), 0.0, 1.0)
        value_logits = value_model(observation)
        _lower, _upper = get_value_bounds(config)
        v_dist = make_value_distribution(value_logits, config.rl.critic.num_value_bins, _lower, _upper, config.rl.critic.value_target_type)
        mc_lp = v_dist.log_prob(mc_return)
        td_lp = v_dist.log_prob(jax.lax.stop_gradient(bootstrap_target))
        if task_index is None:
            mc_loss = -jnp.mean(mc_lp)
            td_loss = -jnp.mean(td_lp)
        else:
            mc_loss = -per_task_mean(mc_lp, task_index, num_tasks)
            td_loss = -per_task_mean(td_lp, task_index, num_tasks)
        value_mean = jnp.mean(v_dist.mean())
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
        aux = {
            "value_mean": value_mean,
            "td_loss": td_loss,
            "mc_loss": mc_loss,
            "td_weight": td_weight,
        }
        if task_index is not None:
            aux |= _per_task_aux(td_lp, td_weight, mc_lp, task_index, num_tasks)
        return loss, aux

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
