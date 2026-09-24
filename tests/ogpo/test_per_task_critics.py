# ruff: noqa: F722
"""Per-task critics (``rl.critic.num_tasks``): the claims the change stands on.

docs/changes/2026-08-21-per-task-critics/ — decisions D1-D8, R1-R5.

What is certified here, in order of how badly it would fail silently:

1. **Routing** — a sample of task t is scored by task t's sub-critic and no
   other (D2, "never use another task's critic for policy extraction").
2. **Gradient disjointness** — a batch containing only task t leaves every other
   task's gradient tree exactly zero (D2, "no shared params").
3. **The ``num_tasks=None`` path is bit-identical** to the pre-change critic
   steps, checked against a VERBATIM copy of those steps
   (``_reference_train_q_step`` / ``_reference_train_value_step``, the
   ``test_split_equivalence.py`` pattern — comparing a refactor to itself
   proves nothing).
4. **T=1 reproduces a standalone critic** through the per-task loss and the
   per-task clip chain (tolerance: reassociation only).
5. **The per-task clip isolates tasks** (D5) — tested on the clip chain
   directly, because Adam's scale invariance would hide it one step later.
6. ``per_task_mean`` (D6), ``TaskRegistry`` (D3/D4), config validation (D7),
   and the jit-1 ``task_index`` sidecar (the Tier-2 signature edit) — the
   ``None`` default is an identity, and routing survives the G-expansion.

Tolerances: ``_ATOL_EXACT = 0`` where the ops are literally the same graph;
``_ATOL_REASSOC = 1e-5`` where a sum is re-associated (per-task sums vs one
batch mean) or a masked clip replaces a global one.
"""
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import tyro

import openpi.shared.array_typing as at
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from src.rl.advantage_weighted_sft.update_critic import (
    CriticBatch,
    _as_scalar_batch,
    _critic_optimizer,
    _kernel_param_norm,
    _task_subtree_mask,
    _update_train_state,
    create_critic,
    critic_values_per_head,
    flatten_action_horizon,
    init_state_action_critic_train_state,
    init_state_value_train_state,
    per_task_clip_chain,
    per_task_mean,
    summarize_critic_values,
    train_q_step,
    train_value_step,
)
from src.rl.best_of_n.update_critic import _build_pi0_backbone_critic_defs
from src.rl.networks.bronet_critic import BroNetStateActionCritic, BroNetStateValue
from src.rl.networks.per_task_critic import (
    TASK_INDEX_NAME,
    PerTaskStateActionCritic,
    PerTaskStateValue,
)
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.rl.task_registry import TaskRegistry
from src.rl.value_distribution import get_value_bounds, make_value_distribution
from src.training.config import (
    AdvantageWeightedSFTLearnerConfig,
    OGPOSFTLearnerConfig,
    OnlineTrainConfig,
    get_config,
)

_ATOL_EXACT = 0.0
_ATOL_REASSOC = 1e-5
_S, _E, _AH, _AD = 8, 16, 2, 4  # state dim, prefix-embed dim, action horizon, action dim


# --------------------------------------------------------------------------- #
# Config + synthetic critic data (no PaliGemma needed for the critic legs)
# --------------------------------------------------------------------------- #

def _critic_config(num_tasks, *, use_bronet=True, num_qs=2):
    base = get_config("pi05_libero_online_ogpo_sft")
    critic = dataclasses.replace(
        base.rl.critic,
        use_bronet=use_bronet, bronet_hidden_dim=16, bronet_depth=1,
        encoder_hidden_dims=(16,), decoder_hidden_dims=(16,),
        num_qs=num_qs, num_vs=num_qs, num_tasks=num_tasks,
    )
    return dataclasses.replace(base, rl=dataclasses.replace(base.rl, critic=critic))


def _critic_defs(config):
    if config.rl.critic.use_bronet:
        hd, dep = config.rl.critic.bronet_hidden_dim, config.rl.critic.bronet_depth
        nq, nv = config.rl.critic.num_qs, config.rl.critic.num_vs

        def sa_def(observation, action, rngs):
            return BroNetStateActionCritic(
                observation=observation, action=action, hidden_dim=hd, depth=dep, num_qs=nq, rngs=rngs
            )

        def sv_def(observation, rngs):
            return BroNetStateValue(observation=observation, hidden_dim=hd, depth=dep, num_vs=nv, rngs=rngs)

        return sa_def, sv_def
    return _build_pi0_backbone_critic_defs(config)


def _dummy_obs(with_task_index):
    obs = {"state": jnp.zeros((1, _S)), PREFIX_EMBEDDING_NAME: jnp.zeros((1, _E))}
    if with_task_index:
        obs[TASK_INDEX_NAME] = jnp.zeros((1,), jnp.int32)
    return obs


def _obs(rng, b, task_index=None):
    k1, k2 = jax.random.split(rng)
    obs = {
        "state": jax.random.normal(k1, (b, _S)),
        PREFIX_EMBEDDING_NAME: jax.random.normal(k2, (b, _E)),
    }
    if task_index is not None:
        obs[TASK_INDEX_NAME] = jnp.asarray(task_index, jnp.int32)
    return obs


def _batch(rng, task_index) -> CriticBatch:
    """A CriticBatch with task_index on obs AND next_obs (as the learner builds it)."""
    b = len(task_index)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    obs = _obs(k1, b, task_index)
    next_obs = _obs(k2, b, task_index)
    actions = jax.random.normal(k3, (b, _AH, _AD))
    reward = -jnp.ones((b,))
    discount = jnp.full((b,), 0.99)
    mc_return = -100.0 + 50.0 * jax.random.normal(k4, (b,))
    return obs, actions, next_obs, reward, discount, mc_return


def _strip_task_index(batch: CriticBatch) -> CriticBatch:
    obs, a, nobs, r, d, mc = batch
    drop = lambda o: {k: v for k, v in o.items() if k != TASK_INDEX_NAME}
    return drop(obs), a, drop(nobs), r, d, mc


def _init_states(config, rng):
    mesh = sharding.make_mesh(1)
    sa_def, sv_def = _critic_defs(config)
    q_rng, v_rng = jax.random.split(rng)
    dummy_obs = _dummy_obs(config.rl.critic.num_tasks is not None)
    q_state, _ = init_state_action_critic_train_state(
        config, q_rng, mesh, critic_def=sa_def, dummy_obs=dummy_obs, dummy_act=jnp.zeros((1, _AH, _AD))
    )
    v_state, _ = init_state_value_train_state(config, v_rng, mesh, critic_def=sv_def, dummy_obs=dummy_obs)
    return q_state, v_state


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree)]


def _task_leaves(state: nnx.State, t: int):
    """Leaves of a per-task wrapper State under tasks/<t>, in flat order."""
    return [np.asarray(v.value) for p, v in state.flat_state() if p[0] == "tasks" and p[1] == t]


def _standalone_state(pt_state: training_utils.TrainState, t: int, config_none) -> training_utils.TrainState:
    """A TrainState for sub-critic ``t`` alone, params shared by value with ``pt_state``,
    under today's optimizer (global clip + AdamW) — the reference a T=1 run must match."""
    wrapper = nnx.merge(pt_state.model_def, pt_state.params)
    sub = wrapper.tasks[t]
    params = nnx.state(sub)
    tx = _critic_optimizer(config_none)
    return training_utils.TrainState(
        step=pt_state.step, params=params, model_def=nnx.graphdef(sub), tx=tx,
        opt_state=tx.init(nnx.filter_state(params, nnx.Param)),
        ema_decay=pt_state.ema_decay, ema_params=None if pt_state.ema_params is None else params,
    )


# --------------------------------------------------------------------------- #
# VERBATIM copies of the pre-change critic steps (git HEAD 5b94510,
# src/rl/advantage_weighted_sft/update_critic.py:232-358). Only the helpers they
# call are imported from the live module; those helpers are untouched by the
# change (git diff). This is the differential reference for legs 3 and 4.
# --------------------------------------------------------------------------- #

@at.typecheck
def _reference_train_q_step(
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
    observation, actions, next_observation, reward, discount, mc_return = batch
    reward = _as_scalar_batch(reward)
    discount = _as_scalar_batch(discount)
    mc_return = _as_scalar_batch(mc_return)
    actions = flatten_action_horizon(actions)

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
        td_loss = -jnp.mean(q_dist.log_prob(td_targets))
        mc_loss = -jnp.mean(q_dist.log_prob(mc_return))
        value_mean = jnp.mean(q_dist.mean())
        loss = td_weight * td_loss + (1 - td_weight) * mc_loss
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
def _reference_train_value_step(
    config: OnlineTrainConfig,
    rng: at.KeyArrayLike,
    value_state: training_utils.TrainState,
    q_state: training_utils.TrainState,
    batch: CriticBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del rng
    assert isinstance(config.rl, AdvantageWeightedSFTLearnerConfig)
    step = value_state.step // config.rl.critic.num_updates_per_batch
    value_model = nnx.merge(value_state.model_def, value_state.params)
    value_model.train()

    q_model = create_critic(q_state, config)
    q_model.eval()

    observation, actions, _, _, _, mc_return = batch
    actions = flatten_action_horizon(actions)
    mc_return = _as_scalar_batch(mc_return)

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


# --------------------------------------------------------------------------- #
# 1. Routing (D2) — both wrappers, both backbones
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("use_bronet", [True, False], ids=["bronet", "pi0_mlp"])
def test_routing_matches_standalone_subcritic(use_bronet):
    config = _critic_config(3, use_bronet=use_bronet)
    sa_def, sv_def = _critic_defs(config)
    dummy = _dummy_obs(True)
    q = PerTaskStateActionCritic(sa_def, 3, dummy, jnp.zeros((1, _AH * _AD)), rngs=nnx.Rngs(jax.random.key(0)))
    v = PerTaskStateValue(sv_def, 3, dummy, rngs=nnx.Rngs(jax.random.key(1)))
    task_index = [0, 2, 1, 2, 0, 1]
    obs = _obs(jax.random.key(2), len(task_index), task_index)
    act = jax.random.normal(jax.random.key(3), (len(task_index), _AH * _AD))
    q_out, v_out = q(obs, act), v(obs)
    assert q_out.shape == (2, 6) and v_out.shape == (2, 6)  # (num_heads, b): today's contract
    for i, t in enumerate(task_index):
        np.testing.assert_allclose(np.asarray(q_out[:, i]), np.asarray(q.tasks[t](obs, act)[:, i]), atol=_ATOL_EXACT)
        np.testing.assert_allclose(np.asarray(v_out[:, i]), np.asarray(v.tasks[t](obs)[:, i]), atol=_ATOL_EXACT)
    # And NOT another task's: the sub-critics are initialized independently.
    for i, t in enumerate(task_index):
        other = (t + 1) % 3
        assert not np.allclose(np.asarray(q_out[:, i]), np.asarray(q.tasks[other](obs, act)[:, i]))


def test_subcritics_have_disjoint_independently_initialized_params():
    config = _critic_config(2)
    sa_def, _ = _critic_defs(config)
    q = PerTaskStateActionCritic(sa_def, 2, _dummy_obs(True), jnp.zeros((1, _AH * _AD)), rngs=nnx.Rngs(jax.random.key(0)))
    params = nnx.state(q, nnx.Param)
    paths = [p for p, _ in params.flat_state()]
    assert all(p[0] == "tasks" for p in paths), "every Param leaf lives under a task slot — nothing shared"
    assert {p[1] for p in paths} == {0, 1}
    # Compare a KERNEL leaf (biases are zero-initialized in every slot).
    k0 = [np.asarray(v.value) for p, v in params.flat_state() if p[1] == 0 and p[-1] == "kernel"]
    k1 = [np.asarray(v.value) for p, v in params.flat_state() if p[1] == 1 and p[-1] == "kernel"]
    assert k0 and len(k0) == len(k1)
    assert not np.allclose(k0[0], k1[0]), "slots must be initialized independently"


def test_missing_task_index_raises_not_defaults():
    config = _critic_config(2)
    sa_def, _ = _critic_defs(config)
    q = PerTaskStateActionCritic(sa_def, 2, _dummy_obs(True), jnp.zeros((1, _AH * _AD)), rngs=nnx.Rngs(jax.random.key(0)))
    with pytest.raises(KeyError, match="task_index"):
        q(_obs(jax.random.key(1), 3), jnp.zeros((3, _AH * _AD)))


# --------------------------------------------------------------------------- #
# 2. Gradient disjointness (D2, "no shared params")
# --------------------------------------------------------------------------- #

def test_single_task_batch_gives_exactly_zero_grads_to_other_tasks():
    config = _critic_config(3)
    sa_def, _ = _critic_defs(config)
    q = PerTaskStateActionCritic(sa_def, 3, _dummy_obs(True), jnp.zeros((1, _AH * _AD)), rngs=nnx.Rngs(jax.random.key(0)))
    t = 1
    obs = _obs(jax.random.key(1), 4, [t] * 4)
    act = jax.random.normal(jax.random.key(2), (4, _AH * _AD))

    def loss(m, o, a):
        return jnp.sum(m(o, a) ** 2)

    grads = nnx.grad(loss, argnums=nnx.DiffState(0, nnx.Param))(q, obs, act)
    for u in range(3):
        if u == t:
            continue
        for g in _task_leaves(grads, u):
            assert np.all(g == 0.0), f"task {u} received gradient from a task-{t}-only batch"
    # Task t's gradient is the standalone sub-critic's gradient on the same batch.
    sub_grads = nnx.grad(loss, argnums=nnx.DiffState(0, nnx.Param))(q.tasks[t], obs, act)
    for g_pt, g_sub in zip(_task_leaves(grads, t), _leaves(sub_grads)):
        # scatter-add of one index per row: literally the same values
        np.testing.assert_allclose(g_pt, g_sub, atol=1e-6)


# --------------------------------------------------------------------------- #
# 6a. per_task_mean (D6)
# --------------------------------------------------------------------------- #

def test_per_task_mean_hand_built_with_absent_task():
    # heads=2, b=5, T=3; task 2 absent. per-task means over the batch axis, then
    # over heads, then over PRESENT tasks only.
    per_sample = jnp.asarray([[1.0, 2.0, 3.0, 4.0, 10.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0]])
    task_index = jnp.asarray([0, 0, 1, 1, 1], jnp.int32)
    got = float(per_task_mean(per_sample, task_index, 3))
    head0 = ((1 + 2) / 2 + (3 + 4 + 10) / 3) / 2
    head1 = 0.0
    expected = (head0 + head1) / 2
    assert abs(got - expected) < 1e-6
    assert np.isfinite(got), "an absent task must contribute 0, never NaN"


def test_per_task_mean_all_tasks_absent_is_zero_not_nan():
    per_sample = jnp.ones((2, 3))
    got = per_task_mean(per_sample, jnp.asarray([5, 5, 5], jnp.int32), 2)  # out-of-range -> one_hot all zero
    assert float(got) == 0.0


def test_per_task_mean_single_task_equals_plain_mean():
    per_sample = jax.random.normal(jax.random.key(0), (2, 9))
    got = per_task_mean(per_sample, jnp.zeros((9,), jnp.int32), 1)
    np.testing.assert_allclose(float(got), float(jnp.mean(per_sample)), atol=1e-6)  # reassociation


# --------------------------------------------------------------------------- #
# 5. Per-task clip (D5) — on the clip chain itself, and the optimizer factory
# --------------------------------------------------------------------------- #

def test_task_subtree_mask_selects_exactly_that_task():
    config = _critic_config(2)
    sa_def, _ = _critic_defs(config)
    q = PerTaskStateActionCritic(sa_def, 2, _dummy_obs(True), jnp.zeros((1, _AH * _AD)), rngs=nnx.Rngs(jax.random.key(0)))
    params = nnx.filter_state(nnx.state(q), nnx.Param)
    mask = _task_subtree_mask(2, 1)(params)
    assert jax.tree.structure(mask) == jax.tree.structure(params)
    for (p, _), m in zip(params.flat_state(), jax.tree.leaves(mask)):
        assert bool(m) == (p[1] == 1), p


def test_per_task_clip_chain_leaves_other_tasks_untouched():
    config = _critic_config(2)
    sa_def, _ = _critic_defs(config)
    q = PerTaskStateActionCritic(sa_def, 2, _dummy_obs(True), jnp.zeros((1, _AH * _AD)), rngs=nnx.Rngs(jax.random.key(0)))
    params = nnx.filter_state(nnx.state(q), nnx.Param)
    # task 0: huge gradient; task 1: small gradient (norm well under the clip).
    grads = params.map(lambda p, v: v.replace(jnp.full_like(v.value, 1e3 if p[1] == 0 else 1e-3)))
    small_norm = float(optax.global_norm(jax.tree.leaves(nnx.State.from_flat_path(
        {p: v for p, v in grads.flat_state() if p[1] == 1}))))
    assert small_norm < 1.0

    chain = per_task_clip_chain(2, 1.0)
    clipped, _ = chain.update(grads, chain.init(grads), params)
    np.testing.assert_allclose(
        optax.global_norm(_task_leaves(clipped, 0)), 1.0, rtol=1e-5, err_msg="task 0 should be clipped to 1.0"
    )
    for g_new, g_old in zip(_task_leaves(clipped, 1), _task_leaves(grads, 1)):
        np.testing.assert_allclose(g_new, g_old, atol=_ATOL_EXACT, err_msg="task 1 must be untouched by task 0's clip")

    # The coupling the change removes: ONE global clip shrinks task 1 as well.
    glob = optax.clip_by_global_norm(1.0)
    g_clipped, _ = glob.update(grads, glob.init(grads), params)
    assert not np.allclose(_task_leaves(g_clipped, 1)[0], _task_leaves(grads, 1)[0])


def test_critic_optimizer_none_path_is_openpi_create_optimizer_bit_for_bit():
    config = _critic_config(None)
    sa_def, _ = _critic_defs(config)
    q = sa_def(_dummy_obs(False), jnp.zeros((1, _AH * _AD)), nnx.Rngs(jax.random.key(0)))
    params = nnx.filter_state(nnx.state(q), nnx.Param)
    grads = params.map(lambda p, v: v.replace(jax.random.normal(jax.random.key(hash(p) % 1000), v.value.shape)))
    ours = _critic_optimizer(config)
    theirs = _optimizer.create_optimizer(config.rl.critic.optimizer, config.rl.critic.lr_schedule, weight_decay_mask=None)
    u_ours, _ = ours.update(grads, ours.init(params), params)
    u_theirs, _ = theirs.update(grads, theirs.init(params), params)
    for a, b in zip(_leaves(u_ours), _leaves(u_theirs)):
        assert np.array_equal(a, b)


def test_critic_optimizer_rejects_non_adamw_with_per_task_critics():
    config = _critic_config(2)
    crit = dataclasses.replace(config.rl.critic)
    object.__setattr__(crit, "optimizer", _optimizer.SGD())  # class-level attr; override on the instance
    bad = dataclasses.replace(config, rl=dataclasses.replace(config.rl, critic=crit))
    with pytest.raises(ValueError, match="AdamW"):
        _critic_optimizer(bad)


# --------------------------------------------------------------------------- #
# 3. num_tasks=None is bit-identical to the verbatim pre-change steps
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("use_bronet", [True, False], ids=["bronet", "pi0_mlp"])
def test_num_tasks_none_train_steps_bit_identical_to_reference(use_bronet):
    config = _critic_config(None, use_bronet=use_bronet)
    q_state, v_state = _init_states(config, jax.random.key(0))
    batch = _strip_task_index(_batch(jax.random.key(1), [0, 0, 0, 0]))
    rng = jax.random.key(2)

    q_new, q_info_new = train_q_step(config, rng, q_state, v_state, batch)
    q_ref, q_info_ref = _reference_train_q_step(config, rng, q_state, v_state, batch)
    v_new, v_info_new = train_value_step(config, rng, v_state, q_state, batch)
    v_ref, v_info_ref = _reference_train_value_step(config, rng, v_state, q_state, batch)

    for new, ref in ((q_new, q_ref), (v_new, v_ref)):
        for a, b in zip(_leaves(new.params), _leaves(ref.params)):
            assert np.array_equal(a, b)
        for a, b in zip(_leaves(new.ema_params), _leaves(ref.ema_params)):
            assert np.array_equal(a, b)
    for new, ref in ((q_info_new, q_info_ref), (v_info_new, v_info_ref)):
        assert set(new) == set(ref), "num_tasks=None must emit exactly today's info keys"
        for k in ref:
            assert np.array_equal(np.asarray(new[k]), np.asarray(ref[k])), k


# --------------------------------------------------------------------------- #
# 4. T=1 reproduces a standalone critic through loss + clip + Adam
# --------------------------------------------------------------------------- #

def test_single_task_train_steps_match_standalone_reference():
    config_pt = _critic_config(1)
    config_none = _critic_config(None)
    q_pt, v_pt = _init_states(config_pt, jax.random.key(0))
    q_ref, v_ref = _standalone_state(q_pt, 0, config_none), _standalone_state(v_pt, 0, config_none)
    batch = _batch(jax.random.key(1), [0] * 6)
    rng = jax.random.key(2)

    q_pt2, q_info = train_q_step(config_pt, rng, q_pt, v_pt, batch)
    q_ref2, q_info_ref = _reference_train_q_step(config_none, rng, q_ref, v_ref, _strip_task_index(batch))
    v_pt2, v_info = train_value_step(config_pt, rng, v_pt, q_pt, batch)
    v_ref2, v_info_ref = _reference_train_value_step(config_none, rng, v_ref, q_ref, _strip_task_index(batch))

    for pt2, ref2 in ((q_pt2, q_ref2), (v_pt2, v_ref2)):
        for a, b in zip(_task_leaves(pt2.params, 0), _leaves(ref2.params)):
            np.testing.assert_allclose(a, b, atol=_ATOL_REASSOC)
    for info, info_ref in ((q_info, q_info_ref), (v_info, v_info_ref)):
        for k in ("loss", "td_loss", "mc_loss", "grad_norm", "value_mean"):
            np.testing.assert_allclose(float(info[k]), float(info_ref[k]), rtol=1e-5, atol=_ATOL_REASSOC, err_msg=k)
        assert float(info["n_task_0"]) == 6.0
        np.testing.assert_allclose(float(info["loss_task_0"]), float(info_ref["loss"]), rtol=1e-5, atol=_ATOL_REASSOC)


def test_two_tasks_absent_task_is_untouched_and_reported():
    config = _critic_config(2)
    q_state, v_state = _init_states(config, jax.random.key(0))
    batch = _batch(jax.random.key(1), [0, 0, 0, 0])  # task 1 absent
    q2, info = train_q_step(config, jax.random.key(2), q_state, v_state, batch)
    assert float(info["n_task_0"]) == 4.0 and float(info["n_task_1"]) == 0.0
    assert float(info["loss_task_1"]) == 0.0 and np.isfinite(float(info["loss"]))
    for a, b in zip(_task_leaves(q2.params, 1), _task_leaves(q_state.params, 1)):
        # zero gradient + first Adam step: m = v = 0, decay*lr = 1e-14 relative -> bit-identical
        assert np.array_equal(a, b), "absent task's critic must not move"
    assert any(not np.array_equal(a, b) for a, b in zip(_task_leaves(q2.params, 0), _task_leaves(q_state.params, 0)))


# --------------------------------------------------------------------------- #
# 6b. TaskRegistry (D3, D4)
# --------------------------------------------------------------------------- #

def test_registry_first_seen_assignment_and_overflow_message():
    # Keys are task IDs (libero_90_NN), not prompts — see test_per_task_critics_verifier
    # for the libero_90 prompt collision that forced this.
    reg = TaskRegistry(2)
    assert reg.index_for("libero_90_79") == 0
    assert reg.index_for("libero_90_82") == 1
    assert reg.index_for("libero_90_79") == 0  # idempotent
    with pytest.raises(ValueError) as e:
        reg.index_for("libero_90_38")
    msg = str(e.value)
    assert "libero_90_38" in msg and "2" in msg and "num_tasks" in msg
    assert len(reg) == 2


def test_registry_json_round_trip_and_mismatch(tmp_path):
    reg = TaskRegistry(3)
    reg.index_for("b"); reg.index_for("a")
    path = tmp_path / "task_registry_100.json"
    reg.to_json(path)
    back = TaskRegistry.from_json(path, 3)
    assert back.tasks == {"b": 0, "a": 1} and back.num_tasks == 3
    assert back.index_for("c") == 2  # continues from the persisted state
    with pytest.raises(ValueError, match="num_tasks"):
        TaskRegistry.from_json(path, 4)
    with pytest.raises(FileNotFoundError, match="D8"):
        TaskRegistry.from_json(tmp_path / "missing.json", 3)


def test_collect_data_threads_slot_aligned_task_ids_to_the_agent():
    """F1 plumbing: ``collect_data`` must hand the agent the task ID of each env slot —
    ``task_id=current_task_ids`` (slot-aligned list) on every ``sample_actions`` and
    ``task_id=current_task_ids[env_index]`` (the FINISHED episode's id, before the slot
    is reassigned) on every ``save_episode``. Stub env/agent; no simulator, no model."""
    import types
    from src.training.collect import collect_data

    E, H, D, A = 2, 3, 4, 2
    # Distinct, equal-length descriptions: the alignment assertion below must be
    # able to FAIL on a permuted id list (verifier F12); the shared-description
    # case is covered in test_per_task_critics_verifier2.
    desc = {"libero_90_79": "pick up the book", "libero_90_82": "open the drawer"}
    steps_to_done = 2  # episodes end on the 2nd chunk

    class Env:
        env_num = E

        def __init__(self):
            self.tasks = [None] * E
            self.age = [0] * E

        def seed(self, s):
            pass

        def _obs(self, n):
            return {"state": np.zeros((n, H, D), np.float32)}

        def reset(self, id=None, options=None):
            # info leaves are arrays (as the vector env returns them), so that
            # collect_data's jax.tree.map slot-update treats them as leaves.
            if id is None:
                self.tasks = list(options["task_id"]); self.age = [0] * E
                return self._obs(E), {"task_description": np.array([desc[t] for t in self.tasks], dtype=object)}
            self.tasks[id] = options["task_id"]; self.age[id] = 0
            return self._obs(1), {"task_description": np.array([desc[self.tasks[id]]], dtype=object)}

        def step(self, action):
            assert action.shape[0] == E
            term = np.zeros((E, H), bool)
            for i in range(E):
                self.age[i] += 1
                if self.age[i] >= steps_to_done:
                    term[i, -1] = True
            return self._obs(E), -np.ones((E, H), np.float32), term, np.zeros((E, H), bool), {}

    seen_sample, seen_save = [], []

    class Agent:
        total_collected_episodes = 0

        def start_data_collection(self, step=None): pass
        def end_data_collection(self, step=None): return 0
        def add_data(self, step_data): pass

        def sample_actions(self, obs, **kw):
            seen_sample.append((list(kw["task_description"]), list(kw["task_id"])))
            return np.zeros((E, H, A), np.float32)

        def save_episode(self, is_success, env_index, task_description, task_id):
            seen_save.append((env_index, task_description, task_id))

    cfg = types.SimpleNamespace(
        seed=0,
        collect=types.SimpleNamespace(
            tasks=["libero_90_79", "libero_90_82"], num_rollouts=1, num_initial_rollouts=None,
            replan_steps=H,
        ),
    )
    collect_data(Agent(), Env(), cfg, step=0)
    # sample_actions: the id list is slot-aligned with the description list.
    assert seen_sample, "no sample_actions call"
    for descs, ids in seen_sample:
        assert len(descs) == len(ids) == E
        assert all(desc[i] == d for d, i in zip(descs, ids)), (descs, ids)
    # save_episode: each slot's own id, with the (shared) description alongside.
    assert sorted(tid for _, _, tid in seen_save) == ["libero_90_79", "libero_90_82"]
    for _, d, tid in seen_save:
        assert d == desc[tid]


# --------------------------------------------------------------------------- #
# 6c. Config surface (D7) + the success-buffer config swap
# --------------------------------------------------------------------------- #

def _parse(name, *args):
    return tyro.cli(OnlineTrainConfig, default=get_config(name), args=["--exp-name", "t", *args])


def test_config_num_tasks_cli_and_registered_config():
    assert _parse("pi05_libero_online_ogpo_sft", "--rl.critic.num_tasks", "None").rl.critic.num_tasks is None
    assert _parse("pi05_libero_online_ogpo_sft", "--rl.critic.num_tasks", "4").rl.critic.num_tasks == 4
    # The mt4 script emits `None` unconditionally when PER_TASK_CRITIC=0; it must
    # override the _pertask config's own value (authoritative env var contract).
    assert _parse("pi05_libero_online_ogpo_sft_pertask", "--rl.critic.num_tasks", "None").rl.critic.num_tasks is None
    pt = get_config("pi05_libero_online_ogpo_sft_pertask")
    assert isinstance(pt.rl, OGPOSFTLearnerConfig) and pt.rl.critic.num_tasks == 4
    base = get_config("pi05_libero_online_ogpo_sft")
    assert dataclasses.replace(pt.rl, critic=base.rl.critic) == base.rl, "_pertask differs from _sft only in the critic"
    assert dataclasses.replace(pt.rl.critic, num_tasks=None) == base.rl.critic


def test_config_rejects_num_tasks_outside_ogpo():
    with pytest.raises(ValueError, match="OGPOSFTLearnerConfig"):
        _parse("pi05_libero_online_aw_sft", "--rl.critic.num_tasks", "2")
    with pytest.raises(ValueError, match="OGPOSFTLearnerConfig"):
        _parse("pi05_libero_online_best_of_n", "--rl.critic.num_tasks", "2")
    with pytest.raises(ValueError, match=">= 1"):
        _parse("pi05_libero_online_ogpo_sft", "--rl.critic.num_tasks", "0")


def test_success_buffer_config_swap_preserves_num_tasks():
    # Mirrors OGPOAgentLearner.__init__'s temporary self._config swap (OGL:88-90).
    config = get_config("pi05_libero_online_ogpo_sft_pertask")
    swapped = dataclasses.replace(
        config, rl=dataclasses.replace(config.rl, buffer_capacity=config.rl.success_buffer_capacity)
    )
    assert swapped.rl.critic.num_tasks == 4


# --------------------------------------------------------------------------- #
# 6d. jit-1: the task_index sidecar (Tier-2 signature edit)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def actor_fx():
    # Reuses the split-equivalence fixture builders (dummy PaliGemma + expert).
    # NOT `from tests.ogpo...`: a site-packages package named `tests` shadows the
    # repo's tests/ (no tests/__init__.py), so that import fails at fixture setup.
    # pytest imports siblings as `ogpo.<module>` (tests/ogpo/__init__.py exists).
    import functools
    import importlib
    _se = importlib.import_module("ogpo.test_split_equivalence")
    _B, _build_config, _build_critics, _build_policy_state, _make_params, _original_recompute = (
        _se._B, _se._build_config, _se._build_critics, _se._build_policy_state,
        _se._make_params, _se._original_recompute,
    )
    from src.rl.ogpo.update_actor import sample_and_advantage
    import openpi.shared.nnx_utils as nnx_utils

    config = _build_config()
    assert _B == 2
    model = config.model.create(jax.random.key(0))
    params = _make_params(config, model)
    mesh = sharding.make_mesh(1)
    q_state, value_state = _build_critics(config, model, mesh, jax.random.key(1))
    policy_observation = config.model.fake_obs(batch_size=_B)
    model_cast = nnx.merge(nnx.graphdef(model), params)
    model_cast.eval()
    prefix = _original_recompute(model_cast, policy_observation)
    policy_state = _build_policy_state(config, model, params, ema_params=None)
    ema = nnx.filter_state(params, config.trainable_filter)
    # Per-task critics with T=2, both slots seeded FROM the shared critics'
    # params, so routing is the only difference.
    config_pt = dataclasses.replace(
        config, rl=dataclasses.replace(config.rl, critic=dataclasses.replace(config.rl.critic, num_tasks=2))
    )
    q_pt, v_pt = _build_critics(config_pt, model, mesh, jax.random.key(1))

    def _fill(pt_state, shared_state):
        flat = dict(pt_state.params.flat_state())
        for p, v in shared_state.params.flat_state():
            for t in range(2):
                flat[("tasks", t) + tuple(p)] = v
        new_params = nnx.State.from_flat_path(flat)
        return dataclasses.replace(pt_state, params=new_params, ema_params=new_params)

    return dict(
        config=config, sa_jit=jax.jit(functools.partial(sample_and_advantage, config)),
        policy_state=policy_state, q_state=q_state, value_state=value_state,
        q_pt=_fill(q_pt, q_state), v_pt=_fill(v_pt, value_state),
        policy_observation=policy_observation, prefix=prefix, ema=ema,
    )


def _sa(fx, q, v, *extra):
    rng = jax.random.key(7)
    return fx["sa_jit"](rng, fx["policy_state"], q, v, fx["policy_observation"], fx["prefix"], fx["ema"], *extra)


def test_jit1_task_index_none_is_identity(actor_fx):
    ref = _sa(actor_fx, actor_fx["q_state"], actor_fx["value_state"])
    got = _sa(actor_fx, actor_fx["q_state"], actor_fx["value_state"], None)
    for a, b in zip(jax.tree.leaves(ref), jax.tree.leaves(got)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_jit1_routes_each_state_to_its_own_task_critics(actor_fx):
    task_index = jnp.asarray([0, 1], jnp.int32)
    ref = _sa(actor_fx, actor_fx["q_state"], actor_fx["value_state"])
    same = _sa(actor_fx, actor_fx["q_pt"], actor_fx["v_pt"], task_index)
    # Both slots hold the shared critic's params -> identical advantage for any routing.
    np.testing.assert_allclose(np.asarray(same[5]), np.asarray(ref[5]), atol=1e-6)
    # Perturb slot 1 only: row 0 (task 0) must not move; row 1 (task 1) must.
    q_pt = actor_fx["q_pt"]
    bumped = q_pt.params.map(lambda p, v: v.replace(v.value * 1.5) if p[1] == 1 else v)
    q_bumped = dataclasses.replace(q_pt, params=bumped, ema_params=bumped)
    out = _sa(actor_fx, q_bumped, actor_fx["v_pt"], task_index)
    adv_ref, adv_out = np.asarray(ref[5]), np.asarray(out[5])
    np.testing.assert_allclose(adv_out[0], adv_ref[0], atol=1e-6)
    assert not np.isclose(adv_out[1], adv_ref[1], atol=1e-6)
