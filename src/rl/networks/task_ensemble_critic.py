# ruff: noqa: F722
"""Per-task critic ensembles for the privileged-critic diagnostic.

A single critic shared across tasks has to spend capacity separating tasks
before it can rank actions within one. These modules give every training task
its OWN critic (its own parameters, end to end) and select the right one per
sample from a task id carried in the observation dict.

All T task critics are evaluated and the per-sample selection is a one-hot
contraction, rather than gathering per-sample parameters: the gather's
[batch, in, out] intermediate would dwarf the T-fold forward for any realistic
batch. Only the selected task's parameters receive gradient — the others'
outputs are multiplied by zero — so this is exactly T independent critics
trained on their own task's transitions.
"""
from typing import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.training.optimizer as _optimizer

from src.rl.networks.bronet_critic import BroNetStateActionCritic, BroNetStateValue
from src.rl.networks.rl_networks import ObsType, ActionType


# Key in the critic observation dict holding the per-sample task index (float,
# one column). Not `privileged_task_id`: by the time the batch reaches the
# critic the buffer keys have been remapped onto the critic's own obs schema.
TASK_ID_KEY = "task_id"


def _select_task(
    per_task: jnp.ndarray, task_id: jnp.ndarray, num_tasks: int
) -> jnp.ndarray:
    """Contract the leading task axis of ``per_task`` against the per-sample id.

    ``per_task`` is ``(T, n, B, ...)`` (T tasks, n ensemble heads, B samples);
    the result is ``(n, B, ...)``. A task id outside ``[0, T)`` — the sentinel
    for a task with no head, e.g. a held-out eval task — one-hots to the zero
    vector and yields exactly 0, rather than silently borrowing another task's
    critic.
    """
    ids = jnp.round(jnp.reshape(task_id, (-1,))).astype(jnp.int32)
    one_hot = jax.nn.one_hot(ids, num_tasks, dtype=per_task.dtype)  # (B, T)
    return jnp.einsum("tnb...,bt->nb...", per_task, one_hot)


class TaskEnsembleStateActionCritic(nnx.Module):
    """``num_tasks`` independent :class:`BroNetStateActionCritic` ensembles.

    Output shape matches the single-task module — ``(num_qs, B)`` for
    ``num_bins == 1``, ``(num_qs, B, num_bins)`` otherwise — so every consumer
    (``summarize_critic_values``, ``critic_values_per_head``, the advantage
    combinations) is unchanged.
    """

    def __init__(
        self,
        observation: ObsType,
        action: ActionType,
        hidden_dim: int,
        depth: int,
        num_qs: int,
        num_tasks: int,
        num_bins: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")
        self.num_tasks = int(num_tasks)
        self.nets = [
            BroNetStateActionCritic(
                observation=observation,
                action=action,
                hidden_dim=hidden_dim,
                depth=depth,
                num_qs=num_qs,
                num_bins=num_bins,
                rngs=rngs,
            )
            for _ in range(self.num_tasks)
        ]

    def __call__(
        self, observation: ObsType, action: ActionType, training: bool = False
    ) -> jnp.ndarray:
        per_task = jnp.stack(
            [net(observation, action, training=training) for net in self.nets], axis=0
        )
        return _select_task(per_task, observation[TASK_ID_KEY], self.num_tasks)


class TaskEnsembleStateValue(nnx.Module):
    """``num_tasks`` independent :class:`BroNetStateValue` ensembles."""

    def __init__(
        self,
        observation: ObsType,
        hidden_dim: int,
        depth: int,
        num_vs: int,
        num_tasks: int,
        num_bins: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")
        self.num_tasks = int(num_tasks)
        self.nets = [
            BroNetStateValue(
                observation=observation,
                hidden_dim=hidden_dim,
                depth=depth,
                num_vs=num_vs,
                num_bins=num_bins,
                rngs=rngs,
            )
            for _ in range(self.num_tasks)
        ]

    def __call__(self, observation: ObsType, training: bool = False) -> jnp.ndarray:
        per_task = jnp.stack(
            [net(observation, training=training) for net in self.nets], axis=0
        )
        return _select_task(per_task, observation[TASK_ID_KEY], self.num_tasks)


# Attribute on the ensemble modules holding the T sub-critics, hence the second
# element of every parameter path: ("nets", <task>, ...). `per_task_optimizer`
# reads the task index from there.
TASK_SUBTREE_ATTR = "nets"


def _task_of_path(path) -> int:
    """Task index of a parameter, from its pytree key path."""
    if not path or getattr(path[0], "key", None) != TASK_SUBTREE_ATTR:
        raise ValueError(
            f"Expected a per-task critic parameter path rooted at "
            f"{TASK_SUBTREE_ATTR!r}, got {path!r}. per_task_optimizer must only be "
            "given a TaskEnsemble* parameter tree."
        )
    return int(path[1].key)


def clip_by_global_norm_per_task(
    max_norm: float, num_tasks: int
) -> optax.GradientTransformation:
    """``optax.clip_by_global_norm`` applied to each task's subtree separately.

    The T critics live in ONE parameter tree, so one global norm over the whole
    tree would make task A's gradient magnitude throttle task B's update — the
    tasks would be coupled through the clip even though their parameters and
    gradients are disjoint. Clipping per subtree gives each task exactly the
    update a standalone critic would get.

    Matches ``optax.clip_by_global_norm``'s rule (scale by ``min(1, c/||g||)``)
    per task, and is stateless like it.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        sq = [jnp.zeros((), jnp.float32) for _ in range(num_tasks)]
        for path, g in jax.tree_util.tree_flatten_with_path(updates)[0]:
            t = _task_of_path(path)
            sq[t] = sq[t] + jnp.sum(jnp.square(g.astype(jnp.float32)))
        # max() rather than a where(): a task absent from the batch has an
        # all-zero gradient, and 0/0 would poison its (zero) update with NaN.
        scales = [
            jnp.minimum(1.0, max_norm / jnp.maximum(jnp.sqrt(s), 1e-6)) for s in sq
        ]
        scaled = jax.tree_util.tree_map_with_path(
            lambda path, g: (g * scales[_task_of_path(path)]).astype(g.dtype), updates
        )
        return scaled, state

    return optax.GradientTransformation(init_fn, update_fn)


def per_task_optimizer(
    optimizer_cfg, lr_schedule_cfg, num_tasks: int
) -> optax.GradientTransformation:
    """AdamW with a PER-TASK gradient clip instead of one global clip.

    Mirrors ``openpi/src/openpi/training/optimizer.py:78-85`` (AdamW.create),
    substituting the clip. openpi is a quarantined submodule, so the core is
    rebuilt here from the same config fields rather than patched there.
    """
    if not isinstance(optimizer_cfg, _optimizer.AdamW):
        raise NotImplementedError(
            "per_task_optimizer mirrors openpi's AdamW.create; the critic optimizer "
            f"is {type(optimizer_cfg).__name__}."
        )
    core = optax.adamw(
        lr_schedule_cfg.create(),
        b1=optimizer_cfg.b1,
        b2=optimizer_cfg.b2,
        eps=optimizer_cfg.eps,
        weight_decay=optimizer_cfg.weight_decay,
        mask=None,
    )
    return optax.chain(
        clip_by_global_norm_per_task(optimizer_cfg.clip_gradient_norm, num_tasks),
        core,
    )


def make_task_ensemble_defs(
    *,
    hidden_dim: int,
    depth: int,
    num_qs: int,
    num_vs: int,
    num_tasks: int,
    num_bins: int = 1,
) -> tuple[Callable, Callable]:
    """``(state_action_critic_def, state_value_def)`` in the learner's calling convention."""

    def state_action_critic_def(observation, action, rngs):
        return TaskEnsembleStateActionCritic(
            observation=observation,
            action=action,
            hidden_dim=hidden_dim,
            depth=depth,
            num_qs=num_qs,
            num_tasks=num_tasks,
            num_bins=num_bins,
            rngs=rngs,
        )

    def state_value_def(observation, rngs):
        return TaskEnsembleStateValue(
            observation=observation,
            hidden_dim=hidden_dim,
            depth=depth,
            num_vs=num_vs,
            num_tasks=num_tasks,
            num_bins=num_bins,
            rngs=rngs,
        )

    return state_action_critic_def, state_value_def
