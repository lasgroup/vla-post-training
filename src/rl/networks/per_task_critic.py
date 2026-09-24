# ruff: noqa: F722
"""Per-task critic ensembles: T disjoint copies of any critic, routed by task.

Implements the per-task-critic treatment for multi-task OGPO
(``docs/changes/2026-08-21-per-task-critics/``): each task gets its own Q and V
ensemble with NO shared parameters, and a sample of task ``t`` is only ever
scored by task ``t``'s parameters — in the critic update and in the advantage.

The wrapper is backbone-agnostic on purpose. The learner already selects the
critic as a ``critic_def`` closure (BroNet or the pi0-backbone MLP,
``src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py:52-63``);
this module instantiates ``num_tasks`` copies of whatever it is given, runs all
of them on the batch and gathers per sample by ``observation["task_index"]``.
The output keeps today's ``(n_heads, batch[, bins])`` contract, so everything
downstream of ``create_critic`` (``summarize_critic_values``,
``critic_values_per_head``, the advantage branches, the burst) is untouched.

Why a hard gather and not a soft one-hot mix: the gather selects exactly one
task's output per sample, so (a) no other task's parameters enter the value
used for policy extraction, and (b) the cotangent of every non-selected
sub-critic is exactly zero — gradients are disjoint by construction, with no
loss mask. Both are pinned by ``tests/ogpo/test_per_task_critics.py``.

Cost: ``num_tasks`` x the critic forward/backward. Critics are small MLPs, so
this is cheap on the default mt4 path and accepted for the 4-task frontier; it
is not sized for tens of tasks.
"""
from collections.abc import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from src.rl.networks.rl_networks import ActionType, ObsType

TASK_INDEX_NAME = "task_index"


def _task_index(observation: ObsType, num_tasks: int) -> jax.Array:
    if not isinstance(observation, (dict, FrozenDict)) or TASK_INDEX_NAME not in observation:
        raise KeyError(
            f"Per-task critics (rl.critic.num_tasks={num_tasks}) need "
            f"observation[{TASK_INDEX_NAME!r}] (int32 [b]) on every critic call, "
            "but the critic observation has no such key. Every path that builds a "
            "critic observation must thread the buffer's task_index through; there "
            "is deliberately no default slot (decision D4)."
        )
    idx = jnp.asarray(observation[TASK_INDEX_NAME])
    if idx.ndim != 1:
        raise ValueError(
            f"observation[{TASK_INDEX_NAME!r}] must be rank-1 [b], got shape {idx.shape}."
        )
    return idx.astype(jnp.int32)


def _gather_task(outs: jax.Array, idx: jax.Array) -> jax.Array:
    """``outs``: (T, n, b[, bins]); ``idx``: (b,) -> (n, b[, bins]) selecting outs[idx[i], :, i]."""
    b = idx.shape[0]
    shape = (1, 1, b) + (1,) * (outs.ndim - 3)
    idx_full = jnp.broadcast_to(idx.reshape(shape), (1,) + outs.shape[1:])
    return jnp.take_along_axis(outs, idx_full, axis=0)[0]


def _split_rngs(rngs: nnx.Rngs, num_tasks: int) -> list[nnx.Rngs]:
    # One independent key per sub-critic, derived from the caller's stream, so
    # the T copies start from different initializations.
    keys = jax.random.split(rngs(), num_tasks)
    return [nnx.Rngs(k) for k in keys]


class PerTaskStateActionCritic(nnx.Module):
    """``num_tasks`` disjoint Q ensembles; ``__call__`` routes by task_index."""

    def __init__(
        self,
        critic_def: Callable[[ObsType, ActionType, nnx.Rngs], nnx.Module],
        num_tasks: int,
        observation: ObsType,
        action: ActionType,
        *,
        rngs: nnx.Rngs,
    ):
        assert num_tasks >= 1, f"num_tasks must be >= 1, got {num_tasks}"
        self.num_tasks = int(num_tasks)
        # The sub-critics see the observation as-is: both backbones read only
        # their declared vector keys (BroNet `_obs_vector_keys`, MLPEncoder
        # `state_vector_keys`) and ignore `task_index`.
        self.tasks = [
            critic_def(observation, action, task_rngs)
            for task_rngs in _split_rngs(rngs, self.num_tasks)
        ]

    def __call__(
        self, observation: ObsType, action: ActionType, training: bool = False
    ) -> jnp.ndarray:
        idx = _task_index(observation, self.num_tasks)
        outs = jnp.stack(
            [net(observation, action, training=training) for net in self.tasks], axis=0
        )  # (T, n, b[, bins])
        return _gather_task(outs, idx)


class PerTaskStateValue(nnx.Module):
    """``num_tasks`` disjoint V ensembles; ``__call__`` routes by task_index."""

    def __init__(
        self,
        critic_def: Callable[[ObsType, nnx.Rngs], nnx.Module],
        num_tasks: int,
        observation: ObsType,
        *,
        rngs: nnx.Rngs,
    ):
        assert num_tasks >= 1, f"num_tasks must be >= 1, got {num_tasks}"
        self.num_tasks = int(num_tasks)
        self.tasks = [
            critic_def(observation, task_rngs)
            for task_rngs in _split_rngs(rngs, self.num_tasks)
        ]

    def __call__(self, observation: ObsType, training: bool = False) -> jnp.ndarray:
        idx = _task_index(observation, self.num_tasks)
        outs = jnp.stack(
            [net(observation, training=training) for net in self.tasks], axis=0
        )  # (T, n, b[, bins])
        return _gather_task(outs, idx)
