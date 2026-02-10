from __future__ import annotations

import dataclasses
import gc
from typing import Any, Dict, Mapping, Protocol, cast

import numpy as np
from flax.training import checkpoints
from flax.training.train_state import TrainState
import jax

from src.rl_training.common import (
    eval_actions_jit,
    eval_log_prob_jit,
    eval_mse_jit,
    eval_reward_function_jit,
    sample_actions_jit,
    DatasetDict,
)

from collections import namedtuple
import openpi.training.utils as training_utils

StepData = namedtuple(
    "StepData", ["obs", "action", "next_obs", "reward", "terminate", "truncate"]
)


class ActorLike(Protocol):
    def infer(
        self,
        element: Mapping[str, Any],
        *,
        sharding_spec: Any,
    ) -> Mapping[str, Any]: ...

    def infer_with_model(self, **kwargs: Any) -> Mapping[str, Any]: ...


class DatasetLike(Protocol):
    @property
    def num_episodes(self) -> int: ...

    def add_frame(self, frame: Mapping[str, Any]) -> None: ...

    def save_episode(self) -> None: ...


def get_batch_stats(actor):
    if hasattr(actor, "batch_stats"):
        return actor.batch_stats
    else:
        return None


class Agent(object):
    _actor: TrainState | training_utils.TrainState | ActorLike | None = None
    _critic: TrainState
    _rng: jax.random.PRNGKey
    _dataset: DatasetLike | None = None
    training_steps: int = 0
    env_steps: int = 0
    episodes: int = 0
    base_dir: str = "/agent"
    _phase: str = "training"
    _offloaded_training_state: dict[str, Any] | None = None

    @property
    def actor(self) -> TrainState | training_utils.TrainState | ActorLike:
        if self._actor is None:
            raise NotImplementedError(
                "Agent.actor is not configured. Set `self._actor` or override `actor`."
            )
        return self._actor

    @actor.setter
    def actor(self, actor: TrainState | training_utils.TrainState | ActorLike) -> None:
        self._actor = actor

    @property
    def dataset(self) -> DatasetLike:
        if self._dataset is None:
            raise NotImplementedError(
                "Agent.dataset is not configured. Set `self._dataset` or override `dataset`."
            )
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: DatasetLike) -> None:
        self._dataset = dataset

    def infer(
        self, observations: Mapping[str, Any], *, sharding_spec: Any = None
    ) -> Mapping[str, Any]:
        actor = self.actor
        if not hasattr(actor, "infer"):
            raise NotImplementedError(
                "Agent.actor does not implement `infer(...)`; cannot run collection inference."
            )
        return cast(ActorLike, actor).infer(observations, sharding_spec=sharding_spec)

    def infer_with_model(self, **kwargs: Any) -> Mapping[str, Any]:
        actor = self.actor
        if not hasattr(actor, "infer_with_model"):
            raise NotImplementedError(
                "Agent.actor does not implement `infer_with_model(...)`; cannot run model-conditioned inference."
            )
        return cast(ActorLike, actor).infer_with_model(**kwargs)

    def _require_train_actor(self) -> TrainState:
        actor = self.actor
        if (
            isinstance(actor, training_utils.TrainState)
            or not hasattr(actor, "apply_fn")
            or not hasattr(actor, "params")
        ):
            raise NotImplementedError(
                "Agent.actor must be a flax TrainState-like actor for this method."
            )
        return cast(TrainState, actor)

    def eval_actions(self, observations: np.ndarray | Dict) -> np.ndarray:
        actor = self._require_train_actor()
        actions = eval_actions_jit(
            actor.apply_fn,
            actor.params,
            observations,
            get_batch_stats(actor),
        )
        return np.asarray(actions)

    def eval_log_probs(self, batch: DatasetDict | Dict) -> float:
        actor = self._require_train_actor()
        return eval_log_prob_jit(
            actor.apply_fn,
            actor.params,
            get_batch_stats(actor),
            batch,
        )

    def eval_mse(self, batch: DatasetDict) -> float:
        actor = self._require_train_actor()
        return eval_mse_jit(
            actor.apply_fn,
            actor.params,
            get_batch_stats(actor),
            batch,
        )

    def eval_reward_function(self, batch: DatasetDict) -> float:
        actor = self._require_train_actor()
        return eval_reward_function_jit(
            actor.apply_fn, actor.params, actor.batch_stats, batch
        )

    def sample_actions(
        self, observations: np.ndarray | Dict, batch_actions: bool = True
    ) -> np.ndarray:
        actor = self._require_train_actor()
        rng, actions = sample_actions_jit(
            self._rng,
            actor.apply_fn,
            actor.params,
            observations,
            get_batch_stats(actor),
        )

        self._rng = rng
        return np.asarray(actions)

    @property
    def _save_dict(self):
        return None

    def save_checkpoint(self, step, keep_every_n_steps: int | None = None):
        checkpoints.save_checkpoint(
            self.base_dir,
            self._save_dict,
            step,
            prefix="checkpoint",
            overwrite=False,
            keep_every_n_steps=keep_every_n_steps,
        )

    def restore_checkpoint(self, dir):
        raise NotImplementedError

    def add_data(self, step_data: StepData):
        raise NotImplementedError

    def save_episode(self, is_success: bool = False, env_index: int = 0):
        self.episodes += 1
        return True

    def start_data_collection(self) -> bool:
        """Offload training-only state to CPU before data collection."""
        train_state = getattr(self, "_train_state", None)
        if train_state is None:
            self._phase = "collect"
            return True

        offloaded: dict[str, Any] = {}
        state_for_collection = train_state

        if hasattr(state_for_collection, "opt_state") and state_for_collection.opt_state is not None:
            offloaded_opt_state = jax.device_get(state_for_collection.opt_state)
            offloaded["opt_state"] = offloaded_opt_state
            state_for_collection = dataclasses.replace(
                state_for_collection,
                opt_state=offloaded_opt_state,
            )

        if hasattr(state_for_collection, "ema_params") and state_for_collection.ema_params is not None:
            offloaded_ema_params = jax.device_get(state_for_collection.ema_params)
            offloaded["ema_params"] = offloaded_ema_params
            state_for_collection = dataclasses.replace(
                state_for_collection,
                ema_params=offloaded_ema_params,
            )

        if offloaded:
            self._offloaded_training_state = offloaded
            self._train_state = state_for_collection
            gc.collect()

        self._phase = "collect"
        return True

    def start_training(self) -> bool:
        """Restore offloaded training state to devices before optimization."""
        train_state = getattr(self, "_train_state", None)
        offloaded = self._offloaded_training_state or {}
        if not offloaded or train_state is None:
            self._phase = "train"
            return True

        sharding = getattr(self, "_train_state_sharding", None)
        restored_fields: dict[str, Any] = {}
        for key, value in offloaded.items():
            field_sharding = getattr(sharding, key, None) if sharding is not None else None
            restored_fields[key] = (
                jax.device_put(value, field_sharding)
                if field_sharding is not None
                else jax.device_put(value)
            )

        self._train_state = dataclasses.replace(train_state, **restored_fields)
        jax.block_until_ready(self._train_state)
        self._offloaded_training_state = None
        gc.collect()
        self._phase = "train"
        return True

    def update(self):
        raise NotImplementedError
