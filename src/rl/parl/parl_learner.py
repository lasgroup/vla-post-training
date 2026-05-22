import functools
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

from src.rl.agent import Agent
from src.rl.filtered_sft_agent.filtered_sft_learner import FilteredSFTLearner
from src.rl.filtered_sft_agent.update import train_step as sft_train_step
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig, PARLConfig


class PARLLearner(Agent):
    """PARL wrapper: alternates between base RL updates and periodic SFT anchoring.

    Every `parl.frequency` calls to `update()`, the policy is fine-tuned via SFT
    on a separate buffer of (optionally success-only) trajectories instead of
    running the base RL update.  This prevents policy drift and anchors exploration
    back towards high-quality demonstrations.
    """

    def __getattr__(self, name: str):
        # Fall through to the base agent for any attribute not defined on PARLLearner.
        # This makes _checkpoint_manager, _online_data_buffer, _resuming, etc. visible
        # to runtime helpers (save_epoch_state, init_wandb, ...) without re-exposing them.
        # Guard on "_base" prevents infinite recursion if __init__ hasn't run yet.
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)

    def __init__(self, base_agent: FilteredSFTLearner, config: OnlineTrainConfig):
        assert config.parl is not None, "OnlineTrainConfig.parl must be set to use PARLLearner"
        self._base = base_agent
        self._parl_cfg: PARLConfig = config.parl
        env_num = config.collect.env_num

        dummy_data = base_agent._make_buffer_dummy_data()
        self._parl_buffer = ShardedReplayBuffer(
            dummy_data=dummy_data,
            max_capacity=self._parl_cfg.buffer_capacity,
            data_sharding=base_agent._data_sharding,
            seed=config.seed + 999,
            preprocess_fn=None,
            postprocess_fn=None,
            freeze_dict=False,
        )

        self._parl_episode_storage: list[list] = [[] for _ in range(env_num)]
        self._parl_step_count: int = 0

        # Jitted SFT step reusing FilteredSFT train_step with base agent sharding.
        # donate_argnums=(1,) avoids copying the large train state.
        self._sft_train_step = jax.jit(
            functools.partial(sft_train_step, config),
            in_shardings=(
                base_agent._replicated_sharding,
                base_agent._train_state_sharding,
                base_agent._data_sharding,
            ),
            out_shardings=(base_agent._train_state_sharding, base_agent._replicated_sharding),
            donate_argnums=(1,),
        )

    # training_steps is a property rather than an instance attribute so it always
    # reflects the base agent's counter.  Without this, __getattr__ would be called
    # for reads (returning base.training_steps) but writes would create a shadow
    # instance attribute on PARLLearner that diverges from the base.
    @property
    def training_steps(self) -> int:
        return self._base.training_steps

    @training_steps.setter
    def training_steps(self, value: int):
        self._base.training_steps = value

    # --- Inference ---

    def eval_actions(self, observations: np.ndarray | Dict, **kwargs) -> np.ndarray:
        return self._base.eval_actions(observations, **kwargs)

    def sample_actions(self, observations: np.ndarray | Dict, **kwargs):
        return self._base.sample_actions(observations, **kwargs)

    # --- Persistence ---

    def save_checkpoint(self, step: int | None = None):
        return self._base.save_checkpoint(step)

    # --- Data collection ---

    def add_data(self, step_data: StepData):
        self._base.add_data(step_data)
        # Maintain a separate episode storage because the base's _episode_storage
        # is cleared inside save_episode; we need our own copy to write into
        # the PARL buffer after the base has already consumed and cleared its data.
        for i in range(len(self._parl_episode_storage)):
            self._parl_episode_storage[i].append(
                jax.tree.map(lambda x, _i=i: x[_i], step_data)
            )

    def save_episode(self, is_success: bool, env_index: int, task_description: str):
        # Let the base agent handle its own buffer (with its own filtering logic).
        self._base.save_episode(is_success, env_index, task_description)

        store_all = not self._parl_cfg.store_success_only
        if store_all or is_success:
            episode_data = self._parl_episode_storage[env_index]
            if episode_data:
                # Reuse base preprocessing via target_buffer= to avoid duplicating
                # the windowing / action-chunking / transform logic.
                self._base._save_episode_in_buffer(
                    episode_data,
                    task_description,
                    is_success=is_success,
                    target_buffer=self._parl_buffer,
                )
        self._parl_episode_storage[env_index] = []

    def start_data_collection(self, step: int | None = None):
        self._base.start_data_collection(step)
        self._parl_episode_storage = [[] for _ in range(len(self._parl_episode_storage))]

    def end_data_collection(self, step: int | None = None) -> int:
        result = self._base.end_data_collection(step)
        self._parl_episode_storage = [[] for _ in range(len(self._parl_episode_storage))]
        return result

    # --- Training ---

    @at.typecheck
    def update(self) -> dict:
        self._parl_step_count += 1
        parl_cfg = self._parl_cfg
        batch_size = parl_cfg.batch_size or self._base._config.batch_size

        is_parl_step = (
            self._parl_step_count % parl_cfg.frequency == 0
            and self._parl_buffer.size >= batch_size
        )

        if is_parl_step:
            # Manually bump the step counter because we are not calling base.update().
            self._base.training_steps += 1
            info: dict = {}
            for _ in range(parl_cfg.num_updates):
                batch_raw = self._parl_buffer.sample(batch_size=batch_size)
                sft_batch = self._base._online_batch_to_sft_batch(batch_raw)
                sft_rng, self._base._rng = jax.random.split(self._base._rng)
                with sharding.set_mesh(self._base._mesh):
                    new_state, step_info = self._sft_train_step(
                        sft_rng, self._base._train_state, sft_batch
                    )
                # donate_argnums=(1,) on _sft_train_step donates the old train_state
                # buffer; replace immediately so the donated reference is not reused.
                self._base._train_state = new_state
                for k, v in step_info.items():
                    info[f"parl/{k}"] = v
            info["parl_buffer_size"] = jnp.asarray(
                float(self._parl_buffer.size), dtype=jnp.float32
            )
            info["online_buffer_size"] = jnp.asarray(
                float(self._base._online_data_buffer.size), dtype=jnp.float32
            )
            return info

        info = self._base.update()
        info["parl_buffer_size"] = jnp.asarray(
            float(self._parl_buffer.size), dtype=jnp.float32
        )
        return info
