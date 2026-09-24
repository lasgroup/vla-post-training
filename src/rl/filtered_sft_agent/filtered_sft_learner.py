import copy
import dataclasses
import functools
import gc
import json
import logging
import weakref
from typing import Any, Dict

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from orbax.checkpoint.checkpoint_utils import construct_restore_args

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
from openpi.policies import policy_config
from openpi_client import image_tools
from src.rl.ema_utils import compose_full_params
from src.rl.filtered_sft_agent.update import train_step
from src.rl.lora_utils import zero_lora_b_params, zero_lora_params
from src.rl.replay_buffer import ShardedReplayBuffer
from src.rl.task_registry import TaskRegistry
from src.rl.types import StepData
from src.training.config import OnlineTrainConfig, FilteredSFTLearnerConfig
from src.training.data_loader import create_data_loader
from src.envs.wrappers import (
    TimeToSuccessAsRewardWrapper,
    Pi0ObservationWrapper,
    PrefixEmbeddingVectorEnvWrapper,
    QueryFrequencyWrapper,
)
from src.envs.venv import SubprocVectorEnv, DummyVectorEnv
from src.rl.agent import Agent, EnvFn
from src.rl.prefix_embedding import PREFIX_EMBEDDING_NAME
from src.training.runtime_state import (
    ResumeState,
    load_resume_state,
    resolve_resume_step,
    resume_state_path,
    step_manifest_steps,
)


def filtered_sft_wrap_env(
    env_fn: EnvFn,
    config,
    env_num: int | None = None,
):
    env_num = env_num if env_num is not None else config.collect.env_num
    replan_steps = config.collect.replan_steps
    env_class = config.collect.domain
    seed = config.seed
    env_factories = []
    for i in range(env_num):

        def _make_env(rank=i):
            # Create the base environment
            base_env = env_fn(rank)
            if config.collect.use_time_to_success_as_reward:
                base_env = TimeToSuccessAsRewardWrapper(
                    base_env, success_bonus=config.collect.success_reward_bonus
                )
            # Add Pi related obs to the environment
            base_env = Pi0ObservationWrapper(
                env=base_env,
                env_class=env_class,
            )
            # Add query-frequency wrapper to rollout action chunks.
            query_wrapper = (
                PrefixEmbeddingVectorEnvWrapper
                if config.collect.store_prefix_rep
                else QueryFrequencyWrapper
            )
            base_env = query_wrapper(
                env=base_env,
                query_frequency=replan_steps,
            )
            return base_env

        env_factories.append(_make_env)

    env = (
        SubprocVectorEnv(env_factories)
        if env_num > 1
        else DummyVectorEnv(env_factories)
    )
    # This sets the seed for all environment all at once to be [seed, seed + i, ..., seed + num_envs]
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    # re-use training seed
    return env


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True
    )

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


def _batch_axis_sharding(pytree, mesh: jax.sharding.Mesh):
    n = mesh.shape[sharding.BATCH_AXIS]
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def shard(arr):
        if n > 1 and arr.ndim >= 2:
            for axis in np.argsort(arr.shape)[::-1]:
                if arr.shape[axis] % n == 0:
                    spec = [None] * arr.ndim
                    spec[axis] = sharding.BATCH_AXIS
                    return jax.sharding.NamedSharding(
                        mesh, jax.sharding.PartitionSpec(*spec)
                    )
        return replicated

    return jax.tree.map(shard, pytree)


@at.typecheck
def init_train_state(
    config: OnlineTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, weight_decay_mask=None
    )

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            nnx.replace_by_pure_dict(state, partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # R3 (docs/changes/2026-08-29-backbone-lora/): zero `lora_b` so the
        # model at step 0 IS the loaded SFT policy. openpi's LoRAConfig has a
        # single init_fn for both factors, so as shipped `lora_b` is
        # normal(0.01) — a randomly perturbed policy at step 0. Zeroing BOTH
        # factors would dead-end adapter gradients (each factor's grad is
        # proportional to the other); a-random/b-zero is the standard LoRA
        # init. Fresh-init only by construction: the resume path returns at
        # the `if resume:` short-circuit below and orbax overwrites the whole
        # tree; the jax.eval_shape pass is shape-only (zeros_like preserves
        # shape/dtype). Exact identity on a lora-less tree.
        params = zero_lora_b_params(params)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(nnx.filter_state(params, config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # `state_sharding` is a RESTORE CONTRACT on this path, not just the jits'
        # `in_shardings`: `_restore_state_sharded` turns it into orbax restore
        # args, which is what makes the restored arrays land on THIS run's mesh
        # instead of on the device ids recorded in the checkpoint. Changing what
        # this returns changes where a resumed checkpoint is placed.
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(
        config.weight_loader, nnx.to_pure_dict(train_state_shape.params)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _params_item_sharding(
    state_sharding: training_utils.TrainState,
    trainable_filter: nnx.filterlib.Filter,
    replicated_sharding: jax.sharding.NamedSharding,
) -> nnx.State:
    """Restore sharding for the on-disk `params` (EMA) item, mirroring the save.

    `save_checkpoint` (FSL:854-870) writes that item MIXED: trainable leaves come
    from `self._ema` after a REPLICATED `device_put`, frozen and non-Param leaves
    from the FSDP-sharded `train_state.params`. Asking for the same composition on
    restore -- through the same `compose_full_params`, so the two stay symmetric by
    construction -- keeps a same-`fsdp_devices` resume placement-identical to what
    the `_sharding`-metadata fallback produced before this helper existed.

    A uniform `fsdp_sharding` target would be shorter but would change even the
    working single-GPU path: the ~11.3 GiB trainable EMA would land FSDP-sharded
    and then be all-gathered by the replicated `device_put` at FSL:447-449, an
    unmeasured peak-init spike on a path where `del ema_dev` (OGL:734) is already
    load-bearing to hold the ~34.7 GiB floor.

    Certified by tests/ogpo/test_cross_topology_resume_verifier.py (leaf-for-leaf
    against what `save_checkpoint` writes); at real pi0.5 scale, peak device memory
    and placement matched the pre-change restore at 1 and 2 GPUs
    (docs/changes/2026-09-21-fsdp-topology-resume/VERIFICATION.md, B.5).
    """
    replicated = jax.tree.map(lambda _: replicated_sharding, state_sharding.params)
    return compose_full_params(state_sharding.params, replicated, trainable_filter)


def _restore_state_sharded(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    state_sharding: training_utils.TrainState,
    *,
    trainable_filter: nnx.filterlib.Filter,
    replicated_sharding: jax.sharding.NamedSharding,
    step: int | None = None,
) -> training_utils.TrainState:
    """openpi's `restore_state` plus explicit, current-mesh `restore_args`.

    openpi's version (openpi/src/openpi/training/checkpoints.py:89-107) passes the
    restore target and NO restore args, so every leaf falls back to the sharding
    recorded in the checkpoint's `_sharding` metadata file, which orbax rebuilds
    from the SAVED device ids. Resuming under a different `fsdp_devices` then dies
    inside orbax ("sharding passed to deserialization should be ... Got None", job
    10324589, 2026-09-05) or, in the 1->2 direction, at the first jit that touches
    a leaf the current mesh wants sharded. Passing the args restores onto this
    run's mesh instead, so a run checkpointed at N devices resumes at any M.

    Annotating the target's leaves is NOT an alternative: `PyTreeCheckpointHandler`
    never derives restore args from its item (only `StandardCheckpointHandler`
    does), so the args have to be built and passed. Measured on CPU with two forced
    devices -- bare target: not resharded; annotated target: not resharded;
    explicit restore args: resharded (docs/changes/2026-09-21-fsdp-topology-resume/).

    Lives here rather than in openpi so that submodule stays clean
    (docs/code/best_practices.md:316-318). `data_loader`, which openpi's signature
    takes and immediately `del`s, is dropped.

    vs openpi's version: identical values and placement at an unchanged
    `fsdp_devices` (tests/ogpo/test_cross_topology_resume_verifier.py,
    `test_same_topology_restore_is_identical_to_openpi_restore_state`); STRICTER on
    a target-vs-stored leaf SHAPE mismatch, which now raises where openpi's returned
    the stored shape unvalidated (same file, `test_new_path_is_STRICTER_than_old_...`).
    """
    with at.disable_typechecking():
        # openpi privates, deliberately (best_practices.md:319-320): the public
        # `restore_state` has no seam for restore args, and the two-item split has
        # to be applied to the shape tree and the sharding tree in exactly the same
        # way or the args would not line up with the items. The guard is required
        # because `dataclasses.replace` inside `_split_params` re-runs TrainState's
        # beartype-checked `__init__` outside a tree-unflatten stack (which is what
        # `at._check_dataclass_annotations` whitelists), and `step: at.Int[...]`
        # rejects a NamedSharding -- openpi guards its own call the same way
        # (checkpoints.py:97).
        train_state, params = _checkpoints._split_params(state)
        train_state_sharding, _ = _checkpoints._split_params(state_sharding)
        params_sharding = _params_item_sharding(
            state_sharding, trainable_filter, replicated_sharding
        )
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                train_state=ocp.args.PyTreeRestore(
                    item=train_state,
                    restore_args=construct_restore_args(
                        train_state, train_state_sharding
                    ),
                ),
                params=ocp.args.PyTreeRestore(
                    item={"params": params},
                    restore_args=construct_restore_args(
                        {"params": params}, {"params": params_sharding}
                    ),
                ),
            ),
        )
    return _checkpoints._merge_params(restored["train_state"], restored["params"])


def _get_post_step_action_filter(domain: str):
    if domain == "libero":
        # see https://arxiv.org/pdf/2501.09747, Appendix C - clipping low-magnitude actions
        # the LIBERO dataset seems to have been filtered accordingly
        return lambda x: np.where(np.abs(x) < 0.0011, 0.0, x)
    return lambda x: x


def _get_obs_key_process_fn(domain: str):
    if domain == "libero":
        return lambda k: k.replace("observation/", "")
    return lambda k: k


class FilteredSFTLearner(Agent):
    # Per-task critics (rl.critic.num_tasks): the buffer gains a `task_index`
    # transition field and _save_episode_in_buffer stamps it from a first-seen
    # registry. Both are instance-attribute gates, NOT config reads: `rl.critic`
    # exists only on the AWR/BofN config branches, so a plain filtered-SFT run has
    # no such field and a direct read would raise (getattr defaults are banned).
    # AdvantageWeightedSFTLearner sets both BEFORE super().__init__ (the buffer is
    # allocated inside it), the same pattern as `_prefix_embed_dim` (AWR:64-67).
    _num_critic_tasks: int | None = None
    _task_registry: TaskRegistry | None = None

    def __init__(self, config: OnlineTrainConfig):
        self._config = config
        self.post_step_action_filter = _get_post_step_action_filter(self._config.collect.domain)
        self.obs_key_process_fn = _get_obs_key_process_fn(self._config.collect.domain)

        if self._config.batch_size % jax.device_count() != 0:
            raise ValueError(
                f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
            )
        jax.config.update(
            "jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser())
        )
        self._rng = jax.random.key(self._config.seed)
        init_rng, self._rng = jax.random.split(self._rng, 2)

        # set up sharding
        self._mesh = sharding.make_mesh(self._config.fsdp_devices)
        self._data_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        self._replicated_sharding = jax.sharding.NamedSharding(
            self._mesh, jax.sharding.PartitionSpec()
        )

        # Initialize checkpoint manager.
        self._checkpoint_manager, self._resuming = (
            _checkpoints.initialize_checkpoint_dir(
                self._config.checkpoint_dir,
                keep_period=self._config.keep_period,
                overwrite=not self._config.resume,
                resume=self._config.resume,
            )
        )

        # initialize data loader
        assert 0.0 <= self._config.rl.online_ratio <= 1.0, "Online ratio must be between 0 and 1."
        self._data_config = self._config.data.create(
            self._config.assets_dirs,
            self._config.model,
        )
        # note that offline data is not seeded upon resume, may induce non-determinism
        self._offline_batch_size = max(len(jax.devices()), int(self._config.batch_size * (1 - self._config.rl.online_ratio)))

        if self._config.rl.online_ratio < 1.0:
            self._data_loader = create_data_loader(
                config, batch_size=self._offline_batch_size, sharding=self._data_sharding, shuffle=True
            )
            self._data_iter = iter(self._data_loader)
        else:
            class DummyDataLoader:
                def __init__(self, data_config):
                    self._data_config = data_config

                def data_config(self):
                    return self._data_config

            self._data_loader = DummyDataLoader(self._data_config)
            self._data_iter = None

        self._online_data_buffer = self._get_online_replay_buffer()
        if self._resuming:
            self._resume_state = self._resolve_resume_state()
            self._online_data_buffer.restore_shards(
                self._resume_state.replay_shard_dir,
                rng_state_json=self._resume_state.replay_rng_state_json,
                max_step=self._resume_state.step,
            )
        self._collection_success_episodes = 0

        # Initialize train state. Both returns MUST come from this one call: on
        # resume `_restore_state_sharded` zips them, and `tx` / `ema_decay` are
        # static TrainState fields compared by identity, so a sharding tree from a
        # second `init_train_state` call raises "Mismatch custom dataclass node data".
        self._train_state, self._train_state_sharding = init_train_state(
            self._config, init_rng, self._mesh, resume=self._resuming
        )
        if self._resuming:
            self._train_state = _restore_state_sharded(
                self._checkpoint_manager,
                self._train_state,
                self._train_state_sharding,
                trainable_filter=self._config.trainable_filter,
                replicated_sharding=self._replicated_sharding,
                step=self._resume_state.step,
            )
            logging.info(
                "Restored training checkpoint from %s at committed step %d onto mesh %s",
                self._config.checkpoint_dir,
                self._resume_state.step,
                dict(self._mesh.shape),
            )
            self.training_steps = int(self._resume_state.step)
            self.total_collected_episodes = self._resume_state.total_collected_episodes
            self.set_rng_state_json(self._resume_state.agent_rng_state_json)
        else:
            self.training_steps = 0

        jax.block_until_ready(self._train_state)
        logging.info(
            f"Initialized train state:\n{training_utils.array_tree_to_info(self._train_state.params)}"
        )

        # Keep ONLY the trainable subset of the EMA on device. Frozen (SigLIP) leaves are
        # bit-invariant across the run (the optimizer touches only trainable leaves), so every
        # full-model consumer recomposes them from self._train_state.params via compose_full_params.
        # Runs unconditionally after the resume-restore block, so one slice covers fresh init
        # (ema_params == params, FSL:174) and resume (restored full-tree EMA); the template FSL:174
        # stays FULL so old full-tree on-disk EMAs structure-match. Every reachable config sets
        # ema_decay, so ema_params is a real State here (no is-not-None guard needed).
        ema_trainable = nnx.filter_state(self._train_state.ema_params, self._config.trainable_filter)
        self._ema_sharding = _batch_axis_sharding(ema_trainable, self._mesh)
        self._ema = jax.device_put(ema_trainable, self._ema_sharding)
        self._train_state = dataclasses.replace(self._train_state, ema_params=None, ema_decay=None)
        decay = self._config.ema_decay
        self._ema_update_fn = jax.jit(
            lambda ema, params: jax.tree.map(
                lambda e, p: decay * e + (1.0 - decay) * p, ema, params
            ),
            # Both operands are now the trainable-only tree, so the params sharding must be the
            # trainable slice (matches self._ema_sharding, rebuilt trainable-only above).
            in_shardings=(
                self._ema_sharding,
                nnx.filter_state(self._train_state_sharding.params, self._config.trainable_filter),
            ),
            out_shardings=self._ema_sharding,
            donate_argnums=0,
        )
        gc.collect()

        # prepare train_step
        self._refresh_train_step()

        # Create temporary episode storage
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]

        def _get_prefix_rep_with_model_fn(
            m: _model.BaseModel, observation: _model.Observation
        ):
            prefix_rep = m.get_prefix_rep(observation)
            # TODO: remove if below
            return prefix_rep[0] if isinstance(prefix_rep, tuple) else prefix_rep

        self._get_prefix_rep_with_model = nnx.jit(_get_prefix_rep_with_model_fn)

        # Create policy for data collection
        policy_checkpoint_dir = self._config.weight_loader.params_path[
            : -len("/params")
        ]
        # The model this loads is DROPPED at _drop_policy_model() below — the
        # policy exists only for its input/output transforms and norm stats.
        # Load it with a lora-less twin of the config: create_trained_policy
        # goes through BaseModelConfig.load (openpi model.py:233-240), which
        # has no lora `missing_regex` and raises on the checkpoint's missing
        # lora keys ("symmetric difference of key sets: {'lora'}"). Nothing on
        # this path reads paligemma_variant. backbone_lora=False must ride in
        # the replace, or __post_init__ rewrites the variant right back
        # (tests/ogpo/test_backbone_lora_config.py::test_policy_twin_config_is_lora_less
        # mirrors this expression — keep in sync). Unconditional: for non-LoRA
        # configs the twin is a valued no-op.
        policy_train_config = dataclasses.replace(
            self._config,
            backbone_lora=False,
            model=dataclasses.replace(
                self._config.model,
                paligemma_variant=self._config.model.paligemma_variant.removesuffix(
                    "_lora"
                ),
            ),
        )
        self._policy = policy_config.create_trained_policy(
            policy_train_config,
            policy_checkpoint_dir,
        )
        # This learner always calls `infer_with_model(...)` with the current train-state model.
        # Drop policy-owned model references to avoid keeping an extra model copy in memory.
        self._drop_policy_model()

        # prepare transforms for preprocessing episode data into model input format
        self._policy_transforms = self._get_policy_transforms(self._config.collect.domain)

    def _resolve_resume_state(self) -> ResumeState:
        """Pick the newest step that is complete on disk and load its manifest.

        The pointer manifest can name a step whose orbax commit never happened
        (killed inside save_epoch_state), so the step to resume is resolved from
        what is actually there rather than from the pointer alone.
        """
        pointer_step = None
        if resume_state_path(self._config).exists():
            pointer_step = int(load_resume_state(self._config).step)
        manifest_steps = step_manifest_steps(self._config)
        orbax_steps = {int(s) for s in self._checkpoint_manager.all_steps()}
        step = resolve_resume_step(
            orbax_steps=orbax_steps,
            manifest_steps=manifest_steps,
            required_ok=lambda s: all(p.exists() for p in self._resume_required_paths(s)),
            pointer_step=pointer_step,
        )
        if pointer_step is not None and step < pointer_step:
            logging.warning(
                "Resuming at step %d, behind the manifest pointer's step %d: that "
                "step is missing an orbax checkpoint or a learner sidecar (the "
                "process died inside save_epoch_state). Everything collected after "
                "step %d is discarded.",
                step,
                pointer_step,
                step,
            )
        elif pointer_step is not None and step > pointer_step:
            logging.warning(
                "Resuming at step %d, ahead of the manifest pointer's step %d: step "
                "%d is fully durable and the pointer refresh is what did not run "
                "(the process died between the two). Nothing is lost.",
                step,
                pointer_step,
                step,
            )
        logging.info(
            "Resume resolved to step %d (orbax steps=%s, per-step manifests=%s)",
            step,
            sorted(orbax_steps),
            sorted(manifest_steps),
        )
        return load_resume_state(
            self._config, step=step if step in manifest_steps else None
        )

    def _get_policy_transforms(self, domain: str):
        if domain == "libero":
            return _transforms.compose([*self._data_config.repack_transforms.inputs, *self._policy._input_transform.transforms])
        if domain == "molmo":
            # TODO: extract these transforms from the policy config instead of hardcoding the order here.
            # Unfortunately, the policy input transforms do not match what was used for training:
            # padding should occur before normalization, and delta actions should be considered.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            input_transforms = [
                copy.deepcopy(self._policy._input_transform.transforms[1]),  # droid inputs
                _transforms.DeltaActions(delta_action_mask),                 # delta actions
                copy.deepcopy(self._policy._input_transform.transforms[6]),  # padding
                copy.deepcopy(self._policy._input_transform.transforms[2]),  # normalize
                copy.deepcopy(self._policy._input_transform.transforms[4]),  # resizeimages
                copy.deepcopy(self._policy._input_transform.transforms[0]),  # inject prompt
                copy.deepcopy(self._policy._input_transform.transforms[5]),  # tokenizer
            ]
            return _transforms.compose([*self._data_config.repack_transforms.inputs, *input_transforms])
        raise NotImplementedError(f"Unknown domain: {domain}")

    def _drop_policy_model(self):
        # For PyTorch policies `infer_with_model` ignores the provided model and uses internal state,
        # so we cannot safely drop the internal model there.
        if getattr(self._policy, "_is_pytorch_model", False):
            return

        model = getattr(self._policy, "_model", None)
        model_ref = None
        if model is not None:
            try:
                model_ref = weakref.ref(model)
            except TypeError:
                model_ref = None

        self._policy._model = None
        # These JAX callables are created from bound model methods and can capture model state.
        if hasattr(self._policy, "_sample_actions"):
            self._policy._sample_actions = None
        if hasattr(self._policy, "_get_prefix_rep"):
            self._policy._get_prefix_rep = None

        del model
        gc.collect()

        if model_ref is not None and model_ref() is not None:
            logging.warning(
                "Policy model object is still alive after cleanup; other references remain."
            )

    def _refresh_train_step(self):
        self._train_state_sharding = sharding.fsdp_sharding(
            self._train_state, self._mesh, log=False
        )
        self._train_step = jax.jit(
            functools.partial(train_step, self._config),
            in_shardings=(
                self._replicated_sharding,
                self._train_state_sharding,
                self._data_sharding,
            ),
            out_shardings=(self._train_state_sharding, self._replicated_sharding),
            donate_argnums=(1,),
        )

    def _make_buffer_dummy_data(self) -> dict:
        obs_spec, act_spec = self._config.model.inputs_spec(batch_size=1)
        obs_spec_dict = obs_spec.to_dict()
        dummy_obs_dict = jax.tree.map(
            lambda spec: np.zeros(spec.shape, dtype=spec.dtype), obs_spec_dict
        )
        dummy_obs_dict = {k: v for k, v in dummy_obs_dict.items() if v is not None}
        if "image" in dummy_obs_dict:
            dummy_obs_dict["image"] = jax.tree.map(
                lambda v: v.astype(np.uint8), dummy_obs_dict["image"]
            )
        dummy = {
            "observations": dummy_obs_dict,
            "actions": np.zeros(act_spec.shape, dtype=act_spec.dtype),
            "reward": np.zeros((1,), dtype=np.float32),
            "mc_return": np.zeros((1,), dtype=np.float32),
            "discount": np.zeros((1,), dtype=np.float32),
            "is_success": np.zeros((1,), dtype=np.float32),
        }
        if self._num_critic_tasks is not None:
            # Per-task critics: which critic slot owns this transition. A
            # transition field (not an observation key) so it survives
            # sample(drop_obs_keys=...) and lands top-level in every batch.
            dummy["task_index"] = np.zeros((1,), dtype=np.int32)
        return dummy

    def _get_online_replay_buffer(
        self,
    ) -> ShardedReplayBuffer:
        dummy_data = self._make_buffer_dummy_data()
        logging.info(
            "Initializing online replay buffer (capacity=%d)",
            self._config.rl.buffer_capacity,
        )
        return ShardedReplayBuffer(
            dummy_data=dummy_data,
            max_capacity=self._config.rl.buffer_capacity,
            data_sharding=self._data_sharding,
            seed=self._config.seed,
            freeze_dict=False,
        )

    def _process_obs_for_pi0(
        self,
        observations: Dict,
        task_description: list[str],
    ) -> Dict[str, Any]:
        # With per-step collection enabled, each env step contains a short chunk of
        # observations. Use the most recent one for policy inference.
        obs = jax.tree_util.tree_map(lambda x: x[:, -1], observations)
        h, w = int(self._config.collect.resize_image_h), int(self._config.collect.resize_image_w)
        resize_fn = lambda x: image_tools.convert_to_uint8(image_tools.resize_with_pad(x, h, w))
        obs = {k: resize_fn(v) if "image" in k else v for k, v in obs.items()}
        obs["prompt"] = task_description
        return obs

    @staticmethod
    def _batch_transform_inputs(inputs: dict, batch_size: int) -> dict:
        for key in ("image_mask", "image_masks"):
            if key in inputs:
                inputs[key] = {k: np.full((batch_size,), bool(v), dtype=bool) for k, v in inputs[key].items()}
        for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
            if key in inputs and inputs[key] is not None:
                arr = np.asarray(inputs[key])
                if arr.ndim == 1:
                    inputs[key] = np.repeat(arr[np.newaxis], batch_size, axis=0)
        return inputs

    def _sample_action(
        self,
        observations: Dict,
        rng: jax.random.PRNGKey,
        train_state: training_utils.TrainState,
        return_prefix_rep: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        # Compose the full model from the (trainable-only, after Phase E) EMA, sourcing frozen leaves
        # from the live params. Only in the is-not-None branch — keep the else fallback byte-identical.
        params = (
            compose_full_params(
                train_state.params, train_state.ema_params, self._config.trainable_filter
            )
            if train_state.ema_params is not None
            else train_state.params
        )
        model = nnx.merge(train_state.model_def, params)
        model.eval()
        first_obs = next(iter(observations.values()), None)
        if first_obs is None:
            raise ValueError("Observation dictionary is empty.")
        first_obs = np.asarray(first_obs)
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1
        noise = jax.random.normal(
            rng, (batch_size, self._policy.action_horizon, self._policy.action_dim)
        )
        # Vector envs expect a batch dimension for actions. Policy inference
        # unbatches when batch_size == 1, so add it back for single-env runs.
        num_devices = len(jax.devices())
        sharding_spec = (
            self._data_sharding if batch_size % num_devices == 0 else None
        )
        if not return_prefix_rep:
            actions = self._policy.infer_with_model(
                model=model,
                obs=observations,
                noise=noise,
                return_prefix_rep=False,
                sharding_spec=sharding_spec,
            )["actions"]
            if batch_size == 1 and actions.ndim == 2:
                actions = actions[np.newaxis, ...]
            return actions

        # Bypass infer_with_model: _output_transform cannot handle the (actions, prefix) tuple
        inputs = self._policy._input_transform(observations)
        inputs = self._batch_transform_inputs(inputs, batch_size=batch_size)
        if sharding_spec is not None:
            inputs = jax.device_put(inputs, sharding_spec)
        observation = _model.Observation.from_dict(inputs)
        _, sample_rng = jax.random.split(rng)
        if self._config.backbone_lora:
            # R1 (docs/changes/2026-08-29-backbone-lora/): the critic's prefix
            # must come from the UNADAPTED backbone, but the ACTIONS from the
            # adapted policy — and sample_actions returns both out of ONE
            # prefix forward (openpi pi0.py). So split: actions from the full
            # model, the prefix from a second, adapter-zeroed, prefix-only
            # forward. Cost: one extra 968-token stack-0 forward per policy
            # query at batch=env_num (compute_v_t runs only the action expert,
            # so the prefix pass IS the backbone cost of a query — this
            # roughly doubles it). NOTE this fires at EVAL queries too —
            # _generate_actions passes return_prefix_rep=store_prefix_rep and
            # evaluate_policy discards the prefix — so eval pays the doubled
            # backbone cost for nothing (verifier finding F3; threading an
            # eval flag through is a deferred optimization). Gated on the
            # flag so every non-LoRA arm keeps the fused call below,
            # bit-for-bit.
            raw_actions = self._policy._sample_actions_with_model(
                m=model,
                observation=observation,
                noise=noise,
                rng=sample_rng,
                return_prefix_rep=False,
                **self._policy._sample_kwargs,
            )
            base_model = nnx.merge(train_state.model_def, zero_lora_params(params))
            base_model.eval()
            prefix = self._get_prefix_rep_with_model(m=base_model, observation=observation)
        else:
            raw_actions, prefix = self._policy._sample_actions_with_model(
                m=model,
                observation=observation,
                noise=noise,
                rng=sample_rng,
                return_prefix_rep=True,
                **self._policy._sample_kwargs,
            )
        outputs = {"state": inputs["state"], "actions": raw_actions}
        if batch_size == 1:
            outputs = jax.tree.map(lambda x: np.asarray(x[0]), outputs)
        outputs = self._policy._output_transform(outputs)
        actions = outputs["actions"]
        if batch_size == 1 and actions.ndim == 2:
            actions = actions[np.newaxis, ...]
        return actions, np.asarray(prefix, dtype=np.float32)

    def _generate_actions(
        self,
        observations: np.ndarray | Dict,
        task_description: list[str],
        task_id: list[str] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        # task_id: slot-aligned task ids from collect.py. Unused on this path --
        # the prompt is what conditions the policy; the id is consumed only by
        # the best-of-N Q-scoring override (AWR) for per-task critic routing.
        del task_id
        rng, self._rng = jax.random.split(self._rng)
        processed_obs = self._process_obs_for_pi0(
            observations, task_description=task_description
        )
        actions = self._sample_action(
            observations=processed_obs,
            rng=rng,
            train_state=self._train_state,
            return_prefix_rep=self._config.collect.store_prefix_rep,
        )
        # TODO: if store_prefix_rep is True, this will crash because (i) openpi output transforms
        # cannot process tuples and (ii) venvs do not accept tuples as input

        # TODO: check if casting is necessary
        if isinstance(actions, (tuple, list)):
            return tuple(np.asarray(x, dtype=np.float32) for x in actions)

        return np.asarray(actions, dtype=np.float32)

    def sample_actions(
        self, observations: np.ndarray | Dict, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        return self._generate_actions(observations, **kwargs)

    def _online_batch_to_sft_batch(
        self, online_batch: Dict[str, Any]
    ) -> tuple[_model.Observation, _model.Actions]:
        return (
            _model.Observation.from_dict(online_batch["observation"]),
            online_batch["actions"],
        )

    def save_extra_resume_state(self, step: int) -> dict[str, Any]:
        """Learner-owned payload for the resume manifest's `extra` block.

        Called by `save_epoch_state` before the orbax commit, so anything written
        here is durable for `step` on either side of a crash. Everything this
        learner needs already lives in the checkpoint and the online shard;
        `OGPOAgentLearner` is the only override (success buffer, its task ranges,
        and the advantage-normalizer scale).
        """
        return {}

    def _resume_required_paths(self, step: int) -> list[epath.Path]:
        """Learner-owned sidecars that must exist for `step` to be resumable.

        The resume resolver skips a step whose orbax checkpoint committed but
        whose sidecars did not. This learner writes none.
        """
        return []

    def save_checkpoint(self, step: int | None = None):
        state_to_save = self._train_state
        if self._ema is not None:
            # self._ema is trainable-only (and host-resident for OGPO). Reconstruct the FULL EMA so
            # the on-disk `params` inference item stays complete (frozen SigLIP included) -> zero
            # migration; old checkpoints, inference/eval loaders, and resume keep working.
            ema_rep = jax.device_put(self._ema, self._replicated_sharding)
            full_ema = compose_full_params(
                self._train_state.params, ema_rep, self._config.trainable_filter
            )
            state_to_save = dataclasses.replace(
                state_to_save, ema_params=full_ema, ema_decay=self._config.ema_decay
            )
        _checkpoints.save_state(
            self._checkpoint_manager, state_to_save, self._data_loader, step
        )
        self._checkpoint_manager.wait_until_finished()

    def rng_state_json(self) -> str:
        """Serialize the agent PRNG key as JSON for resumable runtime state."""
        key_data = np.asarray(jax.device_get(jax.random.key_data(self._rng)))
        return json.dumps(
            {
                "dtype": str(key_data.dtype),
                "shape": list(key_data.shape),
                "data": key_data.reshape(-1).tolist(),
            }
        )

    def set_rng_state_json(self, rng_state_json: str | None) -> None:
        """Restore the agent PRNG key from a `rng_state_json` snapshot."""
        if rng_state_json is None:
            return
        payload = json.loads(rng_state_json)
        key_data = np.array(payload["data"], dtype=payload["dtype"]).reshape(
            payload["shape"]
        )
        self._rng = jax.random.wrap_key_data(jnp.asarray(key_data))

    def add_data(self, step_data: StepData):
        for i in range(self._config.collect.env_num):
            self._episode_storage[i].append(jax.tree.map(lambda x: x[i], step_data))

    def _attach_prefix_embeddings_to_episode_data(
        self,
        episode_data: list[Dict[str, Any]],
        task_description: str,
    ) -> None:
        """Unpack `(actions, prefix)` payloads and align prefixes to transitions.

        - Replaces each `ep["action"]` payload with pure actions.
        - Writes the unpacked prefix embedding to `ep["observation"]`.
        - Writes the shifted prefix embedding to `ep["next_observation"]`
          using the next step's payload.
        - Computes the final next-observation prefix with the current model.
        """

        next_observation = jax.tree.map(
            lambda x: x[np.newaxis], episode_data[-1]["next_observation"]
        )
        processed_obs = self._process_obs_for_pi0(next_observation, task_description)
        # G4: the stored prefix rep is what the critic trains on — it must come from the FULL EMA model,
        # so compose the frozen leaves back from live params (the EMA is trainable-only after Phase E).
        # R1 (docs/changes/2026-08-29-backbone-lora/): additionally zero any
        # LoRA leaves — the critic's features come from the UNADAPTED backbone,
        # matching the per-step prefixes produced in _sample_action. Exact
        # identity on a lora-less tree.
        model = nnx.merge(
            self._train_state.model_def,
            zero_lora_params(
                compose_full_params(
                    self._train_state.params,
                    self._train_state.ema_params,
                    self._config.trainable_filter,
                )
            ),
        )
        model.eval()
        inputs = self._policy._input_transform(processed_obs)
        inputs = self._batch_transform_inputs(inputs, batch_size=1)
        observation = _model.Observation.from_dict(inputs)
        next_prefix = self._get_prefix_rep_with_model(m=model, observation=observation)
        next_prefix = np.asarray(next_prefix, dtype=np.float32)
        if next_prefix.ndim == 3:
            next_prefix = next_prefix[0]
        next_prefix = next_prefix.reshape((-1, next_prefix.shape[-1])).mean(axis=0)

        for idx in reversed(range(len(episode_data))):
            ep = episode_data[idx]
            ep["action"], prefix = ep["action"]
            prefix = np.asarray(prefix, dtype=np.float32)
            prefix = prefix.reshape((-1, prefix.shape[-1])).mean(axis=0)
            horizon = next(iter(ep["observation"].values())).shape[0]
            ep["observation"][f"observation/{PREFIX_EMBEDDING_NAME}"] = np.repeat(
                prefix[None, ...], horizon, axis=0
            )
            ep["next_observation"][f"observation/{PREFIX_EMBEDDING_NAME}"] = np.repeat(
                next_prefix[None, ...], horizon, axis=0
            )
            next_prefix = prefix

    def save_episode(
        self, is_success: bool, env_index: int, task_description: str, task_id: str | None = None
    ):

        assert env_index in range(
            len(self._episode_storage)
        ), f"env_index must be between 0 and {len(self._episode_storage) - 1}, but got {env_index}."
        # extract episode data from storage and empty it
        episode_data = self._episode_storage[env_index]
        self._episode_storage[env_index] = []
        # filtered SFT keeps only successful episodes.
        if is_success:
            self._save_episode_in_buffer(
                episode_data, task_description, is_success=True, task_id=task_id
            )

    def _task_slot(self, task_id: str | None) -> int:
        """Critic slot for ``task_id`` under per-task critics (registry on).

        Keyed on the task ID (``libero_90_79``), NOT on ``task_description``:
        libero_90 language strings are not unique (79 and 82 share one), and a
        prompt-keyed registry silently merged them (verifier finding F1). A
        missing id is a caller bug, never a fallback to the prompt.
        """
        if task_id is None:
            raise ValueError(
                "Per-task critics (rl.critic.num_tasks) need the task ID, but "
                "task_id=None was passed. collect.py must pass "
                "task_id=current_task_ids[env_index] to save_episode and "
                "task_id=current_task_ids to sample_actions."
            )
        return self._task_registry.index_for(str(task_id))

    def _save_episode_in_buffer(
        self, episode_data, task_description, is_success: bool = False, target_buffer=None, task_id=None
    ):
        # target_buffer allows PARL (and other wrappers) to redirect an episode
        # into a separate buffer without subclassing or duplicating preprocessing.

        assert isinstance(self._config.rl, FilteredSFTLearnerConfig), (
            "Only Filtered SFT config should be passed " "to the filtered SFT agent"
        )

        if self._config.collect.store_prefix_rep:
            self._attach_prefix_embeddings_to_episode_data(
                episode_data, task_description=task_description
            )

        # concatenate all chunks
        episode_data = jax.tree_util.tree_map(
            lambda *xs: np.concatenate(xs, axis=0), *episode_data
        )
        done = np.logical_or(episode_data["terminate"], episode_data["truncate"])
        terminate = episode_data["terminate"]  # only true terminations should zero out the discount
        n_steps = np.where(done)[0][0] + 1
        act_h = int(self._config.model.action_horizon)
        last_gamma = float(self._config.rl.discount**act_h)
        all_gammas = np.array([self._config.rl.discount**i for i in range(n_steps)])
        w_gammas = all_gammas[:act_h]
        n_windows = n_steps - act_h + 1
        if n_windows <= 0:
            return

        # in order to optimize memory usage, we join observations and next observations
        # into a single array, and store it directly
        # `observation` and `next_observation` are the same series shifted by one
        _full_obs = {
            self.obs_key_process_fn(k): np.concatenate(
                [v[:n_steps], episode_data["next_observation"][k][n_steps - 1 : n_steps]], axis=0
            )
            for k, v in episode_data["observation"].items()
        }
        _actions = np.stack([episode_data["action"][start : start + act_h] for start in range(n_windows)])
        _actions = self.post_step_action_filter(_actions)
        _reward = np.asarray([(episode_data["reward"][start : start + act_h] * w_gammas).sum() for start in range(n_windows)])
        # Bootstrap after truncation: only zero out discount on true termination, not truncation.
        # Truncated episodes ended due to time limit — the next state still has value, so we bootstrap.
        _discount = np.asarray([0.0 if np.any(terminate[start : start + act_h]) else last_gamma for start in range(n_windows)])
        _mc_return = ((all_gammas * episode_data["reward"][:n_steps])[::-1].cumsum()[::-1] / all_gammas)[:n_windows]

        # if the reward is constant, set the MC returns to reward/(1-gamma)
        if self._config.collect.fix_mc_returns and np.all(episode_data["reward"] == episode_data["reward"][0]):
            _mc_return = np.full_like(_mc_return, episode_data["reward"][0] / (1 - self._config.rl.discount))

        def transform(input):
            obs = self._policy_transforms(input)
            actions = obs.pop("actions")
            return obs, actions

        prefix_emb = _full_obs.pop(self.obs_key_process_fn(f"observation/{PREFIX_EMBEDDING_NAME}"), None)
        actions_padded = np.concatenate([_actions, np.repeat(_actions[-1:], act_h, axis=0)], axis=0)
        _full_obs, actions_out = transform(
            {**_full_obs, "actions": actions_padded, "prompt": str(task_description)}
        )
        if prefix_emb is not None:
            _full_obs[PREFIX_EMBEDDING_NAME] = prefix_emb
        _actions = actions_out[:n_windows]

        obs_index = np.arange(n_windows, dtype=np.int64)
        next_obs_index = obs_index + act_h
        _is_success = np.full((n_windows,), float(is_success), dtype=np.float32)
        # Per-task critics: stamp the slot for this episode's task ID. First-seen
        # assignment; a task beyond rl.critic.num_tasks raises (no fallback
        # slot). Idempotent, so OGPO's success-buffer pass followed by the
        # online-buffer pass for the same episode yields the same slot.
        if self._task_registry is not None:
            _task_index = np.full((n_windows,), self._task_slot(task_id), dtype=np.int32)
        # Subclasses (e.g. Best-of-N with cached prefixes) may declare observation keys
        # that are never read during training; drop them so what we store matches the
        # buffer schema built by _make_buffer_dummy_data. Base class declares none.
        for k in getattr(self, "_buffer_obs_drop_keys", ()):
            _full_obs.pop(k, None)
        buf = target_buffer if target_buffer is not None else self._online_data_buffer
        insert_data = {
            "observations": _full_obs,
            "obs_index": obs_index,
            "next_obs_index": next_obs_index,
            "actions": _actions.astype(np.float32),
            "reward": _reward.astype(np.float32),
            "mc_return": _mc_return.astype(np.float32),
            "discount": _discount.astype(np.float32),
            "is_success": _is_success,
        }
        if self._task_registry is not None:
            insert_data["task_index"] = _task_index
        buf.insert(insert_data)
        if target_buffer is None:
            self._collection_success_episodes += 1

    def start_data_collection(self, step: int | None = None):
        # Reset episode storage
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0
        assert self._train_state.ema_params is None, "EMA parameters should be offloaded except during data collection."
        ema_rep = jax.device_put(self._ema, self._replicated_sharding)
        self._train_state = dataclasses.replace(self._train_state, ema_params=ema_rep)

    def end_data_collection(self, step: int | None = None) -> int:
        # Per-task critics: after a COLLECTION round (step given; eval calls this
        # without one) every collect.tasks id must own a slot. Collection cycles
        # through all of collect.tasks, so an unfilled registry means an id<->slot
        # drift (e.g. ids collapsing onto one key) -- the 3-critics-for-4-tasks
        # failure, caught here instead of 100k steps later.
        if self._task_registry is not None and step is not None:
            n_reg, n_slots = len(self._task_registry), self._num_critic_tasks
            if n_reg != n_slots:
                raise ValueError(
                    f"Per-task critics: after the collection round at step {step} the "
                    f"task registry holds {n_reg} of {n_slots} slots "
                    f"({sorted(self._task_registry.tasks)}) but collect.tasks has "
                    f"{sorted(set(self._config.collect.tasks))}. Every train task must be "
                    "collected and registered in the first round; check that collect.py "
                    "passes task_id and that num_tasks == len(set(collect.tasks))."
                )
        collected_episodes = int(self._collection_success_episodes)
        # Reset episode storage and counter for the next collection round.
        self._episode_storage = [[] for _ in range(self._config.collect.env_num)]
        self._collection_success_episodes = 0
        # offload EMA
        assert self._train_state.ema_params is not None, "EMA parameters should be on device during data collection."
        self._train_state = dataclasses.replace(self._train_state, ema_params=None)
        gc.collect()
        return collected_episodes

    def update(self):
        self.training_steps += 1
        update_policy = (
            self.training_steps >= self._config.rl.policy.training_start_step
            and self.training_steps % self._config.rl.policy.update_interval == 0
        )
        if not update_policy:
            return {"online_buffer_size": self._online_data_buffer.size}

        if self._online_data_buffer.size == 0:
            return {}
        online_ratio = self._config.rl.online_ratio
        if online_ratio < 1.0:
            batch = next(self._data_iter)
        if online_ratio > 0.0:
            online_batch_size = int(self._config.batch_size * min(1.0, online_ratio))
            online_batch_raw = self._online_data_buffer.sample(
                batch_size=online_batch_size
            )
            online_batch = self._online_batch_to_sft_batch(online_batch_raw)
            batch = (
                online_batch
                if online_ratio >= 1.0
                else jax.tree.map(
                    lambda x, y: jnp.concatenate([x, y], axis=0),
                    batch,
                    online_batch,
                )
            )

        train_rng, self._rng = jax.random.split(self._rng)
        rl_config = self._config.rl
        assert isinstance(rl_config, FilteredSFTLearnerConfig)
        with sharding.set_mesh(self._mesh):
            policy_state, info = self._train_step(train_rng, self._train_state, batch)
        self._train_state = policy_state
        # _ema_update_fn now maps over the trainable-only tree; slice the params arg to match.
        self._ema = self._ema_update_fn(
            self._ema,
            nnx.filter_state(self._train_state.params, self._config.trainable_filter),
        )
        info = info | {
            "online_buffer_size": jnp.asarray(
                float(self._online_data_buffer.size), dtype=jnp.float32
            )
        }
        return info
