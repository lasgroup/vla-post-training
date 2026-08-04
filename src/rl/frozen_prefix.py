# ruff: noqa: F722
from __future__ import annotations

import gc
import logging

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.weight_loaders as _weight_loaders

PARAM_DTYPES = {
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
}


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Load and validate the weights. Returns a loaded subset of the weights.

    Mirrors the identical helpers in ``filtered_sft_learner`` / ``dsrl_env``; kept
    local so this module has no dependency on either learner.
    """
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True
    )
    # Drop the jax.ShapeDtypeStruct placeholders so only loaded params are returned.
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


class FrozenPrefixBackbone:
    """Pretrained backbone params held off-device, lent out for a collection round.

    Use ``activate()`` / ``deactivate()`` around data collection; ``model`` is only
    valid in between. Nothing here is ever updated or checkpointed -- the params are
    reloaded from the same fixed checkpoint on every run, including after a requeue.
    """

    def __init__(
        self,
        graphdef: nnx.GraphDef[_model.BaseModel],
        host_params: nnx.State,
        replicated_sharding: jax.sharding.Sharding,
    ):
        self._graphdef = graphdef
        self._host_params = host_params
        self._replicated_sharding = replicated_sharding
        self._device_params: nnx.State | None = None
        self._model: _model.BaseModel | None = None

    @property
    def is_active(self) -> bool:
        return self._model is not None

    @property
    def host_params(self) -> nnx.State:
        """The off-device params. Exposed for diagnostics; do not mutate."""
        return self._host_params

    @property
    def model(self) -> _model.BaseModel:
        if self._model is None:
            raise RuntimeError(
                "The frozen prefix backbone is not on device. It is only materialised "
                "between start_data_collection() and end_data_collection()."
            )
        return self._model

    def activate(self) -> None:
        """Replicate the params onto the accelerators and build the model."""
        if self._model is not None:
            return
        self._device_params = jax.device_put(
            self._host_params, self._replicated_sharding
        )
        model = nnx.merge(self._graphdef, self._device_params)
        model.eval()
        self._model = model

    def deactivate(self) -> None:
        """Drop the device copy. The host copy is kept for the next round."""
        self._model = None
        self._device_params = None
        gc.collect()


def load_frozen_prefix_backbone(
    config,
    *,
    replicated_sharding: jax.sharding.Sharding,
) -> FrozenPrefixBackbone:
    """Build the frozen backbone from ``collect.frozen_prefix_*`` and park it on the host.

    The params are materialised once on device (to reuse the accelerator for model
    creation and the dtype cast) and immediately pulled back to host memory, so the
    steady-state accelerator cost outside collection is zero.
    """
    param_dtype = PARAM_DTYPES['bfloat16']
    weight_loader = config.weight_loader

    # A fixed key rather than the learner's rng: every param is overwritten by the
    # checkpoint below, and this keeps the learner's rng stream identical to a run
    # with the frozen backbone disabled.
    init_rng = jax.random.key(config.seed)

    def init(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> tuple[nnx.GraphDef[_model.BaseModel], nnx.State]:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            nnx.replace_by_pure_dict(state, partial_params)
            model = nnx.merge(graphdef, state)
        return nnx.graphdef(model), nnx.state(model)

    def init_and_cast(
        rng: at.KeyArrayLike, partial_params: at.Params | None = None
    ) -> tuple[nnx.GraphDef[_model.BaseModel], nnx.State]:
        graphdef, params = init(rng, partial_params)
        # Inference only, so cast every float leaf -- not just config.freeze_filter's.
        params = jax.tree.map(
            lambda x: x.astype(param_dtype)
            if jnp.issubdtype(x.dtype, jnp.floating)
            else x,
            params,
        )
        return graphdef, params

    # Validate against the *uncast* shapes: the checkpoint stores float32 params.
    _, params_shape = jax.eval_shape(init, init_rng)
    partial_params = _load_weights_and_validate(
        weight_loader, nnx.to_pure_dict(params_shape)
    )
    # No donate_argnums here: the restored checkpoint buffers are not donatable, so it
    # only produces a long "donated buffers were not usable" warning at startup.
    graphdef, device_params = jax.jit(
        init_and_cast,
        in_shardings=replicated_sharding,
        out_shardings=replicated_sharding,
    )(init_rng, partial_params)

    host_params = jax.device_get(device_params)
    del device_params
    gc.collect()

    n_params = sum(int(x.size) for x in jax.tree.leaves(host_params))
    logging.info(
        "Loaded frozen prefix backbone: %.2fB params (%.1f GiB replicated per device)",
        n_params / 1e9,
        n_params * np.dtype(param_dtype).itemsize / 2**30,
    )
    return FrozenPrefixBackbone(graphdef, host_params, replicated_sharding)
