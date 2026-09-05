"""Policy construction for the online learners.

Wraps ``openpi.policies.policy_config.create_trained_policy`` so that a LoRA
model config can be seeded from a checkpoint that has no adapter weights.
"""

import logging
import pathlib
import re

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.policies.policy_config as policy_config
import openpi.shared.download as download
import openpi.transforms as transforms
from openpi.training import checkpoints as _checkpoints


def _uses_lora(model_config) -> bool:
    return "lora" in getattr(model_config, "paligemma_variant", "") or "lora" in getattr(
        model_config, "action_expert_variant", ""
    )


def _restore_params_with_fresh_lora(model_config, params_dir: pathlib.Path, rng: jax.Array):
    """Restore checkpoint params and add the adapter weights it does not carry.

    ``BaseModelConfig.load`` requires the params to match the model state
    exactly, so a ``*_lora`` variant cannot be loaded from the released pi05
    checkpoint. Sample the missing ``lora_a``/``lora_b`` entries the way the
    model itself would (``LoRAConfig.init_fn`` is ``normal(stddev=0.01)``).
    """
    params = _model.restore_params(params_dir, dtype=jnp.bfloat16)

    abstract_model = nnx.eval_shape(model_config.create, jax.random.key(0))
    _, state = nnx.split(abstract_model)
    flat_ref = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
    flat = traverse_util.flatten_dict(params, sep="/")

    missing = sorted(k for k in flat_ref if k not in flat and re.fullmatch(".*lora.*", k))
    for key in missing:
        rng, init_rng = jax.random.split(rng)
        flat[key] = 0.01 * jax.random.normal(init_rng, flat_ref[key].shape, dtype=jnp.bfloat16)
    logging.info("Initialized %d LoRA params missing from %s", len(missing), params_dir)

    return traverse_util.unflatten_dict(flat, sep="/")


def create_collection_policy(
    train_config,
    checkpoint_dir: pathlib.Path | str,
    *,
    seed: int = 0,
) -> _policy.Policy:
    """``create_trained_policy`` that also works for a LoRA model config.

    Non-LoRA configs are delegated unchanged. For a LoRA config the JAX branch of
    ``create_trained_policy`` is repeated here with the adapter-aware restore --
    the upstream helper takes a directory, not params, and openpi is a submodule.
    """
    if not _uses_lora(train_config.model):
        return policy_config.create_trained_policy(train_config, checkpoint_dir)

    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    logging.info("Loading model (LoRA adapters initialized fresh)...")
    model = train_config.model.load(
        _restore_params_with_fresh_lora(
            train_config.model, checkpoint_dir / "params", jax.random.key(seed)
        )
    )

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
    )
