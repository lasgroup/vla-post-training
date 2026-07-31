# ruff: noqa: E402
"""Entry point: OGPO online post-training with the PaliGemma LLM backbone
UNFROZEN and only the SigLIP vision tower FROZEN.

This is a thin wrapper around scripts/exp.py. It registers a new config
``pi05_libero_online_ogpo_sft_unfrozen_backbone`` at runtime -- a copy of the
frozen ``pi05_libero_online_ogpo_sft`` config with a different ``freeze_filter``
-- and then delegates to the exact same ``main()`` and CLI as scripts/exp.py.
Nothing in src/training/config.py is edited.

Why a wrapper instead of editing config.py:
  * All the fragile process setup (mp "spawn" start method, warning/logging
    filters, progress-bar suppression) and the whole training loop live in
    scripts/exp.py. Importing it keeps a single source of truth, so this
    variant can never drift from the frozen run.

Freeze filter (``.*PaliGemma/img.*``) freezes ONLY the SigLIP image tower.
Trainable parameters after the filter:
  * PaliGemma Gemma LLM backbone (LLM stack index 0)   <-- UNFROZEN here
  * action expert (LLM stack index 1)
  * action heads: state_proj, action_in_proj, action_time_mlp_in/out,
    action_out_proj

Contrast with src/training/config.py:_make_ogpo_freeze_filter, which ALSO
freezes the LLM backbone (``.*llm.*`` minus the action expert). Frozen params
are cast to bfloat16 by init_train_state; the backbone we leave trainable here
stays fp32 and carries optimizer state, so this run uses substantially more
memory than the frozen OGPO config.
"""
import dataclasses
import os
import sys

# Make this script's own directory (scripts/) importable so ``import exp``
# resolves to scripts/exp.py regardless of how the process was launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importing exp runs its module-level process setup exactly once and hands us
# the same main() + config module the frozen run uses.
from exp import main
import openpi.shared.nnx_utils as nnx_utils
import src.training.config as _config

FROZEN_CONFIG_NAME = "pi05_libero_online_ogpo_sft"
UNFROZEN_CONFIG_NAME = "pi05_libero_online_ogpo_sft_unfrozen_backbone"


def _make_siglip_only_freeze_filter():
    """Freeze only the SigLIP vision tower (``.*PaliGemma/img.*``), leaving the
    PaliGemma LLM backbone, the action expert, and the action heads trainable."""
    return nnx_utils.PathRegex(".*PaliGemma/img.*")


def _register_unfrozen_config() -> None:
    """Register a copy of the frozen OGPO config that only freezes SigLIP.

    Uses dataclasses.replace so EVERY other hyperparameter is identical to
    ``pi05_libero_online_ogpo_sft``; only ``name`` and ``freeze_filter`` change.
    """
    if UNFROZEN_CONFIG_NAME in _config._CONFIGS_DICT:
        return
    base = _config.get_config(FROZEN_CONFIG_NAME)
    unfrozen = dataclasses.replace(
        base,
        name=UNFROZEN_CONFIG_NAME,
        freeze_filter=_make_siglip_only_freeze_filter(),
    )
    _config._CONFIGS.append(unfrozen)
    _config._CONFIGS_DICT[UNFROZEN_CONFIG_NAME] = unfrozen


if __name__ == "__main__":
    _register_unfrozen_config()
    main(_config.cli())
