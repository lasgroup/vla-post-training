"""Config-surface tests for backbone LoRA (docs/changes/2026-08-29-backbone-lora/).

Pure-Python: filter polarity is checked with ``nnx.filterlib.to_predicate``
against literal param paths (the ``test_ema_utils.py`` synthetic pattern taken
one step cheaper — no module at all). The path list mirrors the real
``gemma_2b_lora`` Pi0 tree enumerated in the change record's BLAST-RADIUS.md;
tree-level coverage on an actual (dummy-width) LoRA model lives in
``test_backbone_lora_init.py`` / ``test_backbone_lora_grads.py``.
"""

import dataclasses

import flax.nnx as nnx
import pytest

import src.training.config as _config
from src.training.config import OGPOSFTLearnerConfig, OnlineTrainConfig, _make_ogpo_freeze_filter

# The 10 adapter leaves of the gemma_2b_lora backbone (LLM stack 0). The mlp
# names are flat ("gating_einsum_lora_a"), the attn names nested — both shapes
# appear in the real tree.
_ADAPTER_PATHS = [
    ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"),
    ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_b"),
    ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "lora_a"),
    ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "lora_b"),
    ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "lora_a"),
    ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "lora_b"),
    ("PaliGemma", "llm", "layers", "mlp", "gating_einsum_lora_a"),
    ("PaliGemma", "llm", "layers", "mlp", "gating_einsum_lora_b"),
    ("PaliGemma", "llm", "layers", "mlp", "linear_lora_a"),
    ("PaliGemma", "llm", "layers", "mlp", "linear_lora_b"),
]

# Stack-0 base leaves (must stay frozen in BOTH filter modes).
_BASE_LLM_PATHS = [
    ("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"),
    ("PaliGemma", "llm", "layers", "attn", "kv_einsum", "w"),
    ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "w"),
    ("PaliGemma", "llm", "layers", "mlp", "gating_einsum"),
    ("PaliGemma", "llm", "layers", "mlp", "linear"),
    ("PaliGemma", "llm", "layers", "pre_attention_norm", "scale"),
    ("PaliGemma", "llm", "embedder", "input_embedding"),
    ("PaliGemma", "llm", "final_norm", "scale"),
]

# Action expert (stack 1, `_1` suffix) — trainable in both modes.
_EXPERT_PATHS = [
    ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"),
    ("PaliGemma", "llm", "layers", "mlp_1", "gating_einsum"),
    ("PaliGemma", "llm", "final_norm_1", "scale"),
]

# SigLIP — frozen in both modes.
_SIGLIP_PATHS = [
    ("PaliGemma", "img", "embedding", "kernel"),
    ("PaliGemma", "img", "head", "kernel"),
    ("PaliGemma", "img", "pos_embedding"),
    ("PaliGemma", "img", "Transformer", "encoderblock", "MlpBlock_0", "Dense_0", "kernel"),
]

# Action heads — trainable in both modes.
_HEAD_PATHS = [
    ("state_proj", "kernel"),
    ("action_in_proj", "kernel"),
    ("action_time_mlp_in", "kernel"),
    ("action_time_mlp_out", "kernel"),
    ("action_out_proj", "kernel"),
]

_LEAF = nnx.VariableState(nnx.Param, 0.0)


def _frozen(filter_, path):
    return nnx.filterlib.to_predicate(filter_)(path, _LEAF)


def test_lora_filter_leaves_adapters_trainable():
    f = _make_ogpo_freeze_filter(allow_lora=True)
    for p in _ADAPTER_PATHS:
        assert not _frozen(f, p), f"adapter frozen under allow_lora=True: {p}"
    for p in _BASE_LLM_PATHS + _SIGLIP_PATHS:
        assert _frozen(f, p), f"should stay frozen under allow_lora=True: {p}"
    for p in _EXPERT_PATHS + _HEAD_PATHS:
        assert not _frozen(f, p), f"should stay trainable under allow_lora=True: {p}"


def test_old_filter_freezes_the_adapters():
    # Pins the trap this change guards against: the plain OGPO filter's
    # `.*llm.*`-minus-`_1` branch captures every adapter leaf, so a LoRA
    # variant without the flag trains nothing new — silently.
    f = _make_ogpo_freeze_filter()
    for p in _ADAPTER_PATHS:
        assert _frozen(f, p), f"expected the plain filter to freeze {p}"


def test_lora_filter_matches_plain_filter_on_lora_less_paths():
    # "Inert-but-safe under a variant flip": on a tree with no lora leaves the
    # two modes are indistinguishable — the property Pi0Config.get_freeze_filter
    # lacks (it returns nnx.Nothing for gemma_2b, unfreezing everything).
    plain = _make_ogpo_freeze_filter()
    lora = _make_ogpo_freeze_filter(allow_lora=True)
    for p in _BASE_LLM_PATHS + _EXPERT_PATHS + _SIGLIP_PATHS + _HEAD_PATHS:
        assert _frozen(plain, p) == _frozen(lora, p), p


def test_backbone_lora_rewrites_model_and_filter():
    base = _config.get_config("pi05_libero_online_ogpo_sft")
    cfg = dataclasses.replace(base, backbone_lora=True)
    assert cfg.model.paligemma_variant == "gemma_2b_lora"
    # dataclasses.replace-not-restate: the base model's fields survive.
    assert cfg.model.pi05 is True
    assert cfg.model.action_horizon == 10
    assert cfg.model.discrete_state_input is False
    assert not _frozen(cfg.freeze_filter, _ADAPTER_PATHS[0])
    assert _frozen(cfg.freeze_filter, _BASE_LLM_PATHS[0])
    assert _frozen(cfg.freeze_filter, _SIGLIP_PATHS[0])


def test_backbone_lora_rewrite_is_idempotent():
    # tyro re-instantiates the dataclass from the registered default, so
    # __post_init__ re-runs on its own output.
    cfg = dataclasses.replace(
        _config.get_config("pi05_libero_online_ogpo_sft"), backbone_lora=True
    )
    again = dataclasses.replace(cfg)
    assert again.model.paligemma_variant == "gemma_2b_lora"
    assert not _frozen(again.freeze_filter, _ADAPTER_PATHS[0])


def test_lora_variant_without_the_flag_raises():
    base = _config.get_config("pi05_libero_online_ogpo_sft")
    with pytest.raises(ValueError, match="--backbone_lora"):
        dataclasses.replace(
            base,
            model=dataclasses.replace(base.model, paligemma_variant="gemma_2b_lora"),
        )


def test_backbone_lora_on_a_non_ogpo_config_raises():
    base = _config.get_config("pi05_libero_online_aw_sft")
    with pytest.raises(ValueError, match="OGPOSFTLearnerConfig"):
        dataclasses.replace(base, backbone_lora=True)


def test_every_registered_config_has_backbone_lora_off():
    online = [c for c in _config._CONFIGS if isinstance(c, OnlineTrainConfig)]
    assert online, "no OnlineTrainConfigs registered?"
    for cfg in online:
        assert cfg.backbone_lora is False, f"{cfg.name} has backbone_lora on by default"
        if isinstance(cfg.rl, OGPOSFTLearnerConfig):
            # Registered OGPO configs keep the plain (adapter-freezing) filter.
            assert _frozen(cfg.freeze_filter, _ADAPTER_PATHS[0]), cfg.name


def test_mt4_recipe_lora_knob_resolves(tmp_path):
    # DRY=1 emits the command instead of running it; re-parsing it through the
    # real tyro CLI proves the flag SPELLING parses (tyro renders the bool as a
    # flag pair — `--backbone_lora True` would not) and that LORA=1/0 resolve
    # to the right config. Harness borrowed from test_verifier_alignment.
    import importlib

    va = importlib.import_module("ogpo.test_verifier_alignment")
    on = va._resolve(va._dry("ogpo_multitask_4task.sh", tmp_path, LORA="1"))
    assert on.backbone_lora is True
    assert on.model.paligemma_variant == "gemma_2b_lora"
    off = va._resolve(va._dry("ogpo_multitask_4task.sh", tmp_path, LORA="0"))
    assert off.backbone_lora is False
    assert off.model.paligemma_variant == "gemma_2b"
    unset = va._resolve(va._dry("ogpo_multitask_4task.sh", tmp_path))
    assert unset.backbone_lora is False


def test_policy_twin_config_is_lora_less():
    # Mirrors the create_trained_policy twin construction in
    # FilteredSFTLearner.__init__ (keep in sync): backbone_lora=False must ride
    # in the replace, or __post_init__ rewrites the variant right back.
    base = _config.get_config("pi05_libero_online_ogpo_sft")
    cfg = dataclasses.replace(base, backbone_lora=True)
    twin = dataclasses.replace(
        cfg,
        backbone_lora=False,
        model=dataclasses.replace(
            cfg.model,
            paligemma_variant=cfg.model.paligemma_variant.removesuffix("_lora"),
        ),
    )
    assert twin.model.paligemma_variant == "gemma_2b"
