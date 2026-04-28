#!/usr/bin/env python3
"""Shared-image smoke for the docker-testing pi0.5 stack."""

from __future__ import annotations

import os
import platform
import sys
import traceback

import jax
import torch


def _stage(name: str) -> None:
    print(f"\n=== {name} ===", flush=True)


def main() -> int:
    _stage("runtime")
    print(f"python={sys.version.split()[0]} platform={platform.platform()}")
    print(f"jax={jax.__version__}")
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"cuda_device0={torch.cuda.get_device_name(0)}")
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH')}")
    print(f"LIBERO_CONFIG_PATH={os.environ.get('LIBERO_CONFIG_PATH')}")

    _stage("imports")
    try:
        import libero  # noqa: F401
        from openpi.models import pi0_config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.shared import nnx_utils
        from openpi.training.config import get_config as get_openpi_config
        from src.training.config import get_config as get_repo_config

        print("OK import libero/openpi/src.training")
    except Exception:
        print("IMPORT_FAILED")
        traceback.print_exc()
        return 1

    _stage("repo_config")
    try:
        repo_cfg = get_repo_config("pi05_libero_online_filtered_sft")
        obs_spec, action_spec = repo_cfg.model.inputs_spec(batch_size=1)
        fake_obs = repo_cfg.model.fake_obs(batch_size=1)
        print(f"repo_config={repo_cfg.name}")
        print(f"repo_model_type={repo_cfg.model.model_type.value}")
        print(f"repo_action_horizon={repo_cfg.model.action_horizon}")
        print(f"repo_obs_images={sorted(obs_spec.images.keys())}")
        print(f"repo_state_shape={obs_spec.state.shape}")
        print(f"repo_action_shape={action_spec.shape}")
        print(f"repo_fake_prompt_shape={fake_obs.tokenized_prompt.shape}")
    except Exception:
        print("REPO_CONFIG_FAILED")
        traceback.print_exc()
        return 1

    _stage("openpi_debug_pi05")
    try:
        debug_cfg = get_openpi_config("debug_pi05")
        debug_model = debug_cfg.model.create(jax.random.key(0))
        debug_obs = debug_cfg.model.fake_obs(batch_size=1)
        debug_actions = nnx_utils.module_jit(
            debug_model.sample_actions,
            static_argnames=("num_steps",),
        )(
            jax.random.key(1),
            debug_obs,
            num_steps=1,
        )
        debug_actions = jax.device_get(debug_actions)
        print(f"debug_config={debug_cfg.name}")
        print(f"debug_action_shape={debug_actions.shape}")
        print(f"debug_action_dtype={debug_actions.dtype}")
    except Exception:
        print("OPENPI_DEBUG_FAILED")
        traceback.print_exc()
        return 1

    _stage("pytorch_pi05")
    try:
        pt_cfg = pi0_config.Pi0Config(
            pi05=True,
            dtype="float32",
            paligemma_variant="dummy",
            action_expert_variant="dummy",
        )
        policy = PI0Pytorch(pt_cfg)
        print(f"pytorch_policy={policy.__class__.__name__}")
        print(f"pytorch_parameters={sum(p.numel() for p in policy.parameters())}")
    except Exception:
        print("PYTORCH_PI05_FAILED")
        traceback.print_exc()
        return 1

    print("PI05_DOCKER_TESTING_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
