import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module_name", ["scripts.exp", "scripts.eval"])
def test_actual_subproc_worker_overrides_inherited_gpu_jax_platform(
    tmp_path, module_name
):
    driver = tmp_path / "worker_bootstrap_probe.py"
    driver.write_text(
        f"""import importlib
import json
import os

importlib.import_module({module_name!r})
from src.envs.venv import SubprocEnvWorker


class ProbeEnv:
    def __init__(self):
        self.boot_environment = {{
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        }}

    def close(self):
        return None


def make_probe_env():
    return ProbeEnv()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "7"
    worker = SubprocEnvWorker(make_probe_env)
    try:
        print(json.dumps(worker.get_env_attr("boot_environment")))
    finally:
        worker.close()
"""
    )
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, str(driver)],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {
        "cuda_visible_devices": "0",
        "jax_platforms": "cpu",
        "mujoco_egl_device_id": "0",
    }
