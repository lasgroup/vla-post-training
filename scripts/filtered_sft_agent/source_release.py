"""Fail-closed verification for an immutable external source release."""

from __future__ import annotations

import hashlib
import json
import os.path
import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Mapping

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CREDENTIAL_KEY_RE = re.compile(r"TOKEN|PASSWORD|SECRET|API_?KEY|AUTHORIZATION")
_REQUIRED_SUBMODULES = {"molmospaces", "openpi"}
_REQUIRED_FILES = {
    "scripts/filtered_sft_agent/create_fixed_eval_manifests.py",
    "scripts/filtered_sft_agent/eval_trained_policy_manifest.py",
    "scripts/filtered_sft_agent/preloaded_sft_exp.py",
    "scripts/filtered_sft_agent/probe_libero_init_state_counts.py",
    "scripts/filtered_sft_agent/source_release.py",
    "src/envs/__init__.py",
    "src/envs/libero.py",
    "src/envs/venv.py",
    "src/envs/wrappers.py",
    "src/rl/agent.py",
    "src/rl/filtered_sft_agent/filtered_sft_learner.py",
    "src/rl/filtered_sft_agent/update.py",
    "src/rl/prefix_embedding.py",
    "src/rl/replay_buffer.py",
    "src/rl/types.py",
    "src/training/collect.py",
    "src/training/config.py",
    "src/training/data_loader.py",
    "src/training/runtime_state.py",
    "src/training/utils.py",
}
_RELEASE_METADATA_NAMES = {".git"}
_RUNTIME_CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}
_EDF_TOP_LEVEL_KEYS = {"image", "mounts", "workdir", "env"}
_EDF_ENVIRONMENT = {
    "PATH": "/.venv/bin:${PATH}",
    "PYTHONPATH": "/workspace:/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src",
    "LD_LIBRARY_PATH": "/usr/lib64:${LD_LIBRARY_PATH:-}",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "EGL_PLATFORM": "surfaceless",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for immutable source verification")
    return value


def _runtime_file(path_value: Any, *, field: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"source release runtime.{field} is missing")
    candidate = Path(path_value)
    if candidate.is_symlink():
        raise ValueError(f"source release runtime.{field} must not be a symlink")
    path = candidate.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"source release runtime.{field} is not a regular file")
    return path


def _confined_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"invalid source-release path: {relative_path!r}")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(
            f"source-release file must not be a symlink: {relative_path!r}"
        )
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"source-release path escapes root: {relative_path!r}"
        ) from error
    if not path.is_file():
        raise ValueError(f"source-release path is not a regular file: {path}")
    return path


def _source_inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for current_root, directory_names, file_names in os.walk(root):
        current = Path(current_root)
        for directory_name in directory_names:
            directory_path = current / directory_name
            if directory_path.is_symlink():
                raise ValueError(
                    "source release contains a directory symlink: "
                    f"{directory_path.relative_to(root).as_posix()}"
                )
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _RELEASE_METADATA_NAMES
            and name not in _RUNTIME_CACHE_DIR_NAMES
        )
        for file_name in sorted(file_names):
            if file_name in _RELEASE_METADATA_NAMES or file_name.endswith(
                (".pyc", ".pyo")
            ):
                continue
            path = current / file_name
            relative_path = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f"source release contains a symlink: {relative_path}")
            if not path.is_file():
                raise ValueError(
                    f"source release contains a non-regular file: {relative_path}"
                )
            inventory[relative_path] = _sha256_file(path)
    return inventory


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _gitlinks(root: Path) -> dict[str, str]:
    gitlinks: dict[str, str] = {}
    for line in _git(root, "ls-files", "--stage").splitlines():
        metadata, path = line.split("\t", 1)
        mode, sha, stage = metadata.split()
        if mode == "160000":
            if stage != "0":
                raise ValueError(f"source release has an unmerged gitlink: {path}")
            gitlinks[path] = sha
    return gitlinks


def _verify_git_checkout(
    root: Path,
    *,
    source_sha: str,
    source_parent_sha: str,
    submodules: Mapping[str, str],
) -> None:
    if _git(root, "rev-parse", "HEAD") != source_sha:
        raise ValueError("source checkout HEAD does not match source_sha")
    if _git(root, "rev-parse", "HEAD^1") != source_parent_sha:
        raise ValueError("source checkout parent does not match source_parent_sha")
    if _git(root, "symbolic-ref", "-q", "HEAD", check=False):
        raise ValueError("source checkout must be detached at the pinned commit")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout is dirty")
    if _gitlinks(root) != dict(submodules):
        raise ValueError("source checkout gitlinks do not match receipt submodules")
    for submodule_path, expected_sha in sorted(submodules.items()):
        nested_root = root / submodule_path
        if _git(nested_root, "rev-parse", "HEAD") != expected_sha:
            raise ValueError(
                f"submodule {submodule_path} HEAD does not match receipt identity"
            )
        if _git(nested_root, "symbolic-ref", "-q", "HEAD", check=False):
            raise ValueError(f"submodule {submodule_path} must be detached")
        if _git(nested_root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"submodule {submodule_path} is dirty")


def _reject_credential_keys(value: object, *, path: str = "") -> None:
    if not isinstance(value, Mapping):
        return
    for key, nested_value in value.items():
        key_text = str(key)
        key_path = f"{path}.{key_text}" if path else key_text
        if _CREDENTIAL_KEY_RE.search(key_text.upper()):
            raise ValueError(
                f"source release runtime EDF contains a credential-bearing key: {key_path}"
            )
        _reject_credential_keys(nested_value, path=key_path)


def _verify_edf_definition(
    edf_definition: Mapping[str, Any],
    *,
    root: Path,
    receipt: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    _reject_credential_keys(edf_definition)
    if set(edf_definition) != _EDF_TOP_LEVEL_KEYS:
        raise ValueError(
            "source release runtime EDF has unsupported top-level keys: "
            f"{sorted(edf_definition)}"
        )
    source_host_path_value = receipt.get("source_host_path")
    if not isinstance(source_host_path_value, str) or not source_host_path_value:
        raise ValueError("source release source_host_path is missing")
    source_host_path = Path(source_host_path_value).resolve(strict=True)
    if not source_host_path.is_dir() or not source_host_path.samefile(root):
        raise ValueError(
            "source release source_host_path is not the mounted source root"
        )
    expected_mounts = [
        f"{source_host_path}:/workspace:ro",
        "/capstor",
        "/iopsstor",
    ]
    if edf_definition.get("image") != runtime.get("image_path"):
        raise ValueError("source release runtime EDF image does not match receipt")
    if edf_definition.get("mounts") != expected_mounts:
        raise ValueError(
            "source release runtime EDF mounts do not match release layout"
        )
    if edf_definition.get("workdir") != "/workspace":
        raise ValueError("source release runtime EDF workdir must be /workspace")
    if edf_definition.get("env") != _EDF_ENVIRONMENT:
        raise ValueError(
            "source release runtime EDF environment is not the exact allowlist"
        )


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"source release {field} must be a JSON object")
    return value


def verify_source_release() -> dict[str, Any]:
    root = Path(_required_env("VLA_SOURCE_ROOT")).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"VLA_SOURCE_ROOT is not a directory: {root}")
    receipt_path = Path(_required_env("VLA_SOURCE_RECEIPT")).resolve(strict=True)
    receipt_sha256 = _required_env("VLA_SOURCE_RECEIPT_SHA256")
    expected_source_sha = _required_env("VLA_SOURCE_SHA")
    expected_runtime_id = _required_env("VLA_RUNTIME_ID")

    if not _SHA256_RE.fullmatch(receipt_sha256):
        raise ValueError("VLA_SOURCE_RECEIPT_SHA256 must be lowercase SHA-256")
    if _sha256_file(receipt_path) != receipt_sha256:
        raise ValueError("source release receipt hash mismatch")
    if not _GIT_SHA_RE.fullmatch(expected_source_sha):
        raise ValueError("VLA_SOURCE_SHA must be a lowercase 40-hex Git SHA")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise ValueError("unsupported source release receipt")
    if receipt.get("source_sha") != expected_source_sha:
        raise ValueError("source release SHA does not match VLA_SOURCE_SHA")
    source_parent_sha = str(receipt.get("source_parent_sha", ""))
    if not _GIT_SHA_RE.fullmatch(source_parent_sha):
        raise ValueError("source release parent SHA is invalid")
    if (
        not isinstance(receipt.get("source_repository"), str)
        or not receipt["source_repository"]
    ):
        raise ValueError("source release repository is missing")
    if Path(str(receipt.get("source_root"))).resolve(strict=True) != root:
        raise ValueError("source release root does not match VLA_SOURCE_ROOT")

    runtime = _mapping(receipt.get("runtime"), field="runtime")
    if runtime.get("runtime_id") != expected_runtime_id:
        raise ValueError("source release runtime identity mismatch")
    if not isinstance(runtime.get("edf_name"), str) or not runtime["edf_name"]:
        raise ValueError("source release runtime.edf_name is missing")
    for field in ("edf_sha256", "image_sha256"):
        if not _SHA256_RE.fullmatch(str(runtime.get(field, ""))):
            raise ValueError(f"source release runtime.{field} is invalid")
    image_bytes = runtime.get("image_bytes")
    if (
        not isinstance(image_bytes, int)
        or isinstance(image_bytes, bool)
        or image_bytes <= 0
    ):
        raise ValueError("source release runtime.image_bytes must be positive")
    edf_path = _runtime_file(runtime.get("edf_path"), field="edf_path")
    image_path = _runtime_file(runtime.get("image_path"), field="image_path")
    if _sha256_file(edf_path) != runtime["edf_sha256"]:
        raise ValueError("source release runtime EDF hash mismatch")
    try:
        edf_definition = tomllib.loads(edf_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError("source release runtime EDF is not valid TOML") from error
    _verify_edf_definition(
        edf_definition,
        root=root,
        receipt=receipt,
        runtime=runtime,
    )
    if image_path.stat().st_size != image_bytes:
        raise ValueError("source release runtime image size mismatch")
    if _sha256_file(image_path) != runtime["image_sha256"]:
        raise ValueError("source release runtime image hash mismatch")
    derived_runtime_id = f"edf:{runtime['edf_sha256']}:image:{runtime['image_sha256']}"
    if runtime["runtime_id"] != derived_runtime_id:
        raise ValueError(
            "source release runtime_id is not derived from EDF and image hashes"
        )

    submodules = _mapping(receipt.get("submodules"), field="submodules")
    if set(submodules) != _REQUIRED_SUBMODULES:
        raise ValueError(
            "source release submodule identities must be exactly "
            f"{sorted(_REQUIRED_SUBMODULES)}"
        )
    for path, sha in submodules.items():
        relative = Path(path) if isinstance(path, str) else None
        if (
            relative is None
            or not path
            or relative.is_absolute()
            or ".." in relative.parts
            or not _GIT_SHA_RE.fullmatch(str(sha))
        ):
            raise ValueError(
                f"invalid source release submodule entry: {path!r}={sha!r}"
            )

    _verify_git_checkout(
        root,
        source_sha=expected_source_sha,
        source_parent_sha=source_parent_sha,
        submodules=submodules,
    )

    files = _mapping(receipt.get("source_file_sha256"), field="source_file_sha256")
    if not _REQUIRED_FILES.issubset(files):
        missing = sorted(_REQUIRED_FILES.difference(files))
        raise ValueError(f"source release is missing required file hashes: {missing}")
    for relative_path, expected_sha256 in files.items():
        if not isinstance(relative_path, str) or not _SHA256_RE.fullmatch(
            str(expected_sha256)
        ):
            raise ValueError(f"invalid source release file entry: {relative_path!r}")
        path = _confined_file(root, relative_path)
        if _sha256_file(path) != expected_sha256:
            raise ValueError(f"source release file hash mismatch: {relative_path}")

    source_file_count = receipt.get("source_file_count")
    if (
        not isinstance(source_file_count, int)
        or isinstance(source_file_count, bool)
        or source_file_count <= 0
        or source_file_count != len(files)
    ):
        raise ValueError("source release source_file_count is invalid")
    actual_files = _source_inventory(root)
    if files != actual_files:
        missing = sorted(set(actual_files).difference(files))
        unexpected = sorted(set(files).difference(actual_files))
        changed = sorted(
            path
            for path in set(files).intersection(actual_files)
            if files[path] != actual_files[path]
        )
        raise ValueError(
            "source release file inventory is incomplete or stale: "
            f"missing={missing} unexpected={unexpected} changed={changed}"
        )

    return receipt
