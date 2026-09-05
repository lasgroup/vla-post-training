import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.filtered_sft_agent import source_release


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _nested_repo(root: Path, file_name: str) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / file_name).write_text(f"{file_name}\n", encoding="utf-8")
    _git(root, "add", file_name)
    _git(root, "commit", "-q", "-m", f"Add {file_name}")
    return _git(root, "rev-parse", "HEAD")


def _rewrite_receipt(receipt_path: Path, receipt: dict, monkeypatch) -> None:
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("VLA_SOURCE_RECEIPT_SHA256", _sha256(receipt_path))
    monkeypatch.setenv("VLA_RUNTIME_ID", receipt["runtime"]["runtime_id"])


def _release(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    openpi_repo = tmp_path / "openpi-repo"
    molmospaces_repo = tmp_path / "molmospaces-repo"
    openpi_sha = _nested_repo(openpi_repo, "openpi.txt")
    molmospaces_sha = _nested_repo(molmospaces_repo, "molmospaces.txt")

    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "base.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", "base.txt")
    _git(root, "commit", "-q", "-m", "Base")
    parent_sha = _git(root, "rev-parse", "HEAD")

    for relative_path in sorted(source_release._REQUIRED_FILES):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"runtime file: {relative_path}\n", encoding="utf-8")
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(openpi_repo),
        "openpi",
    )
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(molmospaces_repo),
        "molmospaces",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "Add runtime release")
    source_sha = _git(root, "rev-parse", "HEAD")
    _git(root, "remote", "add", "origin", str(root))
    _git(root, "checkout", "-q", "--detach", source_sha)
    _git(root / "openpi", "checkout", "-q", "--detach", openpi_sha)
    _git(root / "molmospaces", "checkout", "-q", "--detach", molmospaces_sha)

    files = source_release._source_inventory(root)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    edf_path = runtime_root / "runtime.edf.toml"
    image_path = runtime_root / "test.sqsh"
    image_path.write_bytes(b"test-runtime-image")
    edf_path.write_text(
        "".join(
            [
                f'image = "{image_path}"\n\n',
                "mounts = [\n",
                f'  "{root.resolve()}:/workspace:ro",\n',
                '  "/capstor",\n',
                '  "/iopsstor"\n',
                "]\n\n",
                'workdir = "/workspace"\n\n',
                "[env]\n",
                'PATH = "/.venv/bin:${PATH}"\n',
                'PYTHONPATH = "/workspace:/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src"\n',
                'LD_LIBRARY_PATH = "/usr/lib64:${LD_LIBRARY_PATH:-}"\n',
                'MUJOCO_GL = "egl"\n',
                'PYOPENGL_PLATFORM = "egl"\n',
                'EGL_PLATFORM = "surfaceless"\n',
            ]
        ),
        encoding="utf-8",
    )
    edf_sha256 = _sha256(edf_path)
    image_sha256 = _sha256(image_path)
    receipt = {
        "schema_version": 1,
        "source_repository": str(root),
        "source_sha": source_sha,
        "source_parent_sha": parent_sha,
        "source_root": str(root.resolve()),
        "source_host_path": str(root.resolve()),
        "runtime": {
            "runtime_id": f"edf:{edf_sha256}:image:{image_sha256}",
            "edf_name": "test-edf",
            "edf_path": str(edf_path),
            "edf_sha256": edf_sha256,
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "image_bytes": image_path.stat().st_size,
        },
        "submodules": {
            "molmospaces": molmospaces_sha,
            "openpi": openpi_sha,
        },
        "source_file_count": len(files),
        "source_file_sha256": files,
    }
    receipt_path = tmp_path / "SOURCE_RELEASE.json"
    _rewrite_receipt(receipt_path, receipt, monkeypatch)
    monkeypatch.setenv("VLA_SOURCE_ROOT", str(root))
    monkeypatch.setenv("VLA_SOURCE_RECEIPT", str(receipt_path))
    monkeypatch.setenv("VLA_SOURCE_SHA", receipt["source_sha"])
    return root, receipt_path, receipt


def test_source_release_accepts_hash_bound_runtime(tmp_path, monkeypatch):
    _, _, receipt = _release(tmp_path, monkeypatch)
    assert source_release.verify_source_release() == receipt


def test_source_release_rejects_forged_repository_identity(tmp_path, monkeypatch):
    _, receipt_path, receipt = _release(tmp_path, monkeypatch)
    receipt["source_repository"] = "https://forged.invalid/not-the-checkout.git"
    _rewrite_receipt(receipt_path, receipt, monkeypatch)
    with pytest.raises(ValueError, match="repository.*configured origin"):
        source_release.verify_source_release()


def test_source_release_rejects_tampered_runtime_file(tmp_path, monkeypatch):
    root, _, _ = _release(tmp_path, monkeypatch)
    target = root / min(source_release._REQUIRED_FILES)
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty|file hash mismatch"):
        source_release.verify_source_release()


def test_source_release_rejects_receipt_hash_mismatch(tmp_path, monkeypatch):
    _release(tmp_path, monkeypatch)
    monkeypatch.setenv("VLA_SOURCE_RECEIPT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        source_release.verify_source_release()


def test_source_release_rejects_escaping_file_path(tmp_path, monkeypatch):
    _, receipt_path, receipt = _release(tmp_path, monkeypatch)
    outside = tmp_path / "outside.py"
    outside.write_text("outside\n", encoding="utf-8")
    receipt["source_file_sha256"]["../outside.py"] = _sha256(outside)
    receipt["source_file_count"] += 1
    _rewrite_receipt(receipt_path, receipt, monkeypatch)
    with pytest.raises(ValueError, match="invalid source-release path"):
        source_release.verify_source_release()


def test_source_release_rejects_unlisted_runtime_file(tmp_path, monkeypatch):
    root, _, _ = _release(tmp_path, monkeypatch)
    extra = root / "src" / "runtime_override.py"
    extra.write_text("override = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty|inventory is incomplete or stale"):
        source_release.verify_source_release()


def test_source_inventory_rejects_directory_symlink(tmp_path, monkeypatch):
    root, _, _ = _release(tmp_path, monkeypatch)
    os.symlink(tmp_path / "runtime", root / "uninventoried_runtime")
    with pytest.raises(ValueError, match="directory symlink"):
        source_release._source_inventory(root)


def test_source_release_rejects_incomplete_submodule_inventory(tmp_path, monkeypatch):
    _, receipt_path, receipt = _release(tmp_path, monkeypatch)
    del receipt["submodules"]["molmospaces"]
    _rewrite_receipt(receipt_path, receipt, monkeypatch)
    with pytest.raises(ValueError, match="submodule identities must be exactly"):
        source_release.verify_source_release()


def test_source_release_rejects_forged_submodule_identities(tmp_path, monkeypatch):
    _, receipt_path, receipt = _release(tmp_path, monkeypatch)
    receipt["submodules"] = {
        "molmospaces": "0" * 40,
        "openpi": "f" * 40,
    }
    _rewrite_receipt(receipt_path, receipt, monkeypatch)
    with pytest.raises(ValueError, match="gitlinks do not match"):
        source_release.verify_source_release()


def test_source_release_rejects_tampered_nested_submodule(tmp_path, monkeypatch):
    root, _, _ = _release(tmp_path, monkeypatch)
    (root / "openpi" / "openpi.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        source_release.verify_source_release()


def test_source_release_rejects_attached_nested_submodule(tmp_path, monkeypatch):
    root, _, receipt = _release(tmp_path, monkeypatch)
    _git(
        root / "openpi",
        "checkout",
        "-q",
        "-B",
        "attached-test-branch",
        receipt["submodules"]["openpi"],
    )
    with pytest.raises(ValueError, match="submodule openpi must be detached"):
        source_release.verify_source_release()


def test_source_release_rejects_tampered_runtime_image(tmp_path, monkeypatch):
    _, _, receipt = _release(tmp_path, monkeypatch)
    Path(receipt["runtime"]["image_path"]).write_bytes(b"tampered-image")
    with pytest.raises(ValueError, match="image size mismatch|image hash mismatch"):
        source_release.verify_source_release()


def test_source_release_rejects_quoted_credential_bearing_edf(tmp_path, monkeypatch):
    _, receipt_path, receipt = _release(tmp_path, monkeypatch)
    edf_path = Path(receipt["runtime"]["edf_path"])
    edf_path.write_text('[env]\n"HF_TOKEN" = "credential-canary"\n', encoding="utf-8")
    edf_sha256 = _sha256(edf_path)
    receipt["runtime"]["edf_sha256"] = edf_sha256
    receipt["runtime"]["runtime_id"] = (
        f"edf:{edf_sha256}:image:{receipt['runtime']['image_sha256']}"
    )
    _rewrite_receipt(receipt_path, receipt, monkeypatch)
    with pytest.raises(ValueError, match="credential-bearing key"):
        source_release.verify_source_release()


@pytest.mark.parametrize("credential_key", ["ACCESS_KEY", "PRIVATE_KEY", "CREDENTIAL"])
def test_source_release_rejects_any_non_allowlisted_edf_environment_key(
    tmp_path, monkeypatch, credential_key
):
    _, receipt_path, receipt = _release(tmp_path, monkeypatch)
    edf_path = Path(receipt["runtime"]["edf_path"])
    with edf_path.open("a", encoding="utf-8") as handle:
        handle.write(f'{credential_key} = "credential-canary"\n')
    edf_sha256 = _sha256(edf_path)
    receipt["runtime"]["edf_sha256"] = edf_sha256
    receipt["runtime"]["runtime_id"] = (
        f"edf:{edf_sha256}:image:{receipt['runtime']['image_sha256']}"
    )
    _rewrite_receipt(receipt_path, receipt, monkeypatch)
    with pytest.raises(ValueError, match="credential-bearing key|exact allowlist"):
        source_release.verify_source_release()
