from pathlib import Path

import pytest

from scripts.filtered_sft_agent.eval_trained_policy_manifest import _checkpoint_id


def _checkpoint(tmp_path: Path, step: int = 4) -> Path:
    root = tmp_path / "checkpoints"
    artifact = root / str(step) / "default" / "state"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"checkpoint-state")
    return root


def test_checkpoint_id_hashes_restored_artifact_inventory(tmp_path):
    root = _checkpoint(tmp_path)
    first_id, first_hash = _checkpoint_id(root, 4, "source@abc/task0/random_val_full")
    assert first_id.endswith(f"#artifact_inventory_sha256={first_hash}")
    assert "#train_state_step=4#" in first_id

    (root / "4" / "default" / "state").write_bytes(b"changed-checkpoint-state")
    second_id, second_hash = _checkpoint_id(root, 4, "source@abc/task0/random_val_full")
    assert second_hash != first_hash
    assert second_id != first_id


def test_checkpoint_id_rejects_unusable_identity_and_symlinks(tmp_path):
    root = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="portable identity"):
        _checkpoint_id(root, 4, "bad#identity")

    symlink = root / "4" / "linked"
    symlink.symlink_to(root / "4" / "default" / "state")
    with pytest.raises(ValueError, match="forbids symlinks"):
        _checkpoint_id(root, 4, "source@abc/task0/random_val_full")


def test_checkpoint_id_rejects_symlinked_step_directory(tmp_path):
    root = _checkpoint(tmp_path, step=3)
    (root / "4").symlink_to(root / "3", target_is_directory=True)
    with pytest.raises(ValueError, match="step directory must not be a symlink"):
        _checkpoint_id(root, 4, "source@abc/task0/random_val_full")
