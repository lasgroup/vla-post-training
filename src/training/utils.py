import dataclasses
import etils.epath as epath
import logging
import numpy as np
from pathlib import Path
import subprocess
import wandb


from openpi.training import config as _config


def _run_git_command(args: list[str], repo_root: Path) -> str:
    result = subprocess.run(
        args,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _resolve_git_repo_root() -> Path | None:
    repo_hint = Path(__file__).resolve().parents[2]
    try:
        repo_root = _run_git_command(
            ["git", "rev-parse", "--show-toplevel"], repo_hint
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return Path(repo_root)


def save_git_snapshot(checkpoint_dir: epath.Path) -> None:
    repo_root = _resolve_git_repo_root()
    if repo_root is None:
        logging.warning("Skipping git snapshot: unable to resolve repository root.")
        return

    try:
        commit_hash = _run_git_command(["git", "rev-parse", "HEAD"], repo_root).strip()
        diff_patch = _run_git_command(["git", "diff", "--binary", "HEAD"], repo_root)
        untracked_files = _run_git_command(
            ["git", "ls-files", "--others", "--exclude-standard"], repo_root
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logging.warning("Skipping git snapshot: %s", exc)
        return

    snapshot_lines = [
        f"repository_root: {repo_root}",
        f"commit_hash: {commit_hash}",
        "",
        "diff_patch_vs_HEAD:",
        diff_patch.rstrip(),
    ]
    if untracked_files:
        snapshot_lines.extend(
            [
                "",
                "untracked_files:",
                untracked_files,
            ]
        )

    snapshot_text = "\n".join(snapshot_lines).rstrip() + "\n"
    (checkpoint_dir / "git_state.txt").write_text(snapshot_text)


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    save_git_snapshot(ckpt_dir)

    if not enabled:
        wandb.init(mode="disabled")
        return

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def log_images(batch):
    images_to_log = [
        wandb.Image(
            np.concatenate(
                [(np.array(img[i]) + 1.0) * 127.5 for img in batch[0].images.values()],
                axis=1,
            )
        )
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)
