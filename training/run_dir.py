"""Assign a unique on-disk run directory for checkpoints and HF exports."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _git_sha_short(root: Path) -> str | None:
    """Return a short git SHA when training inside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        sha = result.stdout.strip()
        return sha or None
    except (OSError, subprocess.SubprocessError):
        return None


def _default_run_name(root: Path, dataset_slug: str) -> str:
    """Build a filesystem-safe run folder name when W&B is not inventing one."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    parts = [stamp, dataset_slug]
    sha = _git_sha_short(root)
    if sha is not None:
        parts.append(sha)
    return "_".join(parts)


def allocate_run_paths(cfg: DictConfig) -> Path:
    """Pick a run folder, wire checkpoint paths into cfg, and save the resolved config.

    Prefer ``cfg.run.name`` when set (including a W&B-generated name from
    ``train.py``). Otherwise fall back to timestamp + dataset slug + git SHA.

    Layout:
        runs/{run_name}/checkpoints/   — Lightning .ckpt files
        runs/{run_name}/hf/            — HuggingFace exports
        runs/{run_name}/config.yaml    — frozen Hydra config for this job
    """
    root = Path(cfg.root)
    dataset_slug = str(cfg.dataset.slug)
    run_name = OmegaConf.select(cfg, "run.name", default=None)
    if run_name is None or str(run_name).strip() == "":
        run_name = _default_run_name(root, dataset_slug)
    else:
        run_name = str(run_name).strip()

    run_dir = root / "runs" / run_name
    checkpoints = run_dir / "checkpoints"
    hf_export = run_dir / "hf"
    checkpoints.mkdir(parents=True, exist_ok=True)
    hf_export.mkdir(parents=True, exist_ok=True)

    OmegaConf.set_struct(cfg, False)
    cfg.run.name = run_name
    cfg.run.dir = str(run_dir)
    cfg.paths.checkpoints = str(checkpoints)
    cfg.paths.hf_export = str(hf_export)

    config_path = run_dir / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    log.info("Run directory: %s", run_dir)
    return run_dir
