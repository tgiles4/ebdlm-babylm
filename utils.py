from enum import StrEnum
from pathlib import Path

from huggingface_hub import snapshot_download


class BabyLMSize(StrEnum):
    """BabyLM 2026 track identifiers matching HuggingFace dataset repo names."""

    STRICT = "Strict"
    STRICT_SMALL = "Strict-Small"


def download_babylm_raw(path: Path, size: BabyLMSize) -> Path:
    """Download BabyLM training data if missing. Returns the raw data directory."""
    if not path.exists():
        snapshot_download(
            f"BabyLM-community/BabyLM-2026-{size.value}",
            repo_type="dataset",
            local_dir=path,
            local_dir_use_symlinks=False,
        )
    return path
