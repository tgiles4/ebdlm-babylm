from enum import StrEnum
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast


class BabyLMSize(StrEnum):
    """BabyLM 2026 track identifiers matching HuggingFace dataset repo names."""

    STRICT = "Strict"
    STRICT_SMALL = "Strict-Small"


def count_whitespace_words(text: str) -> int:
    """Return BabyLM whitespace-separated word count (repeated exposures count)."""
    text = text.strip()
    if not text:
        return 0
    return len(text.split())


def get_tokenizer(tokenizer_path: Path) -> PreTrainedTokenizerFast:
    """Load the trained BPE tokenizer with diffusion special tokens."""
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        mask_token="<mask>",
    )


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
