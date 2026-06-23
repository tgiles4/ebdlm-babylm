import re
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import hydra
from datasets import concatenate_datasets, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerFast

from tokenizer import train_tokenizer
from utils import BabyLMSize, download_babylm_raw

# Joins packed sub-lines in packed text files (one physical line per example).
FILE_PACK_JOIN = "\x1e"


def get_tokenizer(tokenizer_path: Path) -> PreTrainedTokenizerFast:
    """
    Wraps the tokenizer in a PreTrainedTokenizerFast object to access special tokens
    and tokenizer methods.
    """
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        mask_token="<mask>",
    )


def count_tokens(text: str, tokenizer: PreTrainedTokenizerFast) -> int:
    """Return token count including BOS/EOS."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def normalize_text_line(text: str) -> str:
    """Restore packed newlines from packed file storage."""
    return text.replace(FILE_PACK_JOIN, "\n")


def split_on_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part for part in parts if part]


def split_on_fallback(text: str, separators: list[str]) -> list[str]:
    """Split text on the first separator that yields multiple parts."""
    for separator in separators:
        if separator not in text:
            continue
        parts = text.split(separator)
        if len(parts) > 1:
            return [part + separator for part in parts[:-1]] + [parts[-1]]
    return [text]


def split_into_fitting_units(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    context_length: int,
) -> list[str]:
    """Recursively split overlong text until each piece fits within context_length."""
    text = text.strip()
    if not text:
        return []

    if count_tokens(text, tokenizer) <= context_length:
        return [text]

    sentences = split_on_sentences(text)
    if len(sentences) > 1:
        units: list[str] = []
        for sentence in sentences:
            units.extend(split_into_fitting_units(sentence, tokenizer, context_length))
        return units

    for separators in [["; "], [", "]]:
        parts = split_on_fallback(text, separators)
        if len(parts) > 1:
            units = []
            for part in parts:
                units.extend(split_into_fitting_units(part, tokenizer, context_length))
            return units

    token_count = count_tokens(text, tokenizer)
    preview = text[:120].encode("ascii", errors="backslashreplace").decode("ascii")
    raise ValueError(
        f"Cannot split text within {context_length} tokens ({token_count} tok): {preview!r}"
    )


def tokenize_batch(
    examples: dict[str, list[str]],
    tokenizer_path: Path,
    context_length: int,
) -> dict[str, list[list[int]]]:
    """Tokenize a batch of packed examples."""
    tokenizer = get_tokenizer(tokenizer_path)
    texts = [normalize_text_line(line.rstrip("\n")) for line in examples["text"]]
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding="max_length",
        max_length=context_length,
    )
    return {"input_ids": encoded["input_ids"]}


def prep_corpus(raw_dir: Path, prepped_dir: Path) -> None:
    """Copy raw corpora to prepped/, splitting simple_wiki at the first dialogue line."""
    prepped_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(raw_dir.glob("*.train.txt")):
        if path.name != "simple_wiki.train.txt":
            shutil.copy2(path, prepped_dir / path.name)
            continue

        with (
            path.open(encoding="utf-8") as src,
            (prepped_dir / "simple_wiki_articles.train.txt").open("w", encoding="utf-8") as articles,
            (prepped_dir / "simple_wiki_conversations.train.txt").open("w", encoding="utf-8") as conversations,
        ):
            out = articles
            for line in src:
                if out is articles and line.startswith("A:\t"):
                    out = conversations
                out.write(line)


def pack_file(
    input_path: Path,
    output_path: Path,
    tokenizer_path: Path,
    context_length: int,
) -> dict[str, int]:
    """Stream a corpus file; write pre-packed lines; return stats."""
    tokenizer = get_tokenizer(tokenizer_path)
    stats = {
        "source_lines": 0,
        "splits_count": 0,
        "output_lines": 0,
        "max_tokens": 0,
    }
    pending: deque[str] = deque()
    chunk_lines: list[str] = []

    def flush(outf) -> None:
        nonlocal chunk_lines
        if not chunk_lines:
            return

        packed = FILE_PACK_JOIN.join(chunk_lines)
        token_len = count_tokens("\n".join(chunk_lines), tokenizer)
        stats["max_tokens"] = max(stats["max_tokens"], token_len)
        stats["output_lines"] += 1
        outf.write(packed + "\n")
        chunk_lines = []

    with input_path.open(encoding="utf-8") as inf, output_path.open("w", encoding="utf-8") as outf:
        while True:
            if pending:
                line = pending.popleft()
            else:
                raw = inf.readline()
                if not raw:
                    break
                line = raw.rstrip("\n")
                if not line:
                    continue
                stats["source_lines"] += 1

            candidate = chunk_lines + [line]
            if count_tokens("\n".join(candidate), tokenizer) <= context_length:
                chunk_lines = candidate
                continue

            flush(outf)

            if count_tokens(line, tokenizer) <= context_length:
                chunk_lines = [line]
                continue

            stats["splits_count"] += 1
            sub_lines = split_into_fitting_units(line, tokenizer, context_length)
            pending.extendleft(reversed(sub_lines))

        flush(outf)

    return stats


def pack_corpus(
    dataset_dir: Path,
    output_dir: Path,
    tokenizer_path: Path,
    context_length: int,
    num_workers: int,
) -> None:
    """Pack all *.train.txt files to context_length; print per-file and total stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(dataset_dir.glob("*.train.txt"))

    totals = {
        "source_lines": 0,
        "splits_count": 0,
        "output_lines": 0,
        "max_tokens": 0,
    }

    if num_workers <= 1 or len(input_paths) <= 1:
        file_stats = [
            (input_path.name, pack_file(input_path, output_dir / input_path.name, tokenizer_path, context_length))
            for input_path in input_paths
        ]
    else:
        file_stats = []
        with ProcessPoolExecutor(max_workers=min(num_workers, len(input_paths))) as executor:
            future_to_name = {
                executor.submit(
                    pack_file,
                    input_path,
                    output_dir / input_path.name,
                    tokenizer_path,
                    context_length,
                ): input_path.name
                for input_path in input_paths
            }
            for future in as_completed(future_to_name):
                file_stats.append((future_to_name[future], future.result()))

    for filename, stats in sorted(file_stats):
        print(
            f"{filename}: "
            f"source={stats['source_lines']} "
            f"splits={stats['splits_count']} "
            f"output={stats['output_lines']} "
            f"max_tok={stats['max_tokens']}",
            flush=True,
        )
        for key in totals:
            if key == "max_tokens":
                totals[key] = max(totals[key], stats[key])
            else:
                totals[key] += stats[key]

    print(
        f"TOTAL: source={totals['source_lines']} "
        f"splits={totals['splits_count']} "
        f"output={totals['output_lines']} "
        f"max_tok={totals['max_tokens']}",
        flush=True,
    )


def pretokenize_data(
    context_length: int,
    dataset_dir: Path,
    save_dir: Path,
    num_workers: int,
    tokenizer_path: Path,
    batch_size: int,
) -> None:
    """Pretokenize packed text files to a HuggingFace dataset on disk."""
    train_files = sorted(dataset_dir.glob("*.txt"))
    datasets = [
        load_dataset("text", data_files=str(path), split="train") for path in train_files
    ]

    process_function = partial(
        tokenize_batch,
        tokenizer_path=tokenizer_path,
        context_length=context_length,
    )

    tokenized_datasets = [
        dataset.map(
            process_function,
            batched=True,
            batch_size=batch_size,
            num_proc=num_workers,
            remove_columns=["text"],
        )
        for dataset in datasets
    ]

    packed_dataset = concatenate_datasets(tokenized_datasets)
    save_dir.mkdir(parents=True, exist_ok=True)
    packed_dataset.save_to_disk(save_dir)


@hydra.main(version_base=None, config_path="conf", config_name="prepare_data")
def main(cfg: DictConfig) -> None:
    """Download raw data, ensure tokenizer, prep, pack, then pretokenize."""
    size = BabyLMSize(cfg.dataset.size)
    raw_dir = Path(cfg.paths.raw)
    prepped_dir = Path(cfg.paths.prepped)
    packed_dir = Path(cfg.paths.packed)
    pretokenized_dir = Path(cfg.paths.pretokenized)
    tokenizer_path = Path(cfg.paths.tokenizer)

    download_babylm_raw(raw_dir, size)

    if not tokenizer_path.is_file():
        train_tokenizer(
            data_path=raw_dir,
            output_file=tokenizer_path,
            size=size,
            vocab_size=cfg.tokenizer.vocab_size,
            min_frequency=cfg.tokenizer.min_frequency,
        )

    prep_corpus(raw_dir, prepped_dir)

    pack_corpus(
        dataset_dir=prepped_dir,
        output_dir=packed_dir,
        tokenizer_path=tokenizer_path,
        context_length=cfg.context_length,
        num_workers=cfg.num_workers,
    )

    pretokenize_data(
        context_length=cfg.context_length,
        dataset_dir=packed_dir,
        save_dir=pretokenized_dir,
        num_workers=cfg.num_workers,
        tokenizer_path=tokenizer_path,
        batch_size=cfg.batch_size,
    )


if __name__ == "__main__":
    main()
