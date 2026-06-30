"""Pretokenize BabyLM corpora by document (title-boundary) slices."""
import html
import re
from functools import partial
from itertools import chain
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import DictConfig

from tokenizer import train_tokenizer
from utils import BabyLMSize, count_whitespace_words, download_babylm_raw, get_tokenizer

# Compiled once at import; module-level so datasets.map(num_proc=...) workers can pickle preprocess_line.
TITLE_LINE = re.compile(r"^= = = .+ = = =\s*$")


def preprocess_line(example: dict[str, str]) -> dict[str, str | bool]:
    """Normalize a raw line; mark title headers (document boundaries, not training text)."""
    # Collapse whitespace
    text = re.sub(r"\s+", " ", example["text"]).strip()
    # Unescape HTML entities
    text = html.unescape(text)
    return {"text": text, "is_title": bool(TITLE_LINE.match(text))}


def group_lines_into_documents(lines: Dataset) -> Dataset:
    """Join lines per doc_id via Polars group_by (one text blob per document)."""
    frame = lines.to_polars()
    if frame.is_empty():
        return Dataset.from_dict({"text": []})

    documents = (
        frame.group_by("doc_id", maintain_order=True)
        .agg(pl.col("text").str.join("\n"))
        .select("text")
    )
    return Dataset.from_polars(documents)


def tokenize_documents_batch(
    examples: dict[str, list[str]],
    tokenizer_path: Path,
    context_length: int,
) -> dict[str, list[list[int]] | list[int]]:
    """
    Tokenize each document with truncate+overflow; split word_count across sequences.

    Maintains word count across sequences for BabyLM logging.
    """
    tokenizer = get_tokenizer(tokenizer_path)

    def tokenize_document(text: str) -> tuple[list[list[int]], list[int]]:
        total_words = count_whitespace_words(text)
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            padding="max_length",
            stride=0,
        )
        num_sequences = len(encoded["input_ids"])
        num_words = total_words // num_sequences
        counts = [num_words] * num_sequences
        if (remainder := total_words % num_sequences) > 0:
            counts[-1] += remainder
        return list(encoded["input_ids"]), counts

    tokenized = [tokenize_document(text) for text in examples["text"]]
    
    # Flatten the list of tokenized documents into a single list of input_ids and word_counts
    input_ids, word_counts = map(list, zip(*chain.from_iterable(zip(*doc) for doc in tokenized)))
    return {"input_ids": input_ids, "word_count": word_counts}


def pretokenize_lines(
    lines: Dataset,
    tokenizer_path: Path,
    context_length: int,
    num_workers: int,
    batch_size: int,
) -> Dataset:
    """Tag lines, group by title boundary, tokenize each document."""
    lines = lines.map(preprocess_line, num_proc=num_workers)
    lines = lines.add_column("doc_id", np.cumsum(lines["is_title"], dtype=np.int64).tolist())
    lines = lines.filter(lambda row: not row["is_title"] and row["text"])

    documents = group_lines_into_documents(lines)

    tokenize_fn = partial(
        tokenize_documents_batch,
        tokenizer_path=tokenizer_path,
        context_length=context_length,
    )
    return documents.map(
        tokenize_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_workers,
        remove_columns=documents.column_names,
    )


def load_corpora(raw_dir: Path) -> list[Dataset]:
    """Load raw BabyLM files; split simple_wiki into articles and conversations."""

    def corpus_slices(path: Path) -> list[Dataset]:
        lines = load_dataset("text", data_files=str(path), split="train")
        if path.name == "simple_wiki.train.txt":
            # Find the first dialogue line and split into articles and conversations
            index = next(
                (i for i, text in enumerate(lines["text"]) if text.startswith("A:\t")),
                len(lines),
            )
            return [
                lines.select(range(index)),
                lines.select(range(index, len(lines))),
            ]
        return [lines]

    return list(chain.from_iterable(
        corpus_slices(path) for path in sorted(raw_dir.glob("*.train.txt"))
    ))


def pretokenize_raw_corpora(
    raw_dir: Path,
    save_dir: Path,
    tokenizer_path: Path,
    context_length: int,
    num_workers: int,
    batch_size: int,
) -> Dataset:
    """Pretokenize all raw corpora and save a single dataset to disk."""
    tokenized_parts = [
        pretokenize_lines(lines, tokenizer_path, context_length, num_workers, batch_size)
        for lines in load_corpora(raw_dir)
    ]

    tokenized = concatenate_datasets(tokenized_parts)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(save_dir)
    return tokenized


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Download raw data, ensure tokenizer, then pretokenize by document."""
    size = BabyLMSize(cfg.dataset.size)
    raw_dir = Path(cfg.paths.raw)
    pretokenized_dir = Path(cfg.paths.pretokenized)
    tokenizer_path = Path(cfg.paths.tokenizer)

    download_babylm_raw(raw_dir, size)
    if not tokenizer_path.is_file():
        train_tokenizer(
            data_path=raw_dir,
            output_file=tokenizer_path,
            size=size,
            vocab_size=cfg.vocab_size,
            min_frequency=cfg.tokenizer.min_frequency,
        )

    pretokenize_raw_corpora(
        raw_dir=raw_dir,
        save_dir=pretokenized_dir,
        tokenizer_path=tokenizer_path,
        context_length=cfg.context_length,
        num_workers=cfg.pretokenize.num_workers,
        batch_size=cfg.pretokenize.batch_size,
    )


if __name__ == "__main__":
    main()
