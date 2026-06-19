import argparse
from pathlib import Path

from tokenizers import Tokenizer, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from huggingface_hub import snapshot_download


def train_tokenizer(
    data_dir: Path,
    output_path: Path,
    size: str,
    vocab_size: int = 16384,
    min_frequency: int = 2,
    show_progress: bool = True,
) -> None:
    """
    Trains a BPE tokenizer for diffusion models.

    data_dir: Path to the directory containing the text files to train the tokenizer on.
    output_path: Path to the directory to save the trained tokenizer.
    vocab_size: The size of the vocabulary.
    min_frequency: The minimum frequency of a token to be included in the vocabulary.
    show_progress: Whether to show a progress bar during training.
    """
    root = data_dir / size.lower()

    if not root.exists():
        snapshot_download(
            f"BabyLM-community/BabyLM-2026-{size}",
            repo_type="dataset",
            local_dir=data_dir / size.lower(),
            local_dir_use_symlinks=False,
        )


    train_files = [str(p) for p in root.glob("*.train.txt")]

    special_tokens = [
        "<pad>",
        "<unk>",
        "<mask>",
        "<bos>",
        "<eos>",
    ]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=show_progress,
        initial_alphabet=ByteLevel.alphabet(),
        min_frequency=min_frequency,
    )

    tokenizer.train(train_files, trainer)

    tokenizer.post_processor = processors.BertProcessing(
        ("<eos>", tokenizer.token_to_id("<eos>")),
        ("<bos>", tokenizer.token_to_id("<bos>")),
    )
    
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path / f"tokenizer_{size.lower()}.json"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--size", type=str, choices=["Strict-Small", "Strict"], default="Strict")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--min_frequency", type=int, default=2)

    args = parser.parse_args()

    train_tokenizer(
        data_dir=args.data_dir,
        output_path=args.output_path,
        size=args.size,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
