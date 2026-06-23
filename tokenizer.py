from pathlib import Path

import hydra
from omegaconf import DictConfig
from tokenizers import Tokenizer, processors
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from utils import BabyLMSize, download_babylm_raw


def train_tokenizer(
    data_path: Path,
    output_file: Path,
    size: BabyLMSize,
    vocab_size: int,
    min_frequency: int,
    show_progress: bool = True,
) -> None:
    """
    Trains a BPE tokenizer for diffusion models.

    data_path: Directory for raw BabyLM train files.
    output_file: Path to save the trained tokenizer JSON.
    size: BabyLM track used for download and naming.
    vocab_size: The size of the vocabulary.
    min_frequency: The minimum frequency of a token to be included in the vocabulary.
    show_progress: Whether to show a progress bar during training.
    """
    raw_dir = download_babylm_raw(data_path, size)
    train_files = [str(p) for p in raw_dir.glob("*.train.txt")]

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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_file))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a BPE tokenizer from Hydra config."""
    train_tokenizer(
        data_path=Path(cfg.paths.raw),
        output_file=Path(cfg.paths.tokenizer),
        size=BabyLMSize(cfg.dataset.size),
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.tokenizer.min_frequency,
    )


if __name__ == "__main__":
    main()
