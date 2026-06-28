"""Lightning data module for pretokenized BabyLM training data."""

from pathlib import Path

import lightning as L
from datasets import load_from_disk
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


class BabyLMTrain(L.LightningDataModule):
    """Load pretokenized training data from disk."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pretokenized_path = Path(cfg.paths.pretokenized)
        self.batch_size = int(
            OmegaConf.select(cfg, "trainer.per_device_batch_size", default=16)
        )
        self.num_workers = int(
            OmegaConf.select(cfg, "trainer.dataloader_num_workers", default=4)
        )
        self._train_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Load the pretokenized dataset from cfg.paths.pretokenized."""
        dataset = load_from_disk(str(self.pretokenized_path))
        self._train_dataset = dataset.with_format(
            type="torch",
            columns=["input_ids", "word_count"],
        )

    def train_dataloader(self) -> DataLoader:
        """Return a shuffled training dataloader."""
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader().")

        loader_kwargs: dict[str, object] = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        return DataLoader(self._train_dataset, **loader_kwargs)
