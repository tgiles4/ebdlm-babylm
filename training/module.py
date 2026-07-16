from pathlib import Path

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import get_scheduler

from models.ebdlm import LLaDAMDLM
from utils import get_tokenizer


class LLaDAPretrainModule(L.LightningModule):
    """
    Lightning wrapper for LLaDA masked discrete diffusion pre-training.
    """

    def __init__(
        self,
        model: LLaDAMDLM,
        mask_token_id: int,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.mask_token_id = mask_token_id
        self.cfg = cfg
        self.learning_rate = float(
            OmegaConf.select(cfg, "trainer.learning_rate", default=5.0e-5)
        )
        self.weight_decay = float(
            OmegaConf.select(cfg, "trainer.weight_decay", default=0.01)
        )
        self.lr_scheduler_type = str(
            OmegaConf.select(cfg, "trainer.lr_scheduler_type", default="cosine")
        )
        self.num_warmup_steps = int(
            OmegaConf.select(cfg, "trainer.num_warmup_steps", default=1000)
        )
        if not hasattr(model.config, "words_seen"):
            model.config.words_seen = 0

    def save_hf(self, hf_dir: Path) -> None:
        """Write model and tokenizer in HuggingFace save_pretrained layout."""
        hf_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(hf_dir)
        get_tokenizer(Path(self.cfg.paths.tokenizer)).save_pretrained(hf_dir)

    def _add_words_seen(self, word_count: torch.Tensor) -> int:
        """Increment model.config.words_seen by the global batch word count."""
        delta = word_count.sum().to(dtype=torch.long, device=self.device)
        if self.trainer.world_size > 1:
            torch.distributed.all_reduce(delta, op=torch.distributed.ReduceOp.SUM)
        self.model.config.words_seen = int(self.model.config.words_seen) + int(
            delta.item()
        )
        return int(self.model.config.words_seen)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Forward one batch, log scalars, and return the loss for backprop.

        Expects batch["input_ids"] and batch["word_count"] from the dataloader.
        """
        input_ids = batch["input_ids"]
        loss, mask_fraction = self.model.loss(input_ids)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "mask_fraction",
            mask_fraction,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        words_seen = self._add_words_seen(batch["word_count"])
        self.log(
            "words_seen",
            float(words_seen),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        return loss

    def configure_optimizers(self) -> dict[str, object]:
        """
        Return AdamW and a HuggingFace LR scheduler stepped every optimizer step.

        Only parameters with requires_grad=True are optimized (EDLM freezes the
        backbone and trains the energy head).
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters for AdamW")
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
