from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import ModernBertForMaskedLM, get_scheduler

from utils import get_tokenizer


class LLaDAPretrainModule(L.LightningModule):
    """
    Lightning wrapper for LLaDA masked discrete diffusion pre-training.
    """
    def __init__(
        self,
        model: ModernBertForMaskedLM,
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
        self.random_length_fraction = float(
            OmegaConf.select(cfg, "trainer.random_length_fraction", default=0.01)
        )

    def _maybe_random_length_crop(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Truncate to a uniform random prefix length on ~1% of steps (LLaDA pretrain)."""
        if self.random_length_fraction <= 0:
            return input_ids
        if torch.rand(1, device=input_ids.device) >= self.random_length_fraction:
            return input_ids
        seq_len = input_ids.shape[1]
        random_length = int(
            torch.randint(1, seq_len + 1, (1,), device=input_ids.device).item()
        )
        return input_ids[:, :random_length]

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

    def _diffusion_loss(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one forward-noising step and compute the LLaDA pre-training loss.

        1. Sample t ~ U(0, 1] per sequence (broadcast across length) and
           clamp below 1e-5 so the 1/t divisor stays finite.
        2. Draw an independent Bernoulli mask per token with probability t.
        3. Build x_t by replacing masked positions with mask_token_id.
        4. Forward through the bidirectional encoder; logits at position i
           predict the token x_0^i
        5. Cross entropy zeroed on unmasked positions

        Args:
            input_ids: Clean token ids x_0, shape [B, L].

        Returns:
            Scalar mean loss and the batch mean mask fraction (fraction of
            positions replaced by M), useful as a training health metric.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        t = (
            torch.rand(batch_size, 1, device=device)
            .expand(batch_size, seq_len)
            .clamp_min(1e-5)
        )

        mask = torch.bernoulli(t).bool()
        masked_ids = input_ids.masked_fill(mask, self.mask_token_id)
        labels = input_ids.masked_fill(~mask, -100)
        logits = self.model(
            input_ids=masked_ids, attention_mask=attention_mask
        ).logits

        num_classes = logits.shape[-1]
        per_token_loss = F.cross_entropy(
            logits.reshape(batch_size * seq_len, num_classes),
            labels.flatten(),
            reduction="none",
        )

        loss = per_token_loss.reshape(batch_size, seq_len) / t
        loss = loss.mean()
        mask_fraction = mask.float().mean()
        return loss, mask_fraction

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Forward one batch, log scalars, and return the loss for backprop.

        Expects batch["input_ids"] and batch["word_count"] from the dataloader.
        """
        input_ids = batch["input_ids"]
        input_ids = self._maybe_random_length_crop(input_ids)
        loss, mask_fraction = self._diffusion_loss(input_ids)
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
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
