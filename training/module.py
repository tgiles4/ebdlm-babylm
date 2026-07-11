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
        self, input_ids: torch.Tensor, *, mask_eps: float = 1e-3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one forward-noising step and compute the LLaDA pretraining loss.

        Follows the LLaDA pretraining procedure: sample a masking rate t uniformly
        per sequence, set p_mask = (1 - epsilon) * t + epsilon, mask tokens
        independently with probability p_mask, predict clean tokens at masked
        positions, and weight cross-entropy by 1 / p_mask. The loss is summed
        over masked positions and normalized by batch size times sequence length.

        Args:
            input_ids: Clean token ids x_0, shape [B, L].
            mask_eps: Minimum per-token mask probability in the LLaDA forward process.

        Returns:
            Scalar mean loss and the batch mean mask fraction (fraction of
            positions replaced by the mask token), useful as a training metric.
        """
        if torch.rand(1, device=input_ids.device) < 0.01:
            random_length = int(
                torch.randint(
                    1, input_ids.shape[1] + 1, (1,), device=input_ids.device
                ).item()
            )
            input_ids = input_ids[:, :random_length]

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

        t = torch.rand(batch_size, device=device)
        p_mask = (mask_eps + (1.0 - mask_eps) * t).unsqueeze(1).expand(batch_size, seq_len)

        mask = torch.rand(batch_size, seq_len, device=device) < p_mask
        masked_ids = input_ids.masked_fill(mask, self.mask_token_id)
        logits = self.model(
            input_ids=masked_ids, attention_mask=attention_mask
        ).logits

        token_loss = F.cross_entropy(
            logits[mask], input_ids[mask], reduction="none"
        ) / p_mask[mask]
        loss = token_loss.sum() / (batch_size * seq_len)
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
