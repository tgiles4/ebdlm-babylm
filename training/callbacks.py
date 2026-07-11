"""Lightning callbacks for epoch checkpoints and HuggingFace export."""

import logging
from pathlib import Path
from typing import cast

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from transformers import PretrainedConfig

from training.generation import RemaskingStrategy, unconditional_generate
from training.module import LLaDAPretrainModule
from utils import BabyLMSize, get_tokenizer

log = logging.getLogger(__name__)


def words_seen_from_config(config: PretrainedConfig) -> int:
    """Read cumulative BabyLM words_seen stored on the model config."""
    return int(getattr(config, "words_seen", 0) or 0)


class HFSaveCallback(L.Callback):
    """Export HF checkpoints on epoch end, train end, and BabyLM word milestones.

    Epoch exports land in hf_export/epoch-{epoch}/; train end writes last/;
    BabyLM milestones write chck_{N}M/. Rank 0 only; barriers after milestone saves.
    """

    def __init__(self, cfg: DictConfig) -> None:
        # BabyLM 2026 strict README: 1M–10M @1M, 10M–100M @10M, 100M–1000M @100M.
        milestones_m: dict[BabyLMSize, tuple[int, ...]] = {
            BabyLMSize.STRICT_SMALL: (
                *range(1, 11),
                *range(20, 101, 10),
            ),
            BabyLMSize.STRICT: (
                *range(1, 11),
                *range(20, 101, 10),
                *range(200, 1001, 100),
            ),
        }
        self._milestones_m = list(milestones_m[BabyLMSize(cfg.dataset.size)])
        self._hf_dir = Path(cfg.paths.hf_export)
        self._prev_words_seen = 0

    def on_train_start(
        self, trainer: L.Trainer, pl_module: LLaDAPretrainModule
    ) -> None:
        """Seed crossing detection from words_seen already on the loaded model."""
        self._prev_words_seen = words_seen_from_config(pl_module.model.config)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: LLaDAPretrainModule,
        outputs: object,
        batch: dict[str, object],
        batch_idx: int,
    ) -> None:
        """Save chck_{N}M when model.config.words_seen newly crosses a threshold."""
        words_seen = words_seen_from_config(pl_module.model.config)

        for milestone_m in self._milestones_m:
            threshold = milestone_m * 1_000_000
            if self._prev_words_seen < threshold <= words_seen:
                if trainer.is_global_zero:
                    pl_module.save_hf(self._hf_dir / f"chck_{milestone_m}M")

                if trainer.world_size > 1:
                    trainer.strategy.barrier()

        self._prev_words_seen = words_seen

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: LLaDAPretrainModule
    ) -> None:
        """Mirror epoch-based ModelCheckpoint saves with an HF directory."""
        if trainer.is_global_zero:
            pl_module.save_hf(self._hf_dir / f"epoch-{trainer.current_epoch}")

        if trainer.world_size > 1:
            trainer.strategy.barrier()

    def on_train_end(self, trainer: L.Trainer, pl_module: LLaDAPretrainModule) -> None:
        """Write a last/ HF export when training finishes."""
        if not trainer.is_global_zero:
            return

        pl_module.save_hf(self._hf_dir / "last")


class SampleGenerationCallback(L.Callback):
    """Log unconditional diffusion samples to WandB at the end of selected epochs."""

    def __init__(self, cfg: DictConfig) -> None:
        self._enabled = bool(OmegaConf.select(cfg, "samples.enabled", default=False))
        self._every_n_epochs = int(
            OmegaConf.select(cfg, "samples.every_n_epochs", default=1)
        )
        self._batch_size = int(OmegaConf.select(cfg, "samples.batch_size", default=4))
        self._seq_len = int(
            OmegaConf.select(cfg, "samples.seq_len", default=cfg.context_length)
        )
        self._num_steps = int(OmegaConf.select(cfg, "samples.num_steps", default=64))
        self._remasking = cast(
            RemaskingStrategy,
            str(OmegaConf.select(cfg, "samples.remasking", default="low_confidence")),
        )
        self._num_log = int(OmegaConf.select(cfg, "samples.num_log", default=4))
        self._max_log_chars = int(
            OmegaConf.select(cfg, "samples.max_log_chars", default=2048)
        )
        self._tokenizer_path = Path(cfg.paths.tokenizer)

    @staticmethod
    def _truncate_for_log(text: str, max_chars: int) -> str:
        """Cap sample length for W&B tables; full-length 512-token rows blow up artifacts."""
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: LLaDAPretrainModule
    ) -> None:
        """Generate samples on rank 0 and log a W&B table; sync all ranks under DDP."""
        if not self._enabled:
            return

        if (trainer.current_epoch + 1) % self._every_n_epochs != 0:
            return

        if trainer.world_size > 1:
            trainer.strategy.barrier()

        if trainer.is_global_zero:
            wandb_logger = trainer.logger
            if wandb_logger is not None and hasattr(wandb_logger, "log_table"):
                tokenizer = get_tokenizer(self._tokenizer_path)
                was_training = pl_module.training
                pl_module.eval()
                try:
                    ids = unconditional_generate(
                        pl_module.model,
                        seq_len=self._seq_len,
                        mask_token_id=pl_module.mask_token_id,
                        num_steps=self._num_steps,
                        remasking=self._remasking,
                        batch_size=self._batch_size,
                    )
                    texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
                finally:
                    if was_training:
                        pl_module.train()

                rows = [
                    [self._truncate_for_log(text, self._max_log_chars)]
                    for text in texts[: self._num_log]
                ]
                try:
                    wandb_logger.log_table(
                        key=f"samples/epoch{trainer.current_epoch}",
                        columns=["Generated Samples"],
                        data=rows,
                        step=trainer.global_step,
                    )
                except Exception:
                    log.exception(
                        "Failed to log samples for epoch %s", trainer.current_epoch
                    )

        if trainer.world_size > 1:
            trainer.strategy.barrier()


def build_checkpoint_callbacks(cfg: DictConfig) -> list[L.Callback]:
    """Build epoch checkpointing, LR logging, and HF export callbacks.

    LearningRateMonitor is included only when logging.enabled is true (requires
    a Trainer logger). No monitor is set — there is no validation loop; BabyLM
    eval runs on exported HF checkpoints later.
    """
    checkpoint_dir = Path(cfg.paths.checkpoints)

    model_checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch-{epoch}",
        every_n_epochs=1,
        save_last=True,
        save_top_k=-1,
        auto_insert_metric_name=False,
    )
    callbacks: list[L.Callback] = [model_checkpoint]

    if OmegaConf.select(cfg, "logging.enabled", default=False):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    callbacks.append(HFSaveCallback(cfg))
    callbacks.append(SampleGenerationCallback(cfg))
    return callbacks
