"""Hydra entrypoint for LLaDA masked-diffusion pretraining."""

import logging
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from omegaconf import DictConfig, OmegaConf

from models.factory import create_model_from_cfg
from training.callbacks import build_checkpoint_callbacks
from training.data import BabyLMTrain
from training.module import LLaDAPretrainModule
from training.trainer_runtime import resolve_trainer_hardware
from utils import get_tokenizer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a randomly initialized ModernBERT diffusion LM on pretokenized BabyLM."""
    tokenizer = get_tokenizer(Path(cfg.paths.tokenizer))
    model = create_model_from_cfg(cfg, tokenizer)
    module = LLaDAPretrainModule(
        model,
        mask_token_id=int(tokenizer.mask_token_id),
        cfg=cfg,
    )
    datamodule = BabyLMTrain(cfg)

    wandb_logger: WandbLogger | None = None
    if OmegaConf.select(cfg, "logging.enabled", default=False):
        wandb_logger = WandbLogger(
            project=str(cfg.logging.project),
            entity=cfg.logging.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    hardware = resolve_trainer_hardware()
    plugins = (
        LightningEnvironment()
        if hardware.get("plugins") == "lightning_environment"
        else None
    )
    logger.info(
        "Trainer hardware: accelerator=%s devices=%s strategy=%s plugins=%s",
        hardware["accelerator"],
        hardware["devices"],
        hardware["strategy"],
        "LightningEnvironment" if plugins is not None else "default",
    )

    limit_train_batches = OmegaConf.select(cfg, "trainer.limit_train_batches", default=None)

    trainer = L.Trainer(
        accelerator=str(hardware["accelerator"]),
        devices=hardware["devices"],
        strategy=str(hardware["strategy"]),
        plugins=plugins,
        max_epochs=int(cfg.trainer.max_epochs),
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        precision=str(cfg.trainer.precision),
        logger=wandb_logger,
        callbacks=build_checkpoint_callbacks(cfg),
        log_every_n_steps=int(cfg.logging.log_every_n_steps),
        enable_checkpointing=True,
        limit_val_batches=0,
        limit_train_batches=int(limit_train_batches)
        if limit_train_batches is not None
        else None,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
