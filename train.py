"""Hydra entrypoint for LLaDA masked-diffusion pretraining."""

import logging
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from models.factory import create_model_from_cfg
from training.callbacks import build_checkpoint_callbacks
from training.data import BabyLMTrain
from training.module import LLaDAPretrainModule
from training.run_dir import allocate_run_paths
from training.trainer_runtime import resolve_trainer_hardware
from utils import get_tokenizer

logger = logging.getLogger(__name__)


def _configured_run_name(cfg: DictConfig) -> str | None:
    """Return an explicit run.name override, or None to let W&B / timestamp invent one."""
    run_name = OmegaConf.select(cfg, "run.name", default=None)
    if run_name is None or str(run_name).strip() == "":
        return None
    return str(run_name).strip()


def _build_wandb_logger(cfg: DictConfig, *, name: str | None) -> WandbLogger:
    """Create a WandbLogger; name=None lets W&B invent curious-sunset-42 style ids."""
    entity = OmegaConf.select(cfg, "logging.entity", default=None)
    return WandbLogger(
        project=str(cfg.logging.project),
        entity=None if entity is None else str(entity),
        name=name,
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a randomly initialized ModernBERT diffusion LM on pretokenized BabyLM."""
    wandb_logger: WandbLogger | None = None
    explicit_name = _configured_run_name(cfg)
    if OmegaConf.select(cfg, "logging.enabled", default=False):
        # Init W&B before allocate_run_paths so a generated name can become the folder.
        # Rank > 0 gets DummyExperiment; never touch .experiment there (DDP re-exec).
        wandb_logger = _build_wandb_logger(cfg, name=explicit_name)
        if explicit_name is None and rank_zero_only.rank == 0:
            generated = wandb_logger.experiment.name
            if not generated:
                raise RuntimeError("W&B did not return a run name after init.")
            OmegaConf.set_struct(cfg, False)
            cfg.run.name = str(generated)

    allocate_run_paths(cfg)

    if wandb_logger is not None:
        # rank_zero_only inside WandbLogger; DummyExperiment.config is a nop function.
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    tokenizer = get_tokenizer(Path(cfg.paths.tokenizer))
    model = create_model_from_cfg(cfg, tokenizer)
    module = LLaDAPretrainModule(
        model,
        mask_token_id=int(tokenizer.mask_token_id),
        cfg=cfg,
    )
    datamodule = BabyLMTrain(cfg)

    hardware = resolve_trainer_hardware()
    plugins = (
        LightningEnvironment()
        if hardware.get("plugins") == "lightning_environment"
        else None
    )
    # EDLM freezes the backbone; DDP must allow unused params or it can stall.
    strategy: str | DDPStrategy = str(hardware["strategy"])
    if strategy == "ddp" and bool(
        OmegaConf.select(cfg, "train_energy", default=False)
    ):
        strategy = DDPStrategy(find_unused_parameters=True)
    logger.info(
        "Trainer hardware: accelerator=%s devices=%s strategy=%s plugins=%s",
        hardware["accelerator"],
        hardware["devices"],
        strategy,
        "LightningEnvironment" if plugins is not None else "default",
    )

    limit_train_batches = OmegaConf.select(cfg, "trainer.limit_train_batches", default=None)

    trainer = L.Trainer(
        accelerator=str(hardware["accelerator"]),
        devices=hardware["devices"],
        strategy=strategy,
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
