"""Resolve Lightning Trainer hardware settings from the local environment."""

import os

import torch


def _slurm_int(name: str) -> int | None:
    """Parse a SLURM env var as int, or return None if unset."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def resolve_trainer_hardware() -> dict[str, object]:
    """Return accelerator, devices, strategy, and optional plugins for Trainer.

    On SLURM with ``--ntasks-per-node=1`` and multiple GPUs (``--gres=gpu:N``),
    Lightning's default SlurmEnvironment expects one task per GPU. Use
    ``LightningEnvironment`` so DDP spawns locally inside the single task.

    With ``--ntasks-per-node=N`` and one GPU per task, each rank uses
    ``devices=1`` and the default SlurmEnvironment.
    """
    if not torch.cuda.is_available():
        return {
            "accelerator": "cpu",
            "devices": "auto",
            "strategy": "auto",
            "plugins": None,
        }

    num_gpus = torch.cuda.device_count()
    slurm_ntasks_per_node = _slurm_int("SLURM_NTASKS_PER_NODE")
    in_slurm = os.environ.get("SLURM_JOB_ID") is not None

    if in_slurm and slurm_ntasks_per_node == 1 and num_gpus > 1:
        return {
            "accelerator": "gpu",
            "devices": num_gpus,
            "strategy": "ddp",
            "plugins": "lightning_environment",
        }

    if in_slurm and slurm_ntasks_per_node is not None and slurm_ntasks_per_node > 1:
        return {
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "ddp",
            "plugins": None,
        }

    return {
        "accelerator": "gpu",
        "devices": num_gpus,
        "strategy": "ddp" if num_gpus > 1 else "auto",
        "plugins": None,
    }
