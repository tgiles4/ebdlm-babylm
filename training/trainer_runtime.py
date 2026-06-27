"""Resolve Lightning Trainer hardware settings from the local environment."""

import torch


def resolve_trainer_hardware() -> dict[str, object]:
    """Return accelerator, devices, and strategy from visible CUDA devices.

    Uses DDP when multiple GPUs are available; single-GPU and CPU use strategy
    ``auto``.
    """
    if not torch.cuda.is_available():
        return {"accelerator": "cpu", "devices": "auto", "strategy": "auto"}

    num_gpus = torch.cuda.device_count()
    return {
        "accelerator": "gpu",
        "devices": num_gpus,
        "strategy": "ddp" if num_gpus > 1 else "auto",
    }
