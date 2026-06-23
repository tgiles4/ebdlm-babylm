"""Build randomly initialized ModernBERT masked-LM models from Hydra config."""

from __future__ import annotations

import logging

from omegaconf import DictConfig
from transformers import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger(__name__)


def modernbert_config_from_cfg(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> ModernBertConfig:
    """Map shared Hydra fields + model profile + live tokenizer IDs to config."""
    return ModernBertConfig(
        vocab_size=int(cfg.vocab_size),
        max_position_embeddings=int(cfg.context_length),
        num_hidden_layers=int(cfg.model.num_hidden_layers),
        hidden_size=int(cfg.model.hidden_size),
        num_attention_heads=int(cfg.model.num_attention_heads),
        intermediate_size=int(cfg.model.intermediate_size),
        pad_token_id=int(tokenizer.pad_token_id),
        bos_token_id=int(tokenizer.bos_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        mask_token_id=int(tokenizer.mask_token_id),
    )


def _attn_implementation() -> str:
    """Use Flash Attention 2 when installed; otherwise SDPA."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


def create_model(config: ModernBertConfig) -> ModernBertForMaskedLM:
    """Randomly initialize ``ModernBertForMaskedLM`` from config."""
    attn = _attn_implementation()
    logger.info("Using attention implementation: %s", attn)
    model = ModernBertForMaskedLM(config, attn_implementation=attn)
    param_count = model.num_parameters()
    logger.info(
        "Initialized ModernBertForMaskedLM with %s parameters", f"{param_count:,}"
    )
    return model


def create_model_from_cfg(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> ModernBertForMaskedLM:
    """Build config from Hydra + tokenizer, then randomly initialize the model."""
    config = modernbert_config_from_cfg(cfg, tokenizer)
    return create_model(config)
