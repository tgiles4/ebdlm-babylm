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
    )


def create_model(config: ModernBertConfig) -> ModernBertForMaskedLM:
    """Randomly initialize ``ModernBertForMaskedLM`` from config."""
    model = ModernBertForMaskedLM(config)
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
