"""Build LLaDA / EDLM ModernBERT models from Hydra config."""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from transformers import ModernBertConfig, PreTrainedTokenizerFast

from models.ebdlm import EDLM, LLaDAMDLM

logger = logging.getLogger(__name__)


def modernbert_config_from_cfg(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> ModernBertConfig:
    """Map shared Hydra fields + model profile + live tokenizer IDs to config."""
    config = ModernBertConfig(
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
        cls_token_id=None,
        sep_token_id=None,
    )
    _apply_energy_config(config, cfg)
    return config


def _apply_energy_config(config: ModernBertConfig, cfg: DictConfig) -> None:
    """Copy Hydra energy knobs onto a ModernBertConfig when present."""
    energy = cfg.get("energy")
    if energy is None:
        return
    config.is_size = int(energy.is_size)
    config.is_temp = float(energy.is_temp)
    config.is_start = float(energy.is_start)
    config.is_stop = float(energy.is_stop)


def _attn_implementation() -> str:
    """Use Flash Attention 2 when installed; otherwise SDPA."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    return "flash_attention_2"


def create_model(config: ModernBertConfig) -> LLaDAMDLM:
    """Randomly initialize LLaDAMDLM from config."""
    attn = _attn_implementation()
    logger.info("Using attention implementation: %s", attn)
    config.attn_implementation = attn
    config.architectures = ["LLaDAMDLM"]
    if not hasattr(config, "words_seen") or config.words_seen is None:
        config.words_seen = 0
    model = LLaDAMDLM(config)
    param_count = model.num_parameters()
    logger.info("Initialized LLaDAMDLM with %s parameters", f"{param_count:,}")
    return model


def create_model_from_cfg(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerFast,
) -> LLaDAMDLM:
    """Build LLaDAMDLM from config, or EDLM from paths.llada_checkpoint."""
    train_energy = bool(OmegaConf.select(cfg, "train_energy", default=False))
    if train_energy:
        ckpt = OmegaConf.select(cfg, "paths.llada_checkpoint", default=None)
        if ckpt is None or str(ckpt).strip() == "":
            raise ValueError(
                "train_energy=true requires paths.llada_checkpoint "
                "(HF directory of a trained LLaDAMDLM)"
            )
        ckpt_path = Path(str(ckpt))
        attn = _attn_implementation()
        logger.info(
            "Loading LLaDAMDLM from %s (attn=%s) for EDLM", ckpt_path, attn
        )
        llada = LLaDAMDLM.from_pretrained(
            ckpt_path, attn_implementation=attn
        )
        _apply_energy_config(llada.config, cfg)
        model = EDLM(llada)
        logger.info(
            "Initialized EDLM with %s parameters", f"{model.num_parameters():,}"
        )
        return model

    config = modernbert_config_from_cfg(cfg, tokenizer)
    return create_model(config)
