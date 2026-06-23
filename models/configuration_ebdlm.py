"""Configuration for EBDLM diffusion and energy-diffusion models."""

from __future__ import annotations

from transformers import PretrainedConfig


class EbdlmConfig(PretrainedConfig):
    """Config for ``DiffusionModel`` and ``EnergyDiffusionModel``.

    Backbone defaults target ~115M params (16 layers, 768 hidden, 12 heads,
    vocab 16384, max length 512). Energy head adds ~2–5M when present.
    """

    model_type = "ebdlm"

    def __init__(
        self,
        vocab_size: int = 16384,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        layer_norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        mask_token_id: int | None = None,
        cls_token_id: int | None = None,
        initializer_range: float = 0.02,
        use_energy_head: bool = False,
        energy_hidden_ratio: float = 0.25,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.initializer_range = initializer_range
        self.use_energy_head = use_energy_head
        self.energy_hidden_ratio = energy_hidden_ratio

        super().__init__(pad_token_id=pad_token_id, **kwargs)


class DiffusionConfig(EbdlmConfig):
    """Phase-1 diffusion-only checkpoint."""

    model_type = "ebdlm_diffusion"

    def __init__(self, **kwargs):
        kwargs.setdefault("use_energy_head", False)
        super().__init__(**kwargs)


class EnergyDiffusionConfig(EbdlmConfig):
    """Phase-2 checkpoint with energy head weights."""

    model_type = "ebdlm_energy_diffusion"

    def __init__(self, **kwargs):
        kwargs.setdefault("use_energy_head", True)
        super().__init__(**kwargs)
