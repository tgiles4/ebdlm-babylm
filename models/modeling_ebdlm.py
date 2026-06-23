"""EBDLM model stubs: masked diffusion LM + optional energy head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from transformers import ModernBertConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

RemaskingStrategy = Literal["low_confidence", "random", "energy_gradient"]


@dataclass
class DiffusionModelOutput(ModelOutput):
    """Single masked forward pass (train, GLUE, eval scoring, one inference step)."""

    logits: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    last_hidden_state: Tensor | None = None
    loss: Tensor | None = None


@dataclass
class InferenceOutput(ModelOutput):
    """Result of the full LLaDA reverse loop."""

    sequences: Tensor
    # Optional trace for debugging / demos.
    num_steps: int | None = None


class EnergyHead(nn.Module):
    """MLP on backbone hidden states → scalar sequence energy E(x).

    Architecture (plan):
        LayerNorm → Linear(d,d) → GELU → Linear(d,d) → mean_pool → Linear(d,d/4) → GELU → Linear(d/4,1)
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        # TODO: LayerNorm → Linear → GELU → Linear → pool → Linear → GELU → Linear

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Return per-batch scalar energy ``E(x)`` of shape ``(batch,)``."""
        raise NotImplementedError


class DiffusionModel(PreTrainedModel):
    """Masked diffusion language model (ModernBERT-style backbone + lm_head).

    - ``forward()``: one bidirectional masked forward → logits (+ hidden states).
    - ``inference()``: LLaDA reverse loop (fill → remask → next timestep).

    Training (phase 1) uses ``forward()`` only; the generative loop lives in
    ``inference()`` and is never called from the pretrain script.
    """

    config_class = ModernBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        # TODO: self.model (ModernBertModel) + self.lm_head; init from ModernBERT-base.
        self.post_init()

    # ------------------------------------------------------------------
    # HF core
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> DiffusionModelOutput | tuple:
        """One masked forward pass. Does **not** run the generative loop."""
        raise NotImplementedError

    def get_input_embeddings(self) -> nn.Module:
        raise NotImplementedError

    def set_input_embeddings(self, value: nn.Module) -> None:
        raise NotImplementedError

    def get_output_embeddings(self) -> nn.Module:
        raise NotImplementedError

    def tie_weights(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_modernbert_pretrained(
        cls,
        pretrained_name_or_path: str = "answerdotai/ModernBERT-base",
        config: ModernBertConfig | None = None,
        **kwargs,
    ) -> DiffusionModel:
        """Init backbone weights from a ModernBERT checkpoint (+ resize vocab if needed)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Generation (LLaDA Alg. 5 + random variant)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        steps: int = 64,
        remasking: RemaskingStrategy = "low_confidence",
        prompt_mask: Tensor | None = None,
        temperature: float = 0.0,
        return_dict: bool = True,
    ) -> InferenceOutput | Tensor:
        """Iterative fill + remask from t=1 … 0.

        Args:
            input_ids: Prompt + masked answer region. Positions to generate start as
                ``config.mask_token_id``.
            attention_mask: Standard HF padding mask.
            steps: Number of denoising steps ``N`` (schedule uses ``N+1`` time points).
            remasking: ``"low_confidence"`` | ``"random"``. Energy remasking is
                handled by ``EnergyDiffusionModel``.
            prompt_mask: Boolean mask of pinned positions (prompt); excluded from
                fill and remask.
            temperature: Sampling temperature for fill step (0 = argmax).
        """
        raise NotImplementedError

    def _iter_schedule(self, steps: int) -> list[tuple[float, float]]:
        """Return consecutive ``(t, s)`` pairs with ``s = t - 1/steps``."""
        raise NotImplementedError

    def _fill(
        self,
        input_ids: Tensor,
        logits: Tensor,
        mask: Tensor,
        *,
        temperature: float = 0.0,
    ) -> Tensor:
        """Reveal tokens at currently masked positions; leave pinned slots unchanged."""
        raise NotImplementedError

    def _remask(
        self,
        input_ids: Tensor,
        logits: Tensor,
        mask: Tensor,
        *,
        t: float,
        s: float,
        remasking: RemaskingStrategy,
        prompt_mask: Tensor | None = None,
    ) -> Tensor:
        """Dispatch to confidence / random remasking. Energy path overridden in subclass."""
        raise NotImplementedError

    def _remask_low_confidence(
        self,
        input_ids: Tensor,
        logits: Tensor,
        mask: Tensor,
        *,
        t: float,
        s: float,
        prompt_mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    def _remask_random(
        self,
        input_ids: Tensor,
        mask: Tensor,
        *,
        t: float,
        s: float,
        prompt_mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Eval scoring (diffusion-only; sentence + word tasks)
    # ------------------------------------------------------------------

    def masked_forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> DiffusionModelOutput:
        """One forward with an explicit mask pattern (word surprisal, pseudo_ll internals)."""
        raise NotImplementedError

    def word_surprisal(
        self,
        input_ids: Tensor,
        target_positions: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """`-log p(word | context)` via a single masked forward."""
        raise NotImplementedError

    def pseudo_ll(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        response_mask: Tensor | None = None,
        n_mc: int = 128,
    ) -> Tensor:
        """Algorithm 3 Monte Carlo pseudo log-likelihood (BLiMP / minimal pairs)."""
        raise NotImplementedError

    def score_sentence(
        self,
        candidate_a: Tensor,
        candidate_b: Tensor,
        attention_mask_a: Tensor | None = None,
        attention_mask_b: Tensor | None = None,
        *,
        n_mc: int = 128,
    ) -> Tensor:
        """Return ``+1`` if ``score(a) > score(b)``, else ``-1`` (diffusion-only score)."""
        raise NotImplementedError


class EnergyDiffusionModel(DiffusionModel):
    """Diffusion LM + energy head for NCE training and energy-gradient remasking.

    Inference is **one loop**: fill (diffusion) → remask (energy or fallback) →
    next timestep. The energy head is optional at runtime via ``remasking=``.
    """

    config_class = ModernBertConfig

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.energy_head = EnergyHead(config)
        self.post_init()

    @property
    def has_energy_head(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Energy (phase-2 train + eval + remasking)
    # ------------------------------------------------------------------

    def energy(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Scalar sequence energy ``E(x)`` of shape ``(batch,)``. Lower = more coherent."""
        raise NotImplementedError

    def token_energy_scores(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        prompt_mask: Tensor | None = None,
    ) -> Tensor:
        """Per-position directional derivative ``s_i = (∂E/∂e_i) · e_i / ||e_i||``.

        Gradients w.r.t. input embeddings only; backbone weights stay frozen.
        Positions in ``prompt_mask`` should be excluded by the caller / remask step.
        """
        raise NotImplementedError

    def nce_loss(
        self,
        input_ids_pos: Tensor,
        input_ids_neg: Tensor,
        attention_mask_pos: Tensor | None = None,
        attention_mask_neg: Tensor | None = None,
    ) -> Tensor:
        """Noise-contrastive energy loss for phase-2 training."""
        raise NotImplementedError

    def freeze_backbone(self) -> None:
        """Freeze backbone + lm_head for energy-head-only training."""
        raise NotImplementedError

    def _energy_remask(
        self,
        input_ids: Tensor,
        mask: Tensor,
        *,
        t: float,
        s: float,
        prompt_mask: Tensor | None = None,
    ) -> Tensor:
        """Remask positions with largest ``s_i`` (hurts global energy)."""
        raise NotImplementedError

    def _remask(
        self,
        input_ids: Tensor,
        logits: Tensor,
        mask: Tensor,
        *,
        t: float,
        s: float,
        remasking: RemaskingStrategy,
        prompt_mask: Tensor | None = None,
    ) -> Tensor:
        """Route ``energy_gradient`` to ``_energy_remask``; delegate others to parent."""
        raise NotImplementedError

    def inference(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        steps: int = 64,
        remasking: RemaskingStrategy = "energy_gradient",
        prompt_mask: Tensor | None = None,
        temperature: float = 0.0,
        return_dict: bool = True,
    ) -> InferenceOutput | Tensor:
        """Same reverse loop as ``DiffusionModel``; remask branch selects energy or fallback."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Eval scoring (sentence tasks add ``-E(x)``)
    # ------------------------------------------------------------------

    def pseudo_ll(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        response_mask: Tensor | None = None,
        n_mc: int = 128,
    ) -> Tensor:
        """``pseudo_ll(x) + (-E(x))`` when energy head is active."""
        raise NotImplementedError

    def score_sentence(
        self,
        candidate_a: Tensor,
        candidate_b: Tensor,
        attention_mask_a: Tensor | None = None,
        attention_mask_b: Tensor | None = None,
        *,
        n_mc: int = 128,
    ) -> Tensor:
        """Compare ``pseudo_ll + (-E)`` for minimal-pair sentence eval."""
        raise NotImplementedError

    def energy_score(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """``-E(x)`` alone (debug / ablation)."""
        raise NotImplementedError

    @classmethod
    def from_diffusion_checkpoint(
        cls,
        diffusion_model: DiffusionModel | str,
        config: ModernBertConfig | None = None,
        **kwargs,
    ) -> EnergyDiffusionModel:
        """Attach a fresh energy head to a phase-1 diffusion checkpoint."""
        raise NotImplementedError
