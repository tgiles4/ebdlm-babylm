"""LLaDA masked diffusion LM on ModernBERT."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import ModernBertConfig, ModernBertForMaskedLM

LLaDARemasking = Literal["random", "low_confidence"]
EnergyRemasking = Literal["energy_gradient", "energy_importance_sampling"]
RemaskingStrategy = LLaDARemasking | EnergyRemasking


class Energy(nn.Module):
    """
    Aligned RelationNet-style head on frozen (h_t, h_0) pairs.

    Per position: norm each stream, r_i = MLP([h_t^i ; h_0^i]), then scalar e_i.
    Sequence energy is the mean of e_i over length.
    """
    def __init__(self, hidden_size: int, *, norm_eps: float = 1e-5) -> None:
        super().__init__()
        # Match ModernBERT: LayerNorm with scale, no bias.
        self.norm_t = nn.LayerNorm(hidden_size, eps=norm_eps, bias=False)
        self.norm_0 = nn.LayerNorm(hidden_size, eps=norm_eps, bias=False)
        pair_dim = 2 * hidden_size
        self.relation_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, h_t: Tensor, h_0: Tensor) -> Tensor:
        """Sequence energy. h_t, h_0 are [B, L, H] → [B]."""
        pair = torch.cat([self.norm_t(h_t), self.norm_0(h_0)], dim=-1)
        return self.relation_mlp(pair).squeeze(-1).mean(dim=-1)


class LLaDAMDLM(ModernBertForMaskedLM):
    """ModernBERT MaskedLM with LLaDA diffusion loss and reverse sampling."""

    def __init__(self, config: ModernBertConfig) -> None:
        super().__init__(config)

    def loss(self, input_ids: Tensor, *, mask_eps: float = 1e-3) -> tuple[Tensor, Tensor]:
        """
        Run one forward-noising step and compute the LLaDA pretraining loss.

        Follows the LLaDA pretraining procedure: sample a masking rate t uniformly
        per sequence, set p_mask = (1 - epsilon) * t + epsilon, mask tokens
        independently with probability p_mask, predict clean tokens at masked
        positions, and weight cross-entropy by 1 / p_mask. The loss is summed
        over masked positions and normalized by batch size times sequence length.

        Args:
            input_ids: Clean token ids x_0, shape [B, L].
            mask_eps: Minimum per-token mask probability in the LLaDA forward process.

        Returns:
            Scalar mean loss and the batch mean mask fraction (fraction of
            positions replaced by the mask token), useful as a training metric.
        """
        mask_token_id = self.config.mask_token_id

        # Randomly truncate the input sequence to 1% of the original length to learn PAD tokens.
        if torch.rand(1, device=input_ids.device) < 0.01:
            random_length = int(
                torch.randint(
                    1, input_ids.shape[1] + 1, (1,), device=input_ids.device
                ).item()
            )
            input_ids = input_ids[:, :random_length]

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

        # Sample a masking rate uniformly per sequence.
        t = torch.rand(batch_size, device=device)
        p_mask = (mask_eps + (1.0 - mask_eps) * t).unsqueeze(1).expand(
            batch_size, seq_len
        )

        mask = torch.rand(batch_size, seq_len, device=device) < p_mask
        masked_ids = input_ids.masked_fill(mask, mask_token_id)

        # Get the logits for the masked positions.
        logits = self(input_ids=masked_ids, attention_mask=attention_mask).logits

        # Compute the loss for the masked positions.
        token_loss = F.cross_entropy(
            logits[mask], input_ids[mask], reduction="none"
        ) / p_mask[mask]
        loss = token_loss.sum() / (batch_size * seq_len)
        mask_fraction = mask.float().mean()
        return loss, mask_fraction

    def _fill(self, input_tokens: Tensor, mask: Tensor, logits: Tensor) -> Tensor:
        """
        Fill masked positions with softmax probabilities.
        """
        probs = F.softmax(logits[mask], dim=-1)
        input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return input_tokens

    def _remask_random(self, input_tokens: Tensor, mask: Tensor, ratio: float) -> tuple[Tensor, Tensor]:
        """
        Random remask strategy from LLaDA Algorithm 4.
        https://arxiv.org/pdf/2502.09992

        Args:
            input_tokens: Tensor of shape [B, L] containing the input tokens.
            mask: Tensor of shape [B, L] containing the mask.
            ratio: Float between 0 and 1 indicating the ratio of positions to remask.

        Returns:
            Tuple of [B, L] tensors containing the input tokens and mask.
        """
        # Sample random probabilities for each position 
        remask_probs = torch.rand_like(mask, dtype=torch.float) < ratio

        # Update mask and fill masked positions with mask token
        mask = mask & remask_probs
        input_tokens[mask] = self.config.mask_token_id
        return input_tokens, mask
    
    def _remask_low_confidence(self, input_tokens: Tensor, mask: Tensor, logits: Tensor, ratio: float) -> tuple[Tensor, Tensor]:
        """
        Low-confidence remask strategy from LLaDA Algorithm 5.
        https://arxiv.org/pdf/2502.09992

        Args:
            input_tokens: Tensor of shape [B, L] containing the input tokens.
            mask: Tensor of shape [B, L] containing the mask.
            logits: Tensor of shape [B, L, V] containing the logits.
            ratio: Float between 0 and 1 indicating the ratio of positions to remask.

        Returns:
            Tuple of [B, L] tensors containing the input tokens and mask.
        """
        probs_all = F.softmax(logits, dim=-1)

        # Get probabilities for the chosen tokens
        chosen_probs = torch.gather(
            probs_all, 
            dim=-1, 
            index=input_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        chosen_probs = chosen_probs.masked_fill(~mask, 1.0)
        new_mask = torch.zeros_like(mask)

        # Iterate through batch to remask the lowest-confidence tokens
        for batch_idx in range(input_tokens.shape[0]):
            num_masked = int(mask[batch_idx].sum().item())
            num_to_remask = int(ratio * num_masked)
            if num_to_remask > 0:
                lowest_idx = torch.topk(
                    chosen_probs[batch_idx], num_to_remask, largest=False
                ).indices
                new_mask[batch_idx, lowest_idx] = True
        
        # Update mask and fill masked positions with mask token
        mask = new_mask
        input_tokens[mask] = self.config.mask_token_id
        return input_tokens, mask
    
    def _reverse_step(self,
        input_tokens: Tensor, 
        mask: Tensor, 
        logits: Tensor, 
        attention_mask: Tensor, 
        remasking: LLaDARemasking, 
        ratio: float,
    ) -> tuple[Tensor, Tensor]:
        """
        For standard LLaDA reverse step, options are random and low confidence remasking.

        Args:
            input_tokens: Tensor of shape [B, L] containing the input tokens.
            mask: Tensor of shape [B, L] containing the mask.
            logits: Tensor of shape [B, L, V] containing the logits.
            attention_mask: Tensor of shape [B, L] containing the attention mask.
            remasking: String indicating the remasking strategy to use.
            ratio: Float between 0 and 1 indicating the ratio of positions to remask.

        Returns:
            Tuple of [B, L] tensors containing the input tokens and mask.
        """
        input_tokens = self._fill(input_tokens, mask, logits)

        if remasking == "random":
            return self._remask_random(input_tokens, mask, ratio)
        
        if remasking == "low_confidence":
            return self._remask_low_confidence(input_tokens, mask, logits, ratio)
        
        raise ValueError(f"Invalid remasking strategy: {remasking}")

    def generate(
        self,
        num_steps: int,
        remasking: RemaskingStrategy = "low_confidence",
        batch_size: int = 1,
        prompt_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Generate via fill-then-remask.

        Args:
            num_steps: Integer indicating the number of steps to perform. More steps means more accurate generation.
            remasking: String indicating the remasking strategy to use.
            batch_size: Integer indicating the batch size.
            prompt_ids: Tensor of shape [B, L] containing the prompt ids.

        Returns:
            Tensor of shape [B, L] containing the generated tokens.

        Raises:
            ValueError: If the remasking strategy is invalid.
        """
        device = next(self.parameters()).device

        # Generate a full [MASK] canvas
        input_tokens = torch.full(
            (batch_size, self.config.max_position_embeddings),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device
        )

        # If given prompt_ids, overwrite the left prefix
        if prompt_ids is not None:
            input_tokens[:, : prompt_ids.shape[1]] = prompt_ids.to(device)

        attention_mask = torch.ones(
            batch_size, self.config.max_position_embeddings, dtype=torch.long, device=device
        )

        # Create mask so prompt_ids are not remasked
        mask = input_tokens == self.config.mask_token_id

        # Perform reverse process
        times = torch.linspace(1, 0, num_steps + 1, device=input_tokens.device).tolist()
        for t, s in zip(times[:-1], times[1:], strict=True):
            self._t = t
            ratio = s / t
            
            with torch.no_grad():
                # Get logits for current input_ids
                logits = self(input_tokens, attention_mask=attention_mask).logits

            input_tokens, mask = self._reverse_step(
                input_tokens, mask, logits, attention_mask, remasking, ratio
            )

        return input_tokens


class EDLM(LLaDAMDLM):
    """LLaDA MDLM + energy head for NCE and energy-guided remasking."""

    def __init__(self, llada: LLaDAMDLM) -> None:
        """
        Wrap an existing LLaDAMDLM: copy its weights, freeze the backbone, add
        the energy head. Does not construct a backbone from config alone.
        """
        if not isinstance(llada, LLaDAMDLM) or isinstance(llada, EDLM):
            raise TypeError("EDLM requires a LLaDAMDLM instance")
        super().__init__(llada.config)
        self.load_state_dict(llada.state_dict(), strict=True)
        for param in self.parameters():
            param.requires_grad = False
        self.energy_head = Energy(llada.config.hidden_size)
        self.config.architectures = ["EDLM"]

    def _backbone_hidden(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Frozen ModernBERT last hidden states [B, L, H]."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    def energy(
        self,
        xt_ids: Tensor,
        x0_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Transition energy E(x_t, x_0) of shape [B].

        Runs the frozen backbone on each sequence, then the RelationNet head.
        Backbone is under no_grad so only energy_head receives gradients.
        """
        with torch.no_grad():
            h_t = self._backbone_hidden(xt_ids, attention_mask)
            h_0 = self._backbone_hidden(x0_ids, attention_mask)

        return self.energy_head(h_t, h_0)

    def loss(
        self, input_ids: Tensor, *, mask_eps: float = 1e-3
    ) -> tuple[Tensor, Tensor]:
        """
        Noise-contrastive energy loss on transitions (x_t, x_0).

        Mask with the LLaDA forward process, propose x_0^- by filling under the
        frozen backbone, then NCE so real (x_t, x_0^+) scores lower energy than
        (x_t, x_0^-).

        Args:
            input_ids: Clean token ids x_0^+, shape [B, L].
            mask_eps: Minimum per-token mask probability in the LLaDA forward process.

        Returns:
            Scalar mean NCE loss and the batch mean mask fraction.
        """
        mask_token_id = self.config.mask_token_id
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

        t = torch.rand(batch_size, device=device)
        p_mask = (mask_eps + (1.0 - mask_eps) * t).unsqueeze(1).expand(
            batch_size, seq_len
        )
        mask = torch.rand(batch_size, seq_len, device=device) < p_mask
        xt_ids = input_ids.masked_fill(mask, mask_token_id)

        with torch.no_grad():
            logits = self(input_ids=xt_ids, attention_mask=attention_mask).logits
            x0_neg = self._fill(xt_ids.clone(), mask, logits)

        energy_pos = self.energy(xt_ids, input_ids, attention_mask)
        energy_neg = self.energy(xt_ids, x0_neg, attention_mask)
        loss = -(F.logsigmoid(-energy_pos) + F.logsigmoid(energy_neg)).mean()
        mask_fraction = mask.float().mean()
        return loss, mask_fraction

    def _remask_energy_importance(
        self,
        input_tokens: Tensor,
        mask: Tensor,
        logits: Tensor,
        attention_mask: Tensor,
        ratio: float,
    ) -> tuple[Tensor, Tensor]:
        """
        K fills of the current mask, pick one by transition energy IS, then
        randomly remask with probability ratio (= s/t).
        """
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device
        k = self.config.is_size
        xt_ids = input_tokens

        # Get the probabilities for the masked positions.
        probs = F.softmax(logits[mask], dim=-1)

        # Create k candidates by repeating the input tokens and masking them.
        candidates = input_tokens.repeat(k, 1).clone()
        mask_k = mask.repeat(k, 1)
        candidates[mask_k] = torch.multinomial(
            probs.repeat(k, 1), num_samples=1
        ).squeeze(-1)
        candidates = candidates.view(k, batch_size, seq_len).transpose(0, 1)

        # Get the transition energies for the candidates.
        xt_flat = xt_ids.unsqueeze(1).expand(-1, k, -1).reshape(batch_size * k, seq_len)
        x0_flat = candidates.reshape(batch_size * k, seq_len)
        attn_flat = attention_mask.unsqueeze(1).expand(-1, k, -1).reshape(
            batch_size * k, seq_len
        )

        # Low energy = better (matches NCE); weight by exp(-E / T).
        scores = self.energy(xt_flat, x0_flat, attn_flat).view(batch_size, k)
        neg_scores = -scores
        neg_scores = neg_scores - neg_scores.max(dim=-1, keepdim=True).values
        weights = F.softmax(neg_scores / self.config.is_temp, dim=-1)
        idx = torch.multinomial(weights, num_samples=1).squeeze(-1)

        # Select the candidate with the lowest energy.
        input_tokens = candidates[torch.arange(batch_size, device=device), idx]

        # Remask the selected candidate with the random remasking strategy.
        return self._remask_random(input_tokens, mask, ratio)

    def _remask_energy_gradient(
        self,
        input_tokens: Tensor,
        mask: Tensor,
        logits: Tensor,
        attention_mask: Tensor,
        ratio: float,
    ) -> tuple[Tensor, Tensor]:
        """
        Fill once, then remask positions with largest directional energy derivative.

        s_i = (∂E/∂e_i) · e_i / ||e_i|| on input embeddings of the filled
        x_0; x_t is held fixed. Backbone weights stay frozen; grads flow
        only into the embedding tensor.
        """
        xt_ids = input_tokens.clone()

        # Fill the masked positions with the softmax probabilities.
        input_tokens = self._fill(input_tokens, mask, logits)

        with torch.no_grad():
            h_t = self._backbone_hidden(xt_ids, attention_mask)

        # Differentiable embeds for filled x_0 (lookup weights are frozen).
        with torch.enable_grad():
            embeds = (
                self.get_input_embeddings()(input_tokens)
                .detach()
                .requires_grad_(True)
            )
            h_0 = self.model(
                inputs_embeds=embeds, attention_mask=attention_mask
            ).last_hidden_state
            energy = self.energy_head(h_t, h_0)
            (grad_embeds,) = torch.autograd.grad(energy.sum(), embeds)

        # Compute the directional energy derivative.
        emb = embeds.detach()
        s_i = (grad_embeds * emb).sum(dim=-1) / emb.norm(dim=-1).clamp_min(1e-12)
        s_i = s_i.masked_fill(~mask, float("-inf"))

        # Iterate through the batch to remask the positions with the highest directional energy derivative.
        new_mask = torch.zeros_like(mask)
        for batch_idx in range(input_tokens.shape[0]):
            num_masked = int(mask[batch_idx].sum().item())
            num_to_remask = int(ratio * num_masked)
            if num_to_remask > 0:
                top_idx = torch.topk(
                    s_i[batch_idx], num_to_remask, largest=True
                ).indices
                new_mask[batch_idx, top_idx] = True

        # Update the mask and fill the masked positions with the mask token.
        mask = new_mask
        input_tokens[mask] = self.config.mask_token_id
        return input_tokens, mask

    def _reverse_step(
        self,
        input_tokens: Tensor,
        mask: Tensor,
        logits: Tensor,
        attention_mask: Tensor,
        remasking: RemaskingStrategy,
        ratio: float,
    ) -> tuple[Tensor, Tensor]:
        """Dispatch LLaDA remasking to the base class; energy strategies here."""
        if remasking == "energy_importance_sampling":
            in_band = self.config.is_stop <= self._t <= self.config.is_start
            if not in_band:
                return super()._reverse_step(
                    input_tokens,
                    mask,
                    logits,
                    attention_mask,
                    "random",
                    ratio,
                )
            return self._remask_energy_importance(
                input_tokens, mask, logits, attention_mask, ratio
            )

        if remasking == "energy_gradient":
            return self._remask_energy_gradient(
                input_tokens, mask, logits, attention_mask, ratio
            )
        
        return super()._reverse_step(
                input_tokens,
                mask,
                logits,
                attention_mask,
                remasking,
                ratio,
            )
