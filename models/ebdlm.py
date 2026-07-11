"""LLaDA masked diffusion LM on ModernBERT."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import ModernBertConfig, ModernBertForMaskedLM

LLaDARemasking = Literal["random", "low_confidence"]
EnergyRemasking = Literal["energy_gradient", "energy_importance_sampling"]
RemaskingStrategy = LLaDARemasking | EnergyRemasking


class LLaDAMDLM(ModernBertForMaskedLM):
    """ModernBERT MaskedLM with LLaDA diffusion loss and reverse sampling."""

    def __init__(self, config: ModernBertConfig) -> None:
        super().__init__(config)

    def diffusion_loss(
        self, input_ids: Tensor, *, mask_eps: float = 1e-3
    ) -> tuple[Tensor, Tensor]:
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

        t = torch.rand(batch_size, device=device)
        p_mask = (mask_eps + (1.0 - mask_eps) * t).unsqueeze(1).expand(
            batch_size, seq_len
        )

        mask = torch.rand(batch_size, seq_len, device=device) < p_mask
        masked_ids = input_ids.masked_fill(mask, mask_token_id)
        logits = self(input_ids=masked_ids, attention_mask=attention_mask).logits

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
        ratio: float
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
        if remasking not in ["random", "low_confidence"]:
            raise ValueError(f"Invalid remasking strategy: {remasking}")

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
        times = torch.linspace(1, 0, num_steps + 1, device=input_tokens.device)
        for t, s in zip(times[:-1], times[1:], strict=True):
            ratio = s / t
            
            with torch.no_grad():
                # Get logits for current input_ids
                logits = self(input_tokens, attention_mask=attention_mask).logits

            input_tokens, mask = self._reverse_step(
                input_tokens, mask, logits, attention_mask, remasking, ratio
            )

            # For future energy subclass
            # elif remasking == "energy_importance_sampling":
            #     k = 3
            #     is_temp = 1.0

            #     # p_x0; K fills of current mask only (MDLM propose)
            #     probs = F.softmax(logits[mask], dim=-1)
            #     candidates = input_tokens.repeat(k, 1).clone()
            #     mask_k = mask.repeat(k, 1)
            #     candidates[mask_k] = torch.multinomial(
            #         probs.repeat(k, 1), num_samples=1
            #     ).squeeze(-1)
            #     candidates = candidates.view(k, batch_size, seq_len).transpose(0, 1)

            #     # sequence-level IS pick
            #     scores = self.energy(
            #         candidates.reshape(batch_size * k, seq_len)
            #     ).view(batch_size, k)
            #     scores = scores - scores.max(dim=-1, keepdim=True).values
            #     weights = F.softmax(scores / is_temp, dim=-1)
            #     idx = torch.multinomial(weights, num_samples=1).squeeze(-1)
            #     input_tokens = candidates[torch.arange(batch_size, device=device), idx]

            #     # same remask schedule as your random branch (MDLM multi-step)
            #     remask_probs = torch.rand_like(mask, dtype=torch.float) < s / t
            #     mask = mask & remask_probs
            #     input_tokens[mask] = mask_token_id

            # elif remasking == "energy_gradient":
            #     # e_i as autograd leaves (plan steps 1–3)
            #     embeds = self.get_input_embeddings()(input_tokens)
            #     embeds = embeds.detach().requires_grad_(True)
            #     out = self(
            #         inputs_embeds=embeds,
            #         attention_mask=attention_mask,
            #         output_hidden_states=True,
            #     )
            #     # hypothetical: energy head on last hidden → (B,)
            #     E = self.energy(out.hidden_states[-1])
            #     E.sum().backward()
            #     grad = embeds.grad  # ∂E/∂e_i, [B, L, H]
            #     # s_i = (∂E/∂e_i) · e_i / ||e_i||
            #     s_i = (grad * embeds).sum(dim=-1) / embeds.norm(dim=-1).clamp_min(1e-8)
            #     # only rank positions filled this step; prompt stays excluded via ~mask
            #     s_i = s_i.masked_fill(~mask, float("-inf"))
            #     new_mask = torch.zeros_like(mask)
            #     for b in range(batch_size):
            #         n = int(mask[b].sum().item())
            #         k = int(ratio * n)
            #         if k > 0:
            #             new_mask[b, torch.topk(s_i[b], k, largest=True).indices] = True
            #     mask = new_mask
            #     input_tokens[mask] = mask_token_id

        return input_tokens
