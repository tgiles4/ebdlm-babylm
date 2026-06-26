"""
Unconditional reverse-diffusion sampling for WandB sample logging.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from transformers import ModernBertForMaskedLM

RemaskingStrategy = Literal["random", "low_confidence"]


@torch.no_grad()
def unconditional_generate(
    model: ModernBertForMaskedLM,
    seq_len: int,
    mask_token_id: int,
    num_steps: int,
    remasking: RemaskingStrategy = "low_confidence",
    batch_size: int = 1,
    greedy: bool = True,
) -> torch.Tensor:
    """Generate text from an all-mask sequence via reverse masked diffusion.

    Starts with x_1 = [M, ..., M] and walks timesteps from t=1 down to 0. At
    each step the bidirectional mask predictor fills currently masked positions,
    then a remasking rule sets some positions back to M so the mask ratio tracks
    the next time s.

    Remasking modes (faithful to reference inference.py for v1):

    * "random" — among positions masked at this step, remask each with
      probability s / t (§8.1).
    * "low_confidence" — remask the floor((s/t) * n_masked) lowest-confidence
      predicted tokens (§8.2; reference uses int((s/t)*mask.sum()) rather than
      floor(L * (1-s))).

    Args:
        model: Bidirectional mask predictor in eval mode.
        seq_len: Fixed generation length L.
        mask_token_id: Vocabulary id for the mask token M.
        num_steps: Number of reverse-diffusion steps N.
        remasking: "random" or "low_confidence".
        batch_size: Number of independent samples to generate in parallel.
        greedy: If True, take argmax at masked positions; otherwise multinomial
            sample (reference inference default).

    Returns:
        Generated token ids [batch_size, seq_len]. Decode with
        tokenizer.batch_decode(..., skip_special_tokens=True).
    """
    device = next(model.parameters()).device
    input_ids = torch.full(
        (batch_size, seq_len), mask_token_id, dtype=torch.long, device=device
    )
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    times = torch.linspace(1, 0, num_steps + 1, device=device)
    for t, s in zip(times[:-1], times[1:], strict=True):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        input_ids = _fill_masked_positions(input_ids, mask, logits, greedy=greedy)
        input_ids, mask = _remask(
            input_ids=input_ids,
            mask=mask,
            logits=logits,
            mask_token_id=mask_token_id,
            remasking=remasking,
            t=float(t),
            s=float(s),
        )

    return input_ids


def _fill_masked_positions(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    *,
    greedy: bool,
) -> torch.Tensor:
    """Write predicted tokens at positions where mask is True."""
    if greedy:
        predicted = logits.argmax(dim=-1)
        return torch.where(mask, predicted, input_ids)

    probs = F.softmax(logits[mask], dim=-1)
    filled = input_ids.clone()
    filled[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return filled


def _remask(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    mask_token_id: int,
    remasking: RemaskingStrategy,
    t: float,
    s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one remasking step; return updated ids and the new mask bitmap."""
    if remasking == "random":
        return _random_remask(input_ids, mask, mask_token_id, t=t, s=s)
    if remasking == "low_confidence":
        return _low_confidence_remask(
            input_ids, mask, logits, mask_token_id, t=t, s=s
        )
    raise ValueError(f"Unknown remasking strategy: {remasking!r}")


def _random_remask(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
    *,
    t: float,
    s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remask a random subset of currently masked positions with probability s/t."""
    ratio = s / t if t > 0 else 0.0
    remask = (torch.rand_like(mask, dtype=torch.float) < ratio) & mask
    updated_ids = input_ids.masked_fill(remask, mask_token_id)
    return updated_ids, remask


def _low_confidence_remask(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    mask_token_id: int,
    *,
    t: float,
    s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remask the lowest-confidence positions among those masked this step."""
    probs_all = F.softmax(logits, dim=-1)
    chosen_probs = probs_all.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    chosen_probs = chosen_probs.masked_fill(~mask, 1.0)

    ratio = s / t if t > 0 else 0.0
    new_mask = torch.zeros_like(mask)
    for batch_idx in range(input_ids.shape[0]):
        num_masked = int(mask[batch_idx].sum().item())
        num_to_remask = int(ratio * num_masked)
        if num_to_remask > 0:
            lowest_idx = torch.topk(
                chosen_probs[batch_idx], num_to_remask, largest=False
            ).indices
            new_mask[batch_idx, lowest_idx] = True

    updated_ids = input_ids.masked_fill(new_mask, mask_token_id)
    return updated_ids, new_mask
