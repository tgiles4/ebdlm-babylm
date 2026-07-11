"""Reverse-diffusion sampling for BabyLM / WandB logging.

LLaDA (arXiv:2502.09992) defines two inference procedures:

* **Algorithm 4** — fill all masked positions, then remask a subset (random or
  low-confidence). Tokens can be re-predicted each step.
* **Algorithm 5** — predict masked positions but only *commit* top-k per step;
  committed tokens are never remasked (ML-GSAI/LLaDA ``generate.py``).

``unconditional_generate`` / ``generate`` default to Algorithm 4.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from transformers import ModernBertForMaskedLM

RemaskingStrategy = Literal["random", "low_confidence"]
Sampler = Literal["algorithm4", "algorithm5"]


# ---------------------------------------------------------------------------
# Algorithm 4 — fill → remask (arXiv:2502.09992 Alg. 4; §8.1 / §8.2)
# ---------------------------------------------------------------------------


def _fill_masked_positions(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    *,
    greedy: bool,
) -> torch.Tensor:
    """Write predicted tokens at positions where ``mask`` is True."""
    if greedy:
        predicted = logits.argmax(dim=-1)
        return torch.where(mask, predicted, input_ids)

    probs = F.softmax(logits[mask], dim=-1)
    filled = input_ids.clone()
    filled[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return filled


def _random_remask(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
    *,
    t: float,
    s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remask each currently-masked position independently with probability ``s/t``."""
    ratio = s / t if t > 0 else 0.0
    stay_masked = (torch.rand_like(mask, dtype=torch.float) < ratio) & mask
    updated_ids = input_ids.masked_fill(stay_masked, mask_token_id)
    return updated_ids, stay_masked


def _low_confidence_remask(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    mask_token_id: int,
    *,
    t: float,
    s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remask the lowest-confidence positions among those filled this step."""
    probs_all = F.softmax(logits, dim=-1)
    chosen_probs = probs_all.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    chosen_probs = chosen_probs.masked_fill(~mask, 1.0)

    ratio = s / t if t > 0 else 0.0
    stay_masked = torch.zeros_like(mask)
    for batch_idx in range(input_ids.shape[0]):
        num_masked = int(mask[batch_idx].sum().item())
        num_to_remask = int(ratio * num_masked)
        if num_to_remask > 0:
            lowest_idx = torch.topk(
                chosen_probs[batch_idx], num_to_remask, largest=False
            ).indices
            stay_masked[batch_idx, lowest_idx] = True

    updated_ids = input_ids.masked_fill(stay_masked, mask_token_id)
    return updated_ids, stay_masked


def _remask_step(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    mask_token_id: int,
    remasking: RemaskingStrategy,
    *,
    t: float,
    s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one Algorithm 4 remasking step."""
    if remasking == "random":
        return _random_remask(input_ids, mask, mask_token_id, t=t, s=s)
    if remasking == "low_confidence":
        return _low_confidence_remask(
            input_ids, mask, logits, mask_token_id, t=t, s=s
        )
    raise ValueError(f"Unknown remasking strategy: {remasking!r}")


@torch.no_grad()
def _sample_algorithm4(
    model: ModernBertForMaskedLM,
    input_ids: torch.Tensor,
    active_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    num_steps: int,
    remasking: RemaskingStrategy,
    *,
    greedy: bool,
) -> torch.Tensor:
    """Run Algorithm 4 on a batch already initialized with ids + active mask."""
    times = torch.linspace(1, 0, num_steps + 1, device=input_ids.device)
    for t, s in zip(times[:-1], times[1:], strict=True):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        input_ids = _fill_masked_positions(
            input_ids, active_mask, logits, greedy=greedy
        )
        input_ids, active_mask = _remask_step(
            input_ids,
            active_mask,
            logits,
            mask_token_id,
            remasking,
            t=float(t),
            s=float(s),
        )
    return input_ids


# ---------------------------------------------------------------------------
# Algorithm 5 — commit top-k per step (LLaDA generate.py)
# ---------------------------------------------------------------------------


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Precompute how many masked tokens to commit at each step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for batch_idx in range(mask_num.size(0)):
        num_transfer_tokens[batch_idx, : remainder[batch_idx, 0]] += 1
    return num_transfer_tokens


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max noise for non-greedy sampling (temperature=0 disables)."""
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _commit_scores(
    logits: torch.Tensor,
    x0: torch.Tensor,
    mask_index: torch.Tensor,
    remasking: RemaskingStrategy,
) -> torch.Tensor:
    """Per-position scores for Algorithm 5 commit ranking."""
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    if remasking == "low_confidence":
        probs = F.softmax(logits, dim=-1)
        scores = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        scores = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise ValueError(f"Unknown remasking strategy: {remasking!r}")
    return torch.where(mask_index, scores, neg_inf)


@torch.no_grad()
def _sample_algorithm5(
    model: ModernBertForMaskedLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    num_steps: int,
    remasking: RemaskingStrategy,
    *,
    greedy: bool,
    gen_start: int,
    gen_end: int,
) -> torch.Tensor:
    """Run Algorithm 5 on generation region ``[gen_start, gen_end)``."""
    batch_size = input_ids.shape[0]
    gen_length = gen_end - gen_start
    block_length = gen_length
    num_blocks = 1
    steps_per_block = num_steps
    temperature = 0.0 if greedy else 1.0

    for block_idx in range(num_blocks):
        block_start = gen_start + block_idx * block_length
        block_end = gen_start + (block_idx + 1) * block_length
        block_mask_index = input_ids[:, block_start:block_end] == mask_token_id
        num_transfer_tokens = _get_num_transfer_tokens(
            block_mask_index, steps_per_block
        )

        for step in range(steps_per_block):
            mask_index = input_ids == mask_token_id
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            logits_for_conf = logits
            logits = _add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits, dim=-1)
            x0 = torch.where(mask_index, x0, input_ids)

            scores = _commit_scores(logits_for_conf, x0, mask_index, remasking)
            scores[:, block_end:] = float("-inf")

            transfer_index = torch.zeros_like(input_ids, dtype=torch.bool)
            for batch_idx in range(batch_size):
                k = int(num_transfer_tokens[batch_idx, step].item())
                if k > 0:
                    _, select_index = torch.topk(scores[batch_idx], k=k)
                    transfer_index[batch_idx, select_index] = True

            input_ids = torch.where(transfer_index, x0, input_ids)

    return input_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    model: ModernBertForMaskedLM,
    seq_len: int,
    mask_token_id: int,
    num_steps: int,
    remasking: RemaskingStrategy = "low_confidence",
    batch_size: int = 1,
    greedy: bool = True,
    prompt_ids: torch.Tensor | None = None,
    pin_bos: bool = True,
    sampler: Sampler = "algorithm4",
) -> torch.Tensor:
    """Generate with optional prompt prefix.

    Defaults to Algorithm 4 (fill + remask). Pass ``sampler="algorithm5"`` for
    the commit schedule used in LLaDA ``generate.py``.
    """
    device = next(model.parameters()).device
    bos_token_id = int(model.config.bos_token_id)

    input_ids = torch.full(
        (batch_size, seq_len), mask_token_id, dtype=torch.long, device=device
    )

    if prompt_ids is None and pin_bos:
        prompt_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )
    elif prompt_ids is not None and prompt_ids.shape[0] == 1 and batch_size > 1:
        prompt_ids = prompt_ids.expand(batch_size, -1)

    prompt_len = 0 if prompt_ids is None else int(prompt_ids.shape[1])
    if prompt_len >= seq_len:
        raise ValueError(
            f"prompt length {prompt_len} must be less than seq_len {seq_len}"
        )

    if prompt_len > 0:
        input_ids[:, :prompt_len] = prompt_ids.to(device)

    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

    if sampler == "algorithm4":
        active_mask = input_ids == mask_token_id
        return _sample_algorithm4(
            model,
            input_ids,
            active_mask,
            attention_mask,
            mask_token_id,
            num_steps,
            remasking,
            greedy=greedy,
        )

    if sampler == "algorithm5":
        return _sample_algorithm5(
            model,
            input_ids,
            attention_mask,
            mask_token_id,
            num_steps,
            remasking,
            greedy=greedy,
            gen_start=prompt_len,
            gen_end=seq_len,
        )

    raise ValueError(f"Unknown sampler: {sampler!r}")


@torch.no_grad()
def unconditional_generate(
    model: ModernBertForMaskedLM,
    seq_len: int,
    mask_token_id: int,
    num_steps: int,
    remasking: RemaskingStrategy = "low_confidence",
    batch_size: int = 1,
    greedy: bool = True,
    pin_bos: bool = True,
    sampler: Sampler = "algorithm4",
) -> torch.Tensor:
    """Unconditional generation via Algorithm 4 by default."""
    return generate(
        model,
        seq_len=seq_len,
        mask_token_id=mask_token_id,
        num_steps=num_steps,
        remasking=remasking,
        batch_size=batch_size,
        greedy=greedy,
        prompt_ids=None,
        pin_bos=pin_bos,
        sampler=sampler,
    )
