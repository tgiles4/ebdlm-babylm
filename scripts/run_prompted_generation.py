"""Prompted low_confidence generation inspect for saved HF checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.ebdlm import LLaDAMDLM

PROMPTS = [
    "*MOT:\twhat is that?",
    "*CHI:\tI want",
    "*MOT:\tcome on",
    "The cat sat on the",
]


def _run_prompted(model_dir: Path, out_lines: list[str]) -> None:
    """Run low_confidence continuation from fixed prompt prefixes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append(f"model={model_dir}")
    out_lines.append(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = LLaDAMDLM.from_pretrained(model_dir, attn_implementation="sdpa")
    model.eval().to(device)

    seq_len = int(model.config.max_position_embeddings)
    steps = 64

    for prompt_text in PROMPTS:
        encoded = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt")
        prompt_ids = encoded["input_ids"].to(device)
        prompt_len = int(prompt_ids.shape[1])
        out_lines.append("")
        out_lines.append("=" * 70)
        out_lines.append(f'prompt="{prompt_text}" ({prompt_len} tokens)')
        out_lines.append(
            f"remasking=low_confidence, seq_len={seq_len}, steps={steps}"
        )

        ids = model.generate(
            num_steps=steps,
            remasking="low_confidence",
            batch_size=1,
            prompt_ids=prompt_ids,
        )
        row = ids[0]
        n_unique = len(set(row.tolist()))
        pad_id = int(model.config.pad_token_id)
        pad_count = int((row == pad_id).sum().item())
        text = tokenizer.decode(row, skip_special_tokens=True)
        out_lines.append(
            f"unique={n_unique}, pad={pad_count}/{seq_len}, prompt_len={prompt_len}"
        )
        out_lines.append(text)
        out_lines.append("")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    """Compare prompted low_confidence on new and old checkpoints."""
    out_path = Path("runs/new/generation_prompted.txt")
    out_lines = ["Prompted low_confidence generation"]

    for label, path in [
        ("NEW", Path("runs/new/hf/last")),
        ("OLD (213248)", Path("runs/213248/hf/last")),
    ]:
        out_lines.append("")
        out_lines.append(f"# {label}")
        _run_prompted(path, out_lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
