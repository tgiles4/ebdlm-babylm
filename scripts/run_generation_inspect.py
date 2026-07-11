"""One-off unconditional generation inspect for saved HF checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.ebdlm import LLaDAMDLM


def _run_model(model_dir: Path, out_lines: list[str]) -> None:
    """Generate samples from one checkpoint and append formatted lines."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append(f"model={model_dir}")
    out_lines.append(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = LLaDAMDLM.from_pretrained(model_dir, attn_implementation="sdpa")
    model.eval().to(device)

    mask_token_id = int(model.config.mask_token_id)
    seq_len = int(model.config.max_position_embeddings)
    out_lines.append(f"mask_token_id={mask_token_id}")
    out_lines.append(f"seq_len={seq_len}")

    steps = 64
    batch = 4
    for remasking in ("low_confidence", "random"):
        out_lines.append("")
        out_lines.append("=" * 70)
        out_lines.append(
            f"remasking={remasking}, batch={batch}, seq_len={seq_len}, steps={steps}"
        )
        ids = model.generate(
            num_steps=steps,
            remasking=remasking,  # type: ignore[arg-type]
            batch_size=batch,
        )
        texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
        for i, text in enumerate(texts, 1):
            n_unique = len(set(ids[i - 1].tolist()))
            out_lines.append(
                f"--- sample {i}: {n_unique} unique token ids / {seq_len} ---"
            )
            out_lines.append(text)
            out_lines.append("")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    """Compare new and old HF checkpoints with both remasking strategies."""
    out_path = Path("runs/new/generation_compare.txt")
    new_only_path = Path("runs/new/generation_inspect.txt")
    out_lines = ["Generation comparison: new vs old checkpoint"]

    models = [
        ("NEW", Path("runs/new/hf/last")),
        ("OLD (213248)", Path("runs/213248/hf/last")),
    ]
    for label, path in models:
        out_lines.append("")
        out_lines.append(f"# {label}")
        _run_model(path, out_lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

    try:
        old_idx = next(i for i, line in enumerate(out_lines) if line == "# OLD (213248)")
    except StopIteration:
        new_lines = out_lines
    else:
        new_lines = out_lines[:old_idx]
    new_only_path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"Wrote {new_only_path} ({new_only_path.stat().st_size} bytes)")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
