"""One-off unconditional generation inspect for saved HF checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, ModernBertForMaskedLM

from training.generation import unconditional_generate


def _run_model(model_dir: Path, out_lines: list[str], *, pin_bos: bool) -> None:
    """Generate samples from one checkpoint and append formatted lines."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append(f"model={model_dir}")
    out_lines.append(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ModernBertForMaskedLM.from_pretrained(
        model_dir, attn_implementation="sdpa"
    )
    model.eval().to(device)

    mask_token_id = int(model.config.mask_token_id)
    out_lines.append(f"mask_token_id={mask_token_id}")
    out_lines.append(f"pin_bos={pin_bos}")

    seq_len = 512
    steps = 64
    batch = 4
    for remasking in ("low_confidence", "random"):
        out_lines.append("")
        out_lines.append("=" * 70)
        out_lines.append(
            f"remasking={remasking}, batch={batch}, seq_len={seq_len}, steps={steps}"
        )
        ids = unconditional_generate(
            model,
            seq_len=seq_len,
            mask_token_id=mask_token_id,
            num_steps=steps,
            remasking=remasking,  # type: ignore[arg-type]
            batch_size=batch,
            pin_bos=pin_bos,
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
    import sys

    pin_bos = "--no-bos" not in sys.argv
    suffix = "" if pin_bos else "_no_bos"
    out_path = Path(f"runs/new/generation_compare{suffix}.txt")
    new_only_path = Path(f"runs/new/generation_inspect{suffix}.txt")
    out_lines = [
        f"Generation comparison: new vs old checkpoint (pin_bos={pin_bos})"
    ]

    models = [
        ("NEW", Path("runs/new/hf/last")),
        ("OLD (213248)", Path("runs/213248/hf/last")),
    ]
    for label, path in models:
        out_lines.append("")
        out_lines.append(f"# {label}")
        _run_model(path, out_lines, pin_bos=pin_bos)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

    try:
        old_idx = next(i for i, line in enumerate(out_lines) if line == "# OLD (213248)")
        new_lines = out_lines[:old_idx]
    except StopIteration:
        new_lines = out_lines
    new_only_path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"Wrote {new_only_path} ({new_only_path.stat().st_size} bytes)")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
