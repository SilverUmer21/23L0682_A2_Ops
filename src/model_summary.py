"""Task 2.2 evidence: baseline CNN summary and a per-layer table (output shape, parameter count).

Usage (from the repository root, after `source scripts/gpu_env.sh`):
    python src/model_summary.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from visionguard.models import build_baseline_cnn


def layer_table(model) -> pd.DataFrame:
    rows = []
    for layer in model.layers:
        rows.append({
            "layer": layer.name,
            "type": type(layer).__name__,
            "output_shape": str(tuple(layer.output.shape)),   # None = batch size
            "parameters": layer.count_params(),
            "trainable": int(sum(np.prod(w.shape) for w in layer.trainable_weights)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = build_baseline_cnn()
    lines: list[str] = []
    model.summary(print_fn=lambda line, **_: lines.append(line))
    print("\n".join(lines))
    (args.out / "baseline_cnn_summary.txt").write_text("\n".join(lines) + "\n")

    table = layer_table(model)
    table.to_csv(args.out / "baseline_cnn_layers.csv", index=False)
    trainable = int(table["trainable"].sum())
    print(f"\n{len(table)} layers | total parameters {model.count_params():,} | trainable {trainable:,} "
          f"| non-trainable {model.count_params() - trainable:,}")


if __name__ == "__main__":
    main()
