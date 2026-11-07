"""Task 1.3 - visual EDA for VisionGuard.

Reads the audit table written by data_audit.py and produces:
  * a grid of random sample images per class (fixed seed, so the grid is reproducible),
  * histograms of width, height and file size per class,
  * a composition figure (class balance, file format, colour mode),
  * a small table of extra statistics (aspect ratio, bytes per pixel, ...) used to interpret the figures.

Usage (from the repository root):
    python src/eda.py --audit reports/dataset_audit.csv --root data/raw --out reports/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window needed (we run inside WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from PIL import Image, ImageOps

SEED = 42
CLASS_ORDER = ["ai_generated", "real"]
COLOURS = {"ai_generated": "#d95f02", "real": "#1b9e77"}  # same colour for the same class in every figure


def sample_grid(df: pd.DataFrame, root: Path, label: str, out_path: Path, rows: int = 4, cols: int = 4, thumb: int = 256) -> None:
    """Random (seeded) rows x cols grid of images for one class. Each title is the ORIGINAL pixel size."""
    pool = df[(df["class"] == label) & df["readable"]]
    rng = np.random.default_rng(SEED)
    picks = pool.iloc[rng.choice(len(pool), size=rows * cols, replace=False)]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.5))
    for ax, (_, row) in zip(axes.ravel(), picks.iterrows()):
        with Image.open(root / row["path"]) as img:
            # convert("RGB") because the audit found RGBA / palette / grayscale files; contain() shrinks
            # WITHOUT stretching, so we see the true aspect ratio of each image.
            thumbnail = ImageOps.contain(img.convert("RGB"), (thumb, thumb))
        ax.imshow(thumbnail)
        ax.set_title(f"{row['width']}x{row['height']}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Random samples - class '{label}' (seed {SEED}, titles = original width x height)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def histograms(df: pd.DataFrame, out_path: Path) -> None:
    """2 rows (classes) x 3 columns (width, height, file size). Log-spaced bins: values span ~2 orders of magnitude."""
    ok = df[df["readable"]].copy()
    ok["size_kb"] = ok["size_bytes"] / 1024
    metrics = [("width", "Width (pixels, log scale)"), ("height", "Height (pixels, log scale)"),
               ("size_kb", "File size (KB, log scale)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex="col")
    for col_idx, (col, xlabel) in enumerate(metrics):
        bins = np.logspace(np.log10(ok[col].min()), np.log10(ok[col].max()), 30)  # same bins for both classes
        for row_idx, label in enumerate(CLASS_ORDER):
            ax = axes[row_idx, col_idx]
            values = ok.loc[ok["class"] == label, col]
            ax.hist(values, bins=bins, color=COLOURS[label], edgecolor="white")
            ax.axvline(values.median(), color="black", linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())  # minor-tick labels overlapped each other on the log axis
            ax.set_title(f"{label} - median {values.median():,.0f}", fontsize=9)
            ax.set_ylabel("images")
            if row_idx == 1:
                ax.set_xlabel(xlabel)
    fig.suptitle("Image size distributions per class (dashed line = median)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def composition(df: pd.DataFrame, out_path: Path) -> None:
    """Class balance, real file format and colour mode, per class."""
    ok = df[df["readable"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    counts = df["class"].value_counts().reindex(CLASS_ORDER)
    axes[0].bar(counts.index, counts.values, color=[COLOURS[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v} ({v / len(df):.1%})", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Images per class")
    for ax, col, title in ((axes[1], "format", "File format (detected from content, not extension)"),
                           (axes[2], "mode", "Colour mode")):
        table = pd.crosstab(ok[col], ok["class"]).reindex(columns=CLASS_ORDER)
        table.plot.bar(ax=ax, color=[COLOURS[c] for c in table.columns], rot=0)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def extra_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Numbers used when interpreting the figures (and for spotting shortcuts a model could learn)."""
    ok = df[df["readable"]].copy()
    ok["aspect"] = ok["width"] / ok["height"]
    ok["megapixels"] = ok["width"] * ok["height"] / 1e6
    ok["bytes_per_pixel"] = ok["size_bytes"] / (ok["width"] * ok["height"])
    grouped = ok.groupby("class")
    return pd.DataFrame({
        "images": grouped.size(),
        "median_megapixels": grouped["megapixels"].median(),
        "median_aspect_w_over_h": grouped["aspect"].median(),
        "pct_landscape(>1.1)": grouped["aspect"].apply(lambda s: (s > 1.1).mean() * 100),
        "pct_square(0.9-1.1)": grouped["aspect"].apply(lambda s: s.between(0.9, 1.1).mean() * 100),
        "pct_portrait(<0.9)": grouped["aspect"].apply(lambda s: (s < 0.9).mean() * 100),
        "pct_width_exactly_1024": grouped["width"].apply(lambda s: (s == 1024).mean() * 100),
        "pct_png": grouped["format"].apply(lambda s: (s == "PNG").mean() * 100),
        "median_bytes_per_pixel": grouped["bytes_per_pixel"].median(),
        "pct_under_256px_side": grouped.apply(lambda g: ((g["width"] < 256) | (g["height"] < 256)).mean() * 100),
    }).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", type=Path, default=Path("reports/dataset_audit.csv"))
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.audit)
    args.out.mkdir(parents=True, exist_ok=True)
    for label in CLASS_ORDER:
        sample_grid(df, args.root, label, args.out / f"eda_samples_{label}.png")
    histograms(df, args.out / "eda_histograms.png")
    composition(df, args.out / "eda_composition.png")

    stats = extra_statistics(df)
    print(stats.T.to_string())
    stats.to_csv(args.out.parent / "eda_extra_statistics.csv")
    print("\nFigures written to", args.out)
    for path in sorted(args.out.glob("eda_*.png")):
        print(" ", path.name, f"{path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
