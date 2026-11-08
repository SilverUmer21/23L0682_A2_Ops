"""Task 2.1 evidence: one augmented training batch, and the same images before / after augmentation.

Usage (from the repository root, after `source scripts/gpu_env.sh`):
    python src/make_aug_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visionguard.config import CLASS_NAMES, DATA_ROOT, SPLITS_DIR, set_seeds
from visionguard.data import build_augmentation, make_dataset


def show(ax, image, title):
    ax.imshow(np.clip(image / 255.0, 0.0, 1.0))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_seeds()

    # 1) One batch exactly as the model will see it during training (shuffled + augmented)
    train = make_dataset(SPLITS_DIR / "train.csv", DATA_ROOT, training=True, augment=True)
    images, labels = next(iter(train))
    fig, axes = plt.subplots(4, 8, figsize=(14.5, 8))
    for ax, img, y in zip(axes.ravel(), images.numpy(), labels.numpy().ravel()):
        show(ax, img, CLASS_NAMES[int(y)])
    fig.suptitle(f"One augmented training batch: {images.shape[0]} images, {images.shape[1]}x{images.shape[2]} px", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "pipeline_augmented_batch.png", dpi=110)
    plt.close(fig)

    # 2) Same 8 validation images before and after augmentation (4 AI-generated + 4 real)
    val = make_dataset(SPLITS_DIR / "val.csv", DATA_ROOT, batch_size=146, training=False, augment=False)
    x, y = next(iter(val))
    y = y.numpy().ravel()
    picked = np.concatenate([np.where(y == 1)[0][:4], np.where(y == 0)[0][:4]])
    original = x.numpy()[picked]
    augmented = build_augmentation()(original, training=True).numpy()
    fig, axes = plt.subplots(2, 8, figsize=(14.5, 4.4))
    for col, idx in enumerate(picked):
        show(axes[0, col], original[col], f"original ({CLASS_NAMES[int(y[idx])]})")
        show(axes[1, col], augmented[col], "augmented")
    fig.suptitle("Before (top) and after (bottom) random augmentation - the label never changes", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "pipeline_before_after.png", dpi=110)
    plt.close(fig)
    print("wrote", args.out / "pipeline_augmented_batch.png", "and", args.out / "pipeline_before_after.png")


if __name__ == "__main__":
    main()
