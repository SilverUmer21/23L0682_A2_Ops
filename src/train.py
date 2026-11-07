"""Training entry point for VisionGuard (Tasks 2.2, 2.3, 2.4).

Usage (from the repository root, after activating the environment):
    python src/train.py --model baseline    --name baseline_cnn
    python src/train.py --model mobilenetv2 --name mobilenetv2_ft

mobilenetv2 trains in two stages: (1) frozen backbone, new head only; (2) last block unfrozen, much smaller learning rate.
The TEST set is never used here. It is evaluated exactly once, at the very end, by src/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visionguard.config import (
    BATCH_SIZE,
    DATA_ROOT,
    IMG_SIZE,
    SEED,
    SPLITS_DIR,
    set_seeds,
)
from visionguard.data import make_dataset
from visionguard.models import (
    build_baseline_cnn,
    build_transfer_model,
    compile_model,
    trainable_layers,
    unfreeze_last_block,
)


def plot_curves(history: dict, best_epoch: int, title: str, path: Path) -> None:
    """Loss and accuracy per epoch for train and validation, with the best epoch marked."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in ((axes[0], "loss", "Loss (binary cross-entropy)"), (axes[1], "accuracy", "Accuracy")):
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.axvline(best_epoch, color="red", linestyle="--", linewidth=1, label=f"best epoch = {best_epoch}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_stage(model: keras.Model, name: str, train, val, epochs: int, lr: float, out_dir: Path, patience: int) -> dict:
    """Compile and fit one training stage with early stopping, best-model checkpointing and LR reduction."""
    compile_model(model, lr)  # compile again every stage: required after changing which layers are trainable
    checkpoint = out_dir / f"{name}_best.keras"
    callbacks = [
        # stop when validation loss has not improved for `patience` epochs
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        # keep the file of the epoch with the lowest validation loss
        keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
        # halve the learning rate when validation loss stalls for 2 epochs
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    print(f"\n===== stage '{name}' | learning rate {lr} | up to {epochs} epochs =====")
    history = model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks, verbose=2).history
    model.load_weights(str(checkpoint))  # make sure the model object holds the best epoch's weights
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    plot_curves(history, best_epoch, f"{out_dir.name} - {name}", out_dir / f"curves_{name}.png")
    (out_dir / f"history_{name}.json").write_text(json.dumps(history, default=float))
    i = best_epoch - 1
    result = {
        "stage": name, "learning_rate": lr, "epochs_run": len(history["loss"]), "best_epoch": best_epoch,
        "best_val_loss": float(history["val_loss"][i]), "val_accuracy_at_best": float(history["val_accuracy"][i]),
        "val_auc_at_best": float(history["val_auc"][i]), "train_accuracy_at_best": float(history["accuracy"][i]),
        "checkpoint": str(checkpoint),
    }
    print(f"best epoch {best_epoch}/{result['epochs_run']} | val_loss {result['best_val_loss']:.4f} "
          f"| val_acc {result['val_accuracy_at_best']:.3f} | val_auc {result['val_auc_at_best']:.3f}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["baseline", "mobilenetv2"], required=True)
    p.add_argument("--name", required=True, help="run name; outputs go to runs/<name>/")
    p.add_argument("--epochs", type=int, default=60, help="max epochs for the first (or only) stage")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune-epochs", type=int, default=40, help="max epochs for the fine-tuning stage")
    p.add_argument("--finetune-lr", type=float, default=1e-5, help="much smaller than --lr, as the task requires")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--resize-mode", choices=["stretch", "pad"], default="stretch")
    p.add_argument("--no-augment", action="store_true", help="turn augmentation off (used for one experiment)")
    p.add_argument("--patience", type=int, default=6, help="early-stopping patience in epochs")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    common = {"img_size": args.img_size, "batch_size": args.batch_size, "resize_mode": args.resize_mode, "seed": args.seed}
    train = make_dataset(SPLITS_DIR / "train.csv", DATA_ROOT, training=True, augment=not args.no_augment, **common)
    val = make_dataset(SPLITS_DIR / "val.csv", DATA_ROOT, **common)   # validation: no shuffling, no augmentation

    results = []
    if args.model == "baseline":
        model = build_baseline_cnn(args.img_size, dropout=args.dropout)
        results.append(run_stage(model, "baseline", train, val, args.epochs, args.lr, out_dir, args.patience))
    else:
        model, backbone = build_transfer_model(args.img_size, args.dropout)
        (out_dir / "trainable_layers_stage1.txt").write_text("\n".join(trainable_layers(model)) + "\n")
        results.append(run_stage(model, "stage1_head", train, val, args.epochs, args.lr, out_dir, args.patience))
        unfreeze_last_block(backbone)
        (out_dir / "trainable_layers_stage2.txt").write_text("\n".join(trainable_layers(model)) + "\n")
        print("\nlayers with trainable weights in stage 2:", *trainable_layers(model), sep="\n  ")
        results.append(run_stage(model, "stage2_finetune", train, val, args.finetune_epochs, args.finetune_lr,
                                 out_dir, args.patience))

    best = min(results, key=lambda r: r["best_val_loss"])   # final model = stage with the lowest VALIDATION loss
    shutil.copy(best["checkpoint"], out_dir / "best.keras")
    summary = {"args": vars(args), "stages": results, "final_model_from_stage": best["stage"],
               "best_epoch_of_final_stage": best["best_epoch"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nfinal model = stage '{best['stage']}', best epoch {best['best_epoch']} -> {out_dir / 'best.keras'}")


if __name__ == "__main__":
    main()
