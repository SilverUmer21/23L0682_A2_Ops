"""Training entry point for VisionGuard (Tasks 2.2 - 2.5 and 3.3 - 3.4).

Usage (from the repository root, after activating the environment):
    python src/train.py --model baseline    --name baseline_cnn
    python src/train.py --model mobilenetv2 --name mobilenetv2_ft
    python src/train.py --model baseline --name baseline_manual --mlflow manual     # log to MLflow by hand (callback)
    python src/train.py --model baseline --name baseline_auto   --mlflow autolog    # log with mlflow.tensorflow.autolog()

mobilenetv2 trains in two stages: (1) frozen backbone, new head only; (2) last block unfrozen, much smaller learning rate.
The TEST set is used only by the final evaluation step (--evaluate, or any --mlflow run): once per trained model.
"""
from __future__ import annotations

import argparse
import json
import shutil
from contextlib import nullcontext
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visionguard import tracking
from visionguard.config import (
    BATCH_SIZE,
    DATA_ROOT,
    IMG_SIZE,
    SEED,
    SPLITS_DIR,
    set_seeds,
)
from visionguard.data import make_dataset, read_split
from visionguard.evaluate import (
    choose_threshold,
    metrics_at_threshold,
    plot_confusion_matrix,
    plot_roc,
    predict_probabilities,
)
from visionguard.models import (
    build_baseline_cnn,
    build_transfer_model,
    compile_model,
    trainable_layers,
    unfreeze_last_block,
)

EXPERIMENT_NAME = "23L0682_A2_Ops"


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


def run_stage(model, name, train, val, epochs, lr, out_dir, patience, initial_epoch=0, extra_callbacks=()) -> dict:
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
        *extra_callbacks,
    ]
    print(f"\n===== stage '{name}' | learning rate {lr} | up to {epochs} epochs =====")
    # initial_epoch makes epoch numbers continue across stages (stage 2 starts after stage 1), so logged curves line up
    history = model.fit(train, validation_data=val, epochs=initial_epoch + epochs, initial_epoch=initial_epoch,
                        callbacks=callbacks, verbose=2).history
    model.load_weights(str(checkpoint))  # make sure the model object holds the best epoch's weights
    best_epoch = int(np.argmin(history["val_loss"])) + 1   # counted inside this stage, starting at 1
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


def evaluate_final(model, args, common: dict, out_dir: Path) -> dict:
    """Choose the threshold on VALIDATION data, then measure the TEST set once at that threshold."""
    val_ds = make_dataset(SPLITS_DIR / "val.csv", DATA_ROOT, **common)
    test_ds = make_dataset(SPLITS_DIR / "test.csv", DATA_ROOT, **common)
    _, y_val = read_split(SPLITS_DIR / "val.csv", DATA_ROOT)
    _, y_test = read_split(SPLITS_DIR / "test.csv", DATA_ROOT)
    p_val = predict_probabilities(model, val_ds)
    threshold = choose_threshold(y_val, p_val, beta=args.beta)
    p_test = predict_probabilities(model, test_ds)
    result = {
        "threshold": threshold, "threshold_rule": f"maximise F-beta (beta={args.beta}) on validation data",
        "validation": metrics_at_threshold(y_val, p_val, threshold),
        "test": metrics_at_threshold(y_test, p_test, threshold),
        "test_at_threshold_0.5": metrics_at_threshold(y_test, p_test, 0.5),
    }
    plot_confusion_matrix(y_test, p_test, threshold, out_dir / "confusion_matrix.png", f"{args.name} (test set)")
    plot_roc(y_test, p_test, threshold, out_dir / "roc_curve.png", f"{args.name} (test set)")
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2))
    t = result["test"]
    print(f"\nthreshold {threshold:.3f} (chosen on validation) | TEST: accuracy {t['accuracy']:.3f} precision "
          f"{t['precision']:.3f} recall {t['recall']:.3f} F1 {t['f1']:.3f} ROC-AUC {t['roc_auc']:.3f} "
          f"| real called AI: {t['fp']}, AI called real: {t['fn']}")
    return result


def log_to_mlflow(mlflow, args, out_dir, evaluation, summary, counts, model, autolog: bool) -> None:
    """Parameters, tags, final metrics and artifacts. (Per-epoch metrics come from the callback or from autolog.)"""
    params = {k: v for k, v in vars(args).items() if k not in {"mlflow", "tracking_uri", "name"}}
    params.update(counts)
    tracking.log_params_safely(mlflow, {k: str(v) for k, v in params.items()})
    mlflow.set_tags(tracking.run_tags(args.seed))
    val, test, final = evaluation["validation"], evaluation["test"], summary
    mlflow.log_metrics({
        "selected_threshold": evaluation["threshold"],
        "final_val_accuracy": val["accuracy"], "final_val_precision": val["precision"],
        "final_val_recall": val["recall"], "final_val_f1": val["f1"], "final_val_roc_auc": val["roc_auc"],
        "test_accuracy": test["accuracy"], "test_precision": test["precision"], "test_recall": test["recall"],
        "test_f1": test["f1"], "test_roc_auc": test["roc_auc"],
        "test_real_called_ai": test["fp"], "test_ai_called_real": test["fn"],
        "best_val_loss": final["best_val_loss"], "best_epoch_in_final_stage": final["best_epoch"],
    })
    mlflow.set_tag("final_model_stage", final["stage"])
    for pattern in ("curves_*.png", "history_*.json", "trainable_layers_*.txt", "summary.json", "evaluation.json",
                    "confusion_matrix.png", "roc_curve.png"):
        for path in sorted(out_dir.glob(pattern)):
            mlflow.log_artifact(str(path))
    mlflow.log_artifacts(str(SPLITS_DIR), artifact_path="split_lists")                        # train/val/test CSVs
    mlflow.log_artifact(str(tracking.write_frozen_requirements(out_dir / "requirements_frozen.txt")))   # package list
    if not autolog:  # autolog already saved the model itself; manual runs save it explicitly
        sample = np.zeros((2, args.img_size, args.img_size, 3), dtype="float32")
        signature = mlflow.models.infer_signature(sample, model.predict(sample, verbose=0))
        mlflow.keras.log_model(model, name="model", signature=signature)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["baseline", "mobilenetv2"], required=True)
    p.add_argument("--name", required=True, help="run name; outputs go to runs/<name>/")
    p.add_argument("--epochs", type=int, default=60, help="max epochs for the first (or only) stage")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune-epochs", type=int, default=40, help="max epochs for stage 2; 0 = keep the backbone frozen")
    p.add_argument("--finetune-lr", type=float, default=1e-5, help="much smaller than --lr, as the task requires")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--resize-mode", choices=["stretch", "pad"], default="stretch")
    p.add_argument("--no-augment", action="store_true", help="turn augmentation off (used for one experiment)")
    p.add_argument("--patience", type=int, default=6, help="early-stopping patience in epochs")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--evaluate", action="store_true", help="choose the threshold on validation, measure TEST once")
    p.add_argument("--beta", type=float, default=0.5, help="F-beta used to choose the threshold (<1 favours precision)")
    p.add_argument("--mlflow", choices=["off", "manual", "autolog"], default="off")
    p.add_argument("--tracking-uri", default=None, help="default: MLFLOW_TRACKING_URI from .env (DagsHub)")
    p.add_argument("--experiment", default=EXPERIMENT_NAME)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tracked = args.mlflow != "off"
    two_stages = args.model == "mobilenetv2" and args.finetune_epochs > 0
    if args.mlflow == "autolog" and two_stages:
        raise SystemExit("autolog supports one fit() per run; use --model baseline or --finetune-epochs 0")
    set_seeds(args.seed)
    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    common = {"img_size": args.img_size, "batch_size": args.batch_size, "resize_mode": args.resize_mode, "seed": args.seed}
    train = make_dataset(SPLITS_DIR / "train.csv", DATA_ROOT, training=True, augment=not args.no_augment, **common)
    val = make_dataset(SPLITS_DIR / "val.csv", DATA_ROOT, **common)   # validation: no shuffling, no augmentation
    counts = {f"n_{s}": len(read_split(SPLITS_DIR / f"{s}.csv", DATA_ROOT)[1]) for s in ("train", "val", "test")}

    mlflow = tracking.connect(args.tracking_uri, args.experiment) if tracked else None
    if args.mlflow == "autolog":
        mlflow.tensorflow.autolog(log_datasets=False, log_input_examples=False, log_model_signatures=True)
    logger = [tracking.EpochLogger(mlflow)] if args.mlflow == "manual" else []

    with mlflow.start_run(run_name=args.name) if tracked else nullcontext():
        results = []
        if args.model == "baseline":
            model = build_baseline_cnn(args.img_size, dropout=args.dropout)
            results.append(run_stage(model, "baseline", train, val, args.epochs, args.lr, out_dir, args.patience,
                                     extra_callbacks=logger))
        else:
            model, backbone = build_transfer_model(args.img_size, args.dropout)
            (out_dir / "trainable_layers_stage1.txt").write_text("\n".join(trainable_layers(model)) + "\n")
            results.append(run_stage(model, "stage1_head", train, val, args.epochs, args.lr, out_dir, args.patience,
                                     extra_callbacks=logger))
            if two_stages:
                unfreeze_last_block(backbone)
                (out_dir / "trainable_layers_stage2.txt").write_text("\n".join(trainable_layers(model)) + "\n")
                print("\nlayers with trainable weights in stage 2:", *trainable_layers(model), sep="\n  ")
                results.append(run_stage(model, "stage2_finetune", train, val, args.finetune_epochs, args.finetune_lr,
                                         out_dir, args.patience, initial_epoch=results[0]["epochs_run"],
                                         extra_callbacks=logger))

        best = min(results, key=lambda r: r["best_val_loss"])   # final model = stage with the lowest VALIDATION loss
        shutil.copy(best["checkpoint"], out_dir / "best.keras")
        summary = {"args": vars(args), "stages": results, "final_model_from_stage": best["stage"],
                   "best_epoch_of_final_stage": best["best_epoch"]}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nfinal model = stage '{best['stage']}', best epoch {best['best_epoch']} -> {out_dir / 'best.keras'}")

        if args.evaluate or tracked:
            final_model = keras.models.load_model(out_dir / "best.keras")
            evaluation = evaluate_final(final_model, args, common, out_dir)
            if tracked:
                log_to_mlflow(mlflow, args, out_dir, evaluation, best, counts, final_model, args.mlflow == "autolog")
                print("MLflow run:", mlflow.active_run().info.run_id, "| tracking server:", mlflow.get_tracking_uri())


if __name__ == "__main__":
    main()
