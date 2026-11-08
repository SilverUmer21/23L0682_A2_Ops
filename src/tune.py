"""Task 2.7 (apply part): Keras Tuner search for the VisionGuard transfer model.

Searches three settings with random search (10 trials by default):
  * learning_rate   - 1e-5 ... 1e-3 (log scale)
  * dropout         - 0.2 ... 0.6
  * unfrozen_layers - how many of the last MobileNetV2 layers are trainable: 0, 10, 30 or 60
The score of a trial is its best validation ROC-AUC. Only TRAIN and VALIDATION data are used (never the test set).

Usage (from the repository root, after activating the environment):
    python src/tune.py --trials 10
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import keras
import keras_tuner as kt
import pandas as pd

from visionguard.config import BATCH_SIZE, DATA_ROOT, SEED, SPLITS_DIR, set_seeds
from visionguard.data import make_dataset
from visionguard.models import build_transfer_model, compile_model, unfreeze_last_layers

IMG_SIZE = 160   # the size of the current champion; also faster than 224


def build_model(hp: kt.HyperParameters) -> keras.Model:
    # Free the previous trial's model and graph first: without this, memory grows by about 1 GB per trial.
    keras.backend.clear_session()
    gc.collect()
    dropout = hp.Float("dropout", 0.2, 0.6, step=0.1)
    learning_rate = hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
    unfrozen_layers = hp.Choice("unfrozen_layers", [0, 10, 30, 60])
    model, backbone = build_transfer_model(IMG_SIZE, dropout)
    unfreeze_last_layers(backbone, unfrozen_layers)
    return compile_model(model, learning_rate)


def results_table(tuner: kt.RandomSearch) -> pd.DataFrame:
    rows = []
    for trial in tuner.oracle.trials.values():
        row = {"trial": trial.trial_id, **trial.hyperparameters.values, "val_auc": trial.score}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("val_auc", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12, help="max epochs per trial (early stopping may end it sooner)")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--resume", action="store_true", help="continue an interrupted search instead of starting over")
    args = parser.parse_args()
    set_seeds()

    common = {"img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "seed": SEED}
    train = make_dataset(SPLITS_DIR / "train.csv", DATA_ROOT, training=True, augment=True, **common)
    val = make_dataset(SPLITS_DIR / "val.csv", DATA_ROOT, **common)

    tuner = kt.RandomSearch(
        build_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=args.trials,
        seed=SEED,
        directory="runs/tuner",
        project_name="visionguard",
        overwrite=not args.resume,
    )
    tuner.search(train, validation_data=val, epochs=args.epochs, verbose=2,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])

    table = results_table(tuner)
    args.out.mkdir(exist_ok=True)
    table.to_csv(args.out / "tuner_results.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== all trials, best first ===")
    print(table.round(4).to_string(index=False))
    print("\n=== best settings ===")
    print(tuner.get_best_hyperparameters(1)[0].values)

    print("\n=== which setting mattered most? (mean validation AUC per value; spread = max - min of those means) ===")
    for name in ("unfrozen_layers", "dropout", "learning_rate"):
        if name == "learning_rate":   # continuous: compare the lower and the upper half of the tried values
            half = table[name].median()
            groups = table.groupby(table[name] > half)["val_auc"].mean().rename({False: f"lr <= {half:.1e}", True: f"lr > {half:.1e}"})
        else:
            groups = table.groupby(name)["val_auc"].mean()
        print(f"{name:<16} spread {groups.max() - groups.min():.4f} | " + ", ".join(f"{k}: {v:.3f}" for k, v in groups.items()))
    print(f"\nscore range of the {len(table)} trials: {table['val_auc'].min():.3f} - {table['val_auc'].max():.3f}")


if __name__ == "__main__":
    main()
