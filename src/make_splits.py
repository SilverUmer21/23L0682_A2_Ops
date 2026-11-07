"""Reproducible train / validation / test split lists for VisionGuard (needed by Task 2.1).

Steps
  1. Read the audit table (reports/dataset_audit.csv) and keep only readable images.
  2. Drop the EXTRA copy of every exact duplicate, so the same picture can never sit in train and test.
  3. STRATIFIED random split (70 / 15 / 15) with a fixed seed: every split keeps the same class ratio.
  4. Write one CSV per split plus a summary, and prove there is no overlap between the splits.

Label convention: 1 = ai_generated (the "positive" class the product must catch), 0 = real.

Usage (from the repository root):
    python src/make_splits.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABELS = {"real": 0, "ai_generated": 1}
COLUMNS = ["path", "label", "class", "sha256"]  # path is relative to data/raw


def make_splits(audit: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = SEED):
    """Return (train, val, test) DataFrames. Fractions are of the de-duplicated, readable images."""
    df = audit[audit["readable"] & ~audit["is_extra_copy"]].copy()
    df["label"] = df["class"].map(LABELS)
    # First cut off the test set, then split the rest into train/val.
    # val_frac is rescaled because it is now a fraction of the smaller remaining pool.
    train_val, test = train_test_split(df, test_size=test_frac, stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_frac / (1 - test_frac), stratify=train_val["label"], random_state=seed
    )
    return tuple(part[COLUMNS].sort_values("path").reset_index(drop=True) for part in (train, val, test))


def check_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if any file, or any identical picture (same hash), appears in two splits."""
    for (name_a, a), (name_b, b) in [(("train", train), ("val", val)), (("train", train), ("test", test)),
                                      (("val", val), ("test", test))]:
        assert not set(a["path"]) & set(b["path"]), f"same file in {name_a} and {name_b}"
        assert not set(a["sha256"]) & set(b["sha256"]), f"identical image in {name_a} and {name_b}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", type=Path, default=Path("reports/dataset_audit.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/splits"))
    args = parser.parse_args()

    audit = pd.read_csv(args.audit)
    train, val, test = make_splits(audit)
    check_no_leakage(train, val, test)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "label_convention": {"1": "ai_generated", "0": "real"},
               "dropped_duplicate_copies": int(audit["is_extra_copy"].sum())}
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(args.out / f"{name}.csv", index=False)
        counts = part["class"].value_counts()
        summary[name] = {"images": len(part), "ai_generated": int(counts.get("ai_generated", 0)),
                         "real": int(counts.get("real", 0)),
                         "pct_ai_generated": round(100 * counts.get("ai_generated", 0) / len(part), 1)}
        print(f"{name:<5} {len(part):>4} images | ai_generated {summary[name]['ai_generated']:>3} "
              f"({summary[name]['pct_ai_generated']}%) | real {summary[name]['real']:>3}")
    (args.out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"dropped duplicate copies: {summary['dropped_duplicate_copies']} | no file or hash overlap between splits")
    print("wrote", ", ".join(str(args.out / f"{n}.csv") for n in ("train", "val", "test")))


if __name__ == "__main__":
    main()
