"""Task 2.6: show misclassified TEST images of a trained model, to form hypotheses about why it fails.

Picks the 5 most confident mistakes of each kind:
  * a REAL image called AI-generated (false accusation)
  * an AI-generated image called real (missed fake)
and draws them in a 2 x 5 grid. The test set was already evaluated once; this only looks at that model's mistakes.

Usage (from the repository root, after activating the environment):
    python src/error_analysis.py --run mobilenetv2_img160
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps

from visionguard.config import DATA_ROOT, SPLITS_DIR
from visionguard.data import make_dataset
from visionguard.evaluate import predict_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="folder name under runs/ (holds best.keras and evaluation.json)")
    parser.add_argument("--per-kind", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    run_dir = Path("runs") / args.run
    threshold = json.loads((run_dir / "evaluation.json").read_text())["threshold"]
    resize_mode = json.loads((run_dir / "summary.json").read_text())["args"]["resize_mode"]
    model = keras.models.load_model(run_dir / "best.keras")
    img_size = int(model.input_shape[1])

    test = pd.read_csv(SPLITS_DIR / "test.csv")
    audit = pd.read_csv("reports/dataset_audit.csv")[["path", "width", "height", "format", "size_bytes"]]
    probs = predict_probabilities(model, make_dataset(SPLITS_DIR / "test.csv", DATA_ROOT, img_size=img_size,
                                                      resize_mode=resize_mode))
    test["prob_ai"] = probs
    test["predicted"] = (probs >= threshold).astype(int)
    test = test.merge(audit, on="path", how="left")
    test["confidence_in_wrong_label"] = test["prob_ai"].where(test["predicted"] == 1, 1 - test["prob_ai"])

    wrong = test[test["predicted"] != test["label"]]
    print(f"{args.run}: threshold {threshold:.3f} | {len(wrong)} mistakes out of {len(test)} test images "
          f"({(wrong['label'] == 0).sum()} real called AI, {(wrong['label'] == 1).sum()} AI called real)")
    real_called_ai = wrong[wrong["label"] == 0].nlargest(args.per_kind, "confidence_in_wrong_label")
    ai_called_real = wrong[wrong["label"] == 1].nlargest(args.per_kind, "confidence_in_wrong_label")
    picked = pd.concat([real_called_ai, ai_called_real]).reset_index(drop=True)

    fig, axes = plt.subplots(2, args.per_kind, figsize=(3.2 * args.per_kind, 7.4), squeeze=False)
    for row, (title, frame) in enumerate((("REAL image called AI-generated", real_called_ai),
                                          ("AI-generated image called REAL", ai_called_real))):
        for col in range(args.per_kind):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(frame):
                continue
            item = frame.iloc[col]
            with Image.open(Path(DATA_ROOT) / item["path"]) as img:
                ax.imshow(ImageOps.contain(img.convert("RGB"), (360, 360)))
            ax.set_title(f"#{row * args.per_kind + col + 1}  P(AI)={item['prob_ai']:.2f}\n"
                         f"{int(item['width'])}x{int(item['height'])} {item['format']}", fontsize=9)
        axes[row, 0].text(0, 1.28, title, transform=axes[row, 0].transAxes, fontsize=12, fontweight="bold")
    fig.suptitle(f"Most confident mistakes on the test set - {args.run} (threshold {threshold:.2f})", fontsize=13)
    fig.tight_layout()
    args.out.mkdir(exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    fig.savefig(args.out / "figures" / "error_analysis.png", dpi=105)
    plt.close(fig)

    picked.insert(0, "figure_number", range(1, len(picked) + 1))
    picked["kind"] = picked["label"].map({0: "real called AI", 1: "AI called real"})
    picked[["figure_number", "kind", "path", "prob_ai", "width", "height", "format", "size_bytes"]].to_csv(
        args.out / "error_analysis.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(picked[["figure_number", "kind", "path", "prob_ai", "width", "height", "format"]].to_string(index=False))

    # context for hypotheses: are the mistakes unusual in size or format compared with all test images?
    print("\nmedian width x height  | all test images:", int(test["width"].median()), "x", int(test["height"].median()),
          "| these 10 mistakes:", int(picked["width"].median()), "x", int(picked["height"].median()))
    print("share of images under 400 px wide | all test:", f"{(test['width'] < 400).mean():.0%}",
          "| the 10 mistakes:", f"{(picked['width'] < 400).mean():.0%}")
    print("share PNG | all test:", f"{(test['format'] == 'PNG').mean():.0%}",
          "| the 10 mistakes:", f"{(picked['format'] == 'PNG').mean():.0%}")


if __name__ == "__main__":
    main()
