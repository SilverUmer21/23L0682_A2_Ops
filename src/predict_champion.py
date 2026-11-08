"""Task 3.6: load the champion model from the MLflow model registry BY ALIAS and classify one image.

Usage (from the repository root):
    python src/predict_champion.py path/to/image.jpg
    python src/predict_champion.py path/to/image.jpg --alias champion --model-name 23L0682_A2_Ops

The registry login comes from .env (DagsHub). This script is a demonstration; the web app never calls DagsHub per
request. It loads one model file at start-up.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import keras

from visionguard import tracking
from visionguard.inference import classify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-name", default="23L0682_A2_Ops")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--threshold", type=float, default=None, help="default: the threshold stored on the model version")
    args = parser.parse_args()

    mlflow = tracking.connect(None, "23L0682_A2_Ops")
    # resolve the alias to a version in the registry, then use that version's file (downloaded once into ./models)
    model_file, version, origin = tracking.fetch_champion(mlflow, args.model_name, args.alias)
    print(f"registry: {args.model_name}@{args.alias} -> version {version.version} (from run {version.run_id[:8]})")
    print(f"model file: {model_file} ({origin})")
    model = keras.models.load_model(model_file)
    threshold = args.threshold if args.threshold is not None else float(version.tags.get("selected_threshold", 0.5))
    resize_mode = version.tags.get("resize_mode", "stretch")

    result = classify(model, args.image.read_bytes(), threshold, resize_mode)
    print(f"model input: {model.input_shape[1]}x{model.input_shape[2]} px | resize mode: {resize_mode}")
    print(f"{args.image.name}: {result['label']} | confidence {result['confidence']:.1%} | "
          f"P(ai_generated) = {result['prob_ai_generated']:.3f} | threshold {result['threshold']:.3f}")


if __name__ == "__main__":
    main()
