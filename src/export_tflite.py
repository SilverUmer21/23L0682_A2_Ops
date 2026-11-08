"""Task 7 (Vercel route): export the champion Keras model to TensorFlow Lite for the serverless model API.

TensorFlow itself (about 1.5 GB installed) cannot be deployed to a serverless function, but a TensorFlow Lite
model (about 9 MB) runs with a ~22 MB runtime (ai-edge-litert). The exported model still contains the pixel
scaling layer, so it takes the same raw 0..255 pixels as the Keras model.

Usage (from the repository root, after activating the environment):
    python src/export_tflite.py --model models/champion.keras --out vercel_api/model.tflite
"""
from __future__ import annotations

import argparse
from pathlib import Path

import keras
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/champion.keras"))
    parser.add_argument("--out", type=Path, default=Path("vercel_api/model.tflite"))
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()          # float32 weights: no quantisation, so predictions stay almost identical
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(tflite_bytes)
    print(f"input shape {model.input_shape} | wrote {args.out} ({len(tflite_bytes) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
