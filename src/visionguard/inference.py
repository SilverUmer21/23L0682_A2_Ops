"""Inference helpers shared by the registry demo script (Task 3.6), the Flask app and the model API (Task 4.1).

The image is preprocessed with the SAME function as in training (decode_bytes_and_resize), and the verdict rule is
the same one used when the threshold was chosen, so serving behaves exactly like evaluation.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from visionguard.preprocess import decode_bytes_and_resize


def verdict(prob_ai: float, threshold: float) -> tuple[str, float]:
    """Label and confidence from P(ai_generated). An image is called AI-generated when the probability >= threshold.

    The confidence is the model's probability for the label it gave. It is NOT a calibrated guarantee.
    """
    if prob_ai >= threshold:
        return "ai_generated", prob_ai
    return "real", 1.0 - prob_ai


def preprocess(raw: bytes, img_size: int, resize_mode: str = "stretch") -> np.ndarray:
    """Encoded image bytes -> float32 array of shape (1, img_size, img_size, 3) with pixel values 0..255."""
    image = decode_bytes_and_resize(tf.constant(raw), img_size, resize_mode)
    return tf.cast(image, tf.float32)[None].numpy()


def classify(model, raw: bytes, threshold: float, resize_mode: str = "stretch") -> dict:
    """Run one image through the model. The input size is read from the model itself."""
    img_size = int(model.input_shape[1])
    prob_ai = float(model.predict(preprocess(raw, img_size, resize_mode), verbose=0)[0, 0])
    label, confidence = verdict(prob_ai, threshold)
    return {"label": label, "confidence": confidence, "prob_ai_generated": prob_ai, "threshold": threshold}
