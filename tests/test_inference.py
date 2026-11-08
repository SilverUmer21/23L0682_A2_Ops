"""Tests for the shared inference helpers (synthetic image, untrained model: no dataset or network needed)."""
import io

import numpy as np
import pytest
from PIL import Image

from visionguard.inference import classify, preprocess, verdict
from visionguard.models import build_baseline_cnn


def png_bytes(width=90, height=50, mode="RGB"):
    buffer = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)).convert(mode).save(
        buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("prob, threshold, label", [(0.9, 0.5, "ai_generated"), (0.4, 0.5, "real"),
                                                    (0.6, 0.7, "real"), (0.7, 0.7, "ai_generated")])
def test_verdict_uses_the_threshold(prob, threshold, label):
    assert verdict(prob, threshold)[0] == label


def test_confidence_is_the_probability_of_the_chosen_label():
    assert verdict(0.9, 0.5)[1] == pytest.approx(0.9)
    assert verdict(0.2, 0.5)[1] == pytest.approx(0.8)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_preprocess_gives_a_float_batch_of_one_rgb_image(mode):
    x = preprocess(png_bytes(mode=mode), img_size=64)
    assert x.shape == (1, 64, 64, 3) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 255.0


def test_classify_reads_the_input_size_from_the_model():
    model = build_baseline_cnn(img_size=64)
    result = classify(model, png_bytes(), threshold=0.5)
    assert set(result) == {"label", "confidence", "prob_ai_generated", "threshold"}
    assert result["label"] in {"ai_generated", "real"} and 0.0 <= result["confidence"] <= 1.0
