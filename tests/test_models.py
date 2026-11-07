"""Tests for the model definitions (no dataset needed)."""
import numpy as np

from visionguard.models import build_baseline_cnn, compile_model


def test_baseline_cnn_output_is_one_probability_per_image():
    model = compile_model(build_baseline_cnn(img_size=64))
    x = np.random.default_rng(0).uniform(0, 255, (4, 64, 64, 3)).astype("float32")
    p = model.predict(x, verbose=0)
    assert p.shape == (4, 1)
    assert ((p >= 0) & (p <= 1)).all()          # sigmoid output


def test_baseline_cnn_is_small_enough_for_a_few_hundred_images():
    model = build_baseline_cnn(img_size=64)
    assert model.count_params() < 500_000


def test_baseline_cnn_trains_and_reports_the_expected_metrics():
    model = compile_model(build_baseline_cnn(img_size=64))
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (8, 64, 64, 3)).astype("float32")
    y = (rng.uniform(size=(8, 1)) > 0.5).astype("float32")
    history = model.fit(x, y, epochs=1, batch_size=4, verbose=0)
    assert {"loss", "accuracy", "auc", "precision", "recall"} <= set(history.history)
