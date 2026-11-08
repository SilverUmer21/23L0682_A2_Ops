"""Tests for the model definitions (no dataset and no internet needed: ImageNet weights are not downloaded)."""
import numpy as np

from visionguard.models import (
    build_baseline_cnn,
    build_transfer_model,
    compile_model,
    trainable_layers,
    unfreeze_last_block,
)


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


def test_transfer_model_stage1_trains_only_the_new_head():
    model, _ = build_transfer_model(img_size=96, weights=None)
    assert trainable_layers(model) == ["p_ai_generated"]   # backbone fully frozen


def test_unfreeze_last_block_opens_only_block_16_onward():
    model, backbone = build_transfer_model(img_size=96, weights=None)
    unfreeze_last_block(backbone)
    opened = [name.split("/")[1] for name in trainable_layers(model) if "/" in name]
    assert opened[0] == "block_16_expand"
    assert all(name.startswith(("block_16", "Conv_1")) for name in opened)   # nothing from blocks 1-15
    assert "p_ai_generated" in trainable_layers(model)                        # head still trainable


def test_transfer_model_predicts_one_probability_per_image():
    model, _ = build_transfer_model(img_size=96, weights=None)
    x = np.random.default_rng(1).uniform(0, 255, (2, 96, 96, 3)).astype("float32")
    p = model.predict(x, verbose=0)
    assert p.shape == (2, 1) and ((p >= 0) & (p <= 1)).all()
