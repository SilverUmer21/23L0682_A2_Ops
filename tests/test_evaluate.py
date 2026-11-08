"""Tests for the evaluation helpers (pure NumPy/scikit-learn, no dataset needed)."""
import numpy as np

from visionguard.evaluate import (
    choose_threshold,
    metrics_at_threshold,
    plot_confusion_matrix,
    plot_roc,
)


def test_perfect_predictions_give_perfect_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = metrics_at_threshold(y, p, 0.5)
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == m["roc_auc"] == 1.0
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 0, 2)


def test_confusion_counts_add_up_to_the_number_of_images():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    m = metrics_at_threshold(y, p, 0.4)
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 200


def test_precision_weighted_threshold_is_not_lower_than_the_recall_friendly_one():
    # a real image (label 0) scores 0.6: accepting it as AI-generated needs a threshold <= 0.6
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.10, 0.20, 0.60, 0.55, 0.70, 0.80, 0.90])
    precision_first = choose_threshold(y, p, beta=0.25)
    recall_first = choose_threshold(y, p, beta=4.0)
    assert precision_first >= recall_first


def test_chosen_threshold_beats_or_matches_the_default_on_its_own_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.25 + rng.normal(0.4, 0.2, 300), 0, 1)
    thr = choose_threshold(y, p, beta=0.5)

    def f_beta(t):
        m = metrics_at_threshold(y, p, t)
        d = 0.25 * m["precision"] + m["recall"]
        return 1.25 * m["precision"] * m["recall"] / d if d else 0.0

    assert f_beta(thr) >= f_beta(0.5) - 1e-12


def test_plots_are_written(tmp_path):
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.2, 0.6, 0.7, 0.9, 0.4, 0.1])
    plot_confusion_matrix(y, p, 0.5, tmp_path / "cm.png", "test")
    plot_roc(y, p, 0.5, tmp_path / "roc.png", "test")
    assert (tmp_path / "cm.png").stat().st_size > 1000 and (tmp_path / "roc.png").stat().st_size > 1000
