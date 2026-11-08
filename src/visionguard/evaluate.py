"""Evaluation helpers (Task 2.5): metrics, a decision threshold chosen on VALIDATION data, and the two standard plots.

Convention: label 1 = ai_generated (the "positive" class). An image is called AI-generated when probability >= threshold.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from visionguard.config import CLASS_NAMES


def predict_probabilities(model, dataset) -> np.ndarray:
    """P(ai_generated) for every image, in dataset order (evaluation datasets are never shuffled)."""
    return np.asarray(model.predict(dataset, verbose=0)).ravel()


def choose_threshold(y_true, probs, beta: float = 0.5) -> float:
    """Threshold that maximises the F-beta score on the given (validation) data.

    beta < 1 weights PRECISION more than recall: with beta = 0.5, precision counts twice as much as recall. We use it
    because calling a real photo "AI-generated" (a false accusation) is treated as the worse error for this product.
    Among equally good thresholds the one closest to 0.5 is returned.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    p, r = precision[:-1], recall[:-1]            # one entry per threshold
    denominator = beta**2 * p + r
    f_beta = np.divide((1 + beta**2) * p * r, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    ties = thresholds[np.isclose(f_beta, f_beta.max())]
    return float(ties[np.argmin(np.abs(ties - 0.5))])


def metrics_at_threshold(y_true, probs, threshold: float) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC and the four confusion-matrix counts at one threshold."""
    y_true = np.asarray(y_true).astype(int)
    predicted = (np.asarray(probs) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),   # threshold-free
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion_matrix(y_true, probs, threshold: float, path: Path, title: str) -> None:
    predicted = (np.asarray(probs) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"{title}\nthreshold {threshold:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_roc(y_true, probs, threshold: float, path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    point = metrics_at_threshold(y_true, probs, threshold)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {point['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.scatter([point["fp"] / max(point["fp"] + point["tn"], 1)], [point["recall"]], color="red", zorder=3,
               label=f"threshold {threshold:.2f}")
    ax.set_xlabel("false positive rate (real photo called AI-generated)")
    ax.set_ylabel("true positive rate (AI-generated caught)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
