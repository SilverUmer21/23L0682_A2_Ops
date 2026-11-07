"""Project-wide constants, so training, evaluation and the web app all use the same values."""
from __future__ import annotations

from pathlib import Path

SEED = 42
IMG_SIZE = 224          # MobileNetV2's native input size; the baseline CNN uses the same size for a fair comparison
BATCH_SIZE = 32
CLASS_NAMES = ["real", "ai_generated"]   # index == label: 0 = real, 1 = ai_generated (the "positive" class)

DATA_ROOT = Path("data/raw")
SPLITS_DIR = Path("data/splits")


def set_seeds(seed: int = SEED) -> None:
    """Fix Python, NumPy and TensorFlow random seeds (Task 2.4: reproducibility).

    Caveat: this makes runs repeatable in practice, but some GPU kernels are non-deterministic,
    so two runs can still differ in the last decimals.
    """
    import keras

    keras.utils.set_random_seed(seed)
