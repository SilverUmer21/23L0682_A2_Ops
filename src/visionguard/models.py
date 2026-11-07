"""Model definitions (Task 2.2 baseline CNN; the transfer-learning model is added in Task 2.3)."""
from __future__ import annotations

import keras
from keras import layers

from visionguard.config import IMG_SIZE


def conv_block(x, filters: int, name: str):
    """Conv -> BatchNorm -> ReLU -> MaxPool: the repeating unit of the baseline CNN."""
    # use_bias=False because BatchNorm adds its own learnable shift (beta), so a conv bias would be redundant
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def build_baseline_cnn(
    img_size: int = IMG_SIZE, filters: tuple[int, ...] = (32, 64, 128, 128), dropout: float = 0.3
) -> keras.Model:
    """Small CNN trained from scratch: 4 conv blocks, global average pooling, dropout, one sigmoid unit.

    Input : float pixels 0..255, shape (img_size, img_size, 3)
    Output: probability that the image is AI-generated, shape (1,)
    """
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)   # normalisation lives in the model, so serving needs no extra code
    for i, n_filters in enumerate(filters, start=1):
        x = conv_block(x, n_filters, f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)           # 14x14x128 -> 128 numbers; far fewer weights than Flatten
    x = layers.Dropout(dropout, name="dropout")(x)             # randomly silence 30% of features while training: fights overfitting
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai_generated")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Binary cross-entropy + Adam. Metrics are all computed at the default 0.5 threshold except AUC."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
