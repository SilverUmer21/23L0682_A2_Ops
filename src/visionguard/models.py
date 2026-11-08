"""Model definitions: baseline CNN (Task 2.2) and MobileNetV2 transfer learning (Task 2.3)."""
from __future__ import annotations

import keras
from keras import layers

from visionguard.config import IMG_SIZE

# MobileNetV2 is made of 16 "inverted residual" blocks. Block 16 is the last one; in the Keras implementation its
# first layer is called block_16_expand and everything after it (block 16, the final 1x1 conv Conv_1, out_relu) is
# what we unfreeze in the fine-tuning stage.
LAST_BLOCK_FIRST_LAYER = "block_16_expand"


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


def build_transfer_model(
    img_size: int = IMG_SIZE, dropout: float = 0.3, weights: str | None = "imagenet"
) -> tuple[keras.Model, keras.Model]:
    """MobileNetV2 (ImageNet weights, no classifier) + our own head. Returns (model, backbone). Backbone starts FROZEN.

    Preprocessing: MobileNetV2 was trained on pixels scaled to [-1, 1] (x / 127.5 - 1), which is exactly what
    keras.applications.mobilenet_v2.preprocess_input does. Feeding 0..255 or 0..1 instead would shift every
    activation away from what the pretrained filters expect and waste the pretraining.
    """
    backbone = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights
    )
    backbone.trainable = False
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_v2_preprocess")(inputs)
    # training=False keeps BatchNorm layers in inference mode (uses ImageNet statistics, not our tiny batches),
    # even after we unfreeze part of the backbone. This is the standard Keras fine-tuning recipe.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai_generated")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer"), backbone


def unfreeze_last_block(backbone: keras.Model) -> None:
    """Stage 2: make only the last MobileNetV2 block (block_16 onward) trainable; everything earlier stays frozen.

    The model must be compiled again after this call, otherwise Keras keeps the old trainable set.
    """
    names = [layer.name for layer in backbone.layers]
    first_open = names.index(LAST_BLOCK_FIRST_LAYER)
    backbone.trainable = True
    for layer in backbone.layers[:first_open]:
        layer.trainable = False


def unfreeze_last_layers(backbone: keras.Model, n: int) -> None:
    """Make the last `n` backbone layers trainable and keep everything before them frozen (n = 0 keeps all frozen).

    Used by the Keras Tuner to search "how many layers to unfreeze". Compile the model again afterwards.
    """
    backbone.trainable = n > 0
    if n > 0:
        for layer in backbone.layers[:-n]:
            layer.trainable = False


def trainable_layers(model: keras.Model) -> list[str]:
    """Names of layers that currently have trainable weights (backbone layers are written 'backbone/layer')."""
    names: list[str] = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):  # the nested backbone
            names += [f"{layer.name}/{inner.name}" for inner in layer.layers if inner.trainable and inner.trainable_weights]
        elif layer.trainable and layer.trainable_weights:
            names.append(layer.name)
    return names


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
