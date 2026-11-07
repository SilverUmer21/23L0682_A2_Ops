"""tf.data input pipeline (Task 2.1).

Stages, in order (each one is explained in the report):
  1. read the split list (CSV of image path + label)
  2. read the file bytes and decode to a 3-channel uint8 image   (tf.io.decode_image, channels=3)
  3. resize to IMG_SIZE x IMG_SIZE                               (see `resize_mode` for what happens to aspect ratio)
  4. cache the resized uint8 images in memory                    (decode+resize happens once, not every epoch)
  5. shuffle (training only)                                     (new order every epoch)
  6. batch
  7. random augmentation (training only, and only if asked)
  8. prefetch                                                    (prepare the next batch while the GPU works on this one)

The pipeline yields float32 pixels in the range 0..255 and labels of shape (batch, 1).
Normalisation (scaling to 0..1 or -1..1) is done INSIDE the model, so the saved model carries its own
preprocessing and the web app only has to decode + resize the same way.
"""
from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from visionguard.config import BATCH_SIZE, IMG_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE


def read_split(csv_path: str | Path, data_root: str | Path) -> tuple[list[str], np.ndarray]:
    """Read a split CSV (columns: path, label) and return absolute-ish file paths and float32 labels."""
    df = pd.read_csv(csv_path)
    paths = [str(Path(data_root) / p) for p in df["path"]]
    return paths, df["label"].to_numpy(dtype="float32")


def decode_and_resize(path: tf.Tensor, img_size: int = IMG_SIZE, resize_mode: str = "stretch") -> tf.Tensor:
    """File path -> uint8 tensor (img_size, img_size, 3).

    decode_image(channels=3) turns grayscale, palette and RGBA files into plain RGB (alpha is dropped).
    It looks at the file CONTENT, not the extension, which matters because 20 files have the wrong extension.

    resize_mode:
      "stretch" - squash/stretch to a square. Aspect ratio is NOT preserved (median image is 1.5:1, so objects look
                  ~1.5x narrower), but no artificial borders are added.
      "pad"     - keep aspect ratio and fill the rest with black bars. Risky here: portrait images are twice as common
                  among real images (19% vs 8%), so black bars could become a shortcut that reveals the class.
    antialias=True low-pass filters before shrinking, which avoids jagged aliasing for big photos (up to 8,495 px).
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    if resize_mode == "pad":
        image = tf.image.resize_with_pad(image, img_size, img_size, method="bilinear", antialias=True)
    elif resize_mode == "stretch":
        image = tf.image.resize(image, [img_size, img_size], method="bilinear", antialias=True)
    else:
        raise ValueError(f"resize_mode must be 'stretch' or 'pad', got {resize_mode!r}")
    # back to uint8 so the cache holds 150 KB per image instead of 600 KB
    return tf.cast(tf.round(tf.clip_by_value(image, 0.0, 255.0)), tf.uint8)


def build_augmentation(seed: int = SEED) -> keras.Sequential:
    """Random, label-preserving changes applied to TRAINING images only.

    Kept mild on purpose: strong colour/blur changes could erase the very cues (texture, noise) that separate
    AI-generated from real images.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=seed),   # mirror image; a flipped fake is still fake
            keras.layers.RandomRotation(0.05, seed=seed),       # +-5% of a full turn = +-18 degrees
            keras.layers.RandomZoom(0.1, seed=seed),            # zoom in/out by up to 10%
            keras.layers.RandomContrast(0.1, seed=seed),        # contrast change of up to 10%
        ],
        name="augmentation",
    )


def make_dataset(
    csv_path: str | Path,
    data_root: str | Path = "data/raw",
    *,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    training: bool = False,
    augment: bool = False,
    resize_mode: str = "stretch",
    seed: int = SEED,
    cache: bool = True,
) -> tf.data.Dataset:
    """Build the dataset for one split. Shuffling and augmentation happen ONLY when training=True."""
    paths, labels = read_split(csv_path, data_root)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: (decode_and_resize(p, img_size, resize_mode), tf.expand_dims(y, -1)),  # label shape (1,) not ()
        num_parallel_calls=AUTOTUNE,
    )
    if cache:
        ds = ds.cache()  # cache BEFORE shuffle/augment, otherwise every epoch would replay the same order/augmentation
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augment:
        augmenter = build_augmentation(seed)
        ds = ds.map(lambda x, y: (augmenter(tf.cast(x, tf.float32), training=True), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
