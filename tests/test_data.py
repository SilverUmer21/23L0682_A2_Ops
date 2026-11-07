"""Tests for the input pipeline. They use tiny synthetic images, so they need no dataset (they also run in CI)."""
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image

from visionguard.data import build_augmentation, decode_and_resize, make_dataset

IMG = 64  # small size keeps the tests fast


@pytest.fixture()
def tiny_split(tmp_path):
    """12 non-square images (mixed PNG/JPEG, plus one grayscale and one RGBA) and a split CSV."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        width, height = 60 + 13 * i, 100 - 4 * i  # deliberately not square
        array = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        if i == 3:
            image = image.convert("L")      # grayscale
        if i == 4:
            image = image.convert("RGBA")   # has an alpha channel
        name = f"img{i}.png" if i % 2 == 0 or i == 4 else f"img{i}.jpg"
        image.save(tmp_path / name)
        rows.append({"path": name, "label": i % 2})
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, tmp_path


def first_batch(ds):
    images, labels = next(iter(ds))
    return images.numpy(), labels.numpy()


def test_shapes_dtypes_and_value_range(tiny_split):
    csv_path, root = tiny_split
    ds = make_dataset(csv_path, root, img_size=IMG, batch_size=12)
    images, labels = first_batch(ds)
    assert images.shape == (12, IMG, IMG, 3)          # grayscale and RGBA files became 3 channels
    assert images.dtype == np.float32
    assert 0.0 <= images.min() and images.max() <= 255.0
    assert labels.shape == (12, 1) and labels.dtype == np.float32   # (batch, 1), not (batch,)


def test_evaluation_pipeline_is_deterministic_and_never_augments(tiny_split):
    csv_path, root = tiny_split
    plain = make_dataset(csv_path, root, img_size=IMG, batch_size=12, training=False, augment=False)
    asked_to_augment = make_dataset(csv_path, root, img_size=IMG, batch_size=12, training=False, augment=True)
    a, _ = first_batch(plain)
    b, _ = first_batch(plain)           # second pass over the same dataset
    c, _ = first_batch(asked_to_augment)
    np.testing.assert_array_equal(a, b)  # no shuffling, no randomness
    np.testing.assert_array_equal(a, c)  # augment=True is ignored unless training=True


def test_augmentation_layer_changes_pixels():
    x = tf.random.uniform((8, IMG, IMG, 3), 0, 255)
    augmented = build_augmentation(seed=1)(x, training=True).numpy()
    assert augmented.shape == (8, IMG, IMG, 3)
    assert not np.allclose(augmented, x.numpy())


def test_training_pipeline_keeps_every_image(tiny_split):
    csv_path, root = tiny_split
    ds = make_dataset(csv_path, root, img_size=IMG, batch_size=5, training=True, augment=True)
    total = sum(int(images.shape[0]) for images, _ in ds)
    assert total == 12


def test_pad_mode_adds_black_bars_but_stretch_does_not(tiny_split):
    _, root = tiny_split
    wide_path = str(root / "img0.png")   # 60 wide x 100 tall -> portrait, so pad adds bars left/right
    padded = decode_and_resize(tf.constant(wide_path), IMG, "pad").numpy()
    stretched = decode_and_resize(tf.constant(wide_path), IMG, "stretch").numpy()
    assert padded[:, 0, :].max() == 0 and padded[:, -1, :].max() == 0   # black columns at the edges
    assert stretched[:, 0, :].max() > 0                                  # image content reaches the edge


def test_unknown_resize_mode_is_rejected(tiny_split):
    _, root = tiny_split
    with pytest.raises(ValueError):
        decode_and_resize(tf.constant(str(root / "img0.png")), IMG, "crop")
