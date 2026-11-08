"""The ONE preprocessing function used in training and in serving (Task 4.1).

It lives in its own small module (no pandas, no training code) so the web image stays light, and so training and
serving can never drift apart: both call exactly this function.
"""
from __future__ import annotations

import tensorflow as tf

from visionguard.config import IMG_SIZE


def decode_bytes_and_resize(raw: tf.Tensor, img_size: int = IMG_SIZE, resize_mode: str = "stretch") -> tf.Tensor:
    """Encoded image bytes -> uint8 tensor (img_size, img_size, 3).

    decode_image(channels=3) turns grayscale, palette and RGBA files into plain RGB (alpha is dropped).
    It looks at the file CONTENT, not the extension, which matters because 20 files have the wrong extension.

    resize_mode:
      "stretch" - squash/stretch to a square. Aspect ratio is NOT preserved (median image is 1.5:1, so objects look
                  ~1.5x narrower), but no artificial borders are added.
      "pad"     - keep aspect ratio and fill the rest with black bars. Risky here: portrait images are twice as common
                  among real images (19% vs 8%), so black bars could become a shortcut that reveals the class.
    antialias=True low-pass filters before shrinking, which avoids jagged aliasing for big photos (up to 8,495 px).
    """
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
