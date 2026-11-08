"""Where the verdict comes from: a model loaded in this process, or a remote model API (Task 8)."""
from __future__ import annotations

import requests

from webapp.config import Settings


class InferenceError(Exception):
    """The image could not be classified (undecodable image, or the remote API failed)."""


class LocalBackend:
    """Loads the Keras model ONCE, when the app starts, and reuses it for every request."""

    source = "local-model"

    def __init__(self, settings: Settings):
        import keras  # imported here so that remote-only deployments do not need TensorFlow at all

        self.settings = settings
        self.model = keras.models.load_model(settings.model_path)   # loading per request would take seconds each time

    def classify(self, raw: bytes, filename: str = "image") -> dict:
        import tensorflow as tf

        from visionguard.inference import classify

        try:
            result = classify(self.model, raw, self.settings.threshold)
        except tf.errors.InvalidArgumentError as exc:   # TensorFlow's decoder rejected the bytes
            raise InferenceError("the image could not be decoded") from exc
        return {**result, "model_name": self.settings.model_name, "model_version": self.settings.model_version}

    def describe(self) -> dict:
        return {"mode": self.source, "model_loaded": True, "input_size": int(self.model.input_shape[1])}


class RemoteBackend:
    """Sends the image to the model API deployed in the cloud and returns its JSON answer."""

    source = "remote-api"

    def __init__(self, settings: Settings):
        self.settings = settings

    def classify(self, raw: bytes, filename: str = "image") -> dict:
        try:
            response = requests.post(
                f"{self.settings.inference_url}/predict",
                files={"image": (filename, raw)},
                headers={"X-API-Key": self.settings.inference_api_key},
                timeout=60,
            )
        except requests.RequestException as exc:
            raise InferenceError(f"the model API could not be reached ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise InferenceError(f"the model API answered HTTP {response.status_code}")
        return response.json()

    def describe(self) -> dict:
        return {"mode": self.source, "model_loaded": False, "inference_url": self.settings.inference_url}


def build_backend(settings: Settings):
    return RemoteBackend(settings) if settings.inference_url else LocalBackend(settings)
