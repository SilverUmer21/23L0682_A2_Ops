"""The model API (Task 7): a small JSON service that classifies an uploaded image. Deployed to the cloud.

    POST /predict   multipart form field "image", header X-API-Key   ->  JSON verdict
    GET  /health                                                      ->  JSON status

The web UI (Task 8) calls this API instead of loading a model itself.
"""
from __future__ import annotations

import hmac

from flask import Flask, jsonify, request

from webapp.backends import InferenceError, LocalBackend
from webapp.config import Settings
from webapp.validation import InMemoryRequest, UploadError, check_upload


def create_api_app(settings: Settings | None = None, backend=None) -> Flask:
    settings = settings or Settings.from_env()
    if not settings.api_key:
        raise RuntimeError("API_KEY must be set: the model API refuses to run without a secret key")
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config["MAX_CONTENT_LENGTH"] = int(settings.max_upload_mb * 1024 * 1024)
    backend = backend or LocalBackend(settings)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_name": settings.model_name, "model_version": settings.model_version,
                        "threshold": settings.threshold, **backend.describe()})

    @app.post("/predict")
    def predict():
        # constant-time comparison, so the key cannot be guessed from response timing
        if not hmac.compare_digest(request.headers.get("X-API-Key", ""), settings.api_key):
            return jsonify({"error": "missing or wrong API key"}), 401
        try:
            raw, name = check_upload(request.files.get("image"))
            return jsonify(backend.classify(raw, name))
        except UploadError as exc:
            return jsonify({"error": exc.message}), exc.status
        except InferenceError as exc:
            return jsonify({"error": str(exc)}), 422

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": f"image larger than {settings.max_upload_mb:g} MB"}), 413

    return app
