"""VisionGuard web app (Tasks 4 and 5): upload an image, get a verdict, keep a permanent history."""
from __future__ import annotations

import hashlib

from flask import Flask, jsonify, render_template, request

from webapp import db
from webapp.backends import InferenceError, build_backend
from webapp.config import Settings
from webapp.validation import (
    InMemoryRequest,
    UploadError,
    check_upload,
    thumbnail_data_uri,
)

PAGE_SIZE = 25


def create_app(settings: Settings | None = None, backend=None) -> Flask:
    """Application factory. `settings` and `backend` can be passed in by tests; otherwise they come from the environment."""
    settings = settings or Settings.from_env()
    app = Flask(__name__)
    app.request_class = InMemoryRequest
    app.config["MAX_CONTENT_LENGTH"] = int(settings.max_upload_mb * 1024 * 1024)
    engine = db.make_engine(settings.database_url)
    backend = backend or build_backend(settings)   # a local model is loaded here, once, at start-up

    def error_page(status: int, title: str, message: str):
        return render_template("error.html", status=status, title=title, message=message), status

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=settings.max_upload_mb)

    @app.post("/predict")
    def predict():
        try:
            raw, name = check_upload(request.files.get("image"))
        except UploadError as exc:
            return error_page(exc.status, "Upload rejected", exc.message)
        try:
            result = backend.classify(raw, name)
        except InferenceError as exc:
            status = 502 if backend.source == "remote-api" else 422
            return error_page(status, "Could not classify the image", str(exc).capitalize() + ".")
        db.log_prediction(
            engine, filename=name, label=result["label"], confidence=result["confidence"],
            prob_ai_generated=result["prob_ai_generated"], threshold=result["threshold"],
            model_name=result["model_name"], model_version=str(result["model_version"]),
            image_sha256=hashlib.sha256(raw).hexdigest(), source=backend.source,
        )
        return render_template("result.html", result=result, filename=name, preview=thumbnail_data_uri(raw))

    @app.get("/history")
    def history():
        page = max(request.args.get("page", 1, type=int), 1)
        rows = db.recent(engine, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
        count = db.total(engine)
        return render_template("history.html", rows=rows, page=page, count=count,
                               has_next=page * PAGE_SIZE < count)

    @app.get("/health")
    def health():
        database_up = db.is_reachable(engine)
        body = {"status": "ok" if database_up else "degraded", "database": "up" if database_up else "down",
                "model_name": settings.model_name, "model_version": settings.model_version,
                "threshold": settings.threshold, **backend.describe()}
        return jsonify(body), 200 if database_up else 503

    @app.errorhandler(413)
    def too_large(_):
        return error_page(413, "File too large", f"The image is larger than the {settings.max_upload_mb:g} MB limit.")

    @app.errorhandler(404)
    def not_found(_):
        return error_page(404, "Page not found", "There is nothing at this address.")

    @app.errorhandler(405)
    def wrong_method(_):
        return error_page(405, "Not allowed", "This page does not accept that kind of request.")

    @app.errorhandler(500)
    def server_error(_):
        return error_page(500, "Something went wrong", "The server hit an unexpected problem. Please try again.")

    return app
