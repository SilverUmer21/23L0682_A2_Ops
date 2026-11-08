"""Tests for the web UI and the model API. They use a tiny untrained model and SQLite in a temp folder (no dataset, no network)."""
import io

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import select

from visionguard.models import build_baseline_cnn
from webapp import create_app, db
from webapp.api_app import create_api_app
from webapp.config import Settings

MAX_MB = 0.2


@pytest.fixture(scope="module")
def model_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "tiny.keras"
    build_baseline_cnn(img_size=64).save(path)
    return str(path)


def make_settings(tmp_path, model_file, **extra):
    return Settings(model_path=model_file, threshold=0.5, database_url=f"sqlite:///{tmp_path / 'test.db'}",
                    max_upload_mb=MAX_MB, model_name="test-model", model_version="9", api_key="secret", **extra)


@pytest.fixture()
def client(tmp_path, model_file):
    settings = make_settings(tmp_path, model_file)
    app = create_app(settings)
    app.testing = True
    return app.test_client(), settings


def image_bytes(fmt="PNG", size=(80, 60)):
    buffer = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(test_client, data, name="photo.png", route="/predict", **kwargs):
    return test_client.post(route, data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data", **kwargs)


def history_rows(settings):
    engine = db.make_engine(settings.database_url)
    with engine.connect() as connection:
        return [dict(r._mapping) for r in connection.execute(select(db.predictions))]


def test_pages_load(client):
    test_client, _ = client
    assert test_client.get("/").status_code == 200
    assert test_client.get("/history").status_code == 200
    assert b"No images have been checked yet" in test_client.get("/history").data


def test_valid_upload_gives_a_verdict_and_is_logged(client):
    test_client, settings = client
    response = upload(test_client, image_bytes())
    assert response.status_code == 200
    assert b"Confidence" in response.data and b"test-model v9" in response.data
    (row,) = history_rows(settings)
    assert row["filename"] == "photo.png" and row["label"] in {"real", "ai_generated"}
    assert row["model_name"] == "test-model" and row["source"] == "local-model" and len(row["image_sha256"]) == 64
    assert b"photo.png" in test_client.get("/history").data


def test_jpeg_upload_works(client):
    test_client, _ = client
    assert upload(test_client, image_bytes("JPEG"), name="photo.jpg").status_code == 200


def test_wrong_file_type_is_rejected_by_name_and_by_content(client):
    test_client, settings = client
    assert upload(test_client, b"hello world", name="notes.txt").status_code == 415
    assert upload(test_client, b"this is text, not a picture", name="fake.jpg").status_code == 415   # name lies
    assert upload(test_client, image_bytes(), name="picture.gif").status_code == 415
    assert history_rows(settings) == []   # nothing was logged for rejected files


def test_image_that_cannot_be_decoded_is_rejected(client):
    test_client, _ = client
    damaged = image_bytes("PNG")[:200]   # valid PNG header, truncated data
    response = upload(test_client, damaged, name="damaged.png")
    assert response.status_code == 422 and b"cannot be decoded" in response.data


def test_oversized_upload_is_rejected(client):
    test_client, _ = client
    big = image_bytes("PNG", size=(900, 900))   # random noise: far larger than the 0.2 MB limit
    assert len(big) > MAX_MB * 1024 * 1024
    assert upload(test_client, big).status_code == 413


def test_missing_file_is_rejected(client):
    test_client, _ = client
    assert test_client.post("/predict", data={}, content_type="multipart/form-data").status_code == 400


def test_file_names_are_sanitised(client):
    test_client, settings = client
    upload(test_client, image_bytes(), name="../../etc/evil name.png")
    assert history_rows(settings)[0]["filename"] == "etc_evil_name.png"


def test_health_reports_database_and_model(client):
    test_client, _ = client
    body = test_client.get("/health").get_json()
    assert test_client.get("/health").status_code == 200
    assert body["status"] == "ok" and body["database"] == "up" and body["model_loaded"] is True


def test_unknown_page_gets_a_friendly_error(client):
    test_client, _ = client
    response = test_client.get("/nope")
    assert response.status_code == 404 and b"Page not found" in response.data


def test_model_api_needs_the_key(tmp_path, model_file):
    api = create_api_app(make_settings(tmp_path, model_file)).test_client()
    assert upload(api, image_bytes()).status_code == 401
    assert upload(api, image_bytes(), headers={"X-API-Key": "wrong"}).status_code == 401
    ok = upload(api, image_bytes(), headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    assert {"label", "confidence", "prob_ai_generated", "threshold", "model_name", "model_version"} <= set(ok.get_json())
    assert api.get("/health").status_code == 200


def test_model_api_rejects_bad_uploads_with_json(tmp_path, model_file):
    api = create_api_app(make_settings(tmp_path, model_file)).test_client()
    bad = upload(api, b"text", name="x.jpg", headers={"X-API-Key": "secret"})
    assert bad.status_code == 415 and "error" in bad.get_json()


def test_ui_can_use_a_remote_api_instead_of_a_local_model(tmp_path):
    class FakeRemote:
        source = "remote-api"

        def classify(self, raw, filename="image"):
            return {"label": "ai_generated", "confidence": 0.9, "prob_ai_generated": 0.9, "threshold": 0.6,
                    "model_name": "cloud-model", "model_version": "2"}

        def describe(self):
            return {"mode": "remote-api", "model_loaded": False}

    settings = Settings(database_url=f"sqlite:///{tmp_path / 'r.db'}", max_upload_mb=MAX_MB)
    test_client = create_app(settings, backend=FakeRemote()).test_client()
    assert upload(test_client, image_bytes()).status_code == 200
    (row,) = history_rows(settings)
    assert row["source"] == "remote-api" and row["model_name"] == "cloud-model"
