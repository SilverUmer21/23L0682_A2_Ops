"""The oversized-upload guard must read the whole body before the server answers (otherwise browsers show a broken connection)."""
from flask import request

from webapp import create_app
from webapp.config import Settings
from webapp.validation import reject_if_too_large


class FakeBackend:
    source = "local-model"

    def classify(self, raw, filename="image"):
        raise AssertionError("an oversized upload must be refused before the model is asked")

    def describe(self):
        return {"mode": "local-model", "model_loaded": True}


def make_app(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'u.db'}", max_upload_mb=0.001)   # limit: about 1 KB
    return create_app(settings, backend=FakeBackend())


def test_oversized_body_is_fully_read_before_answering(tmp_path):
    app = make_app(tmp_path)
    with app.test_request_context("/predict", method="POST", data=b"x" * 400_000, content_type="application/octet-stream"):
        assert reject_if_too_large(max_bytes=1000) is True
        assert request.environ["wsgi.input"].read() == b""   # nothing is left unread


def test_small_bodies_are_left_alone(tmp_path):
    app = make_app(tmp_path)
    with app.test_request_context("/predict", method="POST", data=b"x" * 500, content_type="application/octet-stream"):
        assert reject_if_too_large(max_bytes=1000) is False
        assert len(request.environ["wsgi.input"].read()) == 500   # untouched, the route can still read it


def test_oversized_post_gets_the_friendly_413_page(tmp_path):
    client = make_app(tmp_path).test_client()
    response = client.post("/predict", data=b"x" * 5000, content_type="multipart/form-data; boundary=b")
    assert response.status_code == 413 and b"File too large" in response.data
