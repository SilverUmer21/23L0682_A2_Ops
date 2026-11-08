"""Upload validation shared by the web UI and the model API (Task 4.2).

Three kinds of bad upload are rejected with a friendly message:
  1. wrong file type  - extension AND the first bytes of the file must say JPEG or PNG (names can lie)
  2. too large        - handled by Flask's MAX_CONTENT_LENGTH (HTTP 413)
  3. looks like an image but cannot be decoded - e.g. a truncated or damaged JPEG
File names are sanitised and the image is never written to disk: it only lives in memory for the request.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from flask import Request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")   # first bytes of JPEG and PNG files


class UploadError(Exception):
    """A rejected upload: carries the HTTP status and a message that is safe to show to the user."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class InMemoryRequest(Request):
    """Werkzeug normally spools uploads above 500 KB into a temporary FILE on disk. Keep everything in memory instead,
    so an uploaded image never touches the disk."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()


def check_upload(file_storage) -> tuple[bytes, str]:
    """Return (image bytes, sanitised file name) or raise UploadError."""
    if file_storage is None or not file_storage.filename:
        raise UploadError(400, "No file was selected. Choose a JPEG or PNG image and try again.")
    name = secure_filename(file_storage.filename) or "upload"   # removes paths such as ../../etc/passwd
    raw = file_storage.read()
    if not raw:
        raise UploadError(400, "The file is empty.")
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS or not raw.startswith(SIGNATURES):
        raise UploadError(415, "Only JPEG and PNG images are accepted. This file is a different type.")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()   # forces a full decode; open() alone only reads the header
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise UploadError(422, "This file looks like an image but cannot be decoded. It may be damaged.") from exc
    return raw, name


def thumbnail_data_uri(raw: bytes, size: int = 320) -> str:
    """Small preview embedded in the result page. Built in memory, not saved anywhere."""
    with Image.open(io.BytesIO(raw)) as image:
        preview = image.convert("RGB")
        preview.thumbnail((size, size))
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
