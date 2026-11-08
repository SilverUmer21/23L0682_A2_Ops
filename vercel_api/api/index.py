"""VisionGuard model API for Vercel (Task 7): classifies one uploaded image.

    POST /predict   multipart form field "image", header X-API-Key   ->  JSON verdict
    GET  /health                                                      ->  JSON status

TensorFlow does not fit in a serverless function, so the champion Keras model was exported to TensorFlow Lite
(model.tflite, see src/export_tflite.py) and runs with the small ai-edge-litert runtime. The JSON answer has the
same fields as the Docker version of the API, so the web app can call either one.
"""
from __future__ import annotations

import hmac
import io
import os
import threading
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
THRESHOLD = float(os.environ.get("THRESHOLD", "0.6362"))      # chosen on validation data
MODEL_NAME = os.environ.get("MODEL_NAME", "23L0682_A2_Ops")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "2")
API_KEY = os.environ.get("API_KEY", "")                        # set in Vercel's project settings, never in code
MAX_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "4.5")) * 1024 * 1024)   # Vercel itself rejects bodies above ~4.5 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")           # first bytes of JPEG and PNG files

interpreter = Interpreter(model_path=str(ROOT / "model.tflite"))   # loaded once per cold start, reused by later requests
interpreter.allocate_tensors()
INPUT = interpreter.get_input_details()[0]
OUTPUT = interpreter.get_output_details()[0]
SIZE = int(INPUT["shape"][1])
lock = threading.Lock()   # a TensorFlow Lite interpreter handles one image at a time

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


def preprocess(raw: bytes) -> np.ndarray:
    """Decode, drop any alpha channel, resize to the model's input size (the same stretch resize as in training)."""
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        resized = image.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None]          # shape (1, SIZE, SIZE, 3), pixel values 0..255


def predict_probability(raw: bytes) -> float:
    batch = preprocess(raw)
    with lock:
        interpreter.set_tensor(INPUT["index"], batch)
        interpreter.invoke()
        return float(interpreter.get_tensor(OUTPUT["index"])[0][0])


@app.get("/health")
def health():
    return jsonify({"status": "ok", "mode": "tflite-on-vercel", "model_name": MODEL_NAME, "model_version": MODEL_VERSION,
                    "threshold": THRESHOLD, "input_size": SIZE})


@app.post("/predict")
def predict():
    if not API_KEY or not hmac.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):   # constant-time compare
        return jsonify({"error": "missing or wrong API key"}), 401
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "No file was sent in the form field 'image'."}), 400
    raw = upload.read()
    if not raw:
        return jsonify({"error": "The file is empty."}), 400
    if Path(upload.filename).suffix.lower() not in ALLOWED_EXTENSIONS or not raw.startswith(SIGNATURES):
        return jsonify({"error": "Only JPEG and PNG images are accepted. This file is a different type."}), 415
    try:
        probability = predict_probability(raw)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        return jsonify({"error": "This file looks like an image but cannot be decoded. It may be damaged."}), 422
    label = "ai_generated" if probability >= THRESHOLD else "real"
    return jsonify({"label": label, "confidence": probability if label == "ai_generated" else 1.0 - probability,
                    "prob_ai_generated": probability, "threshold": THRESHOLD,
                    "model_name": MODEL_NAME, "model_version": MODEL_VERSION})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"image larger than {MAX_BYTES / 1024 / 1024:g} MB"}), 413
