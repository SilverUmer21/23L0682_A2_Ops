"""All web-app settings come from environment variables (Task 4.4). Nothing secret is written in code."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str = "models/champion.keras"       # the Keras file the app loads once at start-up
    threshold: float = 0.5                          # verdict threshold, chosen on validation data
    database_url: str = "sqlite:///visionguard.db"  # SQLite for local runs, PostgreSQL in Docker Compose
    max_upload_mb: float = 5.0                      # larger uploads are rejected with HTTP 413
    model_name: str = "23L0682_A2_Ops"
    model_version: str = "2"
    inference_url: str = ""                         # set => send images to this remote model API instead of a local model
    inference_api_key: str = ""                     # secret key for the remote model API
    api_key: str = ""                               # the model API (api_app.py) requires this key from its callers

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            model_path=env.get("MODEL_PATH", cls.model_path),
            threshold=float(env.get("THRESHOLD", cls.threshold)),
            database_url=env.get("DATABASE_URL", cls.database_url),
            max_upload_mb=float(env.get("MAX_UPLOAD_MB", cls.max_upload_mb)),
            model_name=env.get("MODEL_NAME", cls.model_name),
            model_version=env.get("MODEL_VERSION", cls.model_version),
            inference_url=env.get("INFERENCE_URL", "").rstrip("/"),
            inference_api_key=env.get("INFERENCE_API_KEY", ""),
            api_key=env.get("API_KEY", ""),
        )
