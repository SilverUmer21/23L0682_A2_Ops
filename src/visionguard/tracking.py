"""MLflow helpers (Tasks 3.2 - 3.4): connect to the tracking server, build tags, log per-epoch metrics.

Secrets are never written in code: the DagsHub login is read from environment variables
(loaded from the git-ignored .env file by python-dotenv).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import keras
import yaml


def connect(tracking_uri: str | None, experiment: str):
    """Point MLflow at the tracking server and select the experiment. Returns the mlflow module."""
    import mlflow
    from dotenv import load_dotenv

    load_dotenv()  # reads .env from the working directory (git-ignored)
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit("No tracking server: set MLFLOW_TRACKING_URI in .env or pass --tracking-uri")
    if "dagshub.com" in uri:
        # DagsHub uses HTTP basic auth: username + access token, which MLflow reads from these two variables
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow


def _run(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_info() -> dict:
    """Commit hash of the code that produced the run, and whether tracked files had uncommitted changes."""
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]) or "unknown",
        "git_dirty": "true" if _run(["git", "status", "--porcelain", "--untracked-files=no"]) else "false",
    }


def dvc_data_version(dvc_file: str | Path = "data/raw.dvc") -> str:
    """Hash of the dataset version: the md5 stored in the DVC pointer file (ends in .dir for a folder)."""
    try:
        return str(yaml.safe_load(Path(dvc_file).read_text())["outs"][0]["md5"])
    except (OSError, KeyError, IndexError, TypeError):
        return "unknown"


def run_tags(seed: int) -> dict:
    """Tags that make a run traceable: code version, data version and random seed."""
    return {**git_info(), "dvc_data_version": dvc_data_version(), "seed": str(seed)}


class EpochLogger(keras.callbacks.Callback):
    """The MANUAL approach: after every epoch send exactly the numbers we chose to MLflow."""

    def __init__(self, mlflow_module):
        super().__init__()
        self.mlflow = mlflow_module

    def on_epoch_end(self, epoch, logs=None):
        metrics = {name: float(value) for name, value in (logs or {}).items()}
        metrics["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        self.mlflow.log_metrics(metrics, step=epoch)


def log_params_safely(mlflow_module, params: dict) -> None:
    """Log parameters one by one. MLflow refuses to change an existing parameter, which can happen when autologging
    already stored a parameter with the same name; in that case ours is stored with a cfg_ prefix."""
    from mlflow.exceptions import MlflowException

    for key, value in params.items():
        try:
            mlflow_module.log_param(key, value)
        except MlflowException:
            mlflow_module.log_param(f"cfg_{key}", value)


def write_frozen_requirements(path: Path) -> Path:
    """`pip freeze` of the environment that produced the run (the 'package list' artifact)."""
    path.write_text(_run([sys.executable, "-m", "pip", "freeze"]) + "\n")
    return path
