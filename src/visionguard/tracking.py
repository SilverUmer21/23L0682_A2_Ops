"""MLflow helpers (Tasks 3.2 - 3.4): connect to the tracking server, build tags, log per-epoch metrics.

Secrets are never written in code: the DagsHub login is read from environment variables
(loaded from the git-ignored .env file by python-dotenv).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
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


def fetch_champion(mlflow_module, model_name: str, alias: str, cache_dir: str | Path = "models", attempts: int = 3):
    """Resolve a registry alias to a model version and keep ONE local .keras file for that version.

    The file name contains the version number, so a new champion is downloaded again but an unchanged one is reused
    from the local copy. A download goes to a temporary folder first, so a half-finished download is never mistaken
    for a complete model. Returns (path to the .keras file, ModelVersion, where the file came from). The web app
    uses the local file, so it never has to ask DagsHub while serving requests.
    """
    from mlflow.exceptions import MlflowException

    version = mlflow_module.MlflowClient().get_model_version_by_alias(model_name, alias)
    cached = Path(cache_dir) / f"{model_name}_v{version.version}.keras"
    if cached.exists() and cached.stat().st_size > 0:
        return cached, version, "local copy"
    for attempt in range(1, attempts + 1):
        scratch = tempfile.mkdtemp(prefix="champion_")
        try:
            folder = Path(mlflow_module.artifacts.download_artifacts(
                artifact_uri=f"models:/{model_name}@{alias}", dst_path=scratch))
            weights = folder / "data" / "model.keras"
            if weights.exists() and weights.stat().st_size > 0:
                cached.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(weights, cached)
                return cached, version, "downloaded from the registry"
        except (MlflowException, OSError) as exc:  # a dropped connection surfaces as one of these
            print(f"download attempt {attempt}/{attempts} failed: {type(exc).__name__}", flush=True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        time.sleep(2 * attempt)
    raise SystemExit(f"could not download {model_name}@{alias} after {attempts} attempts")


def write_frozen_requirements(path: Path) -> Path:
    """`pip freeze` of the environment that produced the run (the 'package list' artifact)."""
    path.write_text(_run([sys.executable, "-m", "pip", "freeze"]) + "\n")
    return path
