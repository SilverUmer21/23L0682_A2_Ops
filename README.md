# 23L0682_A2_Ops - VisionGuard

[![CI](https://github.com/SilverUmer21/23L0682_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/SilverUmer21/23L0682_A2_Ops/actions/workflows/ci.yml)

VisionGuard tells whether an uploaded image is **AI-generated** or **real**.

Course assignment (MLOps, FAST NUCES): TensorFlow/Keras models, MLflow tracking on DagsHub,
a Flask web app with a request-history database, Docker Compose, GitHub Actions CI and a
cloud-hosted model API.

- Dataset: [AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) (Kaggle)
- Data versioning: DVC with DagsHub storage (project name `23L0682_A2_Ops`)

## Setup for training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
dvc pull            # downloads data/raw from DagsHub storage
```

## Run the web app with Docker

```bash
cp .env.example .env          # then set POSTGRES_PASSWORD in .env
mkdir -p models               # put the champion model at models/champion.keras
docker compose up -d --build
```

Open http://localhost:8000. The `web` service waits for the `db` (PostgreSQL) service to be healthy; the database
publishes no port to the host. The model file is mounted into the container from `./models` (read-only).

## Tests

```bash
pip install -r requirements-dev.txt
ruff check src tests && pytest -q
```
