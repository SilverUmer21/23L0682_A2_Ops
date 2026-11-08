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

## Cloud deployment of the model API (Vercel)

The live model API runs on Vercel as a serverless Flask function (`vercel_api/`). TensorFlow is too large for a
serverless function, so the champion model is exported to TensorFlow Lite and served with the small `ai-edge-litert`
runtime. On the 145 test images below 4.5 MB it gives the same accuracy as the Keras model (0.703 vs 0.697).

```bash
python src/export_tflite.py --model models/champion.keras --out vercel_api/model.tflite   # model.tflite is not committed
cd vercel_api
npx vercel link --yes
npx vercel env add API_KEY production      # the secret key callers must send in the X-API-Key header
npx vercel deploy --prod --yes
```

Endpoints: `GET /health` and `POST /predict` (form field `image`, header `X-API-Key`). Vercel rejects request bodies
above 4.5 MB, so the web app is run with `MAX_UPLOAD_MB=4` and `INFERENCE_URL` set to the Vercel URL.
