# 23L0682_A2_Ops - VisionGuard

VisionGuard tells whether an uploaded image is **AI-generated** or **real**.

Course assignment (MLOps, FAST NUCES): TensorFlow/Keras models, MLflow tracking on DagsHub,
a Flask web app with a request-history database, Docker Compose, GitHub Actions CI and a
cloud-hosted model API.

- Dataset: [AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) (Kaggle)
- Data versioning: DVC with DagsHub storage (project name `23L0682_A2_Ops`)

> Work in progress. Run instructions will be added as each task is completed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
dvc pull            # downloads data/raw from DagsHub storage
```
