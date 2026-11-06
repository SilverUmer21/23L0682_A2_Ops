# 23L0682_A2_Ops - VisionGuard

VisionGuard tells whether an uploaded image is **AI-generated** or **real**.

Course assignment (MLOps, FAST NUCES): TensorFlow/Keras models, MLflow tracking on DagsHub,
a Flask web app with a request-history database, Docker Compose, GitHub Actions CI and a
cloud-hosted model API.

- Dataset: [AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) (Kaggle)
- Data and model versioning: DVC + DagsHub (project name `23L0682_A2_Ops`)

> Work in progress. Setup notes and run instructions will be added as each task is completed.

## Local environment (WSL2 Debian, GPU)

```bash
source scripts/gpu_env.sh   # activates .venv and exposes the pip-installed CUDA libraries to TensorFlow
```
