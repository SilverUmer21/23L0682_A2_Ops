# syntax=docker/dockerfile:1
# One Dockerfile, three stages:
#   base - Python + dependencies + our code, runs as a non-root user      (shared)
#   api  - base + the champion model BUILT IN, serves the JSON model API  (deployed to the cloud, Task 7)
#   web  - base, serves the web UI; the model is MOUNTED at run time      (default target, used by Compose)
#
#   docker build --target web -t visionguard-web .
#   docker build --target api -t visionguard-api .     (needs models/champion.keras)

FROM python:3.13-slim AS base
# slim = Debian without compilers or docs (~120 MB) instead of the full image (~1 GB)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONPATH=/app/src
WORKDIR /app

# 1) Dependencies FIRST. Docker caches each instruction as a layer and re-runs an instruction only when its inputs
#    changed. requirements-web.txt changes rarely, so this slow layer (TensorFlow) is reused on almost every build.
COPY requirements-web.txt .
# The two rm commands are the size optimisation (Task 6.4). They run in the SAME layer as pip install, because files
# deleted in a later layer would still be stored in the earlier one. tensorflow/include (289 MB) holds C++ headers that
# are only needed to compile custom TensorFlow operations, and clang (62 MB) is a tool binding TensorFlow lists as a
# dependency but never imports when it serves a model. Neither is used at run time.
RUN pip install -r requirements-web.txt \
 && rm -rf /usr/local/lib/python3.13/site-packages/tensorflow/include \
           /usr/local/lib/python3.13/site-packages/clang

# 2) Our source code LAST. It changes all the time; because it comes after the dependency layer, editing code
#    rebuilds only these cheap layers instead of reinstalling TensorFlow.
COPY src/visionguard src/visionguard
COPY src/webapp src/webapp

# Run as an unprivileged user: if the app were ever compromised, the attacker would not be root in the container.
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000

# Docker marks the container "unhealthy" if /health stops answering with HTTP 200 (e.g. the database is down).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

FROM base AS api
# The cloud container has no folder to mount from, so the model file is built into the image.
COPY --chown=appuser:appuser models/champion.keras /models/champion.keras
ENV MODEL_PATH=/models/champion.keras
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "webapp.api_app:create_api_app()"]

FROM base AS web
# gunicorn instead of Flask's built-in server: the built-in one is for development only. One worker, because each
# worker would load its own copy of the model into memory; four threads handle concurrent requests.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "webapp:create_app()"]
