#!/usr/bin/env bash
# Task 1.1 evidence: environment versions and GPU visibility.   Run:  bash scripts/show_env.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/gpu_env.sh
echo "Task 1.1 environment | user: $(whoami) | $(date '+%F %T') | project: 23L0682_A2_Ops"

echo; echo "--- OS / Python / virtual environment ---"
. /etc/os-release; echo "$PRETTY_NAME | kernel $(uname -r)"
python --version
echo "python: $(which python)"

echo; echo "--- key package versions (pip list) ---"
pip list 2>/dev/null | grep -i -E "^(tensorflow|keras|keras-tuner|mlflow|dagshub|dvc|numpy|pandas|pillow|matplotlib|scikit-learn|python-dotenv|pytest|ruff) "

echo; echo "--- GPU as the NVIDIA driver sees it ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo; echo "--- GPU as TensorFlow sees it ---"
python - <<'PY' 2>/dev/null
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow", tf.__version__, "| built with CUDA:", tf.test.is_built_with_cuda())
print("Physical GPUs:", gpus)
if gpus:
    print("Device details:", tf.config.experimental.get_device_details(gpus[0]))
PY
