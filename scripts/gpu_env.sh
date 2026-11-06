# Source this file (do not execute it):   source scripts/gpu_env.sh
# Activates the project venv and lets TensorFlow find the pip-installed CUDA libraries.
# Why: TensorFlow 2.21's pip wheel does not locate the nvidia-* pip libraries by itself in this
# WSL setup, so without LD_LIBRARY_PATH it silently falls back to CPU ("Skipping registering GPU devices").
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$_here/../.venv/bin/activate"
_sp="$(python -c 'import site; print(site.getsitepackages()[0])')"
_cuda_libs="$(ls -d "$_sp"/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${_cuda_libs}/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _here _sp _cuda_libs
