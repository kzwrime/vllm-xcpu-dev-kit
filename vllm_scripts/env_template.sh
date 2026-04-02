export OMP_NUM_THREADS=4
export VLLM_CPU_OMP_THREADS_BIND=nobind
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

export USER_VLLM_LOGGING_LEVEL="DEBUG"

export VLLM_USE_MODELSCOPE=True
# export MODELSCOPE_CACHE="/modelscope/hub"

# Set LD_PRELOAD for CPU backend (TCMalloc and Intel OpenMP)
# Find libiomp5.so based on Python location
# _TC_PATH="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
# _PYTHON_BIN_DIR="$(dirname "$(which python)")"
# _IOMP_PATH="${_PYTHON_BIN_DIR}/../lib/libiomp5.so"
# export LD_PRELOAD="${_TC_PATH}:${_IOMP_PATH}:${LD_PRELOAD}"
