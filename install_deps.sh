#!/bin/bash

# assume in docker

# gcc version should >= 13
apt install gcc g++ cmake ninja-build wget curl vim python3 python3-dev mpich libopenblas-dev tmux tree git net-tools gpg

# assume you can use huggingface or modelscope to download models
# If you use modelscope, and store models to spec dir
# export VLLM_USE_MODELSCOPE=True
# export MODELSCOPE_CACHE="/modelscope/hub"

python3 -m venv .venv
source .venv/bin/activate

MPICC=mpicc MPILD=mpicc python -m pip install git+https://github.com/mpi4py/mpi4py

git clone https://github.com/kzwrime/vllm-xcpu-plugin.git
git clone https://github.com/kzwrime/vllm.git
git clone https://github.com/kzwrime/torch_xcpu.git
git clone https://github.com/kzwrime/torch_mpi_ext.git

pip install -r vllm/requirements/build.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r vllm/requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install setuptools_scm
pip install modelscope

# modify if needed
cp vllm/scripts/env_template.sh vllm/scripts/env.sh

pip install clangd==18.1.8.1 clang-format==18.1.3

