#!/bin/bash

# assume in docker

# gcc version should >= 13

apt install gcc g++ cmake ninja-build wget curl vim python3 python3-dev libopenblas-dev tmux tree git net-tools gpg net-tools libnuma-dev ssh openssh-server
# apt install mpich
apt install openmpi-bin libopenmpi-dev

echo -e "\n# Enable OpenMPI root access\nexport OMPI_ALLOW_RUN_AS_ROOT=1\nexport OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" >> ~/.bashrc && source ~/.bashrc

# assume you can use huggingface or modelscope to download models
# If you use modelscope, and store models to spec dir
# export VLLM_USE_MODELSCOPE=True
# export MODELSCOPE_CACHE="/modelscope/hub"

python3 -m venv .venv
source .venv/bin/activate

git clone https://github.com/kzwrime/vllm-xcpu-plugin.git
git clone https://github.com/kzwrime/vllm.git
git clone https://github.com/kzwrime/torch_xcpu.git
git clone https://github.com/kzwrime/torch_mpi_ext.git

pip install -r vllm/requirements/build.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r vllm/requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install setuptools_scm
pip install modelscope
pip install clangd==18.1.8.1 clang-format==18.1.3

python -m pip cache remove mpi4py
MPICC=mpicc MPILD=mpicc python -m pip install --no-binary=mpi4py mpi4py

# try to run
# mpirun -np 4 python3 tests/test_mpi4py.py

# modify if needed
cp vllm/scripts/env_template.sh vllm/scripts/env.sh
# 

