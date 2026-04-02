#!/bin/bash

# assume in docker

# gcc version should >= 13

apt update
apt install -y \
    gcc g++ cmake ninja-build \
    python3 python3-dev python3.12-venv \
    libopenblas-dev libnuma-dev \
    git ssh openssh-server gpg \
    wget curl net-tools \
    vim tmux tree \
    zip unzip xz-utils
# apt install mpich
apt install -y openmpi-bin libopenmpi-dev
apt-get install -y --no-install-recommends libtcmalloc-minimal4

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
pip install modelscope ruff mypy
pip install clangd==18.1.8.1 clang-format==18.1.3

python -m pip cache remove mpi4py
MPICC=mpicc MPILD=mpicc python -m pip install --no-binary=mpi4py mpi4py

# try to run
# mpirun -np 4 python3 tests/test_mpi4py.py

# modify if needed
cp vllm/scripts/env_template.sh vllm/scripts/env.sh

# !!! warning "set `LD_PRELOAD`"
#     Before use vLLM CPU installed via wheels, make sure TCMalloc and Intel OpenMP are installed and added to `LD_PRELOAD`:
#     ```bash
#     # install TCMalloc, Intel OpenMP is installed with vLLM CPU
#     sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

#     # manually find the path
#     sudo find / -iname *libtcmalloc_minimal.so.4
#     sudo find / -iname *libiomp5.so
#     TC_PATH=...
#     IOMP_PATH=...

#     # add them to LD_PRELOAD
#     export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
#     ```
#
# e.g. find .venv -iname *libiomp5.so
#      .venv/lib/libiomp5.so
# modify if needed
TC_PATH=/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
IOMP_PATH=/shared/vllm_dev_v2/.venv/lib/libiomp5.so

cat >> vllm/scripts/env.sh << EOF

# LD_PRELOAD for TCMalloc and Intel OpenMP
export TC_PATH=$TC_PATH
export IOMP_PATH=$IOMP_PATH
export LD_PRELOAD="\$TC_PATH:\$IOMP_PATH:\$LD_PRELOAD"
EOF

