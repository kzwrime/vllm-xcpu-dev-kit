#!/bin/bash

set -e

source .venv/bin/activate

PROJ_HOME=$(pwd)

cd ${PROJ_HOME}/vllm
VLLM_TARGET_DEVICE=cpu python setup.py develop
pip install pre-commit
pre-commit install


cd ${PROJ_HOME}/torch_mpi_ext
CXX=mpicxx pip install --no-build-isolation .

cd ${PROJ_HOME}/torch_xcpu
./build-all.sh
./scripts/install-hooks.sh

cd ${PROJ_HOME}/vllm-xcpu-plugin
pip install --no-build-isolation -e .
./scripts/install-hooks.sh

