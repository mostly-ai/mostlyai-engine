#!/bin/bash

# build bitsandbytes from source for MI200s (gfx90a) / MI300 (gfx942) GPUs
git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled_multi_backend
uv pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx90a;gfx942" -S .
make
uv pip install .
cd ..

# build vllm from source for MI200s (gfx90a) / MI300 (gfx942) GPUs
git clone https://github.com/vllm-project/vllm.git
source .venv/bin/activate
uv pip install /opt/rocm/share/amd_smi
uv pip install setuptools_scm numba
cd vllm
uv pip install -r requirements-rocm.txt
PYTORCH_ROCM_ARCH="gfx90a;gfx942" python setup.py develop
cd ..

# set environment variables for performance optimization
# ref: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#performance-environment-variables
export HIP_FORCE_DEV_KERNARG=1
export NCCL_MIN_NCHANNELS=112
export TORCH_BLAS_PREFER_HIPBLASLT=1
