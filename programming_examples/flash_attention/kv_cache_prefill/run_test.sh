#!/bin/bash
set -e

# Source XRT first
source /opt/xilinx/xrt/setup.sh 2>/dev/null || true

# Activate sandbox venv (after XRT to preserve PATH)
source /home/strixminipc/new_session/mlir-air/sandbox/bin/activate

# Set paths - build/bin first so we get the C++ aircc binary
export PATH=/home/strixminipc/new_session/mlir-air/build/bin:/home/strixminipc/new_session/mlir-air/mlir-aie/install/bin:$PATH
export PYTHONPATH=/home/strixminipc/new_session/mlir-air/build/python:/home/strixminipc/new_session/mlir-air/mlir-aie/install/python:$PYTHONPATH
export LD_LIBRARY_PATH=/home/strixminipc/new_session/mlir-air/build/lib:/home/strixminipc/new_session/mlir-air/mlir-aie/install/lib:$LD_LIBRARY_PATH

# Peano compiler (llvm-aie) installed as pip package in sandbox
export PEANO_INSTALL_DIR=/home/strixminipc/new_session/mlir-air/sandbox/lib/python3.13/site-packages/llvm-aie

cd /home/strixminipc/new_session/mlir-air/programming_examples/flash_attention/kv_cache_prefill/build_peano

exec python3 ../attn_npu2.py "$@"
