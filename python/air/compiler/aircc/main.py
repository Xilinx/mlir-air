# ./python/air/compiler/aircc/main.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
aircc - AIR compiler driver for MLIR tools

The Python aircc driver has been retired in favor of the native C++ aircc
binary (tools/aircc/aircc.cpp). All compilation is now handled by the C++
binary, which is invoked by XRTBackend via subprocess.

This module is kept only to preserve the configure utilities imported by
other Python backends (cpu_backend, linalg_on_tensors).
"""
