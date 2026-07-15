# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Architecture-orthogonal build infrastructure shared by every LLM-on-NPU
# example: kernel compile/cache/profiling (cache.py), external C++ kernel
# compilation (external_kernels.py), backend kwarg presets (backend_presets.py),
# and the text-MLIR stitching primitives (stitching.py) that the assembly
# builders in shared.builders use to fuse sub-kernels into one ELF.
