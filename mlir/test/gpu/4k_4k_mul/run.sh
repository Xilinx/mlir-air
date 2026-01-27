#===- run.sh ---------------------------------------*- C++
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===------------------------------------------------------------------===//

~/public_realease_mlir_air/mlir-air/install/bin/air-opt air_sync.mlir -air-to-rocdl  -o mul_gpu_.mlir
~/public_realease_mlir_air/mlir-air/install/bin/air-opt mul_gpu_.mlir -air-gpu-outlining  -o mul_gpu11_outline.mlir
~/public_realease_mlir_air/mlir-air/llvm/build/bin/mlir-opt "--pass-pipeline=builtin.module(func.func(lower-affine, convert-linalg-to-loops,convert-scf-to-cf), gpu-kernel-outlining)"  mul_gpu11_outline.mlir -o mul_gpu_outline.llvm
~/public_realease_mlir_air/mlir-air/llvm/build/bin/mlir-opt "--pass-pipeline=builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary, func.func(gpu-async-region),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" mul_gpu_outline.llvm -o mul_gpu_.llvm

~/public_realease_mlir_air/mlir-air/llvm/build/bin/mlir-runner --debug-only=serialize-to-isa --entry-point-result=void --shared-libs=/root/schowdha/mlir-air_latest/llvm/install/lib/libmlir_rocm_runtime.so  mul_gpu_.llvm  > out

cat out
