//===- sym_atomic_syncscope.mlir - cross-XGMI atomic preservation --------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// The symmetric-heap producer/consumer test relies on a contract that
// `llvm.atomicrmw release` and `llvm.load atomic acquire` ops emitted with
// `syncscope("")` (= LLVM IR's System scope = cross-device on AMDGPU)
// survive the GPU compilation pipeline unchanged. Without that, the
// producer's release-store on rank 0's GPU is not seen by the consumer's
// acquire-load on rank 1's GPU, and the consumer hangs forever (test
// times out — appears as "no crash, no signal, just dead").
//
// The empty-string syncscope is LLVM IR's canonical spelling of System
// scope (LLVM's textual IR omits the `syncscope(...)` token entirely when
// scope == System; MLIR's LLVM dialect round-trips it as `syncscope("")`).
// AMDGPU's LangRef defines System as cross-device:
//   https://llvm.org/docs/AMDGPUUsage.html#memory-model
//
// This test asserts that after `convert-gpu-to-rocdl` the atomic ops
// retain their ordering and the explicit `syncscope("")` qualifier.
//
//===-----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt --pass-pipeline='builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts))' %s | FileCheck %s

// CHECK-LABEL: gpu.module @kernels
// CHECK-LABEL: llvm.func @atomic_kernel
// CHECK:       llvm.atomicrmw xchg %{{.*}}, %{{.*}} syncscope("") release : !llvm.ptr, i32
// CHECK:       llvm.load %{{.*}} atomic syncscope("") acquire {{.*}} : !llvm.ptr -> i32
gpu.module @kernels {
  gpu.func @atomic_kernel(%ptr : !llvm.ptr, %v : i32) kernel {
    %old = llvm.atomicrmw xchg %ptr, %v syncscope("") release : !llvm.ptr, i32
    %loaded = llvm.load %ptr atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
    gpu.return
  }
}
