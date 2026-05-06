//===- sym_atomic_syncscope.mlir - cross-XGMI atomic preservation --------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// The symmetric-heap producer/consumer test relies on a contract that
// `llvm.atomicrmw release` and `llvm.load atomic acquire` ops emitted with
// NO syncscope qualifier survive the GPU compilation pipeline as LLVM
// "system" syncscope (= cross-device on AMDGPU). Without that, the
// producer's release-store on rank 0's GPU is not seen by the consumer's
// acquire-load on rank 1's GPU, and the consumer hangs forever (test
// times out — appears as "no crash, no signal, just dead").
//
// AMDGPU's LLVM backend rejects an explicit `syncscope("system")` keyword
// (it recognizes "agent", "workgroup", "wavefront", "one-as", etc., but
// not "system" by name). Default = LLVM IR's System scope, which AMDGPU
// LangRef defines as cross-device:
//   https://llvm.org/docs/AMDGPUUsage.html#memory-model
//
// This test asserts that after `convert-gpu-to-rocdl` the atomic ops
// retain their ordering and continue to have NO syncscope qualifier.
//
//===-----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt --pass-pipeline='builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts))' %s | FileCheck %s

// CHECK-LABEL: gpu.module @kernels
// CHECK-LABEL: llvm.func @atomic_kernel
// CHECK:       llvm.atomicrmw xchg %{{.*}}, %{{.*}} release : !llvm.ptr, i32
// CHECK-NOT:   syncscope
// CHECK:       llvm.load %{{.*}} atomic acquire {{.*}} : !llvm.ptr -> i32
// CHECK-NOT:   syncscope
gpu.module @kernels {
  gpu.func @atomic_kernel(%ptr : !llvm.ptr, %v : i32) kernel {
    %old = llvm.atomicrmw xchg %ptr, %v release : !llvm.ptr, i32
    %loaded = llvm.load %ptr atomic acquire {alignment = 4 : i64} : !llvm.ptr -> i32
    gpu.return
  }
}
