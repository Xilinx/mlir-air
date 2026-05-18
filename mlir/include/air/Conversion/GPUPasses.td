//===- GPUPasses.td ----------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_CONVERSION_GPU_PASSES
#define AIR_CONVERSION_GPU_PASSES

include "mlir/Pass/PassBase.td"

def ConvertAIRToROCDL : Pass<"air-to-rocdl", "ModuleOp"> {
  let summary = "Lower AIR dialect to ROCDL dialect";
  let constructor = "xilinx::air::createAIRToROCDLPass()";
  let description = [{
  }];
  let dependentDialects = [
    "gpu::GPUDialect", "ROCDL::ROCDLDialect", "LLVM::LLVMDialect"
  ];
  let options = [];
}

def AIRTranslateToLLVM : Pass<"air-translate-to-llvm", "ModuleOp"> {
  let summary = "Lower air.translate to memref.reinterpret_cast + LLVM-dialect address arithmetic";
  let description = [{
    Expands each `air.translate` op into the pointer-rebase computation:
    `bases[to_rank] - bases[from_rank]`, converted from bytes to elements
    of the source memref's element type, then applied as a new offset
    via `memref.reinterpret_cast`. The expansion is pure arithmetic; it
    works identically on host functions and inside `gpu.func`.
  }];
  let constructor = "xilinx::air::createAIRTranslateToLLVMPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertGPUKernelOutline : Pass<"air-gpu-outlining", "ModuleOp"> {
  let summary = "Outline GPU Kernel Func from GPU Launch";
  let constructor = "xilinx::air::createGPUKernelOutlinePass()";
  let description = [{
  }];
  let dependentDialects = [
    "gpu::GPUDialect", "ROCDL::ROCDLDialect", "LLVM::LLVMDialect"
  ];
  let options = [];
}

def AIRSymmetricAllocToMgpu : Pass<"air-symmetric-alloc-to-mgpu", "ModuleOp"> {
  let summary = "Lower memref.alloc whose result memref has #air.symmetric_heap "
                "memory_space to mgpuSymmetricAlloc and matching memref.dealloc "
                "to mgpuSymmetricFree";
  let constructor = "xilinx::air::createAIRSymmetricAllocToMgpuPass()";
  let description = [{
    Replaces each `memref.alloc` whose result type has memory_space
    `#air.symmetric_heap` with a call to `mgpuSymmetricAlloc(size_in_bytes,
    stream)` returning `!llvm.ptr`, then builds an LLVM memref descriptor
    (struct) wrapping that pointer and projects it back to the original
    memref type via `builtin.unrealized_conversion_cast` so downstream uses
    keep working.

    For every `memref.dealloc` whose operand traces back (through a single
    `unrealized_conversion_cast`) to such a symmetric alloc, the pass emits
    `mgpuSymmetricFree(ptr, stream)` and erases the dealloc.

    Should run before `convert-to-llvm`. Does nothing if no symmetric-heap
    allocations are present.
  }];
  let dependentDialects = [
    "func::FuncDialect", "arith::ArithDialect", "LLVM::LLVMDialect",
    "xilinx::air::airDialect"
  ];
}

def AIRGpuChannelToCacheline : Pass<"air-gpu-channel-to-cacheline", "ModuleOp"> {
  let summary = "Lower air.channel ops with channel_type = \"gpu_symmetric_heap\" "
                "to the cache-line atomicity pattern inside their enclosing herd "
                "bodies";
  let constructor = "xilinx::air::createAIRGpuChannelToCachelinePass()";
  let description = [{
    For each `air.channel` op with attribute `channel_type = "gpu_symmetric_heap"`,
    finds the matching `air.channel.put` and `air.channel.get` (initial scope:
    exactly one of each) and expands them into the kernel-driven cacheline
    pattern that the handwritten `test/gpu/multi_gpu/handwritten/cacheline.mlir`
    reference uses:

    * **put** -> `air.translate %src, %producer_rank, %consumer_rank, %bases`
      + cooperative `memref.store` (lanes 0..30 write payload, lane 31 writes
      the sync flag = 1).
    * **get** -> `scf.while` spin loop that loads `%dst[lane]` and broadcasts
      lane 31 via `gpu.shuffle idx ..., 31, 64` until the flag is observed.

    The pass requires:
    * Both put and get must live inside `air.herd` bodies.
    * The producer and consumer ranks must be derivable from an enclosing
      `scf.if (arith.cmpi eq %rid, %const)` rank-dispatch nest.
    * The enclosing herd must thread a `memref<?xindex, #air.symmetric_heap>`
      (the heap_bases table) through its kernel args.
    * The put's source memref and the get's destination memref must be
      `memref<32xi32, #air.symmetric_heap>` (initial-PR shape constraint;
      generalising is a follow-up).

    Erases the `air.channel` symbol after expansion.

    Should run before `air-rank-to-mgpu`, `air-translate-to-llvm`,
    `air-to-rocdl`, and `air-gpu-outlining`. After the full chain runs, the
    resulting IR is functionally equivalent to the handwritten cacheline
    reference.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::gpu::GPUDialect",
    "xilinx::air::airDialect"
  ];
}

def AIRRankToMgpu : Pass<"air-rank-to-mgpu", "ModuleOp"> {
  let summary = "Lower air.rank to mgpu* runtime calls (multi-GPU process model)";
  let constructor = "xilinx::air::createAIRRankToMgpuPass()";
  let description = [{
    Each `air.rank` op is replaced by inlining its body in place, with rank
    IDs computed from `mgpuGetRank()` (delinearized into the rank's N-D
    iteration space) and rank sizes substituted from the static size operands.

    The pass also inserts `mgpuSymmetricHeapInit(heap_size)` at the entry of
    the enclosing `func.func` (default 256 MB; configurable via the
    `heap-size` option) and `mgpuSymmetricHeapDestroy()` before each
    `func.return` in that function.

    This replaces `air-rank-to-launch` for the GPU pipeline. Unlike
    `air-rank-to-launch` (which serializes ranks via `scf.for`), this pass
    assumes each process executes the whole rank body once and runtime
    coordinates across processes via env vars (RANK / WORLD_SIZE / LOCAL_RANK)
    and the symmetric-heap fabric.
  }];
  let options = [
    Option<"heapSize", "heap-size", "uint64_t", "/*default=*/268435456",
           "Symmetric heap size in bytes (default: 256 MB)">
  ];
  let dependentDialects = [
    "func::FuncDialect", "arith::ArithDialect",
    "xilinx::air::airDialect"
  ];
}

#endif // AIR_CONVERSION_GPU_PASSES
