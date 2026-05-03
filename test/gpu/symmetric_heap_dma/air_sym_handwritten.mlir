//===- air_sym_handwritten.mlir - hand-written multi-GPU e2e test --------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Hand-written reference IR exercising the symmetric-heap multi-GPU runtime
// from MLIR. This is what the (future) air-rank-to-mgpu + cross-rank-DMA
// lowering passes should produce.
//
// Each process executes this main once. With WORLD_SIZE=2:
//   1. Init symmetric heap.
//   2. Allocate a 1024xf32 symmetric buffer.
//   3. Each rank fills its buffer with (rank + 1).0 from host.
//   4. Barrier.
//   5. Each rank reads peer's buffer via mgpuGetHeapBases()[peer]+offset,
//      copies it D2D into a local hipMalloc-style buffer, then D2H into a
//      host buffer, and verifies every element == (peer + 1).0.
//   6. Print PASS / FAIL.
//
// Launcher: run.sh forks N processes with RANK / WORLD_SIZE / LOCAL_RANK.
//
//===------------------------------------------------------------------===//

module {
  // ---- mgpu* C ABI declarations -----------------------------------------
  func.func private @mgpuSymmetricHeapInit(i64)
  func.func private @mgpuSymmetricHeapDestroy()
  func.func private @mgpuGetRank() -> i32
  func.func private @mgpuGetWorldSize() -> i32
  func.func private @mgpuSymmetricAlloc(i64, !llvm.ptr) -> !llvm.ptr
  func.func private @mgpuSymmetricFree(!llvm.ptr, !llvm.ptr)
  func.func private @mgpuGetHeapBase(i32) -> !llvm.ptr
  func.func private @mgpuGetHeapBases() -> !llvm.ptr
  func.func private @mgpuBarrier()
  func.func private @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
  func.func private @mgpuMemFree(!llvm.ptr, !llvm.ptr)
  func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

  // libc helpers
  func.func private @malloc(i64) -> !llvm.ptr
  func.func private @free(!llvm.ptr)
  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.mlir.global internal constant @msg_init("[mlir] rank %d / world %d, init OK\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass("[mlir] rank %d: cross-rank read PASS (peer=%d, expected=%.1f)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_fail("[mlir] rank %d: MISMATCH at idx=%ld got=%.1f expected=%.1f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_only1("[mlir] rank %d: world_size=1, skipping cross-rank read\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done("[mlir] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // ---- main -------------------------------------------------------------
  func.func @main() {
    // Constants
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64                    // sizeof(f32)
    %c1024_i64 = arith.constant 1024 : i64              // N
    %c4096_i64 = arith.constant 4096 : i64              // N * sizeof(f32)
    %heap_size = arith.constant 268435456 : i64         // 256 MB
    %nullptr = llvm.mlir.zero : !llvm.ptr
    %false = arith.constant false

    // Init symmetric heap (collective)
    func.call @mgpuSymmetricHeapInit(%heap_size) : (i64) -> ()
    %rank = func.call @mgpuGetRank() : () -> i32
    %world = func.call @mgpuGetWorldSize() : () -> i32

    // printf("[mlir] rank %d / world %d, init OK\n", rank, world)
    %fmt_init = llvm.mlir.addressof @msg_init : !llvm.ptr
    llvm.call @printf(%fmt_init, %rank, %world) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32

    // Symmetric alloc 1024 floats
    %buf = func.call @mgpuSymmetricAlloc(%c4096_i64, %nullptr) : (i64, !llvm.ptr) -> !llvm.ptr

    // Allocate host buffer of 1024 floats and fill with (rank + 1).0
    %hostbuf = func.call @malloc(%c4096_i64) : (i64) -> !llvm.ptr
    %rank_plus1_i32 = arith.addi %rank, %c1_i32 : i32
    %rank_plus1_f32 = arith.sitofp %rank_plus1_i32 : i32 to f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    scf.for %i = %c0 to %c1024 step %c1 {
      %i_i64 = arith.index_cast %i : index to i64
      %addr = llvm.getelementptr %hostbuf[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %rank_plus1_f32, %addr : f32, !llvm.ptr
    }

    // mgpuMemcpy(buf, hostbuf, 4096, nullptr)  // H2D
    func.call @mgpuMemcpy(%buf, %hostbuf, %c4096_i64, %nullptr) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

    // Barrier so all ranks have written before any reads
    func.call @mgpuBarrier() : () -> ()

    // If world_size > 1, read from peer = (rank + 1) % world
    %is_multi = arith.cmpi sgt, %world, %c1_i32 : i32
    scf.if %is_multi {
      %sum = arith.addi %rank, %c1_i32 : i32
      %peer = arith.remsi %sum, %world : i32

      // bases = mgpuGetHeapBases()
      %bases = func.call @mgpuGetHeapBases() : () -> !llvm.ptr

      // peer_base = bases[peer]
      %peer_i64 = arith.extsi %peer : i32 to i64
      %peer_base_addr = llvm.getelementptr %bases[%peer_i64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %peer_base = llvm.load %peer_base_addr : !llvm.ptr -> !llvm.ptr

      // local_base = bases[rank]
      %rank_i64 = arith.extsi %rank : i32 to i64
      %local_base_addr = llvm.getelementptr %bases[%rank_i64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
      %local_base = llvm.load %local_base_addr : !llvm.ptr -> !llvm.ptr

      // local_offset = (uintptr_t)buf - (uintptr_t)local_base
      %buf_int = llvm.ptrtoint %buf : !llvm.ptr to i64
      %local_base_int = llvm.ptrtoint %local_base : !llvm.ptr to i64
      %offset = arith.subi %buf_int, %local_base_int : i64

      // peer_buf = (char*)peer_base + offset
      %peer_buf = llvm.getelementptr %peer_base[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8

      // Allocate a local D2D-target buffer via mgpuMemAlloc(N*sizeof(f32))
      %local_copy = func.call @mgpuMemAlloc(%c4096_i64, %nullptr, %false) : (i64, !llvm.ptr, i1) -> !llvm.ptr

      // mgpuMemcpy(local_copy, peer_buf, 4096, nullptr)  // D2D
      func.call @mgpuMemcpy(%local_copy, %peer_buf, %c4096_i64, %nullptr) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

      // Allocate host readback and copy D2H
      %host_rb = func.call @malloc(%c4096_i64) : (i64) -> !llvm.ptr
      func.call @mgpuMemcpy(%host_rb, %local_copy, %c4096_i64, %nullptr) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

      // Verify: every element == (peer + 1).0
      %peer_plus1_i32 = arith.addi %peer, %c1_i32 : i32
      %expected = arith.sitofp %peer_plus1_i32 : i32 to f32

      %nfail_init = arith.constant 0 : i32
      %nfail = scf.for %i = %c0 to %c1024 step %c1
                      iter_args(%nfail_acc = %nfail_init) -> (i32) {
        %i_i64 = arith.index_cast %i : index to i64
        %addr = llvm.getelementptr %host_rb[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %v = llvm.load %addr : !llvm.ptr -> f32
        %ne = arith.cmpf une, %v, %expected : f32
        %new_nfail = scf.if %ne -> i32 {
          // Print first few mismatches
          %fmt_fail = llvm.mlir.addressof @msg_fail : !llvm.ptr
          %v64 = arith.extf %v : f32 to f64
          %e64 = arith.extf %expected : f32 to f64
          llvm.call @printf(%fmt_fail, %rank, %i_i64, %v64, %e64) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i64, f64, f64) -> i32
          %inc = arith.addi %nfail_acc, %c1_i32 : i32
          scf.yield %inc : i32
        } else {
          scf.yield %nfail_acc : i32
        }
        scf.yield %new_nfail : i32
      }

      // If no failures, print PASS
      %ok = arith.cmpi eq, %nfail, %c0_i32 : i32
      scf.if %ok {
        %fmt_pass = llvm.mlir.addressof @msg_pass : !llvm.ptr
        %e64 = arith.extf %expected : f32 to f64
        llvm.call @printf(%fmt_pass, %rank, %peer, %e64) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, f64) -> i32
      }

      // Cleanup
      func.call @free(%host_rb) : (!llvm.ptr) -> ()
      func.call @mgpuMemFree(%local_copy, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
    } else {
      %fmt_only1 = llvm.mlir.addressof @msg_only1 : !llvm.ptr
      llvm.call @printf(%fmt_only1, %rank) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    }

    func.call @mgpuBarrier() : () -> ()

    // Cleanup
    func.call @free(%hostbuf) : (!llvm.ptr) -> ()
    func.call @mgpuSymmetricFree(%buf, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
    func.call @mgpuSymmetricHeapDestroy() : () -> ()

    %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
    llvm.call @printf(%fmt_done, %rank) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

    return
  }
}
