//===- air_sym_with_rank.mlir - High-level air.rank multi-GPU e2e --------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Higher-level version of air_sym_handwritten.mlir that uses `air.rank` to
// express the multi-process world. The `air-rank-to-mgpu` pass lowers
// air.rank to inline body + mgpuGetRank() / mgpuSymmetricHeapInit / Destroy.
//
// Once lowered, the IR matches air_sym_handwritten.mlir's behavior. After
// `mlir-opt --pass-pipeline=...`, both forms should run identically under
// the multi-process driver run.sh.
//
//===-----------------------------------------------------------------------===//

module {
  // ---- mgpu* C ABI declarations --------------------------------------
  func.func private @mgpuGetRank() -> i32
  func.func private @mgpuGetWorldSize() -> i32
  func.func private @mgpuSymmetricAlloc(i64, !llvm.ptr) -> !llvm.ptr
  func.func private @mgpuSymmetricFree(!llvm.ptr, !llvm.ptr)
  func.func private @mgpuGetHeapBases() -> !llvm.ptr
  func.func private @mgpuBarrier()
  func.func private @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
  func.func private @mgpuMemFree(!llvm.ptr, !llvm.ptr)
  func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

  // libc helpers
  func.func private @malloc(i64) -> !llvm.ptr
  func.func private @free(!llvm.ptr)
  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.mlir.global internal constant @msg_pass("[mlir/rank] rank %d: cross-rank read PASS (peer=%d, expected=%.1f)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_only1("[mlir/rank] rank %d: world_size=1, skipping cross-rank read\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done("[mlir/rank] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  func.func @main() {
    %c2 = arith.constant 2 : index

    // High-level: a 2-rank world. The body executes once per rank.
    air.rank (%rid) in (%rsize = %c2) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c4096_i64 = arith.constant 4096 : i64
      %nullptr = llvm.mlir.zero : !llvm.ptr
      %false = arith.constant false
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index

      // Convert rank id (index) to i32 for printf and arithmetic.
      %rid_i64 = arith.index_cast %rid : index to i64
      %rid_i32 = arith.trunci %rid_i64 : i64 to i32
      %rsize_i64 = arith.index_cast %rsize : index to i64
      %rsize_i32 = arith.trunci %rsize_i64 : i64 to i32

      %buf = func.call @mgpuSymmetricAlloc(%c4096_i64, %nullptr) : (i64, !llvm.ptr) -> !llvm.ptr

      // Fill buf with (rank+1).0 from host
      %hostbuf = func.call @malloc(%c4096_i64) : (i64) -> !llvm.ptr
      %r1_i32 = arith.addi %rid_i32, %c1_i32 : i32
      %r1_f = arith.sitofp %r1_i32 : i32 to f32
      scf.for %i = %c0 to %c1024 step %c1 {
        %i_i64 = arith.index_cast %i : index to i64
        %addr = llvm.getelementptr %hostbuf[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %r1_f, %addr : f32, !llvm.ptr
      }
      func.call @mgpuMemcpy(%buf, %hostbuf, %c4096_i64, %nullptr) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      func.call @mgpuBarrier() : () -> ()

      %is_multi = arith.cmpi sgt, %rsize_i32, %c1_i32 : i32
      scf.if %is_multi {
        %sum = arith.addi %rid_i32, %c1_i32 : i32
        %peer_i32 = arith.remsi %sum, %rsize_i32 : i32
        %bases = func.call @mgpuGetHeapBases() : () -> !llvm.ptr
        %peer_i64 = arith.extsi %peer_i32 : i32 to i64
        %peer_base_addr = llvm.getelementptr %bases[%peer_i64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %peer_base = llvm.load %peer_base_addr : !llvm.ptr -> !llvm.ptr
        %local_base_addr = llvm.getelementptr %bases[%rid_i64] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
        %local_base = llvm.load %local_base_addr : !llvm.ptr -> !llvm.ptr
        %buf_int = llvm.ptrtoint %buf : !llvm.ptr to i64
        %lb_int = llvm.ptrtoint %local_base : !llvm.ptr to i64
        %offset = arith.subi %buf_int, %lb_int : i64
        %peer_buf = llvm.getelementptr %peer_base[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, i8

        %local_copy = func.call @mgpuMemAlloc(%c4096_i64, %nullptr, %false) : (i64, !llvm.ptr, i1) -> !llvm.ptr
        func.call @mgpuMemcpy(%local_copy, %peer_buf, %c4096_i64, %nullptr) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
        %host_rb = func.call @malloc(%c4096_i64) : (i64) -> !llvm.ptr
        func.call @mgpuMemcpy(%host_rb, %local_copy, %c4096_i64, %nullptr) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

        %p1_i32 = arith.addi %peer_i32, %c1_i32 : i32
        %expected = arith.sitofp %p1_i32 : i32 to f32
        %c0_i64 = arith.constant 0 : i64
        %addr0 = llvm.getelementptr %host_rb[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %v0 = llvm.load %addr0 : !llvm.ptr -> f32
        %ok = arith.cmpf oeq, %v0, %expected : f32
        scf.if %ok {
          %fmt = llvm.mlir.addressof @msg_pass : !llvm.ptr
          %e64 = arith.extf %expected : f32 to f64
          llvm.call @printf(%fmt, %rid_i32, %peer_i32, %e64) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32, f64) -> i32
        }

        func.call @free(%host_rb) : (!llvm.ptr) -> ()
        func.call @mgpuMemFree(%local_copy, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
      } else {
        %fmt = llvm.mlir.addressof @msg_only1 : !llvm.ptr
        llvm.call @printf(%fmt, %rid_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      }

      func.call @mgpuBarrier() : () -> ()
      func.call @free(%hostbuf) : (!llvm.ptr) -> ()
      func.call @mgpuSymmetricFree(%buf, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()

      %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
      llvm.call @printf(%fmt_done, %rid_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      air.rank_terminator
    }
    return
  }
}
