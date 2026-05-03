//===- air_sym_with_alloc.mlir - air.rank + memref.alloc air.symmetric e2e ===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Variant of air_sym_with_rank.mlir that uses `memref.alloc {air.symmetric}`
// instead of a direct call to `mgpuSymmetricAlloc`. Exercises Phase 3
// (`air-rank-to-mgpu`) AND Phase 4 (`air-symmetric-alloc-to-mgpu`).
//
// The symmetric memref is wrapped/unwrapped via the standard
// `memref.extract_aligned_pointer_as_index` -> `llvm.inttoptr` idiom to
// recover the !llvm.ptr that the runtime ABI expects.
//
//===-----------------------------------------------------------------------===//

module {
  func.func private @mgpuGetWorldSize() -> i32
  func.func private @mgpuGetHeapBases() -> !llvm.ptr
  func.func private @mgpuBarrier()
  func.func private @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
  func.func private @mgpuMemFree(!llvm.ptr, !llvm.ptr)
  func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
  func.func private @malloc(i64) -> !llvm.ptr
  func.func private @free(!llvm.ptr)
  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.mlir.global internal constant @msg_pass("[mlir/alloc] rank %d: cross-rank read PASS (peer=%d, expected=%.1f)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_only1("[mlir/alloc] rank %d: world_size=1, skipping cross-rank read\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done("[mlir/alloc] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  func.func @main() {
    %c2 = arith.constant 2 : index

    air.rank (%rid) in (%rsize = %c2) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c4096_i64 = arith.constant 4096 : i64
      %nullptr = llvm.mlir.zero : !llvm.ptr
      %false = arith.constant false
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index

      %rid_i64 = arith.index_cast %rid : index to i64
      %rid_i32 = arith.trunci %rid_i64 : i64 to i32
      %rsize_i64 = arith.index_cast %rsize : index to i64
      %rsize_i32 = arith.trunci %rsize_i64 : i64 to i32

      // === Phase 4 lowering target: memref.alloc {air.symmetric} ===
      %buf_memref = memref.alloc() {air.symmetric} : memref<1024xf32>

      // Extract the underlying pointer for use with the mgpu* runtime ABI.
      // (Symmetric heap memory is GPU-only; CPU writes go through mgpuMemcpy.)
      %intptr = memref.extract_aligned_pointer_as_index %buf_memref
          : memref<1024xf32> -> index
      %buf_int = arith.index_cast %intptr : index to i64
      %buf = llvm.inttoptr %buf_int : i64 to !llvm.ptr

      // Fill (rid+1).0 from a host buffer via mgpuMemcpy H2D.
      %r1_i32 = arith.addi %rid_i32, %c1_i32 : i32
      %r1_f = arith.sitofp %r1_i32 : i32 to f32
      %hostbuf = func.call @malloc(%c4096_i64) : (i64) -> !llvm.ptr
      scf.for %i = %c0 to %c1024 step %c1 {
        %i_i64 = arith.index_cast %i : index to i64
        %addr = llvm.getelementptr %hostbuf[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %r1_f, %addr : f32, !llvm.ptr
      }
      func.call @mgpuMemcpy(%buf, %hostbuf, %c4096_i64, %nullptr) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      func.call @free(%hostbuf) : (!llvm.ptr) -> ()

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
      memref.dealloc %buf_memref : memref<1024xf32>

      %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
      llvm.call @printf(%fmt_done, %rid_i32) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
      air.rank_terminator
    }
    return
  }
}
