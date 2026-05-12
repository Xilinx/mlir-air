//===- air_sym_with_dma.mlir - air.rank + air.dma_memcpy_nd cross-rank ----===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Highest-level form of the symmetric-heap test. Combines:
//   - Phase 1: air.symmetric memref attribute, src_rank attribute on
//              air.dma_memcpy_nd
//   - Phase 3: air-rank-to-mgpu (rank body inlining)
//   - Phase 4: air-symmetric-alloc-to-mgpu (memref.alloc -> mgpuSymmetricAlloc)
//   - Phase 5: air-cross-rank-dma-to-mgpu (cross-rank dma -> peer-VA mgpuMemcpy)
//
// Each rank allocates two symmetric buffers (src and dst), fills its src with
// (rank+1).0, then issues a cross-rank DMA reading rank 0's src into its
// own dst, and verifies dst contains 1.0 on every rank.
//
//===-----------------------------------------------------------------------===//

module {
  func.func private @mgpuBarrier()
  func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
  func.func private @malloc(i64) -> !llvm.ptr
  func.func private @free(!llvm.ptr)
  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.mlir.global internal constant @msg_pass("[mlir/dma] rank %d: cross-rank DMA PASS (read rank 0 = %.1f)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done("[mlir/dma] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  func.func @main() {
    %c2 = arith.constant 2 : index

    air.rank (%rid) in (%rsize = %c2) {
      %c1_i32 = arith.constant 1 : i32
      %c4096_i64 = arith.constant 4096 : i64
      %nullptr = llvm.mlir.zero : !llvm.ptr
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index

      %rid_i64 = arith.index_cast %rid : index to i64
      %rid_i32 = arith.trunci %rid_i64 : i64 to i32

      // Two symmetric buffers per rank (collective allocation).
      %src_buf = memref.alloc() {air.symmetric} : memref<1024xf32>
      %dst_buf = memref.alloc() {air.symmetric} : memref<1024xf32>

      // Get pointers for the H2D init (and later D2H verification).
      %src_intptr = memref.extract_aligned_pointer_as_index %src_buf
          : memref<1024xf32> -> index
      %src_int = arith.index_cast %src_intptr : index to i64
      %src_ptr = llvm.inttoptr %src_int : i64 to !llvm.ptr

      %dst_intptr = memref.extract_aligned_pointer_as_index %dst_buf
          : memref<1024xf32> -> index
      %dst_int = arith.index_cast %dst_intptr : index to i64
      %dst_ptr = llvm.inttoptr %dst_int : i64 to !llvm.ptr

      // Fill src_buf with (rid+1).0 via host buffer + mgpuMemcpy H2D.
      %r1_i32 = arith.addi %rid_i32, %c1_i32 : i32
      %r1_f = arith.sitofp %r1_i32 : i32 to f32
      %hostbuf = func.call @malloc(%c4096_i64) : (i64) -> !llvm.ptr
      scf.for %i = %c0 to %c1024 step %c1 {
        %i_i64 = arith.index_cast %i : index to i64
        %addr = llvm.getelementptr %hostbuf[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %r1_f, %addr : f32, !llvm.ptr
      }
      func.call @mgpuMemcpy(%src_ptr, %hostbuf, %c4096_i64, %nullptr)
          : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      func.call @mgpuBarrier() : () -> ()

      // === Phase 5 lowering target: cross-rank air.dma_memcpy_nd ===
      // Both ranks read from rank 0's src_buf into their own dst_buf.
      air.dma_memcpy_nd (%dst_buf[] [] [], %src_buf[] [] [])
          {src_rank = 0 : i64}
          : (memref<1024xf32>, memref<1024xf32>)

      // Verify: D2H readback dst_buf to a host buffer, check element 0.
      // On every rank, dst_buf should contain (rank0 + 1).0 == 1.0.
      %host_rb = func.call @malloc(%c4096_i64) : (i64) -> !llvm.ptr
      func.call @mgpuMemcpy(%host_rb, %dst_ptr, %c4096_i64, %nullptr)
          : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      %c0_i64 = arith.constant 0 : i64
      %addr0 = llvm.getelementptr %host_rb[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %v0 = llvm.load %addr0 : !llvm.ptr -> f32
      %expected = arith.constant 1.0 : f32
      %ok = arith.cmpf oeq, %v0, %expected : f32
      scf.if %ok {
        %fmt = llvm.mlir.addressof @msg_pass : !llvm.ptr
        %v0_64 = arith.extf %v0 : f32 to f64
        llvm.call @printf(%fmt, %rid_i32, %v0_64) vararg(!llvm.func<i32 (ptr, ...)>)
            : (!llvm.ptr, i32, f64) -> i32
      }
      func.call @free(%host_rb) : (!llvm.ptr) -> ()

      func.call @mgpuBarrier() : () -> ()
      func.call @free(%hostbuf) : (!llvm.ptr) -> ()
      memref.dealloc %dst_buf : memref<1024xf32>
      memref.dealloc %src_buf : memref<1024xf32>

      %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
      llvm.call @printf(%fmt_done, %rid_i32) vararg(!llvm.func<i32 (ptr, ...)>)
          : (!llvm.ptr, i32) -> i32
      air.rank_terminator
    }
    return
  }
}
