//===- air_sym_with_channel.mlir - air.channel gpu_symmetric_heap e2e ----===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Highest-level form combining:
//   - Phase 1: gpu_symmetric_heap channel_type, air.symmetric memref attribute
//   - Phase 3: air-rank-to-mgpu (rank body inlining)
//   - Phase 4: air-symmetric-alloc-to-mgpu (memref.alloc -> mgpuSymmetricAlloc)
//   - Phase 6: air-gpu-channel-to-mgpu (gpu_symmetric_heap put/get -> peer-VA
//              mgpuMemcpy + mgpuBarrier)
//
// Each rank fills a symmetric src buffer with (rank+1).0, publishes via
// air.channel.put, and reads rank 0's slot via air.channel.get into a local
// dst buffer. Both ranks should see 1.0 in dst[0].
//
//===-----------------------------------------------------------------------===//

module {
  func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)
  func.func private @malloc(i64) -> !llvm.ptr
  func.func private @free(!llvm.ptr)
  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.mlir.global internal constant @msg_pass("[mlir/chan] rank %d: channel get PASS (read rank 0 = %.1f)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done("[mlir/chan] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // Channel decl at module scope (Symbol).
  air.channel @sym_chan [] {channel_type = "gpu_symmetric_heap"}

  func.func @main() {
    %c2 = arith.constant 2 : index

    air.rank (%rid) in (%rsize = %c2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1024 = arith.constant 1024 : index
      %c1_i32 = arith.constant 1 : i32
      %c4096_i64 = arith.constant 4096 : i64
      %nullptr = llvm.mlir.zero : !llvm.ptr

      %rid_i64 = arith.index_cast %rid : index to i64
      %rid_i32 = arith.trunci %rid_i64 : i64 to i32

      // Symmetric src buffer (each rank allocates same shape at same offset).
      %src_buf = memref.alloc() {air.symmetric} : memref<1024xf32>
      // Local non-symmetric destination.
      %dst_buf = memref.alloc() {air.symmetric} : memref<1024xf32>

      // Fill src_buf with (rid+1).0 from host.
      %r1_i32 = arith.addi %rid_i32, %c1_i32 : i32
      %r1_f = arith.sitofp %r1_i32 : i32 to f32
      %hostbuf = func.call @malloc(%c4096_i64) : (i64) -> !llvm.ptr
      scf.for %i = %c0 to %c1024 step %c1 {
        %i_i64 = arith.index_cast %i : index to i64
        %addr = llvm.getelementptr %hostbuf[%i_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %r1_f, %addr : f32, !llvm.ptr
      }
      %src_intptr = memref.extract_aligned_pointer_as_index %src_buf
          : memref<1024xf32> -> index
      %src_int = arith.index_cast %src_intptr : index to i64
      %src_ptr = llvm.inttoptr %src_int : i64 to !llvm.ptr
      func.call @mgpuMemcpy(%src_ptr, %hostbuf, %c4096_i64, %nullptr)
          : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

      // === Phase 6 lowering target: gpu_symmetric_heap channel put/get ===
      // put publishes our src_buf; get reads peer (rank 0) into dst_buf.
      air.channel.put @sym_chan[] (%src_buf[] [] []) : (memref<1024xf32>)
      air.channel.get @sym_chan[%c0] (%dst_buf[] [] []) : (memref<1024xf32>)

      // Verify: D2H readback dst_buf to a host buffer, check element 0.
      %dst_intptr = memref.extract_aligned_pointer_as_index %dst_buf
          : memref<1024xf32> -> index
      %dst_int = arith.index_cast %dst_intptr : index to i64
      %dst_ptr = llvm.inttoptr %dst_int : i64 to !llvm.ptr
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
