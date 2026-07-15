//===- air_rank/cacheline.mlir - air.rank wrap of handwritten cacheline --===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// High-level version of handwritten/cacheline.mlir.
//
// This file is a 1:1 wrap of the cacheline producer/consumer test inside
// an `air.rank` op:
//
//   air.rank (%rid) in (%rsize = %c2) { <cacheline main body> }
//
// The `air-rank-to-mgpu` pass (Phase 3) lowers this to:
//   - mgpuSymmetricHeapInit at function entry
//   - %rid := index_cast(mgpuGetRank())   (delinearization for 1D)
//   - %rsize := constant 2
//   - body inlined
//   - mgpuSymmetricHeapDestroy before each func.return
//
// After lowering the IR is functionally equivalent to
// handwritten/cacheline.mlir (same kernels, same launch dispatch, same
// validation). This file's job is to demonstrate that the user can
// write the multi-process world declaratively via air.rank and have
// the pass produce the handwritten reference.
//
// The kernels and helpers (gpu.module @sym_kernels, @wrap_bytes) are
// duplicated verbatim from the cacheline test. Only @main differs in
// being wrapped in air.rank and using %rid / %rsize where the
// handwritten test calls mgpuGetRank() / mgpuGetWorldSize().
//
// Pinned to W=2 because air.translate today requires static-shape
// source memref (see AIRTranslateToLLVMPass.cpp). Same constraint as
// the handwritten cacheline test.
//
// Launcher: `make INPUT=cacheline` from this subdir forks 2 processes.
// The air-rank-to-mgpu pass converts air.rank to runtime dispatch.
//
//===-----------------------------------------------------------------------===//

module attributes {gpu.container_module} {
  // ---- mgpu* C ABI declarations -----------------------------------------
  // Note: SymmetricHeapInit/Destroy/GetRank/GetWorldSize are emitted by
  // the air-rank-to-mgpu pass; user IR doesn't reference them directly.
  func.func private @mgpuSymmetricAlloc(i64, !llvm.ptr) -> !llvm.ptr
  func.func private @mgpuSymmetricFree(!llvm.ptr, !llvm.ptr)
  func.func private @mgpuGetHeapBases() -> !llvm.ptr
  func.func private @mgpuBarrier()
  func.func private @mgpuMemAlloc(i64, !llvm.ptr, i1) -> !llvm.ptr
  func.func private @mgpuMemFree(!llvm.ptr, !llvm.ptr)
  func.func private @mgpuMemcpy(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

  func.func private @exit(i32)

  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.mlir.global internal constant @msg_init(
      "[mlir/rank] rank %d / world %d, init OK\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_p(
      "[mlir/rank] rank 0 (producer): kernel returned\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_c(
      "[mlir/rank] rank 1 (consumer): cache-line message PASS (data[0]=%d, flag=%d)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_fail(
      "[mlir/rank] rank 1 (consumer): MISMATCH at idx=%ld got=%d expected=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done(
      "[mlir/rank] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // ---- GPU kernels (verbatim from air_sym_handwritten_cacheline.mlir) ---
  gpu.module @sym_kernels {

    gpu.func @producer(%data : memref<32xi32>,
                       %bases : memref<?xindex>) kernel
                       attributes {gpu.known_block_size = array<i32: 64, 1, 1>,
                                   gpu.known_grid_size  = array<i32: 1, 1, 1>} {
      %c1_i32   = arith.constant 1   : i32
      %c100_i32 = arith.constant 100 : i32
      %c31      = arith.constant 31  : index
      %c32      = arith.constant 32  : index
      %from = arith.constant 0 : index
      %to   = arith.constant 1 : index

      %tid = gpu.thread_id x
      %active = arith.cmpi ult, %tid, %c32 : index
      %peer_data = air.translate %data, %from, %to, %bases
          : memref<32xi32>, memref<?xindex>

      scf.if %active {
        %is_flag  = arith.cmpi eq, %tid, %c31 : index
        %tid_i32  = arith.index_cast %tid : index to i32
        %payload  = arith.addi %tid_i32, %c100_i32 : i32
        %val      = arith.select %is_flag, %c1_i32, %payload : i32
        memref.store %val, %peer_data[%tid] : memref<32xi32>
      }
      gpu.return
    }

    gpu.func @consumer(%data       : memref<32xi32>,
                       %verify_buf : memref<32xi32>) kernel
                       attributes {gpu.known_block_size = array<i32: 64, 1, 1>,
                                   gpu.known_grid_size  = array<i32: 1, 1, 1>} {
      %c0_i32  = arith.constant 0  : i32
      %c1_i32  = arith.constant 1  : i32
      %c31_i32 = arith.constant 31 : i32
      %c64_i32 = arith.constant 64 : i32
      %c32     = arith.constant 32 : index

      %tid = gpu.thread_id x
      %active = arith.cmpi ult, %tid, %c32 : index

      %final_v = scf.while (%dummy = %c0_i32) : (i32) -> i32 {
        %v = scf.if %active -> i32 {
          %loaded = memref.load %data[%tid] : memref<32xi32>
          scf.yield %loaded : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %flag, %valid = gpu.shuffle idx %v, %c31_i32, %c64_i32 : i32
        %not_ready = arith.cmpi ne, %flag, %c1_i32 : i32
        scf.condition(%not_ready) %v : i32
      } do {
      ^bb0(%v_iter : i32):
        scf.yield %v_iter : i32
      }

      scf.if %active {
        memref.store %final_v, %verify_buf[%tid] : memref<32xi32>
      }
      gpu.return
    }
  }

  // ---- Helpers (verbatim from air_sym_handwritten_cacheline.mlir) -------
  func.func private @wrap_bytes(%ptr : !llvm.ptr, %size : i64) -> memref<?xi8> {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %d0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d1 = llvm.insertvalue %ptr,    %d0[0]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d2 = llvm.insertvalue %ptr,    %d1[1]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d3 = llvm.insertvalue %c0_i64, %d2[2]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d4 = llvm.insertvalue %size,   %d3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d5 = llvm.insertvalue %c1_i64, %d4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %m  = builtin.unrealized_conversion_cast %d5
        : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xi8>
    return %m : memref<?xi8>
  }

  // ---- main: cacheline test wrapped in air.rank ------------------------
  func.func @main() {
    // World size declared at the source level. air-rank-to-mgpu uses
    // this as %rsize and resolves %rid from mgpuGetRank() at runtime.
    %c2 = arith.constant 2 : index

    air.rank (%rid) in (%rsize = %c2) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i64 = arith.constant 0 : i64
      %c128_bytes = arith.constant 128 : i64
      %nullptr = llvm.mlir.zero : !llvm.ptr
      %false = arith.constant false

      %c0_idx = arith.constant 0 : index
      %c1_idx = arith.constant 1 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index

      // Convert air.rank's %rid (index) and %rsize (index) to i32 for
      // printf and the rank-dispatch comparisons. The handwritten test
      // gets these from mgpuGetRank() / mgpuGetWorldSize() directly.
      %rid_i64 = arith.index_cast %rid : index to i64
      %rid_i32 = arith.trunci %rid_i64 : i64 to i32
      %rsize_i64 = arith.index_cast %rsize : index to i64
      %rsize_i32 = arith.trunci %rsize_i64 : i64 to i32

      %fmt_init = llvm.mlir.addressof @msg_init : !llvm.ptr
      llvm.call @printf(%fmt_init, %rid_i32, %rsize_i32)
          vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32

      %data_ptr = func.call @mgpuSymmetricAlloc(%c128_bytes, %nullptr)
          : (i64, !llvm.ptr) -> !llvm.ptr

      // Zero-init data from host.
      %data_host = memref.alloc() : memref<32xi32>
      %dc32 = arith.constant 32 : index
      scf.for %i = %c0_idx to %dc32 step %c1_idx {
        memref.store %c0_i32, %data_host[%i] : memref<32xi32>
      }
      %data_host_intptr = memref.extract_aligned_pointer_as_index %data_host
          : memref<32xi32> -> index
      %data_host_int = arith.index_cast %data_host_intptr : index to i64
      %data_host_ptr = llvm.inttoptr %data_host_int : i64 to !llvm.ptr
      func.call @mgpuMemcpy(%data_ptr, %data_host_ptr, %c128_bytes, %nullptr)
          : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      memref.dealloc %data_host : memref<32xi32>

      func.call @mgpuBarrier() : () -> ()

      %c0_view = arith.constant 0 : index
      %data_bytes = func.call @wrap_bytes(%data_ptr, %c128_bytes)
          : (!llvm.ptr, i64) -> memref<?xi8>
      %data_m = memref.view %data_bytes[%c0_view][]
          : memref<?xi8> to memref<32xi32>

      // heap_bases (host ptr → device copy).
      %c8_i64 = arith.constant 8 : i64
      %bases_size = arith.muli %rsize_i64, %c8_i64 : i64
      %bases_host = func.call @mgpuGetHeapBases() : () -> !llvm.ptr
      %bases_devptr = func.call @mgpuMemAlloc(%bases_size, %nullptr, %false)
          : (i64, !llvm.ptr, i1) -> !llvm.ptr
      func.call @mgpuMemcpy(%bases_devptr, %bases_host, %bases_size, %nullptr)
          : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      %bases_bytes = func.call @wrap_bytes(%bases_devptr, %bases_size)
          : (!llvm.ptr, i64) -> memref<?xi8>
      %bases = memref.view %bases_bytes[%c0_view][%rsize]
          : memref<?xi8> to memref<?xindex>

      // Rank dispatch (rank 0 = producer, rank 1 = consumer). Compares
      // %rid directly as index — equivalent to the handwritten test's
      // i32 comparison after index_cast(mgpuGetRank()).
      %is_producer = arith.cmpi eq, %rid, %c0_idx : index
      scf.if %is_producer {
        gpu.launch_func @sym_kernels::@producer
            blocks  in (%c1, %c1, %c1)
            threads in (%c64, %c1, %c1)
            args(%data_m : memref<32xi32>,
                 %bases  : memref<?xindex>)
        %fmt_p = llvm.mlir.addressof @msg_pass_p : !llvm.ptr
        llvm.call @printf(%fmt_p)
            vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      } else {
        %is_consumer = arith.cmpi eq, %rid, %c1_idx : index
        scf.if %is_consumer {
          %verify_ptr = func.call @mgpuMemAlloc(%c128_bytes, %nullptr, %false)
              : (i64, !llvm.ptr, i1) -> !llvm.ptr
          %verify_bytes = func.call @wrap_bytes(%verify_ptr, %c128_bytes)
              : (!llvm.ptr, i64) -> memref<?xi8>
          %verify_m = memref.view %verify_bytes[%c0_view][]
              : memref<?xi8> to memref<32xi32>
          gpu.launch_func @sym_kernels::@consumer
              blocks  in (%c1, %c1, %c1)
              threads in (%c64, %c1, %c1)
              args(%data_m  : memref<32xi32>,
                   %verify_m: memref<32xi32>)

          // D2H readback verify_buf and check all 32 ints.
          %hb = memref.alloc() : memref<32xi32>
          %hb_intptr = memref.extract_aligned_pointer_as_index %hb
              : memref<32xi32> -> index
          %hb_int = arith.index_cast %hb_intptr : index to i64
          %hb_ptr = llvm.inttoptr %hb_int : i64 to !llvm.ptr
          func.call @mgpuMemcpy(%hb_ptr, %verify_ptr, %c128_bytes, %nullptr)
              : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

          %c31_idx  = arith.constant 31  : index
          %c32_idx  = arith.constant 32  : index
          %c100_i32 = arith.constant 100 : i32

          %nfail = scf.for %i = %c0_idx to %c32_idx step %c1_idx
                          iter_args(%nfail_acc = %c0_i32) -> (i32) {
            %v = memref.load %hb[%i] : memref<32xi32>
            %is_flag_idx = arith.cmpi eq, %i, %c31_idx : index
            %expected = scf.if %is_flag_idx -> i32 {
              scf.yield %c1_i32 : i32
            } else {
              %i_i32 = arith.index_cast %i : index to i32
              %e = arith.addi %i_i32, %c100_i32 : i32
              scf.yield %e : i32
            }
            %ne = arith.cmpi ne, %v, %expected : i32
            %new_nfail = scf.if %ne -> i32 {
              %is_first = arith.cmpi eq, %nfail_acc, %c0_i32 : i32
              scf.if %is_first {
                %fmt_fail = llvm.mlir.addressof @msg_fail : !llvm.ptr
                %i_i64 = arith.index_cast %i : index to i64
                llvm.call @printf(%fmt_fail, %rid_i32, %i_i64, %v, %expected)
                    vararg(!llvm.func<i32 (ptr, ...)>)
                    : (!llvm.ptr, i32, i64, i32, i32) -> i32
              }
              %inc = arith.addi %nfail_acc, %c1_i32 : i32
              scf.yield %inc : i32
            } else {
              scf.yield %nfail_acc : i32
            }
            scf.yield %new_nfail : i32
          }

          %ok_all = arith.cmpi eq, %nfail, %c0_i32 : i32
          scf.if %ok_all {
            %fmt_c = llvm.mlir.addressof @msg_pass_c : !llvm.ptr
            %v0 = memref.load %hb[%c0_idx] : memref<32xi32>
            %vf = memref.load %hb[%c31_idx] : memref<32xi32>
            llvm.call @printf(%fmt_c, %v0, %vf)
                vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32
          } else {
            func.call @exit(%c1_i32) : (i32) -> ()
          }

          memref.dealloc %hb : memref<32xi32>
          func.call @mgpuMemFree(%verify_ptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
        }
      }

      func.call @mgpuBarrier() : () -> ()
      func.call @mgpuMemFree(%bases_devptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call @mgpuSymmetricFree(%data_ptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()

      %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
      llvm.call @printf(%fmt_done, %rid_i32)
          vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

      air.rank_terminator
    }
    return
  }
}
