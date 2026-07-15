//===- air_rank/allgather.mlir - air.rank wrap of handwritten allgather --===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// High-level version of handwritten/allgather.mlir.
//
// This file is a 1:1 wrap of the SIMD-across-ranks all-gather test inside
// an `air.rank` op:
//
//   air.rank (%rid) in (%rsize = %c2) { <allgather main body> }
//
// The `air-rank-to-mgpu` pass (Phase 3) lowers this to:
//   - mgpuSymmetricHeapInit at function entry
//   - %rid := index_cast(mgpuGetRank())   (delinearization for 1D)
//   - %rsize := constant 2
//   - body inlined
//   - mgpuSymmetricHeapDestroy before each func.return
//
// After lowering the IR is functionally equivalent to
// handwritten/allgather.mlir (same kernel, same launch dispatch, same
// validation). Sister file: air_rank/cacheline.mlir does the analogous
// wrap of the producer/consumer cacheline test.
//
// The kernel and helpers (gpu.module @sym_kernels, @wrap_bytes) are
// duplicated verbatim from the handwritten allgather. Only @main differs
// in being wrapped in air.rank and using %rid / %rsize where the
// handwritten test calls mgpuGetRank() / mgpuGetWorldSize().
//
// Pinned to W=2 because air.translate today requires static-shape
// source memref (see AIRTranslateToLLVMPass.cpp). Same constraint as
// the handwritten allgather.
//
// Launcher: `make INPUT=allgather` from this subdir forks 2 processes.
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
  llvm.mlir.global internal constant @msg_pass(
      "[mlir/rank] rank %d: all-gather PASS (slot[0]={data=%d, flag=%d}, slot[%d]={data=%d, flag=%d})\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_fail(
      "[mlir/rank] rank %d: MISMATCH at idx=%ld got=%d expected=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done(
      "[mlir/rank] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // ---- GPU kernel (verbatim from air_sym_handwritten_allgather.mlir) ----
  gpu.module @sym_kernels {

    gpu.func @allgather(%output     : memref<64xi32>,
                        %verify_buf : memref<64xi32>,
                        %my_rank    : index,
                        %world      : index,
                        %bases      : memref<?xindex>) kernel
                        attributes {gpu.known_block_size = array<i32: 64, 1, 1>,
                                    gpu.known_grid_size  = array<i32: 1, 1, 1>} {
      %c0        = arith.constant 0    : index
      %c1        = arith.constant 1    : index
      %c31       = arith.constant 31   : index
      %c32       = arith.constant 32   : index
      %c0_i32    = arith.constant 0    : i32
      %c1_i32    = arith.constant 1    : i32
      %c31_i32   = arith.constant 31   : i32
      %c64_i32   = arith.constant 64   : i32
      %c100_i32  = arith.constant 100  : i32
      %c1000_i32 = arith.constant 1000 : i32

      %tid    = gpu.thread_id x
      %active = arith.cmpi ult, %tid, %c32 : index

      %my_rank_i32  = arith.index_cast %my_rank : index to i32
      %my_rank_x1k  = arith.muli %my_rank_i32, %c1000_i32 : i32
      %tid_i32      = arith.index_cast %tid : index to i32
      %payload_lane = arith.addi %tid_i32, %c100_i32 : i32
      %payload      = arith.addi %payload_lane, %my_rank_x1k : i32
      %is_flag_lane = arith.cmpi eq, %tid, %c31 : index
      %my_val       = arith.select %is_flag_lane, %c1_i32, %payload : i32

      %my_slot_off = arith.muli %my_rank, %c32 : index
      %my_addr     = arith.addi %my_slot_off, %tid : index

      // Phase 1: publish my slice to every peer's slot[my_rank].
      scf.for %peer = %c0 to %world step %c1 {
        %peer_out = air.translate %output, %my_rank, %peer, %bases
            : memref<64xi32>, memref<?xindex>
        scf.if %active {
          memref.store %my_val, %peer_out[%my_addr] : memref<64xi32>
        }
      }

      // Phase 2: spin on every slot in my LOCAL output.
      scf.for %peer = %c0 to %world step %c1 {
        %slot_off = arith.muli %peer, %c32 : index
        %addr     = arith.addi %slot_off, %tid : index

        %final_v = scf.while (%dummy = %c0_i32) : (i32) -> i32 {
          %v = scf.if %active -> i32 {
            %loaded = memref.load %output[%addr] : memref<64xi32>
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
          memref.store %final_v, %verify_buf[%addr] : memref<64xi32>
        }
      }
      gpu.return
    }
  }

  // ---- Helpers (verbatim from air_sym_handwritten_allgather.mlir) -------
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

  // ---- main: allgather test wrapped in air.rank ------------------------
  func.func @main() {
    // World size declared at the source level. air-rank-to-mgpu uses
    // this as %rsize and resolves %rid from mgpuGetRank() at runtime.
    %c2 = arith.constant 2 : index

    air.rank (%rid) in (%rsize = %c2) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c8_i64 = arith.constant 8 : i64
      %nullptr = llvm.mlir.zero : !llvm.ptr
      %false = arith.constant false

      %c0_idx  = arith.constant 0  : index
      %c1_idx  = arith.constant 1  : index
      %c31_idx = arith.constant 31 : index
      %c32_idx = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index

      %rid_i64   = arith.index_cast %rid : index to i64
      %rid_i32   = arith.trunci %rid_i64 : i64 to i32
      %rsize_i64 = arith.index_cast %rsize : index to i64
      %rsize_i32 = arith.trunci %rsize_i64 : i64 to i32

      %fmt_init = llvm.mlir.addressof @msg_init : !llvm.ptr
      llvm.call @printf(%fmt_init, %rid_i32, %rsize_i32)
          vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32

      // Output buffer = 2 cache lines = 256 bytes (64 i32).
      %output_bytes_static = arith.constant 256 : i64
      %output_ptr = func.call @mgpuSymmetricAlloc(%output_bytes_static, %nullptr)
          : (i64, !llvm.ptr) -> !llvm.ptr

      %total_elems = arith.constant 64 : index

      // Zero-init from host so spins start at flag=0.
      %output_host = memref.alloc() : memref<64xi32>
      scf.for %i = %c0_idx to %total_elems step %c1_idx {
        memref.store %c0_i32, %output_host[%i] : memref<64xi32>
      }
      %output_host_intptr = memref.extract_aligned_pointer_as_index %output_host
          : memref<64xi32> -> index
      %output_host_int = arith.index_cast %output_host_intptr : index to i64
      %output_host_ptr = llvm.inttoptr %output_host_int : i64 to !llvm.ptr
      func.call @mgpuMemcpy(%output_ptr, %output_host_ptr, %output_bytes_static, %nullptr)
          : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      memref.dealloc %output_host : memref<64xi32>

      func.call @mgpuBarrier() : () -> ()

      %c0_view = arith.constant 0 : index
      %output_bytes_m = func.call @wrap_bytes(%output_ptr, %output_bytes_static)
          : (!llvm.ptr, i64) -> memref<?xi8>
      %output_m = memref.view %output_bytes_m[%c0_view][]
          : memref<?xi8> to memref<64xi32>

      %verify_ptr = func.call @mgpuMemAlloc(%output_bytes_static, %nullptr, %false)
          : (i64, !llvm.ptr, i1) -> !llvm.ptr
      %verify_bytes_m = func.call @wrap_bytes(%verify_ptr, %output_bytes_static)
          : (!llvm.ptr, i64) -> memref<?xi8>
      %verify_m = memref.view %verify_bytes_m[%c0_view][]
          : memref<?xi8> to memref<64xi32>

      // heap_bases (host ptr → device copy).
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

      // Same kernel on every rank — only the rank-id arg differs.
      // (No is_producer/is_consumer dispatch: the SIMD all-gather body
      // is symmetric.)
      gpu.launch_func @sym_kernels::@allgather
          blocks  in (%c1, %c1, %c1)
          threads in (%c64, %c1, %c1)
          args(%output_m  : memref<64xi32>,
               %verify_m  : memref<64xi32>,
               %rid       : index,
               %rsize     : index,
               %bases     : memref<?xindex>)

      // D2H readback verify_buf.
      %hb = memref.alloc() : memref<64xi32>
      %hb_intptr = memref.extract_aligned_pointer_as_index %hb
          : memref<64xi32> -> index
      %hb_int = arith.index_cast %hb_intptr : index to i64
      %hb_ptr = llvm.inttoptr %hb_int : i64 to !llvm.ptr
      func.call @mgpuMemcpy(%hb_ptr, %verify_ptr, %output_bytes_static, %nullptr)
          : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

      %c100_i32  = arith.constant 100  : i32
      %c1000_i32 = arith.constant 1000 : i32

      // Check every (slot, lane) pair: slot*1000 + lane+100, except
      // lane 31 which should be the flag (1).
      %nfail = scf.for %i = %c0_idx to %total_elems step %c1_idx
                      iter_args(%nfail_acc = %c0_i32) -> (i32) {
        %slot = arith.divui %i, %c32_idx : index
        %lane = arith.remui %i, %c32_idx : index
        %v = memref.load %hb[%i] : memref<64xi32>
        %is_flag_idx = arith.cmpi eq, %lane, %c31_idx : index
        %expected = scf.if %is_flag_idx -> i32 {
          scf.yield %c1_i32 : i32
        } else {
          %lane_i32  = arith.index_cast %lane : index to i32
          %slot_i32  = arith.index_cast %slot : index to i32
          %slot_x1k  = arith.muli %slot_i32, %c1000_i32 : i32
          %lane_p100 = arith.addi %lane_i32, %c100_i32 : i32
          %e = arith.addi %lane_p100, %slot_x1k : i32
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

      %ok = arith.cmpi eq, %nfail, %c0_i32 : i32
      scf.if %ok {
        %fmt_p = llvm.mlir.addressof @msg_pass : !llvm.ptr
        %v00 = memref.load %hb[%c0_idx] : memref<64xi32>
        %v0f = memref.load %hb[%c31_idx] : memref<64xi32>
        %last_slot     = arith.subi %rsize, %c1_idx : index
        %last_slot_off = arith.muli %last_slot, %c32_idx : index
        %vL_idx        = arith.addi %last_slot_off, %c0_idx : index
        %vL            = memref.load %hb[%vL_idx] : memref<64xi32>
        %vLf_idx       = arith.addi %last_slot_off, %c31_idx : index
        %vLf           = memref.load %hb[%vLf_idx] : memref<64xi32>
        %last_slot_i32 = arith.index_cast %last_slot : index to i32
        llvm.call @printf(%fmt_p, %rid_i32, %v00, %v0f, %last_slot_i32, %vL, %vLf)
            vararg(!llvm.func<i32 (ptr, ...)>)
            : (!llvm.ptr, i32, i32, i32, i32, i32, i32) -> i32
      } else {
        func.call @exit(%c1_i32) : (i32) -> ()
      }

      memref.dealloc %hb : memref<64xi32>
      func.call @mgpuMemFree(%verify_ptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()

      func.call @mgpuBarrier() : () -> ()
      func.call @mgpuMemFree(%bases_devptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
      func.call @mgpuSymmetricFree(%output_ptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()

      %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
      llvm.call @printf(%fmt_done, %rid_i32)
          vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

      air.rank_terminator
    }
    return
  }
}
