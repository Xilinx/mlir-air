//===- air_sym_handwritten_allgather.mlir - multi-GPU all-gather (cache line) ===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Symmetric-heap all-gather e2e (WORLD_SIZE >= 2). SIMD across ranks:
// every rank runs the SAME kernel; the only per-rank variation is the
// rank ID, which determines both (a) the payload value and (b) which
// slot of each peer's output buffer to write into.
//
// Sister file: air_sym_handwritten_cacheline.mlir is the producer/
// consumer (1-to-1) version of the same cache-line atomicity mechanism.
// This file generalizes it to a many-to-many collective.
//
// Layout
// ======
// Each rank R has:
//   - a 32-i32 input slice with payload `R*1000 + lane+100` (lanes
//     0..30) and flag=1 (lane 31), all values fitting in one cache line.
//   - an output buffer of `world * 32` i32 (= world cache lines), laid
//     out as the concatenation `slot[0] | slot[1] | ... | slot[W-1]`.
//
// Phase 1 (publish):
//   For each peer P in 0..W-1, rank R writes its slice into the
//   sub-buffer P_output[R*32 .. (R+1)*32]. Self-write (P == R) is
//   covered by the same code path — `air.translate` with from == to
//   returns the same VA, so it becomes a local store.
//
// Phase 2 (collect):
//   For each peer P in 0..W-1, rank R spins on its LOCAL
//   output[P*32 .. (P+1)*32] until lane 31 (flag) shows up, then copies
//   the validated cache line into verify_buf for host check.
//
// Why this is interesting:
//   - SIMD across ranks: no producer/consumer asymmetry — same IR runs
//     on every rank with only the rank-id arg differing.
//   - N independent cache-line publish events vs the cacheline file's 1.
//   - Validates that air.translate composes correctly inside a peer
//     loop (one translate per destination per kernel).
//
// Cache-line atomicity contract is identical to the cacheline producer/
// consumer variant — see that file's header for the gfx940 / MI300
// reasoning.
//
// World-size constraint: this test is pinned to W=2 because air.translate
// today requires a static-shape source memref (see AIRTranslateToLLVMPass
// `requires a static-shape source memref`), and the output buffer's
// element count is W*32. The KERNEL itself is W-agnostic — its peer loop
// iterates `0..world` and the memref is just sized to fit. The host
// refuses any world other than 2 with exit(1). To lift this restriction,
// extend buildPeerDescriptor in AIRTranslateToLLVMPass to thread dynamic
// dim values into the descriptor's size/stride fields.
//
// Launcher: run.sh with INPUT=allgather forks 2 processes.
//
//===------------------------------------------------------------------===//

module attributes {gpu.container_module} {
  // ---- mgpu* C ABI declarations -----------------------------------------
  func.func private @mgpuSymmetricHeapInit(i64)
  func.func private @mgpuSymmetricHeapDestroy()
  func.func private @mgpuGetRank() -> i32
  func.func private @mgpuGetWorldSize() -> i32
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
      "[mlir] rank %d / world %d, init OK\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass(
      "[mlir] rank %d: all-gather PASS (slot[0]={data=%d, flag=%d}, slot[%d]={data=%d, flag=%d})\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_fail(
      "[mlir] rank %d: MISMATCH at idx=%ld got=%d expected=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done(
      "[mlir] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // ---- GPU kernel -------------------------------------------------------
  gpu.module @sym_kernels {

    // SIMD-across-ranks all-gather kernel. 1 wave × 64 lanes; lanes
    // 0..31 do all memory work, lanes 32..63 idle but participate in
    // gpu.shuffle to keep it wave-uniform.
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

      // My payload: my_rank*1000 + lane+100 for lanes 0..30; 1 for lane 31.
      %my_rank_i32  = arith.index_cast %my_rank : index to i32
      %my_rank_x1k  = arith.muli %my_rank_i32, %c1000_i32 : i32
      %tid_i32      = arith.index_cast %tid : index to i32
      %payload_lane = arith.addi %tid_i32, %c100_i32 : i32
      %payload      = arith.addi %payload_lane, %my_rank_x1k : i32
      %is_flag_lane = arith.cmpi eq, %tid, %c31 : index
      %my_val       = arith.select %is_flag_lane, %c1_i32, %payload : i32

      // Index of my slot in any output buffer: my_rank * 32 + lane.
      %my_slot_off = arith.muli %my_rank, %c32 : index
      %my_addr     = arith.addi %my_slot_off, %tid : index

      // Phase 1: publish my slice to every peer's slot[my_rank].
      // Self-write (peer == my_rank) goes through air.translate too —
      // with from == to the runtime returns the same VA, so it becomes
      // a local store. Keeps the kernel uniform across ranks.
      scf.for %peer = %c0 to %world step %c1 {
        %peer_out = air.translate %output, %my_rank, %peer, %bases
            : memref<64xi32>, memref<?xindex>
        scf.if %active {
          memref.store %my_val, %peer_out[%my_addr] : memref<64xi32>
        }
      }

      // Phase 2: spin on every slot in my LOCAL output until its flag
      // (lane 31) shows up, then copy that slot to verify_buf.
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

  // ---- Helpers ----------------------------------------------------------
  // Wrap a raw runtime !llvm.ptr as a 1-D byte memref. Phase 4's
  // AIRSymmetricAllocToMgpuPass replaces this entirely.
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

  // ---- main ------------------------------------------------------------
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c4_i64 = arith.constant 4 : i64    // bytes per i32
    %c8_i64 = arith.constant 8 : i64    // bytes per heap-base ptr
    %c32_i64 = arith.constant 32 : i64  // i32 per slot (one cache line)
    %heap_size = arith.constant 268435456 : i64  // 256 MB
    %nullptr = llvm.mlir.zero : !llvm.ptr
    %false = arith.constant false

    %c0_idx  = arith.constant 0  : index
    %c1_idx  = arith.constant 1  : index
    %c31_idx = arith.constant 31 : index
    %c32_idx = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    func.call @mgpuSymmetricHeapInit(%heap_size) : (i64) -> ()
    %rank = func.call @mgpuGetRank() : () -> i32
    %world = func.call @mgpuGetWorldSize() : () -> i32
    %fmt_init = llvm.mlir.addressof @msg_init : !llvm.ptr
    llvm.call @printf(%fmt_init, %rank, %world)
        vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32

    // Pinned to W=2: output memref must be static-shape because of the
    // current air.translate verifier. See file-level comment.
    %c2_i32 = arith.constant 2 : i32
    %is_w2 = arith.cmpi eq, %world, %c2_i32 : i32
    %not_w2 = arith.cmpi ne, %world, %c2_i32 : i32
    scf.if %not_w2 {
      func.call @exit(%c1_i32) : (i32) -> ()
    }

    // Output buffer = 2 cache lines = 2 * 128 = 256 bytes (64 i32).
    %output_bytes_static = arith.constant 256 : i64
    %output_ptr = func.call @mgpuSymmetricAlloc(%output_bytes_static, %nullptr)
        : (i64, !llvm.ptr) -> !llvm.ptr

    %total_elems = arith.constant 64 : index

    // Zero-init from host so spins start at flag=0 (and a "never written"
    // bug surfaces as a 0 at lane positions 0..30 too).
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

    func.call @mgpuBarrier() : () -> ()  // zero-init visible to all ranks

    %c0_view = arith.constant 0 : index
    %output_bytes_m = func.call @wrap_bytes(%output_ptr, %output_bytes_static)
        : (!llvm.ptr, i64) -> memref<?xi8>
    %output_m = memref.view %output_bytes_m[%c0_view][]
        : memref<?xi8> to memref<64xi32>

    // verify_buf in local HBM (mgpuMemAlloc; non-symmetric).
    %verify_ptr = func.call @mgpuMemAlloc(%output_bytes_static, %nullptr, %false)
        : (i64, !llvm.ptr, i1) -> !llvm.ptr
    %verify_bytes_m = func.call @wrap_bytes(%verify_ptr, %output_bytes_static)
        : (!llvm.ptr, i64) -> memref<?xi8>
    %verify_m = memref.view %verify_bytes_m[%c0_view][]
        : memref<?xi8> to memref<64xi32>

    // heap_bases (host ptr → device copy; same as cacheline test).
    %world_i64 = arith.extui %world : i32 to i64
    %bases_size = arith.muli %world_i64, %c8_i64 : i64
    %bases_host = func.call @mgpuGetHeapBases() : () -> !llvm.ptr
    %bases_devptr = func.call @mgpuMemAlloc(%bases_size, %nullptr, %false)
        : (i64, !llvm.ptr, i1) -> !llvm.ptr
    func.call @mgpuMemcpy(%bases_devptr, %bases_host, %bases_size, %nullptr)
        : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
    %bases_bytes = func.call @wrap_bytes(%bases_devptr, %bases_size)
        : (!llvm.ptr, i64) -> memref<?xi8>
    %world_idx = arith.index_cast %world_i64 : i64 to index
    %bases = memref.view %bases_bytes[%c0_view][%world_idx]
        : memref<?xi8> to memref<?xindex>

    // Same kernel on every rank — only the rank-id arg differs.
    %rank_idx = arith.index_cast %rank : i32 to index
    gpu.launch_func @sym_kernels::@allgather
        blocks  in (%c1, %c1, %c1)
        threads in (%c64, %c1, %c1)
        args(%output_m  : memref<64xi32>,
             %verify_m  : memref<64xi32>,
             %rank_idx  : index,
             %world_idx : index,
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

    // Check every (slot, lane) pair: slot's payload should equal
    // slot*1000 + lane+100, except lane 31 which should be the flag (1).
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
          llvm.call @printf(%fmt_fail, %rank, %i_i64, %v, %expected)
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
      // slot[0] sample (data at lane 0, flag at lane 31)
      %v00 = memref.load %hb[%c0_idx] : memref<64xi32>
      %v0f = memref.load %hb[%c31_idx] : memref<64xi32>
      // slot[world-1] sample
      %last_slot     = arith.subi %world_idx, %c1_idx : index
      %last_slot_off = arith.muli %last_slot, %c32_idx : index
      %vL_idx        = arith.addi %last_slot_off, %c0_idx : index
      %vL            = memref.load %hb[%vL_idx] : memref<64xi32>
      %vLf_idx       = arith.addi %last_slot_off, %c31_idx : index
      %vLf           = memref.load %hb[%vLf_idx] : memref<64xi32>
      %last_slot_i32 = arith.index_cast %last_slot : index to i32
      llvm.call @printf(%fmt_p, %rank, %v00, %v0f, %last_slot_i32, %vL, %vLf)
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
    func.call @mgpuSymmetricHeapDestroy() : () -> ()

    %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
    llvm.call @printf(%fmt_done, %rank)
        vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

    return
  }
}
