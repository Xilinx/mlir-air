//===- air_sym_handwritten_cacheline.mlir - multi-GPU e2e (cache line) ----===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Symmetric-heap producer/consumer e2e (WORLD_SIZE=2), cache-line variant.
// Sister file: air_sym_handwritten_atomic.mlir uses LLVM atomicrmw / atomic
// load with syncscope("") for the cross-rank handoff.
//
//   rank 0 launches @producer; rank 1 launches @consumer.
//
// Message-passing via cache-line atomicity (no atomics, no fences)
// ================================================================
//
// Assuming one cache line = 128 bytes = 32 i32:
//
//        ┌─────────────────────────────────────────────────────┐
//        │                  128-byte cache line                │
//        ├────┬────┬────┬────┬─── ··· ───┬────┬───────────────┤
//  lane: │  0 │  1 │  2 │  3 │           │ 30 │  31 ◄── flag  │
//        ├────┼────┼────┼────┤           ├────┼───────────────┤
//  init: │  0 │  0 │  0 │  0 │    0 ···  │  0 │   0           │
//        ├────┼────┼────┼────┤           ├────┼───────────────┤
//  prod: │100 │101 │102 │103 │ lane+100  │130 │   1           │
//        └────┴────┴────┴────┴─── ··· ───┴────┴───────────────┘
//
// Producer (rank 0, 1 wave × 64 lanes):
//   data[lane] = (lane == 31) ? 1 : (lane + 100)   // single vec store
//
// Consumer (rank 1, 1 wave × 64 lanes), spin loop:
//   v    = data[lane]                              // single vec load
//   flag = gpu.shuffle idx v, lane=31, width=64   // broadcast lane 31's val
//   if flag == 1: break, else retry
//
// Why this works on gfx940 / MI300:
//   - Producer's vec-store commits the whole 128-byte cache line as one HW
//     transaction; lane 31's "1" is published with the same coherence event
//     as lanes 0..30's payload (the compiler cannot split a uniform vector
//     store of 32 i32 into per-lane sub-stores).
//   - The XGMI coherence fabric on MI300 publishes peer cache lines whole
//     (not per-lane), so when consumer's lane 31 observes flag==1, lanes
//     0..30 of the same line are guaranteed visible from this load.
//   - shuffle-broadcast of the flag is wave-uniform, so all 64 lanes break
//     in lockstep; no need for control-flow synchronization.
//
// Trade-off vs the previous LLVM-atomic design: this trades a spec-defined
// ordering contract (atomicrmw release / atomic load acquire with
// syncscope("") = AMDGPUUsage System) for a microarchitectural one. It is
// simpler and matches how real GPU code does fast intra-rank handoff, but
// the atomicity guarantee is not in the AMDGPU LangRef the way LLVM atomic
// scopes are.
//
// Note on lanes 32..63: data is sized to one cache line (32 i32), so only
// lanes 0..31 access it. Lanes 32..63 still participate in gpu.shuffle so
// the shuffle stays wave-uniform; their loads are guarded by `lane < 32`.
//
// Launcher: run.sh forks N processes with RANK / WORLD_SIZE / LOCAL_RANK.
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

  // libc exit — verify branch calls this on any mismatch so run.sh
  // sees a non-zero process exit (no green-without-validation).
  func.func private @exit(i32)

  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.mlir.global internal constant @msg_init(
      "[mlir] rank %d / world %d, init OK\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_p(
      "[mlir] rank 0 (producer): kernel returned\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_c(
      "[mlir] rank 1 (consumer): cache-line message PASS (data[0]=%d, flag=%d)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_fail(
      "[mlir] rank 1 (consumer): MISMATCH at idx=%ld got=%d expected=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done(
      "[mlir] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // ---- GPU kernels ------------------------------------------------------
  gpu.module @sym_kernels {

    // Producer: 1 wave × 64 lanes; lanes 0..31 write one cache line into
    // peer's data buffer, with lane 31 == 1 (flag) and lanes 0..30 ==
    // lane+100 (payload). Lanes 32..63 idle.
    gpu.func @producer(%data : memref<32xi32>,
                       %bases : memref<?xindex>) kernel
                       attributes {gpu.known_block_size = array<i32: 64, 1, 1>,
                                   gpu.known_grid_size  = array<i32: 1, 1, 1>} {
      %c1_i32   = arith.constant 1   : i32
      %c100_i32 = arith.constant 100 : i32
      %c31      = arith.constant 31  : index
      %c32      = arith.constant 32  : index
      %from = arith.constant 0 : index   // rank 0 (producer)
      %to   = arith.constant 1 : index   // rank 1 (consumer)

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

    // Consumer: 1 wave × 64 lanes; spin on local data (already peer-mapped
    // by symmetric heap), broadcasting lane 31 via gpu.shuffle until it
    // observes flag==1. Then lanes 0..31 store their loaded value into
    // verify_buf for host check.
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

      // Spin loop: all 64 lanes participate so the shuffle stays uniform.
      // Lanes 32..63 contribute a poison value to the shuffle (shfl reads
      // lane 31, so their input is irrelevant) and do no memory work.
      // The loop's exit predicate is wave-uniform (flag is a broadcast),
      // so all lanes break together.
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

  // ---- Helpers ----------------------------------------------------------
  // Single ABI-leaking helper: wrap a raw runtime !llvm.ptr as a 1-D byte
  // memref. All typed views below derive from this via memref.view, so the
  // hand-built LLVM-struct descriptor literal lives in exactly one place.
  // Phase 4's AIRSymmetricAllocToMgpuPass will replace this entirely.
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
    %c128_bytes  = arith.constant 128 : i64       // 32 i32 = one cache line
    %heap_size   = arith.constant 268435456 : i64 // 256 MB
    %nullptr = llvm.mlir.zero : !llvm.ptr
    %false = arith.constant false

    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    // Heap init (collective).
    func.call @mgpuSymmetricHeapInit(%heap_size) : (i64) -> ()
    %rank = func.call @mgpuGetRank() : () -> i32
    %world = func.call @mgpuGetWorldSize() : () -> i32
    %fmt_init = llvm.mlir.addressof @msg_init : !llvm.ptr
    llvm.call @printf(%fmt_init, %rank, %world)
        vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32

    // Single 128-byte symmetric allocation (32 i32 = one cache line).
    %data_ptr  = func.call @mgpuSymmetricAlloc(%c128_bytes, %nullptr)
        : (i64, !llvm.ptr) -> !llvm.ptr

    // Zero-init data from host so the consumer's spin starts seeing flag=0
    // (and so the validation can distinguish "never written" from "wrote 0").
    %data_host = memref.alloc() : memref<32xi32>
    %dc0 = arith.constant 0 : index
    %dc1 = arith.constant 1 : index
    %dc32 = arith.constant 32 : index
    scf.for %i = %dc0 to %dc32 step %dc1 {
      memref.store %c0_i32, %data_host[%i] : memref<32xi32>
    }
    %data_host_intptr = memref.extract_aligned_pointer_as_index %data_host
        : memref<32xi32> -> index
    %data_host_int = arith.index_cast %data_host_intptr : index to i64
    %data_host_ptr = llvm.inttoptr %data_host_int : i64 to !llvm.ptr
    func.call @mgpuMemcpy(%data_ptr, %data_host_ptr, %c128_bytes, %nullptr)
        : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
    memref.dealloc %data_host : memref<32xi32>

    func.call @mgpuBarrier() : () -> ()  // zero-init visible to all ranks

    %c0_view = arith.constant 0 : index
    %data_bytes = func.call @wrap_bytes(%data_ptr, %c128_bytes)
        : (!llvm.ptr, i64) -> memref<?xi8>
    %data_m = memref.view %data_bytes[%c0_view][]
        : memref<?xi8> to memref<32xi32>

    // mgpuGetHeapBases() returns a HOST pointer; GPU can't deref it, so
    // copy to device. TODO(airgpu): make heap_bases device-accessible
    // (hipMallocManaged / hipHostMalloc-Mapped) and drop this copy.
    %world_i64 = arith.extui %world : i32 to i64
    %c8_i64 = arith.constant 8 : i64
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

    // Rank 0 = producer, rank 1 = consumer. Ranks > 1 idle.
    // (Future: extend to all-pairs producer/consumer mesh.)
    // Precondition: world >= 2 — enforced by run.sh, not re-checked here.
    %is_producer = arith.cmpi eq, %rank, %c0_i32 : i32
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
      %is_consumer = arith.cmpi eq, %rank, %c1_i32 : i32
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

        // D2H readback verify_buf and check all 32 ints:
        //   verify[i] == i + 100 for i in 0..30,
        //   verify[31] == 1 (flag).
        %hb = memref.alloc() : memref<32xi32>
        %hb_intptr = memref.extract_aligned_pointer_as_index %hb
            : memref<32xi32> -> index
        %hb_int = arith.index_cast %hb_intptr : index to i64
        %hb_ptr = llvm.inttoptr %hb_int : i64 to !llvm.ptr
        func.call @mgpuMemcpy(%hb_ptr, %verify_ptr, %c128_bytes, %nullptr)
            : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

        %c0_idx   = arith.constant 0   : index
        %c1_idx   = arith.constant 1   : index
        %c31_idx  = arith.constant 31  : index
        %c32_idx  = arith.constant 32  : index
        %c100_i32 = arith.constant 100 : i32

        // Count mismatches; print msg_fail on the first.
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
    func.call @mgpuSymmetricHeapDestroy() : () -> ()

    %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
    llvm.call @printf(%fmt_done, %rank)
        vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

    return
  }
}
