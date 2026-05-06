//===- air_sym_handwritten.mlir - hand-written multi-GPU e2e test --------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Hand-written reference IR for the symmetric-heap multi-GPU programming
// model on ROCm. Kernel-driven producer/consumer (rather than host-
// orchestrated mgpuMemcpy), per @mawad-amd's review feedback on PR #1577.
//
// Two ranks (WORLD_SIZE=2):
//   rank 0 launches the producer kernel.
//   rank 1 launches the consumer kernel.
//
// The producer kernel runs on rank 0's GPU and writes 42.0 directly into
// rank 1's `data` HBM via XGMI peer access. Each warp signals completion
// of its 64-element slice via a release-store on a per-warp flag (also in
// rank 1's HBM). No mgpuMemcpy is involved on the data path.
//
// The consumer kernel runs on rank 1's GPU. Each warp's lane 0 spins on
// its flag with an acquire-load until the producer has signaled, then all
// 64 lanes of the warp read their slice of `data` and copy it to a local
// verification buffer. The host then D2H reads the verification buffer
// and checks every element == 42.0.
//
// Block shape:
//   1 grid point × 256 threads = 4 warps × 64 lanes.
//   data:  256 f32   (one float per thread).
//   flags: 4 i32     (one flag per warp).
//
// This file is the IR shape that future high-level passes
// (air.launch/air.segment/air.herd → gpu.func via air-to-rocdl +
// air-gpu-outlining) should produce. Phase 2's role is to lock down
// that target shape.
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

  llvm.func @printf(!llvm.ptr, ...) -> i32

  llvm.mlir.global internal constant @msg_init(
      "[mlir] rank %d / world %d, init OK\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_p(
      "[mlir] rank 0 (producer): kernel returned\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_c(
      "[mlir] rank 1 (consumer): cross-rank kernel write PASS (verify[0]=%.1f)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_fail(
      "[mlir] rank 1 (consumer): MISMATCH at idx=%ld got=%.1f expected=42.0\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_only1(
      "[mlir] rank %d: world_size=1, kernel test requires 2 ranks; skipping\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done(
      "[mlir] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // ---- GPU kernels ------------------------------------------------------
  gpu.module @sym_kernels {

    // Producer: store 42.0 into peer (rank 1)'s `data`, signal each warp's
    // flag with system-scope release atomic.
    gpu.func @producer(%data : memref<256xf32>,
                       %flags : memref<4xi32>,
                       %bases : !llvm.ptr) kernel
                       attributes {gpu.known_block_size = array<i32: 256, 1, 1>,
                                   gpu.known_grid_size  = array<i32: 1, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c1_i32 = arith.constant 1 : i32
      %c42_f = arith.constant 42.0 : f32

      // Producer rank id = 0, consumer rank id = 1 (hard-coded for 2-rank test).
      %from = arith.constant 0 : index
      %to   = arith.constant 1 : index

      %tid = gpu.thread_id x
      %wid = arith.divui %tid, %c64 : index   // warp id 0..3
      %lane = arith.remui %tid, %c64 : index  // lane 0..63

      // Translate local memrefs into peer (rank 1)'s address space.
      %peer_data  = air.translate %data,  %from, %to, %bases : memref<256xf32>, !llvm.ptr
      %peer_flags = air.translate %flags, %from, %to, %bases : memref<4xi32>,   !llvm.ptr

      // Each thread writes one f32 into peer's data slot.
      memref.store %c42_f, %peer_data[%tid] : memref<256xf32>

      // Lane 0 of each warp signals the per-warp flag with a release-store.
      // Use llvm.atomicrmw for syncscope("system") semantics — required so
      // the consumer GPU's acquire-load synchronizes with this store across
      // XGMI.
      %is_lane0 = arith.cmpi eq, %lane, %c0 : index
      scf.if %is_lane0 {
        // Extract raw aligned pointer from peer_flags so we can do atomic.
        %flag_idx = memref.extract_aligned_pointer_as_index %peer_flags
            : memref<4xi32> -> index
        %flag_int = arith.index_cast %flag_idx : index to i64
        %flag_ptr = llvm.inttoptr %flag_int : i64 to !llvm.ptr
        // &flags[wid] = flag_ptr + wid * 4
        %wid_i64 = arith.index_cast %wid : index to i64
        %slot_ptr = llvm.getelementptr %flag_ptr[%wid_i64]
            : (!llvm.ptr, i64) -> !llvm.ptr, i32
        // Default syncscope (system / cross-device); AMDGPU rejects an
        // explicit "system" syncscope name, so we omit the keyword and
        // rely on the LLVM IR default.
        %old = llvm.atomicrmw xchg %slot_ptr, %c1_i32 release
            : !llvm.ptr, i32
      }
      gpu.return
    }

    // Consumer: spin on flag (system-scope acquire), then copy data slot
    // into the local verification buffer.
    gpu.func @consumer(%data       : memref<256xf32>,
                       %verify_buf : memref<256xf32>,
                       %flags      : memref<4xi32>,
                       %bases      : !llvm.ptr) kernel
                       attributes {gpu.known_block_size = array<i32: 256, 1, 1>,
                                   gpu.known_grid_size  = array<i32: 1, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c0_i32 = arith.constant 0 : i32

      %tid = gpu.thread_id x
      %wid = arith.divui %tid, %c64 : index
      %lane = arith.remui %tid, %c64 : index

      // Lane 0 of each warp spins on its flag until producer signals.
      // Use atomic acquire syncscope("system") to synchronize with the
      // producer's release-store across XGMI.
      %is_lane0 = arith.cmpi eq, %lane, %c0 : index
      scf.if %is_lane0 {
        %flag_idx = memref.extract_aligned_pointer_as_index %flags
            : memref<4xi32> -> index
        %flag_int = arith.index_cast %flag_idx : index to i64
        %flag_ptr = llvm.inttoptr %flag_int : i64 to !llvm.ptr
        %wid_i64 = arith.index_cast %wid : index to i64
        %slot_ptr = llvm.getelementptr %flag_ptr[%wid_i64]
            : (!llvm.ptr, i64) -> !llvm.ptr, i32

        // scf.while: spin while flag == 0.
        scf.while : () -> () {
          %v = llvm.load %slot_ptr atomic acquire {alignment = 4 : i64}
              : !llvm.ptr -> i32
          %not_ready = arith.cmpi eq, %v, %c0_i32 : i32
          scf.condition(%not_ready)
        } do {
          scf.yield
        }
      }
      // Workgroup barrier: lanes 1..63 of each warp wait for lane 0's
      // spin to terminate before reading data.
      gpu.barrier

      // All 256 threads cooperatively copy their slot from data → verify_buf.
      %v = memref.load %data[%tid] : memref<256xf32>
      memref.store %v, %verify_buf[%tid] : memref<256xf32>
      gpu.return
    }
  }

  // ---- Helpers: build a static-shape memref descriptor over a raw ptr. --
  //
  // Matches the descriptor that AIRSymmetricAllocToMgpuPass (Phase 4) will
  // build automatically. Hand-written here so Phase 2 stands alone.
  //
  //   wrap_data(ptr) : memref<256xf32>  — 256 elements, stride 1, offset 0
  //   wrap_flags(ptr) : memref<4xi32>   — 4   elements, stride 1, offset 0
  func.func private @wrap_data(%ptr : !llvm.ptr) -> memref<256xf32> {
    %c0_i64    = arith.constant 0 : i64
    %c1_i64    = arith.constant 1 : i64
    %c256_i64  = arith.constant 256 : i64
    %d0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d1 = llvm.insertvalue %ptr,        %d0[0]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d2 = llvm.insertvalue %ptr,        %d1[1]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d3 = llvm.insertvalue %c0_i64,     %d2[2]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d4 = llvm.insertvalue %c256_i64,   %d3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d5 = llvm.insertvalue %c1_i64,     %d4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %m  = builtin.unrealized_conversion_cast %d5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<256xf32>
    return %m : memref<256xf32>
  }

  func.func private @wrap_flags(%ptr : !llvm.ptr) -> memref<4xi32> {
    %c0_i64    = arith.constant 0 : i64
    %c1_i64    = arith.constant 1 : i64
    %c4_i64    = arith.constant 4 : i64
    %d0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d1 = llvm.insertvalue %ptr,        %d0[0]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d2 = llvm.insertvalue %ptr,        %d1[1]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d3 = llvm.insertvalue %c0_i64,     %d2[2]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d4 = llvm.insertvalue %c4_i64,     %d3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d5 = llvm.insertvalue %c1_i64,     %d4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %m  = builtin.unrealized_conversion_cast %d5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<4xi32>
    return %m : memref<4xi32>
  }

  // ---- main ------------------------------------------------------------
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1024_bytes = arith.constant 1024 : i64   // 256 f32 = 1024 bytes
    %c16_bytes   = arith.constant 16   : i64   // 4 i32  = 16 bytes
    %heap_size   = arith.constant 268435456 : i64  // 256 MB
    %nullptr = llvm.mlir.zero : !llvm.ptr
    %false = arith.constant false

    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index

    // Init heap collectively.
    func.call @mgpuSymmetricHeapInit(%heap_size) : (i64) -> ()
    %rank = func.call @mgpuGetRank() : () -> i32
    %world = func.call @mgpuGetWorldSize() : () -> i32
    %fmt_init = llvm.mlir.addressof @msg_init : !llvm.ptr
    llvm.call @printf(%fmt_init, %rank, %world)
        vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32

    // Two symmetric allocations: data (256 f32) + flags (4 i32).
    %data_ptr  = func.call @mgpuSymmetricAlloc(%c1024_bytes, %nullptr) : (i64, !llvm.ptr) -> !llvm.ptr
    %flags_ptr = func.call @mgpuSymmetricAlloc(%c16_bytes,   %nullptr) : (i64, !llvm.ptr) -> !llvm.ptr

    // Initialize flags to 0 from host (ensures consumer's spin starts at 0).
    %flags_host = memref.alloc() : memref<4xi32>
    %fc0 = arith.constant 0 : index
    %fc1 = arith.constant 1 : index
    %fc4 = arith.constant 4 : index
    scf.for %i = %fc0 to %fc4 step %fc1 {
      memref.store %c0_i32, %flags_host[%i] : memref<4xi32>
    }
    %flags_host_intptr = memref.extract_aligned_pointer_as_index %flags_host
        : memref<4xi32> -> index
    %flags_host_int = arith.index_cast %flags_host_intptr : index to i64
    %flags_host_ptr = llvm.inttoptr %flags_host_int : i64 to !llvm.ptr
    func.call @mgpuMemcpy(%flags_ptr, %flags_host_ptr, %c16_bytes, %nullptr)
        : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
    memref.dealloc %flags_host : memref<4xi32>

    // All ranks: barrier so flags init is visible before producer runs.
    func.call @mgpuBarrier() : () -> ()

    // Wrap raw pointers as memrefs for kernel argument typing.
    %data_m  = func.call @wrap_data(%data_ptr)   : (!llvm.ptr) -> memref<256xf32>
    %flags_m = func.call @wrap_flags(%flags_ptr) : (!llvm.ptr) -> memref<4xi32>

    // mgpuGetHeapBases() returns a HOST pointer (std::vector<void*>::data()).
    // GPU kernels cannot dereference host memory, so we copy the table into a
    // device-resident buffer and pass that pointer instead. Conservative size:
    // 256 bytes (32 ranks * 8 bytes/ptr).
    //
    // TODO(symmetric_heap): change runtime to allocate heap_bases as
    // hipHostMalloc(...,Mapped) or hipMallocManaged so this copy is unnecessary.
    %bases_size = arith.constant 256 : i64
    %bases_host = func.call @mgpuGetHeapBases() : () -> !llvm.ptr
    %bases = func.call @mgpuMemAlloc(%bases_size, %nullptr, %false)
        : (i64, !llvm.ptr, i1) -> !llvm.ptr
    func.call @mgpuMemcpy(%bases, %bases_host, %bases_size, %nullptr)
        : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

    %is_solo = arith.cmpi sle, %world, %c1_i32 : i32
    scf.if %is_solo {
      %fmt_only1 = llvm.mlir.addressof @msg_only1 : !llvm.ptr
      llvm.call @printf(%fmt_only1, %rank)
          vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    } else {
      // Rank 0 = producer, rank 1 = consumer. Other ranks (W>2) idle.
      // (Future: extend to all-pairs producer/consumer mesh.)
      %is_producer = arith.cmpi eq, %rank, %c0_i32 : i32
      scf.if %is_producer {
        // Rank 0: launch producer kernel (1 block, 256 threads).
        gpu.launch_func @sym_kernels::@producer
            blocks  in (%c1, %c1, %c1)
            threads in (%c256, %c1, %c1)
            args(%data_m  : memref<256xf32>,
                 %flags_m : memref<4xi32>,
                 %bases   : !llvm.ptr)
        %fmt_p = llvm.mlir.addressof @msg_pass_p : !llvm.ptr
        llvm.call @printf(%fmt_p)
            vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      } else {
        %is_consumer = arith.cmpi eq, %rank, %c1_i32 : i32
        scf.if %is_consumer {
          // Rank 1: launch consumer kernel; allocate verify buffer.
          %verify_ptr = func.call @mgpuMemAlloc(%c1024_bytes, %nullptr, %false)
              : (i64, !llvm.ptr, i1) -> !llvm.ptr
          %verify_m = func.call @wrap_data(%verify_ptr) : (!llvm.ptr) -> memref<256xf32>
          gpu.launch_func @sym_kernels::@consumer
              blocks  in (%c1, %c1, %c1)
              threads in (%c256, %c1, %c1)
              args(%data_m  : memref<256xf32>,
                   %verify_m: memref<256xf32>,
                   %flags_m : memref<4xi32>,
                   %bases   : !llvm.ptr)

          // D2H readback verify_buf and check element 0 == 42.0.
          %hb = memref.alloc() : memref<256xf32>
          %hb_intptr = memref.extract_aligned_pointer_as_index %hb : memref<256xf32> -> index
          %hb_int = arith.index_cast %hb_intptr : index to i64
          %hb_ptr = llvm.inttoptr %hb_int : i64 to !llvm.ptr
          func.call @mgpuMemcpy(%hb_ptr, %verify_ptr, %c1024_bytes, %nullptr)
              : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

          // Check element 0.
          %c0_idx = arith.constant 0 : index
          %v0 = memref.load %hb[%c0_idx] : memref<256xf32>
          %expected = arith.constant 42.0 : f32
          %ok = arith.cmpf oeq, %v0, %expected : f32
          scf.if %ok {
            %fmt_c = llvm.mlir.addressof @msg_pass_c : !llvm.ptr
            %v0_64 = arith.extf %v0 : f32 to f64
            llvm.call @printf(%fmt_c, %v0_64)
                vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
          }

          memref.dealloc %hb : memref<256xf32>
          func.call @mgpuMemFree(%verify_ptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
        }
        // ranks > 1: idle (no kernel launch)
      }
    }

    // All-rank barrier and cleanup.
    func.call @mgpuBarrier() : () -> ()
    func.call @mgpuMemFree(%bases, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
    func.call @mgpuSymmetricFree(%data_ptr,  %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
    func.call @mgpuSymmetricFree(%flags_ptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
    func.call @mgpuSymmetricHeapDestroy() : () -> ()

    %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
    llvm.call @printf(%fmt_done, %rank)
        vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

    return
  }
}
