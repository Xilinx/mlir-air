//===- air_sym_handwritten.mlir - hand-written multi-GPU e2e test --------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Symmetric-heap producer/consumer e2e (WORLD_SIZE=2):
//   rank 0 launches @producer; rank 1 launches @consumer.
//   producer writes 42.0 into rank 1's `data` over XGMI; per-warp flags
//   (4 i32, in rank 1's HBM) signal completion via release atomicrmw.
//   consumer's lane 0 acquires on its flag, then all 64 lanes copy
//   the local data slot to verify_buf for host check.
//   Block: 1 grid × 256 threads = 4 warps × 64 lanes.
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
      "[mlir] rank 1 (consumer): cross-rank kernel write PASS (verify[0]=%.1f)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_fail(
      "[mlir] rank 1 (consumer): MISMATCH at idx=%ld got=%.1f expected=42.0\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_only1(
      "[mlir] rank %d: world_size=1, kernel test requires 2 ranks; skipping\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done(
      "[mlir] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // ---- GPU kernels ------------------------------------------------------
  gpu.module @sym_kernels {

    // Producer: each thread stores 42.0 into peer's data; lane 0 of each
    // warp release-atomicrmws peer's per-warp flag.
    gpu.func @producer(%data : memref<256xf32>,
                       %flags : memref<4xi32>,
                       %bases : memref<?xindex>) kernel
                       attributes {gpu.known_block_size = array<i32: 256, 1, 1>,
                                   gpu.known_grid_size  = array<i32: 1, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1_i32 = arith.constant 1 : i32
      %c42_f = arith.constant 42.0 : f32
      %from = arith.constant 0 : index   // rank 0 (producer)
      %to   = arith.constant 1 : index   // rank 1 (consumer)

      %tid = gpu.thread_id x
      %wid = arith.divui %tid, %c64 : index
      %lane = arith.remui %tid, %c64 : index

      %peer_data  = air.translate %data,  %from, %to, %bases : memref<256xf32>, memref<?xindex>
      %peer_flags = air.translate %flags, %from, %to, %bases : memref<4xi32>,   memref<?xindex>
      memref.store %c42_f, %peer_data[%tid] : memref<256xf32>

      %is_lane0 = arith.cmpi eq, %lane, %c0 : index
      scf.if %is_lane0 {
        // Drop to llvm.ptr for the atomic — AMDGPU rejects an explicit
        // syncscope("system"); default = LLVM System = cross-device.
        // See sym_atomic_syncscope.mlir for the contract test.
        %flag_idx = memref.extract_aligned_pointer_as_index %peer_flags
            : memref<4xi32> -> index
        %flag_int = arith.index_cast %flag_idx : index to i64
        %flag_ptr = llvm.inttoptr %flag_int : i64 to !llvm.ptr
        %wid_i64 = arith.index_cast %wid : index to i64
        %slot_ptr = llvm.getelementptr %flag_ptr[%wid_i64]
            : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %old = llvm.atomicrmw xchg %slot_ptr, %c1_i32 release
            : !llvm.ptr, i32
      }
      gpu.return
    }

    // Consumer: lane 0 acquires on its flag; then all 64 lanes copy
    // their data slot into verify_buf for host check.
    gpu.func @consumer(%data       : memref<256xf32>,
                       %verify_buf : memref<256xf32>,
                       %flags      : memref<4xi32>) kernel
                       attributes {gpu.known_block_size = array<i32: 256, 1, 1>,
                                   gpu.known_grid_size  = array<i32: 1, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c0_i32 = arith.constant 0 : i32

      %tid = gpu.thread_id x
      %wid = arith.divui %tid, %c64 : index
      %lane = arith.remui %tid, %c64 : index

      %is_lane0 = arith.cmpi eq, %lane, %c0 : index
      scf.if %is_lane0 {
        // Drop to llvm.ptr for the atomic; default = LLVM System scope
        // (cross-device on AMDGPU). See sym_atomic_syncscope.mlir.
        %flag_idx = memref.extract_aligned_pointer_as_index %flags
            : memref<4xi32> -> index
        %flag_int = arith.index_cast %flag_idx : index to i64
        %flag_ptr = llvm.inttoptr %flag_int : i64 to !llvm.ptr
        %wid_i64 = arith.index_cast %wid : index to i64
        %slot_ptr = llvm.getelementptr %flag_ptr[%wid_i64]
            : (!llvm.ptr, i64) -> !llvm.ptr, i32
        // Spin: flag == 0.
        scf.while : () -> () {
          %v = llvm.load %slot_ptr atomic acquire {alignment = 4 : i64}
              : !llvm.ptr -> i32
          %not_ready = arith.cmpi eq, %v, %c0_i32 : i32
          scf.condition(%not_ready)
        } do {
          scf.yield
        }
      }
      gpu.barrier  // lanes 1..63 wait for lane 0's spin to terminate
      %v = memref.load %data[%tid] : memref<256xf32>
      memref.store %v, %verify_buf[%tid] : memref<256xf32>
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
    %c1024_bytes = arith.constant 1024 : i64   // 256 f32 = 1024 bytes
    %c16_bytes   = arith.constant 16   : i64   // 4 i32  = 16 bytes
    %heap_size   = arith.constant 268435456 : i64  // 256 MB
    %nullptr = llvm.mlir.zero : !llvm.ptr
    %false = arith.constant false

    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index

    // Heap init (collective).
    func.call @mgpuSymmetricHeapInit(%heap_size) : (i64) -> ()
    %rank = func.call @mgpuGetRank() : () -> i32
    %world = func.call @mgpuGetWorldSize() : () -> i32
    %fmt_init = llvm.mlir.addressof @msg_init : !llvm.ptr
    llvm.call @printf(%fmt_init, %rank, %world)
        vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32

    // Symmetric allocations: data (256 f32) + flags (4 i32).
    %data_ptr  = func.call @mgpuSymmetricAlloc(%c1024_bytes, %nullptr) : (i64, !llvm.ptr) -> !llvm.ptr
    %flags_ptr = func.call @mgpuSymmetricAlloc(%c16_bytes,   %nullptr) : (i64, !llvm.ptr) -> !llvm.ptr

    // Zero-init flags from host so the consumer's spin starts at 0.
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

    func.call @mgpuBarrier() : () -> ()  // flags init visible to all ranks

    %c0_view = arith.constant 0 : index
    %data_bytes  = func.call @wrap_bytes(%data_ptr,  %c1024_bytes) : (!llvm.ptr, i64) -> memref<?xi8>
    %flags_bytes = func.call @wrap_bytes(%flags_ptr, %c16_bytes)   : (!llvm.ptr, i64) -> memref<?xi8>
    %data_m  = memref.view %data_bytes[%c0_view][]  : memref<?xi8> to memref<256xf32>
    %flags_m = memref.view %flags_bytes[%c0_view][] : memref<?xi8> to memref<4xi32>

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
    %bases_bytes = func.call @wrap_bytes(%bases_devptr, %bases_size) : (!llvm.ptr, i64) -> memref<?xi8>
    %world_idx = arith.index_cast %world_i64 : i64 to index
    %bases = memref.view %bases_bytes[%c0_view][%world_idx] : memref<?xi8> to memref<?xindex>

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
        gpu.launch_func @sym_kernels::@producer
            blocks  in (%c1, %c1, %c1)
            threads in (%c256, %c1, %c1)
            args(%data_m  : memref<256xf32>,
                 %flags_m : memref<4xi32>,
                 %bases   : memref<?xindex>)
        %fmt_p = llvm.mlir.addressof @msg_pass_p : !llvm.ptr
        llvm.call @printf(%fmt_p)
            vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      } else {
        // Rank 1 = consumer; ranks > 1 idle.
        %is_consumer = arith.cmpi eq, %rank, %c1_i32 : i32
        scf.if %is_consumer {
          %verify_ptr = func.call @mgpuMemAlloc(%c1024_bytes, %nullptr, %false)
              : (i64, !llvm.ptr, i1) -> !llvm.ptr
          %verify_bytes = func.call @wrap_bytes(%verify_ptr, %c1024_bytes) : (!llvm.ptr, i64) -> memref<?xi8>
          %verify_m = memref.view %verify_bytes[%c0_view][] : memref<?xi8> to memref<256xf32>
          gpu.launch_func @sym_kernels::@consumer
              blocks  in (%c1, %c1, %c1)
              threads in (%c256, %c1, %c1)
              args(%data_m  : memref<256xf32>,
                   %verify_m: memref<256xf32>,
                   %flags_m : memref<4xi32>)

          // D2H readback verify_buf and check ALL 256 elements == 42.0.
          // (Checking only element 0 would mask a bug where warps 1..3
          // didn't write their slice. exit(1) on mismatch makes the
          // multi-process driver see a non-zero exit code.)
          %hb = memref.alloc() : memref<256xf32>
          %hb_intptr = memref.extract_aligned_pointer_as_index %hb : memref<256xf32> -> index
          %hb_int = arith.index_cast %hb_intptr : index to i64
          %hb_ptr = llvm.inttoptr %hb_int : i64 to !llvm.ptr
          func.call @mgpuMemcpy(%hb_ptr, %verify_ptr, %c1024_bytes, %nullptr)
              : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

          %c0_idx = arith.constant 0 : index
          %c1_idx = arith.constant 1 : index
          %c256_idx = arith.constant 256 : index
          %expected = arith.constant 42.0 : f32

          // Count mismatches; print msg_fail on the first.
          %nfail = scf.for %i = %c0_idx to %c256_idx step %c1_idx
                          iter_args(%nfail_acc = %c0_i32) -> (i32) {
            %v = memref.load %hb[%i] : memref<256xf32>
            %ne = arith.cmpf une, %v, %expected : f32
            %new_nfail = scf.if %ne -> i32 {
              %is_first = arith.cmpi eq, %nfail_acc, %c0_i32 : i32
              scf.if %is_first {
                %fmt_fail = llvm.mlir.addressof @msg_fail : !llvm.ptr
                %i_i64 = arith.index_cast %i : index to i64
                %v_64 = arith.extf %v : f32 to f64
                %e_64 = arith.extf %expected : f32 to f64
                llvm.call @printf(%fmt_fail, %rank, %i_i64, %v_64, %e_64)
                    vararg(!llvm.func<i32 (ptr, ...)>)
                    : (!llvm.ptr, i32, i64, f64, f64) -> i32
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
            %v0 = memref.load %hb[%c0_idx] : memref<256xf32>
            %v0_64 = arith.extf %v0 : f32 to f64
            llvm.call @printf(%fmt_c, %v0_64)
                vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
          } else {
            func.call @exit(%c1_i32) : (i32) -> ()
          }

          memref.dealloc %hb : memref<256xf32>
          func.call @mgpuMemFree(%verify_ptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
        }
      }
    }

    func.call @mgpuBarrier() : () -> ()
    func.call @mgpuMemFree(%bases_devptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
    func.call @mgpuSymmetricFree(%data_ptr,  %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
    func.call @mgpuSymmetricFree(%flags_ptr, %nullptr) : (!llvm.ptr, !llvm.ptr) -> ()
    func.call @mgpuSymmetricHeapDestroy() : () -> ()

    %fmt_done = llvm.mlir.addressof @msg_done : !llvm.ptr
    llvm.call @printf(%fmt_done, %rank)
        vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

    return
  }
}
