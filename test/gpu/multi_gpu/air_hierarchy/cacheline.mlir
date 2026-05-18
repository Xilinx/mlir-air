//===- air_hierarchy/cacheline.mlir - air.launch/segment/herd cacheline ---===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// High-level version of air_rank/cacheline.mlir using the full AIR
// compute-model hierarchy (air.launch / air.segment / air.herd) instead
// of explicit gpu.module + gpu.func + gpu.launch_func.
//
// Per docs/AIRComputeModel.md §2.3, on AMD MI3xx the herd is the GPU
// kernel boundary: "A herd executes entirely within a single Compute
// Unit (CU), with PE instances mapped to individual warps." So this
// file's air.herd bodies become gpu.func bodies after lowering through
// the existing air-to-rocdl + air-gpu-outlining passes.
//
// Lowering chain (after this file):
//   air-rank-to-mgpu             # phase 3 — inlines air.rank
//   air-translate-to-llvm        # phase 2 — expands air.translate
//   air-to-rocdl                 # existing — AIR ops → ROCDL primitives
//   air-gpu-outlining            # existing — air.launch/segment/herd → gpu.func
//   <standard MLIR GPU pipeline> # convert-gpu-to-rocdl, gpu-module-to-binary, etc.
//
// After the chain the IR is functionally equivalent to
// handwritten/cacheline.mlir (same kernel structure, same cache-line
// atomicity, same validation). This file's job is to demonstrate that
// the AIR hierarchy correctly composes with the multi-rank context
// (air.rank wrapping) and produces the handwritten reference.
//
// This is also the planned lowering target for the future phase 6
// pass `air-gpu-channel-to-cacheline` (see docs/MultiGPUPhase56Redesign.md):
// channel.put/get inside air.herd bodies will expand to the same shape
// as the cacheline ops handwritten here.
//
// Pinned to W=2 because air.translate today requires static-shape source
// memref. Same constraint as the handwritten / air_rank / air_alloc
// cacheline tests.
//
// Launcher: `make` from this subdir forks 2 processes.
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
      "[mlir/hier] rank %d / world %d, init OK\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_p(
      "[mlir/hier] rank 0 (producer): kernel returned\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_c(
      "[mlir/hier] rank 1 (consumer): cache-line message PASS (data[0]=%d, flag=%d)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_fail(
      "[mlir/hier] rank 1 (consumer): MISMATCH at idx=%ld got=%d expected=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done(
      "[mlir/hier] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // ---- Helpers ---------------------------------------------------------
  // wrap_bytes produces a #air.symmetric_heap-tagged byte memref so that
  // typed views inherit the tag. The herd verifier accepts loads/stores on
  // memrefs with this memory_space because the backing storage (HBM) is
  // directly addressable from GPU kernels via XGMI.
  func.func private @wrap_bytes(%ptr : !llvm.ptr, %size : i64)
      -> memref<?xi8, #air.symmetric_heap> {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %d0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d1 = llvm.insertvalue %ptr,    %d0[0]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d2 = llvm.insertvalue %ptr,    %d1[1]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d3 = llvm.insertvalue %c0_i64, %d2[2]    : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d4 = llvm.insertvalue %size,   %d3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %d5 = llvm.insertvalue %c1_i64, %d4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %m  = builtin.unrealized_conversion_cast %d5
        : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        to memref<?xi8, #air.symmetric_heap>
    return %m : memref<?xi8, #air.symmetric_heap>
  }

  // ---- main: cacheline test using AIR hierarchy ------------------------
  func.func @main() {
    %c2 = arith.constant 2 : index

    air.rank (%rid) in (%rsize = %c2) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c128_bytes = arith.constant 128 : i64
      %nullptr = llvm.mlir.zero : !llvm.ptr
      %false = arith.constant false

      %c0_idx = arith.constant 0 : index
      %c1_idx = arith.constant 1 : index
      %c1 = arith.constant 1 : index

      %rid_i64 = arith.index_cast %rid : index to i64
      %rid_i32 = arith.trunci %rid_i64 : i64 to i32
      %rsize_i64 = arith.index_cast %rsize : index to i64
      %rsize_i32 = arith.trunci %rsize_i64 : i64 to i32

      %fmt_init = llvm.mlir.addressof @msg_init : !llvm.ptr
      llvm.call @printf(%fmt_init, %rid_i32, %rsize_i32)
          vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32, i32) -> i32

      // Symmetric data buffer.
      %data_ptr = func.call @mgpuSymmetricAlloc(%c128_bytes, %nullptr)
          : (i64, !llvm.ptr) -> !llvm.ptr

      // Zero-init data from host so spins start at flag=0.
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
          : (!llvm.ptr, i64) -> memref<?xi8, #air.symmetric_heap>
      %data_m = memref.view %data_bytes[%c0_view][]
          : memref<?xi8, #air.symmetric_heap>
          to memref<32xi32, #air.symmetric_heap>

      // heap_bases (host ptr → device copy).
      %c8_i64 = arith.constant 8 : i64
      %bases_size = arith.muli %rsize_i64, %c8_i64 : i64
      %bases_host = func.call @mgpuGetHeapBases() : () -> !llvm.ptr
      %bases_devptr = func.call @mgpuMemAlloc(%bases_size, %nullptr, %false)
          : (i64, !llvm.ptr, i1) -> !llvm.ptr
      func.call @mgpuMemcpy(%bases_devptr, %bases_host, %bases_size, %nullptr)
          : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()
      %bases_bytes = func.call @wrap_bytes(%bases_devptr, %bases_size)
          : (!llvm.ptr, i64) -> memref<?xi8, #air.symmetric_heap>
      %bases = memref.view %bases_bytes[%c0_view][%rsize]
          : memref<?xi8, #air.symmetric_heap>
          to memref<?xindex, #air.symmetric_heap>

      // Rank dispatch. Producer: rank 0; consumer: rank 1.
      %is_producer = arith.cmpi eq, %rid, %c0_idx : index
      scf.if %is_producer {
        // ---- Producer: AIR hierarchy nest -----------------------------
        // 1 block, 1 segment, 64 herd PEs (one full MI3xx wavefront).
        // Per AIRComputeModel.md §4.1 the herd iteration space maps
        // directly to gpu blockDim and herd tile ids to thread ids — so
        // `air.herd tile (%tx, %ty) in (%ntx=64, %nty=1)` lowers to
        // `gpu.launch threads in (64,1,1)` with `%tx = threadIdx.x`.
        // We use 64 (the wavefront size) so the consumer's
        // `gpu.shuffle width=64` can see every producer lane.
        // Only lanes 0..31 do real work; lanes 32..63 are idle but live.
        // Declare size constants locally to avoid cross-scf.if-region uses
        // that the air.rank inliner doesn't track through nested cloning.
        %p_c1 = arith.constant 1 : index
        air.launch (%bx) in (%nx = %p_c1)
            args(%ldata = %data_m, %lbases = %bases)
            : memref<32xi32, #air.symmetric_heap>,
              memref<?xindex, #air.symmetric_heap> {
          air.segment args(%sdata = %ldata, %sbases = %lbases)
              : memref<32xi32, #air.symmetric_heap>,
                memref<?xindex, #air.symmetric_heap> {
            %p_c1_s = arith.constant 1 : index
            %p_c64_s = arith.constant 64 : index
            air.herd tile (%tx, %ty) in (%ntx = %p_c64_s, %nty = %p_c1_s)
                args(%hdata = %sdata, %hbases = %sbases)
                : memref<32xi32, #air.symmetric_heap>,
                  memref<?xindex, #air.symmetric_heap> {
              %c1_i32_h   = arith.constant 1   : i32
              %c100_i32_h = arith.constant 100 : i32
              %c31_h      = arith.constant 31  : index
              %c32_h      = arith.constant 32  : index
              %from = arith.constant 0 : index
              %to   = arith.constant 1 : index

              %active = arith.cmpi ult, %tx, %c32_h : index
              %peer_data = air.translate %hdata, %from, %to, %hbases
                  : memref<32xi32, #air.symmetric_heap>,
                    memref<?xindex, #air.symmetric_heap>

              scf.if %active {
                %is_flag  = arith.cmpi eq, %tx, %c31_h : index
                %tid_i32  = arith.index_cast %tx : index to i32
                %payload  = arith.addi %tid_i32, %c100_i32_h : i32
                %val      = arith.select %is_flag, %c1_i32_h, %payload : i32
                memref.store %val, %peer_data[%tx]
                    : memref<32xi32, #air.symmetric_heap>
              }
              air.herd_terminator
            }
            air.segment_terminator
          }
          air.launch_terminator
        }
        %fmt_p = llvm.mlir.addressof @msg_pass_p : !llvm.ptr
        llvm.call @printf(%fmt_p)
            vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
      } else {
        %is_consumer = arith.cmpi eq, %rid, %c1_idx : index
        scf.if %is_consumer {
          %verify_ptr = func.call @mgpuMemAlloc(%c128_bytes, %nullptr, %false)
              : (i64, !llvm.ptr, i1) -> !llvm.ptr
          %verify_bytes = func.call @wrap_bytes(%verify_ptr, %c128_bytes)
              : (!llvm.ptr, i64) -> memref<?xi8, #air.symmetric_heap>
          %verify_m = memref.view %verify_bytes[%c0_view][]
              : memref<?xi8, #air.symmetric_heap>
              to memref<32xi32, #air.symmetric_heap>

          // ---- Consumer: AIR hierarchy nest ---------------------------
          // Same shape as producer: 1×1 launch, 64×1 herd → blockDim
          // (64,1,1), one full wavefront so that the gpu.shuffle below
          // (width=64) can read the producer's flag lane.
          %c_c1 = arith.constant 1 : index
          air.launch (%bx) in (%nx = %c_c1)
              args(%ldata = %data_m, %lvb = %verify_m)
              : memref<32xi32, #air.symmetric_heap>,
                memref<32xi32, #air.symmetric_heap> {
            air.segment args(%sdata = %ldata, %svb = %lvb)
                : memref<32xi32, #air.symmetric_heap>,
                  memref<32xi32, #air.symmetric_heap> {
              %c_c1_s = arith.constant 1 : index
              %c_c64_s = arith.constant 64 : index
              air.herd tile (%tx, %ty) in (%ntx = %c_c64_s, %nty = %c_c1_s)
                  args(%hdata = %sdata, %hvb = %svb)
                  : memref<32xi32, #air.symmetric_heap>,
                    memref<32xi32, #air.symmetric_heap> {
                %c0_i32_h  = arith.constant 0  : i32
                %c1_i32_h  = arith.constant 1  : i32
                %c31_i32_h = arith.constant 31 : i32
                %c64_i32_h = arith.constant 64 : i32
                %c32_h     = arith.constant 32 : index

                %active = arith.cmpi ult, %tx, %c32_h : index

                %final_v = scf.while (%dummy = %c0_i32_h) : (i32) -> i32 {
                  %v = scf.if %active -> i32 {
                    %loaded = memref.load %hdata[%tx]
                        : memref<32xi32, #air.symmetric_heap>
                    scf.yield %loaded : i32
                  } else {
                    scf.yield %c0_i32_h : i32
                  }
                  %flag, %valid = gpu.shuffle idx %v, %c31_i32_h, %c64_i32_h : i32
                  %not_ready = arith.cmpi ne, %flag, %c1_i32_h : i32
                  scf.condition(%not_ready) %v : i32
                } do {
                ^bb0(%v_iter : i32):
                  scf.yield %v_iter : i32
                }

                scf.if %active {
                  memref.store %final_v, %hvb[%tx]
                      : memref<32xi32, #air.symmetric_heap>
                }
                air.herd_terminator
              }
              air.segment_terminator
            }
            air.launch_terminator
          }

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
