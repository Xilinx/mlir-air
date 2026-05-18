//===- air_channel/cacheline.mlir - air.channel-based cacheline -----------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------===//
//
// Cacheline cross-rank message using air.channel.put / air.channel.get
// (channel_type = "gpu_symmetric_heap") inside air.herd bodies, lowered
// by the phase-6 pass `air-gpu-channel-to-cacheline` into the same
// kernel-driven cacheline shape that ../air_hierarchy/cacheline.mlir
// writes by hand.
//
// Lowering chain:
//   air-gpu-channel-to-cacheline   # phase 6 — expand channel.put/get
//   air-rank-to-mgpu               # phase 3 — inline air.rank
//   air-translate-to-llvm          # phase 2 — expand air.translate
//   air-to-rocdl + air-gpu-outlining  # existing — herd → gpu.func
//   <standard MLIR GPU pipeline>   # convert-gpu-to-rocdl, ...
//
// After the chain runs, the resulting IR is functionally equivalent to
// air_hierarchy/cacheline.mlir.
//
// Notes on the producer fill:
//   The handwritten + air_hierarchy variants compute the payload inline
//   inside the cooperative store (`%val = select is_flag, 1, tx + 100`).
//   With air.channel.put the payload must already be in the source
//   memref — channel.put publishes a memref's contents. So the producer
//   herd first writes the payload to its local %hdata[lane] for lanes
//   0..30, then issues the put. The put expansion stores src[lane] to
//   peer[lane] for lanes 0..30 and the sync flag = 1 at lane 31.
//
//===-----------------------------------------------------------------------===//

module attributes {gpu.container_module} {
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
      "[mlir/chan] rank %d / world %d, init OK\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_p(
      "[mlir/chan] rank 0 (producer): kernel returned\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_pass_c(
      "[mlir/chan] rank 1 (consumer): cache-line message PASS (data[0]=%d, flag=%d)\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_fail(
      "[mlir/chan] rank %d: MISMATCH at idx=%ld got=%d expected=%d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @msg_done(
      "[mlir/chan] rank %d: ALL PASSED\0A\00") {addr_space = 0 : i32}

  // Channel declared at module scope. Single-wire bundle [1] for the
  // 1-producer / 1-consumer cacheline pattern.
  air.channel @C [1] {channel_type = "gpu_symmetric_heap"}

  // Wrap a raw runtime pointer + byte length as a
  // memref<?xi8, #air.symmetric_heap> so subsequent memref.view ops
  // can project typed views that inherit the tag (which makes the herd
  // verifier accept direct loads/stores on the resulting views).
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

  func.func @main() {
    %c2 = arith.constant 2 : index

    air.rank (%rid) in (%rsize = %c2) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c100_i32 = arith.constant 100 : i32
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

      // Rank dispatch. Producer = rank 0, Consumer = rank 1.
      %is_producer = arith.cmpi eq, %rid, %c0_idx : index
      scf.if %is_producer {
        // ---- Producer: fill local data, then channel.put ---------------
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
              %c100_h  = arith.constant 100 : i32
              %c31_h   = arith.constant 31  : index
              %c32_h   = arith.constant 32  : index
              %c0_idx_h = arith.constant 0  : index

              // Fill producer's local %hdata[lane] with payload `lane + 100`
              // for lanes 0..30; leave lane 31 alone (the put expansion
              // overwrites it with the sync flag).
              %active = arith.cmpi ult, %tx, %c32_h : index
              %not_flag = arith.cmpi ne, %tx, %c31_h : index
              %do_fill = arith.andi %active, %not_flag : i1
              scf.if %do_fill {
                %tid_i32  = arith.index_cast %tx : index to i32
                %payload  = arith.addi %tid_i32, %c100_h : i32
                memref.store %payload, %hdata[%tx]
                    : memref<32xi32, #air.symmetric_heap>
              }

              // Publish to consumer (rank 1).
              air.channel.put @C[%c0_idx_h] (%hdata[][][])
                  : (memref<32xi32, #air.symmetric_heap>)
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
          // ---- Consumer: channel.get spins until producer publishes ----
          %c_c1 = arith.constant 1 : index
          air.launch (%bx) in (%nx = %c_c1)
              args(%ldata = %data_m, %lbases = %bases)
              : memref<32xi32, #air.symmetric_heap>,
                memref<?xindex, #air.symmetric_heap> {
            air.segment args(%sdata = %ldata, %sbases = %lbases)
                : memref<32xi32, #air.symmetric_heap>,
                  memref<?xindex, #air.symmetric_heap> {
              %c_c1_s = arith.constant 1 : index
              %c_c64_s = arith.constant 64 : index
              air.herd tile (%tx, %ty) in (%ntx = %c_c64_s, %nty = %c_c1_s)
                  args(%hdata = %sdata, %hbases = %sbases)
                  : memref<32xi32, #air.symmetric_heap>,
                    memref<?xindex, #air.symmetric_heap> {
                %c0_idx_h = arith.constant 0 : index
                // Wait until producer's writes (incl. flag at lane 31) arrive.
                air.channel.get @C[%c0_idx_h] (%hdata[][][])
                    : (memref<32xi32, #air.symmetric_heap>)
                air.herd_terminator
              }
              air.segment_terminator
            }
            air.launch_terminator
          }

          // D2H readback the symmetric-heap data buffer and check all 32 ints.
          %hb = memref.alloc() : memref<32xi32>
          %hb_intptr = memref.extract_aligned_pointer_as_index %hb
              : memref<32xi32> -> index
          %hb_int = arith.index_cast %hb_intptr : index to i64
          %hb_ptr = llvm.inttoptr %hb_int : i64 to !llvm.ptr
          func.call @mgpuMemcpy(%hb_ptr, %data_ptr, %c128_bytes, %nullptr)
              : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> ()

          %c31_idx  = arith.constant 31  : index
          %c32_idx  = arith.constant 32  : index

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
