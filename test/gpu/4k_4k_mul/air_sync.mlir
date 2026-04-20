//===- air_sync.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 4)>
module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @str0("Output match = %d\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str1("Val = %f:%f\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @str2("Input = %d:%d\0A\00") {addr_space = 0 : i32}
  llvm.func @mgpuStreamCreate() -> !llvm.ptr
  llvm.func @mgpuStreamDestroy(!llvm.ptr)
  llvm.func @mgpuEventSynchronize(!llvm.ptr)
  llvm.func @mgpuStreamSynchronize(!llvm.ptr)
  llvm.func @mgpuStreamWaitEvent(!llvm.ptr, !llvm.ptr)
  llvm.func @mgpuEventCreate() -> !llvm.ptr
  llvm.func @mgpuEventDestroy(!llvm.ptr)
  llvm.func @mgpuEventRecord(!llvm.ptr, !llvm.ptr)
  llvm.func @mgpuEventElapsedTime(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  llvm.func @mgpuCheckOutput(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64)
  llvm.func @mgpuInit(!llvm.ptr, !llvm.ptr, i64, i64)
  func.func @print_time(%arg0: f32) {
    %0 = llvm.mlir.constant(0 : i32) : i32
    return
  }
  func.func @main() {
    call @test_matmul() : () -> ()
    return
  }
   func.func @test_matmul() {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %alloc = memref.alloc() : memref<4096x4096xf32>
    %alloc_0 = memref.alloc() : memref<4096x4096xf32>
    %alloc_1 = memref.alloc() : memref<4096x4096xf32>
    %alloc_2 = memref.alloc() : memref<4096x4096xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %intptr = memref.extract_aligned_pointer_as_index %alloc : memref<4096x4096xf32> -> index
    %intptr_4 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<4096x4096xf32> -> index
    %2 = arith.index_cast %intptr : index to i64
    %3 = arith.index_cast %intptr_4 : index to i64
    %4 = llvm.inttoptr %2 : i64 to !llvm.ptr
    %5 = llvm.inttoptr %3 : i64 to !llvm.ptr
    %6 = arith.index_cast %c4096 : index to i64
    llvm.call @mgpuInit(%4, %5, %6, %6) : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()
    %memref = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref, %alloc : memref<4096x4096xf32>, memref<4096x4096xf32>
    %memref_5 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_5, %alloc_0 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %memref_6 = gpu.alloc  () : memref<4096x4096xf32>
    gpu.memcpy  %memref_6, %alloc_1 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %7 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr
    %8 = llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    %9 = llvm.call @mgpuEventCreate() : () -> !llvm.ptr
    llvm.call @mgpuEventRecord(%8, %7) : (!llvm.ptr, !llvm.ptr) -> ()
    call @forward(%memref, %memref_5, %memref_6) : (memref<4096x4096xf32>, memref<4096x4096xf32>, memref<4096x4096xf32>) -> ()
    llvm.call @mgpuEventRecord(%9, %7) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.call @mgpuEventSynchronize(%9) : (!llvm.ptr) -> ()
    %c1_i32 = arith.constant 1 : i32
    %10 = llvm.alloca %c1_i32 x f32 : (i32) -> !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %11 = llvm.call @mgpuEventElapsedTime(%10, %8, %9) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    llvm.call @mgpuStreamDestroy(%7) : (!llvm.ptr) -> ()
    llvm.call @mgpuEventDestroy(%8) : (!llvm.ptr) -> ()
    llvm.call @mgpuEventDestroy(%9) : (!llvm.ptr) -> ()
    gpu.memcpy  %alloc_1, %memref_6 : memref<4096x4096xf32>, memref<4096x4096xf32>
    %intptr_7 = memref.extract_aligned_pointer_as_index %alloc_1 : memref<4096x4096xf32> -> index
    %intptr_8 = memref.extract_aligned_pointer_as_index %alloc : memref<4096x4096xf32> -> index
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloc_0 : memref<4096x4096xf32> -> index
    %12 = arith.index_cast %intptr_7 : index to i64
    %13 = arith.index_cast %intptr_8 : index to i64
    %14 = arith.index_cast %intptr_9 : index to i64
    %15 = llvm.inttoptr %12 : i64 to !llvm.ptr
    %16 = llvm.inttoptr %13 : i64 to !llvm.ptr
    %17 = llvm.inttoptr %14 : i64 to !llvm.ptr
    llvm.call @mgpuCheckOutput(%15, %16, %17, %6, %6) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64) -> ()
    return
  }
  func.func @forward(%arg0: memref<4096x4096xf32>, %arg1: memref<4096x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %c32 = arith.constant 32 : index
    air.launch (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<4096x4096xf32>, memref<4096x4096xf32>, memref<4096x4096xf32> {
      air.segment @forward_0  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<4096x4096xf32>, memref<4096x4096xf32>, memref<4096x4096xf32> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %c4096 = arith.constant 4096 : index
        %cst = arith.constant 0.000000e+00 : f32

        %row_off = affine.apply #map()[%arg11]
        %col_off = affine.apply #map()[%arg10]

        %a_reg = memref.alloc() : memref<8xf32, 2>
        %b_reg = memref.alloc() : memref<8xf32, 2>
        %acc = memref.alloc() : memref<64xf32, 2>

        scf.for %i = %c0 to %c64 step %c1 {
          memref.store %cst, %acc[%i] : memref<64xf32, 2>
        }

        scf.for %k = %c0 to %c4096 step %c8 {
          %As = memref.alloc() : memref<128x8xf32, 1>
          %Bs = memref.alloc() : memref<8x128xf32, 1>

          // Phase 1: L3 -> L2 (global -> shared)
          air.dma_memcpy_nd (%As[] [] [], %arg12[%row_off, %k] [%c128, %c8] [%c4096, %c1]) : (memref<128x8xf32, 1>, memref<4096x4096xf32>)
          air.dma_memcpy_nd (%Bs[] [] [], %arg13[%k, %col_off] [%c8, %c128] [%c4096, %c1]) : (memref<8x128xf32, 1>, memref<4096x4096xf32>)

          gpu.barrier

          // Phase 2 + Compute
          air.herd @herd_0 tile (%tx, %ty) in (%ntx=%c256, %nty=%c1) args(%hAs=%As, %hBs=%Bs, %ha=%a_reg, %hb=%b_reg, %hacc=%acc) : memref<128x8xf32, 1>, memref<8x128xf32, 1>, memref<8xf32, 2>, memref<8xf32, 2>, memref<64xf32, 2> {
            %c0_h = arith.constant 0 : index
            %c1_h = arith.constant 1 : index
            %c8_h = arith.constant 8 : index
            %c16_h = arith.constant 16 : index

            // This thread's 8x8 sub-tile within the 128x128 output tile
            // tx in [0..255] -> 16x16 grid of 8x8 tiles
            %tile_row_idx = arith.remsi %tx, %c16_h : index
            %tile_col_idx = arith.divsi %tx, %c16_h : index
            %row_start = arith.muli %tile_row_idx, %c8_h : index
            %col_start = arith.muli %tile_col_idx, %c8_h : index

            scf.for %kk = %c0_h to %c8_h step %c1_h {
              // Phase 2a: L2 -> L1: load column kk from As
              air.dma_memcpy_nd (%ha[] [] [], %hAs[%row_start, %kk] [%c8_h] [%c8_h]) : (memref<8xf32, 2>, memref<128x8xf32, 1>)

              // Phase 2b: L2 -> L1: load row kk from Bs
              air.dma_memcpy_nd (%hb[] [] [], %hBs[%kk, %col_start] [%c8_h] [%c1_h]) : (memref<8xf32, 2>, memref<8x128xf32, 1>)

              // Outer product accumulate (all L1 only)
              scf.for %yt = %c0_h to %c8_h step %c1_h {
                scf.for %xt = %c0_h to %c8_h step %c1_h {
                  %flat = arith.muli %yt, %c8_h : index
                  %idx = arith.addi %flat, %xt : index
                  %av = memref.load %ha[%yt] : memref<8xf32, 2>
                  %bv = memref.load %hb[%xt] : memref<8xf32, 2>
                  %cv = memref.load %hacc[%idx] : memref<64xf32, 2>
                  %prod = arith.mulf %av, %bv : f32
                  %sum = arith.addf %cv, %prod : f32
                  memref.store %sum, %hacc[%idx] : memref<64xf32, 2>
                }
              }
            }
            gpu.barrier
            air.herd_terminator
          }
        }

        // Phase 3: L1 -> L3 writeback
        // Each thread writes its 8x8 sub-tile from acc back to global C
        air.herd @writeback tile (%tx, %ty) in (%ntx=%c256, %nty=%c1) args(%wacc=%acc, %wC=%arg14, %wrow=%row_off, %wcol=%col_off) : memref<64xf32, 2>, memref<4096x4096xf32>, index, index {
          %c0_w = arith.constant 0 : index
          %c8_w = arith.constant 8 : index
          %c16_w = arith.constant 16 : index
          %c4096_w = arith.constant 4096 : index
          %c1_w = arith.constant 1 : index
          %tr = arith.remsi %tx, %c16_w : index
          %tc = arith.divsi %tx, %c16_w : index
          %dr = arith.muli %tr, %c8_w : index
          %dc = arith.muli %tc, %c8_w : index
          %dst_r = arith.addi %wrow, %dr : index
          %dst_c = arith.addi %wcol, %dc : index
          air.dma_memcpy_nd (%wC[%dst_r, %dst_c] [%c8_w, %c8_w] [%c4096_w, %c1_w], %wacc[] [] []) : (memref<4096x4096xf32>, memref<64xf32, 2>)
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
