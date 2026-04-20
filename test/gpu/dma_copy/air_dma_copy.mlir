//===- air_dma_copy.mlir - DMA data movement e2e test --------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Focused e2e test exercising air.dma_memcpy_nd through all three memory
// levels on GPU:
//
//   L3 (global) -> L2 (shared/LDS) -> L1 (private/VGPRs) -> L3 (global)
//
// A 64x64 input matrix is tiled into 4 blocks of 32x32.  Each workgroup
// copies one 32x32 tile through the full memory hierarchy and writes it
// back to a 64x64 output matrix.  The test harness compares input and
// output element-by-element.
//
//===------------------------------------------------------------------===//

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @pass_msg("PASS: all elements match\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @fail_msg("FAIL at [%ld,%ld]: expected %f, got %f\0A\00") {addr_space = 0 : i32}

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst = arith.constant 0.000000e+00 : f32

    %input = memref.alloc() : memref<64x64xf32>
    %output = memref.alloc() : memref<64x64xf32>

    // Initialize input: val = row * 64 + col (as float)
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        %row = arith.muli %i, %c64 : index
        %flat = arith.addi %row, %j : index
        %flat_i32 = arith.index_cast %flat : index to i32
        %val = arith.sitofp %flat_i32 : i32 to f32
        memref.store %val, %input[%i, %j] : memref<64x64xf32>
        memref.store %cst, %output[%i, %j] : memref<64x64xf32>
      }
    }

    // Allocate GPU buffers and copy input
    %g_in = gpu.alloc () : memref<64x64xf32>
    gpu.memcpy %g_in, %input : memref<64x64xf32>, memref<64x64xf32>
    %g_out = gpu.alloc () : memref<64x64xf32>
    gpu.memcpy %g_out, %output : memref<64x64xf32>, memref<64x64xf32>

    // Run kernel: copy through L3 -> L2 -> L1 -> L3
    call @copy_kernel(%g_in, %g_out) : (memref<64x64xf32>, memref<64x64xf32>) -> ()

    // Copy result back to host
    gpu.memcpy %output, %g_out : memref<64x64xf32>, memref<64x64xf32>

    // Verify
    %pass = memref.alloc() : memref<1xi32>
    %c1_i32 = arith.constant 1 : i32
    memref.store %c1_i32, %pass[%c0] : memref<1xi32>

    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        %expected = memref.load %input[%i, %j] : memref<64x64xf32>
        %actual = memref.load %output[%i, %j] : memref<64x64xf32>
        %diff = arith.subf %expected, %actual : f32
        %abs = math.absf %diff : f32
        %eps = arith.constant 1.0e-6 : f32
        %ok = arith.cmpf "olt", %abs, %eps : f32
        scf.if %ok {
        } else {
          %c0_i32 = arith.constant 0 : i32
          memref.store %c0_i32, %pass[%c0] : memref<1xi32>
          %fmt = llvm.mlir.addressof @fail_msg : !llvm.ptr
          %ii = arith.index_cast %i : index to i64
          %jj = arith.index_cast %j : index to i64
          %e64 = arith.extf %expected : f32 to f64
          %a64 = arith.extf %actual : f32 to f64
          llvm.call @printf(%fmt, %ii, %jj, %e64, %a64) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64, i64, f64, f64) -> i32
        }
      }
    }

    %result = memref.load %pass[%c0] : memref<1xi32>
    %c1_check = arith.constant 1 : i32
    %passed = arith.cmpi "eq", %result, %c1_check : i32
    scf.if %passed {
      %fmt = llvm.mlir.addressof @pass_msg : !llvm.ptr
      llvm.call @printf(%fmt) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
    }

    return
  }

  // Kernel: 2x2 grid of workgroups, each copies a 32x32 tile through
  // L3 -> L2 -> L1 -> L3 using air.dma_memcpy_nd at every level.
  func.func @copy_kernel(%in: memref<64x64xf32>, %out: memref<64x64xf32>) {
    %c2 = arith.constant 2 : index
    air.launch (%bx, %by) in (%nbx=%c2, %nby=%c2)
        args(%arg_in=%in, %arg_out=%out)
        : memref<64x64xf32>, memref<64x64xf32> {

      air.segment @tile_copy
          args(%sbx=%bx, %sby=%by, %sin=%arg_in, %sout=%arg_out)
          : index, index, memref<64x64xf32>, memref<64x64xf32> {

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index

        %row_off = affine.apply #map()[%sby]
        %col_off = affine.apply #map()[%sbx]

        // L2 tile buffer (shared memory)
        %tile_l2 = memref.alloc() : memref<32x32xf32, 1>

        // Phase 1: L3 -> L2  (cooperative DMA)
        air.dma_memcpy_nd (%tile_l2[] [] [],
                           %sin[%row_off, %col_off] [%c32, %c32] [%c64, %c1])
            : (memref<32x32xf32, 1>, memref<64x64xf32>)
        gpu.barrier

        // Phase 2+3: L2 -> L1 -> L3 inside herd
        // 32 threads, each handles one row of the 32x32 tile
        air.herd @copy_herd tile (%tx, %ty) in (%ntx=%c32, %nty=%c1)
            args(%hl2=%tile_l2, %hout=%sout, %hrow=%row_off, %hcol=%col_off)
            : memref<32x32xf32, 1>, memref<64x64xf32>, index, index {

          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c32_h = arith.constant 32 : index
          %c64_h = arith.constant 64 : index

          // L1 row buffer (private memory) — one row of 32 elements
          %row_l1 = memref.alloc() : memref<32xf32, 2>

          // Phase 2: L2 -> L1  (per-thread DMA, one row)
          air.dma_memcpy_nd (%row_l1[] [] [],
                             %hl2[%tx, %c0_h] [%c32_h] [%c1_h])
              : (memref<32xf32, 2>, memref<32x32xf32, 1>)

          // Phase 3: L1 -> L3  (per-thread DMA, write row to global output)
          %dst_r = arith.addi %hrow, %tx : index
          air.dma_memcpy_nd (%hout[%dst_r, %hcol] [%c1_h, %c32_h] [%c64_h, %c1_h],
                             %row_l1[] [] [])
              : (memref<64x64xf32>, memref<32xf32, 2>)

          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
