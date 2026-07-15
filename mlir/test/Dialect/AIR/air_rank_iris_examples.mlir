//===- air_rank_iris_examples.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Round-trip tests for AIR IR representations of multi-GPU IRIS examples.
// These demonstrate that air.rank + air.channel + air.launch/segment/herd
// have sufficient expressibility to represent cross-rank communication
// patterns with per-device parallel compute.
//
// Each air.rank instance models one GPU. Inside each rank, air.launch
// provides the GPU grid, air.segment groups shared L2 memory, and
// air.herd provides data-parallel threads (PEs).
//
// GPU mapping (via air-to-rocdl):
//   air.launch  -> gpu.launch grid dimensions
//   air.segment -> transparent (inlined)
//   air.herd    -> gpu.launch block/thread dimensions
//   L2 (memspace 1) -> GPU shared/workgroup memory (space 3)
//   L1 (memspace 2) -> GPU private/register memory (space 5)
//   L3 (memspace 0) -> GPU global memory (function args)
//
// Note: all hierarchy ops are IsolatedFromAbove, so constants must
// be redefined in each scope body. Herds must be 2D.
//
// Reference: https://github.com/ROCm/iris
// Reference: https://xilinx.github.io/mlir-air/AIRComputeModel.html
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// ============================================================
// Example 1: Message Passing (Producer/Consumer)
//
// Based on: iris/examples/06_message_passing/message_passing_load_store.py
//
// Pattern:
//   - 2 ranks (GPUs), rank 0 = producer, rank 1 = consumer
//   - Producer sends data via cross-rank channel
//   - Consumer receives data, then uses air.launch/segment/herd
//     to compute dst[i] = dst[i] * 2.0 in parallel
//   - Channel implicit synchronization replaces IRIS atomics
//
// GPU mapping per rank (consumer):
//   air.launch(1,1) > air.segment > air.herd(4,1)
//   Memory: L3 --DMA--> L2 --DMA--> L1 (compute) --DMA--> L2 --DMA--> L3
// ============================================================

// CHECK: air.channel @msg_data_ch []
air.channel @msg_data_ch []

// CHECK-LABEL: func.func @message_passing
func.func @message_passing(%arg0 : memref<1024xf32>, %arg1 : memref<1024xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %u = air.universe.alloc(%c2)

  // CHECK: air.rank universe
  air.rank universe(%u) (%rx) in (%sx = %c2)
      args(%src = %arg0, %dst = %arg1) : memref<1024xf32>, memref<1024xf32> {
    %c0_r = arith.constant 0 : index
    %c1_r = arith.constant 1 : index
    %c1024_r = arith.constant 1024 : index

    %is_producer = arith.cmpi eq, %rx, %c0_r : index
    // CHECK: scf.if
    scf.if %is_producer {
      // Producer (rank 0): send 1024 floats via cross-rank channel.
      // CHECK: air.channel.put @msg_data_ch
      air.channel.put @msg_data_ch[]
          (%src[%c0_r] [%c1024_r] [%c1_r]) : (memref<1024xf32>)
    } else {
      // Consumer (rank 1): receive data, then compute dst *= 2.
      // CHECK: air.channel.get @msg_data_ch
      air.channel.get @msg_data_ch[]
          (%dst[%c0_r] [%c1024_r] [%c1_r]) : (memref<1024xf32>)

      // CHECK: air.launch
      %c1_l = arith.constant 1 : index
      air.launch (%bx, %by) in (%nbx = %c1_l, %nby = %c1_l)
          args(%l_dst = %dst) : memref<1024xf32> {
        %c4_s = arith.constant 4 : index
        %c1_s = arith.constant 1 : index
        air.segment @consumer_seg
            args(%s_dst = %l_dst) : memref<1024xf32> {
          %c0_ss = arith.constant 0 : index
          %c1_ss = arith.constant 1 : index
          %c4_ss = arith.constant 4 : index
          %c1024_ss = arith.constant 1024 : index
          // L2 staging buffer
          %l2_buf = memref.alloc() : memref<1024xf32, 1>
          air.dma_memcpy_nd (%l2_buf[] [] [],
              %s_dst[%c0_ss] [%c1024_ss] [%c1_ss])
              : (memref<1024xf32, 1>, memref<1024xf32>)

          // CHECK: air.herd
          air.herd @compute tile (%tx, %ty) in (%ntx = %c4_ss, %nty = %c1_ss)
              args(%h_buf = %l2_buf) : memref<1024xf32, 1> {
            %c1_h = arith.constant 1 : index
            %c256_h = arith.constant 256 : index
            %base = arith.muli %tx, %c256_h : index
            %two = arith.constant 2.0 : f32
            %l1_tile = memref.alloc() : memref<256xf32, 2>
            air.dma_memcpy_nd (%l1_tile[] [] [],
                %h_buf[%base] [%c256_h] [%c1_h])
                : (memref<256xf32, 2>, memref<1024xf32, 1>)
            affine.for %i = 0 to 256 {
              %val = memref.load %l1_tile[%i] : memref<256xf32, 2>
              %result = arith.mulf %val, %two : f32
              memref.store %result, %l1_tile[%i] : memref<256xf32, 2>
            }
            air.dma_memcpy_nd (%h_buf[%base] [%c256_h] [%c1_h],
                %l1_tile[] [] [])
                : (memref<1024xf32, 1>, memref<256xf32, 2>)
            memref.dealloc %l1_tile : memref<256xf32, 2>
            air.herd_terminator
          }

          // L2 -> L3
          air.dma_memcpy_nd (%s_dst[%c0_ss] [%c1024_ss] [%c1_ss],
              %l2_buf[] [] [])
              : (memref<1024xf32>, memref<1024xf32, 1>)
          memref.dealloc %l2_buf : memref<1024xf32, 1>
        }
      }
    }
    air.rank_terminator
  }
  return
}

// ============================================================
// Example 2: All-to-All Store
//
// Based on: iris/examples/03_all_store/all_store_bench.py
//
// Pattern:
//   - 4 ranks, each rank stores data into every other rank
//   - Uses N x N channel array indexed by [src_rank, dst_rank]
//   - Each rank uses air.launch/segment/herd to prepare data,
//     then cross-rank channel put/get exchanges it
//
// GPU mapping per rank:
//   air.launch(1,1) > air.segment > air.herd(4,1)
//   Each PE fills 1024 elements via L1, staged through L2
// ============================================================

// CHECK: air.channel @a2a_ch [4, 4]
air.channel @a2a_ch [4, 4]

// CHECK-LABEL: func.func @all_to_all_store
func.func @all_to_all_store(%arg0 : memref<4096xf32>) {
  %c4 = arith.constant 4 : index
  %u = air.universe.alloc(%c4)

  // CHECK: air.rank universe
  air.rank universe(%u) (%rx) in (%sx = %c4)
      args(%buf = %arg0) : memref<4096xf32> {
    %c0_r = arith.constant 0 : index
    %c1_r = arith.constant 1 : index
    %c4_r = arith.constant 4 : index
    %c4096_r = arith.constant 4096 : index

    // CHECK: air.launch
    %c1_l = arith.constant 1 : index
    air.launch (%bx, %by) in (%nbx = %c1_l, %nby = %c1_l)
        args(%l_buf = %buf, %l_rx = %rx) : memref<4096xf32>, index {
      %c4_s = arith.constant 4 : index
      %c1_s = arith.constant 1 : index
      air.segment @fill_seg
          args(%s_buf = %l_buf, %s_rx = %l_rx) : memref<4096xf32>, index {
        %c0_ss = arith.constant 0 : index
        %c1_ss = arith.constant 1 : index
        %c4_ss = arith.constant 4 : index
        %c4096_ss = arith.constant 4096 : index
        %l2_buf = memref.alloc() : memref<4096xf32, 1>

        // CHECK: air.herd
        air.herd @fill tile (%tx, %ty) in (%ntx = %c4_ss, %nty = %c1_ss)
            args(%h_buf = %l2_buf, %h_rx = %s_rx)
            : memref<4096xf32, 1>, index {
          %c1_h = arith.constant 1 : index
          %c1024_h = arith.constant 1024 : index
          %l1_tile = memref.alloc() : memref<1024xf32, 2>
          %rx_idx = arith.index_cast %h_rx : index to i32
          %rx_f32 = arith.sitofp %rx_idx : i32 to f32
          affine.for %i = 0 to 1024 {
            memref.store %rx_f32, %l1_tile[%i] : memref<1024xf32, 2>
          }
          %pe_offset = arith.muli %tx, %c1024_h : index
          air.dma_memcpy_nd (%h_buf[%pe_offset] [%c1024_h] [%c1_h],
              %l1_tile[] [] [])
              : (memref<4096xf32, 1>, memref<1024xf32, 2>)
          memref.dealloc %l1_tile : memref<1024xf32, 2>
          air.herd_terminator
        }

        air.dma_memcpy_nd (%s_buf[%c0_ss] [%c4096_ss] [%c1_ss],
            %l2_buf[] [] [])
            : (memref<4096xf32>, memref<4096xf32, 1>)
        memref.dealloc %l2_buf : memref<4096xf32, 1>
      }
    }

    // Cross-rank all-to-all communication
    // CHECK: scf.for
    scf.for %dst = %c0_r to %c4_r step %c1_r {
      %not_self = arith.cmpi ne, %rx, %dst : index
      // Use addi to produce a non-IV index so the channel verifier accepts it.
      %dst_idx = arith.addi %dst, %c0_r : index
      scf.if %not_self {
        // CHECK: air.channel.put @a2a_ch
        air.channel.put @a2a_ch[%rx, %dst_idx]
            (%buf[%c0_r] [%c4096_r] [%c1_r]) : (memref<4096xf32>)
      }
    }
    scf.for %src = %c0_r to %c4_r step %c1_r {
      %not_self = arith.cmpi ne, %src, %rx : index
      %src_idx = arith.addi %src, %c0_r : index
      scf.if %not_self {
        air.channel.get @a2a_ch[%src_idx, %rx]
            (%buf[%c0_r] [%c4096_r] [%c1_r]) : (memref<4096xf32>)
      }
    }
    air.rank_terminator
  }
  return
}

// ============================================================
// Example 3: GEMM + AllScatter
//
// Based on: iris/examples/07_gemm_all_scatter/gemm_all_scatter.py
//
// Pattern:
//   - 4 ranks, each computes local GEMM on a partition of B
//   - After GEMM, each rank scatters its C_local to all ranks
//   - Full air.launch/segment/herd hierarchy for tiled GEMM
//
// Global problem: C[128, 128] = A[128, 64] x B[64, 128]
// Per-rank: C_local[128, 32] = A[128, 64] x B_local[64, 32]
//
// GPU mapping per rank:
//   air.launch(4,1 grid) for M-dimension tiling
//     air.segment: L2 tiles for A[32x64], B[64x32], C[32x32]
//       air.herd(4,1 PEs): each PE computes 8 rows of 32x32 tile
//         L1: A rows, B tile, accumulator row
// ============================================================

// CHECK: air.channel @scatter_ch [4, 4]
air.channel @scatter_ch [4, 4]

// CHECK-LABEL: func.func @gemm_all_scatter
func.func @gemm_all_scatter(%arg0 : memref<128x64xf32>,
                             %arg1 : memref<64x128xf32>,
                             %arg2 : memref<128x128xf32>) {
  %c4 = arith.constant 4 : index
  %u = air.universe.alloc(%c4)

  // CHECK: air.rank universe
  air.rank universe(%u) (%rx) in (%sx = %c4)
      args(%A = %arg0, %B = %arg1, %C_global = %arg2)
      : memref<128x64xf32>, memref<64x128xf32>, memref<128x128xf32> {
    %c0_r = arith.constant 0 : index
    %c1_r = arith.constant 1 : index
    %c4_r = arith.constant 4 : index
    %c32_r = arith.constant 32 : index
    %c128_r = arith.constant 128 : index

    %col_offset = arith.muli %rx, %c32_r : index
    %C_local = memref.alloc() : memref<128x32xf32>

    // Tiled GEMM: 4 tiles of 32x32 along M dimension
    // CHECK: air.launch
    %c1_l = arith.constant 1 : index
    air.launch (%m_tile, %ly) in (%n_m_tiles = %c4_r, %nly = %c1_l)
        args(%l_A = %A, %l_B = %B, %l_C = %C_local, %l_col_off = %col_offset)
        : memref<128x64xf32>, memref<64x128xf32>, memref<128x32xf32>, index {

      %c1_s = arith.constant 1 : index
      %c4_s = arith.constant 4 : index
      air.segment @gemm_seg
          args(%s_A = %l_A, %s_B = %l_B, %s_C = %l_C,
               %s_col_off = %l_col_off, %s_m_tile = %m_tile)
          : memref<128x64xf32>, memref<64x128xf32>, memref<128x32xf32>,
            index, index {

        %c0_ss = arith.constant 0 : index
        %c1_ss = arith.constant 1 : index
        %c4_ss = arith.constant 4 : index
        %c8_ss = arith.constant 8 : index
        %c32_ss = arith.constant 32 : index
        %c64_ss = arith.constant 64 : index
        %c128_ss = arith.constant 128 : index

        // L2 shared memory tiles
        %A_tile = memref.alloc() : memref<32x64xf32, 1>
        %B_tile = memref.alloc() : memref<64x32xf32, 1>
        %C_tile = memref.alloc() : memref<32x32xf32, 1>

        // DMA: A[m_base:m_base+32, 0:64] from L3 -> L2
        %m_base = arith.muli %s_m_tile, %c32_ss : index
        air.dma_memcpy_nd (%A_tile[] [] [],
            %s_A[%m_base, %c0_ss] [%c32_ss, %c64_ss] [%c64_ss, %c1_ss])
            : (memref<32x64xf32, 1>, memref<128x64xf32>)

        // DMA: B[0:64, col_off:col_off+32] from L3 -> L2
        air.dma_memcpy_nd (%B_tile[] [] [],
            %s_B[%c0_ss, %s_col_off] [%c64_ss, %c32_ss] [%c128_ss, %c1_ss])
            : (memref<64x32xf32, 1>, memref<64x128xf32>)

        // 4 PEs, each computes 8 rows of the 32x32 output tile
        // CHECK: air.herd
        air.herd @matmul tile (%pe_x, %pe_y) in (%n_pe_x = %c4_ss, %n_pe_y = %c1_ss)
            args(%h_A = %A_tile, %h_B = %B_tile, %h_C = %C_tile)
            : memref<32x64xf32, 1>, memref<64x32xf32, 1>,
              memref<32x32xf32, 1> {

          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c8_h = arith.constant 8 : index
          %c32_h = arith.constant 32 : index
          %c64_h = arith.constant 64 : index

          // L1 private buffers
          %a_rows = memref.alloc() : memref<8x64xf32, 2>
          %b_local = memref.alloc() : memref<64x32xf32, 2>
          %acc = memref.alloc() : memref<8x32xf32, 2>

          // This PE handles rows [pe_x*8 : pe_x*8+8]
          %row_base = arith.muli %pe_x, %c8_h : index

          // DMA: load 8 rows of A_tile (L2 -> L1)
          air.dma_memcpy_nd (%a_rows[] [] [],
              %h_A[%row_base, %c0_h] [%c8_h, %c64_h] [%c64_h, %c1_h])
              : (memref<8x64xf32, 2>, memref<32x64xf32, 1>)

          // DMA: load full B_tile (L2 -> L1)
          air.dma_memcpy_nd (%b_local[] [] [],
              %h_B[%c0_h, %c0_h] [%c64_h, %c32_h] [%c32_h, %c1_h])
              : (memref<64x32xf32, 2>, memref<64x32xf32, 1>)

          // Initialize accumulator
          %zero = arith.constant 0.0 : f32
          affine.for %r = 0 to 8 {
            affine.for %col = 0 to 32 {
              memref.store %zero, %acc[%r, %col] : memref<8x32xf32, 2>
            }
          }

          // Matmul: acc[r,c] += a_rows[r,k] * b_local[k,c]
          affine.for %r = 0 to 8 {
            affine.for %k = 0 to 64 {
              %a_val = memref.load %a_rows[%r, %k] : memref<8x64xf32, 2>
              affine.for %col = 0 to 32 {
                %b_val = memref.load %b_local[%k, %col] : memref<64x32xf32, 2>
                %old = memref.load %acc[%r, %col] : memref<8x32xf32, 2>
                %prod = arith.mulf %a_val, %b_val : f32
                %new = arith.addf %old, %prod : f32
                memref.store %new, %acc[%r, %col] : memref<8x32xf32, 2>
              }
            }
          }

          // DMA: store 8 result rows (L1 -> L2)
          air.dma_memcpy_nd (%h_C[%row_base, %c0_h] [%c8_h, %c32_h] [%c32_h, %c1_h],
              %acc[] [] [])
              : (memref<32x32xf32, 1>, memref<8x32xf32, 2>)

          memref.dealloc %a_rows : memref<8x64xf32, 2>
          memref.dealloc %b_local : memref<64x32xf32, 2>
          memref.dealloc %acc : memref<8x32xf32, 2>
          air.herd_terminator
        }

        // DMA: C_tile from L2 -> L3
        air.dma_memcpy_nd (%s_C[%m_base, %c0_ss] [%c32_ss, %c32_ss] [%c32_ss, %c1_ss],
            %C_tile[] [] [])
            : (memref<128x32xf32>, memref<32x32xf32, 1>)

        memref.dealloc %A_tile : memref<32x64xf32, 1>
        memref.dealloc %B_tile : memref<64x32xf32, 1>
        memref.dealloc %C_tile : memref<32x32xf32, 1>
      }
    }

    // AllScatter: distribute C_local to all ranks
    // CHECK: scf.for
    scf.for %dst = %c0_r to %c4_r step %c1_r {
      // Use addi to produce a non-IV index so the channel verifier accepts it.
      %dst_idx = arith.addi %dst, %c0_r : index
      // CHECK: air.channel.put @scatter_ch
      air.channel.put @scatter_ch[%rx, %dst_idx]
          (%C_local[%c0_r, %c0_r] [%c128_r, %c32_r] [%c32_r, %c1_r])
          : (memref<128x32xf32>)
    }
    scf.for %src = %c0_r to %c4_r step %c1_r {
      %src_idx = arith.addi %src, %c0_r : index
      %src_col_offset = arith.muli %src, %c32_r : index
      // CHECK: air.channel.get @scatter_ch
      air.channel.get @scatter_ch[%src_idx, %rx]
          (%C_global[%c0_r, %src_col_offset] [%c128_r, %c32_r] [%c128_r, %c1_r])
          : (memref<128x128xf32>)
    }

    memref.dealloc %C_local : memref<128x32xf32>
    air.rank_terminator
  }
  return
}
