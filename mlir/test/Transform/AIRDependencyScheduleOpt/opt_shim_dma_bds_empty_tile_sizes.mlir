//===- opt_shim_dma_bds_empty_tile_sizes.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Default path (empty `shim-dma-tile-sizes`) for `air-opt-shim-dma-bds`.
// Split out from opt_shim_dma_bds.mlir so this RUN gets its own 30s lit
// budget on Assert builds; NPUTILED + AIE1 RUNs stay in the sibling file.
//
// The default tiles every shim scf.for level by 1: iteration-count neutral
// but invokes tilePerfectlyNested + post-tile fixup. The output fully
// unrolls shim loop nests, so we assert per-function op counts and that
// no scf.for survives.

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1" | FileCheck %s

module {

  // Three scf.for loop nested around air.channle.put containing two effective wrap-and-stride dimensions. 
  // Specialize two inner-most for loops into the wrap-and-stride list, and leave one outer-most for loop unchanged.


  

  // CHECK-LABEL: func.func @func0
  // CHECK-NOT: scf.for
  // CHECK-COUNT-8: air.channel
  // CHECK-NOT: scf.for
  func.func @func0(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3) in (%arg4=%c1) args(%arg5=%arg0) : memref<512x512xbf16> {
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32768 = arith.constant 32768 : index
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg6 = %c0 to %c512 step %c256 iter_args(%arg7 = %1) -> (!air.async.token) {
        %3 = scf.for %arg8 = %c0 to %c512 step %c256 iter_args(%arg9 = %arg7) -> (!air.async.token) {
          %4 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %arg9) -> (!air.async.token) {
            %5 = air.channel.put async [%arg11]  @channel_0[%c0, %c0] (%arg5[%c0, %c0, %arg6, %arg10] [%c1_0, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {metadata = @airMemcpyId22} : (memref<512x512xbf16>)
            %6 = air.channel.put async [%arg11]  @channel_0[%c1_0, %c0] (%arg5[%c1_0, %c0, %arg6, %arg10] [%c1_0, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {metadata = @airMemcpyId23} : (memref<512x512xbf16>)
            %7 = air.channel.put async [%arg11]  @channel_0[%c2, %c0] (%arg5[%c2, %c0, %arg6, %arg10] [%c1_0, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {metadata = @airMemcpyId24} : (memref<512x512xbf16>)
            %8 = air.channel.put async [%arg11]  @channel_0[%c3, %c0] (%arg5[%c3, %c0, %arg6, %arg10] [%c1_0, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {metadata = @airMemcpyId25} : (memref<512x512xbf16>)
            %9 = air.wait_all async [%5, %6, %7, %8] 
            scf.yield %9 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        scf.yield %3 : !air.async.token
      }
    }
    return
  }

  // Three scf.for loop nested around air.channle.put containing two effective wrap-and-stride dimensions. 
  // This test is different from func0 in two ways:
  // The first air.channel.put, after wrap-and-stride canonicalization, is capable of folding all three scf.for loops in the nest into wraps and strides.
  // The second to fourth air.channel.puts can only fold one inner-most for loop into wrap-and-stride list due to having non-zero offsets at 3rd dimension.

  
  

  // CHECK-LABEL: func.func @func1
  // CHECK-NOT: scf.for
  // CHECK-COUNT-4: air.channel
  // CHECK-NOT: scf.for
  func.func @func1(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3) in (%arg4=%c1) args(%arg5=%arg1) : memref<512x512xbf16> {
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32768 = arith.constant 32768 : index
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg6 = %c0 to %c512 step %c256 iter_args(%arg7 = %1) -> (!air.async.token) {
        %3 = scf.for %arg8 = %c0 to %c512 step %c256 iter_args(%arg9 = %arg7) -> (!air.async.token) {
          %4 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %arg9) -> (!air.async.token) {
            %5 = air.channel.put async [%arg11]  @channel_0[%c0, %c0] (%arg5[%c0, %c0, %arg10, %arg8] [%c1_0, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {metadata = @airMemcpyId26} : (memref<512x512xbf16>)
            %6 = air.channel.put async [%arg11]  @channel_0[%c1_0, %c0] (%arg5[%c0, %c1_0, %arg10, %arg8] [%c1_0, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {metadata = @airMemcpyId27} : (memref<512x512xbf16>)
            %7 = air.channel.put async [%arg11]  @channel_0[%c2, %c0] (%arg5[%c0, %c2, %arg10, %arg8] [%c1_0, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {metadata = @airMemcpyId28} : (memref<512x512xbf16>)
            %8 = air.channel.put async [%arg11]  @channel_0[%c3, %c0] (%arg5[%c0, %c3, %arg10, %arg8] [%c1_0, %c1_0, %c64, %c64] [%c32768, %c64, %c512, %c1_0]) {metadata = @airMemcpyId29} : (memref<512x512xbf16>)
            %9 = air.wait_all async [%5, %6, %7, %8] 
            scf.yield %9 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        scf.yield %3 : !air.async.token
      }
    }
    return
  }

  // Two scf.for loop nested around air.channle.get containing two effective wrap-and-stride dimensions. 
  // Both for loops can be folded into the wrap-and-stride list; no scf.for loop remains.

  


  // CHECK-LABEL: func.func @func2
  // CHECK-NOT: scf.for
  // CHECK-COUNT-4: air.channel
  // CHECK-NOT: scf.for
  func.func @func2(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3) in (%arg4=%c1) args(%arg5=%arg2) : memref<512x512xbf16> {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg6 = %c0 to %c512 step %c256 iter_args(%arg7 = %1) -> (!air.async.token) {
        %3 = scf.for %arg8 = %c0 to %c512 step %c256 iter_args(%arg9 = %arg7) -> (!air.async.token) {
          %4 = air.channel.get async [%arg9]  @channel_0[%c0, %c0] (%arg5[%c0, %arg8] [%c64, %c256] [%c512, %c1_0]) {metadata = @airMemcpyId39} : (memref<512x512xbf16>)
          %5 = air.channel.get async [%arg9]  @channel_0[%c1_0, %c0] (%arg5[%c64, %arg8] [%c64, %c256] [%c512, %c1_0]) {metadata = @airMemcpyId41} : (memref<512x512xbf16>)
          %6 = air.channel.get async [%arg9]  @channel_0[%c2, %c0] (%arg5[%c128, %arg8] [%c64, %c256] [%c512, %c1_0]) {metadata = @airMemcpyId43} : (memref<512x512xbf16>)
          %7 = air.channel.get async [%arg9]  @channel_0[%c3, %c0] (%arg5[%c192, %arg8] [%c64, %c256] [%c512, %c1_0]) {metadata = @airMemcpyId45} : (memref<512x512xbf16>)
          %8 = air.wait_all async [%4, %5, %6, %7] 
          scf.yield %8 : !air.async.token
        }
        scf.yield %3 : !air.async.token
      }
    }
    return
  }

  // No air.launch or air.segment.

  


  // CHECK-LABEL: func.func @func4
  // CHECK-NOT: scf.for
  // CHECK-COUNT-16: air.channel
  // CHECK-NOT: scf.for
  func.func @func4(%arg0: memref<128x128xbf16>) {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c128 step %c32 {
      scf.for %arg4 = %c0 to %c128 step %c32 {
        air.channel.put  @channel_0[%c0, %c0] (%arg0[%arg3, %arg4] [%c32, %c32] [%c128, %c1]) {metadata = @airMemcpyId4} : (memref<128x128xbf16>)
      }
    }
    return
  }

  // Repeat dimension promotion.


  

  // CHECK-LABEL: func.func @func5
  // CHECK-NOT: scf.for
  // CHECK-COUNT-12: air.channel
  // CHECK-NOT: scf.for
  func.func @func5(%arg0: memref<8x8xi32>, %arg1: memref<8x8xi32>, %arg2: memref<8x8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %1 = air.wait_all async
    %2 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg7 = %1) -> (!air.async.token) {
      %3 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg8 = %arg7) -> (!air.async.token) {
        %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%arg3]
        %put0 = air.channel.put async [%arg8]  @channel_0[] (%arg0[%4, %c0] [%c4, %c8] [%c8, %c1]) {metadata = @airMemcpyId4} : (memref<8x8xi32>)
        %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%arg4]
        %put1 = air.channel.put async [%arg8]  @channel_1[] (%arg1[%c0, %5] [%c8, %c4] [%c8, %c1]) {metadata = @airMemcpyId5} : (memref<8x8xi32>)
        %get0 = air.channel.get async [%arg8]  @channel_2[] (%arg2[%4, %5] [%c4, %c4] [%c8, %c1]) {metadata = @airMemcpyId16} : (memref<8x8xi32>)
        %w = air.wait_all async [%put0, %put1, %get0]
        scf.yield %w : !air.async.token
      }
      scf.yield %3 : !air.async.token
    }
    return
  }

  // Repeat dimension promotion.


  

  
  // CHECK-LABEL: func.func @func6
  // CHECK-NOT: scf.for
  // CHECK-COUNT-6: air.channel
  // CHECK-NOT: scf.for
  func.func @func6(%arg0: memref<8x16xi32>, %arg1: memref<16x32xi32>, %arg2: memref<8x32xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %1 = air.wait_all async
    %2 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg7 = %1) -> (!air.async.token) {
      %3 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg8 = %arg7) -> (!air.async.token) {
        %put0 = air.channel.put async [%arg8]  @channel_0[] (%arg0[%arg3, %c0] [%c8, %c16] [%c16, %c1]) {metadata = @airMemcpyId4} : (memref<8x16xi32>)
        %5 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%arg4]
        %put1 = air.channel.put async [%arg8]  @channel_1[] (%arg1[%c0, %5] [%c16, %c16] [%c32, %c1]) {metadata = @airMemcpyId5} : (memref<16x32xi32>)
        %get0 = air.channel.get async [%arg8]  @channel_2[] (%arg2[%arg3, %5] [%c8, %c16] [%c32, %c1]) {metadata = @airMemcpyId12} : (memref<8x32xi32>)
        %w = air.wait_all async [%put0, %put1, %get0]
        scf.yield %w : !air.async.token
      }
      scf.yield %3 : !air.async.token
    }
    return
  }

  // Repeat dimension promotion.


  

  // CHECK-LABEL: func.func @func7
  // CHECK-NOT: scf.for
  // CHECK-COUNT-48: air.channel
  // CHECK-NOT: scf.for
  func.func @func7(%arg0: memref<2048x512xi32>, %arg1: memref<512x2048xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c2048 = arith.constant 2048 : index
    %alloc = memref.alloc() : memref<2048x2048xi32>
    %1 = air.wait_all async
    %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg7 = %1) -> (!air.async.token) {
      %3 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg8 = %arg7) -> (!air.async.token) {
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg3]
        %put0 = air.channel.put async [%arg8]  @channel_0[] (%arg0[%c0, %4, %c0] [%c8, %c64, %c64] [%c64, %c512, %c1]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
        %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg4]
        %put1 = air.channel.put async [%arg8]  @channel_1[] (%arg1[%c0, %5] [%c512, %c64] [%c2048, %c1]) {metadata = @airMemcpyId21} : (memref<512x2048xi32>)
        %get0 = air.channel.get async [%arg8]  @channel_2[] (%alloc[%4, %5] [%c64, %c64] [%c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
        %w = air.wait_all async [%put0, %put1, %get0]
        scf.yield %w : !air.async.token
      }
      scf.yield %3 : !air.async.token
    }
    return
  }

  // NPU wrap size limit: [0, 1023].


  
  
  // CHECK-LABEL: func.func @func8
  // CHECK-NOT: scf.for
  // CHECK-COUNT-48: air.channel
  // CHECK-NOT: scf.for
  func.func @func8(%arg0: memref<2048x2048xi32>, %arg1: memref<2048x2048xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c2048 = arith.constant 2048 : index
    %alloc = memref.alloc() : memref<2048x2048xi32>
    %1 = air.wait_all async
    %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg7 = %1) -> (!air.async.token) {
      %3 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg8 = %arg7) -> (!air.async.token) {
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg3]
        %put0 = air.channel.put async [%arg8]  @channel_0[] (%arg0[%c0, %4, %c0] [%c8, %c64, %c256] [%c256, %c2048, %c1]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
        %5 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg4]
        %put1 = air.channel.put async [%arg8]  @channel_1[] (%arg1[%c0, %5] [%c2048, %c64] [%c2048, %c1]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
        %get0 = air.channel.get async [%arg8]  @channel_2[] (%alloc[%4, %5] [%c64, %c64] [%c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
        %w = air.wait_all async [%put0, %put1, %get0]
        scf.yield %w : !air.async.token
      }
      scf.yield %3 : !air.async.token
    }
    return
  }

  // NPU wrap size limit: [0, 1023]; stride limit: [0, 1048576].


  

  // CHECK-LABEL: func.func @func9
  // CHECK-NOT: scf.for
  // CHECK-COUNT-9: air.channel
  // CHECK-NOT: scf.for
  func.func @func9(%arg0: memref<2304x2304xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c2304 = arith.constant 2304 : index
    %1 = air.wait_all async
    %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg7 = %1) -> (!air.async.token) {
      %3 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg7) -> (!air.async.token) {
        %5 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%arg4]
        %put0 = air.channel.put async [%arg8]  @channel_1[] (%arg0[%c0, %5] [%c2304, %c64] [%c2304, %c1]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
        %w = air.wait_all async [%put0]
        scf.yield %w : !air.async.token
      }
      scf.yield %3 : !air.async.token
    }
    return
  }

  // Multiple Shim DMAs.


  
  
  // CHECK-LABEL: func.func @func10
  // CHECK-NOT: scf.for
  // CHECK-COUNT-24: air.channel
  // CHECK-NOT: scf.for
  func.func @func10(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %1 = air.wait_all async
    %2 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg7 = %1) -> (!air.async.token) {
      %3 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg8 = %arg7) -> (!air.async.token) {
        %4 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%arg3]
        %put0 = air.channel.put async [%arg8]  @channel_0[] (%arg0[%c0, %4, %c0] [%c4, %c256, %c256] [%c256, %c1024, %c1]) {metadata = @airMemcpyId7} : (memref<512x1024xbf16>)
        %5 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%arg4]
        %put1 = air.channel.put async [%arg8]  @channel_1[] (%arg1[%c0, %5] [%c1024, %c256] [%c512, %c1]) {metadata = @airMemcpyId12} : (memref<1024x512xbf16>)
        %6 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%arg3]
        %get0 = air.channel.get async [%arg8]  @channel_2[%c0, %c0] (%arg2[%6, %5] [%c64, %c256] [%c512, %c1]) {metadata = @airMemcpyId45} : (memref<512x512xbf16>)
        %7 = affine.apply affine_map<()[s0] -> (s0 * 256 + 64)>()[%arg3]
        %get1 = air.channel.get async [%arg8]  @channel_2[%c0, %c1] (%arg2[%7, %5] [%c64, %c256] [%c512, %c1]) {metadata = @airMemcpyId46} : (memref<512x512xbf16>)
        %8 = affine.apply affine_map<()[s0] -> (s0 * 256 + 128)>()[%arg3]
        %get2 = air.channel.get async [%arg8]  @channel_2[%c1, %c0] (%arg2[%8, %5] [%c64, %c256] [%c512, %c1]) {metadata = @airMemcpyId47} : (memref<512x512xbf16>)
        %9 = affine.apply affine_map<()[s0] -> (s0 * 256 + 192)>()[%arg3]
        %get3 = air.channel.get async [%arg8]  @channel_2[%c1, %c1] (%arg2[%9, %5] [%c64, %c256] [%c512, %c1]) {metadata = @airMemcpyId48} : (memref<512x512xbf16>)
        %w = air.wait_all async [%put0, %put1, %get0, %get1, %get2, %get3]
        scf.yield %w : !air.async.token
      }
      scf.yield %3 : !air.async.token
    }
    return
  }

  // Big memref.


  
  
  // CHECK-LABEL: func.func @func11
  // CHECK-NOT: scf.for
  // CHECK-COUNT-4: air.channel
  // CHECK-NOT: scf.for
  func.func @func11() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c19 = arith.constant 19 : index
    %c28 = arith.constant 28 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c2432 = arith.constant 2432 : index
    %alloc = memref.alloc() : memref<308x2432xi32>
    %1 = air.wait_all async
    %2 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg7 = %1) -> (!air.async.token) {
      %3 = scf.for %arg4 = %c0 to %c1 step %c1 iter_args(%arg8 = %arg7) -> (!air.async.token) {
        %4 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%arg3]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%arg4]
        %put0 = air.channel.put async [%arg8]  @channel_0[] (%alloc[%c0, %c0, %c0] [%c19, %c28, %c128] [%c128, %c2432, %c1]) {metadata = @airMemcpyId26} : (memref<308x2432xi32>)
        %w = air.wait_all async [%put0]
        scf.yield %w : !air.async.token
      }
      scf.yield %3 : !air.async.token
    }
    return
  }

  // Offset field with (1) for loop induction variable, (2) affine map, and (3) existing non-singleton stride.


  
  
  // CHECK-LABEL: func.func @func12
  // CHECK-NOT: scf.for
  // CHECK-COUNT-2: air.channel
  // CHECK-NOT: scf.for
  func.func @func12(%arg0: memref<2x64x64xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c19 = arith.constant 19 : index
    %c28 = arith.constant 28 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c2432 = arith.constant 2432 : index
    %c4096 = arith.constant 4096 : index
    %1 = air.wait_all async
    %2 = scf.for %arg3 = %c0 to %c1 step %c1 iter_args(%arg7 = %1) -> (!air.async.token) {
      %3 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg8 = %arg7) -> (!air.async.token) {
        %4 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg9 = %arg8) -> (!air.async.token) {
          %5 = scf.for %arg6 = %c0 to %c1 step %c1 iter_args(%arg10 = %arg9) -> (!air.async.token) {
            %6 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg6]
            %7 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg5]
            %put0 = air.channel.put async [%arg8]  @channel_0[] (%arg0[%arg4, %7, %6] [%c1, %c64, %c64] [%c4096, %c64, %c1]) {metadata = @airMemcpyId31} : (memref<2x64x64xi32>)
            %w = air.wait_all async [%put0]
            scf.yield %w : !air.async.token
          }
          scf.yield %5 : !air.async.token
        }
        scf.yield %4 : !air.async.token
      }
      scf.yield %3 : !air.async.token
    }
    return
  }

  // Scf.for operating on integer type; scf.for nested inside scf.parallel.


  // CHECK-LABEL: func.func @func13
  // CHECK-NOT: scf.for
  // CHECK-COUNT-4: air.channel
  // CHECK-NOT: scf.for
  func.func @func13(%arg0: memref<*xf32>, %arg1: memref<*xf32>) {
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = air.wait_all async 
    %1 = scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
      %async_token, %results = air.execute -> (index) {
        %4 = arith.muli %arg2, %c128 : index
        air.execute_terminator %4 : index
      } {id = 5 : i32}
      %async_token_0, %results_1 = air.execute [%async_token] -> (i32) {
        %4 = arith.index_cast %results : index to i32
        air.execute_terminator %4 : i32
      } {id = 6 : i32}
      %async_token_2, %results_3 = air.execute -> (index) {
        %4 = arith.muli %arg3, %c64 : index
        air.execute_terminator %4 : index
      } {id = 7 : i32}
      %async_token_4, %results_5 = air.execute [%async_token_2] -> (i32) {
        %4 = arith.index_cast %results_3 : index to i32
        air.execute_terminator %4 : i32
      } {id = 8 : i32}
      %2 = air.wait_all async [%async_token_0, %async_token_4]  {id = 3 : i32}
      %3 = scf.for %arg4 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg5 = %2) -> (!air.async.token)  : i32 {
        %async_token_6, %results_7 = air.execute [%arg5] -> (i32) {
          %6 = arith.muli %arg4, %c4_i32 : i32
          air.execute_terminator %6 : i32
        } {id = 9 : i32}
        %async_token_8, %results_9 = air.execute [%async_token_6] -> (i32) {
          %6 = arith.addi %results_1, %results_7 : i32
          air.execute_terminator %6 : i32
        } {id = 10 : i32}
        %4 = air.wait_all async [%async_token_8, %arg5]  {id = 1 : i32}
        %5 = scf.for %arg6 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg7 = %4) -> (!air.async.token)  : i32 {
          %async_token_10, %results_11 = air.execute [%arg7] -> (i32) {
            %7 = arith.muli %arg6, %c2_i32 : i32
            air.execute_terminator %7 : i32
          } {id = 11 : i32}
          %async_token_12, %results_13 = air.execute [%async_token_10] -> (i32) {
            %7 = arith.addi %results_5, %results_11 : i32
            air.execute_terminator %7 : i32
          } {id = 12 : i32}
          %async_token_14, %results_15 = air.execute [%arg7] -> (index) {
            %7 = arith.index_cast %results_9 : i32 to index
            air.execute_terminator %7 : index
          } {id = 13 : i32}
          %async_token_16, %results_17 = air.execute [%async_token_14] -> (index) {
            %7 = arith.muli %results_15, %c128 : index
            air.execute_terminator %7 : index
          } {id = 14 : i32}
          %async_token_18, %results_19 = air.execute [%arg7, %async_token_12] -> (index) {
            %7 = arith.index_cast %results_13 : i32 to index
            air.execute_terminator %7 : index
          } {id = 15 : i32}
          %async_token_20, %results_21 = air.execute [%async_token_16, %async_token_18] -> (index) {
            %7 = arith.addi %results_17, %results_19 : index
            air.execute_terminator %7 : index
          } {id = 16 : i32}
          %6 = air.channel.put async [%async_token_20]  @channel_0[%arg2, %arg3] (%arg0[%c0, %results_21] [%c4, %c2] [%c128, %c1]) {id = 1 : i32, metadata = @airMemcpyId3} : (memref<*xf32>)
          scf.yield %6 : !air.async.token
        }
        scf.yield %5 : !air.async.token
      }
      scf.reduce(%3 : !air.async.token) {
      ^bb0(%arg4: !air.async.token, %arg5: !air.async.token):
        %4 = air.wait_all async [%arg4, %arg5] 
        scf.reduce.return %4 : !air.async.token
      }
    }
    return
  }

  // Scf.parallel unrolling: reduced tokens must be preserved into the blocking wait_all at launch terminator.


  // CHECK-LABEL: func.func @func14
  // CHECK-NOT: scf.for
  // CHECK-COUNT-4: air.channel
  // CHECK-NOT: scf.for
  func.func @func14(%arg0: memref<*xf32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg9, %arg10) in (%arg11=%c1, %arg12=%c1) args(%arg13=%arg0) : memref<*xf32> attributes {id = 1 : i32} {
      %c128 = arith.constant 128 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %1 = air.wait_all async 
      %2 = scf.parallel (%arg14, %arg15) = (%c0, %c0) to (%c2, %c2) step (%c1_0, %c1_0) init (%1) -> !air.async.token {
        %3 = air.channel.put async  @channel_0[%arg14, %arg15] (%arg13[%c0, %c0] [%c4, %c2] [%c128, %c1_0]) {id = 1 : i32, metadataArray = [{base = "air_channel_0_0", index = 0 : i32}, {base = "air_channel_0_1", index = 1 : i32}, {base = "air_channel_0_2", index = 2 : i32}, {base = "air_channel_0_3", index = 3 : i32}]} : (memref<*xf32>)
        scf.reduce(%3 : !air.async.token) {
        ^bb0(%arg16: !air.async.token, %arg17: !air.async.token):
          %4 = air.wait_all async [%arg16, %arg17] 
          scf.reduce.return %4 : !air.async.token
        }
      }
    }
    return
  }

  // Multi-operand affine.apply in channel.put offset. The offset
  // d0 * 64 + s0 is computed from one dim operand and one symbol operand. When
  // eraseWrapNStrideDim folds adjacent dimensions (stride[0] == size[1] *
  // stride[1]), it must compose the stride factor into the affine expression:
  // new_offset = (d0 * 64 + s0) * 64 / 1 = d0 * 4096 + s0 * 64.
  // Previously, multi-operand affine.apply expressions were skipped by
  // eraseWrapNStrideDim in Util.cpp, causing the stride multiplication to be
  // lost.


  // CHECK-LABEL: func.func @func16
  // CHECK-NOT: scf.for
  // CHECK-COUNT-1: air.channel
  // CHECK-NOT: scf.for
  func.func @func16(%arg0: memref<2x64x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3) in (%arg4=%c1) args(%arg5=%arg0) : memref<2x64x64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      // scf.for loop that won't be unrolled by this pass
      scf.for %arg6 = %c0 to %c2 step %c1_0 {
        // head_offset = arg6 * 64 + 0, i.e., multi-operand affine.apply
        %head_off = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg6)[%c0]
        %1 = air.channel.put async @channel_0[%c0, %c0] (%arg5[%head_off, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 1 : i32} : (memref<2x64x64xbf16>)
      }
    }
    return
  }

  // Canonicalizing repeat dimension at highest dimension.


  // CHECK-LABEL: func.func @func15
  // CHECK-NOT: scf.for
  // CHECK-COUNT-1: air.channel
  // CHECK-NOT: scf.for
  func.func @func15(%arg0: memref<512x512xbf16>) {
    %0 = air.launch async () in () args(%arg8=%arg0) : memref<512x512xbf16> {
      %c65536 = arith.constant 65536 : index
      %c4 = arith.constant 4 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %1 = air.channel.put async  @channel_0[%c0, %c0] (%arg8[%c0, %c0, %c1, %c0, %c256] [%c2, %c4, %c1, %c128, %c64] [%c0, %c65536, %c64, %c512, %c1]) {id = 6 : i32, metadataArray = [{base = "air_channel_13_0", index = 0 : i32}, {base = "air_channel_13_1", index = 1 : i32}, {base = "air_channel_13_2", index = 2 : i32}, {base = "air_channel_13_3", index = 3 : i32}]} : (memref<512x512xbf16>)
    }
    return
  }

  // Multiple air.launch ops in one function. Each launch's shim DMA BDs
  // should be optimized independently. The scf.for loops in each launch
  // are folded into the channel wrap-and-stride dimensions.


  // CHECK-LABEL: func.func @func_multi_launch
  // CHECK-NOT: scf.for
  // CHECK-COUNT-2: air.channel
  // CHECK-NOT: scf.for
  func.func @func_multi_launch(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx) in (%sx=%c1) args(%buf=%arg0) : memref<512x512xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %1 = air.wait_all async
      %2 = scf.for %i = %c0 to %c512 step %c256 iter_args(%tok = %1) -> (!air.async.token) {
        %3 = air.channel.put async [%tok]  @channel_0[%c0, %c0] (%buf[%c0, %i] [%c256, %c64] [%c512, %c1_0]) {id = 1 : i32} : (memref<512x512xbf16>)
        scf.yield %3 : !air.async.token
      }
    }
    %4 = air.launch async [%0] (%ty) in (%sy=%c1) args(%buf2=%arg1) : memref<512x512xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %5 = air.wait_all async
      %6 = scf.for %j = %c0 to %c512 step %c256 iter_args(%tok2 = %5) -> (!air.async.token) {
        %7 = air.channel.get async [%tok2]  @channel_1[%c0, %c0] (%buf2[%c0, %j] [%c256, %c64] [%c512, %c1_0]) {id = 2 : i32} : (memref<512x512xbf16>)
        scf.yield %7 : !air.async.token
      }
    }
    return
  }
}
