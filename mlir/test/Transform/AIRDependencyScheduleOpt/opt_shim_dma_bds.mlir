//===- opt_shim_dma_bds.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds | FileCheck %s

// Optimize logical air.channel.put/get op into efficient shim dma block descriptor (BD).

module {

  // Three scf.for loop nested around air.channle.put containing two effective wrap-and-stride dimensions. 
  // Specialize two inner-most for loops into the wrap-and-stride list, and leave one outer-most for loop unchanged.

  // CHECK-LABEL: func0
  // CHECK: scf.for %[[EVENT0:.*]] = %c0{{.*}} to %c512{{.*}} step %c256{{.*}} iter_args(%[[EVENT1:.*]] = %{{.*}}) -> (!air.async.token) {
  // CHECK-NEXT: %[[EVENT2:.*]] = air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %[[EVENT0]], %c0{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId22} : (memref<512x512xbf16>)
  // CHECK-NEXT: %[[EVENT3:.*]] = air.channel.put async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %[[EVENT0]], %c32768{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId23} : (memref<512x512xbf16>)
  // CHECK-NEXT: %[[EVENT4:.*]] = air.channel.put async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %[[EVENT0]], %c65536{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId24} : (memref<512x512xbf16>)
  // CHECK-NEXT: %[[EVENT5:.*]] = air.channel.put async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %[[EVENT0]], %c98304{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId25} : (memref<512x512xbf16>)
  // CHECK-NEXT: %[[EVENT6:.*]] = air.wait_all async [%[[EVENT2]], %[[EVENT3]], %[[EVENT4]], %[[EVENT5]]] 
  // CHECK-NEXT: scf.yield %[[EVENT6]] : !air.async.token
  // CHECK-NEXT: }

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

  // CHECK-LABEL: func1

  // Three scf.for loop nested around air.channle.put containing two effective wrap-and-stride dimensions. 
  // This test is different from func0 in two ways:
  // The first air.channel.put, after wrap-and-stride canonicalization, is capable of folding all three scf.for loops in the nest into wraps and strides.
  // The second to fourth air.channel.puts can only fold one inner-most for loop into wrap-and-stride list due to having non-zero offsets at 3rd dimension.

  // CHECK: air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId26} : (memref<512x512xbf16>)
  // CHECK: scf.for %[[EVENT0:.*]] = %c0{{.*}} to %c512{{.*}} step %c256{{.*}} iter_args(%{{.*}} = %{{.*}})
  // CHECK-NEXT: %[[EVENT2:.*]] = scf.for %[[EVENT1:.*]] = %c0{{.*}} to %c512{{.*}} step %c256{{.*}} iter_args(%{{.*}} = %{{.*}})
  // CHECK: %[[EVENT3:.*]] = air.channel.put async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c1{{.*}}, %c0{{.*}}, %[[EVENT1]]] [%c8{{.*}}, %c1{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c32768{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId27} : (memref<512x512xbf16>)
  // CHECK: %[[EVENT4:.*]] = air.channel.put async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c2{{.*}}, %c0{{.*}}, %[[EVENT1]]] [%c8{{.*}}, %c1{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c32768{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId28} : (memref<512x512xbf16>)
  // CHECK: %[[EVENT5:.*]] = air.channel.put async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c3{{.*}}, %c0{{.*}}, %[[EVENT1]]] [%c8{{.*}}, %c1{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c32768{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId29} : (memref<512x512xbf16>)
  // CHECK: %[[EVENT6:.*]] = air.wait_all async [%[[EVENT3]], %[[EVENT4]], %[[EVENT5]]] 
  // CHECK: scf.yield %[[EVENT6]] : !air.async.token
  // CHECK: }
  // CHECK: scf.yield %[[EVENT2]] : !air.async.token
  // CHECK: }

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

  // CHECK-LABEL: func2

  // Two scf.for loop nested around air.channle.get containing two effective wrap-and-stride dimensions. 
  // Both for loops can be folded into the wrap-and-stride list; no scf.for loop remains.

  // CHECK: air.channel.get async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId39} : (memref<512x512xbf16>)
  // CHECK: air.channel.get async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId41} : (memref<512x512xbf16>)
  // CHECK: air.channel.get async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c128{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId43} : (memref<512x512xbf16>)
  // CHECK: air.channel.get async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c192{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId45} : (memref<512x512xbf16>)

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

  // CHECK-LABEL: func3

  // Hoisting air.herd op out of any parent scf.for loop nest.

  // CHECK: air.herd
  // CHECK-COUNT-3: scf.for
  // CHECK-COUNT-4: air.dma_memcpy_nd
  // CHECK-COUNT-4: }

  func.func @func3(%arg0: memref<4096x1024x512xi32>, %arg1: memref<4096x1024x512xi32>, %arg2: memref<4096x1024x512xi32>) {
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c4096 step %c128 {
      scf.for %arg4 = %c0 to %c1024 step %c128 {
        scf.for %arg5 = %c0 to %c512 step %c128 {
          air.herd tile (%arg6, %arg7) in (%arg8=%c4, %arg9=%c4) args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg0, %arg13=%arg5, %arg14=%arg1, %arg15=%arg2) : index,index,memref<4096x1024x512xi32>,index,memref<4096x1024x512xi32>,memref<4096x1024x512xi32>attributes {sym_name = "herd_0"} {
            %c1 = arith.constant 1 : index
            %c512_0 = arith.constant 512 : index
            %c524288 = arith.constant 524288 : index
            %c128_1 = arith.constant 128 : index
            %c32 = arith.constant 32 : index
            %0 = arith.muli %arg6, %c32 : index
            %1 = arith.muli %arg7, %c32 : index
            %2 = arith.addi %arg10, %0 : index
            %3 = arith.addi %arg11, %1 : index
            %4 = memref.alloc() : memref<32x32x128xi32, 2>
            %5 = memref.alloc() : memref<32x32x128xi32, 2>
            %6 = memref.alloc() : memref<32x32x128xi32, 2>
            air.dma_memcpy_nd (%4[] [] [], %arg12[%2, %3, %arg13] [%c32, %c32, %c128_1] [%c524288, %c512_0, %c1]) {id = 1 : i32} : (memref<32x32x128xi32, 2>, memref<4096x1024x512xi32>)
            air.dma_memcpy_nd (%5[] [] [], %arg14[%2, %3, %arg13] [%c32, %c32, %c128_1] [%c524288, %c512_0, %c1]) {id = 2 : i32} : (memref<32x32x128xi32, 2>, memref<4096x1024x512xi32>)
            air.dma_memcpy_nd (%6[] [] [], %arg15[%2, %3, %arg13] [%c32, %c32, %c128_1] [%c524288, %c512_0, %c1]) {id = 3 : i32} : (memref<32x32x128xi32, 2>, memref<4096x1024x512xi32>)
            air.dma_memcpy_nd (%arg15[%2, %3, %arg13] [%c32, %c32, %c128_1] [%c524288, %c512_0, %c1], %6[] [] []) {id = 4 : i32} : (memref<4096x1024x512xi32>, memref<32x32x128xi32, 2>)
            memref.dealloc %4 : memref<32x32x128xi32, 2>
            memref.dealloc %5 : memref<32x32x128xi32, 2>
            memref.dealloc %6 : memref<32x32x128xi32, 2>
          }
        }
      }
    }
    return
  }

  // CHECK-LABEL: func4

  // No air.launch or air.segment.

  // CHECK: air.channel.put  @channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c4{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c4096{{.*}}, %c32{{.*}}, %c128{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<128x128xbf16>)

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
}
