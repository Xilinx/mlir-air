//===- opt_shim_dma_bds.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1" | FileCheck %s
// RUN: air-opt %s -air-opt-shim-dma-bds="device=npu1 shim-dma-tile-sizes=2,2" | FileCheck %s --check-prefix=NPUTILED
// RUN: air-opt %s -air-opt-shim-dma-bds="device=xcvc1902" | FileCheck %s --check-prefix=AIE1

// Optimize logical air.channel.put/get op into efficient shim dma block descriptor (BD).

module {

  // Three scf.for loop nested around air.channle.put containing two effective wrap-and-stride dimensions. 
  // Specialize two inner-most for loops into the wrap-and-stride list, and leave one outer-most for loop unchanged.

  // CHECK-LABEL: func0
  // CHECK: %[[PUT0:.*]] = air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId22} : (memref<512x512xbf16>)
  // CHECK-NEXT: %[[PUT1:.*]] = air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId22} : (memref<512x512xbf16>)
  // CHECK: %[[PUT2:.*]] = air.channel.put async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c32768{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId23} : (memref<512x512xbf16>)
  // CHECK-NEXT: %[[PUT3:.*]] = air.channel.put async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c32768{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId23} : (memref<512x512xbf16>)
  // CHECK: %[[PUT4:.*]] = air.channel.put async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c65536{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId24} : (memref<512x512xbf16>)
  // CHECK-NEXT: %[[PUT5:.*]] = air.channel.put async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c65536{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId24} : (memref<512x512xbf16>)
  // CHECK: %[[PUT6:.*]] = air.channel.put async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c98304{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId25} : (memref<512x512xbf16>)
  // CHECK-NEXT: %[[PUT7:.*]] = air.channel.put async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c98304{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId25} : (memref<512x512xbf16>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[PUT3]]{{.*}}%[[PUT4]]{{.*}}%[[PUT5]]{{.*}}%[[PUT6]]{{.*}}%[[PUT7]]{{.*}}]

  // NPUTILED-LABEL: func0
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId22} : (memref<512x512xbf16>)
  // NPUTILED-NEXT: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[PUT0]]{{.*}}] @channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId22} : (memref<512x512xbf16>)
  // NPUTILED: %[[PUT2:.*]] = air.channel.put async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c32768{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId23} : (memref<512x512xbf16>)
  // NPUTILED-NEXT: %[[PUT3:.*]] = air.channel.put async [{{.*}}%[[PUT2]]{{.*}}] @channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c32768{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId23} : (memref<512x512xbf16>)
  // NPUTILED: %[[PUT4:.*]] = air.channel.put async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c65536{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId24} : (memref<512x512xbf16>)
  // NPUTILED-NEXT: %[[PUT5:.*]] = air.channel.put async [{{.*}}%[[PUT4]]{{.*}}] @channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c65536{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId24} : (memref<512x512xbf16>)
  // NPUTILED: %[[PUT6:.*]] = air.channel.put async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c98304{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId25} : (memref<512x512xbf16>)
  // NPUTILED-NEXT: %[[PUT7:.*]] = air.channel.put async [{{.*}}%[[PUT6]]{{.*}}] @channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c98304{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId25} : (memref<512x512xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT1]]{{.*}}%[[PUT3]]{{.*}}%[[PUT5]]{{.*}}%[[PUT7]]{{.*}}]
  
  // AIE1-LABEL: func0
  // AIE1-COUNT-3: scf.for
  // AIE1-COUNT-4: air.channel.put
  // AIE1-COUNT-3: scf.yield

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

  // CHECK-LABEL: func1
  // CHECK: %[[PUT0:.*]] = air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId26} : (memref<512x512xbf16>)
  // CHECK: %[[PUT1:.*]] = air.channel.put async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c64{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId27} : (memref<512x512xbf16>)
  // CHECK: %[[PUT2:.*]] = air.channel.put async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c128{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId28} : (memref<512x512xbf16>)
  // CHECK: %[[PUT3:.*]] = air.channel.put async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c192{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId29} : (memref<512x512xbf16>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[PUT3]]{{.*}}]
  
  // NPUTILED-LABEL: func1
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId26} : (memref<512x512xbf16>)
  // NPUTILED: %[[PUT1:.*]] = air.channel.put async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c64{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId27} : (memref<512x512xbf16>)
  // NPUTILED: %[[PUT2:.*]] = air.channel.put async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c128{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId28} : (memref<512x512xbf16>)
  // NPUTILED: %[[PUT3:.*]] = air.channel.put async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c192{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId29} : (memref<512x512xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[PUT3]]{{.*}}]
  
  // AIE1-LABEL: func1
  // AIE1-COUNT-3: scf.for
  // AIE1-COUNT-4: air.channel.put
  // AIE1-COUNT-3: scf.yield

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

  // CHECK-LABEL: func2
  // CHECK: %[[GET0:.*]] = air.channel.get async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId39} : (memref<512x512xbf16>)
  // CHECK: %[[GET1:.*]] = air.channel.get async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId41} : (memref<512x512xbf16>)
  // CHECK: %[[GET2:.*]] = air.channel.get async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c128{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId43} : (memref<512x512xbf16>)
  // CHECK: %[[GET3:.*]] = air.channel.get async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c192{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId45} : (memref<512x512xbf16>)
  // CHECK: air.wait_all [{{.*}}%[[GET0]]{{.*}}%[[GET1]]{{.*}}%[[GET2]]{{.*}}%[[GET3]]{{.*}}]
  
  // NPUTILED-LABEL: func2
  // NPUTILED: %[[GET0:.*]] = air.channel.get async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId39} : (memref<512x512xbf16>)
  // NPUTILED: %[[GET1:.*]] = air.channel.get async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId41} : (memref<512x512xbf16>)
  // NPUTILED: %[[GET2:.*]] = air.channel.get async{{.*}}@channel_0[%c2{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c128{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId43} : (memref<512x512xbf16>)
  // NPUTILED: %[[GET3:.*]] = air.channel.get async{{.*}}@channel_0[%c3{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c192{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId45} : (memref<512x512xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[GET0]]{{.*}}%[[GET1]]{{.*}}%[[GET2]]{{.*}}%[[GET3]]{{.*}}]

  // AIE1-LABEL: func2
  // AIE1-COUNT-2: scf.for
  // AIE1-COUNT-4: air.channel.get
  // AIE1-COUNT-2: scf.yield

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

  // Hoisting air.herd op out of any parent scf.for loop nest.

  // CHECK-LABEL: func3
  // CHECK: air.herd
  // CHECK-COUNT-3: scf.for
  // CHECK-COUNT-4: air.dma_memcpy_nd
  // CHECK-COUNT-4: }
  
  // NPUTILED-LABEL: func3
  // NPUTILED-COUNT-256: air.dma_memcpy_nd

  // AIE1-LABEL: func3
  // AIE1-COUNT-3: scf.for
  // AIE1: air.herd
  // AIE1-COUNT-4: air.dma_memcpy_nd
  // AIE1-COUNT-4: }

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

  // No air.launch or air.segment.

  // CHECK-LABEL: func4
  // CHECK: air.channel.put  @channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c4{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c4096{{.*}}, %c32{{.*}}, %c128{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<128x128xbf16>)
  
  // NPUTILED-LABEL: func4
  // NPUTILED: air.channel.put  @channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c4096{{.*}}, %c32{{.*}}, %c128{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<128x128xbf16>)
  // NPUTILED-NEXT: air.channel.put  @channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c64{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c4096{{.*}}, %c32{{.*}}, %c128{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<128x128xbf16>)
  // NPUTILED-NEXT: air.channel.put  @channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c4096{{.*}}, %c32{{.*}}, %c128{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<128x128xbf16>)
  // NPUTILED-NEXT: air.channel.put  @channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c32{{.*}}, %c32{{.*}}] [%c4096{{.*}}, %c32{{.*}}, %c128{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<128x128xbf16>)

  // AIE1-LABEL: func4
  // AIE1-COUNT-2: scf.for
  // AIE1: air.channel.put
  // AIE1-COUNT-2: }

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

  // CHECK-LABEL: func5
  // CHECK: %[[WAITALL0:.*]] = air.wait_all async
  // CHECK: %[[PUT0:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c1{{.*}}, %c1{{.*}}, %c32{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<8x8xi32>)
  // CHECK: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[PUT0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c32{{.*}}] [%c2{{.*}}, %c1{{.*}}, %c1{{.*}}, %c32{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<8x8xi32>)
  // CHECK: %[[PUT2:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c8{{.*}}, %c4{{.*}}] [%c0{{.*}}, %c4{{.*}}, %c8{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId5} : (memref<8x8xi32>)
  // CHECK: %[[GET0:.*]] = air.channel.get async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_2[] (%arg2[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c4{{.*}}, %c4{{.*}}] [%c32{{.*}}, %c4{{.*}}, %c8{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId16} : (memref<8x8xi32>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[GET0]]{{.*}}] 

  // NPUTILED-LABEL: func5
  // NPUTILED: %[[WAITALL0:.*]] = air.wait_all async
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c1{{.*}}, %c1{{.*}}, %c32{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<8x8xi32>)
  // NPUTILED: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[PUT0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c32{{.*}}] [%c2{{.*}}, %c1{{.*}}, %c1{{.*}}, %c32{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<8x8xi32>)
  // NPUTILED: %[[PUT2:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c8{{.*}}, %c4{{.*}}] [%c0{{.*}}, %c4{{.*}}, %c8{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId5} : (memref<8x8xi32>)
  // NPUTILED: %[[GET0:.*]] = air.channel.get async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_2[] (%arg2[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c4{{.*}}, %c4{{.*}}] [%c32{{.*}}, %c4{{.*}}, %c8{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId16} : (memref<8x8xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[GET0]]{{.*}}] 
  
  // AIE1-LABEL: func5
  // AIE1-COUNT-2: scf.for
  // AIE1: air.channel.put
  // AIE1: air.channel.put
  // AIE1: air.channel.get
  // AIE1-COUNT-2: }

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

  // CHECK-LABEL: func6
  // CHECK: %[[WAITALL0:.*]] = air.wait_all async
  // CHECK: %[[PUT0:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c1{{.*}}, %c1{{.*}}, %c128{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<8x16xi32>)
  // CHECK: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c16{{.*}}, %c16{{.*}}] [%c16{{.*}}, %c32{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId5} : (memref<16x32xi32>)
  // CHECK: %[[GET0:.*]] = air.channel.get async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_2[] (%arg2[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c16{{.*}}] [%c16{{.*}}, %c32{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId12} : (memref<8x32xi32>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[GET0]]{{.*}}] 

  // NPUTILED-LABEL: func6
  // NPUTILED: %[[WAITALL0:.*]] = air.wait_all async
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c1{{.*}}, %c1{{.*}}, %c128{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId4} : (memref<8x16xi32>)
  // NPUTILED: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c16{{.*}}, %c16{{.*}}] [%c16{{.*}}, %c32{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId5} : (memref<16x32xi32>)
  // NPUTILED: %[[GET0:.*]] = air.channel.get async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_2[] (%arg2[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c8{{.*}}, %c16{{.*}}] [%c16{{.*}}, %c32{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId12} : (memref<8x32xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[GET0]]{{.*}}] 
  
  // AIE1-LABEL: func6
  // AIE1-COUNT-2: scf.for
  // AIE1: air.channel.put
  // AIE1: air.channel.put
  // AIE1: air.channel.get
  // AIE1-COUNT-2: }

  
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

  // CHECK-LABEL: func7
  // CHECK: %[[WAITALL0:.*]] = air.wait_all async
  // CHECK: %[[PUT0:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // CHECK: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[PUT0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // CHECK: %[[PUT2:.*]] = air.channel.put async [{{.*}}%[[PUT1]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c128{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // CHECK: %[[PUT3:.*]] = air.channel.put async [{{.*}}%[[PUT2]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c192{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c8{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // CHECK: %[[PUT4:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c4{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c64{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<512x2048xi32>)
  // CHECK: %[[GET0:.*]] = air.channel.get async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_2[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c4{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c131072{{.*}}, %c64{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[PUT3]]{{.*}}%[[PUT4]]{{.*}}%[[GET0]]{{.*}}] 

  // NPUTILED-LABEL: func7
  // NPUTILED: %[[WAITALL0:.*]] = air.wait_all async
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c0, %c0] [%c2, %c8, %c64, %c64] [%c0, %c64, %c512, %c1]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // NPUTILED: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[PUT0]]{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c64, %c0] [%c2, %c8, %c64, %c64] [%c0, %c64, %c512, %c1]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // NPUTILED: %[[PUT2:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c0] [%c2, %c2, %c512, %c64] [%c0, %c64, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<512x2048xi32>)
  // NPUTILED: %[[GET0:.*]] = air.channel.get async [%{{.*}}]  @channel_2[] (%alloc[%c0, %c0, %c0, %c0] [%c2, %c2, %c64, %c64] [%c131072, %c64, %c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[GET0]]{{.*}}] 
  // NPUTILED: %[[PUT3:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c0, %c0] [%c2, %c8, %c64, %c64] [%c0, %c64, %c512, %c1]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // NPUTILED: %[[PUT4:.*]] = air.channel.put async [{{.*}}%[[PUT3]]{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c64, %c0] [%c2, %c8, %c64, %c64] [%c0, %c64, %c512, %c1]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // NPUTILED: %[[PUT5:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c128] [%c2, %c2, %c512, %c64] [%c0, %c64, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<512x2048xi32>)
  // NPUTILED: %[[GET1:.*]] = air.channel.get async [%{{.*}}]  @channel_2[] (%alloc[%c0, %c0, %c0, %c128] [%c2, %c2, %c64, %c64] [%c131072, %c64, %c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT4]]{{.*}}%[[PUT5]]{{.*}}%[[GET1]]{{.*}}] 
  // NPUTILED: %[[PUT6:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c128, %c0] [%c2, %c8, %c64, %c64] [%c0, %c64, %c512, %c1]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // NPUTILED: %[[PUT7:.*]] = air.channel.put async [{{.*}}%[[PUT6]]{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c192, %c0] [%c2, %c8, %c64, %c64] [%c0, %c64, %c512, %c1]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // NPUTILED: %[[PUT8:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c0] [%c2, %c2, %c512, %c64] [%c0, %c64, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<512x2048xi32>)
  // NPUTILED: %[[GET2:.*]] = air.channel.get async [%{{.*}}]  @channel_2[] (%alloc[%c0, %c0, %c128, %c0] [%c2, %c2, %c64, %c64] [%c131072, %c64, %c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT7]]{{.*}}%[[PUT8]]{{.*}}%[[GET2]]{{.*}}] 
  // NPUTILED: %[[PUT9:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c128, %c0] [%c2, %c8, %c64, %c64] [%c0, %c64, %c512, %c1]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // NPUTILED: %[[PUT10:.*]] = air.channel.put async [{{.*}}%[[PUT9]]{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c192, %c0] [%c2, %c8, %c64, %c64] [%c0, %c64, %c512, %c1]) {metadata = @airMemcpyId20} : (memref<2048x512xi32>)
  // NPUTILED: %[[PUT11:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c128] [%c2, %c2, %c512, %c64] [%c0, %c64, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<512x2048xi32>)
  // NPUTILED: %[[GET3:.*]] = air.channel.get async [%{{.*}}]  @channel_2[] (%alloc[%c0, %c0, %c128, %c128] [%c2, %c2, %c64, %c64] [%c131072, %c64, %c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT10]]{{.*}}%[[PUT11]]{{.*}}%[[GET3]]{{.*}}] 
  
  // AIE1-LABEL: func7
  // AIE1-COUNT-2: scf.for
  // AIE1: air.channel.put
  // AIE1: air.channel.put
  // AIE1: air.channel.get
  // AIE1-COUNT-2: }

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

  // CHECK-LABEL: func8
  // CHECK: %[[WAITALL0:.*]] = air.wait_all async
  // CHECK: %[[PUT0:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c8{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // CHECK: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[PUT0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c8{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // CHECK: %[[PUT2:.*]] = air.channel.put async [{{.*}}%[[PUT1]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c128{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c8{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // CHECK: %[[PUT3:.*]] = air.channel.put async [{{.*}}%[[PUT2]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c192{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c8{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // CHECK: %[[PUT4:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c4{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1048576{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // CHECK: %[[PUT5:.*]] = air.channel.put async [{{.*}}%[[PUT4]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c4{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1048576{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // CHECK: %[[PUT6:.*]] = air.channel.put async [{{.*}}%[[PUT5]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c4{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1048576{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // CHECK: %[[PUT7:.*]] = air.channel.put async [{{.*}}%[[PUT6]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c4{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c64{{.*}}, %c1048576{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // CHECK: %[[GET0:.*]] = air.channel.get async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_2[] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c4{{.*}}, %c64{{.*}}, %c64{{.*}}] [%c131072{{.*}}, %c64{{.*}}, %c2048{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[PUT3]]{{.*}}%[[PUT4]]{{.*}}%[[PUT5]]{{.*}}%[[PUT6]]{{.*}}%[[PUT7]]{{.*}}%[[GET0]]{{.*}}] 

  // NPUTILED-LABEL: func8
  // NPUTILED: %[[WAITALL0:.*]] = air.wait_all async
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c0, %c0] [%c2, %c8, %c64, %c256] [%c0, %c256, %c2048, %c1]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[PUT0]]{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c64, %c0] [%c2, %c8, %c64, %c256] [%c0, %c256, %c2048, %c1]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT2:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c0] [%c2, %c4, %c512, %c64] [%c64, %c1048576, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT3:.*]] = air.channel.put async [{{.*}}%[[PUT2]]{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c0] [%c2, %c4, %c512, %c64] [%c64, %c1048576, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // NPUTILED: %[[GET0:.*]] = air.channel.get async [%{{.*}}]  @channel_2[] (%alloc[%c0, %c0, %c0, %c0] [%c2, %c2, %c64, %c64] [%c131072, %c64, %c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT1]]{{.*}}%[[PUT3]]{{.*}}%[[GET0]]{{.*}}] 
  // NPUTILED: %[[PUT4:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c0, %c0] [%c2, %c8, %c64, %c256] [%c0, %c256, %c2048, %c1]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT5:.*]] = air.channel.put async [{{.*}}%[[PUT4]]{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c64, %c0] [%c2, %c8, %c64, %c256] [%c0, %c256, %c2048, %c1]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT6:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c128] [%c2, %c4, %c512, %c64] [%c64, %c1048576, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT7:.*]] = air.channel.put async [{{.*}}%[[PUT6]]{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c128] [%c2, %c4, %c512, %c64] [%c64, %c1048576, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // NPUTILED: %[[GET1:.*]] = air.channel.get async [%{{.*}}]  @channel_2[] (%alloc[%c0, %c0, %c0, %c128] [%c2, %c2, %c64, %c64] [%c131072, %c64, %c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT5]]{{.*}}%[[PUT7]]{{.*}}%[[GET1]]{{.*}}] 
  // NPUTILED: %[[PUT8:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c128, %c0] [%c2, %c8, %c64, %c256] [%c0, %c256, %c2048, %c1]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT9:.*]] = air.channel.put async [{{.*}}%[[PUT8]]{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c192, %c0] [%c2, %c8, %c64, %c256] [%c0, %c256, %c2048, %c1]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT10:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c0] [%c2, %c4, %c512, %c64] [%c64, %c1048576, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT11:.*]] = air.channel.put async [{{.*}}%[[PUT10]]{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c0] [%c2, %c4, %c512, %c64] [%c64, %c1048576, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // NPUTILED: %[[GET2:.*]] = air.channel.get async [%{{.*}}]  @channel_2[] (%alloc[%c0, %c0, %c128, %c0] [%c2, %c2, %c64, %c64] [%c131072, %c64, %c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT9]]{{.*}}%[[PUT11]]{{.*}}%[[GET2]]{{.*}}] 
  // NPUTILED: %[[PUT12:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c128, %c0] [%c2, %c8, %c64, %c256] [%c0, %c256, %c2048, %c1]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT13:.*]] = air.channel.put async [{{.*}}%[[PUT12]]{{.*}}]  @channel_0[] (%arg0[%c0, %c0, %c192, %c0] [%c2, %c8, %c64, %c256] [%c0, %c256, %c2048, %c1]) {metadata = @airMemcpyId20} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT14:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c128] [%c2, %c4, %c512, %c64] [%c64, %c1048576, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // NPUTILED: %[[PUT15:.*]] = air.channel.put async [{{.*}}%[[PUT14]]{{.*}}]  @channel_1[] (%arg1[%c0, %c0, %c0, %c128] [%c2, %c4, %c512, %c64] [%c64, %c1048576, %c2048, %c1]) {metadata = @airMemcpyId21} : (memref<2048x2048xi32>)
  // NPUTILED: %[[GET3:.*]] = air.channel.get async [%{{.*}}]  @channel_2[] (%alloc[%c0, %c0, %c128, %c128] [%c2, %c2, %c64, %c64] [%c131072, %c64, %c2048, %c1]) {metadata = @airMemcpyId26} : (memref<2048x2048xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT13]]{{.*}}%[[PUT15]]{{.*}}%[[GET3]]{{.*}}] 
  
  // AIE1-LABEL: func8
  // AIE1-COUNT-2: scf.for
  // AIE1: air.channel.put
  // AIE1: air.channel.put
  // AIE1: air.channel.get
  // AIE1-COUNT-2: }
  
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

  // CHECK-LABEL: func9
  // CHECK: %[[WAITALL0:.*]] = air.wait_all async
  // CHECK: %[[PUT0:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c3{{.*}}, %c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c256{{.*}}, %c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // CHECK: %[[PUT1:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c3{{.*}}, %c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c256{{.*}}, %c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // CHECK: %[[PUT2:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c3{{.*}}, %c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c256{{.*}}, %c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}]

  // NPUTILED-LABEL: func9
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT0]]{{.*}}] 
  // NPUTILED: %[[PUT1:.*]] = air.channel.put async  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}] [%c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT1]]{{.*}}] 
  // NPUTILED: %[[PUT2:.*]] = air.channel.put async  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c512{{.*}}] [%c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT2]]{{.*}}] 
  // NPUTILED: %[[PUT3:.*]] = air.channel.put async  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT3]]{{.*}}] 
  // NPUTILED: %[[PUT4:.*]] = air.channel.put async  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}] [%c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT4]]{{.*}}] 
  // NPUTILED: %[[PUT5:.*]] = air.channel.put async  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c512{{.*}}] [%c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT5]]{{.*}}] 
  // NPUTILED: %[[PUT6:.*]] = air.channel.put async  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT6]]{{.*}}] 
  // NPUTILED: %[[PUT7:.*]] = air.channel.put async  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}] [%c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT7]]{{.*}}] 
  // NPUTILED: %[[PUT8:.*]] = air.channel.put async  @channel_1[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c512{{.*}}] [%c768{{.*}}, %c3{{.*}}, %c64{{.*}}] [%c6912{{.*}}, %c2304{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId21} : (memref<2304x2304xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT8]]{{.*}}] 
  
  // AIE1-LABEL: func9
  // AIE1-COUNT-2: scf.for
  // AIE1: air.channel.put
  // AIE1-COUNT-2: }

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

  // CHECK-LABEL: func10
  // CHECK: %[[WAITALL0:.*]] = air.wait_all async
  // CHECK: %[[PUT0:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c4{{.*}}, %c256{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c1024{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId7} : (memref<512x1024xbf16>)
  // CHECK: %[[PUT1:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c4{{.*}}, %c256{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c1024{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId7} : (memref<512x1024xbf16>)
  // CHECK: %[[PUT2:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c256{{.*}}] [%c256{{.*}}, %c262144{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId12} : (memref<1024x512xbf16>)
  // CHECK: %[[PUT3:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c256{{.*}}] [%c256{{.*}}, %c262144{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId12} : (memref<1024x512xbf16>)
  // CHECK: %[[GET0:.*]] = air.channel.get async [%{{.*}}]  @channel_2[%c0{{.*}}, %c0{{.*}}] (%arg2[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c131072{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId45} : (memref<512x512xbf16>)
  // CHECK: %[[GET1:.*]] = air.channel.get async [%{{.*}}]  @channel_2[%c0{{.*}}, %c1{{.*}}] (%arg2[%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c131072{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId46} : (memref<512x512xbf16>)
  // CHECK: %[[GET2:.*]] = air.channel.get async [%{{.*}}]  @channel_2[%c1{{.*}}, %c0{{.*}}] (%arg2[%c0{{.*}}, %c0{{.*}}, %c128{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c131072{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId47} : (memref<512x512xbf16>)
  // CHECK: %[[GET3:.*]] = air.channel.get async [%{{.*}}]  @channel_2[%c1{{.*}}, %c1{{.*}}] (%arg2[%c0{{.*}}, %c0{{.*}}, %c192{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c131072{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId48} : (memref<512x512xbf16>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[PUT3]]{{.*}}%[[GET0]]{{.*}}%[[GET1]]{{.*}}%[[GET2]]{{.*}}%[[GET3]]{{.*}}]

  // NPUTILED-LABEL: func10
  // NPUTILED: %[[WAITALL0:.*]] = air.wait_all async
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c4{{.*}}, %c256{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c1024{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId7} : (memref<512x1024xbf16>)
  // NPUTILED: %[[PUT1:.*]] = air.channel.put async [{{.*}}%[[PUT0]]{{.*}}]  @channel_0[] (%arg0[%c0{{.*}}, %c0{{.*}}, %c256{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c4{{.*}}, %c256{{.*}}, %c256{{.*}}] [%c0{{.*}}, %c256{{.*}}, %c1024{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId7} : (memref<512x1024xbf16>)
  // NPUTILED: %[[PUT2:.*]] = air.channel.put async [%{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c256{{.*}}] [%c256{{.*}}, %c262144{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId12} : (memref<1024x512xbf16>)
  // NPUTILED: %[[PUT3:.*]] = air.channel.put async [{{.*}}%[[PUT2]]{{.*}}]  @channel_1[] (%arg1[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c512{{.*}}, %c256{{.*}}] [%c256{{.*}}, %c262144{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId12} : (memref<1024x512xbf16>)
  // NPUTILED: %[[GET0:.*]] = air.channel.get async [%{{.*}}]  @channel_2[%c0{{.*}}, %c0{{.*}}] (%arg2[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c131072{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId45} : (memref<512x512xbf16>)
  // NPUTILED: %[[GET1:.*]] = air.channel.get async [%{{.*}}]  @channel_2[%c0{{.*}}, %c1{{.*}}] (%arg2[%c0{{.*}}, %c0{{.*}}, %c64{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c131072{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId46} : (memref<512x512xbf16>)
  // NPUTILED: %[[GET2:.*]] = air.channel.get async [%{{.*}}]  @channel_2[%c1{{.*}}, %c0{{.*}}] (%arg2[%c0{{.*}}, %c0{{.*}}, %c128{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c131072{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId47} : (memref<512x512xbf16>)
  // NPUTILED: %[[GET3:.*]] = air.channel.get async [%{{.*}}]  @channel_2[%c1{{.*}}, %c1{{.*}}] (%arg2[%c0{{.*}}, %c0{{.*}}, %c192{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c2{{.*}}, %c64{{.*}}, %c256{{.*}}] [%c131072{{.*}}, %c256{{.*}}, %c512{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId48} : (memref<512x512xbf16>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT1]]{{.*}}%[[PUT3]]{{.*}}%[[GET0]]{{.*}}%[[GET1]]{{.*}}%[[GET2]]{{.*}}%[[GET3]]{{.*}}]
  
  // AIE1-LABEL: func10
  // AIE1-COUNT-2: scf.for
  // AIE1: air.channel.put
  // AIE1: air.channel.put
  // AIE1: air.channel.get
  // AIE1: air.channel.get
  // AIE1: air.channel.get
  // AIE1: air.channel.get
  // AIE1-COUNT-2: }
  
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

  // CHECK-LABEL: func11
  // CHECK: %[[WAITALL0:.*]] = air.wait_all async
  // CHECK: %[[PUT0:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%alloc[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c4{{.*}}, %c19{{.*}}, %c28{{.*}}, %c128{{.*}}] [%c0{{.*}}, %c128{{.*}}, %c2432{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId26} : (memref<308x2432xi32>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}]

  // NPUTILED-LABEL: func11
  // NPUTILED: %[[WAITALL0:.*]] = air.wait_all async
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%alloc[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c19{{.*}}, %c28{{.*}}, %c128{{.*}}] [%c0{{.*}}, %c128{{.*}}, %c2432{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId26} : (memref<308x2432xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT0]]{{.*}}]
  // NPUTILED: %[[PUT1:.*]] = air.channel.put async [%{{.*}}]  @channel_0[] (%alloc[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c2{{.*}}, %c19{{.*}}, %c28{{.*}}, %c128{{.*}}] [%c0{{.*}}, %c128{{.*}}, %c2432{{.*}}, %c1{{.*}}]) {metadata = @airMemcpyId26} : (memref<308x2432xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT1]]{{.*}}]
  
  // AIE1-LABEL: func11
  // AIE1-COUNT-2: scf.for
  // AIE1: air.channel.put
  // AIE1-COUNT-2: }
  
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

  // CHECK-LABEL: func12
  // CHECK: %[[WAITALL0:.*]] = air.wait_all async
  // CHECK: %[[PUT0:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_0[] (%arg0[] [] []) {metadata = @airMemcpyId31} : (memref<2x64x64xi32>)
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}]

  // NPUTILED-LABEL: func12
  // NPUTILED: %[[WAITALL0:.*]] = air.wait_all async
  // NPUTILED: %[[PUT0:.*]] = air.channel.put async [{{.*}}%[[WAITALL0]]{{.*}}]  @channel_0[] (%arg0[] [] []) {metadata = @airMemcpyId31} : (memref<2x64x64xi32>)
  // NPUTILED: air.wait_all [{{.*}}%[[PUT0]]{{.*}}]
  
  // AIE1-LABEL: func12
  // AIE1-COUNT-4: scf.for
  // AIE1: air.channel.put
  // AIE1-COUNT-4: }
  
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

  // CHECK-LABEL: func13
  // CHECK: air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c32{{.*}}, %c32{{.*}}, %c4{{.*}}, %c2{{.*}}] [%c512{{.*}}, %c2{{.*}}, %c128{{.*}}, %c1{{.*}}])
  // CHECK: air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c1{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c64{{.*}}] [%c32{{.*}}, %c32{{.*}}, %c4{{.*}}, %c2{{.*}}] [%c512{{.*}}, %c2{{.*}}, %c128{{.*}}, %c1{{.*}}])
  // CHECK: air.channel.put async{{.*}}@channel_0[%c1{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c16384{{.*}}] [%c32{{.*}}, %c32{{.*}}, %c4{{.*}}, %c2{{.*}}] [%c512{{.*}}, %c2{{.*}}, %c128{{.*}}, %c1{{.*}}])
  // CHECK: air.channel.put async{{.*}}@channel_0[%c1{{.*}}, %c1{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c16448{{.*}}] [%c32{{.*}}, %c32{{.*}}, %c4{{.*}}, %c2{{.*}}] [%c512{{.*}}, %c2{{.*}}, %c128{{.*}}, %c1{{.*}}])

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

  // CHECK-LABEL: func14
  // CHECK: %[[PUT0:.*]] = air.channel.put async{{.*}}@channel_0
  // CHECK: %[[PUT1:.*]] = air.channel.put async{{.*}}@channel_0
  // CHECK: %[[PUT2:.*]] = air.channel.put async{{.*}}@channel_0
  // CHECK: %[[PUT3:.*]] = air.channel.put async{{.*}}@channel_0
  // CHECK: air.wait_all [{{.*}}%[[PUT0]]{{.*}}%[[PUT1]]{{.*}}%[[PUT2]]{{.*}}%[[PUT3]]{{.*}}]

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

  // Canonicalizing repeat dimension at highest dimension.

  // CHECK-LABEL: func15
  // CHECK: air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c320{{.*}}] [%c2{{.*}}, %c1{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c512{{.*}}, %c1{{.*}}])
  // NPUTILED-LABEL: func15
  // NPUTILED: air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c320{{.*}}] [%c2{{.*}}, %c1{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c512{{.*}}, %c1{{.*}}])
  // AIE1-LABEL: func15
  // AIE1: air.channel.put async{{.*}}@channel_0[%c0{{.*}}, %c0{{.*}}] (%{{.*}}[%c0{{.*}}, %c0{{.*}}, %c0{{.*}}, %c320{{.*}}] [%c2{{.*}}, %c1{{.*}}, %c512{{.*}}, %c64{{.*}}] [%c0{{.*}}, %c0{{.*}}, %c512{{.*}}, %c1{{.*}}])

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
}
