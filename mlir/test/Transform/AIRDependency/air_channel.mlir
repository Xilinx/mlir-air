//===- air_channel.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// Trace dependencies with air.channel put/get ops



#map0 = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<24576x1024xbf16>, %arg1: memref<1024x1024xbf16>) -> memref<24576x1024xbf16> {
    %c16 = arith.constant 16 : index
    %c384 = arith.constant 384 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %1 = memref.alloc() {alignment = 128 : i64} : memref<24576x1024xbf16>
    air.launch (%arg2, %arg3) in (%arg4=%c384, %arg5=%c16) args(%arg6=%arg0, %arg7=%arg1, %arg8=%1) : memref<24576x1024xbf16>, memref<1024x1024xbf16>, memref<24576x1024xbf16> {
      air.channel @channel_1 [1]
      air.channel @channel_2 [1]
      air.channel @channel_3 [1]
      air.channel @channel_4 [1]
      %c0_new = arith.constant 0 : index
      %c1_new = arith.constant 1 : index
      %c1024_new = arith.constant 1024 : index
      %c64_new = arith.constant 64 : index
      %c384_new = arith.constant 384 : index
    
      %17 = affine.apply #map0()[%arg2]
      %18 = affine.apply #map0()[%arg3]
// CHECK: %[[EVENT0:.*]] = scf.for{{.*}}iter_args(%[[EVENT1:.*]] = 
      scf.for %arg16 = %c0_new to %c1024_new step %c64_new {
// CHECK: %[[EVENT2:.*]] = air.channel.put async [%[[EVENT1]]]{{.*}}@channel_1[]
        air.channel.put @channel_1[] (%arg6[%17, %arg16] [%c64_new, %c64_new] [%c1024_new, %c1_new]) : (memref<24576x1024xbf16>)
// CHECK: %[[EVENT3:.*]] = air.channel.put async [%[[EVENT1]]]{{.*}}@channel_2[]
        air.channel.put @channel_2[] (%arg7[%arg16, %18] [%c64_new, %c64_new] [%c1024_new, %c1_new]) : (memref<1024x1024xbf16>)
      }
// CHECK: %[[EVENT4:.*]] = air.wait_all async [%[[EVENT1]], %[[EVENT2]], %[[EVENT3]]]
// CHECK: %[[EVENT5:.*]] = air.channel.put async [%[[EVENT6:.*]], %[[EVENT7:.*]]]{{.*}}@channel_3[]
      air.channel.put @channel_3[] (%arg8[%17, %18] [%c64_new, %c64_new] [%c1024_new, %c1_new]) : (memref<24576x1024xbf16>)
      
      air.segment  args(%arg9=%arg2, %arg10=%arg3, %arg11=%arg4, %arg12=%arg5, %arg13=%arg6, %arg14=%arg7) : index, index, index, index, memref<24576x1024xbf16>, memref<1024x1024xbf16> {
        air.channel @channel_5 [2, 2]
        air.channel @channel_6 [2, 2]
        air.channel @channel_7 [2, 2]
        air.channel @channel_8 [2, 2]
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c32_new = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        
// CHECK: %[[EVENT8:.*]], %[[VALUE0:.*]] = air.execute
        %19 = memref.alloc() : memref<64x64xbf16, 1>
// CHECK: %[[EVENT9:.*]] = air.channel.get async [%[[EVENT8]]]{{.*}}@channel_3[]
// CHECK: %[[EVENT10:.*]] = air.wait_all async [%[[EVENT9]]]
        air.channel.get @channel_3[] (%19[] [] []) : (memref<64x64xbf16, 1>)
        scf.for %arg16 = %c0 to %c1024 step %c64 {

          %5 = memref.alloc() : memref<64x64xbf16, 1>
          %6 = memref.alloc() : memref<64x64xbf16, 1>
// CHECK: %[[EVENT11:.*]] = air.channel.get async{{.*}}@channel_1[]
          air.channel.get @channel_1[] (%5[] [] []) : (memref<64x64xbf16, 1>)
// CHECK: %[[EVENT12:.*]] = air.channel.get async{{.*}}@channel_2[]
          air.channel.get @channel_2[] (%6[] [] []) : (memref<64x64xbf16, 1>)

// CHECK: %[[EVENT13:.*]] = scf.parallel
          scf.parallel (%arg17, %arg18) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
            %20 = affine.apply #map1()[%arg17]
            %21 = affine.apply #map1()[%arg18]
// CHECK: %[[EVENT14:.*]] = scf.for{{.*}}iter_args(%[[EVENT15:.*]] = 
            scf.for %arg24 = %c0 to %c64 step %c32_new {
              air.channel.put @channel_5[%arg17, %arg18] (%5[%20, %arg24] [%c64, %c64] [%c1024, %c1]) : (memref<64x64xbf16, 1>)
// CHECK: %[[EVENT16:.*]] = air.channel.put async [%[[EVENT15]]{{.*}}@channel_5[
              air.channel.put @channel_6[%arg17, %arg18] (%6[%arg24, %21] [%c64, %c64] [%c1024, %c1]) : (memref<64x64xbf16, 1>)
// CHECK: %[[EVENT17:.*]] = air.channel.put async [%[[EVENT15]]{{.*}}@channel_6[
            }
// CHECK: %[[EVENT18:.*]] = air.wait_all async [%[[EVENT15]], %[[EVENT16]], %[[EVENT17]]]
// CHECK: %[[EVENT19:.*]] = air.channel.put async [{{.*}}@channel_7[
            air.channel.put @channel_7[%arg17, %arg18] (%19[%20, %21] [%c32_new, %c32_new] [%c64, %c1]) : (memref<64x64xbf16, 1>)
          }
          air.herd @herd_0  tile (%arg17, %arg18) in (%arg19=%c2, %arg20=%c2) args(%arg21=%5, %arg22=%6, %arg23=%19) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> {
            %c1_0 = arith.constant 1 : index
            %c0_1 = arith.constant 0 : index
            %c64_2 = arith.constant 64 : index
            %c32 = arith.constant 32 : index
// CHECK: %[[EVENT20:.*]], %[[VALUE1:.*]] = air.execute
            %12 = memref.alloc() : memref<32x32xbf16, 2>
// CHECK: %[[EVENT21:.*]] = air.channel.get async [%[[EVENT20]]]{{.*}}@channel_7[
            air.channel.get @channel_7[%arg17, %arg18] (%12[] [] []) : (memref<32x32xbf16, 2>)
// CHECK: %[[EVENT22:.*]] = air.wait_all async [%[[EVENT21]]]
// CHECK: %[[EVENT23:.*]] = scf.for{{.*}}iter_args(%[[EVENT24:.*]] = 
            scf.for %arg24 = %c0_1 to %c64_2 step %c32 {
// CHECK: %[[EVENT25:.*]], %[[VALUE2:.*]] = air.execute
              %10 = memref.alloc() : memref<32x32xbf16, 2>
// CHECK: %[[EVENT26:.*]], %[[VALUE3:.*]] = air.execute
              %11 = memref.alloc() : memref<32x32xbf16, 2>
// CHECK: %[[EVENT27:.*]] = air.channel.get async [%[[EVENT24]], %[[EVENT25]]]{{.*}}@channel_5[
              air.channel.get @channel_5[%arg17, %arg18] (%10[] [] []) : (memref<32x32xbf16, 2>)
// CHECK: %[[EVENT28:.*]] = air.channel.get async [%[[EVENT24]], %[[EVENT26]]]{{.*}}@channel_6[
              air.channel.get @channel_6[%arg17, %arg18] (%11[] [] []) : (memref<32x32xbf16, 2>)
// CHECK: %[[EVENT29:.*]] = air.execute [%[[EVENT28]], %[[EVENT27]]
              linalg.matmul ins(%10, %11 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%12 : memref<32x32xbf16, 2>)
              memref.dealloc %10 : memref<32x32xbf16, 2>
              memref.dealloc %11 : memref<32x32xbf16, 2>
            }
// CHECK: %[[EVENT30:.*]] = air.channel.put async [%[[EVENT23]]{{.*}}@channel_8[
            air.channel.put @channel_8[%arg17, %arg18] (%12[] [] []) : (memref<32x32xbf16, 2>)
// CHECK: %[[EVENT31:.*]] = air.execute [%[[EVENT30]]]
            memref.dealloc %12 : memref<32x32xbf16, 2>
            air.herd_terminator
          }
          scf.parallel (%arg17, %arg18) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
            %20 = affine.apply #map1()[%arg17]
            %21 = affine.apply #map1()[%arg18]
// CHECK: %[[EVENT32:.*]] = air.channel.get async{{.*}}@channel_8[
            air.channel.get @channel_8[%arg17, %arg18] (%19[%20, %21] [%c64, %c64] [%c1024, %c1]) : (memref<64x64xbf16, 1>)
// CHECK: %[[EVENT33:.*]] = air.wait_all async [%[[EVENT32]]]
          }
          memref.dealloc %5 : memref<64x64xbf16, 1>
          memref.dealloc %6 : memref<64x64xbf16, 1>
        }
// CHECK: %[[EVENT34:.*]] = air.channel.put async{{.*}}@channel_4[]
        air.channel.put @channel_4[] (%19[] [] []) : (memref<64x64xbf16, 1>)
        memref.dealloc %19 : memref<64x64xbf16, 1>
        air.segment_terminator
      }

// CHECK: %[[EVENT35:.*]] = air.channel.get async{{.*}}@channel_4[]
      air.channel.get @channel_4[] (%arg8[%17, %18] [%c64_new, %c64_new] [%c1024_new, %c1_new]) : (memref<24576x1024xbf16>)
      
      air.launch_terminator
    }
    return %1 : memref<24576x1024xbf16>
  }
}
