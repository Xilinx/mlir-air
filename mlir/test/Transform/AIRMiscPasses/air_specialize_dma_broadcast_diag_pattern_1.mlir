// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-specialize-dma-broadcast | FileCheck %s

// Lowers broadcastable DMAs using affine.if, with herd size 4x4
// CHECK: [[$SET0:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 floordiv 2 + s1 * 2 == 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 floordiv 2 + s1 * 2 - 1 == 0)>
// CHECK: [[$SET2:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 floordiv 2 + s1 * 2 - 2 == 0)>
// CHECK: [[$SET3:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 floordiv 2 + s1 * 2 - 3 == 0)>
// CHECK: [[$SET4:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 floordiv 2 + s1 * 2 - 4 == 0)>
// CHECK: [[$SET5:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 floordiv 2 + s1 * 2 - 5 == 0)>
// CHECK: [[$SET6:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 floordiv 2 + s1 * 2 - 6 == 0)>
// CHECK: [[$SET7:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 floordiv 2 + s1 * 2 - 7 == 0)>
// CHECK: %[[EVENT0:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET0]]{{.*}}
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET1]]{{.*}}
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET2]]{{.*}}
// CHECK: %[[EVENT3:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET3]]{{.*}}
// CHECK: %[[EVENT4:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET4]]{{.*}}
// CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET5]]{{.*}}
// CHECK: %[[EVENT6:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET6]]{{.*}}
// CHECK: %[[EVENT7:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET7]]{{.*}}
// CHECK: %[[EVENT8:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET0]]{{.*}}
// CHECK: %[[EVENT9:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET1]]{{.*}}
// CHECK: %[[EVENT10:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET2]]{{.*}}
// CHECK: %[[EVENT11:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET3]]{{.*}}
// CHECK: %[[EVENT12:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET4]]{{.*}}
// CHECK: %[[EVENT13:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET5]]{{.*}}
// CHECK: %[[EVENT14:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET6]]{{.*}}
// CHECK: %[[EVENT15:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET7]]{{.*}}

#map = affine_map<()[s0] -> (s0 * 32)>
#set0 = affine_set<(d0, d1)[s0] : (d0 floordiv 2 + d1 * 2 - s0 == 0, s0 >= 0, -s0 + 7 >= 0)>
module {
  func.func @matmul(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %asyncToken_1, %valOut_2 = air.region async {
      %8 = memref.alloc() : memref<64x64xbf16, 1>
      air.region_terminator %8 : memref<64x64xbf16, 1>
    } {id = 3 : i32} : (memref<64x64xbf16, 1>)
    %asyncToken_3, %valOut_4 = air.region async {
      %8 = memref.alloc() : memref<64x64xbf16, 1>
      air.region_terminator %8 : memref<64x64xbf16, 1>
    } {id = 4 : i32} : (memref<64x64xbf16, 1>)
    %asyncToken_5, %valOut_6 = air.region async {
      %8 = memref.alloc() : memref<64x64xbf16, 1>
      air.region_terminator %8 : memref<64x64xbf16, 1>
    } {id = 5 : i32} : (memref<64x64xbf16, 1>)
    %5 = air.herd async [%asyncToken_1, %asyncToken_3, %asyncToken_5]  tile (%arg7, %arg8) in (%arg9=%c4, %arg10=%c4) args(%arg11=%valOut_2, %arg12=%valOut_4, %arg13=%valOut_6) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 1 : i32, sym_name = "herd_0"} {
      %c1_10 = arith.constant 1 : index
      %c64_11 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c0_12 = arith.constant 0 : index
      %asyncToken_13, %valOut_14 = air.region async  {
        %12 = affine.apply #map()[%arg7]
        air.region_terminator %12 : index
      } {id = 6 : i32} : (index)
      %asyncToken_15, %valOut_16 = air.region async  {
        %12 = affine.apply #map()[%arg8]
        air.region_terminator %12 : index
      } {id = 7 : i32} : (index)
      %8 = air.wait_all async [%asyncToken_13, %asyncToken_15] 
      %asyncToken_17, %valOut_18 = air.region async  {
        %12 = memref.alloc() : memref<32x32xbf16, 2>
        air.region_terminator %12 : memref<32x32xbf16, 2>
      } {id = 10 : i32} : (memref<32x32xbf16, 2>)
      %9 = air.dma_memcpy_nd async [%8, %asyncToken_17] (%valOut_18[] [] [], %arg13[%valOut_14, %valOut_16] [%c32, %c32] [%c64_11, %c1_10]) {id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
      %10 = scf.for %arg14 = %c0_12 to %c64_11 step %c32 iter_args(%arg15 = %9) -> (!air.async.token) {
        %asyncToken_20, %valOut_21 = air.region async [%arg15]  : (!air.async.token) {
          %15 = memref.alloc() : memref<32x32xbf16, 2>
          air.region_terminator %15 : memref<32x32xbf16, 2>
        } {id = 8 : i32} : (memref<32x32xbf16, 2>)
        %asyncToken_22, %valOut_23 = air.region async [%arg15]  : (!air.async.token) {
          %15 = memref.alloc() : memref<32x32xbf16, 2>
          air.region_terminator %15 : memref<32x32xbf16, 2>
        } {id = 9 : i32} : (memref<32x32xbf16, 2>)
        %12 = air.dma_memcpy_nd async [%asyncToken_20, %arg15] (%valOut_21[] [] [], %arg11[%valOut_14, %arg14] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_pattern = #set0, id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
        %13 = air.dma_memcpy_nd async [%asyncToken_22, %arg15] (%valOut_23[] [] [], %arg12[%arg14, %valOut_16] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_pattern = #set0, id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
        %asyncToken_24 = air.region async [%13, %arg15, %12]  : (!air.async.token, !air.async.token, !air.async.token) {
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%valOut_21, %valOut_23 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%valOut_18 : memref<32x32xbf16, 2>)
          air.region_terminator
        } {id = 11 : i32}
        %asyncToken_25 = air.region async [%asyncToken_24]  : (!air.async.token) {
          memref.dealloc %valOut_21 : memref<32x32xbf16, 2>
          air.region_terminator
        } {id = 12 : i32}
        %asyncToken_26 = air.region async [%asyncToken_24]  : (!air.async.token) {
          memref.dealloc %valOut_23 : memref<32x32xbf16, 2>
          air.region_terminator
        } {id = 13 : i32}
        %14 = air.wait_all async [%asyncToken_24, %asyncToken_25, %asyncToken_26] 
        scf.yield %14 : !air.async.token
      }
      %11 = air.dma_memcpy_nd async [%10] (%arg13[%valOut_14, %valOut_16] [%c32, %c32] [%c64_11, %c1_10], %valOut_18[] [] []) {broadcast = "both", id = 7 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
      %asyncToken_19 = air.region async [%11]  : (!air.async.token) {
        memref.dealloc %valOut_18 : memref<32x32xbf16, 2>
        air.region_terminator
      } {id = 14 : i32}
      air.herd_terminator
    }
    return
  }
}