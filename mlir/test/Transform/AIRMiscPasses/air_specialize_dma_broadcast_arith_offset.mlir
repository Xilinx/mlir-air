// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-specialize-dma-broadcast | FileCheck %s

// Checks for affine_map simplification of src offset in a broadcast dma
// CHECK: [[$SET0:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK: [[$SET2:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
// CHECK: [[$SET3:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
// CHECK: %[[EVENT0:.*]] = affine.if [[$SET0]]
// CHECK: %[[CONST0:.*]] = arith.constant 2080 : index
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}%[[CONST0]]{{.*}}broadcast_set = [[$SET0]]{{.*}}
// CHECK: affine.yield %[[EVENT1]]
// CHECK: %[[CONST1:.*]] = arith.constant 2144 : index
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd {{.*}}%[[CONST1]]{{.*}}broadcast_set = [[$SET1]]{{.*}}
// CHECK: affine.yield %[[EVENT2]]
// CHECK: %[[EVENT3:.*]] = affine.if [[$SET2]]
// CHECK: %[[CONST2:.*]] = arith.constant 0 : index
// CHECK: %[[EVENT4:.*]] = air.dma_memcpy_nd {{.*}}%[[CONST2]]{{.*}}broadcast_set = [[$SET2]]{{.*}}
// CHECK: affine.yield %[[EVENT4]]
// CHECK: %[[CONST3:.*]] = arith.constant 32 : index
// CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd {{.*}}%[[CONST3]]{{.*}}broadcast_set = [[$SET3]]{{.*}}
// CHECK: affine.yield %[[EVENT5]]

#map0 = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0] -> (s0 floordiv 4)>
#map2 = affine_map<()[s0] -> (s0 * 32)>
#set0 = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
module {
  func.func @broadcast_offset(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>, %arg2: memref<64x64xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %5 = air.launch_herd async  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%arg0, %arg12=%arg1, %arg13=%arg2) : memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16> attributes {id = 1 : i32, sym_name = "herd_0"} {
        %c1_10 = arith.constant 1 : index
        %c64_11 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c0_12 = arith.constant 0 : index
        %newAsyncToken_0, %newValOut_0 = air.region async  {
            %12 = affine.apply #map0()[%arg7]
            air.region_terminator %12 : index
        } {id = 6 : i32} : (index)
        %newAsyncToken_1, %newValOut_1 = air.region async [%newAsyncToken_0]  : (!air.async.token) {
            %12 = affine.apply #map1()[%newValOut_0]
            air.region_terminator %12 : index
        } {id = 6 : i32} : (index)
        %newAsyncToken_2, %newValOut_2 = air.region async [%newAsyncToken_1]  : (!air.async.token) {
            %12 = arith.addi %newValOut_1, %c1_10 : index
            air.region_terminator %12 : index
        } {id = 6 : i32} : (index)
        %newAsyncToken_3, %newValOut_3 = air.region async [%newAsyncToken_2]  : (!air.async.token) {
            %12 = arith.addi %newValOut_2, %c64_11 : index
            air.region_terminator %12 : index
        } {id = 6 : i32} : (index)
        %asyncToken_13, %valOut_14 = air.region async [%newAsyncToken_3]  : (!air.async.token) {
            %12 = arith.muli %newValOut_3, %c32 : index
            air.region_terminator %12 : index
        } {id = 6 : i32} : (index)
        %asyncToken_15, %valOut_16 = air.region async  {
            %12 = affine.apply #map2()[%arg8]
            air.region_terminator %12 : index
        } {id = 7 : i32} : (index)
        %8 = air.wait_all async [%asyncToken_13, %asyncToken_15] 
        %asyncToken_17, %valOut_18 = air.region async  {
            %12 = memref.alloc() : memref<32x32xbf16>
            air.region_terminator %12 : memref<32x32xbf16>
        } {id = 10 : i32} : (memref<32x32xbf16>)
        %10 = scf.for %arg14 = %c0_12 to %c64_11 step %c32 iter_args(%arg15 = %8) -> (!air.async.token) {
            %asyncToken_20, %valOut_21 = air.region async [%arg15]  : (!air.async.token) {
                %15 = memref.alloc() : memref<32x32xbf16>
                air.region_terminator %15 : memref<32x32xbf16>
            } {id = 8 : i32} : (memref<32x32xbf16>)
            %asyncToken_22, %valOut_23 = air.region async [%arg15]  : (!air.async.token) {
                %15 = memref.alloc() : memref<32x32xbf16>
                air.region_terminator %15 : memref<32x32xbf16>
            } {id = 9 : i32} : (memref<32x32xbf16>)
            %12 = air.dma_memcpy_nd async [%asyncToken_20, %arg15] (%valOut_21[] [] [], %arg11[%valOut_14, %arg14] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_pattern = #set0, id = 4 : i32} : (memref<32x32xbf16>, memref<64x64xbf16>)
            %13 = air.dma_memcpy_nd async [%asyncToken_22, %arg15] (%valOut_23[] [] [], %arg12[%arg14, %valOut_16] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_pattern = #set1, id = 5 : i32} : (memref<32x32xbf16>, memref<64x64xbf16>)
            %asyncToken_24 = air.region async [%13, %arg15, %12]  : (!air.async.token, !air.async.token, !air.async.token) {
                linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%valOut_21, %valOut_23 : memref<32x32xbf16>, memref<32x32xbf16>) outs(%valOut_18 : memref<32x32xbf16>)
                air.region_terminator
            } {id = 11 : i32}
            %asyncToken_25 = air.region async [%asyncToken_24]  : (!air.async.token) {
                memref.dealloc %valOut_21 : memref<32x32xbf16>
                air.region_terminator
            } {id = 12 : i32}
            %asyncToken_26 = air.region async [%asyncToken_24]  : (!air.async.token) {
                memref.dealloc %valOut_23 : memref<32x32xbf16>
                air.region_terminator
            } {id = 13 : i32}
            %14 = air.wait_all async [%asyncToken_24, %asyncToken_25, %asyncToken_26] 
            scf.yield %14 : !air.async.token
        }
        %asyncToken_19 = air.region async [%10] : (!air.async.token) {
            memref.dealloc %valOut_18 : memref<32x32xbf16>
            air.region_terminator
        } {id = 14 : i32}
        air.herd_terminator
    }
    return
  }
}