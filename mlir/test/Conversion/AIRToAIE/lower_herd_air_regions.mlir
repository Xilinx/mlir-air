// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-to-aie -o /dev/null | air-opt | FileCheck %s
// CHECK: AIE.core({{.*}}) {
// CHECK:      AIE.useLock({{.*}}, Acquire, 1)
// CHECK-NEXT: AIE.useLock({{.*}}, Acquire, 0)
// CHECK-NEXT: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} {
// CHECK-NEXT:   AIE.useLock({{.*}}, Acquire, 1)
// CHECK-NEXT:   AIE.useLock({{.*}}, Acquire, 1)
// CHECK-NEXT:   linalg.matmul ins({{.*}}, {{.*}} : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs({{.*}} : memref<32x32xi32, 2>)
// CHECK-NEXT:   AIE.useLock({{.*}}, Release, 0)
// CHECK-NEXT:   AIE.useLock({{.*}}, Release, 0)
// CHECK-NEXT: }
// CHECK-NEXT: AIE.useLock({{.*}}, Release, 0)
// CHECK-NEXT: AIE.useLock({{.*}}, Release, 1)
#map = affine_map<()[s0] -> (s0 * 32)>
#set0 = affine_set<(d0, d1)[s0] : (d0 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %ci1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<64x64xi32>)
    %1 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    memref.copy %0, %1 : memref<64x64xi32> to memref<64x64xi32>
    air.launch_herd  tile (%arg3, %arg4) in (%arg5=%ci1, %arg6=%ci1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%1) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 1 : i32, sym_name = "herd_0"} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %asyncToken, %valOut = air.region async  {
        %6 = affine.apply #map()[%arg3]
        air.region_terminator %6 : index
      } {id = 5 : i32} : (index)
      %asyncToken_0, %valOut_1 = air.region async  {
        %6 = affine.apply #map()[%arg4]
        air.region_terminator %6 : index
      } {id = 6 : i32} : (index)
      %2 = air.wait_all async [%asyncToken, %asyncToken_0] 
      %asyncToken_2, %valOut_3 = air.region async  {
        %6 = memref.alloc() : memref<32x32xi32, 2>
        air.region_terminator %6 : memref<32x32xi32, 2>
      } {id = 9 : i32} : (memref<32x32xi32, 2>)
      %3 = air.dma_memcpy_nd async [%2, %asyncToken_2] (%valOut_3[] [] [], %arg9[%valOut, %valOut_1] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
      %4 = scf.for %arg10 = %c0 to %c64 step %c32 iter_args(%arg11 = %3) -> (!air.async.token) {
        %asyncToken_5, %valOut_6 = air.region async [%arg11]  : (!air.async.token) {
          %9 = memref.alloc() : memref<32x32xi32, 2>
          air.region_terminator %9 : memref<32x32xi32, 2>
        } {id = 7 : i32} : (memref<32x32xi32, 2>)
        %asyncToken_7, %valOut_8 = air.region async [%arg11]  : (!air.async.token) {
          %9 = memref.alloc() : memref<32x32xi32, 2>
          air.region_terminator %9 : memref<32x32xi32, 2>
        } {id = 8 : i32} : (memref<32x32xi32, 2>)
        %6 = air.dma_memcpy_nd async [%asyncToken_5, %arg11] (%valOut_6[] [] [], %arg7[%valOut, %arg10] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        %7 = air.dma_memcpy_nd async [%asyncToken_7, %arg11] (%valOut_8[] [] [], %arg8[%arg10, %valOut_1] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        %asyncToken_9 = air.region async [%7, %arg11, %6]  : (!air.async.token, !air.async.token, !air.async.token) {
          linalg.matmul ins(%valOut_6, %valOut_8 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%valOut_3 : memref<32x32xi32, 2>)
          air.region_terminator
        } {id = 12 : i32}
        %asyncToken_10 = air.region async [%asyncToken_9]  : (!air.async.token) {
          memref.dealloc %valOut_6 : memref<32x32xi32, 2>
          air.region_terminator
        } {id = 13 : i32}
        %asyncToken_11 = air.region async [%asyncToken_9]  : (!air.async.token) {
          memref.dealloc %valOut_8 : memref<32x32xi32, 2>
          air.region_terminator
        } {id = 14 : i32}
        %8 = air.wait_all async [%asyncToken_9, %asyncToken_10, %asyncToken_11] 
        scf.yield %8 : !air.async.token
      }
      %5 = air.dma_memcpy_nd async [%4] (%arg9[%valOut, %valOut_1] [%c32, %c32] [%c64, %c1], %valOut_3[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
      %asyncToken_4 = air.region async [%5]  : (!air.async.token) {
        memref.dealloc %valOut_3 : memref<32x32xi32, 2>
        air.region_terminator
      } {id = 15 : i32}
      air.herd_terminator
    }
    memref.copy %1, %arg2 : memref<64x64xi32> to memref<64x64xi32>
    return
  }
}

