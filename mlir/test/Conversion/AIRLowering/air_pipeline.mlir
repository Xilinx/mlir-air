// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt -air-to-std %s | FileCheck %s
// CHECK: #set0 = affine_set<(d0, d1) : (d0 == 0, d1 >= 0)>
// CHECK: #set1 = affine_set<(d0, d1) : (d0 - 1 == 0, d1 >= 0)>
// CHECK: #set2 = affine_set<(d0, d1) : (d0 - 2 == 0, d1 >= 0)>
// CHECK: #set3 = affine_set<(d0, d1) : (d0 - 3 == 0, d1 >= 0)>
// CHECK: affine.for %arg3 = 0 to 4 {
// CHECK:   affine.for %arg4 = 0 to 1 {
// CHECK: affine.if #set0(%arg3, %arg4) {
// CHECK: affine.if #set1(%arg3, %arg4) {
// CHECK: affine.if #set2(%arg3, %arg4) {
// CHECK: affine.if #set3(%arg3, %arg4) {
#map0 = affine_map<(d0) -> (d0)>
module  {
  func @launch(%m0: memref<1024xf32>, %m1: memref<1024xf32>, %m2: memref<1024xf32>) {
    %c4 = constant 4 : index
    %c1 = constant 1 : index
    air.launch_herd tile (%x, %y) in (%sx=%c4, %sy=%c1) args(%op0=%m0, %op1=%m1, %op2=%m2) : memref<1024xf32>,memref<1024xf32>,memref<1024xf32> {
      %c1_f32 = constant 1.0 : f32
      %c0 = constant 0 : index
      %c1024 = constant 1024 : index

      air.pipeline {direction = "horiz"} {
        %output1 = air.pipeline.stage {
          %a = memref.alloc() : memref<1024xf32, 2>
          %b = memref.alloc() : memref<1024xf32, 2>
          air.dma_memcpy (%a, %op0, [%c0], [%c0], %c1024) {id = 1 : i32} : (memref<1024xf32, 2>, memref<1024xf32>, [index], [index], index) -> ()
          air.dma_memcpy (%b, %op1, [%c0], [%c0], %c1024) {id = 2 : i32} : (memref<1024xf32, 2>, memref<1024xf32>, [index], [index], index) -> ()
          %init = linalg.init_tensor [1024] : tensor<1024xf32>
          %ta = memref.tensor_load %a : memref<1024xf32, 2>
          %tb = memref.tensor_load %b : memref<1024xf32, 2>
          %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%ta, %tb : tensor<1024xf32>, tensor<1024xf32>) outs(%init : tensor<1024xf32>) {
          ^bb0(%a2: f32, %a3: f32, %a4: f32):  // no predecessors
            %6 = mulf %a2, %a3 : f32
            linalg.yield %6 : f32
          } -> tensor<1024xf32>
          air.pipeline.yield %5 : tensor<1024xf32>
        } : tensor<1024xf32>

        %output2 = air.pipeline.stage args(%in = %output1) : tensor<1024xf32> {
          %init = linalg.init_tensor [1024] : tensor<1024xf32>
          %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%in : tensor<1024xf32>) outs(%init : tensor<1024xf32>) {
          ^bb0(%a2: f32, %a3: f32):  // no predecessors
            %one = constant 1.0 : f32
            %6 = addf %a2, %one : f32
            linalg.yield %6 : f32
          } -> tensor<1024xf32>
          air.pipeline.yield %5 : tensor<1024xf32>
        } : tensor<1024xf32>

        %output3 = air.pipeline.stage args(%in = %output2) : tensor<1024xf32> {
          %init = linalg.init_tensor [1024] : tensor<1024xf32>
          %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%in : tensor<1024xf32>) outs(%init : tensor<1024xf32>) {
          ^bb0(%a2: f32, %a3: f32):  // no predecessors
            %one = constant 1.0 : f32
            %6 = addf %a2, %one : f32
            linalg.yield %6 : f32
          } -> tensor<1024xf32>
          air.pipeline.yield %5 : tensor<1024xf32>
        } : tensor<1024xf32>

        air.pipeline.stage args(%in = %output3) : tensor<1024xf32> {
          %c = memref.alloc() : memref<1024xf32, 2>
          %init = linalg.init_tensor [1024] : tensor<1024xf32>
          %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%in : tensor<1024xf32>) outs(%init : tensor<1024xf32>) {
          ^bb0(%a2: f32, %a3: f32):  // no predecessors
            %one = constant 1.0 : f32
            %6 = addf %a2, %one : f32
            linalg.yield %6 : f32
          } -> tensor<1024xf32>
          memref.tensor_store %5, %c : memref<1024xf32, 2>
          air.dma_memcpy (%op2, %c, [%c0], [%c0], %c1024) {id = 3 : i32} : (memref<1024xf32>, memref<1024xf32, 2>, [index], [index], index) -> ()
          air.pipeline.yield
        }
        air.pipeline.terminator
      }
      air.herd_terminator
    }
    return
  }
}
