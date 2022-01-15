// (c) Copyright 2021 Xilinx Inc.

#map0 = affine_map<(d0) -> (d0)>
module  {
  func @launch(%m0: memref<1024xi32>, %m1: memref<1024xi32>, %m2: memref<1024xi32>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    air.launch_herd tile (%x, %y) in (%sx=%c4, %sy=%c1) args(%op0=%m0, %op1=%m1, %op2=%m2) : memref<1024xi32>,memref<1024xi32>,memref<1024xi32> attributes {sym_name="herd_0"} {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index

      air.pipeline {direction = "horiz"} {

        %output1 = air.pipeline.stage {
          %a = memref.alloc() : memref<1024xi32, 2>
          %b = memref.alloc() : memref<1024xi32, 2>
          air.dma_memcpy (%a, %op0, [%c0], [%c0], %c1024) {id = 1 : i32} : (memref<1024xi32, 2>, memref<1024xi32>, [index], [index], index) -> ()
          air.dma_memcpy (%b, %op1, [%c0], [%c0], %c1024) {id = 2 : i32} : (memref<1024xi32, 2>, memref<1024xi32>, [index], [index], index) -> ()
          %init = linalg.init_tensor [1024] : tensor<1024xi32>
          %ta = memref.tensor_load %a : memref<1024xi32, 2>
          %tb = memref.tensor_load %b : memref<1024xi32, 2>
          %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]} ins(%ta, %tb : tensor<1024xi32>, tensor<1024xi32>) outs(%init : tensor<1024xi32>) {
          ^bb0(%a2: i32, %a3: i32, %a4: i32):  // no predecessors
            %6 = arith.muli %a2, %a3 : i32
            linalg.yield %6 : i32
          } -> tensor<1024xi32>
          air.pipeline.yield %5 : tensor<1024xi32>
        } : tensor<1024xi32>

        %output2 = air.pipeline.stage args(%in = %output1) : tensor<1024xi32> {
          %init = linalg.init_tensor [1024] : tensor<1024xi32>
          %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%in : tensor<1024xi32>) outs(%init : tensor<1024xi32>) {
          ^bb0(%a2: i32, %a3: i32):  // no predecessors
            %one = arith.constant 1 : i32
            %6 = arith.addi %a2, %one : i32
            linalg.yield %6 : i32
          } -> tensor<1024xi32>
          air.pipeline.yield %5 : tensor<1024xi32>
        } : tensor<1024xi32>

        %output3 = air.pipeline.stage args(%in = %output2) : tensor<1024xi32> {
          %init = linalg.init_tensor [1024] : tensor<1024xi32>
          %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%in : tensor<1024xi32>) outs(%init : tensor<1024xi32>) {
          ^bb0(%a2: i32, %a3: i32):  // no predecessors
            %two = arith.constant 2 : i32
            %6 = arith.addi %a2, %two : i32
            linalg.yield %6 : i32
          } -> tensor<1024xi32>
          air.pipeline.yield %5 : tensor<1024xi32>
        } : tensor<1024xi32>

        air.pipeline.stage args(%in = %output3) : tensor<1024xi32> {
          %c = memref.alloc() : memref<1024xi32, 2>
          %init = linalg.init_tensor [1024] : tensor<1024xi32>
          %5 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%in : tensor<1024xi32>) outs(%init : tensor<1024xi32>) {
          ^bb0(%a2: i32, %a3: i32):  // no predecessors
            %three = arith.constant 3 : i32
            %6 = arith.addi %a2, %three : i32
            linalg.yield %6 : i32
          } -> tensor<1024xi32>
          memref.tensor_store %5, %c : memref<1024xi32, 2>
          air.dma_memcpy (%op2, %c, [%c0], [%c0], %c1024) {id = 3 : i32} : (memref<1024xi32>, memref<1024xi32, 2>, [index], [index], index) -> ()
          air.pipeline.yield
        }
        air.pipeline.terminator
      }
      air.herd_terminator
    }
    return
  }
}