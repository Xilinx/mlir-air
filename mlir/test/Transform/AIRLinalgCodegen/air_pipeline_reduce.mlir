
// RUN: air-opt %s -air-pipeline-reduce='pipeline-depth=2 tile-size=16' | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0)>
// CHECK: #map1 = affine_map<(d0) -> ()>
// CHECK: #set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0)>
// CHECK: #set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0)>
// CHECK: air.channel @[[CHAN_0:.*]] [1]
// CHECK: func.func @f0(%[[VAL_0:.*]]: memref<32xf32>) -> memref<f32> {
// CHECK:   %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:   %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK:   air.herd  tile (%[[VAL_4:.*]], %[[VAL_5:.*]]) in (%[[VAL_6:.*]]=%[[VAL_1]], %[[VAL_7:.*]]=%[[VAL_2]]) args(%[[VAL_8:.*]]=%[[VAL_0]], %[[VAL_9:.*]]=%[[VAL_3]]) : memref<32xf32>, memref<f32> {
// CHECK:     %[[VAL_10:.*]] = arith.constant 16 : index
// CHECK:     %[[VAL_11:.*]] = arith.muli %[[VAL_4]], %[[VAL_10]] : index
// CHECK:     %[[VAL_12:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_11]]] [16] [1] : memref<32xf32> to memref<16xf32, strided<[1], offset: ?>>
// CHECK:     affine.if #set(){{\[}}%[[VAL_4]], %[[VAL_5]]] {
// CHECK:       linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%[[VAL_12]] : memref<16xf32, strided<[1], offset: ?>>) outs(%[[VAL_9]] : memref<f32>) {
// CHECK:       ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32):
// CHECK:         %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:         linalg.yield %[[VAL_15]] : f32
// CHECK:       }
// CHECK:       air.channel.put  @[[CHAN_0]][] (%[[VAL_9]][] [] []) : (memref<f32>)
// CHECK:     }
// CHECK:     affine.if #set1(){{\[}}%[[VAL_4]], %[[VAL_5]]] {
// CHECK:       %[[VAL_16:.*]] = memref.alloc() : memref<f32, 2>
// CHECK:       air.channel.get  @[[CHAN_0]][] (%[[VAL_16]][] [] []) : (memref<f32, 2>)
// CHECK:       linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%[[VAL_12]] : memref<16xf32, strided<[1], offset: ?>>) outs(%[[VAL_16]] : memref<f32, 2>) {
// CHECK:       ^bb0(%[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
// CHECK:         %[[VAL_19:.*]] = arith.addf %[[VAL_17]], %[[VAL_18]] : f32
// CHECK:         linalg.yield %[[VAL_19]] : f32
// CHECK:       }
// CHECK:       memref.copy %[[VAL_16]], %[[VAL_9]] : memref<f32, 2> to memref<f32>
// CHECK:     }
// CHECK:     air.herd_terminator
// CHECK:   }
// CHECK:   return %[[VAL_3]] : memref<f32>
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func.func @f0(%arg0: memref<32xf32>) -> memref<f32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
  linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<32xf32>) outs(%alloc : memref<f32>) {
  ^bb0(%in: f32, %out: f32):
    %0 = arith.addf %in, %out : f32
    linalg.yield %0 : f32
  }
  return %alloc : memref<f32>
}