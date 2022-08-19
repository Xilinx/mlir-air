// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s -air-to-aie -o /dev/null | air-opt | FileCheck %s
// CHECK: [[T_0_0:%.*]] = AIE.tile(0, 0)
// CHECK: [[T_1_0:%.*]] = AIE.tile(1, 0)
// CHECK: [[T_0_1:%.*]] = AIE.tile(0, 1)
// CHECK: [[T_1_1:%.*]] = AIE.tile(1, 1)
// CHECK: [[C_1_1:%.*]] = AIE.core([[T_1_1]])
// CHECK: [[V0:%.*]] = arith.constant 10 : i32
// CHECK: [[V1:%.*]] = arith.constant 6 : i32
// CHECK: arith.addi %{{.*}}, [[V1]] : i32
// CHECK: arith.addi %{{.*}}, [[V0]] : i32
// CHECK: AIE.end
// CHECK: [[C_0_1:%.*]] = AIE.core([[T_0_1]])
// CHECK: [[V2:%.*]] = arith.constant 8 : i32
// CHECK: [[V3:%.*]] = arith.constant 6 : i32
// CHECK: arith.addi %{{.*}}, [[V3]] : i32
// CHECK: arith.addi %{{.*}}, [[V2]] : i32
// CHECK: AIE.end
// CHECK: [[C_1_0:%.*]] = AIE.core([[T_1_0]])
// CHECK: [[V4:%.*]] = arith.constant 10 : i32
// CHECK: [[V5:%.*]] = arith.constant 4 : i32
// CHECK: arith.addi %{{.*}}, [[V5]] : i32
// CHECK: arith.addi %{{.*}}, [[V4]] : i32
// CHECK: AIE.end
// CHECK: [[C_0_0:%.*]] = AIE.core([[T_0_0]])
// CHECK: [[V6:%.*]] = arith.constant 8 : i32
// CHECK: [[V7:%.*]] = arith.constant 4 : i32
// CHECK: arith.addi %{{.*}}, [[V7]] : i32
// CHECK: arith.addi %{{.*}}, [[V6]] : i32
// CHECK: AIE.end
#map0 = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#set0 = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0)>
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<64x64xi32>)
    air.herd  tile (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1, %arg9=%0) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {sym_name = "herd_0"} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c10_i32 = arith.constant 10 : i32
      %c8_i32 = arith.constant 8 : i32
      %c6_i32 = arith.constant 6 : i32
      %c4_i32 = arith.constant 4 : i32
      %1 = affine.apply #map0()[%arg3]
      %2 = affine.apply #map0()[%arg4]
      %3 = memref.alloc() : memref<32x32xi32, 2>
      %4 = memref.alloc() : memref<32x32xi32, 2>
      affine.if #set0()[%arg3, %arg4] {
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : memref<32x32xi32, 2>) outs(%3 : memref<32x32xi32, 2>) {
        ^bb0(%arg10: i32, %arg11: i32):
          %8 = arith.addi %arg10, %c4_i32 : i32
          linalg.yield %8 : i32
        }
      } else {
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : memref<32x32xi32, 2>) outs(%3 : memref<32x32xi32, 2>) {
        ^bb0(%arg10: i32, %arg11: i32):
          %8 = arith.addi %arg10, %c6_i32 : i32
          linalg.yield %8 : i32
        }
      }
      affine.if #set2()[%arg3, %arg4] {
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : memref<32x32xi32, 2>) outs(%3 : memref<32x32xi32, 2>) {
        ^bb0(%arg10: i32, %arg11: i32):
          %8 = arith.addi %arg10, %c8_i32 : i32
          linalg.yield %8 : i32
        }
      } else {
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%4 : memref<32x32xi32, 2>) outs(%3 : memref<32x32xi32, 2>) {
        ^bb0(%arg10: i32, %arg11: i32):
          %8 = arith.addi %arg10, %c10_i32 : i32
          linalg.yield %8 : i32
        }
      }
      air.herd_terminator
    }
    return
  }
}

