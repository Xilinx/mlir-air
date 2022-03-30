// (c) Copyright 2022 Xilinx Inc.

// check unchanged with no filter options
// RUN: air-opt %s -air-linalg-codegen | FileCheck %s -check-prefix=UNCH
// UNCH: linalg.matmul {__internal_linalg_transform__ = "mmult"} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>)
// UNCH: linalg.generic {{.*}} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic0"} {
// UNCH: linalg.generic {{.*}} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic1"} {

// check generic1 options applied
// RUN: air-opt %s -air-linalg-codegen='input-filter=generic1' | FileCheck %s -check-prefix=GENERIC1
// GENERIC1: linalg.matmul {__internal_linalg_transform__ = "mmult"} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>)
// GENERIC1: linalg.generic {{.*}} ins({{.*}} : memref<128x128xi32>, memref<128x128xi32>) outs({{.*}} : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic0"} {
// GENERIC1: linalg.generic {{.*}} ins({{.*}} : memref<64x32xi32, 2>, memref<64x32xi32, 2>) outs({{.*}} : memref<64x32xi32, 2>) {


#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {torch.debug_module_name = "mmult"} {
  func @forward(%arg0: memref<128x128xi32>, %arg1: memref<128x128xi32>, %arg2: memref<128x128xi32>, %arg3: memref<128x128xi32>) -> memref<?x?xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() : memref<128x128xi32>
    linalg.fill(%c0_i32, %0) : i32, memref<128x128xi32> 
    %1 = memref.alloc() : memref<128x128xi32>
    linalg.copy(%0, %1) : memref<128x128xi32>, memref<128x128xi32> 
    linalg.matmul {__internal_linalg_transform__ = "mmult"} ins(%arg2, %arg3 : memref<128x128xi32>, memref<128x128xi32>) outs(%1 : memref<128x128xi32>)
    %2 = memref.alloc() : memref<128x128xi32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %1 : memref<128x128xi32>, memref<128x128xi32>) outs(%2 : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic0"} {
    ^bb0(%arg4: i32, %arg5: i32, %arg6: i32):
      %5 = arith.muli %arg4, %arg5 : i32
      linalg.yield %5 : i32
    }
    %3 = memref.alloc() : memref<128x128xi32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %2 : memref<128x128xi32>, memref<128x128xi32>) outs(%3 : memref<128x128xi32>) attrs =  {__internal_linalg_transform__ = "generic1"} {
    ^bb0(%arg4: i32, %arg5: i32, %arg6: i32):
      %5 = arith.addi %arg4, %arg5 : i32
      linalg.yield %5 : i32
    }
    %4 = memref.cast %3 : memref<128x128xi32> to memref<?x?xi32>
    return %4 : memref<?x?xi32>
  }
}

