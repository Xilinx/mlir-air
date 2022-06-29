// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt -air-linalg-name %s | FileCheck %s
// CHECK: linalg.fill {__internal_linalg_transform__ = "linalg.fill0"} {{.*}}
// CHECK: linalg.copy {__internal_linalg_transform__ = "linalg.copy1"} {{.*}}
// CHECK: linalg.matmul {__internal_linalg_transform__ = "linalg.matmul2"}
module attributes {torch.debug_module_name = "mmult"} {
  func.func @forward(%arg0: memref<2304x1024xi32>, %arg1: memref<1024x1024xi32>) -> memref<?x?xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() : memref<2304x1024xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<2304x1024xi32>)
    %1 = memref.alloc() : memref<2304x1024xi32>
    linalg.copy ins(%0 : memref<2304x1024xi32>) outs(%1 : memref<2304x1024xi32>)
    linalg.matmul ins(%arg0, %arg1 : memref<2304x1024xi32>, memref<1024x1024xi32>) outs(%1 : memref<2304x1024xi32>)
    %2 = memref.cast %1 : memref<2304x1024xi32> to memref<?x?xi32>
    return %2 : memref<?x?xi32>
  }
}

