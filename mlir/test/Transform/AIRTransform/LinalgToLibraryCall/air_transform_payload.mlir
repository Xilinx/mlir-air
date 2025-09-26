//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform_default_name.mlir' %s | FileCheck %s
// RUN: air-opt -air-transform='filename=%S/air_transform_explicit_name.mlir' %s | FileCheck %s  --check-prefix=EXPLICITNAME
// RUN: air-opt -air-transform='filename=%S/air_transform_link_with.mlir' %s | FileCheck %s  --check-prefix=LINKWITH

// CHECK: call @linalg_matmul_view4x4xf32_view4x4xf32_view4x4xf32(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
// EXPLICITNAME: call @my_explicit_func(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
// LINKWITH: func.func private @my_linked_func(memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) attributes {link_with = "extern_func.o", llvm.emit_c_interface}
// LINKWITH: call @my_linked_func(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
module {
  func.func @main(%A: memref<4x4xf32>, %B: memref<4x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul ins(%A, %B : memref<4x4xf32>, memref<4x4xf32>) outs(%C : memref<4x4xf32>)
    return
  }
}
