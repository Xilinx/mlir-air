//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// CHECK: %[[apply0:.*]] = affine.apply #map(%{{.*}})
// CHECK: %[[apply1:.*]] = affine.apply #map(%{{.*}})
// CHECK: %[[subview:.*]] = memref.subview %{{.*}}[%[[apply0]], 0] [256, 1024] [1, 1]
// CHECK: %[[subview_3:.*]] = memref.subview %{{.*}}[0, %[[apply1]]] [1024, 256] [1, 1]
// CHECK: %[[subview_4:.*]] = memref.subview %{{.*}}[%[[apply0]], %[[apply1]]] [256, 256] [1, 1]
// CHECK: %[[subview_5:.*]] = memref.subview %[[subview]][0, 0] [256, 32] [1, 1]
// CHECK: %[[expand_shape:.*]] = memref.expand_shape %[[subview_5]] [{{.*}}[0, 1], [2, 3]{{.*}}] output_shape [4, 64, 1, 32]
// CHECK: %[[transpose:.*]] = memref.transpose %[[expand_shape]] (d0, d1, d2, d3) -> (d0, d2, d1, d3)
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[transpose]][] [] [])
// CHECK: %[[subview_6:.*]] = memref.subview %[[subview_3]][0, 0] [32, 256] [1, 1]
// CHECK: %[[expand_shape_7:.*]] = memref.expand_shape %[[subview_6]] [{{.*}}[0, 1], [2, 3]{{.*}}] output_shape [1, 32, 4, 64]
// CHECK: %[[transpose_8:.*]] = memref.transpose %[[expand_shape_7]] (d0, d1, d2, d3) -> (d2, d0, d1, d3)
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[transpose_8]][] [] [])
// CHECK: %[[transpose_9:.*]] = memref.transpose %{{.*}} (d0, d1, d2, d3) -> (d0, d2, d1, d3)
// CHECK: air.dma_memcpy_nd (%[[subview_4]][] [] [], %[[transpose_9]][] [] [])
// CHECK: %[[subview_10:.*]] = memref.subview %{{.*}}[%[[apply0]], %[[apply1]]] [256, 256] [1, 1]
// CHECK: memref.copy %[[subview_4]], %[[subview_10]]

#map = affine_map<(d0) -> (d0 * 256)>
module {
  func.func @func0(%arg0: memref<512x1024xi32>, %arg1: memref<1024x512xi32>) -> memref<512x512xi32> {
    %c0_i32 = arith.constant 0 : i32
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %alloc_1 = memref.alloc() : memref<4x1x32x64xi32, 1>
    %alloc_2 = memref.alloc() : memref<4x1x64x32xi32, 1>
    %alloc_4 = memref.alloc() : memref<4x4x64x64xi32, 1>
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<512x1024xi32> to tensor<512x1024xi32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<1024x512xi32> to tensor<1024x512xi32>
    %2 = tensor.empty() : tensor<512x512xi32>
    %3 = scf.forall (%arg2, %arg3) in (2, 2) shared_outs(%arg4 = %2) -> (tensor<512x512xi32>) {
      %5 = affine.apply #map(%arg2)
      %6 = affine.apply #map(%arg3)
      %extracted_slice = tensor.extract_slice %0[%5, 0] [256, 1024] [1, 1] : tensor<512x1024xi32> to tensor<256x1024xi32>
      %extracted_slice_5 = tensor.extract_slice %1[0, %6] [1024, 256] [1, 1] : tensor<1024x512xi32> to tensor<1024x256xi32>
      %extracted_slice_6 = tensor.extract_slice %arg4[%5, %6] [256, 256] [1, 1] : tensor<512x512xi32> to tensor<256x256xi32>
      %7 = bufferization.to_tensor %alloc_4 restrict writable : memref<4x4x64x64xi32, 1> to tensor<4x4x64x64xi32>
      %extracted_slice_7 = tensor.extract_slice %extracted_slice[0, 0] [256, 32] [1, 1] : tensor<256x1024xi32> to tensor<256x32xi32>
      %9 = bufferization.to_tensor %alloc_2 restrict writable : memref<4x1x64x32xi32, 1> to tensor<4x1x64x32xi32>
      %pack = linalg.pack %extracted_slice_7 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 32] into %9 : tensor<256x32xi32> -> tensor<4x1x64x32xi32>
      %extracted_slice_8 = tensor.extract_slice %extracted_slice_5[0, 0] [32, 256] [1, 1] : tensor<1024x256xi32> to tensor<32x256xi32>
      %10 = bufferization.to_tensor %alloc_1 restrict writable : memref<4x1x32x64xi32, 1> to tensor<4x1x32x64xi32>
      %pack_9 = linalg.pack %extracted_slice_8 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %10 : tensor<32x256xi32> -> tensor<4x1x32x64xi32>
      %unpack = linalg.unpack %7 inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %extracted_slice_6 : tensor<4x4x64x64xi32> -> tensor<256x256xi32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %unpack into %arg4[%5, %6] [256, 256] [1, 1] : tensor<256x256xi32> into tensor<512x512xi32>
      }
    }
    %4 = bufferization.to_buffer %3 : tensor<512x512xi32> to memref<512x512xi32>
    memref.dealloc %alloc_4 : memref<4x4x64x64xi32, 1>
    memref.dealloc %alloc_2 : memref<4x1x64x32xi32, 1>
    memref.dealloc %alloc_1 : memref<4x1x32x64xi32, 1>
    return %4 : memref<512x512xi32>
  }
}
