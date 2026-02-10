// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @softmax_kernel(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant 1.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c4_i32 = arith.constant 4 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x1xf32>) -> tensor<4x1xf32>
    %2 = arith.muli %arg5, %c4_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.muli %3, %c1024 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [4, 1024], strides: [1024, 1] : memref<*xbf16> to memref<4x1024xbf16, strided<[1024, 1], offset: ?>>
    %alloc = memref.alloc() : memref<4x1024xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<4x1024xbf16, strided<[1024, 1], offset: ?>> to memref<4x1024xbf16>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<4x1024xbf16> to tensor<4x1024xbf16>
    %6 = tensor.empty() : tensor<4x1024xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<4x1024xbf16>) outs(%6 : tensor<4x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %22 = arith.extf %in : bf16 to f32
      linalg.yield %22 : f32
    } -> tensor<4x1024xf32>
    %8 = tensor.empty() : tensor<1024x4xf32>
    %transposed = linalg.transpose ins(%7 : tensor<4x1024xf32>) outs(%8 : tensor<1024x4xf32>) permutation = [1, 0] 
    %9 = tensor.empty() : tensor<4xf32>
    %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<4xf32>) -> tensor<4xf32>
    %reduced = linalg.reduce ins(%transposed : tensor<1024x4xf32>) outs(%10 : tensor<4xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %22 = arith.maxnumf %in, %init : f32
        linalg.yield %22 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<4x1xf32>) outs(%6 : tensor<4x1024xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4x1024xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%7, %11 : tensor<4x1024xf32>, tensor<4x1024xf32>) outs(%7 : tensor<4x1024xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %22 = arith.subf %in, %in_6 : f32
      linalg.yield %22 : f32
    } -> tensor<4x1024xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<4x1024xf32>) outs(%12 : tensor<4x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = math.exp %in : f32
      linalg.yield %22 : f32
    } -> tensor<4x1024xf32>
    %transposed_2 = linalg.transpose ins(%13 : tensor<4x1024xf32>) outs(%8 : tensor<1024x4xf32>) permutation = [1, 0] 
    %14 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<4xf32>) -> tensor<4xf32>
    %reduced_3 = linalg.reduce ins(%transposed_2 : tensor<1024x4xf32>) outs(%14 : tensor<4xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %22 = arith.addf %in, %init : f32
        linalg.yield %22 : f32
      }
    %expanded_4 = tensor.expand_shape %reduced_3 [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %expanded_4 : tensor<4x1xf32>, tensor<4x1xf32>) outs(%1 : tensor<4x1xf32>) {
    ^bb0(%in: f32, %in_6: f32, %out: f32):
      %22 = arith.divf %in, %in_6 : f32
      linalg.yield %22 : f32
    } -> tensor<4x1xf32>
    %16 = tensor.empty() : tensor<4x1024xbf16>
    %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<4x1024xf32>) outs(%16 : tensor<4x1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %22 = arith.truncf %in : f32 to bf16
      linalg.yield %22 : bf16
    } -> tensor<4x1024xbf16>
    %18 = tensor.empty() : tensor<4x1xbf16>
    %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<4x1xf32>) outs(%18 : tensor<4x1xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %22 = arith.truncf %in : f32 to bf16
      linalg.yield %22 : bf16
    } -> tensor<4x1xbf16>
    %20 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%19 : tensor<4x1xbf16>) outs(%16 : tensor<4x1024xbf16>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4x1024xbf16>
    %21 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%17, %20 : tensor<4x1024xbf16>, tensor<4x1024xbf16>) outs(%17 : tensor<4x1024xbf16>) {
    ^bb0(%in: bf16, %in_6: bf16, %out: bf16):
      %22 = arith.mulf %in, %in_6 : bf16
      linalg.yield %22 : bf16
    } -> tensor<4x1024xbf16>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%4], sizes: [4, 1024], strides: [1024, 1] : memref<*xbf16> to memref<4x1024xbf16, strided<[1024, 1], offset: ?>>
    bufferization.materialize_in_destination %21 in writable %reinterpret_cast_5 : (tensor<4x1024xbf16>, memref<4x1024xbf16, strided<[1024, 1], offset: ?>>) -> ()
    return
  }
}

