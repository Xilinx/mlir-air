// (c) Copyright 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @softmax_kernel(%arg0: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xbf16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c4_i32 = arith.constant 4 : i32
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg5, %c4_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %1, %c1024 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [4, 1024], strides: [1024, 1] : memref<*xbf16> to memref<4x1024xbf16, strided<[1024, 1], offset: ?>>
    %alloc = memref.alloc() : memref<4x1024xbf16>
    memref.copy %reinterpret_cast, %alloc : memref<4x1024xbf16, strided<[1024, 1], offset: ?>> to memref<4x1024xbf16>
    %3 = bufferization.to_tensor %alloc restrict writable : memref<4x1024xbf16> to tensor<4x1024xbf16>
    %4 = tensor.empty() : tensor<4x1024xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4x1024xbf16>) outs(%4 : tensor<4x1024xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %17 = arith.extf %in : bf16 to f32
      linalg.yield %17 : f32
    } -> tensor<4x1024xf32>
    %6 = tensor.empty() : tensor<1024x4xf32>
    %transposed = linalg.transpose ins(%5 : tensor<4x1024xf32>) outs(%6 : tensor<1024x4xf32>) permutation = [1, 0] 
    %7 = tensor.empty() : tensor<4xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4xf32>) -> tensor<4xf32>
    %reduced = linalg.reduce ins(%transposed : tensor<1024x4xf32>) outs(%8 : tensor<4xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.maxnumf %in, %init : f32
        linalg.yield %17 : f32
      }
    %expanded = tensor.expand_shape %reduced [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<4x1xf32>) outs(%4 : tensor<4x1024xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4x1024xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %9 : tensor<4x1024xf32>, tensor<4x1024xf32>) outs(%5 : tensor<4x1024xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %17 = arith.subf %in, %in_5 : f32
      linalg.yield %17 : f32
    } -> tensor<4x1024xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<4x1024xf32>) outs(%10 : tensor<4x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %17 = math.exp %in : f32
      linalg.yield %17 : f32
    } -> tensor<4x1024xf32>
    %transposed_1 = linalg.transpose ins(%11 : tensor<4x1024xf32>) outs(%6 : tensor<1024x4xf32>) permutation = [1, 0] 
    %12 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<4xf32>) -> tensor<4xf32>
    %reduced_2 = linalg.reduce ins(%transposed_1 : tensor<1024x4xf32>) outs(%12 : tensor<4xf32>) dimensions = [0] 
      (%in: f32, %init: f32) {
        %17 = arith.addf %in, %init : f32
        linalg.yield %17 : f32
      }
    %expanded_3 = tensor.expand_shape %reduced_2 [[0, 1]] output_shape [4, 1] : tensor<4xf32> into tensor<4x1xf32>
    %13 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_3 : tensor<4x1xf32>) outs(%4 : tensor<4x1024xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4x1024xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11, %13 : tensor<4x1024xf32>, tensor<4x1024xf32>) outs(%11 : tensor<4x1024xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %17 = arith.divf %in, %in_5 : f32
      linalg.yield %17 : f32
    } -> tensor<4x1024xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [4, 1024], strides: [1024, 1] : memref<*xbf16> to memref<4x1024xbf16, strided<[1024, 1], offset: ?>>
    %15 = tensor.empty() : tensor<4x1024xbf16>
    %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<4x1024xf32>) outs(%15 : tensor<4x1024xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %17 = arith.truncf %in : f32 to bf16
      linalg.yield %17 : bf16
    } -> tensor<4x1024xbf16>
    bufferization.materialize_in_destination %16 in writable %reinterpret_cast_4 : (tensor<4x1024xbf16>, memref<4x1024xbf16, strided<[1024, 1], offset: ?>>) -> ()
    return
  }
}

