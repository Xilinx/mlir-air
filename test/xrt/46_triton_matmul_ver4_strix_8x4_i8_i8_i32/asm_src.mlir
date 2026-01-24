module {
  func.func @bare_matmul(%arg0: memref<*xi8> {tt.divisibility = 16 : i32}, %arg1: memref<*xi8> {tt.divisibility = 16 : i32}, %arg2: memref<*xi32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1024 = arith.constant 1024 : index
    %c256_i32 = arith.constant 256 : i32
    %c512_i32 = arith.constant 512 : i32
    %0 = arith.muli %arg6, %c512_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.muli %arg7, %c256_i32 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.muli %1, %c1024 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%4], sizes: [512, 1024], strides: [1024, 1] : memref<*xi8> to memref<512x1024xi8, strided<[1024, 1], offset: ?>>
    %alloc = memref.alloc() : memref<512x1024xi8>
    memref.copy %reinterpret_cast, %alloc : memref<512x1024xi8, strided<[1024, 1], offset: ?>> to memref<512x1024xi8>
    %5 = bufferization.to_tensor %alloc restrict writable : memref<512x1024xi8> to tensor<512x1024xi8>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%3], sizes: [1024, 256], strides: [1024, 1] : memref<*xi8> to memref<1024x256xi8, strided<[1024, 1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<1024x256xi8>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<1024x256xi8, strided<[1024, 1], offset: ?>> to memref<1024x256xi8>
    %6 = bufferization.to_tensor %alloc_1 restrict writable : memref<1024x256xi8> to tensor<1024x256xi8>
    %7 = tensor.empty() : tensor<512x256xi32>
    %8 = linalg.fill ins(%c0_i32 : i32) outs(%7 : tensor<512x256xi32>) -> tensor<512x256xi32>
    %9 = linalg.matmul ins(%5, %6 : tensor<512x1024xi8>, tensor<1024x256xi8>) outs(%8 : tensor<512x256xi32>) -> tensor<512x256xi32>
    %10 = arith.addi %4, %3 : index
    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [%10], sizes: [512, 256], strides: [1024, 1] : memref<*xi32> to memref<512x256xi32, strided<[1024, 1], offset: ?>>
    bufferization.materialize_in_destination %9 in writable %reinterpret_cast_2 : (tensor<512x256xi32>, memref<512x256xi32, strided<[1024, 1], offset: ?>>) -> ()
    return
  }
}
