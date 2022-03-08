module {
  func @forward(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<1024x1024xi32>) {
    %c4096 = arith.constant 4096 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %0 = memref.alloc() : memref<64x32xi32, 1>
    %1 = memref.alloc() : memref<32x64xi32, 1>
    %2 = memref.alloc() : memref<64x64xi32, 1>
    %3 = air.dma_memcpy_2d async (%0, %arg0, [%c0, %c0], [%c0, %c0], %c2048, %c1024, %c32) {id = 1 : i32} : (memref<64x32xi32, 1>, memref<1024x1024xi32>, [index, index], [index, index], index, index, index) -> ()
    %4 = air.dma_memcpy_2d async (%1, %arg1, [%c0, %c0], [%c0, %c0], %c2048, %c1024, %c64) {id = 2 : i32} : (memref<32x64xi32, 1>, memref<1024x1024xi32>, [index, index], [index, index], index, index, index) -> ()
    %5 = air.dma_memcpy_2d async (%2, %arg2, [%c0, %c0], [%c0, %c0], %c4096, %c1024, %c64) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<1024x1024xi32>, [index, index], [index, index], index, index, index) -> ()
    %6 = air.dma_memcpy_2d async [%3, %4, %5](%arg2, %2, [%c0, %c0], [%c0, %c0], %c4096, %c1024, %c64) {id = 8 : i32} : (memref<1024x1024xi32>, memref<64x64xi32, 1>, [index, index], [index, index], index, index, index) -> ()
    air.wait_all [%6]
    memref.dealloc %0 : memref<64x32xi32, 1>
    memref.dealloc %1 : memref<32x64xi32, 1>
    memref.dealloc %2 : memref<64x64xi32, 1>
    return
  }

  func @test0(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<?x?xi32>) {
    %c4096 = arith.constant 4096 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() : memref<1024x1024xi32>
    linalg.fill(%c0_i32, %0) : i32, memref<1024x1024xi32> 
    %1 = memref.cast %arg2 : memref<?x?xi32> to memref<1024x1024xi32>
    linalg.copy(%0, %1) : memref<1024x1024xi32>, memref<1024x1024xi32> 
    scf.for %arg3 = %c0 to %c1024 step %c1024 {
      scf.for %arg4 = %c0 to %c1024 step %c1024 {
        scf.for %arg5 = %c0 to %c1024 step %c32 {
          %2 = memref.alloc() : memref<64x32xi32, 1>
          %3 = memref.alloc() : memref<32x64xi32, 1>
          %4 = memref.alloc() : memref<64x64xi32, 1>
          %e0 = air.dma_memcpy_2d async (%2, %arg0, [%c0, %c0], [%arg3, %arg5], %c2048, %c1024, %c32) {id = 1 : i32} : (memref<64x32xi32, 1>, memref<1024x1024xi32>, [index, index], [index, index], index, index, index) -> ()
          %e1 = air.dma_memcpy_2d async (%3, %arg1, [%c0, %c0], [%arg5, %arg4], %c2048, %c1024, %c64) {id = 2 : i32} : (memref<32x64xi32, 1>, memref<1024x1024xi32>, [index, index], [index, index], index, index, index) -> ()
          %e2 = air.dma_memcpy_2d async (%4, %1, [%c0, %c0], [%arg3, %arg4], %c4096, %c1024, %c64) {id = 3 : i32} : (memref<64x64xi32, 1>, memref<1024x1024xi32>, [index, index], [index, index], index, index, index) -> ()
          %e3 = air.dma_memcpy_2d async [%e0, %e1, %e2] (%1, %4, [%arg3, %arg4], [%c0, %c0], %c4096, %c1024, %c64) {id = 8 : i32} : (memref<1024x1024xi32>, memref<64x64xi32, 1>, [index, index], [index, index], index, index, index) -> ()
          air.wait_all [%e3]
          memref.dealloc %2 : memref<64x32xi32, 1>
          memref.dealloc %3 : memref<32x64xi32, 1>
          memref.dealloc %4 : memref<64x64xi32, 1>
        }
      }
    }
    return
  }

  func @test1(%arg0: memref<1024x1024xi32>, %arg1: memref<1024x1024xi32>, %arg2: memref<1024x1024xi32>) {
    %c0_i32 = arith.constant 0 : i32
    linalg.fill(%c0_i32, %arg0) : i32, memref<1024x1024xi32> 
    linalg.copy(%arg1, %arg2) : memref<1024x1024xi32>, memref<1024x1024xi32> 
    linalg.matmul ins(%arg1, %arg2 : memref<1024x1024xi32>, memref<1024x1024xi32>) outs(%arg0 : memref<1024x1024xi32>)
    return
  }
}

