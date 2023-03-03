module {
  func.func @graph(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
    %c1 = arith.constant 1 : index
    air.herd @herd_0  tile (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<4096xi32>, memref<4096xi32>, memref<4096xi32> {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %alloc = memref.alloc() {sym_name = "inA"} : memref<32xi32, 2>
      %alloc_0 = memref.alloc() {sym_name = "inB"} : memref<32xi32, 2>
      %alloc_1 = memref.alloc() {sym_name = "outC"} : memref<32xi32, 2>
      affine.for %arg10 = 0 to 4096 step 32 {
        air.dma_memcpy_nd (%alloc[%c0] [%c32] [%c0], %arg7[%arg10] [%c32] [%c0]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
        air.dma_memcpy_nd (%alloc_0[%c0] [%c32] [%c0], %arg8[%arg10] [%c32] [%c0]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
        affine.for %arg11 = 0 to 32 {
          %0 = affine.load %alloc[%arg11] : memref<32xi32, 2>
          %1 = affine.load %alloc_0[%arg11] : memref<32xi32, 2>
          %2 = arith.addi %1, %0 : i32
          affine.store %2, %alloc_1[%arg11] : memref<32xi32, 2>
        }
        air.dma_memcpy_nd (%arg9[%arg10] [%c32] [%c0], %alloc_1[%c0] [%c32] [%c0]) {id = 2 : i32} : (memref<4096xi32>, memref<32xi32, 2>)
      }
      memref.dealloc %alloc_1 : memref<32xi32, 2>
      memref.dealloc %alloc_0 : memref<32xi32, 2>
      memref.dealloc %alloc : memref<32xi32, 2>
      air.herd_terminator
    }
    return
  }
}

