module {
  func.func @graph(%arg0: memref<4096xi32>, %arg1: memref<1xi32>) {
    %c1 = arith.constant 1 : index
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%arg1) : memref<4096xi32>, memref<1xi32> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
      %alloc_1 = memref.alloc() {sym_name = "scratch_copy"} : memref<1xi32, 2>
      affine.for %arg8 = 0 to 4096 step 32 {
        air.dma_memcpy_nd (%alloc[%c0] [%c32] [%c0], %arg6[%arg8] [%c32] [%c0]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
        affine.for %arg9 = 0 to 32 {
          %0 = affine.load %alloc[%arg9] : memref<32xi32, 2>
          %1 = affine.load %alloc_1[0] : memref<1xi32, 2>
          %2 = arith.addi %0, %1 : i32
          affine.store %2, %alloc_1[0] : memref<1xi32, 2>
        }
      }
      air.dma_memcpy_nd (%arg7[%c0] [%c1_0] [%c0], %alloc_1[%c0] [%c1_0] [%c0]) {id = 2 : i32} : (memref<1xi32>, memref<1xi32, 2>)
      memref.dealloc %alloc_1 : memref<1xi32, 2>
      memref.dealloc %alloc : memref<32xi32, 2>
      air.herd_terminator
    }
    return
  }
}

