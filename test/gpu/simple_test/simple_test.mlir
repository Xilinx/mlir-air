// Simple AIR test for GPU compilation
module {
  func.func @simple_add(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>) {
    %c1 = arith.constant 1 : index

    air.herd tile(%tx, %ty) in (%sx=%c1, %sy=%c1) args(%in0=%arg0, %in1=%arg1, %out=%arg2) : memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32> {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c16 = arith.constant 16 : index

      %buf0 = memref.alloc() : memref<16x16xf32, 2>
      %buf1 = memref.alloc() : memref<16x16xf32, 2>
      %buf2 = memref.alloc() : memref<16x16xf32, 2>

      // Copy input data
      air.dma_memcpy_nd (%buf0[][][], %in0[%c0,%c0][%c16,%c16][%c16,%c1_inner]) : (memref<16x16xf32, 2>, memref<16x16xf32>)
      air.dma_memcpy_nd (%buf1[][][], %in1[%c0,%c0][%c16,%c16][%c16,%c1_inner]) : (memref<16x16xf32, 2>, memref<16x16xf32>)

      // Compute: element-wise add
      affine.for %i = 0 to 16 {
        affine.for %j = 0 to 16 {
          %a = affine.load %buf0[%i, %j] : memref<16x16xf32, 2>
          %b = affine.load %buf1[%i, %j] : memref<16x16xf32, 2>
          %c = arith.addf %a, %b : f32
          affine.store %c, %buf2[%i, %j] : memref<16x16xf32, 2>
        }
      }

      // Copy output data
      air.dma_memcpy_nd (%out[%c0,%c0][%c16,%c16][%c16,%c1_inner], %buf2[][][]) : (memref<16x16xf32>, memref<16x16xf32, 2>)

      memref.dealloc %buf0 : memref<16x16xf32, 2>
      memref.dealloc %buf1 : memref<16x16xf32, 2>
      memref.dealloc %buf2 : memref<16x16xf32, 2>
      air.herd_terminator
    }
    return
  }
}
