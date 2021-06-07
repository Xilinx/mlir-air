module {

func @graph(%arg0 : memref<32x16xi32>, %arg1 : memref<32x16xi32>) -> () {
  %herd_cols = constant 1 : index
  %herd_rows = constant 1 : index
  air.launch_herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<32x16xi32>, memref<32x16xi32> attributes { } {
    %c0 = constant 0 : index
    %c128 = constant 128 : index
    %c32 = constant 32 : index
    %c16 = constant 16 : index
    %buf0 = memref.alloc() {sym_name = "scratch"}: memref<16x8xi32, 2>
    %buf1 = memref.alloc() {sym_name = "scratch_copy"}: memref<16x8xi32, 2>
    air.dma_memcpy_2d (%buf0, %ext0, [%c0, %c0], [%c0, %c0], %c128, %c32, %c16) {id = 1 : i32} : (memref<16x8xi32, 2>, memref<32x16xi32>, [index, index], [index, index], index, index, index) -> ()
    affine.for %i = 0 to 16 {
      affine.for %j = 0 to 8 {
        %0 = affine.load %buf0[%i, %j] : memref<16x8xi32, 2>
        affine.store %0, %buf1[%i, %j] : memref<16x8xi32, 2>
      }
    }
    air.dma_memcpy_2d (%ext1, %buf1, [%c0, %c0], [%c0, %c0], %c128, %c32, %c16) {id = 2 : i32} : (memref<32x16xi32>, memref<16x8xi32, 2>, [index, index], [index, index], index, index, index) -> ()
    memref.dealloc %buf1 : memref<16x8xi32, 2>
    memref.dealloc %buf0 : memref<16x8xi32, 2>
    air.herd_terminator
  }
  return
}

}


