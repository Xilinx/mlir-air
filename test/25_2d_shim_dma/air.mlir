module {

func @graph(%arg0 : memref<32x16xi32>, %arg1 : memref<32x16xi32>) -> () {
  %herd_cols = constant 1 : index
  %herd_rows = constant 1 : index
  air.launch_herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<32x16xi32>, memref<32x16xi32> attributes { } {
    %c0 = constant 0 : index
    %c128 = constant 128 : index
    %buf0 = alloc() {sym_name = "scratch"}: memref<16x8xi32, 2>
    air.dma_memcpy_2d (%buf0, %ext0, [%c0, %c0], [%c0, %c0], %c128) : (memref<16x8xi32, 2>, memref<32x16xi32>, [index, index], [index, index], index) -> ()
    air.dma_memcpy_2d (%ext1, %buf0, [%c0, %c0], [%c0, %c0], %c128) : (memref<32x16xi32>, memref<16x8xi32, 2>, [index, index], [index, index], index) -> ()
    air.herd_terminator
  }
  return
}

}
