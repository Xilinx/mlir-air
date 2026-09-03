air.channel @f0 []
air.channel @f1 []
air.channel @f2 []
func.func @ab(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    air.segment args(%sa=%la) : memref<64x64xi32> {
      %c1_0 = arith.constant 1 : index
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa) : memref<64x64xi32> {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %cst1 = arith.constant 1 : index
        %al0 = memref.alloc() : memref<32x32xi32, 2>
        %al1 = memref.alloc() : memref<32x32xi32, 2>
        %al2 = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%al0[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @f0} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        air.dma_memcpy_nd (%al1[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 2 : i32, channel = @f1} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        air.dma_memcpy_nd (%al2[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 3 : i32, channel = @f2} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        memref.dealloc %al0 : memref<32x32xi32, 2>
        memref.dealloc %al1 : memref<32x32xi32, 2>
        memref.dealloc %al2 : memref<32x32xi32, 2>
      }
    }
  }
  return
}
