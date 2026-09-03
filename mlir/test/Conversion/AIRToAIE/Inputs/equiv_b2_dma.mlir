air.channel @bw [2, 2]
func.func @ab(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    air.segment args(%sa=%la) : memref<64x64xi32> {
      %c2 = arith.constant 2 : index
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%a=%sa) : memref<64x64xi32> {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %cst1 = arith.constant 1 : index
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @bw} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
    }
  }
  return
}
