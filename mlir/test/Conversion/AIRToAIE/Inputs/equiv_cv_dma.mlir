air.channel @cv [] {channel_type = "npu_dma_packet"}
func.func @ab(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %lb=%arg1) : memref<64x64xi32>, memref<64x64xi32> {
    air.segment args(%sa=%la, %sb=%lb) : memref<64x64xi32>, memref<64x64xi32> {
      %c1_0 = arith.constant 1 : index
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa, %b=%sb) : memref<64x64xi32>, memref<64x64xi32> {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %cst1 = arith.constant 1 : index
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        %alloc2 = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @cv} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        air.dma_memcpy_nd (%alloc2[] [] [], %b[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 2 : i32, channel = @cv} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
        memref.dealloc %alloc2 : memref<32x32xi32, 2>
      }
    }
  }
  return
}
