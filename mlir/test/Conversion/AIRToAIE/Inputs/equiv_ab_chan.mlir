air.channel @cw []
func.func @ab(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0_l = arith.constant 0 : index
    %c32_l = arith.constant 32 : index
    %c64_l = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @cw[] (%la[%c0_l, %c0_l] [%c32_l, %c32_l] [%c64_l, %c1_l]) : (memref<64x64xi32>)
    air.segment {
      %c1_0 = arith.constant 1 : index
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.channel.get @cw[] (%alloc[] [] []) : (memref<32x32xi32, 2>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
    }
  }
  return
}
