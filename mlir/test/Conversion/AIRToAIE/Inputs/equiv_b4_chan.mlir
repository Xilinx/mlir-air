air.channel @bw [4, 4]
func.func @ab(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0_l = arith.constant 0 : index
    %c4_l = arith.constant 4 : index
    %c32_l = arith.constant 32 : index
    %c64_l = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    scf.parallel (%i, %j) = (%c0_l, %c0_l) to (%c4_l, %c4_l) step (%c1_l, %c1_l) {
      air.channel.put @bw[%i, %j] (%la[%c0_l, %c0_l] [%c32_l, %c32_l] [%c64_l, %c1_l]) : (memref<64x64xi32>)
      scf.reduce
    }
    air.segment {
      %c4 = arith.constant 4 : index
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c4, %sy=%c4) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.channel.get @bw[%tx, %ty] (%alloc[] [] []) : (memref<32x32xi32, 2>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
    }
  }
  return
}
