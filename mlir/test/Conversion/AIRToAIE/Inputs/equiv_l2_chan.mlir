air.channel @toL2 []
air.channel @toL1 [2, 2]
func.func @ab(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0_l = arith.constant 0 : index
    %c64_l = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @toL2[] (%la[%c0_l, %c0_l] [%c64_l, %c64_l] [%c64_l, %c1_l]) : (memref<64x64xi32>)
    air.segment {
      %c2 = arith.constant 2 : index
      %c0s = arith.constant 0 : index
      %c1s = arith.constant 1 : index
      %c32s = arith.constant 32 : index
      %c64s = arith.constant 64 : index
      %l2 = memref.alloc() : memref<64x64xi32, 1>
      air.channel.get @toL2[] (%l2[] [] []) : (memref<64x64xi32, 1>)
      scf.parallel (%i, %j) = (%c0s, %c0s) to (%c2, %c2) step (%c1s, %c1s) {
        air.channel.put @toL1[%i, %j] (%l2[%c0s, %c0s] [%c32s, %c32s] [%c64s, %c1s]) : (memref<64x64xi32, 1>)
        scf.reduce
      }
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c2, %sy=%c2) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.channel.get @toL1[%tx, %ty] (%alloc[] [] []) : (memref<32x32xi32, 2>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
      memref.dealloc %l2 : memref<64x64xi32, 1>
    }
  }
  return
}
