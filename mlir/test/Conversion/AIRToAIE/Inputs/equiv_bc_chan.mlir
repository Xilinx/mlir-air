#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
air.channel @bc [1, 1] {broadcast_shape = [1, 2]}
func.func @ab(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0_l = arith.constant 0 : index
    %c32_l = arith.constant 32 : index
    %c64_l = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @bc[] (%la[%c0_l, %c0_l] [%c32_l, %c32_l] [%c64_l, %c1_l]) {broadcast_set = #set} : (memref<64x64xi32>)
    air.segment {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c2) {
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        affine.if #set()[%tx, %ty] {
          air.channel.get @bc[%tx, %ty] (%alloc[] [] []) : (memref<32x32xi32, 2>)
        }
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
    }
  }
  return
}
