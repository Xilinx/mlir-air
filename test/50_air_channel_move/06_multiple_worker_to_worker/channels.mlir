#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 2 >= 0)>
module @aie.partition_0 {
  air.channel @channel_0 [1, 2]
  func.func @test() {
    %c2 = arith.constant 2 : index
    air.herd @herd_0 tile (%arg0, %arg1) in (%arg2=%c2, %arg3=%c2) {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      affine.if #set()[%arg0, %arg1] {
        %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
        air.channel.put @channel_0[%c0, %arg1] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
        memref.dealloc %alloc : memref<32xi32, 2>
      } else {
        %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
        air.channel.get @channel_0[%c0, %arg1] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
        memref.dealloc %alloc : memref<32xi32, 2>
      }
      air.herd_terminator
    }
    return
  }
}