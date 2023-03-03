module {
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @graph(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    affine.for %arg2 = 0 to 4096 step 32 {
      air.channel.put  @channel_0[] (%arg0[%arg2] [%c32] [%c0]) : (memref<4096xi32>)
    }
    affine.for %arg2 = 0 to 4096 step 32 {
      air.channel.get  @channel_1[] (%arg1[%arg2] [%c32] [%c0]) : (memref<4096xi32>)
    }
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) {
      %c0_0 = arith.constant 0 : index
      %c32_1 = arith.constant 32 : index
      %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
      %alloc_2 = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
      affine.for %arg6 = 0 to 4096 step 32 {
        air.channel.get  @channel_0[] (%alloc[%c0_0] [%c32_1] [%c0_0]) : (memref<32xi32, 2>)
        affine.for %arg7 = 0 to 32 {
          %0 = affine.load %alloc[%arg7] : memref<32xi32, 2>
          affine.store %0, %alloc_2[%arg7] : memref<32xi32, 2>
        }
        air.channel.put  @channel_1[] (%alloc_2[%c0_0] [%c32_1] [%c0_0]) : (memref<32xi32, 2>)
      }
      memref.dealloc %alloc_2 : memref<32xi32, 2>
      memref.dealloc %alloc : memref<32xi32, 2>
      air.herd_terminator
    }
    return
  }
}

