air.channel @channel_0 [1]
func.func @forward(%arg0 : memref<16x16xi32>, %arg1 : memref<16x16xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  air.channel.put @channel_0[] (%arg0[%c0, %c0] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
  air.channel.get @channel_0[] (%arg1[%c8, %c8] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
  return
}

