air.channel @f0 []
air.channel @f1 []
air.channel @f2 []
func.func @ab(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0_l = arith.constant 0 : index
    %c32_l = arith.constant 32 : index
    %c64_l = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @f0[] (%la[%c0_l, %c0_l] [%c32_l, %c32_l] [%c64_l, %c1_l]) : (memref<64x64xi32>)
    air.channel.put @f1[] (%la[%c0_l, %c0_l] [%c32_l, %c32_l] [%c64_l, %c1_l]) : (memref<64x64xi32>)
    air.channel.put @f2[] (%la[%c0_l, %c0_l] [%c32_l, %c32_l] [%c64_l, %c1_l]) : (memref<64x64xi32>)
    air.segment {
      %c1_0 = arith.constant 1 : index
      air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
        %al0 = memref.alloc() : memref<32x32xi32, 2>
        %al1 = memref.alloc() : memref<32x32xi32, 2>
        %al2 = memref.alloc() : memref<32x32xi32, 2>
        air.channel.get @f0[] (%al0[] [] []) : (memref<32x32xi32, 2>)
        air.channel.get @f1[] (%al1[] [] []) : (memref<32x32xi32, 2>)
        air.channel.get @f2[] (%al2[] [] []) : (memref<32x32xi32, 2>)
        memref.dealloc %al0 : memref<32x32xi32, 2>
        memref.dealloc %al1 : memref<32x32xi32, 2>
        memref.dealloc %al2 : memref<32x32xi32, 2>
      }
    }
  }
  return
}
