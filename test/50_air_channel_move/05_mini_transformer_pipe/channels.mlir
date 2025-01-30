#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
module {
  air.channel @channel_5 [1, 1] // D
  air.channel @channel_4 [1, 1] // C
  air.channel @channel_3 [1, 1] // partial
  air.channel @channel_2 [1, 1] // output
  air.channel @channel_1 [1, 1] // B
  air.channel @channel_0 [1, 1] // A
  func.func @graph(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg01: memref<4096xi32>, %arg11: memref<4096xi32>, %arg2: memref<4096xi32>) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    affine.for %arg3 = 0 to 4096 step 32 {
      air.channel.put  @channel_0[] (%arg0[%arg3] [%c32] [%c0]) : (memref<4096xi32>)
    }
    affine.for %arg3 = 0 to 4096 step 32 {
      air.channel.put  @channel_1[] (%arg1[%arg3] [%c32] [%c0]) : (memref<4096xi32>)
    }
    affine.for %arg3 = 0 to 4096 step 32 {
      air.channel.put  @channel_4[] (%arg01[%arg3] [%c32] [%c0]) : (memref<4096xi32>)
    }
    affine.for %arg3 = 0 to 4096 step 32 {
      air.channel.put  @channel_5[] (%arg11[%arg3] [%c32] [%c0]) : (memref<4096xi32>)
    }
    affine.for %arg3 = 0 to 4096 step 32 {
      air.channel.get  @channel_2[] (%arg2[%arg3] [%c32] [%c0]) : (memref<4096xi32>)
    }
    air.herd @herd_0  tile (%arg3, %arg4) in (%arg5=%c1, %arg6=%c2) {
      %c0_0 = arith.constant 0 : index
      %c32_1 = arith.constant 32 : index
      affine.if #set()[%arg3, %arg4] {
        %alloc = memref.alloc() {sym_name = "inA"} : memref<32xi32, 2>
        %alloc_2 = memref.alloc() {sym_name = "inB"} : memref<32xi32, 2>
        %alloc_3 = memref.alloc() {sym_name = "outC"} : memref<32xi32, 2>
        affine.for %arg7 = 0 to 4096 step 32 {
          air.channel.get  @channel_0[] (%alloc[%c0_0] [%c32_1] [%c0_0]) : (memref<32xi32, 2>)
          air.channel.get  @channel_1[] (%alloc_2[%c0_0] [%c32_1] [%c0_0]) : (memref<32xi32, 2>)
          affine.for %arg8 = 0 to 32 {
            %0 = affine.load %alloc[%arg8] : memref<32xi32, 2>
            %1 = affine.load %alloc_2[%arg8] : memref<32xi32, 2>
            %2 = arith.addi %1, %0 : i32
            affine.store %2, %alloc_3[%arg8] : memref<32xi32, 2>
          }
          air.channel.put  @channel_3[] (%alloc_3[%c0_0] [%c32_1] [%c0_0]) : (memref<32xi32, 2>)
        }
        memref.dealloc %alloc_3 : memref<32xi32, 2>
        memref.dealloc %alloc_2 : memref<32xi32, 2>
        memref.dealloc %alloc : memref<32xi32, 2>
      }
      affine.if #set1()[%arg3, %arg4] {
          %alloc = memref.alloc() {sym_name = "inA"} : memref<32xi32, 2>
          %alloc_1 = memref.alloc() {sym_name = "inB"} : memref<32xi32, 2>
          %alloc_2 = memref.alloc() {sym_name = "inC"} : memref<32xi32, 2>
          %alloc_3 = memref.alloc() {sym_name = "outC"} : memref<32xi32, 2>
          affine.for %arg7 = 0 to 4096 step 32 {
            air.channel.get  @channel_4[] (%alloc[%c0_0] [%c32_1] [%c0_0]) : (memref<32xi32, 2>)
            air.channel.get  @channel_5[] (%alloc_1[%c0_0] [%c32_1] [%c0_0]) : (memref<32xi32, 2>)
            air.channel.get  @channel_3[] (%alloc_2[%c0_0] [%c32_1] [%c0_0]) : (memref<32xi32, 2>)
            affine.for %arg8 = 0 to 32 {
              %0 = affine.load %alloc[%arg8] : memref<32xi32, 2>
              %1 = affine.load %alloc_1[%arg8] : memref<32xi32, 2>
              %2 = arith.addi %1, %0 : i32
              %3 = affine.load %alloc_2[%arg8] : memref<32xi32, 2>
              %4 = arith.addi %2, %3 : i32
              affine.store %4, %alloc_3[%arg8] : memref<32xi32, 2>
            }
            air.channel.put  @channel_2[] (%alloc_3[%c0_0] [%c32_1] [%c0_0]) : (memref<32xi32, 2>)
          }
          memref.dealloc %alloc_3 : memref<32xi32, 2>
          memref.dealloc %alloc_2 : memref<32xi32, 2>
          memref.dealloc %alloc_1 : memref<32xi32, 2>
          memref.dealloc %alloc : memref<32xi32, 2>
        }
        air.herd_terminator
      }
    return
  }
}
