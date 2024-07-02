module {
  air.channel @ChanIn []
  air.channel @ChanOut []
  air.channel @MiddleChannel []
  func.func @copy(%arg0: memref<4x4xi32>, %arg1: memref<4x4xi32>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<4x4xi32>, memref<4x4xi32> {
      air.channel.put  @ChanIn[] (%arg2[] [] []) : (memref<4x4xi32>)
      air.channel.get  @ChanOut[] (%arg3[] [] []) : (memref<4x4xi32>)
      air.segment @seg  {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        air.herd @partial_herd  tile (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1_0) {
          %c0 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_1 = arith.constant 1 : index
          scf.for %arg8 = %c0 to %c16 step %c1_1 {
            %alloc = memref.alloc() : memref<1x1xi32, 2 : i32>
            %alloc_5 = memref.alloc() : memref<1x1xi32, 2 : i32>
            air.channel.get  @ChanIn[] (%alloc[] [] []) : (memref<1x1xi32, 2 : i32>)
            %c0_6 = arith.constant 0 : index
            %0 = memref.load %alloc[%c0_6, %c0_6] : memref<1x1xi32, 2 : i32>
            %1 = arith.index_cast %arg8 : index to i32
            %2 = arith.addi %0, %1 : i32
            memref.store %2, %alloc_5[%c0_6, %c0_6] : memref<1x1xi32, 2 : i32>
            air.channel.put  @MiddleChannel[] (%alloc_5[] [] []) : (memref<1x1xi32, 2 : i32>)
            memref.dealloc %alloc : memref<1x1xi32, 2 : i32>
            memref.dealloc %alloc_5 : memref<1x1xi32, 2 : i32>
          }
          %c0_2 = arith.constant 0 : index
          %c16_3 = arith.constant 16 : index
          %c1_4 = arith.constant 1 : index
          scf.for %arg8 = %c0_2 to %c16_3 step %c1_4 {
            %alloc = memref.alloc() : memref<1x1xi32, 2 : i32>
            %alloc_5 = memref.alloc() : memref<1x1xi32, 2 : i32>
            air.channel.get  @MiddleChannel[] (%alloc[] [] []) : (memref<1x1xi32, 2 : i32>)
            %c0_6 = arith.constant 0 : index
            %0 = memref.load %alloc[%c0_6, %c0_6] : memref<1x1xi32, 2 : i32>
            %1 = arith.index_cast %arg8 : index to i32
            %2 = arith.addi %0, %1 : i32
            memref.store %2, %alloc_5[%c0_6, %c0_6] : memref<1x1xi32, 2 : i32>
            air.channel.put  @ChanOut[] (%alloc_5[] [] []) : (memref<1x1xi32, 2 : i32>)
            memref.dealloc %alloc : memref<1x1xi32, 2 : i32>
            memref.dealloc %alloc_5 : memref<1x1xi32, 2 : i32>
          }
        }
      }
    }
    return
  }
}

