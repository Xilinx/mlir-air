// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

module {
  %t20 = AIE.tile(2, 0)
  %t30 = AIE.tile(3, 0)
  %t60 = AIE.tile(6, 0)
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)
  %t82 = AIE.tile(8, 2)
  %t83 = AIE.tile(8, 3)
  %t342 = AIE.tile(34, 2)

  AIE.flow(%t20, "DMA" : 0, %t72, "DMA" : 0)
  AIE.flow(%t20, "DMA" : 1, %t73, "DMA" : 0)
  AIE.flow(%t30, "DMA" : 0, %t82, "DMA" : 0)
  AIE.flow(%t30, "DMA" : 1, %t83, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 0, %t20, "DMA" : 0)
  AIE.flow(%t73, "DMA" : 0, %t20, "DMA" : 1)
  AIE.flow(%t82, "DMA" : 0, %t30, "DMA" : 0)
  AIE.flow(%t83, "DMA" : 0, %t30, "DMA" : 1)

  %buf72_0 = AIE.buffer(%t72) { sym_name = "t72_ping_in" } : memref<8xi32>
  %buf72_1 = AIE.buffer(%t72) { sym_name = "t72_ping_out" } : memref<8xi32>
  %buf72_2 = AIE.buffer(%t72) { sym_name = "t72_pong_in" } : memref<8xi32>
  %buf72_3 = AIE.buffer(%t72) { sym_name = "t72_pong_out" } : memref<8xi32>
  %buf73_0 = AIE.buffer(%t73) { sym_name = "t73_ping_in" } : memref<8xi32>
  %buf73_1 = AIE.buffer(%t73) { sym_name = "t73_ping_out" } : memref<8xi32>
  %buf73_2 = AIE.buffer(%t73) { sym_name = "t73_pong_in" } : memref<8xi32>
  %buf73_3 = AIE.buffer(%t73) { sym_name = "t73_pong_out" } : memref<8xi32>
  %buf82_0 = AIE.buffer(%t82) { sym_name = "t82_ping_in" } : memref<8xi32>
  %buf82_1 = AIE.buffer(%t82) { sym_name = "t82_ping_out" } : memref<8xi32>
  %buf82_2 = AIE.buffer(%t82) { sym_name = "t82_pong_in" } : memref<8xi32>
  %buf82_3 = AIE.buffer(%t82) { sym_name = "t82_pong_out" } : memref<8xi32>
  %buf83_0 = AIE.buffer(%t83) { sym_name = "t83_ping_in" } : memref<8xi32>
  %buf83_1 = AIE.buffer(%t83) { sym_name = "t83_ping_out" } : memref<8xi32>
  %buf83_2 = AIE.buffer(%t83) { sym_name = "t83_pong_in" } : memref<8xi32>
  %buf83_3 = AIE.buffer(%t83) { sym_name = "t83_pong_out" } : memref<8xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)
  %l72_2 = AIE.lock(%t72, 2)
  %l72_3 = AIE.lock(%t72, 3)
  %l73_0 = AIE.lock(%t73, 0)
  %l73_1 = AIE.lock(%t73, 1)
  %l73_2 = AIE.lock(%t73, 2)
  %l73_3 = AIE.lock(%t73, 3)
  %l82_0 = AIE.lock(%t82, 0)
  %l82_1 = AIE.lock(%t82, 1)
  %l82_2 = AIE.lock(%t82, 2)
  %l82_3 = AIE.lock(%t82, 3)
  %l83_0 = AIE.lock(%t83, 0)
  %l83_1 = AIE.lock(%t83, 1)
  %l83_2 = AIE.lock(%t83, 2)
  %l83_3 = AIE.lock(%t83, 3)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0)
      AIE.dmaBd(<%buf72_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_0, "Release", 1)
      br ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0)
      AIE.dmaBd(<%buf72_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_1, "Release", 1)
      br ^bd0
    ^bd2:
      AIE.useLock(%l72_2, "Acquire", 1)
      AIE.dmaBd(<%buf72_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_2, "Release", 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l72_3, "Acquire", 1)
      AIE.dmaBd(<%buf72_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_3, "Release", 0)
      br ^bd2
    ^end:
      AIE.end
  }
  %m73 = AIE.mem(%t73) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l73_0, "Acquire", 0)
      AIE.dmaBd(<%buf73_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l73_0, "Release", 1)
      br ^bd1
    ^bd1:
      AIE.useLock(%l73_1, "Acquire", 0)
      AIE.dmaBd(<%buf73_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l73_1, "Release", 1)
      br ^bd0
    ^bd2:
      AIE.useLock(%l73_2, "Acquire", 1)
      AIE.dmaBd(<%buf73_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l73_2, "Release", 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l73_3, "Acquire", 1)
      AIE.dmaBd(<%buf73_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l73_3, "Release", 0)
      br ^bd2
    ^end:
      AIE.end
  }
  %m82 = AIE.mem(%t82) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l82_0, "Acquire", 0)
      AIE.dmaBd(<%buf82_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l82_0, "Release", 1)
      br ^bd1
    ^bd1:
      AIE.useLock(%l82_1, "Acquire", 0)
      AIE.dmaBd(<%buf82_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l82_1, "Release", 1)
      br ^bd0
    ^bd2:
      AIE.useLock(%l82_2, "Acquire", 1)
      AIE.dmaBd(<%buf82_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l82_2, "Release", 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l82_3, "Acquire", 1)
      AIE.dmaBd(<%buf82_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l82_3, "Release", 0)
      br ^bd2
    ^end:
      AIE.end
  }
  %m83 = AIE.mem(%t83) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l83_0, "Acquire", 0)
      AIE.dmaBd(<%buf83_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l83_0, "Release", 1)
      br ^bd1
    ^bd1:
      AIE.useLock(%l83_1, "Acquire", 0)
      AIE.dmaBd(<%buf83_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l83_1, "Release", 1)
      br ^bd0
    ^bd2:
      AIE.useLock(%l83_2, "Acquire", 1)
      AIE.dmaBd(<%buf83_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l83_2, "Release", 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l83_3, "Acquire", 1)
      AIE.dmaBd(<%buf83_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l83_3, "Release", 0)
      br ^bd2
    ^end:
      AIE.end
  }

  AIE.core(%t72) {
    %c8 = constant 8 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c1_32 = constant 1 : i32

    AIE.useLock(%l72_0, "Acquire", 1)
    AIE.useLock(%l72_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf72_0[%arg3] : memref<8xi32>
        %1 = addi %0, %c1_32 : i32
        memref.store %1, %buf72_1[%arg3] : memref<8xi32>
    }
    AIE.useLock(%l72_0, "Release", 0)
    AIE.useLock(%l72_2, "Release", 1)

    AIE.useLock(%l72_1, "Acquire", 1)
    AIE.useLock(%l72_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf72_2[%arg4] : memref<8xi32>
        %3 = addi %2, %c1_32 : i32
        memref.store %3, %buf72_3[%arg4] : memref<8xi32>
    }
    AIE.useLock(%l72_1, "Release", 0)
    AIE.useLock(%l72_3, "Release", 1)
    AIE.end
  }
  AIE.core(%t73) {
    %c8 = constant 8 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c1_32 = constant 1 : i32

    AIE.useLock(%l73_0, "Acquire", 1)
    AIE.useLock(%l73_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf73_0[%arg3] : memref<8xi32>
        %1 = addi %0, %c1_32 : i32
        memref.store %1, %buf73_1[%arg3] : memref<8xi32>
    }
    AIE.useLock(%l73_0, "Release", 0)
    AIE.useLock(%l73_2, "Release", 1)

    AIE.useLock(%l73_1, "Acquire", 1)
    AIE.useLock(%l73_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf73_2[%arg4] : memref<8xi32>
        %3 = addi %2, %c1_32 : i32
        memref.store %3, %buf73_3[%arg4] : memref<8xi32>
    }
    AIE.useLock(%l73_1, "Release", 0)
    AIE.useLock(%l73_3, "Release", 1)
    AIE.end
  }
  AIE.core(%t82) {
    %c8 = constant 8 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c1_32 = constant 1 : i32

    AIE.useLock(%l82_0, "Acquire", 1)
    AIE.useLock(%l82_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf82_0[%arg3] : memref<8xi32>
        %1 = addi %0, %c1_32 : i32
        memref.store %1, %buf82_1[%arg3] : memref<8xi32>
    }
    AIE.useLock(%l82_0, "Release", 0)
    AIE.useLock(%l82_2, "Release", 1)

    AIE.useLock(%l82_1, "Acquire", 1)
    AIE.useLock(%l82_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf82_2[%arg4] : memref<8xi32>
        %3 = addi %2, %c1_32 : i32
        memref.store %3, %buf82_3[%arg4] : memref<8xi32>
    }
    AIE.useLock(%l82_1, "Release", 0)
    AIE.useLock(%l82_3, "Release", 1)
    AIE.end
  }
  AIE.core(%t83) {
    %c8 = constant 8 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c1_32 = constant 1 : i32

    AIE.useLock(%l83_0, "Acquire", 1)
    AIE.useLock(%l83_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf83_0[%arg3] : memref<8xi32>
        %1 = addi %0, %c1_32 : i32
        memref.store %1, %buf83_1[%arg3] : memref<8xi32>
    }
    AIE.useLock(%l83_0, "Release", 0)
    AIE.useLock(%l83_2, "Release", 1)

    AIE.useLock(%l83_1, "Acquire", 1)
    AIE.useLock(%l83_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf83_2[%arg4] : memref<8xi32>
        %3 = addi %2, %c1_32 : i32
        memref.store %3, %buf83_3[%arg4] : memref<8xi32>
    }
    AIE.useLock(%l83_1, "Release", 0)
    AIE.useLock(%l83_3, "Release", 1)
    AIE.end
  }

  AIE.flow(%t60,  "DMA" : 0, %t342, "DMA" : 0)
  AIE.flow(%t342, "DMA" : 0, %t60,  "DMA" : 0)

  %buf342_0 = AIE.buffer(%t342) { sym_name = "ping_in2" } : memref<8xi32>
  %buf342_1 = AIE.buffer(%t342) { sym_name = "ping_out2" } : memref<8xi32>
  %buf342_2 = AIE.buffer(%t342) { sym_name = "pong_in2" } : memref<8xi32>
  %buf342_3 = AIE.buffer(%t342) { sym_name = "pong_out2" } : memref<8xi32>

  %l342_0 = AIE.lock(%t342, 0)
  %l342_1 = AIE.lock(%t342, 1)
  %l342_2 = AIE.lock(%t342, 2)
  %l342_3 = AIE.lock(%t342, 3)

  %m342 = AIE.mem(%t342) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l342_0, "Acquire", 0)
      AIE.dmaBd(<%buf342_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l342_0, "Release", 1)
      br ^bd1
    ^bd1:
      AIE.useLock(%l342_1, "Acquire", 0)
      AIE.dmaBd(<%buf342_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l342_1, "Release", 1)
      br ^bd0
    ^bd2:
      AIE.useLock(%l342_2, "Acquire", 1)
      AIE.dmaBd(<%buf342_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l342_2, "Release", 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l342_3, "Acquire", 1)
      AIE.dmaBd(<%buf342_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l342_3, "Release", 0)
      br ^bd2
    ^end:
      AIE.end
  }

  AIE.core(%t342) {
    %c8 = constant 8 : index
    %c0 = constant 0 : index
    %c4 = constant 4 : index
    %c1 = constant 1 : index
    %c1_32 = constant 1 : i32

    scf.for %arg2 = %c0 to %c4 step %c1 {
      AIE.useLock(%l342_0, "Acquire", 1)
      AIE.useLock(%l342_2, "Acquire", 0)
      scf.for %arg3 = %c0 to %c8 step %c1 {
          %0 = memref.load %buf342_0[%arg3] : memref<8xi32>
          %1 = addi %0, %c1_32 : i32
          memref.store %1, %buf342_1[%arg3] : memref<8xi32>
      }
      AIE.useLock(%l342_0, "Release", 0)
      AIE.useLock(%l342_2, "Release", 1)

      AIE.useLock(%l342_1, "Acquire", 1)
      AIE.useLock(%l342_3, "Acquire", 0)
      scf.for %arg4 = %c0 to %c8 step %c1 {
          %2 = memref.load %buf342_2[%arg4] : memref<8xi32>
          %3 = addi %2, %c1_32 : i32
          memref.store %3, %buf342_3[%arg4] : memref<8xi32>
      }
      AIE.useLock(%l342_1, "Release", 0)
      AIE.useLock(%l342_3, "Release", 1)
    }
    AIE.end
  }

}
