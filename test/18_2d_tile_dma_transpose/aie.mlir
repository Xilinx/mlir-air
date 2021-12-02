// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

module {
  %t70 = AIE.tile(7, 0)
  %t72 = AIE.tile(7, 2)

  AIE.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)

  %buf72_0 = AIE.buffer(%t72) { sym_name = "buf0" } : memref<4x8xi32>

  %l72_0 = AIE.lock(%t72, 0)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0, 0)
      AIE.dmaBd2D(4,1,8,1,8,4)
      AIE.dmaBd(<%buf72_0 : memref<4x8xi32>, 0, 32>, 0)
      AIE.useLock(%l72_0, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l72_0, "Acquire", 1, 0)
      AIE.dmaBd2D(4,1,8,1,8,4)
      AIE.dmaBd(<%buf72_0 : memref<4x8xi32>, 0, 32>, 0)
      AIE.useLock(%l72_0, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

}
