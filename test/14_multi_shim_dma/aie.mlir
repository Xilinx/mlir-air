// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

module {
  %t30 = AIE.tile(18, 0)
  %tb0 = AIE.tile(11, 0)
  %t72 = AIE.tile(7, 2)
  %t74 = AIE.tile(7, 4)

  AIE.flow(%t30, "DMA" : 0, %t72, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 0, %t30, "DMA" : 0)
  AIE.flow(%tb0, "DMA" : 0, %t74, "DMA" : 0)
  AIE.flow(%t74, "DMA" : 0, %tb0, "DMA" : 0)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0"} : memref<32xi32>
  %buf74_0 = AIE.buffer(%t74) {sym_name = "buf74_0"}: memref<32xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l74_0 = AIE.lock(%t74, 0)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf72_0 : memref<32xi32>, 0, 32>, 0)
      AIE.useLock(%l72_0, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l72_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf72_0 : memref<32xi32>, 0, 32>, 0)
      AIE.useLock(%l72_0, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

  %m74 = AIE.mem(%t74) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l74_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf74_0 : memref<32xi32>, 0, 32>, 0)
      AIE.useLock(%l74_0, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l74_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf74_0 : memref<32xi32>, 0, 32>, 0)
      AIE.useLock(%l74_0, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

}
