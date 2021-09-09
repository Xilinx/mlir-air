// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

module {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  // Fixup
  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux = AIE.shimmux(%t70) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }

  AIE.flow(%t71, "South" : 0, %t72, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 0, %t71, "South" : 0)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0"} : memref<512xi32>

  %l72_0 = AIE.lock(%t72, 0)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf72_0 : memref<512xi32>, 0, 256>, 0)
      AIE.useLock(%l72_0, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l72_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf72_0 : memref<512xi32>, 0, 256>, 0)
      AIE.useLock(%l72_0, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end
  }

}
