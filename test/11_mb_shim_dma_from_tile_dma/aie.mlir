// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.


module {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)

  %buf72_0 = AIE.buffer(%t72) {sym_name="a"} : memref<256xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name="b"} : memref<256xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("MM2S0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 1)
      AIE.dmaBd(<%buf72_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_0, "Release", 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 1)
      AIE.dmaBd(<%buf72_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_1, "Release", 0)
      br ^bd0
    ^end:
      AIE.end
  }
  AIE.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)
}
