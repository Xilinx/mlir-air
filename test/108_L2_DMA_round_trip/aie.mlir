// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

module @aie.0  {
  %t70 = AIE.tile(7, 0)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  %10 = AIE.lock(%t73, 1)
  %11 = AIE.buffer(%t73) {sym_name = "buf1"} : memref<16xi32, 2>

  %6 = AIE.lock(%t74, 1)
  %7 = AIE.buffer(%t74) {sym_name = "buf2"} : memref<16xi32, 2>

  %8 = AIE.mem(%t74)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2: 
    AIE.useLock(%6, Acquire, 0, 0)
    AIE.dmaBd(<%7 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%6, Release, 1, 0)
    br ^bb2
  ^bb3: 
    AIE.useLock(%6, Acquire, 1, 0)
    AIE.dmaBd(<%7 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%6, Release, 0, 0)
    br ^bb3
  ^end: 
    AIE.end
  }
  %12 = AIE.mem(%t73)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2: 
    AIE.useLock(%10, Acquire, 0, 0)
    AIE.dmaBd(<%11 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%10, Release, 1, 0)
    br ^bb2
  ^bb3: 
    AIE.useLock(%10, Acquire, 1, 0)
    AIE.dmaBd(<%11 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%10, Release, 0, 0)
    br ^bb3
  ^end: 
    AIE.end
  }
  AIE.flow(%t73, DMA : 0, %t70, PLIO : 0)
  AIE.flow(%t74, DMA : 0, %t70, PLIO : 1)
  AIE.flow(%t70, PLIO : 4, %t73, DMA : 0)
  AIE.flow(%t70, PLIO : 5, %t74, DMA : 0)
}
