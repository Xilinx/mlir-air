// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

module @aie.0  {
  %t70 = AIE.tile(7, 0)
  %t74 = AIE.tile(7, 4)

  %4 = AIE.lock(%t74, 0)
  %5 = AIE.lock(%t74, 1)

  %6 = AIE.buffer(%t74) {sym_name = "buf0"} : memref<64xi32, 2>
  %7 = AIE.buffer(%t74) {sym_name = "buf1"} : memref<64xi32, 2>

  %8 = AIE.mem(%t74)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb0, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb2, ^end)
  ^bb0: 
    AIE.useLock(%4, Acquire, 0)
    AIE.dmaBd(<%6 : memref<64xi32, 2>,   0, 64>, 0)
    AIE.useLock(%4, Release, 1)
    br ^bb1
  ^bb1: 
    AIE.useLock(%5, Acquire, 0)
    AIE.dmaBd(<%7 : memref<64xi32, 2>,  0, 64>, 0)
    AIE.useLock(%5, Release, 1)
    br ^bb0
  ^bb2: 
    AIE.useLock(%4, Acquire, 1)
    AIE.dmaBd(<%6 : memref<64xi32, 2>,   0, 64>, 0)
    AIE.useLock(%4, Release, 0)
    br ^bb3
  ^bb3: 
    AIE.useLock(%5, Acquire, 1)
    AIE.dmaBd(<%7 : memref<64xi32, 2>,   0, 64>, 0)
    AIE.useLock(%5, Release, 0)
    br ^bb2
  ^end: 
    AIE.end
  }

  AIE.flow(%t74, DMA : 0, %t70, PLIO : 1)
  AIE.flow(%t70, PLIO : 4, %t74, DMA : 0)
}
