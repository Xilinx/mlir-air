// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

module @aie.0  {
  %t70 = AIE.tile(7, 0)
  %t74 = AIE.tile(7, 4)

  %6 = AIE.lock(%t74, 1)
  %7 = AIE.buffer(%t74) {sym_name = "buf1"} : memref<16xi32, 2>

  %8 = AIE.mem(%t74)  {
    %9 = AIE.dmaStart(MM2S0, ^bb1, ^bb4)
  ^bb1: 
    br ^bb2
  ^bb2: 
    AIE.useLock(%6, Acquire, 1, 0)
    AIE.dmaBd(<%7 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%6, Release, 0, 0)
    br ^bb3
  ^bb3: 
    br ^bb1
  ^bb4: 
    AIE.end
  }
  AIE.flow(%t74, DMA : 0, %t70, PLIO : 0)
}
