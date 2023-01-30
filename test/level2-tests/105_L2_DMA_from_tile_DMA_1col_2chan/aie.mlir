//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module @aie.0  {
  %t70 = AIE.tile(7, 0)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  %6 = AIE.lock(%t74, 1)
  %7 = AIE.buffer(%t74) {sym_name = "buf1"} : memref<16xi32, 2>
  %10 = AIE.lock(%t73, 1)
  %11 = AIE.buffer(%t73) {sym_name = "buf2"} : memref<16xi32, 2>

  %8 = AIE.mem(%t74)  {
    %9 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb4)
  ^bb1: 
    AIE.nextBd ^bb2
  ^bb2: 
    AIE.useLock(%6, Acquire, 1)
    AIE.dmaBd(<%7 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%6, Release, 0)
    AIE.nextBd ^bb3
  ^bb3: 
    AIE.nextBd ^bb1
  ^bb4: 
    AIE.end
  }
  %12 = AIE.mem(%t73)  {
    %13 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb4)
  ^bb1: 
    AIE.nextBd ^bb2
  ^bb2: 
    AIE.useLock(%10, Acquire, 1)
    AIE.dmaBd(<%11 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%10, Release, 0)
    AIE.nextBd ^bb3
  ^bb3: 
    AIE.nextBd ^bb1
  ^bb4: 
    AIE.end
  }
  AIE.flow(%t73, DMA : 0, %t70, PLIO : 1)
  AIE.flow(%t74, DMA : 0, %t70, PLIO : 0)
}
