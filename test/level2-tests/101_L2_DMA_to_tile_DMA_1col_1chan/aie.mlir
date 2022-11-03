//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module @aie.0  {
  %0 = AIE.tile(7, 1)
  %1 = AIE.tile(7, 0)
  %2 = AIE.tile(0, 0)
  %3 = AIE.tile(7, 4)
  %4 = AIE.lock(%3, 2)
  %5 = AIE.buffer(%3) {sym_name = "buf2"} : memref<16xi32, 2>
  %6 = AIE.lock(%3, 1)
  %7 = AIE.buffer(%3) {sym_name = "buf1"} : memref<16xi32, 2>
  %10 = AIE.mem(%3)  {
    %14 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb3
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%6, Acquire, 0)
    AIE.dmaBd(<%7 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%6, Release, 1)
    cf.br ^bb3
  ^bb3:  // pred: ^bb2
    AIE.useLock(%4, Acquire, 0)
    AIE.dmaBd(<%5 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%4, Release, 1)
    cf.br ^bb1
  ^bb4:  // pred: ^bb0
    AIE.end
  }
  %11 = AIE.core(%3)  {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%6, Acquire, 1)
    AIE.useLock(%4, Acquire, 1)
    AIE.end
  }
  AIE.flow(%1, PLIO : 0, %3, DMA : 0)
}
