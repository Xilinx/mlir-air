//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module @aie.0  {
  %t70 = AIE.tile(7, 0)
  %t80 = AIE.tile(8, 0)
  %t90 = AIE.tile(9, 0)
  %ta0 = AIE.tile(10, 0)
  %t74 = AIE.tile(7, 4)
  %t84 = AIE.tile(8, 4)
  %t94 = AIE.tile(9, 4)
  %ta4 = AIE.tile(10, 4)

  %6 = AIE.lock(%t74, 1)
  %7 = AIE.buffer(%t74) {sym_name = "buf1"} : memref<16xi32, 2>
  %10 = AIE.lock(%t84, 1)
  %11 = AIE.buffer(%t84) {sym_name = "buf2"} : memref<16xi32, 2>
  %14 = AIE.lock(%t94, 1)
  %15 = AIE.buffer(%t94) {sym_name = "buf3"} : memref<16xi32, 2>
  %18 = AIE.lock(%ta4, 1)
  %19 = AIE.buffer(%ta4) {sym_name = "buf4"} : memref<16xi32, 2>

  %8 = AIE.mem(%t74)  {
    %9 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb4)
  ^bb1: 
    cf.br ^bb2
  ^bb2: 
    AIE.useLock(%6, Acquire, 1)
    AIE.dmaBd(<%7 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%6, Release, 0)
    cf.br ^bb3
  ^bb3: 
    cf.br ^bb1
  ^bb4: 
    AIE.end
  }
  %12 = AIE.mem(%t84)  {
    %13 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb4)
  ^bb1: 
    cf.br ^bb2
  ^bb2: 
    AIE.useLock(%10, Acquire, 1)
    AIE.dmaBd(<%11 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%10, Release, 0)
    cf.br ^bb3
  ^bb3: 
    cf.br ^bb1
  ^bb4: 
    AIE.end
  }
  %16 = AIE.mem(%t94)  {
    %17 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb4)
  ^bb1: 
    cf.br ^bb2
  ^bb2: 
    AIE.useLock(%14, Acquire, 1)
    AIE.dmaBd(<%15 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%14, Release, 0)
    cf.br ^bb3
  ^bb3: 
    cf.br ^bb1
  ^bb4: 
    AIE.end
  }
  %20 = AIE.mem(%ta4)  {
    %21 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb4)
  ^bb1: 
    cf.br ^bb2
  ^bb2: 
    AIE.useLock(%18, Acquire, 1)
    AIE.dmaBd(<%19 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%18, Release, 0)
    cf.br ^bb3
  ^bb3: 
    cf.br ^bb1
  ^bb4: 
    AIE.end
  }
  AIE.flow(%t74, DMA : 0, %t70, PLIO : 0)
  AIE.flow(%t84, DMA : 0, %t80, PLIO : 0)
  AIE.flow(%t94, DMA : 0, %t90, PLIO : 0)
  AIE.flow(%ta4, DMA : 0, %ta0, PLIO : 0)
}
