//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  %t70 = AIE.tile(0x7, 0)
  %t71 = AIE.tile(0x7, 1)
  %t72 = AIE.tile(0x7, 2)

  %t80 = AIE.tile(0x8, 0)
  %t81 = AIE.tile(0x8, 1)
  %t82 = AIE.tile(0x8, 2)

  %t90 = AIE.tile(0x9, 0)
  %t91 = AIE.tile(0x9, 1)
  %t92 = AIE.tile(0x9, 2)

  %ta0 = AIE.tile(0xa, 0)
  %ta1 = AIE.tile(0xa, 1)
  %ta2 = AIE.tile(0xa, 2)

  AIE.flow(%t70, PLIO : 0, %t72, DMA : 0)
  AIE.flow(%t80, PLIO : 0, %t82, DMA : 0)
  AIE.flow(%t90, PLIO : 0, %t92, DMA : 0)
  AIE.flow(%ta0, PLIO : 0, %ta2, DMA : 0)

  %buf72_0 = AIE.buffer(%t72) { sym_name = "a" } : memref<16xi32, 2>
  %buf72_1 = AIE.buffer(%t72) { sym_name = "b" } : memref<16xi32, 2>
  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)

  %buf82_0 = AIE.buffer(%t82) { sym_name = "c" } : memref<16xi32, 2>
  %buf82_1 = AIE.buffer(%t82) { sym_name = "d" } : memref<16xi32, 2>
  %l82_0 = AIE.lock(%t82, 0)
  %l82_1 = AIE.lock(%t82, 1)

  %buf92_0 = AIE.buffer(%t92) { sym_name = "e" } : memref<16xi32, 2>
  %buf92_1 = AIE.buffer(%t92) { sym_name = "f" } : memref<16xi32, 2>
  %l92_0 = AIE.lock(%t92, 0)
  %l92_1 = AIE.lock(%t92, 1)

  %bufa2_0 = AIE.buffer(%ta2) { sym_name = "g" } : memref<16xi32, 2>
  %bufa2_1 = AIE.buffer(%ta2) { sym_name = "i" } : memref<16xi32, 2>
  %la2_0 = AIE.lock(%ta2, 0)
  %la2_1 = AIE.lock(%ta2, 1)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd2, ^end)
    ^bd2:
      cf.br ^bd0
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0)
      AIE.dmaBd(<%buf72_0 : memref<16xi32,2>, 0, 16>, 0)
      AIE.useLock(%l72_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0)
      AIE.dmaBd(<%buf72_1 : memref<16xi32,2>, 0, 16>, 0)
      AIE.useLock(%l72_1, "Release", 1)
      cf.br ^bd0
    ^end:
      AIE.end
  }

  %m82 = AIE.mem(%t82) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd2, ^end)
    ^bd2:
      cf.br ^bd0
    ^bd0:
      AIE.useLock(%l82_0, "Acquire", 0)
      AIE.dmaBd(<%buf82_0 : memref<16xi32,2>, 0, 16>, 0)
      AIE.useLock(%l82_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l82_1, "Acquire", 0)
      AIE.dmaBd(<%buf82_1 : memref<16xi32,2>, 0, 16>, 0)
      AIE.useLock(%l82_1, "Release", 1)
      cf.br ^bd0
    ^end:
      AIE.end
  }

  %m92 = AIE.mem(%t92) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd2, ^end)
    ^bd2:
      cf.br ^bd0
    ^bd0:
      AIE.useLock(%l92_0, "Acquire", 0)
      AIE.dmaBd(<%buf92_0 : memref<16xi32,2>, 0, 16>, 0)
      AIE.useLock(%l92_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l92_1, "Acquire", 0)
      AIE.dmaBd(<%buf92_1 : memref<16xi32,2>, 0, 16>, 0)
      AIE.useLock(%l92_1, "Release", 1)
      cf.br ^bd0
    ^end:
      AIE.end
  }

  %ma2 = AIE.mem(%ta2) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd2, ^end)
    ^bd2:
      cf.br ^bd0
    ^bd0:
      AIE.useLock(%la2_0, "Acquire", 0)
      AIE.dmaBd(<%bufa2_0 : memref<16xi32,2>, 0, 16>, 0)
      AIE.useLock(%la2_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%la2_1, "Acquire", 0)
      AIE.dmaBd(<%bufa2_1 : memref<16xi32,2>, 0, 16>, 0)
      AIE.useLock(%la2_1, "Release", 1)
      cf.br ^bd0
    ^end:
      AIE.end
  }


}
