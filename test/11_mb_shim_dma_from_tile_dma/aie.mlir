//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  %t70 = aie.tile(6, 0)
  %t71 = aie.tile(6, 1)
  %t72 = aie.tile(6, 2)

  %buf72_0 = aie.buffer(%t72) {sym_name="a"} : memref<256xi32>
  %buf72_1 = aie.buffer(%t72) {sym_name="b"} : memref<256xi32>

  %l72_0 = aie.lock(%t72, 0)
  %l72_1 = aie.lock(%t72, 1)

  %m72 = aie.mem(%t72) {
      %srcDma = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l72_0, "Acquire", 1)
      aie.dma_bd(%buf72_0 : memref<256xi32>, 0, 256)
      aie.use_lock(%l72_0, "Release", 0)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l72_1, "Acquire", 1)
      aie.dma_bd(%buf72_1 : memref<256xi32>, 0, 256)
      aie.use_lock(%l72_1, "Release", 0)
      aie.next_bd ^bd0
    ^end:
      aie.end
  }
  aie.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)

  aie.core(%t72) {
    aie.end

  }

}
