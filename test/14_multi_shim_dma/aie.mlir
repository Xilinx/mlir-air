//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  %t30 = aie.tile(18, 0)
  %tb0 = aie.tile(11, 0)
  %t72 = aie.tile(7, 2)
  %t74 = aie.tile(7, 4)

  aie.flow(%t30, "DMA" : 0, %t72, "DMA" : 0)
  aie.flow(%t72, "DMA" : 0, %t30, "DMA" : 0)
  aie.flow(%tb0, "DMA" : 0, %t74, "DMA" : 0)
  aie.flow(%t74, "DMA" : 0, %tb0, "DMA" : 0)

  %buf72_0 = aie.buffer(%t72) {sym_name = "buf72_0"} : memref<32xi32>
  %buf74_0 = aie.buffer(%t74) {sym_name = "buf74_0"}: memref<32xi32>

  %l72_0 = aie.lock(%t72, 0)
  %l74_0 = aie.lock(%t74, 0)

  %m72 = aie.mem(%t72) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%l72_0, "Acquire", 0)
      aie.dma_bd(%buf72_0 : memref<32xi32>, 0, 32)
      aie.use_lock(%l72_0, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%l72_0, "Acquire", 1)
      aie.dma_bd(%buf72_0 : memref<32xi32>, 0, 32)
      aie.use_lock(%l72_0, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  %m74 = aie.mem(%t74) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%l74_0, "Acquire", 0)
      aie.dma_bd(%buf74_0 : memref<32xi32>, 0, 32)
      aie.use_lock(%l74_0, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%l74_0, "Acquire", 1)
      aie.dma_bd(%buf74_0 : memref<32xi32>, 0, 32)
      aie.use_lock(%l74_0, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  aie.core(%t72) {
    aie.end
  }

  aie.core(%t74) {
    aie.end
  }

}
