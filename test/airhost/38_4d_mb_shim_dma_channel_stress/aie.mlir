//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  %t10 = aie.tile(10, 0)
  %t11 = aie.tile(11, 0)
  %t18 = aie.tile(18, 0)
  %t19 = aie.tile(19, 0)
  %t26 = aie.tile(26, 0)
  %t30 = aie.tile(3, 0)
  %t60 = aie.tile(6, 0)
  %t70 = aie.tile(7, 0)
  %t72 = aie.tile(7, 2)
  %t74 = aie.tile(7, 4)
  %t82 = aie.tile(8, 2)
  %t84 = aie.tile(8, 4)

  aie.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)
  aie.flow(%t72, "DMA" : 0, %t30, "DMA" : 0)
  aie.flow(%t11, "DMA" : 1, %t74, "DMA" : 0)
  aie.flow(%t74, "DMA" : 0, %t19, "DMA" : 1)
  aie.flow(%t60, "DMA" : 0, %t82, "DMA" : 0)
  aie.flow(%t82, "DMA" : 0, %t26, "DMA" : 0)
  aie.flow(%t10, "DMA" : 1, %t84, "DMA" : 0)
  aie.flow(%t84, "DMA" : 0, %t18, "DMA" : 1)

  %buf72_0 = aie.buffer(%t72) {sym_name = "buf72_0"} : memref<128xi32>
  %buf72_1 = aie.buffer(%t72) {sym_name = "buf72_1"} : memref<128xi32>

  %l72_0 = aie.lock(%t72, 0)
  %l72_1 = aie.lock(%t72, 1)

  %m72 = aie.mem(%t72) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%l72_0, "Acquire", 0)
      aie.dma_bd(%buf72_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%l72_0, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l72_1, "Acquire", 0)
      aie.dma_bd(%buf72_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%l72_1, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%l72_0, "Acquire", 1)
      aie.dma_bd(%buf72_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%l72_0, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%l72_1, "Acquire", 1)
      aie.dma_bd(%buf72_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%l72_1, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  %buf74_0 = aie.buffer(%t74) {sym_name = "buf74_0"} : memref<128xi32>
  %buf74_1 = aie.buffer(%t74) {sym_name = "buf74_1"} : memref<128xi32>

  %l74_0 = aie.lock(%t74, 0)
  %l74_1 = aie.lock(%t74, 1)

  %m74 = aie.mem(%t74) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%l74_0, "Acquire", 0)
      aie.dma_bd(%buf74_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%l74_0, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l74_1, "Acquire", 0)
      aie.dma_bd(%buf74_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%l74_1, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%l74_0, "Acquire", 1)
      aie.dma_bd(%buf74_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%l74_0, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%l74_1, "Acquire", 1)
      aie.dma_bd(%buf74_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%l74_1, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  %buf82_0 = aie.buffer(%t82) {sym_name = "buf82_0"} : memref<128xi32>
  %buf82_1 = aie.buffer(%t82) {sym_name = "buf82_1"} : memref<128xi32>

  %l82_0 = aie.lock(%t82, 0)
  %l82_1 = aie.lock(%t82, 1)

  %m82 = aie.mem(%t82) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%l82_0, "Acquire", 0)
      aie.dma_bd(%buf82_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%l82_0, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l82_1, "Acquire", 0)
      aie.dma_bd(%buf82_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%l82_1, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%l82_0, "Acquire", 1)
      aie.dma_bd(%buf82_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%l82_0, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%l82_1, "Acquire", 1)
      aie.dma_bd(%buf82_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%l82_1, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  %buf84_0 = aie.buffer(%t84) {sym_name = "buf84_0"} : memref<128xi32>
  %buf84_1 = aie.buffer(%t84) {sym_name = "buf84_1"} : memref<128xi32>

  %l84_0 = aie.lock(%t84, 0)
  %l84_1 = aie.lock(%t84, 1)

  %m84 = aie.mem(%t84) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%l84_0, "Acquire", 0)
      aie.dma_bd(%buf84_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%l84_0, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l84_1, "Acquire", 0)
      aie.dma_bd(%buf84_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%l84_1, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%l84_0, "Acquire", 1)
      aie.dma_bd(%buf84_0 : memref<128xi32>, 0, 128)
      aie.use_lock(%l84_0, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%l84_1, "Acquire", 1)
      aie.dma_bd(%buf84_1 : memref<128xi32>, 0, 128)
      aie.use_lock(%l84_1, "Release", 0)
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

  aie.core(%t82) {
    aie.end

  }

  aie.core(%t84) {
    aie.end

  }

}
