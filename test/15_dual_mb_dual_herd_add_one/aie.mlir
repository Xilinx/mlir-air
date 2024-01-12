//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  %t70 = aie.tile(7, 0)
  %t72 = aie.tile(7, 2)
  %t340 = aie.tile(34, 0)
  %t342 = aie.tile(34, 2)

  aie.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)
  aie.flow(%t70, "DMA" : 1, %t72, "DMA" : 1)
  aie.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)
  aie.flow(%t72, "DMA" : 1, %t70, "DMA" : 1)

  %buf72_0 = aie.buffer(%t72) { sym_name = "ping_in" } : memref<8xi32>
  %buf72_1 = aie.buffer(%t72) { sym_name = "ping_out" } : memref<8xi32>
  %buf72_2 = aie.buffer(%t72) { sym_name = "pong_in" } : memref<8xi32>
  %buf72_3 = aie.buffer(%t72) { sym_name = "pong_out" } : memref<8xi32>

  %l72_0 = aie.lock(%t72, 0)
  %l72_1 = aie.lock(%t72, 1)
  %l72_2 = aie.lock(%t72, 2)
  %l72_3 = aie.lock(%t72, 3)

  %m72 = aie.mem(%t72) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%l72_0, "Acquire", 0)
      aie.dma_bd(%buf72_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%l72_0, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l72_1, "Acquire", 0)
      aie.dma_bd(%buf72_2 : memref<8xi32>, 0, 8)
      aie.use_lock(%l72_1, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%l72_2, "Acquire", 1)
      aie.dma_bd(%buf72_1 : memref<8xi32>, 0, 8)
      aie.use_lock(%l72_2, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%l72_3, "Acquire", 1)
      aie.dma_bd(%buf72_3 : memref<8xi32>, 0, 8)
      aie.use_lock(%l72_3, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  aie.core(%t72) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_32 = arith.constant 1 : i32

    aie.use_lock(%l72_0, "Acquire", 1)
    aie.use_lock(%l72_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf72_0[%arg3] : memref<8xi32>
        %1 = arith.addi %0, %c1_32 : i32
        memref.store %1, %buf72_1[%arg3] : memref<8xi32>
    }
    aie.use_lock(%l72_0, "Release", 0)
    aie.use_lock(%l72_2, "Release", 1)

    aie.use_lock(%l72_1, "Acquire", 1)
    aie.use_lock(%l72_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf72_2[%arg4] : memref<8xi32>
        %3 = arith.addi %2, %c1_32 : i32
        memref.store %3, %buf72_3[%arg4] : memref<8xi32>
    }
    aie.use_lock(%l72_1, "Release", 0)
    aie.use_lock(%l72_3, "Release", 1)
    aie.end

  }

  aie.flow(%t340, "DMA" : 0, %t342, "DMA" : 0)
  aie.flow(%t340, "DMA" : 1, %t342, "DMA" : 1)
  aie.flow(%t342, "DMA" : 0, %t340, "DMA" : 0)
  aie.flow(%t342, "DMA" : 1, %t340, "DMA" : 1)

  %buf342_0 = aie.buffer(%t342) { sym_name = "ping_in2" } : memref<8xi32>
  %buf342_1 = aie.buffer(%t342) { sym_name = "ping_out2" } : memref<8xi32>
  %buf342_2 = aie.buffer(%t342) { sym_name = "pong_in2" } : memref<8xi32>
  %buf342_3 = aie.buffer(%t342) { sym_name = "pong_out2" } : memref<8xi32>

  %l342_0 = aie.lock(%t342, 0)
  %l342_1 = aie.lock(%t342, 1)
  %l342_2 = aie.lock(%t342, 2)
  %l342_3 = aie.lock(%t342, 3)

  %m342 = aie.mem(%t342) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%l342_0, "Acquire", 0)
      aie.dma_bd(%buf342_0 : memref<8xi32>, 0, 8)
      aie.use_lock(%l342_0, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l342_1, "Acquire", 0)
      aie.dma_bd(%buf342_2 : memref<8xi32>, 0, 8)
      aie.use_lock(%l342_1, "Release", 1)
      aie.next_bd ^bd0
    ^bd2:
      aie.use_lock(%l342_2, "Acquire", 1)
      aie.dma_bd(%buf342_1 : memref<8xi32>, 0, 8)
      aie.use_lock(%l342_2, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%l342_3, "Acquire", 1)
      aie.dma_bd(%buf342_3 : memref<8xi32>, 0, 8)
      aie.use_lock(%l342_3, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  aie.core(%t342) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_32 = arith.constant 1 : i32

    aie.use_lock(%l342_0, "Acquire", 1)
    aie.use_lock(%l342_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf342_0[%arg3] : memref<8xi32>
        %1 = arith.addi %0, %c1_32 : i32
        memref.store %1, %buf342_1[%arg3] : memref<8xi32>
    }
    aie.use_lock(%l342_0, "Release", 0)
    aie.use_lock(%l342_2, "Release", 1)

    aie.use_lock(%l342_1, "Acquire", 1)
    aie.use_lock(%l342_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf342_2[%arg4] : memref<8xi32>
        %3 = arith.addi %2, %c1_32 : i32
        memref.store %3, %buf342_3[%arg4] : memref<8xi32>
    }
    aie.use_lock(%l342_1, "Release", 0)
    aie.use_lock(%l342_3, "Release", 1)
    aie.end

  }

}
