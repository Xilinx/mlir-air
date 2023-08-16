//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  %t60 = AIE.tile(6, 0)
  %t61 = AIE.tile(6, 1)
  %t62 = AIE.tile(6, 2)

  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)

  AIE.flow(%t60, "DMA" : 0, %t62, "DMA" : 0)
  AIE.flow(%t60, "DMA" : 1, %t62, "DMA" : 1)
  AIE.flow(%t62, "DMA" : 0, %t60, "DMA" : 0)
  AIE.flow(%t62, "DMA" : 1, %t60, "DMA" : 1)

  AIE.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)
  AIE.flow(%t70, "DMA" : 1, %t72, "DMA" : 1)
  AIE.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 1, %t70, "DMA" : 1)

  %buf62_0 = AIE.buffer(%t62) { sym_name = "ping_in" } : memref<8xi32>
  %buf62_1 = AIE.buffer(%t62) { sym_name = "ping_out" } : memref<8xi32>
  %buf62_2 = AIE.buffer(%t62) { sym_name = "pong_in" } : memref<8xi32>
  %buf62_3 = AIE.buffer(%t62) { sym_name = "pong_out" } : memref<8xi32>

  %buf72_0 = AIE.buffer(%t72) { sym_name = "ping_in_two" } : memref<8xi32>
  %buf72_1 = AIE.buffer(%t72) { sym_name = "ping_out_two" } : memref<8xi32>
  %buf72_2 = AIE.buffer(%t72) { sym_name = "pong_in_two" } : memref<8xi32>
  %buf72_3 = AIE.buffer(%t72) { sym_name = "pong_out_two" } : memref<8xi32>

  %l62_0 = AIE.lock(%t62, 0)
  %l62_1 = AIE.lock(%t62, 1)
  %l62_2 = AIE.lock(%t62, 2)
  %l62_3 = AIE.lock(%t62, 3)

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)
  %l72_2 = AIE.lock(%t72, 2)
  %l72_3 = AIE.lock(%t72, 3)

  %m62 = AIE.mem(%t62) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart(MM2S, 0, ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l62_0, "Acquire", 0)
      AIE.dmaBd(<%buf62_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l62_0, "Release", 1)
      AIE.nextBd ^bd1
    ^bd1:
      AIE.useLock(%l62_1, "Acquire", 0)
      AIE.dmaBd(<%buf62_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l62_1, "Release", 1)
      AIE.nextBd ^bd0
    ^bd2:
      AIE.useLock(%l62_2, "Acquire", 1)
      AIE.dmaBd(<%buf62_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l62_2, "Release", 0)
      AIE.nextBd ^bd3
    ^bd3:
      AIE.useLock(%l62_3, "Acquire", 1)
      AIE.dmaBd(<%buf62_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l62_3, "Release", 0)
      AIE.nextBd ^bd2
    ^end:
      AIE.end
  }

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart(S2MM, 0, ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart(MM2S, 0, ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0)
      AIE.dmaBd(<%buf72_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_0, "Release", 1)
      AIE.nextBd ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0)
      AIE.dmaBd(<%buf72_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_1, "Release", 1)
      AIE.nextBd ^bd0
    ^bd2:
      AIE.useLock(%l72_2, "Acquire", 1)
      AIE.dmaBd(<%buf72_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_2, "Release", 0)
      AIE.nextBd ^bd3
    ^bd3:
      AIE.useLock(%l72_3, "Acquire", 1)
      AIE.dmaBd(<%buf72_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_3, "Release", 0)
      AIE.nextBd ^bd2
    ^end:
      AIE.end
  }

  AIE.core(%t62) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_32 = arith.constant 1 : i32

    AIE.useLock(%l62_0, "Acquire", 1)
    AIE.useLock(%l62_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf62_0[%arg3] : memref<8xi32>
        %1 = arith.addi %0, %c1_32 : i32
        memref.store %1, %buf62_1[%arg3] : memref<8xi32>
    }
    AIE.useLock(%l62_0, "Release", 0)
    AIE.useLock(%l62_2, "Release", 1)

    AIE.useLock(%l62_1, "Acquire", 1)
    AIE.useLock(%l62_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf62_2[%arg4] : memref<8xi32>
        %3 = arith.addi %2, %c1_32 : i32
        memref.store %3, %buf62_3[%arg4] : memref<8xi32>
    }
    AIE.useLock(%l62_1, "Release", 0)
    AIE.useLock(%l62_3, "Release", 1)
    AIE.end

  }

  AIE.core(%t72) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_32 = arith.constant 1 : i32

    AIE.useLock(%l72_0, "Acquire", 1)
    AIE.useLock(%l72_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf72_0[%arg3] : memref<8xi32>
        %1 = arith.addi %0, %c1_32 : i32
        memref.store %1, %buf72_1[%arg3] : memref<8xi32>
    }
    AIE.useLock(%l72_0, "Release", 0)
    AIE.useLock(%l72_2, "Release", 1)

    AIE.useLock(%l72_1, "Acquire", 1)
    AIE.useLock(%l72_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf72_2[%arg4] : memref<8xi32>
        %3 = arith.addi %2, %c1_32 : i32
        memref.store %3, %buf72_3[%arg4] : memref<8xi32>
    }
    AIE.useLock(%l72_1, "Release", 0)
    AIE.useLock(%l72_3, "Release", 1)
    AIE.end

  }

}
