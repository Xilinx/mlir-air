//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)
  %t72 = aie.tile(7, 2)

  aie.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)
  aie.flow(%t70, "DMA" : 1, %t72, "DMA" : 1)
  aie.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)
  aie.flow(%t72, "DMA" : 1, %t70, "DMA" : 1)

  %buf72_0 = aie.buffer(%t72) { sym_name = "ping_a" } : memref<1024xi32>
  %buf72_4 = aie.buffer(%t72) { sym_name = "ping_b" } : memref<1024xi32>
  %buf72_1 = aie.buffer(%t72) { sym_name = "ping_c" } : memref<1024xi32>
  %buf72_2 = aie.buffer(%t72) { sym_name = "pong_a" } : memref<1024xi32>
  %buf72_5 = aie.buffer(%t72) { sym_name = "pong_b" } : memref<1024xi32>
  %buf72_3 = aie.buffer(%t72) { sym_name = "pong_c" } : memref<1024xi32>

  %l72_0 = aie.lock(%t72, 0)
  %l72_1 = aie.lock(%t72, 1)
  %l72_2 = aie.lock(%t72, 2)
  %l72_3 = aie.lock(%t72, 3)
  %l72_4 = aie.lock(%t72, 4)
  %l72_5 = aie.lock(%t72, 5)

  %m72 = aie.mem(%t72) {
      %srcDma1 = aie.dma_start(S2MM, 0, ^bd0, ^src1)
    ^src1:
      %srcDma2 = aie.dma_start(S2MM, 1, ^bd4, ^dma0)
    ^dma0:
      %dstDma = aie.dma_start(MM2S, 0, ^bd2, ^end)
    ^bd0:
      aie.use_lock(%l72_0, "Acquire", 0)
      aie.dma_bd(%buf72_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%l72_0, "Release", 1)
      aie.next_bd ^bd1
    ^bd1:
      aie.use_lock(%l72_1, "Acquire", 0)
      aie.dma_bd(%buf72_2 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%l72_1, "Release", 1)
      aie.next_bd ^bd0
    ^bd4:
      aie.use_lock(%l72_4, "Acquire", 0)
      aie.dma_bd(%buf72_4 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%l72_4, "Release", 1)
      aie.next_bd ^bd5
    ^bd5:
      aie.use_lock(%l72_5, "Acquire", 0)
      aie.dma_bd(%buf72_5 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%l72_5, "Release", 1)
      aie.next_bd ^bd4
    ^bd2:
      aie.use_lock(%l72_2, "Acquire", 1)
      aie.dma_bd(%buf72_1 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%l72_2, "Release", 0)
      aie.next_bd ^bd3
    ^bd3:
      aie.use_lock(%l72_3, "Acquire", 1)
      aie.dma_bd(%buf72_3 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%l72_3, "Release", 0)
      aie.next_bd ^bd2
    ^end:
      aie.end
  }

  aie.core(%t72) {
    %clp = arith.constant 36 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.for %arg5 = %c0 to %clp step %c1 {  
      aie.use_lock(%l72_0, "Acquire", 1)
      aie.use_lock(%l72_4, "Acquire", 1)
      aie.use_lock(%l72_2, "Acquire", 0)
      scf.for %arg3 = %c0 to %c1024 step %c1 {
          %0 = memref.load %buf72_0[%arg3] : memref<1024xi32>
          %1 = memref.load %buf72_4[%arg3] : memref<1024xi32>
          %2 = arith.addi %0, %1 : i32
          memref.store %2, %buf72_1[%arg3] : memref<1024xi32>
      }
      aie.use_lock(%l72_0, "Release", 0)
      aie.use_lock(%l72_4, "Release", 0)
      aie.use_lock(%l72_2, "Release", 1)

      aie.use_lock(%l72_1, "Acquire", 1)
      aie.use_lock(%l72_5, "Acquire", 1)
      aie.use_lock(%l72_3, "Acquire", 0)
      scf.for %arg4 = %c0 to %c1024 step %c1 {
          %3 = memref.load %buf72_2[%arg4] : memref<1024xi32>
          %4 = memref.load %buf72_5[%arg4] : memref<1024xi32>
          %5 = arith.addi %3, %4 : i32
          memref.store %5, %buf72_3[%arg4] : memref<1024xi32>
      }
      aie.use_lock(%l72_1, "Release", 0)
      aie.use_lock(%l72_5, "Release", 0)
      aie.use_lock(%l72_3, "Release", 1)
    }
    aie.end

  }

}
