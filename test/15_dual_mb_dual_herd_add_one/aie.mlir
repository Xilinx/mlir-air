//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

module {
  %t70 = AIE.tile(7, 0)
  %t72 = AIE.tile(7, 2)
  %t340 = AIE.tile(34, 0)
  %t342 = AIE.tile(34, 2)

  AIE.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)
  AIE.flow(%t70, "DMA" : 1, %t72, "DMA" : 1)
  AIE.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 1, %t70, "DMA" : 1)

  %buf72_0 = AIE.buffer(%t72) { sym_name = "ping_in" } : memref<8xi32>
  %buf72_1 = AIE.buffer(%t72) { sym_name = "ping_out" } : memref<8xi32>
  %buf72_2 = AIE.buffer(%t72) { sym_name = "pong_in" } : memref<8xi32>
  %buf72_3 = AIE.buffer(%t72) { sym_name = "pong_out" } : memref<8xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)
  %l72_2 = AIE.lock(%t72, 2)
  %l72_3 = AIE.lock(%t72, 3)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart(S2MM, 0 ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart(MM2S, 0 ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0)
      AIE.dmaBd(<%buf72_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0)
      AIE.dmaBd(<%buf72_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_1, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%l72_2, "Acquire", 1)
      AIE.dmaBd(<%buf72_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_2, "Release", 0)
      cf.br ^bd3
    ^bd3:
      AIE.useLock(%l72_3, "Acquire", 1)
      AIE.dmaBd(<%buf72_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_3, "Release", 0)
      cf.br ^bd2
    ^end:
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

  AIE.flow(%t340, "DMA" : 0, %t342, "DMA" : 0)
  AIE.flow(%t340, "DMA" : 1, %t342, "DMA" : 1)
  AIE.flow(%t342, "DMA" : 0, %t340, "DMA" : 0)
  AIE.flow(%t342, "DMA" : 1, %t340, "DMA" : 1)

  %buf342_0 = AIE.buffer(%t342) { sym_name = "ping_in2" } : memref<8xi32>
  %buf342_1 = AIE.buffer(%t342) { sym_name = "ping_out2" } : memref<8xi32>
  %buf342_2 = AIE.buffer(%t342) { sym_name = "pong_in2" } : memref<8xi32>
  %buf342_3 = AIE.buffer(%t342) { sym_name = "pong_out2" } : memref<8xi32>

  %l342_0 = AIE.lock(%t342, 0)
  %l342_1 = AIE.lock(%t342, 1)
  %l342_2 = AIE.lock(%t342, 2)
  %l342_3 = AIE.lock(%t342, 3)

  %m342 = AIE.mem(%t342) {
      %srcDma = AIE.dmaStart(S2MM, 0 ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart(MM2S, 0 ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l342_0, "Acquire", 0)
      AIE.dmaBd(<%buf342_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l342_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l342_1, "Acquire", 0)
      AIE.dmaBd(<%buf342_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l342_1, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%l342_2, "Acquire", 1)
      AIE.dmaBd(<%buf342_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l342_2, "Release", 0)
      cf.br ^bd3
    ^bd3:
      AIE.useLock(%l342_3, "Acquire", 1)
      AIE.dmaBd(<%buf342_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l342_3, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

  AIE.core(%t342) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_32 = arith.constant 1 : i32

    AIE.useLock(%l342_0, "Acquire", 1)
    AIE.useLock(%l342_2, "Acquire", 0)
    scf.for %arg3 = %c0 to %c8 step %c1 {
        %0 = memref.load %buf342_0[%arg3] : memref<8xi32>
        %1 = arith.addi %0, %c1_32 : i32
        memref.store %1, %buf342_1[%arg3] : memref<8xi32>
    }
    AIE.useLock(%l342_0, "Release", 0)
    AIE.useLock(%l342_2, "Release", 1)

    AIE.useLock(%l342_1, "Acquire", 1)
    AIE.useLock(%l342_3, "Acquire", 0)
    scf.for %arg4 = %c0 to %c8 step %c1 {
        %2 = memref.load %buf342_2[%arg4] : memref<8xi32>
        %3 = arith.addi %2, %c1_32 : i32
        memref.store %3, %buf342_3[%arg4] : memref<8xi32>
    }
    AIE.useLock(%l342_1, "Release", 0)
    AIE.useLock(%l342_3, "Release", 1)
    AIE.end

  }

}
