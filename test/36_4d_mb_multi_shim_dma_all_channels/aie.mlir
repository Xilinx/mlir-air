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
  %t60 = AIE.tile(6, 0)
  %t70 = AIE.tile(7, 0)
  %t72 = AIE.tile(7, 2)
  %t74 = AIE.tile(7, 4)
  %t82 = AIE.tile(8, 2)
  %t84 = AIE.tile(8, 4)

  AIE.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)
  AIE.flow(%t70, "DMA" : 1, %t74, "DMA" : 0)
  AIE.flow(%t74, "DMA" : 0, %t70, "DMA" : 1)
  AIE.flow(%t60, "DMA" : 0, %t82, "DMA" : 0)
  AIE.flow(%t82, "DMA" : 0, %t60, "DMA" : 0)
  AIE.flow(%t60, "DMA" : 1, %t84, "DMA" : 0)
  AIE.flow(%t84, "DMA" : 0, %t60, "DMA" : 1)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0"} : memref<128xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name = "buf72_1"} : memref<128xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0)
      AIE.dmaBd(<%buf72_0 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l72_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0)
      AIE.dmaBd(<%buf72_1 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l72_1, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%l72_0, "Acquire", 1)
      AIE.dmaBd(<%buf72_0 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l72_0, "Release", 0)
      cf.br ^bd3
    ^bd3:
      AIE.useLock(%l72_1, "Acquire", 1)
      AIE.dmaBd(<%buf72_1 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l72_1, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

  %buf74_0 = AIE.buffer(%t74) {sym_name = "buf74_0"} : memref<128xi32>
  %buf74_1 = AIE.buffer(%t74) {sym_name = "buf74_1"} : memref<128xi32>

  %l74_0 = AIE.lock(%t74, 0)
  %l74_1 = AIE.lock(%t74, 1)

  %m74 = AIE.mem(%t74) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l74_0, "Acquire", 0)
      AIE.dmaBd(<%buf74_0 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l74_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l74_1, "Acquire", 0)
      AIE.dmaBd(<%buf74_1 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l74_1, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%l74_0, "Acquire", 1)
      AIE.dmaBd(<%buf74_0 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l74_0, "Release", 0)
      cf.br ^bd3
    ^bd3:
      AIE.useLock(%l74_1, "Acquire", 1)
      AIE.dmaBd(<%buf74_1 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l74_1, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

  %buf82_0 = AIE.buffer(%t82) {sym_name = "buf82_0"} : memref<128xi32>
  %buf82_1 = AIE.buffer(%t82) {sym_name = "buf82_1"} : memref<128xi32>

  %l82_0 = AIE.lock(%t82, 0)
  %l82_1 = AIE.lock(%t82, 1)

  %m82 = AIE.mem(%t82) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l82_0, "Acquire", 0)
      AIE.dmaBd(<%buf82_0 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l82_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l82_1, "Acquire", 0)
      AIE.dmaBd(<%buf82_1 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l82_1, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%l82_0, "Acquire", 1)
      AIE.dmaBd(<%buf82_0 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l82_0, "Release", 0)
      cf.br ^bd3
    ^bd3:
      AIE.useLock(%l82_1, "Acquire", 1)
      AIE.dmaBd(<%buf82_1 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l82_1, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

  %buf84_0 = AIE.buffer(%t84) {sym_name = "buf84_0"} : memref<128xi32>
  %buf84_1 = AIE.buffer(%t84) {sym_name = "buf84_1"} : memref<128xi32>

  %l84_0 = AIE.lock(%t84, 0)
  %l84_1 = AIE.lock(%t84, 1)

  %m84 = AIE.mem(%t84) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l84_0, "Acquire", 0)
      AIE.dmaBd(<%buf84_0 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l84_0, "Release", 1)
      cf.br ^bd1
    ^bd1:
      AIE.useLock(%l84_1, "Acquire", 0)
      AIE.dmaBd(<%buf84_1 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l84_1, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%l84_0, "Acquire", 1)
      AIE.dmaBd(<%buf84_0 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l84_0, "Release", 0)
      cf.br ^bd3
    ^bd3:
      AIE.useLock(%l84_1, "Acquire", 1)
      AIE.dmaBd(<%buf84_1 : memref<128xi32>, 0, 128>, 0)
      AIE.useLock(%l84_1, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

}
