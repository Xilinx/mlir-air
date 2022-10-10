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
  %t30 = AIE.tile(18, 0)
  %tb0 = AIE.tile(11, 0)
  %t72 = AIE.tile(7, 2)
  %t74 = AIE.tile(7, 4)

  AIE.flow(%t30, "DMA" : 0, %t72, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 0, %t30, "DMA" : 0)
  AIE.flow(%tb0, "DMA" : 0, %t74, "DMA" : 0)
  AIE.flow(%t74, "DMA" : 0, %tb0, "DMA" : 0)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0"} : memref<32xi32>
  %buf74_0 = AIE.buffer(%t74) {sym_name = "buf74_0"}: memref<32xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l74_0 = AIE.lock(%t74, 0)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart(S2MM, 0 ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart(MM2S, 0 ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0)
      AIE.dmaBd(<%buf72_0 : memref<32xi32>, 0, 32>, 0)
      AIE.useLock(%l72_0, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%l72_0, "Acquire", 1)
      AIE.dmaBd(<%buf72_0 : memref<32xi32>, 0, 32>, 0)
      AIE.useLock(%l72_0, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

  %m74 = AIE.mem(%t74) {
      %srcDma = AIE.dmaStart(S2MM, 0 ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart(MM2S, 0 ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l74_0, "Acquire", 0)
      AIE.dmaBd(<%buf74_0 : memref<32xi32>, 0, 32>, 0)
      AIE.useLock(%l74_0, "Release", 1)
      cf.br ^bd0
    ^bd2:
      AIE.useLock(%l74_0, "Acquire", 1)
      AIE.dmaBd(<%buf74_0 : memref<32xi32>, 0, 32>, 0)
      AIE.useLock(%l74_0, "Release", 0)
      cf.br ^bd2
    ^end:
      AIE.end
  }

}
