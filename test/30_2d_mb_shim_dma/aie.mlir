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
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  AIE.flow(%t70, "DMA" : 0, %t72, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 0, %t70, "DMA" : 0)

  %buf72_0 = AIE.buffer(%t72) {sym_name = "buf72_0"} : memref<128xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name = "buf72_1"} : memref<128xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart(S2MM, 0 ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart(MM2S, 0 ^bd2, ^end)
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

}
