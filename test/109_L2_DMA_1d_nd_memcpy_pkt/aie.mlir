//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
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

module @aie.0  {
  %t70 = AIE.tile(7, 0)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  %10 = AIE.lock(%t73, 1)
  %11 = AIE.buffer(%t73) {sym_name = "buf1"} : memref<64xi32, 2>

  %6 = AIE.lock(%t74, 1)
  %7 = AIE.buffer(%t74) {sym_name = "buf2"} : memref<64xi32, 2>

  %8 = AIE.mem(%t74)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2: 
    AIE.useLock(%6, Acquire, 0)
    AIE.dmaBd(<%7 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%6, Release, 1)
    cf.br ^bb2
  ^bb3: 
    AIE.useLock(%6, Acquire, 1)
    AIE.dmaBd(<%7 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%6, Release, 0)
    cf.br ^bb3
  ^end: 
    AIE.end
  }
  %12 = AIE.mem(%t73)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2: 
    AIE.useLock(%10, Acquire, 0)
    AIE.dmaBd(<%11 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%10, Release, 1)
    cf.br ^bb2
  ^bb3: 
    AIE.useLock(%10, Acquire, 1)
    AIE.dmaBd(<%11 : memref<64xi32, 2>, 0, 64>, 0)
    AIE.useLock(%10, Release, 0)
    cf.br ^bb3
  ^end: 
    AIE.end
  }
  AIE.flow(%t73, DMA : 0, %t70, PLIO : 0)
  AIE.flow(%t74, DMA : 0, %t70, PLIO : 1)
  AIE.flow(%t70, PLIO : 4, %t73, DMA : 0)
  AIE.flow(%t70, PLIO : 5, %t74, DMA : 0)
}
