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

module @aie.0  {
  %0 = AIE.tile(7, 1)
  %1 = AIE.tile(7, 0)
  %2 = AIE.tile(0, 0)
  %3 = AIE.tile(7, 4)
  %4 = AIE.lock(%3, 2)
  %5 = AIE.buffer(%3) {sym_name = "buf2"} : memref<16xi32, 2>
  %6 = AIE.lock(%3, 1)
  %7 = AIE.buffer(%3) {sym_name = "buf1"} : memref<16xi32, 2>
  %10 = AIE.mem(%3)  {
    %14 = AIE.dmaStart(S2MM, 0 ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb3
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%6, Acquire, 0)
    AIE.dmaBd(<%7 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%6, Release, 1)
    cf.br ^bb3
  ^bb3:  // pred: ^bb2
    AIE.useLock(%4, Acquire, 0)
    AIE.dmaBd(<%5 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%4, Release, 1)
    cf.br ^bb1
  ^bb4:  // pred: ^bb0
    AIE.end
  }
  %11 = AIE.core(%3)  {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%6, Acquire, 1)
    AIE.useLock(%4, Acquire, 1)
    AIE.end
  }
  AIE.flow(%1, PLIO : 0, %3, DMA : 0)
}
