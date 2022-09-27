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
  %t72 = AIE.tile(7, 2)
  %buf72_0 = AIE.buffer(%t72) {sym_name="buffer"}: memref<4xi32>
  %lock = AIE.lock(%t72, 0)

  %core13 = AIE.core(%t72) {
    %val1 = arith.constant 0xdeadbeef : i32
    %val2 = arith.constant 0xcafecafe : i32
    %val3 = arith.constant 0x000decaf : i32
    %val4 = arith.constant 0x5a1ad000 : i32
    %idx1 = arith.constant 0 : index
    %idx2 = arith.constant 1 : index
    %idx3 = arith.constant 2 : index
    %idx4 = arith.constant 3 : index
    AIE.useLock(%lock, "Acquire", 1)
    memref.store %val1, %buf72_0[%idx1] : memref<4xi32>
    memref.store %val2, %buf72_0[%idx2] : memref<4xi32>
    memref.store %val3, %buf72_0[%idx3] : memref<4xi32>
    memref.store %val4, %buf72_0[%idx4] : memref<4xi32>
    AIE.useLock(%lock, "Release", 0)
    AIE.end
  }
  
}
