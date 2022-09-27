//===- herd.airrt.mlir -----------------------------------------*- MLIR -*-===//
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
  airrt.module_metadata {
    airrt.partition_metadata attributes {
      sym_name = "partition_0"
    } {
      airrt.herd_metadata {
        sym_name = "herd_0",
        dma_allocations =
        [
          {id=1, row=0, col=0, channel=1, location=2},
          {id=2, row=0, col=0, channel=2, location=2},
          {id=3, row=0, col=0, channel=3, location=2}
        ]
      }
      airrt.herd_metadata {
        sym_name = "herd_1",
        dma_allocations =
        [
          {id=4, row=1, col=1, channel=4, location=3},
          {id=5, row=1, col=1, channel=5, location=3},
          {id=6, row=1, col=1, channel=6, location=3}
        ]
      }
    }
    // airrt.partition_metadata attributes {
    //   sym_name = "partition_1"
    // } {
    //   airrt.herd_metadata {
    //     sym_name = "herd_2",
    //     dma_allocations =
    //     [
    //       {id=7, row=2, col=2, channel=7, location=4},
    //       {id=8, row=2, col=2, channel=8, location=4},
    //       {id=9, row=2, col=2, channel=9, location=4}
    //     ]
    //   }
    // }
  }
}
