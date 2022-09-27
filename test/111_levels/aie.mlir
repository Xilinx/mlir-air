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
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)

  %10 = AIE.lock(%t71, 1)
  %11 = AIE.buffer(%t71) {sym_name = "buf1"} : memref<512xi32, 2>
  %14 = AIE.lock(%t72, 1)
  %15 = AIE.buffer(%t72) {sym_name = "buf2"} : memref<512xi32, 2>
  %18 = AIE.lock(%t73, 1)
  %19 = AIE.buffer(%t73) {sym_name = "buf3"} : memref<512xi32, 2>
  %22 = AIE.lock(%t74, 1)
  %23 = AIE.buffer(%t74) {sym_name = "buf4"} : memref<512xi32, 2>

  %12 = AIE.mem(%t71)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2: 
    AIE.useLock(%10, Acquire, 0)
    AIE.dmaBdPacket(0x1, 3)
    AIE.dmaBd(<%11 : memref<512xi32, 2>, 0, 512>, 0)
    AIE.useLock(%10, Release, 1)
    cf.br ^bb2
  ^bb3: 
    AIE.useLock(%10, Acquire, 1)
    AIE.dmaBdPacket(0x6, 10)
    AIE.dmaBd(<%11 : memref<512xi32, 2>, 0, 512>, 0)
    AIE.useLock(%10, Release, 0)
    cf.br ^bb3
  ^end: 
    AIE.end
  }
  %16 = AIE.mem(%t72)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2: 
    AIE.useLock(%14, Acquire, 0)
    AIE.dmaBdPacket(0x2, 3)
    AIE.dmaBd(<%15 : memref<512xi32, 2>, 0, 512>, 0)
    AIE.useLock(%14, Release, 1)
    cf.br ^bb2
  ^bb3: 
    AIE.useLock(%14, Acquire, 1)
    AIE.dmaBdPacket(0x7, 11)
    AIE.dmaBd(<%15 : memref<512xi32, 2>, 0, 512>, 0)
    AIE.useLock(%14, Release, 0)
    cf.br ^bb3
  ^end: 
    AIE.end
  }
  %20 = AIE.mem(%t73)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2: 
    AIE.useLock(%18, Acquire, 0)
    AIE.dmaBdPacket(0x3, 3)
    AIE.dmaBd(<%19 : memref<512xi32, 2>, 0, 512>, 0)
    AIE.useLock(%18, Release, 1)
    cf.br ^bb2
  ^bb3: 
    AIE.useLock(%18, Acquire, 1)
    AIE.dmaBdPacket(0x8, 12)
    AIE.dmaBd(<%19 : memref<512xi32, 2>, 0, 512>, 0)
    AIE.useLock(%18, Release, 0)
    cf.br ^bb3
  ^end: 
    AIE.end
  }
  %24 = AIE.mem(%t74)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2: 
    AIE.useLock(%22, Acquire, 0)
    AIE.dmaBdPacket(0x4, 3)
    AIE.dmaBd(<%23 : memref<512xi32, 2>, 0, 512>, 0)
    AIE.useLock(%22, Release, 1)
    cf.br ^bb2
  ^bb3: 
    AIE.useLock(%22, Acquire, 1)
    AIE.dmaBdPacket(0x9, 13)
    AIE.dmaBd(<%23 : memref<512xi32, 2>, 0, 512>, 0)
    AIE.useLock(%22, Release, 0)
    cf.br ^bb3
  ^end: 
    AIE.end
  }

  AIE.packet_flow(0x3) {
    AIE.packet_source<%t70, South : 4>
    AIE.packet_dest<%t71, DMA : 0>
  }
  AIE.packet_flow(0x3) {
    AIE.packet_source<%t70, South : 4>
    AIE.packet_dest<%t72, DMA : 0>
  }
  AIE.packet_flow(0x3) {
    AIE.packet_source<%t70, South : 4>
    AIE.packet_dest<%t73, DMA : 0>
  }
  AIE.packet_flow(0x3) {
    AIE.packet_source<%t70, South : 4>
    AIE.packet_dest<%t74, DMA : 0>
  }

  AIE.packet_flow(0xA) {
    AIE.packet_source<%t71, DMA : 0>
    AIE.packet_dest<%t70, South : 0>
  }
  AIE.packet_flow(0xB) {
    AIE.packet_source<%t72, DMA : 0>
    AIE.packet_dest<%t70, South : 0>
  }
  AIE.packet_flow(0xC) {
    AIE.packet_source<%t73, DMA : 0>
    AIE.packet_dest<%t70, South : 0>
  }
  AIE.packet_flow(0xD) {
    AIE.packet_source<%t74, DMA : 0>
    AIE.packet_dest<%t70, South : 0>
  }
}
