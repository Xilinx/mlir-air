// RUN: aie-opt --aie-create-flows --aie-find-flows %s | FileCheck %s
// CHECK: AIE.flow(%2, "DMA" : 0, %0, "DMA" : 0)

// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie 

module {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)

  // Fixup
  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 0>
    AIE.connect<"South" : 7, "North" : 1>
    AIE.connect<"North" : 0, "South" : 2>
    AIE.connect<"North" : 1, "South" : 3>
  }
  %mux = AIE.shimmux(%t70) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"DMA" : 1, "South": 7>
    AIE.connect<"South" : 2, "DMA": 0>
    AIE.connect<"South" : 3, "DMA": 1>
  }
  %buf72_0 = AIE.buffer(%t72) {sym_name="a"} : memref<256xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name="b"} : memref<256xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("MM2S0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 1, 0)
      AIE.dmaBd(<%buf72_0 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_0, "Release", 0, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 1, 0)
      AIE.dmaBd(<%buf72_1 : memref<256xi32>, 0, 256>, 0)
      AIE.useLock(%l72_1, "Release", 0, 0)
      br ^bd0
    ^end:
      AIE.end
  }
  %d70 = AIE.shimDMA(%t70) {
    AIE.end
  }
  AIE.flow(%t72, "DMA" : 0, %t71, "South" : 0)
}
