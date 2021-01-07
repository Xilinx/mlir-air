
// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie

module {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)

  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 0, "North" : 0>
  }

//  %mux = AIE.shimmux(%t70) {
//    AIE.connect<"DMA" : 0, "South": 3>
//  }

  AIE.flow(%t71, "South" : 0, %t72, "DMA" : 0)

  %buf62_0 = AIE.buffer(%t72) : memref<16xi32>
  %buf62_1 = AIE.buffer(%t72) : memref<16xi32>

  %l62_0 = AIE.lock(%t72, 0)
  %l62_1 = AIE.lock(%t72, 1)

  %m62 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l62_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf62_0 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l62_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l62_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf62_1 : memref<16xi32>, 0, 16>, 0)
      AIE.useLock(%l62_1, "Release", 1, 0)
      br ^bd0
    ^end:
      AIE.end
  }
}
