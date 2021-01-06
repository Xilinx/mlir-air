
// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie

module {
  %t60 = AIE.tile(6, 0)
  %t61 = AIE.tile(6, 1)
  %t62 = AIE.tile(6, 2)

  %sw = AIE.switchbox(%t60) {
    AIE.connect<"South" : 0, "North" : 0>
  }

//  %mux = AIE.shimmux(%t60) {
//    AIE.connect<"DMA" : 0, "South": 3>
//  }

  AIE.flow(%t61, "South" : 0, %t62, "DMA" : 0)

  %buf62_0 = AIE.buffer(%t62) : memref<16xi32>
  %buf62_1 = AIE.buffer(%t62) : memref<16xi32>

  %l62_0 = AIE.lock(%t62, 0)
  %l62_1 = AIE.lock(%t62, 1)

  %m62 = AIE.mem(%t62) {
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
