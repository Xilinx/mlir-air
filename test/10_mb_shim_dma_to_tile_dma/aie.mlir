
// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie

module {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)

  // Fixup
  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 3>
  }
  %mux = AIE.shimmux(%t70) {
    AIE.connect<"DMA" : 0, "South": 3>
  }

  AIE.flow(%t71, "South" : 3, %t72, "DMA" : 0)

  %buf72_0 = AIE.buffer(%t72) {sym_name="b0"} : memref<512xi32>

  %l72_0 = AIE.lock(%t72, 0)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf72_0 : memref<512xi32>, 0, 512>, 0)
      AIE.useLock(%l72_0, "Release", 1, 0)
      br ^bd0
    ^end:
      AIE.end
  }
  %d70 = AIE.shimDMA(%t70) {
    AIE.end
  }
}
