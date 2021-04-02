
// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie

module {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)

  // Fixup
  %sw = AIE.switchbox(%t70) {
    AIE.connect<"South" : 3, "North" : 3>
    AIE.connect<"North" : 2, "South" : 2>
  }
  %mux = AIE.shimmux(%t70) {
    AIE.connect<"DMA" : 0, "South": 3>
    AIE.connect<"South" : 2, "DMA": 0>
  }

  AIE.flow(%t71, "South" : 3, %t72, "DMA" : 0)
  AIE.flow(%t72, "DMA" : 0, %t71, "South" : 2)

  AIE.core(%t72) {
    AIE.end
  }
  {
    elf_file = "add_one.elf"
  }

  %buf72_0 = AIE.buffer(%t72) {sym_name="a"} : memref<8xi32>
  %buf72_1 = AIE.buffer(%t72) {sym_name="b"} : memref<8xi32>
  %buf72_2 = AIE.buffer(%t72) {sym_name="c"} : memref<8xi32>
  %buf72_3 = AIE.buffer(%t72) {sym_name="d"} : memref<8xi32>

  %l72_0 = AIE.lock(%t72, 0)
  %l72_1 = AIE.lock(%t72, 1)
  %l72_2 = AIE.lock(%t72, 2)
  %l72_3 = AIE.lock(%t72, 3)

  %m72 = AIE.mem(%t72) {
      %srcDma = AIE.dmaStart("S2MM0", ^bd0, ^dma0)
    ^dma0:
      %dstDma = AIE.dmaStart("MM2S0", ^bd2, ^end)
    ^bd0:
      AIE.useLock(%l72_0, "Acquire", 0, 0)
      AIE.dmaBd(<%buf72_0 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_0, "Release", 1, 0)
      br ^bd1
    ^bd1:
      AIE.useLock(%l72_1, "Acquire", 0, 0)
      AIE.dmaBd(<%buf72_1 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_1, "Release", 1, 0)
      br ^bd0
    ^bd2:
      AIE.useLock(%l72_2, "Acquire", 1, 0)
      AIE.dmaBd(<%buf72_2 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_2, "Release", 0, 0)
      br ^bd3
    ^bd3:
      AIE.useLock(%l72_3, "Acquire", 1, 0)
      AIE.dmaBd(<%buf72_3 : memref<8xi32>, 0, 8>, 0)
      AIE.useLock(%l72_3, "Release", 0, 0)
      br ^bd2
    ^end:
      AIE.end

  }
}
