// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

module {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)
  %t72 = AIE.tile(7, 2)
  %t80 = AIE.tile(8, 0)
  %t81 = AIE.tile(8, 1)
  %t82 = AIE.tile(8, 2)

  %buf71_a = AIE.buffer(%t71) {sym_name="a71"}: memref<1024xi32>
  %buf71_b = AIE.buffer(%t71) {sym_name="b71"}: memref<1024xi32>
  %buf72_a = AIE.buffer(%t72) {sym_name="a72"}: memref<1024xi32>
  %buf72_b = AIE.buffer(%t72) {sym_name="b72"}: memref<1024xi32>
  %buf81_a = AIE.buffer(%t81) {sym_name="a81"}: memref<1024xi32>
  %buf81_b = AIE.buffer(%t81) {sym_name="b81"}: memref<1024xi32>
  %buf82_a = AIE.buffer(%t82) {sym_name="a82"}: memref<1024xi32>
  %buf82_b = AIE.buffer(%t82) {sym_name="b82"}: memref<1024xi32>
  %l71_a = AIE.lock(%t71, 0)
  %l71_b = AIE.lock(%t71, 1)
  %l71_arm = AIE.lock(%t71, 8)
  %l72_a = AIE.lock(%t72, 0)
  %l72_b = AIE.lock(%t72, 1)
  %l72_arm = AIE.lock(%t72, 8)
  %l81_a = AIE.lock(%t81, 0)
  %l81_b = AIE.lock(%t81, 1)
  %l81_arm = AIE.lock(%t81, 8)
  %l82_a = AIE.lock(%t82, 0)
  %l82_b = AIE.lock(%t82, 1)
  %l82_arm = AIE.lock(%t82, 8)

  AIE.flow(%t70,PLIO:0, %t71, DMA : 0)
  AIE.flow(%t70,PLIO:0, %t81, DMA : 0)

  AIE.flow(%t80,PLIO:0, %t72, DMA : 0)
  AIE.flow(%t80,PLIO:0, %t82, DMA : 0)

  AIE.flow(%t70,PLIO:1, %t71, DMA : 1)
  AIE.flow(%t70,PLIO:1, %t72, DMA : 1)
  
  AIE.flow(%t80,PLIO:1, %t81, DMA : 1)
  AIE.flow(%t80,PLIO:1, %t82, DMA : 1)
    

 %m71 = AIE.mem(%t71) {
    %dma71_a = AIE.dmaStart(S2MM0,^bb1,^dma71_b)
  ^dma1:
    %dma71_b = AIE.dmaStart(S2MM1,^bb2,^end)
  ^bb1:
    AIE.useLock(%l71_a, Acquire, 0, 0)
    AIE.dmaBd(<%buf71_a : memref<1024xi32>, 2>, 0, 1024, 0)
    AIE.useLock(%l71_a, Release, 1, 0)
    br ^bb1
  ^bb2:
    AIE.useLock(%l71_b, Acquire, 0, 0)
    AIE.dmaBd(<%buf71_b : memref<1024xi32>, 2>, 0, 1024, 0)
    AIE.useLock(%l71_b, Release, 1, 0)
    br ^bb2
  ^end:
    AIE.end
 }

 %core71 = AIE.core(%t71) {
    br ^bb1
  ^bb1:
    AIE.useLock(%l71_a, Acquire, 1, 0)
    AIE.useLock(%l71_b, Acquire, 1, 0)
    AIE.useLock(%l71_arm, Acquire, 1, 0)
    AIE.useLock(%l71_a, Release, 0, 0)
    AIE.useLock(%l71_b, Release, 0, 0)
    AIE.useLock(%l71_arm, Release, 0, 0)
    br ^bb1
    AIE.end
}
 %m72 = AIE.mem(%t72) {
    %dma72_a = AIE.dmaStart(S2MM0,^bb1,^dma72_b)
  ^dma1:
    %dma72_b = AIE.dmaStart(S2MM1,^bb2,^end)
  ^bb1:
    AIE.useLock(%l72_a, Acquire, 0, 0)
    AIE.dmaBd(<%buf72_a : memref<1024xi32>, 2>, 0, 1024, 0)
    AIE.useLock(%l72_a, Release, 1, 0)
    br ^bb1
  ^bb2:
    AIE.useLock(%l72_b, Acquire, 0, 0)
    AIE.dmaBd(<%buf72_b : memref<1024xi32>, 2>, 0, 1024, 0)
    AIE.useLock(%l72_b, Release, 1, 0)
    br ^bb2
  ^end:
    AIE.end
 }

 %core72 = AIE.core(%t72) {
    br ^bb1
  ^bb1:
    AIE.useLock(%l72_a, Acquire, 1, 0)
    AIE.useLock(%l72_b, Acquire, 1, 0)
    AIE.useLock(%l72_arm, Acquire, 1, 0)
    AIE.useLock(%l72_a, Release, 0, 0)
    AIE.useLock(%l72_b, Release, 0, 0)
    AIE.useLock(%l72_arm, Release, 0, 0)
    br ^bb1
    AIE.end
}


 %m81 = AIE.mem(%t81) {
    %dma81_a = AIE.dmaStart(S2MM0,^bb1,^dma81_b)
  ^dma1:
    %dma81_b = AIE.dmaStart(S2MM1,^bb2,^end)
  ^bb1:
    AIE.useLock(%l81_a, Acquire, 0, 0)
    AIE.dmaBd(<%buf81_a : memref<1024xi32>, 2>, 0, 1024, 0)
    AIE.useLock(%l81_a, Release, 1, 0)
    br ^bb1
  ^bb2:
    AIE.useLock(%l81_b, Acquire, 0, 0)
    AIE.dmaBd(<%buf81_b : memref<1024xi32>, 2>, 0, 1024, 0)
    AIE.useLock(%l81_b, Release, 1, 0)
    br ^bb2
  ^end:
    AIE.end
 }

 %core81 = AIE.core(%t81) {
    br ^bb1
  ^bb1:
    AIE.useLock(%l81_a, Acquire, 1, 0)
    AIE.useLock(%l81_b, Acquire, 1, 0)
    AIE.useLock(%l81_arm, Acquire, 1, 0)
    AIE.useLock(%l81_a, Release, 0, 0)
    AIE.useLock(%l81_b, Release, 0, 0)
    AIE.useLock(%l81_arm, Release, 0, 0)
    br ^bb1
    AIE.end
}


 %m82 = AIE.mem(%t82) {
    %dma82_a = AIE.dmaStart(S2MM0,^bb1,^dma82_b)
  ^dma1:
    %dma82_b = AIE.dmaStart(S2MM1,^bb2,^end)
  ^bb1:
    AIE.useLock(%l82_a, Acquire, 0, 0)
    AIE.dmaBd(<%buf82_a : memref<1024xi32>, 2>, 0, 1024, 0)
    AIE.useLock(%l82_a, Release, 1, 0)
    br ^bb1
  ^bb2:
    AIE.useLock(%l82_b, Acquire, 0, 0)
    AIE.dmaBd(<%buf82_b : memref<1024xi32>, 2>, 0, 1024, 0)
    AIE.useLock(%l82_b, Release, 1, 0)
    br ^bb2
  ^end:
    AIE.end
 }

 %core82 = AIE.core(%t82) {
    br ^bb1
  ^bb1:
    AIE.useLock(%l82_a, Acquire, 1, 0)
    AIE.useLock(%l82_b, Acquire, 1, 0)
    AIE.useLock(%l82_arm, Acquire, 1, 0)
    AIE.useLock(%l82_a, Release, 0, 0)
    AIE.useLock(%l82_b, Release, 0, 0)
    AIE.useLock(%l82_arm, Release, 0, 0)
    br ^bb1
    AIE.end
}


}
