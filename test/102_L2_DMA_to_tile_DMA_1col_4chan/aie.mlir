// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

module @aie.0  {
  %0 = AIE.tile(7, 1)
  %1 = AIE.tile(7, 0)
  %2 = AIE.tile(7, 3)
  %3 = AIE.tile(7, 4)
  %99 = AIE.tile(7, 2)
  %24 = AIE.lock(%2, 2)
  %25 = AIE.buffer(%2) {sym_name = "buf5"} : memref<16xi32, 2>
  %26 = AIE.lock(%2, 1)
  %27 = AIE.buffer(%2) {sym_name = "buf4"} : memref<16xi32, 2>
  %28 = AIE.lock(%2, 0)
  %29 = AIE.buffer(%2) {sym_name = "buf3"} : memref<4xi32, 2>
  %30 = AIE.mem(%2)  {
    %214 = AIE.dmaStart(S2MM0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb3
    AIE.useLock(%28, Acquire, 0, 0)
    AIE.dmaBd(<%29 : memref<4xi32, 2>, 0, 1>, 0)
    AIE.useLock(%28, Release, 1, 0)
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%26, Acquire, 0, 0)
    AIE.dmaBd(<%27 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%26, Release, 1, 0)
    br ^bb3
  ^bb3:  // pred: ^bb2
    AIE.useLock(%24, Acquire, 0, 0)
    AIE.dmaBd(<%25 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%24, Release, 1, 0)
    br ^bb1
  ^bb4:  // pred: ^bb0
    AIE.end
  }
  %31 = AIE.core(%2)  {
    br ^bb1
  ^bb1:  // pred: ^bb0
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%28, Acquire, 1, 0)
    AIE.useLock(%26, Acquire, 1, 0)
    AIE.useLock(%24, Acquire, 1, 0)
    AIE.end
  }
  %4 = AIE.lock(%3, 2)
  %5 = AIE.buffer(%3) {sym_name = "buf2"} : memref<16xi32, 2>
  %6 = AIE.lock(%3, 1)
  %7 = AIE.buffer(%3) {sym_name = "buf1"} : memref<16xi32, 2>
  %8 = AIE.lock(%3, 0)
  %9 = AIE.buffer(%3) {sym_name = "buf0"} : memref<4xi32, 2>
  %20 = AIE.mem(%3)  {
    %14 = AIE.dmaStart(S2MM0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb3
    AIE.useLock(%8, Acquire, 0, 0)
    AIE.dmaBd(<%9 : memref<4xi32, 2>, 0, 1>, 0)
    AIE.useLock(%8, Release, 1, 0)
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%6, Acquire, 0, 0)
    AIE.dmaBd(<%7 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%6, Release, 1, 0)
    br ^bb3
  ^bb3:  // pred: ^bb2
    AIE.useLock(%4, Acquire, 0, 0)
    AIE.dmaBd(<%5 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%4, Release, 1, 0)
    br ^bb1
  ^bb4:  // pred: ^bb0
    AIE.end
  }
  %11 = AIE.core(%3)  {
    br ^bb1
  ^bb1:  // pred: ^bb0
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%8, Acquire, 1, 0)
    AIE.useLock(%6, Acquire, 1, 0)
    AIE.useLock(%4, Acquire, 1, 0)
    AIE.end
  }
  %44 = AIE.lock(%0, 2)
  %45 = AIE.buffer(%0) {sym_name = "buf8"} : memref<16xi32, 2>
  %46 = AIE.lock(%0, 1)
  %47 = AIE.buffer(%0) {sym_name = "buf7"} : memref<16xi32, 2>
  %48 = AIE.lock(%0, 0)
  %49 = AIE.buffer(%0) {sym_name = "buf6"} : memref<4xi32, 2>
  %40 = AIE.mem(%0)  {
    %414 = AIE.dmaStart(S2MM0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb3
    AIE.useLock(%48, Acquire, 0, 0)
    AIE.dmaBd(<%49 : memref<4xi32, 2>, 0, 1>, 0)
    AIE.useLock(%48, Release, 1, 0)
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%46, Acquire, 0, 0)
    AIE.dmaBd(<%47 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%46, Release, 1, 0)
    br ^bb3
  ^bb3:  // pred: ^bb2
    AIE.useLock(%44, Acquire, 0, 0)
    AIE.dmaBd(<%45 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%44, Release, 1, 0)
    br ^bb1
  ^bb4:  // pred: ^bb0
    AIE.end
  }
  %41 = AIE.core(%0)  {
    br ^bb1
  ^bb1:  // pred: ^bb0
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%48, Acquire, 1, 0)
    AIE.useLock(%46, Acquire, 1, 0)
    AIE.useLock(%44, Acquire, 1, 0)
    AIE.end
  }
  %54 = AIE.lock(%99, 2)
  %55 = AIE.buffer(%99) {sym_name = "buf11"} : memref<16xi32, 2>
  %56 = AIE.lock(%99, 1)
  %57 = AIE.buffer(%99) {sym_name = "buf10"} : memref<16xi32, 2>
  %58 = AIE.lock(%99, 0)
  %59 = AIE.buffer(%99) {sym_name = "buf9"} : memref<4xi32, 2>
  %50 = AIE.mem(%99)  {
    %514 = AIE.dmaStart(S2MM0, ^bb1, ^bb4)
  ^bb1:  // 2 preds: ^bb0, ^bb3
    AIE.useLock(%58, Acquire, 0, 0)
    AIE.dmaBd(<%59 : memref<4xi32, 2>, 0, 1>, 0)
    AIE.useLock(%58, Release, 1, 0)
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%56, Acquire, 0, 0)
    AIE.dmaBd(<%57 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%56, Release, 1, 0)
    br ^bb3
  ^bb3:  // pred: ^bb2
    AIE.useLock(%54, Acquire, 0, 0)
    AIE.dmaBd(<%55 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%54, Release, 1, 0)
    br ^bb1
  ^bb4:  // pred: ^bb0
    AIE.end
  }
  %51 = AIE.core(%99)  {
    br ^bb1
  ^bb1:  // pred: ^bb0
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%58, Acquire, 1, 0)
    AIE.useLock(%56, Acquire, 1, 0)
    AIE.useLock(%54, Acquire, 1, 0)
    AIE.end
  }
  AIE.flow(%1, PLIO : 0, %0,  DMA : 0)
  AIE.flow(%1, PLIO : 1, %99, DMA : 0)
  AIE.flow(%1, PLIO : 4, %3,  DMA : 0)
  AIE.flow(%1, PLIO : 5, %2,  DMA : 0)
}
