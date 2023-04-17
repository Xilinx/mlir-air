module {
  %0 = AIE.tile(6, 0)
  %1 = AIE.switchbox(%0) {
    AIE.connect<South : 3, North : 0>
    AIE.connect<South : 7, North : 1>
    AIE.connect<North : 0, South : 2>
    AIE.connect<North : 1, South : 3>
  }
  %2 = AIE.tile(6, 1)
  %3 = AIE.switchbox(%2) {
    AIE.connect<South : 0, North : 0>
    AIE.connect<South : 1, North : 1>
    AIE.connect<North : 0, South : 0>
    AIE.connect<North : 1, South : 1>
  }
  %4 = AIE.tile(6, 2)
  %5 = AIE.switchbox(%4) {
    AIE.connect<South : 0, DMA : 0>
    AIE.connect<South : 1, DMA : 1>
    AIE.connect<DMA : 0, South : 0>
    AIE.connect<DMA : 1, South : 1>
  }
  %6 = AIE.shimmux(%0) {
    AIE.connect<DMA : 0, North : 3>
    AIE.connect<DMA : 1, North : 7>
    AIE.connect<North : 2, DMA : 0>
    AIE.connect<North : 3, DMA : 1>
  }
  %7 = AIE.buffer(%4) {address = 4096 : i32, sym_name = "ping_in"} : memref<8xi32>
  %8 = AIE.buffer(%4) {address = 4128 : i32, sym_name = "ping_out"} : memref<8xi32>
  %9 = AIE.buffer(%4) {address = 4160 : i32, sym_name = "pong_in"} : memref<8xi32>
  %10 = AIE.buffer(%4) {address = 4192 : i32, sym_name = "pong_out"} : memref<8xi32>
  %11 = AIE.lock(%4, 0)
  %12 = AIE.lock(%4, 1)
  %13 = AIE.lock(%4, 2)
  %14 = AIE.lock(%4, 3)
  %15 = AIE.mem(%4) {
    %17 = AIE.dmaStart(S2MM, 0, ^bb2, ^bb1)
  ^bb1:  // pred: ^bb0
    %18 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
  ^bb2:  // 2 preds: ^bb0, ^bb3
    AIE.useLock(%11, Acquire, 0)
    AIE.dmaBd(<%7 : memref<8xi32>, 0, 8>, 0)
    AIE.useLock(%11, Release, 1)
    AIE.nextBd ^bb3
  ^bb3:  // pred: ^bb2
    AIE.useLock(%12, Acquire, 0)
    AIE.dmaBd(<%9 : memref<8xi32>, 0, 8>, 0)
    AIE.useLock(%12, Release, 1)
    AIE.nextBd ^bb2
  ^bb4:  // 2 preds: ^bb1, ^bb5
    AIE.useLock(%13, Acquire, 1)
    AIE.dmaBd(<%8 : memref<8xi32>, 0, 8>, 0)
    AIE.useLock(%13, Release, 0)
    AIE.nextBd ^bb5
  ^bb5:  // pred: ^bb4
    AIE.useLock(%14, Acquire, 1)
    AIE.dmaBd(<%10 : memref<8xi32>, 0, 8>, 0)
    AIE.useLock(%14, Release, 0)
    AIE.nextBd ^bb4
  ^bb6:  // pred: ^bb1
    AIE.end
  }
  %16 = AIE.core(%4) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    AIE.useLock(%11, Acquire, 1)
    AIE.useLock(%13, Acquire, 0)
    cf.br ^bb1(%c0 : index)
  ^bb1(%17: index):  // 2 preds: ^bb0, ^bb2
    %18 = arith.cmpi slt, %17, %c8 : index
    cf.cond_br %18, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %19 = memref.load %7[%17] : memref<8xi32>
    %20 = arith.addi %19, %c1_i32 : i32
    memref.store %20, %8[%17] : memref<8xi32>
    %21 = arith.addi %17, %c1 : index
    cf.br ^bb1(%21 : index)
  ^bb3:  // pred: ^bb1
    AIE.useLock(%11, Release, 0)
    AIE.useLock(%13, Release, 1)
    AIE.useLock(%12, Acquire, 1)
    AIE.useLock(%14, Acquire, 0)
    cf.br ^bb4(%c0 : index)
  ^bb4(%22: index):  // 2 preds: ^bb3, ^bb5
    %23 = arith.cmpi slt, %22, %c8 : index
    cf.cond_br %23, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %24 = memref.load %9[%22] : memref<8xi32>
    %25 = arith.addi %24, %c1_i32 : i32
    memref.store %25, %10[%22] : memref<8xi32>
    %26 = arith.addi %22, %c1 : index
    cf.br ^bb4(%26 : index)
  ^bb6:  // pred: ^bb4
    AIE.useLock(%12, Release, 0)
    AIE.useLock(%14, Release, 1)
    AIE.end
  }
  AIE.wire(%6 : North, %1 : South)
  AIE.wire(%0 : DMA, %6 : DMA)
  AIE.wire(%2 : Core, %3 : Core)
  AIE.wire(%2 : DMA, %3 : DMA)
  AIE.wire(%1 : North, %3 : South)
  AIE.wire(%4 : Core, %5 : Core)
  AIE.wire(%4 : DMA, %5 : DMA)
  AIE.wire(%3 : North, %5 : South)
  AIE.flow(%0, DMA : 0, %4, DMA : 0)
  AIE.flow(%0, DMA : 1, %4, DMA : 1)
  AIE.flow(%4, DMA : 0, %0, DMA : 0)
  AIE.flow(%4, DMA : 1, %0, DMA : 1)
}

