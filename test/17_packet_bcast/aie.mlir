module @aie.partition_0 {
  %0 = AIE.tile(7, 2)
  %1 = AIE.tile(8, 2)
  %2 = AIE.tile(7, 3)
  %3 = AIE.tile(8, 3)
  %4 = AIE.tile(2, 0)
  %5 = AIE.tile(3, 0)
  %6 = AIE.tile(6, 0)
  %7 = AIE.lock(%0, 3)
  %8 = AIE.lock(%0, 2)
  %9 = AIE.lock(%0, 1)
  %10 = AIE.lock(%0, 0)
  %11 = AIE.lock(%1, 3)
  %12 = AIE.lock(%1, 2)
  %13 = AIE.lock(%1, 1)
  %14 = AIE.lock(%1, 0)
  %15 = AIE.lock(%2, 3)
  %16 = AIE.lock(%2, 2)
  %17 = AIE.lock(%2, 1)
  %18 = AIE.lock(%2, 0)
  %19 = AIE.lock(%3, 3)
  %20 = AIE.lock(%3, 2)
  %21 = AIE.lock(%3, 1)
  %22 = AIE.lock(%3, 0)
  %23 = AIE.buffer(%3) {sym_name = "buf11"} : memref<32x32xi32, 2>
  %24 = AIE.buffer(%3) {sym_name = "buf10"} : memref<32x32xi32, 2>
  %25 = AIE.buffer(%3) {sym_name = "buf9"} : memref<32x32xi32, 2>
  %26 = AIE.buffer(%2) {sym_name = "buf8"} : memref<32x32xi32, 2>
  %27 = AIE.buffer(%2) {sym_name = "buf7"} : memref<32x32xi32, 2>
  %28 = AIE.buffer(%2) {sym_name = "buf6"} : memref<32x32xi32, 2>
  %29 = AIE.buffer(%1) {sym_name = "buf5"} : memref<32x32xi32, 2>
  %30 = AIE.buffer(%1) {sym_name = "buf4"} : memref<32x32xi32, 2>
  %31 = AIE.buffer(%1) {sym_name = "buf3"} : memref<32x32xi32, 2>
  %32 = AIE.buffer(%0) {sym_name = "buf2"} : memref<32x32xi32, 2>
  %33 = AIE.buffer(%0) {sym_name = "buf1"} : memref<32x32xi32, 2>
  %34 = AIE.buffer(%0) {sym_name = "buf0"} : memref<32x32xi32, 2>
  memref.global "public" @__air_herd_arg_9 : memref<64x64xi32>
  memref.global "public" @__air_herd_arg_10 : memref<64x64xi32>
  memref.global "public" @__air_herd_arg_11 : memref<64x64xi32>
  %35 = AIE.mem(%3) {
    %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb6)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%22, Acquire, 0)
    AIE.dmaBd(<%23 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%22, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb3
    AIE.end
  ^bb3:  // pred: ^bb6
    %44 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
  ^bb4:  // 2 preds: ^bb3, ^bb5
    AIE.useLock(%21, Acquire, 0)
    AIE.dmaBd(<%24 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%21, Release, 1)
    cf.br ^bb5
  ^bb5:  // pred: ^bb4
    AIE.useLock(%20, Acquire, 0)
    AIE.dmaBd(<%25 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%20, Release, 1)
    cf.br ^bb4
  ^bb6:  // pred: ^bb0
    %45 = AIE.dmaStart(MM2S, 0, ^bb7, ^bb3)
  ^bb7:  // 2 preds: ^bb6, ^bb7
    AIE.useLock(%19, Acquire, 1)
    AIE.dmaBd(<%23 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%19, Release, 0)
    cf.br ^bb7
  }
  %36 = AIE.core(%3) {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%19, Acquire, 0)
    AIE.useLock(%22, Acquire, 1)
    AIE.useLock(%20, Acquire, 1)
    AIE.useLock(%21, Acquire, 1)
    affine.for %arg1 = 0 to 32 {
      affine.for %arg2 = 0 to 32 {
        %43 = affine.load %24[%arg1, %arg2] : memref<32x32xi32, 2>
        %44 = affine.load %25[%arg1, %arg2] : memref<32x32xi32, 2>
        %45 = affine.load %23[%arg1, %arg2] : memref<32x32xi32, 2>
        %46 = arith.muli %43, %44 : i32
        %47 = arith.addi %45, %46 : i32
        affine.store %47, %23[%arg1, %arg2] : memref<32x32xi32, 2>
      }
    }
    AIE.useLock(%21, Release, 0)
    AIE.useLock(%20, Release, 0)
    AIE.useLock(%22, Release, 0)
    AIE.useLock(%19, Release, 1)
    AIE.end
  } {elf_file = "partition_0_core_8_3.elf"}
  memref.global "public" @__air_herd_arg_6 : memref<64x64xi32>
  memref.global "public" @__air_herd_arg_7 : memref<64x64xi32>
  memref.global "public" @__air_herd_arg_8 : memref<64x64xi32>
  %37 = AIE.mem(%2) {
    %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb6)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%18, Acquire, 0)
    AIE.dmaBd(<%26 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%18, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb3
    AIE.end
  ^bb3:  // pred: ^bb6
    %44 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
  ^bb4:  // 2 preds: ^bb3, ^bb5
    AIE.useLock(%17, Acquire, 0)
    AIE.dmaBd(<%27 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%17, Release, 1)
    cf.br ^bb5
  ^bb5:  // pred: ^bb4
    AIE.useLock(%16, Acquire, 0)
    AIE.dmaBd(<%28 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%16, Release, 1)
    cf.br ^bb4
  ^bb6:  // pred: ^bb0
    %45 = AIE.dmaStart(MM2S, 0, ^bb7, ^bb3)
  ^bb7:  // 2 preds: ^bb6, ^bb7
    AIE.useLock(%15, Acquire, 1)
    AIE.dmaBd(<%26 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%15, Release, 0)
    cf.br ^bb7
  }
  %38 = AIE.core(%2) {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%15, Acquire, 0)
    AIE.useLock(%18, Acquire, 1)
    AIE.useLock(%16, Acquire, 1)
    AIE.useLock(%17, Acquire, 1)
    affine.for %arg1 = 0 to 32 {
      affine.for %arg2 = 0 to 32 {
        %43 = affine.load %27[%arg1, %arg2] : memref<32x32xi32, 2>
        %44 = affine.load %28[%arg1, %arg2] : memref<32x32xi32, 2>
        %45 = affine.load %26[%arg1, %arg2] : memref<32x32xi32, 2>
        %46 = arith.muli %43, %44 : i32
        %47 = arith.addi %45, %46 : i32
        affine.store %47, %26[%arg1, %arg2] : memref<32x32xi32, 2>
      }
    }
    AIE.useLock(%17, Release, 0)
    AIE.useLock(%16, Release, 0)
    AIE.useLock(%18, Release, 0)
    AIE.useLock(%15, Release, 1)
    AIE.end
  } {elf_file = "partition_0_core_7_3.elf"}
  memref.global "public" @__air_herd_arg_3 : memref<64x64xi32>
  memref.global "public" @__air_herd_arg_4 : memref<64x64xi32>
  memref.global "public" @__air_herd_arg_5 : memref<64x64xi32>
  %39 = AIE.mem(%1) {
    %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb6)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%14, Acquire, 0)
    AIE.dmaBd(<%29 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%14, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb3
    AIE.end
  ^bb3:  // pred: ^bb6
    %44 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
  ^bb4:  // 2 preds: ^bb3, ^bb5
    AIE.useLock(%13, Acquire, 0)
    AIE.dmaBd(<%30 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%13, Release, 1)
    cf.br ^bb5
  ^bb5:  // pred: ^bb4
    AIE.useLock(%12, Acquire, 0)
    AIE.dmaBd(<%31 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%12, Release, 1)
    cf.br ^bb4
  ^bb6:  // pred: ^bb0
    %45 = AIE.dmaStart(MM2S, 0, ^bb7, ^bb3)
  ^bb7:  // 2 preds: ^bb6, ^bb7
    AIE.useLock(%11, Acquire, 1)
    AIE.dmaBd(<%29 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%11, Release, 0)
    cf.br ^bb7
  }
  %40 = AIE.core(%1) {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%11, Acquire, 0)
    AIE.useLock(%14, Acquire, 1)
    AIE.useLock(%12, Acquire, 1)
    AIE.useLock(%13, Acquire, 1)
    affine.for %arg1 = 0 to 32 {
      affine.for %arg2 = 0 to 32 {
        %43 = affine.load %30[%arg1, %arg2] : memref<32x32xi32, 2>
        %44 = affine.load %31[%arg1, %arg2] : memref<32x32xi32, 2>
        %45 = affine.load %29[%arg1, %arg2] : memref<32x32xi32, 2>
        %46 = arith.muli %43, %44 : i32
        %47 = arith.addi %45, %46 : i32
        affine.store %47, %29[%arg1, %arg2] : memref<32x32xi32, 2>
      }
    }
    AIE.useLock(%13, Release, 0)
      AIE.useLock(%12, Release, 0)
    AIE.useLock(%14, Release, 0)
    AIE.useLock(%11, Release, 1)
    AIE.end
  } {elf_file = "partition_0_core_8_2.elf"}
  memref.global "public" @__air_herd_arg_0 : memref<64x64xi32>
  memref.global "public" @__air_herd_arg_1 : memref<64x64xi32>
  memref.global "public" @__air_herd_arg_2 : memref<64x64xi32>
  %41 = AIE.mem(%0) {
    %43 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb6)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%10, Acquire, 0)
    AIE.dmaBd(<%32 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%10, Release, 1)
    cf.br ^bb1
  ^bb2:  // pred: ^bb3
    AIE.end
  ^bb3:  // pred: ^bb6
    %44 = AIE.dmaStart(S2MM, 1, ^bb4, ^bb2)
  ^bb4:  // 2 preds: ^bb3, ^bb5
    AIE.useLock(%9, Acquire, 0)
    AIE.dmaBd(<%33 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%9, Release, 1)
    cf.br ^bb5
  ^bb5:  // pred: ^bb4
    AIE.useLock(%8, Acquire, 0)
    AIE.dmaBd(<%34 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%8, Release, 1)
    cf.br ^bb4
  ^bb6:  // pred: ^bb0
    %45 = AIE.dmaStart(MM2S, 0, ^bb7, ^bb3)
  ^bb7:  // 2 preds: ^bb6, ^bb7
    AIE.useLock(%7, Acquire, 1)
    AIE.dmaBd(<%32 : memref<32x32xi32, 2>, 0, 1024>, 0)
    AIE.useLock(%7, Release, 0)
    cf.br ^bb7
  }
  %42 = AIE.core(%0) {
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%7, Acquire, 0)
    AIE.useLock(%10, Acquire, 1)
    AIE.useLock(%8, Acquire, 1)
    AIE.useLock(%9, Acquire, 1)
    affine.for %arg1 = 0 to 32 {
      affine.for %arg2 = 0 to 32 {
        %43 = affine.load %33[%arg1, %arg2] : memref<32x32xi32, 2>
        %44 = affine.load %34[%arg1, %arg2] : memref<32x32xi32, 2>
        %45 = affine.load %32[%arg1, %arg2] : memref<32x32xi32, 2>
        %46 = arith.muli %43, %44 : i32
        %47 = arith.addi %45, %46 : i32
        affine.store %47, %32[%arg1, %arg2] : memref<32x32xi32, 2>
      }
    }
    AIE.useLock(%9, Release, 0)
    AIE.useLock(%8, Release, 0)
    AIE.useLock(%10, Release, 0)
    AIE.useLock(%7, Release, 1)
    AIE.end
  } {elf_file = "partition_0_core_7_2.elf"}
  AIE.flow(%3, DMA : 0, %4, DMA : 0)
  AIE.flow(%2, DMA : 0, %4, DMA : 1)
  AIE.flow(%1, DMA : 0, %5, DMA : 0)
  AIE.flow(%0, DMA : 0, %5, DMA : 1)
  AIE.broadcast_packet(%4, DMA : 0) {
    AIE.bp_id(4) {
      AIE.bp_dest<%0, DMA : 1>
      AIE.bp_dest<%1, DMA : 1>
    }
    AIE.bp_id(5) {
      AIE.bp_dest<%2, DMA : 1>
      AIE.bp_dest<%3, DMA : 1>
    }
    AIE.bp_id(3) {
      AIE.bp_dest<%1, DMA : 1>
      AIE.bp_dest<%3, DMA : 1>
    }
    AIE.bp_id(2) {
      AIE.bp_dest<%0, DMA : 1>
      AIE.bp_dest<%2, DMA : 1>
    }
  }
  AIE.flow(%5, DMA : 0, %3, DMA : 0)
  AIE.flow(%5, DMA : 1, %2, DMA : 0)
  AIE.flow(%6, DMA : 0, %1, DMA : 0)
  AIE.flow(%6, DMA : 1, %0, DMA : 0)
}
