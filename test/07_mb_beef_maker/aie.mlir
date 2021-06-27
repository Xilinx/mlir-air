// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

module {
  %t72 = AIE.tile(7, 2)
  %buf72_0 = AIE.buffer(%t72) {sym_name="buffer"}: memref<4xi32>
  %lock = AIE.lock(%t72, 0)

  %core13 = AIE.core(%t72) {
    %val1 = constant 0xdeadbeef : i32
    %val2 = constant 0xcafecafe : i32
    %val3 = constant 0x000decaf : i32
    %val4 = constant 0x5a1ad000 : i32
    %idx1 = constant 0 : index
    %idx2 = constant 1 : index
    %idx3 = constant 2 : index
    %idx4 = constant 3 : index
    AIE.useLock(%lock, "Acquire", 1, 0)
    memref.store %val1, %buf72_0[%idx1] : memref<4xi32>
    memref.store %val2, %buf72_0[%idx2] : memref<4xi32>
    memref.store %val3, %buf72_0[%idx3] : memref<4xi32>
    memref.store %val4, %buf72_0[%idx4] : memref<4xi32>
    AIE.useLock(%lock, "Release", 0, 0)
    AIE.end
  }
  
}
