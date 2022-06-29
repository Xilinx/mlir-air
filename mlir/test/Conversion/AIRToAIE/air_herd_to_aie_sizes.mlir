// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt -air-to-aie %s | FileCheck %s

func.func @launch(%arg0: i32) {
  %cst2 = arith.constant 2 : index
  // CHECK: %[[TILE01:.*]] = AIE.tile(0, 1)
  // CHECK: {{.*}} = AIE.core(%[[TILE01]])  {
  // CHECK: memref.store {{.*}}, {{.*}}[{{.*}}] : memref<1024xindex, 2>
  // CHECK: AIE.end
  air.launch_herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) {
    %buf = memref.alloc() : memref<1024xindex,2>
    %0 = arith.addi %x, %y : index
    %1 = arith.muli %sx, %sy : index
    memref.store %0, %buf[%1] : memref<1024xindex,2>
    air.herd_terminator
  }
  return
}
