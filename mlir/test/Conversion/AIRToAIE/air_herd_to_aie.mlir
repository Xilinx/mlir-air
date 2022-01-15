// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -air-to-aie | FileCheck %s
module {

func @foo(%arg0: i32) {
  %cst1 = arith.constant 1 : index
  // CHECK-LABEL: module @aie.herd_0
  // CHECK: %[[VAR1:.*]] = AIE.tile(0, 0)
  // CHECK: %[[BUF1:.*]] = AIE.buffer(%[[VAR1]]) {sym_name = {{.*}}} : memref<1xi32, 2>
  // CHECK: %[[BUF2:.*]] = AIE.buffer(%[[VAR1]]) {sym_name = {{.*}}} : memref<1xi32, 2>
  // CHECK: %[[BUF3:.*]] = AIE.buffer(%[[VAR1]]) {sym_name = {{.*}}} : memref<1xi32, 2>
  // CHECK: %[[VAR2:.*]] = AIE.core(%[[VAR1]])  {
  air.launch_herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) {
    %src0 = memref.alloc() : memref<1xi32, 2>
    %src1 = memref.alloc() : memref<1xi32, 2>
    %zero = arith.constant 0 : index
    // CHECK: load %[[BUF3]]
    %0 = memref.load %src0[%zero] : memref<1xi32, 2>
    // CHECK: load %[[BUF2]]
    %1 = memref.load %src1[%zero] : memref<1xi32, 2>
    %2 = arith.addi %0, %1 :  i32
    %dst0 = memref.alloc() : memref<1xi32, 2>
    // CHECK: memref.store {{.*}}, %[[BUF1]]
    memref.store %2, %dst0[%zero] : memref<1xi32, 2>
    air.herd_terminator
  }
  return
}

}
