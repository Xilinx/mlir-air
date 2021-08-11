// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func @launch
// CHECK: air.launch_herd tile ({{.*}}, {{.*}}) in ({{.*}}={{.*}}, {{.*}}={{.*}})
func @launch(%arg0: i32) {
  %cst2 = constant 2 : index
  air.launch_herd tile (%x, %y) in (%sx=%cst2, %sy=%cst2) args (%op0=%arg0, %op1=%arg0) : i32, i32 attributes { } {
    %0 = addi %x, %y : index
    %1 = muli %sx, %sy : index
    %2 = addi %op0, %op1 : i32
    air.herd_terminator
  }
  return
}
