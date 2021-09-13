// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// CHECK:  %0 = llvm.mlir.addressof @__air_string_elk : !llvm.ptr<array<3 x i8>>
// CHECK:  %1 = llvm.bitcast %0 : !llvm.ptr<array<3 x i8>> to !llvm.ptr<i8>
// CHECK:  %2 = llvm.call @air_herd_load(%1) : (!llvm.ptr<i8>) -> i32
// CHECK:  %3 = llvm.mlir.addressof @__air_string_deer : !llvm.ptr<array<4 x i8>>
// CHECK:  %4 = llvm.bitcast %3 : !llvm.ptr<array<4 x i8>> to !llvm.ptr<i8>
// CHECK:  %5 = llvm.call @air_herd_load(%4) : (!llvm.ptr<i8>) -> i32
module {
    airrt.module_metadata {
        airrt.herd_metadata { sym_name = "elk", shim_allocations = [] }
        airrt.herd_metadata { sym_name = "deer", shim_allocations = [] }
    }
    func @f() {
        %ret1 = airrt.herd_load "elk" () : () -> (i32)
        airrt.herd_load "deer" () : () -> (i32)
        return
    }
}