//===- airrt_herd_load.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// XFAIL:*
// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// CHECK: %0 = llvm.mlir.addressof @__airrt_string_plot : !llvm.ptr
// CHECK: %1 = llvm.bitcast %0 : !llvm.ptr to !llvm.ptr<i8>
// CHECK: %2 = call @__airrt_segment_load(%1) : (!llvm.ptr<i8>) -> i64
// CHECK: %3 = llvm.mlir.addressof @__airrt_string_elk : !llvm.ptr
// CHECK: %4 = llvm.bitcast %3 : !llvm.ptr to !llvm.ptr<i8>
// CHECK: %5 = call @__airrt_herd_load(%4) : (!llvm.ptr<i8>) -> i64
// CHECK: %6 = llvm.mlir.addressof @__airrt_string_deer : !llvm.ptr
// CHECK: %7 = llvm.bitcast %6 : !llvm.ptr to !llvm.ptr<i8>
// CHECK: %8 = call @__airrt_herd_load(%7) : (!llvm.ptr<i8>) -> i64
module {
    airrt.module_metadata {
        airrt.segment_metadata attributes {sym_name="plot"} {
            airrt.herd_metadata { sym_name = "elk", dma_allocations = [] }
            airrt.herd_metadata { sym_name = "deer", dma_allocations = [] }
        }
    }
    func.func @f() {
        %ret0 = airrt.segment_load "plot" : i64
        %ret1 = airrt.herd_load "elk" : i64
        airrt.herd_load "deer" : i64
        return
    }
}