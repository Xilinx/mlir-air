//===- airrt_herd_load.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// CHECK: %0 = llvm.mlir.addressof @__air_string_plot : !llvm.ptr<array<4 x i8>>
// CHECK: %1 = llvm.bitcast %0 : !llvm.ptr<array<4 x i8>> to !llvm.ptr<i8>
// CHECK: %2 = llvm.call @air_partition_load(%1) : (!llvm.ptr<i8>) -> i64
// CHECK: %3 = llvm.mlir.addressof @__air_string_elk : !llvm.ptr<array<3 x i8>>
// CHECK: %4 = llvm.bitcast %3 : !llvm.ptr<array<3 x i8>> to !llvm.ptr<i8>
// CHECK: %5 = llvm.call @air_herd_load(%4) : (!llvm.ptr<i8>) -> i64
// CHECK: %6 = llvm.mlir.addressof @__air_string_deer : !llvm.ptr<array<4 x i8>>
// CHECK: %7 = llvm.bitcast %6 : !llvm.ptr<array<4 x i8>> to !llvm.ptr<i8>
// CHECK: %8 = llvm.call @air_herd_load(%7) : (!llvm.ptr<i8>) -> i64
module {
    airrt.module_metadata {
        airrt.partition_metadata attributes {sym_name="plot"} {
            airrt.herd_metadata { sym_name = "elk", dma_allocations = [] }
            airrt.herd_metadata { sym_name = "deer", dma_allocations = [] }
        }
    }
    func.func @f() {
        %ret0 = airrt.partition_load "plot" : i64
        %ret1 = airrt.herd_load "elk" : i64
        airrt.herd_load "deer" : i64
        return
    }
}