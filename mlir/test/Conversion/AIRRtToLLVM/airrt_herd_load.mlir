//===- airrt_herd_load.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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