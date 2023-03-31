//===- airrt_metadata.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// Note this does not check the contents of the shim descriptors.
// CHECK-LABEL:   llvm.mlir.global internal constant @__airrt_shim_descriptor_1() {addr_space = 0 : i32} : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.addressof @__airrt_shim_location_data_1 : !llvm.ptr<array<1024 x i64>>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0] : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @__airrt_shim_channel_data_1 : !llvm.ptr<array<1024 x i64>>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_2]][1] : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>
// CHECK:           llvm.return %[[VAL_4]] : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>
// CHECK:         }
// CHECK:         llvm.mlir.global internal constant @__airrt_string_herd_1("herd_1")

// CHECK-LABEL:   llvm.mlir.global external constant @__airrt_herd_descriptor_1() {addr_space = 0 : i32} : !llvm.struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.addressof @__airrt_string_herd_1 : !llvm.ptr<array<6 x i8>>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(6 : i32) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]] : (!llvm.ptr<array<6 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_0]][0] : !llvm.struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_5]][1] : !llvm.struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.addressof @__airrt_shim_descriptor_1 : !llvm.ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_6]][2] : !llvm.struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:           llvm.return %[[VAL_8]] : !llvm.struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:         }

// CHECK-LABEL:   llvm.mlir.global internal constant @__airrt_segment_herd_descriptors() {addr_space = 0 : i32} : !llvm.array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.addressof @__airrt_herd_descriptor : !llvm.ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0] : !llvm.array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @__airrt_herd_descriptor_1 : !llvm.ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_2]][1] : !llvm.array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>
// CHECK:           llvm.return %[[VAL_4]] : !llvm.array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>
// CHECK:         }

// CHECK-LABEL:   llvm.mlir.global internal constant @__airrt_module_segment_descriptors() {addr_space = 0 : i32} : !llvm.array<1 x ptr<struct<(i64, ptr<i8>, i64, ptr<array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>>> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<array<1 x ptr<struct<(i64, ptr<i8>, i64, ptr<array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>>>>)>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.addressof @__airrt_module_segment_descriptors : !llvm.ptr<array<1 x ptr<struct<(i64, ptr<i8>, i64, ptr<array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>>>>
// CHECK:           %[[VAL_3:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0] : !llvm.struct<(i64, ptr<array<1 x ptr<struct<(i64, ptr<i8>, i64, ptr<array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>>>>)>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_3]][1] : !llvm.struct<(i64, ptr<array<1 x ptr<struct<(i64, ptr<i8>, i64, ptr<array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>>>>)>
// CHECK:           llvm.return %[[VAL_4]] : !llvm.struct<(i64, ptr<array<1 x ptr<struct<(i64, ptr<i8>, i64, ptr<array<2 x ptr<struct<(i64, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>>>>)>
// CHECK:         }
module {
    airrt.module_metadata {
        airrt.segment_metadata attributes {sym_name="part_0"} {
            airrt.herd_metadata {
                sym_name = "herd_0",
                dma_allocations =
                [
                    {id=1, row=0, col=0, channel=1, location=2},
                    {id=2, row=0, col=0, channel=2, location=2},
                    {id=3, row=0, col=0, channel=3, location=2}
                ]
            }
            airrt.herd_metadata {
                sym_name = "herd_1",
                dma_allocations =
                [
                    {id=4, row=1, col=1, channel=4, location=3},
                    {id=5, row=1, col=1, channel=5, location=3},
                    {id=6, row=1, col=1, channel=6, location=3}
                ]
            }
        }
    }
}
