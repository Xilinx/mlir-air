// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// Note this does not check the contents of the shim descriptors.
// CHECK-LABEL:   llvm.mlir.global internal constant @__air_shim_descriptor_1() : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.addressof @__air_shim_location_data_1 : !llvm.ptr<array<1024 x i64>>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0] : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @__air_shim_channel_data_1 : !llvm.ptr<array<1024 x i64>>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_2]][1] : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>
// CHECK:           llvm.return %[[VAL_4]] : !llvm.struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>
// CHECK:         }
// CHECK:         llvm.mlir.global internal constant @__air_string_herd_1("herd_1")

// CHECK-LABEL:   llvm.mlir.global external constant @__air_herd_descriptor_1() : !llvm.struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.addressof @__air_string_herd_1 : !llvm.ptr<array<6 x i8>>
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]] : (!llvm.ptr<array<6 x i8>>, i32, i32) -> !llvm.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_0]][0] : !llvm.struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:           %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_4]], %[[VAL_5]][1] : !llvm.struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.addressof @__air_shim_descriptor_1 : !llvm.ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>
// CHECK:           %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_6]][2] : !llvm.struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:           llvm.return %[[VAL_8]] : !llvm.struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>
// CHECK:         }

// CHECK-LABEL:   llvm.mlir.global internal constant @__air_module_herd_descriptors() : !llvm.array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.addressof @__air_herd_descriptor : !llvm.ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>
// CHECK:           %[[VAL_2:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0 : i32] : !llvm.array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @__air_herd_descriptor_1 : !llvm.ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_2]][1 : i32] : !llvm.array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>
// CHECK:           llvm.return %[[VAL_4]] : !llvm.array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>
// CHECK:         }

// CHECK-LABEL:   llvm.mlir.global external constant @__air_module_descriptor() : !llvm.struct<(i64, ptr<array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.addressof @__air_module_herd_descriptors : !llvm.ptr<array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>
// CHECK:           %[[VAL_4:.*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_0]][0 : i32] : !llvm.struct<(i64, ptr<array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>
// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_4]][1 : i32] : !llvm.struct<(i64, ptr<array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>
// CHECK:           llvm.return %[[VAL_5]] : !llvm.struct<(i64, ptr<array<2 x ptr<struct<(i32, ptr<i8>, ptr<struct<(ptr<array<1024 x i64>>, ptr<array<1024 x i64>>)>>)>>>>)>
// CHECK:         }
module {
    airrt.module_metadata {
        airrt.herd_metadata {
            sym_name = "herd_0",
            shim_allocations =
            [
                {id=1, row=0, col=0, channel=1, location=2},
                {id=2, row=0, col=0, channel=2, location=2},
                {id=3, row=0, col=0, channel=3, location=2}
            ]
        }
        airrt.herd_metadata {
            sym_name = "herd_1",
            shim_allocations =
            [
                {id=4, row=1, col=1, channel=4, location=3},
                {id=5, row=1, col=1, channel=5, location=3},
                {id=6, row=1, col=1, channel=6, location=3}
            ]
        }
    }
}
