// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s | FileCheck %s

module {

// CHECK-LABEL: module
// CHECK: func.func @test
func.func @test() {

    %p0 = air.alloc : memref<128xbf16>
    air.dealloc %p0 : memref<128xbf16>

    %e1, %p1 = air.alloc async : memref<32xf32>
    air.dealloc [%e1] %p1 : memref<32xf32>

    %e2 = air.wait_all async
    %p2 = air.alloc [%e2] : memref<8xi8>
    %e3 = air.dealloc async %p2 : memref<8xi8>

    %e4 = air.wait_all async
    %e5, %p3 = air.alloc async [%e4] : memref<4xi4>
    %e6  = air.dealloc async [%e4, %e5] %p3 : memref<4xi4>

    air.wait_all [%e3, %e6]
    return
}

}