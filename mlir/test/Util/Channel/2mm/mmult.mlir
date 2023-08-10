//===- mmult.mlir -----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -o %T/mmult.async.llvm.mlir %s -async-to-async-runtime -async-runtime-ref-counting -async-runtime-ref-counting-opt -convert-linalg-to-affine-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-async-to-llvm -convert-memref-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -canonicalize -cse
// RUN: air-translate --mlir-to-llvmir %T/mmult.async.llvm.mlir -o %T/mmult.async.ll
// RUN: %CLANG %T/mmult.async.ll -O2 -std=c++17 -c -o %T/mmult.async.o
// RUN: %CLANG %S/main.cpp -O2 -std=c++17 %airhost_inc -c -o %T/main.o
// RUN: %CLANG %aircpu_lib %mlir_async_lib -o %T/test.exe %T/main.o %T/mmult.async.o
// RUN: %ld_lib_path %T/test.exe

module attributes {torch.debug_module_name = "model"} {
  memref.global "private" @channel_7 : memref<1x1xi64> = dense<0>
  memref.global "private" @channel_6 : memref<1x1xi64> = dense<0>
  memref.global "private" @channel_5 : memref<1x1xi64> = dense<0>
  memref.global "private" @channel_4 : memref<1x1xi64> = dense<0>
  memref.global "private" @channel_3 : memref<1x1xi64> = dense<0>
  memref.global "private" @channel_2 : memref<1x1xi64> = dense<0>
  memref.global "private" @channel_1 : memref<1x1xi64> = dense<0>
  memref.global "private" @channel_0 : memref<1x1xi64> = dense<0>
  func.func @forward(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>, %arg2: memref<32x32xi32>, %arg3: memref<32x32xi32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<32x32xi32>)
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
    memref.copy %alloc, %alloc_0 : memref<32x32xi32> to memref<32x32xi32>
    %0 = memref.get_global @channel_0 : memref<1x1xi64>
    %1 = builtin.unrealized_conversion_cast %0 : memref<1x1xi64> to memref<1x1xi64>
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<32x32xi32> to memref<?x?xi32>
    // put %arg0 into channel_0
    call @air_channel_put_M0D2I64_I64_I64_I64_I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%1, %c1, %c1, %c1, %c1, %c0, %c0, %2, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, index, index, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %3 = memref.get_global @channel_1 : memref<1x1xi64>
    %4 = builtin.unrealized_conversion_cast %3 : memref<1x1xi64> to memref<1x1xi64>
    %5 = builtin.unrealized_conversion_cast %arg1 : memref<32x32xi32> to memref<?x?xi32>
    // put %arg1 into channel_1
    call @air_channel_put_M0D2I64_I64_I64_I64_I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%4, %c1, %c1, %c1, %c1, %c0, %c0, %5, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, index, index, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %6 = memref.get_global @channel_2 : memref<1x1xi64>
    %7 = builtin.unrealized_conversion_cast %6 : memref<1x1xi64> to memref<1x1xi64>
    %8 = builtin.unrealized_conversion_cast %alloc_0 : memref<32x32xi32> to memref<?x?xi32>
    // put %alloc_0 into channel_2 
    call @air_channel_put_M0D2I64_I64_I64_I64_I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%7, %c1, %c1, %c1, %c1, %c0, %c0,%8,%c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, index, index, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %token = async.execute {
      %alloc_2 = memref.alloc() : memref<32x32xi32>
      %alloc_3 = memref.alloc() : memref<32x32xi32>
      %alloc_4 = memref.alloc() : memref<32x32xi32>
      %24 = memref.get_global @channel_0 : memref<1x1xi64>
      %25 = builtin.unrealized_conversion_cast %24 : memref<1x1xi64> to memref<1x1xi64>
      %26 = builtin.unrealized_conversion_cast %alloc_2 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0D2I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%25, %c0, %c0, %26, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      %27 = memref.get_global @channel_1 : memref<1x1xi64>
      %28 = builtin.unrealized_conversion_cast %27 : memref<1x1xi64> to memref<1x1xi64>
      %29 = builtin.unrealized_conversion_cast %alloc_3 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0D2I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%28, %c0, %c0, %29, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      %30 = memref.get_global @channel_2 : memref<1x1xi64>
      %31 = builtin.unrealized_conversion_cast %30 : memref<1x1xi64> to memref<1x1xi64>
      %32 = builtin.unrealized_conversion_cast %alloc_4 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0D2I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%31, %c0, %c0, %32, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      linalg.matmul ins(%alloc_2, %alloc_3 : memref<32x32xi32>, memref<32x32xi32>) outs(%alloc_4 : memref<32x32xi32>)
      %33 = memref.get_global @channel_3 : memref<1x1xi64>
      %34 = builtin.unrealized_conversion_cast %33 : memref<1x1xi64> to memref<1x1xi64>
      %35 = builtin.unrealized_conversion_cast %alloc_4 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_put_M0D2I64_I64_I64_I64_I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%34, %c1, %c1, %c1, %c1, %c0,%c0, %35, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, index, index, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      memref.dealloc %alloc_2 : memref<32x32xi32>
      memref.dealloc %alloc_3 : memref<32x32xi32>
      memref.dealloc %alloc_4 : memref<32x32xi32>
      async.yield
    }
    async.await %token : !async.token
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32> // result of second mm
    memref.copy %alloc, %alloc_1 : memref<32x32xi32> to memref<32x32xi32> // zero init

    // get %alloc_0 from channel_3
    %chn3 = memref.get_global @channel_3 : memref<1x1xi64>
    %chn3_cast = builtin.unrealized_conversion_cast %chn3 : memref<1x1xi64> to memref<1x1xi64>
    %alloc_0_cast = builtin.unrealized_conversion_cast %alloc_0 : memref<32x32xi32> to memref<?x?xi32>
    call @air_channel_get_M0D2I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%chn3_cast, %c0, %c0, %alloc_0_cast, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()

    %12 = memref.get_global @channel_4 : memref<1x1xi64>
    %13 = builtin.unrealized_conversion_cast %12 : memref<1x1xi64> to memref<1x1xi64>
    %14 = builtin.unrealized_conversion_cast %alloc_0 : memref<32x32xi32> to memref<?x?xi32>
    // put %alloc_0 into channel_4
    call @air_channel_put_M0D2I64_I64_I64_I64_I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%13, %c1, %c1, %c1, %c1, %c0,%c0,%14, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, index, index, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %15 = memref.get_global @channel_5 : memref<1x1xi64>
    %16 = builtin.unrealized_conversion_cast %15 : memref<1x1xi64> to memref<1x1xi64>
    %17 = builtin.unrealized_conversion_cast %arg2 : memref<32x32xi32> to memref<?x?xi32>
    // put %arg2 into channel_5
    call @air_channel_put_M0D2I64_I64_I64_I64_I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%16, %c1, %c1, %c1, %c1, %c0,%c0, %17,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, index, index, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %18 = memref.get_global @channel_6 : memref<1x1xi64>
    %19 = builtin.unrealized_conversion_cast %18 : memref<1x1xi64> to memref<1x1xi64>
    %20 = builtin.unrealized_conversion_cast %alloc_1 : memref<32x32xi32> to memref<?x?xi32>
    // put %alloc_1 into channel_6
    call @air_channel_put_M0D2I64_I64_I64_I64_I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%19, %c1, %c1, %c1, %c1, %c0,%c0, %20,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, index, index, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    %token_0 = async.execute {
      %alloc_2 = memref.alloc() : memref<32x32xi32>
      %alloc_3 = memref.alloc() : memref<32x32xi32>
      %alloc_4 = memref.alloc() : memref<32x32xi32>
      %24 = memref.get_global @channel_4 : memref<1x1xi64>
      %25 = builtin.unrealized_conversion_cast %24 : memref<1x1xi64> to memref<1x1xi64>
      %26 = builtin.unrealized_conversion_cast %alloc_2 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0D2I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%25, %c0, %c0,  %26, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      %27 = memref.get_global @channel_5 : memref<1x1xi64>
      %28 = builtin.unrealized_conversion_cast %27 : memref<1x1xi64> to memref<1x1xi64>
      %29 = builtin.unrealized_conversion_cast %alloc_3 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0D2I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%28, %c0, %c0,  %29, %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      %30 = memref.get_global @channel_6 : memref<1x1xi64>
      %31 = builtin.unrealized_conversion_cast %30 : memref<1x1xi64> to memref<1x1xi64>
      %32 = builtin.unrealized_conversion_cast %alloc_4 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_get_M0D2I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%31, %c0, %c0,  %32,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      linalg.matmul ins(%alloc_2, %alloc_3 : memref<32x32xi32>, memref<32x32xi32>) outs(%alloc_4 : memref<32x32xi32>)
      %33 = memref.get_global @channel_7 : memref<1x1xi64>
      %34 = builtin.unrealized_conversion_cast %33 : memref<1x1xi64> to memref<1x1xi64>
      %35 = builtin.unrealized_conversion_cast %alloc_4 : memref<32x32xi32> to memref<?x?xi32>
      func.call @air_channel_put_M0D2I64_I64_I64_I64_I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%34, %c1, %c1, %c1, %c1, %c0,%c0, %35,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, index, index, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
      memref.dealloc %alloc_2 : memref<32x32xi32>
      memref.dealloc %alloc_3 : memref<32x32xi32>
      memref.dealloc %alloc_4 : memref<32x32xi32>
      async.yield
    }
    async.await %token_0 : !async.token
    // get %alloc_1 from channel_7
    %chn7 = memref.get_global @channel_7 : memref<1x1xi64>
    %chn7_cast = builtin.unrealized_conversion_cast %chn7 : memref<1x1xi64> to memref<1x1xi64>
    %alloc_1_cast = builtin.unrealized_conversion_cast %alloc_1 : memref<32x32xi32> to memref<?x?xi32>
    call @air_channel_get_M0D2I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(%chn7_cast,%c0, %c0,  %alloc_1_cast,  %c0, %c0, %c32, %c32, %c32, %c1) : (memref<1x1xi64>, index, index, memref<?x?xi32>, index, index, index, index, index, index) -> ()
    memref.copy %alloc_1, %arg3 : memref<32x32xi32> to memref<32x32xi32>
    return
  }
  func.func private @air_channel_put_M0D2I64_I64_I64_I64_I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<1x1xi64>, index, index, index, index, index, index, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
  func.func private @air_channel_get_M0D2I64_I64_I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<1x1xi64>, index, index, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
}

