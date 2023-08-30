//===- channel_op_lowering.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//


// RUN: air-opt -o %T/channel.async.llvm.mlir %s -buffer-results-to-out-params -air-to-async -async-to-async-runtime -async-runtime-ref-counting -async-runtime-ref-counting-opt -convert-linalg-to-affine-loops -expand-strided-metadata -lower-affine -convert-scf-to-cf -convert-async-to-llvm -convert-memref-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -canonicalize -cse
// RUN: air-translate --mlir-to-llvmir %T/channel.async.llvm.mlir -o %T/channel.async.ll
// RUN: %CLANG %T/channel.async.ll -O2 -std=c++17 -c -o %T/channel.async.o
// RUN: %CLANG %S/main.cpp -O2 -std=c++17 %airhost_inc -c -o %T/main.o
// RUN: %CLANG %aircpu_lib %mlir_async_lib -o %T/test.exe %T/main.o %T/channel.async.o
// RUN: %ld_lib_path %T/test.exe

air.channel @channel_0 [1, 1]
func.func @forward(%arg0 : memref<16x16xi32>, %arg1 : memref<16x16xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  air.channel.put @channel_0[%c0, %c0] (%arg0[%c0, %c0] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
  air.channel.get @channel_0[%c0, %c0] (%arg1[%c8, %c8] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
  return
}

