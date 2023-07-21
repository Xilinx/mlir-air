//===- channel_op_lowering.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//


// RUN: cd %S; make channel.async.o; mv channel.async.o %T/
// RUN: %CLANG %S/main.cpp -O2 -std=c++17 %airhost_libs -c -o %T/main.o
// RUN: %CLANG %airhost_libs %mlir_async_lib -o %T/test.exe %T/main.o %T/channel.async.o
// RUN: %T/test.exe
// RUN: make clean

air.channel @channel_0 [1]
func.func @forward(%arg0 : memref<16x16xi32>, %arg1 : memref<16x16xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  air.channel.put @channel_0[] (%arg0[%c0, %c0] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
  air.channel.get @channel_0[] (%arg1[%c8, %c8] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
  return
}

