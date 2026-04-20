//===- help_output.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that the C++ aircc binary prints expected CLI options in --help.

// RUN: aircc --help 2>&1 | FileCheck %s

// CHECK: AIR Compiler Options:
// CHECK-DAG: --device
// CHECK-DAG: --output-format
// CHECK-DAG: --debug-ir
// CHECK-DAG: --omit-ping-pong-transform
// CHECK-DAG: --omit-while-true-loop
// CHECK-DAG: --trace-size
// CHECK-DAG: --bf16-emulation
