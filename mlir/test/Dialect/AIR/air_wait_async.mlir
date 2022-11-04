//===- air_wait_async.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func.func @wait
// CHECK: %[[V1:.*]] = air.wait_all async [{{.*}}, {{.*}}]
// CHECK: return %[[V1:.*]] : !air.async.token
func.func @wait() -> !air.async.token {
  air.wait_all
  %e0 = air.wait_all async
  %e1 = air.wait_all async [%e0]
  %e2 = air.wait_all async [%e0, %e1]
  return %e2 : !air.async.token
} 
