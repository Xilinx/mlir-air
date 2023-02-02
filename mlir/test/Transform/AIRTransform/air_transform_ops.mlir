// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// RUN: air-opt %s

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
  %matmul_1, %loops:2 = transform.air.linalg_tile %matmul [64, 64, 0]
}