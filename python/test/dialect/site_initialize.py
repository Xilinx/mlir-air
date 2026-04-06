# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

# Verify that air.ir.Context() automatically registers AIR dialect ops
# (including transform.air.* extension ops) via _site_initialize_0.py.
# This must work WITHOUT importing air.dialects.air explicitly.

from air.ir import *

# CHECK-LABEL: TEST: transform_air_ops_registered
# CHECK: transform.named_sequence @test
# CHECK: transform.air.linalg_tile
print("\nTEST: transform_air_ops_registered")

ctx = Context()
ir_string = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @test(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.air.linalg_tile %matmul [32, 32]
    transform.yield
  }
}
"""
module = Module.parse(ir_string, context=ctx)
print(module)

# CHECK-LABEL: TEST: air_dialect_ops_registered
# CHECK: module
print("\nTEST: air_dialect_ops_registered")

ir_string2 = """
module {}
"""
module2 = Module.parse(ir_string2, context=ctx)
# Verify that the AIR dialect is loadable in the context.
ctx.dialects["air"]
print(module2)

print("\nPASSED")
# CHECK: PASSED
