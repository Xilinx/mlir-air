# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

from air.ir import *
from air.dialects import transform
from air.dialects import pdl
from air.dialects import air as airdialect

def run(f):
  with Context() as ctx, Location.unknown():
    module = Module.create()
    with InsertionPoint(module.body):
      print("\nTEST:", f.__name__)
      f()
    print(module)
  return f

# CHECK-LABEL: TEST: testLinalgTile
# CHECK: transform.sequence
# CHECK: transform.air.linalg_tile
@run
def testLinalgTile():
  sequence = transform.SequenceOp(transform.FailurePropagationMode.Propagate, [], pdl.OperationType.get())
  with InsertionPoint(sequence.body):
    airdialect.LinalgTileOp(sequence.bodyTarget, sizes=[32,32])
    transform.YieldOp()


