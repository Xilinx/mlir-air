# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

from air.ir import *
from air.dialects import transform
from air.dialects import pdl
from air.dialects import air as airdialect
from air.dialects.air import module_builder


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: testLinalgTile
# CHECK: transform.sequence
# CHECK: transform.air.linalg_tile
@run
def testLinalgTile():

    @module_builder
    def build_module():
        sequence = transform.SequenceOp(
            transform.FailurePropagationMode.Propagate, [], pdl.OperationType.get()
        )
        with InsertionPoint(sequence.body):
            airdialect.LinalgTileOp(sequence.bodyTarget, sizes=[32, 32])
            transform.YieldOp()

    module = build_module()
    print(module)
