# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

from air.ir import *
from air.dialects.air import *
from air.dialects.arith import AddIOp
from air.dialects.func import FuncOp, ReturnOp


def constructAndPrintInFunc(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            ftype = FunctionType.get([], [])
            fop = FuncOp(f.__name__, ftype)
            bb = fop.add_entry_block()
            with InsertionPoint(bb):
                f()
                ReturnOp([])
    print(module)


# CHECK-LABEL: TEST: test_herd
# CHECK: air.launch
# CHECK-NEXT: air.segment @seg
# CHECK: %[[C3:.*]] = arith.constant 3 : index
# CHECK: %[[C2:.*]] = arith.constant 2 : index
# CHECK: air.herd @hrd  tile (%[[X:.*]], %[[Y:.*]]) in (%[[SX:.*]]=%[[C2]], %[[SY:.*]]=%[[C3]])
# CHECK: arith.addi %[[X]], %[[Y]]
# CHECK: arith.addi %[[SX]], %[[SY]]
@constructAndPrintInFunc
def test_herd():
    @launch
    def launch_body():
        @segment(name="seg")
        def segment_body():
            idx_ty = IndexType.get()
            sz = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 3))

            @herd(name="hrd", sizes=[2, sz])
            def herd_body(x, y, sx, sy):
                AddIOp(x, y)
                AddIOp(sx, sy)
                HerdTerminatorOp()
                return

            SegmentTerminatorOp()

        LaunchTerminatorOp()
