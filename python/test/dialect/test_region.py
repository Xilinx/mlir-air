# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

from air.ir import *
from air.dialects.air import *
from air.dialects.arith import AddIOp


def constructAndPrintInFunc(f):
    print("\nTEST:", f.__name__)
    print(f())


# CHECK-LABEL: TEST: test_herd
# CHECK: air.launch
# CHECK-NEXT: air.segment @seg
# CHECK: %[[C3:.*]] = arith.constant 3 : index
# CHECK: %[[C2:.*]] = arith.constant 2 : index
# CHECK: air.herd @hrd  tile (%[[X:.*]], %[[Y:.*]]) in (%[[SX:.*]]=%[[C2]], %[[SY:.*]]=%[[C3]])
# CHECK: arith.addi %[[X]], %[[Y]]
# CHECK: arith.addi %[[SX]], %[[SY]]
@constructAndPrintInFunc
@module_builder
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


# CHECK-LABEL: TEST: test_rank
# CHECK: air.rank @r (%[[RX:.*]]) in (%[[RSX:.*]]=%{{.*}})
# CHECK:   air.launch
# CHECK:     air.segment @seg
# CHECK:       air.herd @hrd
@constructAndPrintInFunc
@module_builder
def test_rank():
    @rank(name="r", sizes=[2])
    def rank_body(rx, sx):
        @launch
        def launch_body():
            @segment(name="seg")
            def segment_body():
                @herd(name="hrd", sizes=[2, 2])
                def herd_body(x, y, sx, sy):
                    AddIOp(x, y)


# Regression: attributes= kwarg on Launch/Segment/Herd must be attached to
# the underlying op (previously accepted but silently dropped).
# CHECK-LABEL: TEST: test_attributes_kwarg
# CHECK: air.launch
# CHECK-SAME: {air.shim_dma_tile_sizes = array<i64: 0>, launch_tag = "L"}
# CHECK: air.segment @seg
# CHECK-SAME: {segment_tag = "S"}
# CHECK: air.herd @hrd
# CHECK-SAME: {herd_tag = "H"}
@constructAndPrintInFunc
@module_builder
def test_attributes_kwarg():
    @launch(
        attributes={
            "air.shim_dma_tile_sizes": DenseI64ArrayAttr.get([0]),
            "launch_tag": StringAttr.get("L"),
        }
    )
    def launch_body():
        @segment(name="seg", attributes={"segment_tag": StringAttr.get("S")})
        def segment_body():
            @herd(
                name="hrd", sizes=[1, 1], attributes={"herd_tag": StringAttr.get("H")}
            )
            def herd_body(x, y, sx, sy):
                AddIOp(x, y)


# A body may leave out the ids and sizes. Taking one parameter per operand, it is
# passed only the operands; the explicit form above is unchanged.
# CHECK-LABEL: TEST: test_body_without_ids
# CHECK: air.launch (%{{.*}}) in (%{{.*}}=%{{.*}}) args(%[[LA:.*]]=%{{.*}}) : i32
# CHECK: air.segment @seg  unroll(%{{.*}}) in (%{{.*}}=%{{.*}}) args(%[[SA:.*]]=%[[LA]]) : i32
# CHECK: arith.addi %[[SA]], %[[SA]] : i32
# CHECK: air.herd @hrd  tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%{{.*}}, %{{.*}}=%{{.*}}) args(%[[A:.*]]=%[[SA]], %[[B:.*]]=%[[SA]]) : i32, i32
# CHECK: arith.addi %[[A]], %[[B]] : i32
# CHECK: air.herd @one  tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c1{{.*}}, %{{.*}}=%c1{{.*}})
@constructAndPrintInFunc
@module_builder
def test_body_without_ids():
    i32 = IntegerType.get_signless(32)
    c = arith.ConstantOp(i32, IntegerAttr.get(i32, 7))

    @launch(sizes=[2], operands=[c])
    def launch_body(a):
        @segment(name="seg", sizes=[2], operands=[a])
        def segment_body(a):
            AddIOp(a, a)

            @herd(name="hrd", sizes=[2, 2], operands=[a, a])
            def herd_body(x, y):
                AddIOp(x, y)

            @herd(name="one")
            def one_body():
                pass


# A body taking *args still gets every block argument.
# CHECK-LABEL: TEST: test_body_varargs
# CHECK: air.herd @hrd  tile (%[[X:.*]], %[[Y:.*]]) in
# CHECK: arith.addi %[[X]], %[[Y]] : index
@constructAndPrintInFunc
@module_builder
def test_body_varargs():
    @launch
    def launch_body():
        @segment(name="seg")
        def segment_body():
            @herd(name="hrd", sizes=[2, 2])
            def herd_body(*args):
                AddIOp(args[0], args[1])
