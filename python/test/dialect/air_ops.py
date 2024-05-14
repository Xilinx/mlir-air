# ./python/test/dialect/air_ops.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

from air.ir import *
from air.dialects.air import *
from air.dialects.arith import ConstantOp
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

# CHECK-LABEL: TEST: launchOp
# CHECK: air.launch @pyLaunch () in () {
# CHECK:   %{{.*}} = arith.constant 1 : index
# CHECK:   air.launch_terminator
@constructAndPrintInFunc
def launchOp():
  l = Launch("pyLaunch")
  with InsertionPoint(l.body.blocks[0]):
    idx_ty = IndexType.get()
    ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1))
    LaunchTerminatorOp()

# CHECK-LABEL: TEST: segmentOp
# CHECK: air.segment @pySegment {
# CHECK:   %{{.*}} = arith.constant 1 : index
# CHECK:   air.segment_terminator
@constructAndPrintInFunc
def segmentOp():
  s = Segment("pySegment")
  with InsertionPoint(s.body.blocks[0]):
    idx_ty = IndexType.get()
    ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1))
    SegmentTerminatorOp()

# CHECK-LABEL: TEST: herdOp
# CHECK: %[[C0:.*]] = arith.constant 2 : index
# CHECK: %[[C1:.*]] = arith.constant 2 : index
# CHECK: air.herd @pyHerd tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[C0]], %{{.*}}=%[[C1]]) {
# CHECK:   %{{.*}} = arith.constant 1 : index
# CHECK:   air.herd_terminator
@constructAndPrintInFunc
def herdOp():
  H = Herd("pyHerd", [2,2])
  with InsertionPoint(H.body.blocks[0]):
    idx_ty = IndexType.get()
    idx_ty = IndexType.get()
    ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1))
    HerdTerminatorOp()

# CHECK-LABEL: TEST: waitallOp
# CHECK: %0 = air.wait_all async
# CHECK: air.wait_all [%0]
@constructAndPrintInFunc
def waitallOp():
  token_type = AsyncTokenType.get()
  e = WaitAllOp(token_type,[]).result
  WaitAllOp(None,[e])

# CHECK-LABEL: TEST: enum
# CHECK-SAME: L1 L2
print("TEST: enum", MemorySpace.L1, MemorySpace.L2)
