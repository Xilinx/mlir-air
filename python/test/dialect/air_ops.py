# ./python/test/dialect/air_ops.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

import air
from air.mlir.ir import *
from air.dialects import air as airdialect
from air.mlir.dialects import arith
from air.mlir.dialects import func

def constructAndPrintInFunc(f):
  print("\nTEST:", f.__name__)
  with Context() as ctx, Location.unknown():
    airdialect.register_dialect(ctx)
    module = Module.create()
    with InsertionPoint(module.body):
      ftype = FunctionType.get(
          [IntegerType.get_signless(32),
           IntegerType.get_signless(32)], [])
      fop = func.FuncOp(f.__name__, ftype)
      bb = fop.add_entry_block()
      with InsertionPoint(bb):
        f()
        func.ReturnOp([])
  module.operation.verify()
  print(module)

# CHECK-LABEL: TEST: launchOp
# CHECK: air.launch @pyLaunch () in () {
# CHECK:   %{{.*}} = arith.constant 1 : index
# CHECK:   air.launch_terminator
@constructAndPrintInFunc
def launchOp():
  l = airdialect.LaunchOp("pyLaunch")
  with InsertionPoint(l.body):
    idx_ty = IndexType.get()
    arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1))
    airdialect.LaunchTerminatorOp()

# CHECK-LABEL: TEST: segmentOp
# CHECK: air.segment @pySegment {
# CHECK:   %{{.*}} = arith.constant 1 : index
# CHECK:   air.segment_terminator
@constructAndPrintInFunc
def segmentOp():
  P = airdialect.SegmentOp("pySegment")
  with InsertionPoint(P.body):
    idx_ty = IndexType.get()
    arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1))
    airdialect.SegmentTerminatorOp()

# CHECK-LABEL: TEST: herdOp
# CHECK: %[[C0:.*]] = arith.constant 2 : index
# CHECK: %[[C1:.*]] = arith.constant 2 : index
# CHECK: air.herd @pyHerd tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%[[C0]], %{{.*}}=%[[C1]]) {
# CHECK:   %{{.*}} = arith.constant 1 : index
# CHECK:   air.herd_terminator
@constructAndPrintInFunc
def herdOp():
  idx_ty = IndexType.get()
  size_x = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 2))
  size_y = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 2))
  sizes = [size_x.result, size_y.result]
  H = airdialect.HerdOp("pyHerd", sizes, [])
  with InsertionPoint(H.body):
    arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 1))
    airdialect.HerdTerminatorOp()

# CHECK-LABEL: TEST: waitallOp
# CHECK: %0 = air.wait_all async
# CHECK: air.wait_all [%0]
@constructAndPrintInFunc
def waitallOp():
  token_type = airdialect.AsyncTokenType.get()
  e = airdialect.WaitAllOp(token_type,[]).result
  airdialect.WaitAllOp(None,[e])
