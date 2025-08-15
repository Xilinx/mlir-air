# ./python/spensor/spensor/passes/spensor_dce.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern
import spensor.dialects.spensor_dialect as spensor


class RemoveDeadPattern(RewritePattern):
    """
    Removes all DeclareMemoryOp because they don't return any values.
    """
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if isinstance(op, spensor.DeclareMemoryOp) or isinstance(
            op, spensor.GetMemoryOp
        ):
            rewriter.erase_matched_op()


class SpensorDeadCodeElimination(ModulePass):
    name = "spensor_dce"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(RemoveDeadPattern(), walk_reverse=True)
        walker.rewrite_module(op)
