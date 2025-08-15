# ./python/spensor/spensor/passes/append_constant.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.dialects import func
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)

import spensor.utils.spensor_global_analysis as spensor_global


class FuncAddConstantPattern(RewritePattern):
    """
    Insert all arith.ConstantOp in spensor_global.index_to_constant_op
    into the beginning of the matched function.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        block = op.body.block
        first_op = block.first_op
        assert first_op is not None
        block.insert_ops_before(
            list(spensor_global.index_to_constant_op.values()), first_op
        )
        rewriter.has_done_action = True


class AppendConstant(ModulePass):
    name = "append_constant"

    def apply(self, ctx: Context, op: ModuleOp):
        constant_walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([FuncAddConstantPattern()]),
            walk_reverse=False,
            apply_recursively=False,
        )
        constant_walker.rewrite_module(op)
