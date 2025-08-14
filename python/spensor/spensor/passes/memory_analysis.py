# ./python/spensor/spensor/passes/memory_analysis.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from xdsl.dialects.builtin import (
    ModuleOp,
    StringAttr,
)
from xdsl.context import Context
from xdsl.passes import ModulePass
import spensor.dialects.spensor_dialect as spensor
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from spensor.utils.spensor_global_analysis import Memory
import spensor.utils.spensor_global_analysis as spensor_global


class DeclareMemoryPattern(RewritePattern):
    """
    Iterates all DeclareMemoryOp and updates spensor_global.memory_mapping
    make a dictionary from memory name -> Memory class
    """
    def __init__(self, memory_mapping: dict[StringAttr, Memory]):
        self.memory_mapping = memory_mapping
        super().__init__()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: spensor.DeclareMemoryOp, rewriter: PatternRewriter):
        if op.memory_name not in self.memory_mapping:
             # Collect allowed load and store operations
            allowing_load: set[StringAttr] = set()
            allowing_store: set[StringAttr] = set()
            for memory in op.load:
                allowing_load.add(memory)
            for memory in op.store:
                allowing_store.add(memory)
            memory_space = op.attributes.get("memory_space")

            # Create a new Memory instance and update the mapping
            self.memory_mapping[op.memory_name] = Memory(
                op.memory_name,
                op.memory_shape,
                allowing_load,
                allowing_store,
                memory_space,
            )
            rewriter.has_done_action = True


class MemoryAnalysis(ModulePass):
    name = "memory_analysis"
    memory_mapping: dict[StringAttr, Memory]

    def __init__(
        self,
    ):
        self.memory_mapping = {}

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([DeclareMemoryPattern(self.memory_mapping)])
        )
        walker.rewrite_module(op)
        # asserts all memory names in allowing_load/store are existing
        for val in self.memory_mapping.values():
            for s in val.allowing_load:
                assert s in self.memory_mapping
            for s in val.allowing_store:
                assert s in self.memory_mapping
        spensor_global.memory_mapping = self.memory_mapping
