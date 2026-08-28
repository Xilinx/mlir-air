# ./python/air/api/__init__.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A high-level Python DSL for AIR.

``air.api`` sits above ``air.dialects`` and emits ordinary AIR IR: the module a
traced program produces is the same kind of module ``@module_builder`` builds by
hand, and it takes the same ``XRTBackend`` pipeline. Structural declarations live
here; memory transfers and elementwise compute live in ``ops``, which this
package re-exports, so one import reaches the whole DSL::

    from air import api as air
    from air.api.types import bf16

    A = air.tensor([M, N], bf16)
    B = air.tensor([M, N], bf16)
    C = air.tensor([M, N], bf16)

    with air.launch() as launch:
        @launch.body
        def _():
            with air.herd(product(range(0, M, tm), range(0, N, tn))) as h:
                @h.body
                def _(tx, ty):
                    a = air.alloc([tm, tn], bf16, scope=h.private())
                    air.ops.load(a, A[...])
                    ...

Scope of this version: kernels over a 1-D or 2-D herd -- tensors, allocation in
L1 (a core) and L2 (a memtile, via ``air.segment``), DMA between any two levels,
elementwise arithmetic on whole L1 tiles, an ``air.sequential`` loop, and
``ops.dot``, a contraction that dispatches on operand rank onto ``linalg.dot`` /
``vecmat`` / ``matvec`` / ``matmul``. Everything outside that raises
``NotImplementedError`` at the point of use. Nothing degrades quietly into a
kernel that runs and returns wrong numbers.

Element types are ``bf16 f16 f32``, ``i8 i16 i32`` and ``ui8 ui16 ui32``. The
unsigned three are *movable but not computable*: MLIR spells signedness in the
operation rather than the type, so the whole ``arith`` dialect -- and every
named ``linalg`` contraction built out of it -- takes signless operands, and a
``ui8`` one does not verify. An unsigned tile can therefore be allocated, moved
by ``ops.load``/``ops.store`` and ``air.channel``, copied elementwise, and
handed to an ``air.extern`` kernel, which is what a ``uint8`` example in this
tree needs; arithmetic on one raises at the call site naming the signed type to
declare instead.

``launch``, ``segment`` and ``herd`` are independent levels: a kernel that needs
no staging writes a herd on its own, and one that does wraps it in a segment.
Giving ``air.segment`` an iteration space makes it the *launch* grid, one segment
instance per point -- which is where outer tiling has to go, because the L2
staging buffers are refilled per point. Indexing a buffer with a partial
subscript (``staged[tx, 0:n, :]``) names a DMA region for
``ops.load``/``ops.store``; ``buf[:]`` in an expression is an elementwise read,
and is rejected on L2, which has no compute core.

For the AIE2 matmul intrinsic a tile has to be laid out in blocks, so that the
block the instruction consumes is contiguous. That is a *walk*, not a memref
layout -- the buffer stays contiguous either way -- so it is written the way
numpy writes one, with ``reshape`` and ``transpose`` on the region being moved::

    m, k = 4, 8
    a = air.alloc([1, 1, tile_k // k, tile_m // m, m, k], bf16, scope=h.private())
    ops.load(a, l2_a[rows, cols]
                 .reshape(1, 1, tile_m // m, m, tile_k // k, k)
                 .transpose(0, 1, 4, 2, 3, 5))    # the DMA performs the pack
    ops.dot(a, b, acc=acc)                        # rank 6: block_matmul

``reshape`` re-describes a region at a different rank and ``transpose`` permutes
its axes; both are views, and both raise rather than silently copying, since a
copy here would be a hidden L2 transfer. ``<segment>.shared()`` allocates L1
with the segment's lifetime, for an accumulator carried across a reduction loop
at segment scope; its leading dimensions are the herd's, one per axis, which is
what makes each core's slab well defined. Zero it with ``ops.fill(acc, 0.0)``.

``air.sequential`` is named for what ``scf.for`` guarantees -- its trips are
ordered in time on one core -- as against the herd's grid, which is spatial.
``air.parallel`` is its unordered counterpart, and the two are not
interchangeable at segment scope. Staging that fans a memtile buffer out to a
row of cores has to be parallel twice over: the trip index names one slot of a
channel bundle, which ``air-place-herds`` refuses to take from a temporal loop,
and the trips share one set of buffer descriptors, so writing them out as a
Python ``for`` turns one fan-out into that many independent DMAs. See
``herd_dataflow``, where the unrolled form does not fit on npu1 at all.

The DSL has **two conditionals**, and they are not interchangeable.
``ops.select(c, a, b)`` is branchless: ``c`` compares buffer *data*, the
decision is per element, both sides are evaluated, and the arms are values.
``ops.branch(c)`` is a real branch: ``c`` compares *index* expressions
(``tx == 0``), the decision is per core, one side runs, and the arms are
statements -- which is what a channel put or a DMA has to be. They are the two
halves of if-conversion. Reaching for either with the other's condition raises
and the message names the one you wanted; ``_cond.py`` has the full table.

``ops.branch`` has to be a region rather than a Python ``if`` because the herd
body is traced once for the whole herd: a comparison against a coordinate has no
value at trace time, so ``bool()`` on one raises rather than picking a branch for
every core. The else is ``with <branch>.otherwise():``. There is no
``and``/``or``, and conjunction is nesting.

One hardware caveat that the DSL cannot see and so cannot raise on: an
*unstaged* K reduction on a 2-D herd is wrong past a single trip. Both operands
are then broadcast, and the resulting packet flows share one shim channel whose
coalesced buffer descriptors do not line up with the per-trip ring on the
receiving tile. Staging through ``air.segment`` avoids it entirely, which is what
every hand-written matmul in the tree does; a 1-D herd is also unaffected.
"""

from . import ops
from ._channel import Channel, channel
from ._compile import CompiledKernel, LaunchContext, compile, launch
from ._extern import ExternKernel, extern
from ._loop import parallel, sequential
from ._trace import (
    HerdContext,
    Scope,
    SegmentContext,
    Symbol,
    alloc,
    dealloc,
    herd,
    resolve_target,
    segment,
    symbol,
    tensor,
    wait,
)
from ._value import Buffer, BufferSlice, Tensor, TensorSlice, Token
from .types import DType, bf16, f16, f32, i4, i8, i16, i32, ui8, ui16, ui32

__all__ = [
    # operations
    "ops",
    # launch hierarchy
    "launch",
    "segment",
    "herd",
    # declarations
    "tensor",
    "alloc",
    "dealloc",
    "symbol",
    "extern",
    "channel",
    # control flow
    "sequential",
    "wait",
    # micro-tiled (packed) layouts
    # compilation
    "compile",
    # the NPU generation --target resolves to, for a design that has to branch
    # on it before tracing (an object file to link, or an element type)
    "resolve_target",
    # types
    "DType",
    "bf16",
    "f16",
    "f32",
    "i4",
    "i8",
    "i16",
    "i32",
    "ui8",
    "ui16",
    "ui32",
    # objects surfaced for isinstance checks and typing
    "LaunchContext",
    "SegmentContext",
    "HerdContext",
    "ExternKernel",
    "Channel",
    "CompiledKernel",
    "Buffer",
    "BufferSlice",
    "Tensor",
    "TensorSlice",
    "Token",
    "Scope",
    "Symbol",
]


def _unimplemented(name, needs):
    def stub(*args, **kwargs):
        raise NotImplementedError(
            f"air.api.{name} is not implemented yet (needs {needs})"
        )

    stub.__name__ = name
    return stub


# Names from the wider API proposal that this version does not lower. They are
# present, and they raise -- an accepted-but-ignored capability gate is worse
# than an absent one, because the kernel still compiles and still runs.
BlockType = _unimplemented("BlockType", "block floating-point types")
Field = _unimplemented("Field", "block floating-point types")
Scratchpad = _unimplemented("Scratchpad", "fabric property gating")
Cascade = _unimplemented("Cascade", "cascade interconnect support")
Adjacency = _unimplemented("Adjacency", "placement constraints")
# Broadcasting is a property of the channel, not a free-standing capability:
# air.channel(size=[...], broadcast_shape=[...]) is the spelling, matching the
# broadcast_shape attribute on air.channel itself.
Broadcast = _unimplemented(
    "Broadcast",
    "a standalone broadcast capability; pass broadcast_shape= to air.channel() "
    "instead",
)
CacheDomain = _unimplemented("CacheDomain", "GPU/XCD cache domains")
Disjoint = _unimplemented("Disjoint", "placement constraints")
Fabric = _unimplemented("Fabric", "fabric descriptors")
requires = _unimplemented("requires", "body variant capability gating")
jit = _unimplemented("jit", "function-level tracing")
