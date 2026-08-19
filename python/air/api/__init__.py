# ./python/air/api/__init__.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A high-level Python DSL for AIR.

``air.api`` sits above ``air.dialects`` and emits ordinary AIR IR: the module a
traced program produces is the same kind of module ``@module_builder`` builds by
hand, and it takes the same ``XRTBackend`` pipeline. Structural declarations live
here; imperative memory operations live in ``air.api.ops``::

    from air import api as air
    import air.api.ops
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
                    ...

Scope of this version: elementwise kernels over a 1-D or 2-D herd -- tensors,
L1 allocation, DMA in and out, and elementwise arithmetic on whole tiles.
Everything outside that raises ``NotImplementedError`` at the point of use.
Nothing degrades quietly into a kernel that runs and returns wrong numbers.
"""

from ._compile import CompiledKernel, LaunchContext, compile, launch
from ._trace import HerdContext, Scope, Symbol, alloc, herd, symbol, tensor, wait
from ._value import Buffer, Tensor, TensorSlice, Token
from .types import DType, bf16, f16, f32, i8, i16, i32

__all__ = [
    # launch hierarchy
    "launch",
    "herd",
    # declarations
    "tensor",
    "alloc",
    "symbol",
    # control flow
    "wait",
    # compilation
    "compile",
    # types
    "DType",
    "bf16",
    "f16",
    "f32",
    "i8",
    "i16",
    "i32",
    # objects surfaced for isinstance checks and typing
    "LaunchContext",
    "HerdContext",
    "CompiledKernel",
    "Buffer",
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
segment = _unimplemented("segment", "air.segment emission and L2 allocation")
BlockType = _unimplemented("BlockType", "block floating-point types")
Field = _unimplemented("Field", "block floating-point types")
Scratchpad = _unimplemented("Scratchpad", "fabric property gating")
Cascade = _unimplemented("Cascade", "cascade interconnect support")
Adjacency = _unimplemented("Adjacency", "placement constraints")
Broadcast = _unimplemented("Broadcast", "broadcast channel support")
CacheDomain = _unimplemented("CacheDomain", "GPU/XCD cache domains")
Disjoint = _unimplemented("Disjoint", "placement constraints")
Fabric = _unimplemented("Fabric", "fabric descriptors")
requires = _unimplemented("requires", "body variant capability gating")
jit = _unimplemented("jit", "function-level tracing")
