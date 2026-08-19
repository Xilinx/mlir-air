# ./python/air/api/ops.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Imperative memory operations.

Import as::

    import air.api.ops

Every implemented op returns a :class:`~air.api._value.Token`. AIR builds its
real asynchronous dependency graph from program order in the ``air-dependency``
pass, so a v1 token carries no SSA value -- it exists so that ``dependency=``
can be type-checked instead of silently ignored.

Elementwise compute ops (``maximum``, ``minimum``, ``relu``, ``tanh``, and the
``sigmoid``/``silu``/``gelu`` compositions built on them) build lazy expression
nodes instead, and return a :class:`~air.api._value.BufferExpr`.

The remaining compute ops from the wider API proposal (``dot``, ``reduce``,
``stack``, ``dequant``, ``atomic_add``) are not implemented. They raise
rather than returning a plausible-looking placeholder: a DSL that accepts an op
it cannot lower produces a kernel that runs and is silently wrong.
"""

from ._value import Buffer, BufferExpr, TensorSlice, Token

__all__ = [
    "load",
    "store",
    "copy",
    "maximum",
    "minimum",
    "relu",
    "tanh",
    "sigmoid",
    "silu",
    "gelu",
]


def _check_dependency(dependency):
    if dependency is None:
        return
    deps = dependency if isinstance(dependency, (list, tuple)) else [dependency]
    for d in deps:
        if not isinstance(d, Token):
            raise TypeError(
                f"dependency= expects a Token (or list of them), got "
                f"{type(d).__name__}"
            )


def _check_padding(pad_before, pad_after):
    if pad_before is None and pad_after is None:
        return
    raise NotImplementedError(
        "pad_before/pad_after are not supported by air.api yet; the underlying "
        "air.dma_memcpy_nd accepts them, but the DSL does not validate them"
    )


def _check_transfer(buf, sl, direction):
    if not isinstance(buf, Buffer):
        raise TypeError(
            f"air.api.ops.{direction} expects an L1 buffer from air.alloc(), got "
            f"{type(buf).__name__}"
        )
    if not isinstance(sl, TensorSlice):
        raise TypeError(
            f"air.api.ops.{direction} expects a tensor slice such as A[i:i+n], "
            f"got {type(sl).__name__}"
        )
    if buf.value is None:
        raise RuntimeError("buffer used before allocation")
    if tuple(sl.sizes) != buf.shape:
        raise ValueError(
            f"transfer shape mismatch: buffer is {buf.shape} but the tensor "
            f"slice is {tuple(sl.sizes)}"
        )
    if sl.dtype is not buf.dtype:
        raise ValueError(
            f"transfer dtype mismatch: buffer is {buf.dtype} but the tensor is "
            f"{sl.dtype}"
        )


def load(dst, src_slice, pad_before=None, pad_after=None, dependency=None):
    """Copy a tile from a global tensor into an L1 buffer (L3 -> L1)."""
    from air.dialects.air import dma_memcpy_nd

    _check_dependency(dependency)
    _check_padding(pad_before, pad_after)
    _check_transfer(dst, src_slice, "load")

    op = dma_memcpy_nd(
        dst.value,
        src_slice.tensor.value,
        src_offsets=src_slice.materialize_offsets(),
        src_sizes=list(src_slice.sizes),
        src_strides=list(src_slice.strides),
    )
    return Token(op)


def store(src, dst_slice, pad_before=None, pad_after=None, dependency=None):
    """Copy an L1 buffer back into a global tensor (L1 -> L3)."""
    from air.dialects.air import dma_memcpy_nd

    _check_dependency(dependency)
    _check_padding(pad_before, pad_after)
    _check_transfer(src, dst_slice, "store")

    # Being the destination of a store is what makes a tensor an output, which
    # in turn fixes the kernel's calling convention (inputs first, then outputs).
    dst_slice.tensor.is_output = True

    op = dma_memcpy_nd(
        dst_slice.tensor.value,
        src.value,
        dst_offsets=dst_slice.materialize_offsets(),
        dst_sizes=list(dst_slice.sizes),
        dst_strides=list(dst_slice.strides),
    )
    return Token(op)


def copy(src_slice, dst_slice, pad_before=None, pad_after=None, dependency=None):
    """Tensor-to-tensor copy. Not implemented."""
    raise NotImplementedError(
        "air.api.ops.copy (tensor-to-tensor) is not implemented; use load() into "
        "an L1 buffer followed by store()"
    )


# ---------------------------------------------------------------------------
# Elementwise compute
#
# These build lazy expression nodes rather than emitting anything, exactly like
# the `+ - * /` operators on a buffer slice. Nothing reaches the IR until the
# tree is assigned into a buffer (`out[:] = ...`), so a whole expression still
# lowers as one vectorised loop.
# ---------------------------------------------------------------------------


def _elementwise(name, key, a, b):
    for operand, pos in ((a, "first"), (b, "second")):
        if not isinstance(operand, (Buffer, BufferExpr, int, float)):
            raise TypeError(
                f"air.api.ops.{name} expects a buffer slice or a numeric scalar "
                f"as its {pos} argument, got {type(operand).__name__}"
            )
    a, b = BufferExpr.coerce(a), BufferExpr.coerce(b)
    if not a.leaves() and not b.leaves():
        raise ValueError(
            f"air.api.ops.{name} needs at least one buffer operand; both "
            "arguments are scalars, which the emitter cannot shape"
        )
    return BufferExpr("binary", op=key, args=(a, b))


def maximum(a, b):
    """Elementwise max. Lowers to arith.maximumf (float) / arith.maxsi (int)."""
    return _elementwise("maximum", "max", a, b)


def minimum(a, b):
    """Elementwise min. Lowers to arith.minimumf (float) / arith.minsi (int)."""
    return _elementwise("minimum", "min", a, b)


def relu(x):
    """max(x, 0), the composition the hand-written relu kernel emits.

    The zero takes its Python type from the operand's dtype: an integer buffer
    lowers through ``_INT_OPS``, and building an integer ``arith.constant`` from
    a Python float fails with "expected floating point type".
    """
    expr = BufferExpr.coerce(x)
    leaves = expr.leaves()
    if not leaves:
        raise ValueError("air.api.ops.relu needs a buffer operand, got a scalar")
    return maximum(expr, 0.0 if leaves[0].dtype.is_float else 0)


def _unary(name, x):
    if not isinstance(x, (Buffer, BufferExpr)):
        raise TypeError(
            f"air.api.ops.{name} expects a buffer slice, got {type(x).__name__}"
        )
    expr = BufferExpr.coerce(x)
    if not expr.leaves():
        raise ValueError(f"air.api.ops.{name} needs a buffer operand, got a scalar")
    return BufferExpr("unary", op=name, args=(expr,))


def tanh(x):
    """Elementwise hyperbolic tangent. Lowers to math.tanh. Float only."""
    return _unary("tanh", x)


# The three activations below are compositions, not new primitives. Each is
# written the way the hand-written kernel it replaces wrote it -- in particular
# via tanh rather than exp, which keeps them clear of two AIE2 limitations at
# once: there is no vector division on bf16, and exp would need one.


def sigmoid(x):
    """0.5 * (tanh(x/2) + 1), the logistic function without a division."""
    return 0.5 * (tanh(0.5 * BufferExpr.coerce(x)) + 1.0)


def silu(x):
    """x * sigmoid(x), also known as swish."""
    expr = BufferExpr.coerce(x)
    return expr * sigmoid(expr)


# tanh approximation of the Gaussian error linear unit: the constants are
# sqrt(2/pi) and the 0.044715 cubic term from Hendrycks & Gimpel.
GELU_SQRT_2_OVER_PI = 0.7978845608
GELU_BETA = 0.044715


def gelu(x):
    """0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))."""
    expr = BufferExpr.coerce(x)
    inner = GELU_SQRT_2_OVER_PI * (expr + GELU_BETA * (expr * expr * expr))
    return 0.5 * expr * (tanh(inner) + 1.0)


def _unimplemented(name, needs):
    def stub(*args, **kwargs):
        raise NotImplementedError(
            f"air.api.ops.{name} is not implemented yet (needs {needs})"
        )

    stub.__name__ = name
    return stub


# Named so that a program using them fails loudly at the call site rather than
# at compile time with a confusing IR error.
dot = _unimplemented("dot", "linalg/vector.contract lowering")
# math.exp exists in the bindings, but nothing needs it yet and it has not
# been checked for an aievec lowering -- untested surface is worse than none.
exp = _unimplemented("exp", "a checked aievec lowering; use ops.tanh, which has one")
reduce = _unimplemented("reduce", "a reduction emitter")
stack = _unimplemented("stack", "multi-buffer concatenation")
dequant = _unimplemented("dequant", "BlockType support")
atomic_add = _unimplemented("atomic_add", "CacheDomain support")
