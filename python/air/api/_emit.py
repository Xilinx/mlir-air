# ./python/air/api/_emit.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lowering of lazy elementwise expressions into a vectorised loop nest.

The emitted shape follows the hand-tuned kernel in
``programming_examples/eltwise_add/eltwise_add.py``: an ``scf.for`` nest over the
tile, with ``vector.transfer_read`` / ``arith.<op>`` / ``vector.transfer_write``
on the innermost dimension. Reads use a minor-identity permutation map, so a
rank-N L1 tile is read as a rank-1 vector of the innermost dimension and the map
folds away entirely in the printed IR.

This deliberately does not go through ``linalg.generic``: the aircc pipeline
runs ``convert-linalg-to-loops``, which scalarises it, and a scalarised
elementwise body is the documented cause of NPU timeouts on tiles this size.

When the innermost dimension is not a multiple of the vector width the emitter
falls back to a scalar ``memref.load``/``store`` loop rather than failing --
correctness first, with the width left to the caller to fix for speed.
"""

from air.ir import AffineDimExpr, AffineMap, AffineMapAttr, VectorType
from air.dialects import arith
from air.dialects import math as math_dialect
from air.dialects.memref import load as memref_load, store as memref_store
from air.dialects.scf import for_ as range_, yield_
from air.dialects.vector import broadcast, transfer_read, transfer_write

from .types import require_signless

__all__ = ["emit_elementwise"]

_FLOAT_OPS = {
    "add": arith.AddFOp,
    "sub": arith.SubFOp,
    "mul": arith.MulFOp,
    "div": arith.DivFOp,
    # maximumf/minimumf, not maxnumf/minnumf: these are what the hand-written
    # relu kernel uses, so they are the forms known to legalize on AIE2, and the
    # LLVM 24 bump regressed scalar f32 maxnumf specifically.
    "max": arith.MaximumFOp,
    "min": arith.MinimumFOp,
}

# Unary transcendentals. float only -- there is no integer tanh, and an integer
# buffer reaching one of these is a user error rather than something to coerce.
_FLOAT_UNARY_OPS = {
    "tanh": math_dialect.tanh,
}

_INT_OPS = {
    "add": arith.AddIOp,
    "sub": arith.SubIOp,
    "mul": arith.MulIOp,
    "div": arith.DivSIOp,
    "max": arith.MaxSIOp,
    "min": arith.MinSIOp,
}


def emit_elementwise(dst, expr):
    """Emit ``dst[:] = expr`` as a loop nest over ``dst``'s shape."""
    leaves = expr.leaves()
    # A bare scalar on the right-hand side is a fill (`acc[:] = 0.0`), which is
    # how an accumulator is zeroed before a K loop. It needs no leaves: the
    # destination supplies the shape.
    for leaf in leaves:
        if leaf.shape != dst.shape:
            raise ValueError(
                f"shape mismatch in elementwise assignment: destination has "
                f"shape {dst.shape} but operand has shape {leaf.shape}"
            )
        if leaf.dtype is not dst.dtype:
            raise ValueError(
                f"dtype mismatch in elementwise assignment: destination is "
                f"{dst.dtype} but operand is {leaf.dtype}"
            )
        if leaf.value is None:
            raise RuntimeError(
                "buffer used before allocation; air.alloc() must be called "
                "inside the herd body that uses it"
            )

    if dst.dtype.is_unsigned:
        # An unsigned tile can be *copied* elementwise but not computed on. The
        # copy holds because it emits memref.load/store and no arith op at all,
        # which is exactly the loop the hand-written uint8 examples spell out;
        # anything else -- an operator, or a broadcast constant -- reaches an
        # arith builder that rejects a signful operand. The vector path is not
        # available even for the copy: vector.transfer_read takes a padding
        # value, and that padding value is an arith.constant.
        if expr.kind != "buffer":
            require_signless(
                dst.dtype,
                "an elementwise operator or broadcast scalar (a plain copy, "
                "dst[:] = src[:], is)",
            )
        _emit_scalar(dst, expr, _INT_OPS)
        return

    shape = dst.shape
    # A rank-0 buffer is a single scalar -- the accumulator linalg.dot writes
    # into. There is no innermost dimension to vectorise, and the loop nest is
    # empty, so the scalar path handles it with no induction variables at all.
    width = dst.vector_width
    vectorized = bool(shape) and width > 0 and shape[-1] % width == 0

    ops = _FLOAT_OPS if dst.dtype.is_float else _INT_OPS
    if vectorized:
        _emit_vector(dst, expr, ops, width)
    else:
        _emit_scalar(dst, expr, ops)


# ---------------------------------------------------------------------------
# Loop nest construction
# ---------------------------------------------------------------------------


def _nest(bounds, body):
    """Build a nest of scf.for loops and call ``body(ivs)`` innermost."""

    def rec(level, ivs):
        if level == len(bounds):
            body(ivs)
            return
        lo, hi, step = bounds[level]
        for iv in range_(lo, hi, step):
            rec(level + 1, ivs + [iv])
            yield_([])

    rec(0, [])


def _emit_vector(dst, expr, ops, width):
    shape = dst.shape
    rank = len(shape)
    ety = dst.dtype.mlir()
    vec_ty = VectorType.get([width], ety)
    # Read a rank-1 vector out of a rank-N memref along the innermost dim.
    minor = AffineMapAttr.get(AffineMap.get(rank, 0, [AffineDimExpr.get(rank - 1)]))
    zero = arith.ConstantOp(ety, 0.0 if dst.dtype.is_float else 0)

    bounds = [(0, extent, 1) for extent in shape[:-1]]
    bounds.append((0, shape[-1], width))

    def body(ivs):
        value = _eval(
            expr, ivs, ops, ety, vectorized=True, vec_ty=vec_ty, minor=minor, pad=zero
        )
        transfer_write(None, value, dst.value, ivs, minor, [True])

    _nest(bounds, body)


def _emit_scalar(dst, expr, ops):
    ety = dst.dtype.mlir()
    bounds = [(0, extent, 1) for extent in dst.shape]

    def body(ivs):
        value = _eval(expr, ivs, ops, ety, vectorized=False)
        memref_store(value, dst.value, ivs)

    _nest(bounds, body)


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------


def _result(v):
    """Reduce a single-result OpView to its Value.

    Every consumer here has to receive a Value, not an OpView: the generated
    arith builders infer their own result type from ``operands[0].type``, and
    only a Value carries ``.type``. A one-operator expression never exposes this
    -- its operands come straight from ``transfer_read``, which already yields a
    Value -- but any nested expression (``alpha * x[:] + y[:]``) feeds one
    builder's output into the next.
    """
    return v.result if hasattr(v, "result") else v


def _eval(node, ivs, ops, ety, vectorized, vec_ty=None, minor=None, pad=None):
    if node.kind == "buffer":
        if vectorized:
            return _result(
                transfer_read(vec_ty, node.buffer.value, ivs, minor, pad, [True])
            )
        return _result(memref_load(node.buffer.value, ivs))

    if node.kind == "scalar":
        value = node.scalar
        from ._index import IndexExpr

        if isinstance(value, IndexExpr):
            # Materialise first: a constant expression folds back to a Python
            # int and takes the ordinary constant path below, so only a genuine
            # coordinate costs an index_cast.
            value = value.materialize()
            if not isinstance(value, int):
                if ops is not _INT_OPS:
                    raise NotImplementedError(
                        "a herd coordinate or loop variable can be broadcast "
                        "into an integer elementwise expression but not a "
                        f"floating-point one ({ety} here): it would need an "
                        "index_cast followed by a conversion to float, which "
                        "nothing has required yet"
                    )
                scalar = _result(arith.index_cast(ety, value))
                return _result(broadcast(vec_ty, scalar)) if vectorized else scalar
        if ops is _FLOAT_OPS and isinstance(value, int) and not isinstance(value, bool):
            # arith.constant of a float type with a Python int does not raise --
            # it fails an assertion inside cast<IntegerType> and aborts the
            # process, taking the traceback with it. `x[:] + 1` on an f32 buffer
            # is ordinary Python, so widen rather than reject.
            value = float(value)
        if ops is _INT_OPS and isinstance(value, float):
            # arith.constant of an integer type rejects a Python float with
            # "expected floating point type", which says nothing about which
            # scalar in the expression was wrong. A whole-number float is a
            # harmless way to write an integer constant, so accept it; anything
            # else would silently truncate, so refuse it here.
            if not value.is_integer():
                raise ValueError(
                    f"cannot use the non-integral scalar {value} in an integer "
                    "elementwise expression; it would be truncated"
                )
            value = int(value)
        scalar = _result(arith.ConstantOp(ety, value))
        return _result(broadcast(vec_ty, scalar)) if vectorized else scalar

    if node.kind == "unary":
        fn = _FLOAT_UNARY_OPS.get(node.op)
        if fn is None or ops is _INT_OPS:
            raise NotImplementedError(
                f"elementwise operator '{node.op}' is not supported for "
                f"{'integer' if ops is _INT_OPS else 'float'} buffers"
            )
        return _result(
            fn(_eval(node.args[0], ivs, ops, ety, vectorized, vec_ty, minor, pad))
        )

    if node.kind == "binary":
        op = ops.get(node.op)
        if op is None:
            raise NotImplementedError(
                f"elementwise operator '{node.op}' is not supported for dtype "
                f"{'float' if ops is _FLOAT_OPS else 'integer'}"
            )
        lhs = _eval(node.args[0], ivs, ops, ety, vectorized, vec_ty, minor, pad)
        rhs = _eval(node.args[1], ivs, ops, ety, vectorized, vec_ty, minor, pad)
        return _result(op(lhs, rhs))

    raise AssertionError(f"unknown expression node kind {node.kind!r}")
