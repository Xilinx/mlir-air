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
from air.dialects.memref import load as memref_load, store as memref_store
from air.dialects.scf import for_ as range_, yield_
from air.dialects.vector import broadcast, transfer_read, transfer_write

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
    if not leaves:
        raise ValueError(
            "elementwise assignment needs at least one buffer on the "
            "right-hand side; assigning a bare scalar is not supported yet"
        )
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

    shape = dst.shape
    inner = shape[-1]
    width = dst.vector_width
    vectorized = width > 0 and inner % width == 0

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
        scalar = _result(arith.ConstantOp(ety, node.scalar))
        return _result(broadcast(vec_ty, scalar)) if vectorized else scalar

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
