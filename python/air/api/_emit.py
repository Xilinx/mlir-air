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

One expression can hold more than one element type, because ``ops.cast``
converts between them. Everything below a cast node is read and computed in the
source type and only the conversion itself lands in the destination's; a
``_Region`` groups the four things that differ between the two (the MLIR element
type, the arith table, the vector type and the padding constant) so they can be
looked up per node rather than threaded down the walk.
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

# Comparison predicates. The float side uses the *ordered* forms: ordered means
# false when either operand is NaN, which is C's `>=` and what the hand-written
# vector_select kernel named (CmpFPredicate.OGE). The integer side uses the
# signed forms -- air.api's integer dtypes are signless-or-signed, and an
# unsigned buffer never reaches here (see require_signless below).
_CMP_F = {
    "lt": arith.CmpFPredicate.OLT,
    "le": arith.CmpFPredicate.OLE,
    "gt": arith.CmpFPredicate.OGT,
    "ge": arith.CmpFPredicate.OGE,
    "eq": arith.CmpFPredicate.OEQ,
    "ne": arith.CmpFPredicate.ONE,
}

_CMP_I = {
    "lt": arith.CmpIPredicate.slt,
    "le": arith.CmpIPredicate.sle,
    "gt": arith.CmpIPredicate.sgt,
    "ge": arith.CmpIPredicate.sge,
    "eq": arith.CmpIPredicate.eq,
    "ne": arith.CmpIPredicate.ne,
}

_INT_OPS = {
    "add": arith.AddIOp,
    "sub": arith.SubIOp,
    "mul": arith.MulIOp,
    "div": arith.DivSIOp,
    "max": arith.MaxSIOp,
    "min": arith.MinSIOp,
    # Bitwise. Integer only -- there is no arith.andf, so these deliberately
    # have no _FLOAT_OPS counterpart and _eval names the dtype when a float
    # buffer reaches one.
    "and": arith.AndIOp,
    "or": arith.OrIOp,
    "xor": arith.XOrIOp,
}

# Named so the failure can say *why*, not just "not supported for dtype float".
_BITWISE = ("and", "or", "xor")


def _check_region(node, dst, dtype, expected=None):
    """Check every buffer leaf against the element type of *its* region.

    Without a cast in the tree there is one region and this is the loop that
    used to live in ``emit_elementwise``, with its two messages unchanged. A
    cast starts a new region: below it the expected type is the cast's source,
    so leaves there are checked against that instead of against ``dst`` -- and
    ``expected`` renames the type in the message accordingly, because "the
    destination is f32" is untrue of an assignment whose destination is i32 and
    whose cast happens to convert from f32.

    Shape is checked against the destination in either region: a cast changes
    the element type and nothing else.
    """
    expected = expected or f"destination is {dtype}"
    if node.kind == "cast":
        source = node.args[0].element_dtype()
        _check_region(
            node.args[0], dst, source, expected=f"the cast converts from {source}"
        )
        return
    if node.kind == "buffer":
        leaf = node.buffer
        if leaf.shape != dst.shape:
            raise ValueError(
                f"shape mismatch in elementwise assignment: destination has "
                f"shape {dst.shape} but operand has shape {leaf.shape}"
            )
        if leaf.dtype is not dtype:
            raise ValueError(
                f"dtype mismatch in elementwise assignment: {expected} but "
                f"operand is {leaf.dtype}"
            )
        if leaf.value is None:
            raise RuntimeError(
                "buffer used before allocation; air.alloc() must be called "
                "inside the herd body that uses it"
            )
        return
    for arg in node.args:
        _check_region(arg, dst, dtype, expected)


def _regions_in(node, dtype, out):
    """Element types appearing in ``node``, outermost first.

    Order is deliberate and not incidental: it fixes the order the padding
    constants are emitted in, so an expression with no cast in it produces
    byte-identical IR to the version of this emitter that knew only one type.
    """
    if dtype is not None and dtype not in out:
        out.append(dtype)
    if node.kind == "cast":
        _regions_in(node.args[0], node.args[0].element_dtype(), out)
        return out
    for arg in node.args:
        _regions_in(arg, dtype, out)
    return out


def emit_elementwise(dst, expr):
    """Emit ``dst[:] = expr`` as a loop nest over ``dst``'s shape."""
    # A bare scalar on the right-hand side is a fill (`acc[:] = 0.0`), which is
    # how an accumulator is zeroed before a K loop. It needs no leaves: the
    # destination supplies the shape.
    _check_region(expr, dst, dst.dtype)

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
        _emit_scalar(dst, expr)
        return

    shape = dst.shape
    # A rank-0 buffer is a single scalar -- the accumulator linalg.dot writes
    # into. There is no innermost dimension to vectorise, and the loop nest is
    # empty, so the scalar path handles it with no induction variables at all.
    width = dst.vector_width
    vectorized = bool(shape) and width > 0 and shape[-1] % width == 0

    if vectorized:
        _emit_vector(dst, expr, width)
    else:
        _emit_scalar(dst, expr)


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


class _Region:
    """Everything the emitter needs to work in one element type.

    Before ``ops.cast`` there was exactly one of these per assignment and its
    four fields were four separate arguments threaded down ``_eval``. A cast
    makes an expression hold more than one element type at once, so they are
    grouped and looked up per node instead.

    The padding constant is built once here rather than per read, which is what
    keeps it hoisted above the loop nest exactly as before.
    """

    __slots__ = ("dtype", "ops", "ety", "vec_ty", "pad")

    def __init__(self, dtype, width, vectorized):
        self.dtype = dtype
        self.ety = dtype.mlir()
        self.ops = _FLOAT_OPS if dtype.is_float else _INT_OPS
        self.vec_ty = VectorType.get([width], self.ety) if vectorized else None
        self.pad = (
            arith.ConstantOp(self.ety, 0.0 if dtype.is_float else 0)
            if vectorized
            else None
        )


def _emit_vector(dst, expr, width):
    shape = dst.shape
    rank = len(shape)
    # Read a rank-1 vector out of a rank-N memref along the innermost dim.
    minor = AffineMapAttr.get(AffineMap.get(rank, 0, [AffineDimExpr.get(rank - 1)]))
    # One lane count for the whole nest, taken from the destination. A cast does
    # not change how many elements a trip covers, only how wide each one is, so
    # the source is read at the destination's lane count and not at its own
    # dtype's default.
    regions = {d: _Region(d, width, True) for d in _regions_in(expr, dst.dtype, [])}

    bounds = [(0, extent, 1) for extent in shape[:-1]]
    bounds.append((0, shape[-1], width))

    def body(ivs):
        value = _eval(expr, ivs, regions[dst.dtype], regions, True, minor, {})
        transfer_write(None, value, dst.value, ivs, minor, [True])

    _nest(bounds, body)


def _emit_scalar(dst, expr):
    regions = {d: _Region(d, 0, False) for d in _regions_in(expr, dst.dtype, [])}
    bounds = [(0, extent, 1) for extent in dst.shape]

    def body(ivs):
        value = _eval(expr, ivs, regions[dst.dtype], regions, False, None, {})
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


def _eval(node, ivs, region, regions, vectorized, minor, reads=None):
    ops, ety = region.ops, region.ety
    vec_ty, pad = region.vec_ty, region.pad

    def recur(child, into=region):
        return _eval(child, ivs, into, regions, vectorized, minor, reads)

    if node.kind == "buffer":
        # One read per buffer per loop body, not one per mention. A buffer that
        # appears more than once in the tree -- `select(a >= b, a, b)` names
        # both twice, and gelu names x four times -- is the same value at every
        # mention: nothing writes to L1 between them inside a single iteration.
        # Without this the emitter issues a redundant transfer_read for each
        # extra mention, which is what the hand-written kernels avoid by binding
        # the read to a local once.
        #
        # The key is the buffer alone, with no region in it, and that is safe
        # rather than lucky: a buffer has one element type, `_check_region`
        # requires every leaf to match the type of the region it sits in, and a
        # cast is the only thing that starts a new region. So a given buffer is
        # reachable from exactly one region and its cached value can only ever
        # have been read at that region's vector type.
        key = id(node.buffer)
        if reads is not None and key in reads:
            return reads[key]
        if vectorized:
            value = _result(
                transfer_read(vec_ty, node.buffer.value, ivs, minor, pad, [True])
            )
        else:
            value = _result(memref_load(node.buffer.value, ivs))
        if reads is not None:
            reads[key] = value
        return value

    if node.kind == "cast":
        # The operand is evaluated in *its* region -- a different element type,
        # a different arith table, and a different vector type -- and only the
        # conversion itself lands in this one.
        from .ops import _conversion_op

        source = node.args[0].element_dtype()
        operand = recur(node.args[0], regions[source])
        build = _conversion_op(source, region.dtype)
        return _result(build(vec_ty if vectorized else ety, operand))

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
        return _result(fn(recur(node.args[0])))

    if node.kind == "compare":
        # Yields i1 (vector<Wxi1> when vectorised), not the element type. Only
        # a "select" node consumes this, which ops.select enforces at trace
        # time -- nothing else in _eval can receive one. Its *operands* are
        # ordinary values in this region, so they recur normally.
        table = _CMP_I if ops is _INT_OPS else _CMP_F
        predicate = table.get(node.op)
        if predicate is None:
            raise NotImplementedError(
                f"comparison '{node.op}' is not supported for dtype "
                f"{'integer' if ops is _INT_OPS else 'float'}"
            )
        build = arith.cmpi if ops is _INT_OPS else arith.cmpf
        return _result(build(predicate, recur(node.args[0]), recur(node.args[1])))

    if node.kind == "select":
        cond, a, b = node.args
        return _result(arith.select(recur(cond), recur(a), recur(b)))

    if node.kind == "binary":
        op = ops.get(node.op)
        if op is None:
            if node.op in _BITWISE and ops is _FLOAT_OPS:
                raise NotImplementedError(
                    f"the bitwise operator '{node.op}' is integer-only: MLIR "
                    f"has arith.{node.op}i but no floating-point counterpart, "
                    f"so it cannot be applied to a {ety} buffer"
                )
            raise NotImplementedError(
                f"elementwise operator '{node.op}' is not supported for dtype "
                f"{'float' if ops is _FLOAT_OPS else 'integer'}"
            )
        return _result(op(recur(node.args[0]), recur(node.args[1])))

    raise AssertionError(f"unknown expression node kind {node.kind!r}")
