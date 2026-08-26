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

from air.ir import AffineDimExpr, AffineMap, AffineMapAttr, IndexType, VectorType
from air.dialects import arith
from air.dialects import math as math_dialect
from air.dialects.memref import load as memref_load, store as memref_store
from air.dialects.scf import for_ as range_, yield_
from air.dialects.vector import (
    CombiningKind,
    broadcast,
    fma as vector_fma,
    reduction,
    transfer_read,
    transfer_write,
)

from .types import require_computable, require_signless

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
    "exp": math_dialect.exp,
    "rsqrt": math_dialect.rsqrt,
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
    # Shifts, likewise integer-only. The right shift is *arithmetic*
    # (arith.shrsi, sign-replicating) rather than logical, because air.api's
    # integer dtypes are signed and Python's own `>>` on a negative int is
    # arithmetic too -- x >> n == floor(x / 2**n) for both.
    "shl": arith.ShLIOp,
    "shr": arith.ShRSIOp,
}

# Named so the failure can say *why*, not just "not supported for dtype float".
_BITWISE = ("and", "or", "xor")
_SHIFT = ("shl", "shr")

# vector.reduction combining kinds. maximumf mirrors the elementwise _FLOAT_OPS
# choice of arith.maximumf over maxnumf; the signed integer form matches the
# rest of air.api's integer dtypes, and unsigned never reaches here.
_REDUCE_F = {"add": CombiningKind.ADD, "max": CombiningKind.MAXIMUMF}
_REDUCE_I = {"add": CombiningKind.ADD, "max": CombiningKind.MAXSI}


def _broadcast_offset(shape, dst_shape):
    """numpy's broadcast rule, one-sided: can ``shape`` be read as ``dst_shape``?

    Returns how many destination axes the operand does not have -- the amount
    the two shapes are offset by once right-aligned -- or None if they do not
    broadcast. Right-aligned each operand axis must either match the
    destination's or be 1, and leading axes the operand does not have at all
    are the ``1`` case written by omission.

    One-sided because the destination of an air.api assignment is a buffer that
    already exists: numpy would broadcast both sides against each other and
    allocate the result, and here the result's shape is given, which is numpy's
    own rule for an explicit ``out=``.
    """
    offset = len(dst_shape) - len(shape)
    if offset < 0:
        return None
    for extent, target in zip(shape, dst_shape[offset:]):
        if extent != target and extent != 1:
            return None
    return offset


def _pins_an_axis(shape, dst_shape):
    """Does reading ``shape`` as ``dst_shape`` pin one of its axes to 0?

    Deliberately narrower than "is this a broadcast". An operand short of
    *leading* axes -- a [16] bias against an [8, 16] tile -- is stretched
    without any axis it actually has being pinned, because a missing axis
    contributes no index at all. Only an axis that exists and is 1 against a
    wider destination needs the constant.

    The condition is the negation of the one in ``_leaf_index``, and is written
    that way on purpose: the two must agree, or a constant is emitted that
    nothing uses, or -- worse -- used before it is emitted.
    """
    offset = _broadcast_offset(shape, dst_shape)
    return any(extent != dst_shape[i + offset] for i, extent in enumerate(shape))


def _zero_index_if(needed):
    """An index-typed 0 for the pinned axes of a broadcast, or None.

    Conditional so that a kernel that pins nothing emits no constant, and
    therefore the same IR it emitted before broadcasting existed.
    """
    return _result(arith.ConstantOp(IndexType.get(), 0)) if needed else None


def _leaf_index(shape, dst_shape, ivs, zero):
    """The indices to read a leaf of ``shape`` at, inside a nest over ``dst_shape``.

    An axis that matches the destination's is walked with the destination's own
    induction variable; an axis of extent 1 against a wider destination is
    pinned at 0, which is what makes the read a broadcast. Axes the operand does
    not have contribute no index at all.

    When the shapes are equal this returns ``ivs`` unchanged, which is what
    keeps every existing kernel's IR byte-identical.
    """
    offset = _broadcast_offset(shape, dst_shape)
    return [
        ivs[i + offset] if extent == dst_shape[i + offset] else zero
        for i, extent in enumerate(shape)
    ]


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
        if _broadcast_offset(leaf.shape, dst.shape) is None:
            raise ValueError(
                f"shape mismatch in elementwise assignment: destination has "
                f"shape {dst.shape} but operand has shape {leaf.shape}, which "
                f"does not broadcast to it. Right-aligned, every operand axis "
                f"must either match the destination's or be 1"
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
    # A reduction is the one right-hand side whose shape is not the
    # destination's, so it is dispatched before the elementwise shape check
    # rather than being taught to it.
    if expr.kind == "reduce":
        _emit_reduce(dst, expr)
        return

    # A bare scalar on the right-hand side is a fill (`acc[:] = 0.0`), which is
    # how an accumulator is zeroed before a K loop. It needs no leaves: the
    # destination supplies the shape.
    _check_region(expr, dst, dst.dtype)

    # A tile the core has no instructions for -- f16 -- follows the same rule
    # as an unsigned one: it can be *copied* elementwise, because a copy emits a
    # read and a write and no arith op, and the bits arrive unchanged. Measured
    # on npu1: an f16 copy is exact on 2048 of 2048 elements, and an f16 add is
    # wrong on 2048 of 2048. Only the second is refused.
    if not dst.dtype.computes and expr.kind != "buffer":
        require_computable(
            dst.dtype,
            "an elementwise operator or broadcast scalar (a plain copy, "
            "dst[:] = src[:], is)",
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


def _emit_reduce(dst, expr):
    """Emit ``dst[:] = ops.reduce_*(src)`` -- one vector.reduction per row.

    The destination is the operand with its innermost axis collapsed -- kept
    as 1 or dropped, see the shape check below -- so the loop nest walks the
    outer dimensions and each trip reduces one whole innermost axis.

    That axis is read as a *single* vector of its full extent rather than in
    steps of the destination's vector width. This is what the hand-written
    kernels do, and it is the reason there is no accumulator here: stepping
    would need a loop-carried vector accumulator, which is exactly the
    construct ``ops.dot`` documents as failing to legalize on AIE2 (LLVM
    splits it into sub-512-bit pieces). The cost is that the reduced axis has
    to be a vector length the backend accepts.
    """
    operand = expr.args[0]
    dtype = dst.dtype
    require_computable(dtype, "air.api.ops.reduce_add / reduce_max")
    require_signless(dtype, "air.api.ops.reduce_add / reduce_max")

    # The source shape is the destination's with the innermost extent taken
    # from the operand's own leaves: that dimension is what gets collapsed, so
    # it is the one thing the destination cannot tell us.
    leaves = operand.leaves()
    src_shape = leaves[0].shape
    for leaf in leaves:
        if leaf.shape != src_shape:
            raise ValueError(
                f"shape mismatch inside a reduction: operands have shapes "
                f"{src_shape} and {leaf.shape}. Unlike a plain elementwise "
                f"assignment, a reduction does not broadcast its operands: the "
                f"innermost extent is the thing being collapsed, so an operand "
                f"of extent 1 there would change what the reduction means "
                f"rather than being stretched to fit"
            )
        if leaf.dtype is not dtype:
            raise ValueError(
                f"dtype mismatch in elementwise assignment: destination is "
                f"{dtype} but operand is {leaf.dtype}"
            )
        if leaf.value is None:
            raise RuntimeError(
                "buffer used before allocation; air.alloc() must be called "
                "inside the herd body that uses it"
            )

    # Two destination spellings are accepted, matching numpy's keepdims:
    #   [.., n] -> [.., 1]  keeps the reduced axis (keepdims=True)
    #   [.., n] -> [..]     drops it (keepdims=False)
    # Both occur in the kernels this replaces, and in the *same* kernel: the
    # hand-written reduce_add allocates its L1 tile [tile_m, 1] but declares
    # the L3 output [m], so whichever spelling the DSL refused would have
    # forced the example to change shape somewhere.
    kept = tuple(list(src_shape[:-1]) + [1])
    dropped = tuple(src_shape[:-1])
    keepdims = tuple(dst.shape) == kept
    if not keepdims and tuple(dst.shape) != dropped:
        raise ValueError(
            f"shape mismatch in a reduction: reducing {src_shape} along its "
            f"innermost axis gives {kept} (keeping the axis) or {dropped} "
            f"(dropping it), but the destination is {tuple(dst.shape)}"
        )

    lanes = src_shape[-1]
    minor = _minor(len(src_shape))
    regions = {d: _Region(d, lanes, True) for d in _regions_in(operand, dtype, [])}
    kind = (_REDUCE_F if dtype.is_float else _REDUCE_I)[expr.op]

    # Index-typed 0 for the collapsed axis: it is both where the whole
    # row is read from and where the scalar result is stored.
    zero = _result(arith.ConstantOp(IndexType.get(), 0))
    bounds = [(0, extent, 1) for extent in src_shape[:-1]]

    # Every leaf here has the operand's shape -- the check above insists on it,
    # since a reduction's extent is what is being collapsed -- so the read is
    # always the plain one and broadcasting does not arise.
    def load(buf, at, region):
        return _result(
            transfer_read(region.vec_ty, buf.value, at, minor, region.pad, [True])
        )

    def body(ivs):
        reads = {}
        value = _eval(operand, ivs + [zero], regions[dtype], regions, True, load, reads)
        scalar = _result(reduction(dtype.mlir(), kind, value))
        memref_store(scalar, dst.value, ivs + [zero] if keepdims else ivs)

    _nest(bounds, body)


def _minor(rank):
    """A minor-identity permutation map: read a rank-N memref's innermost dim."""
    return AffineMapAttr.get(AffineMap.get(rank, 0, [AffineDimExpr.get(rank - 1)]))


def _emit_vector(dst, expr, width):
    shape = dst.shape
    rank = len(shape)
    # Read a rank-1 vector out of a rank-N memref along the innermost dim.
    minor = _minor(rank)
    # One lane count for the whole nest, taken from the destination. A cast does
    # not change how many elements a trip covers, only how wide each one is, so
    # the source is read at the destination's lane count and not at its own
    # dtype's default.
    regions = {d: _Region(d, width, True) for d in _regions_in(expr, dst.dtype, [])}

    bounds = [(0, extent, 1) for extent in shape[:-1]]
    bounds.append((0, shape[-1], width))

    # Hoisted above the nest, like the padding constants, so a broadcast costs
    # one constant for the whole kernel rather than one per trip.
    zero = _zero_index_if(
        any(_pins_an_axis(leaf.shape, shape) for leaf in expr.leaves())
    )

    def load(buf, ivs, region):
        index = _leaf_index(buf.shape, shape, ivs, zero)
        if buf.shape and buf.shape[-1] == shape[-1]:
            # The operand has the destination's innermost extent, so the vector
            # read is the ordinary one -- at the operand's own rank, which is
            # the destination's unless leading axes were broadcast away.
            return _result(
                transfer_read(
                    region.vec_ty,
                    buf.value,
                    index,
                    _minor(len(buf.shape)),
                    region.pad,
                    [True],
                )
            )
        # The innermost axis is the broadcast one, so there is no run of
        # contiguous elements to read: take the single element the whole vector
        # is made of and splat it. This is what the hand-written
        # vector_broadcast_scalar kernel spells as memref.load + vector.broadcast.
        return _result(broadcast(region.vec_ty, _result(memref_load(buf.value, index))))

    def body(ivs):
        value = _eval(expr, ivs, regions[dst.dtype], regions, True, load, {})
        transfer_write(None, value, dst.value, ivs, minor, [True])

    _nest(bounds, body)


def _emit_scalar(dst, expr):
    shape = dst.shape
    regions = {d: _Region(d, 0, False) for d in _regions_in(expr, dst.dtype, [])}
    bounds = [(0, extent, 1) for extent in shape]

    zero = _zero_index_if(
        any(_pins_an_axis(leaf.shape, shape) for leaf in expr.leaves())
    )

    def load(buf, ivs, region):
        return _result(memref_load(buf.value, _leaf_index(buf.shape, shape, ivs, zero)))

    def body(ivs):
        value = _eval(expr, ivs, regions[dst.dtype], regions, False, load, {})
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


def _eval(node, ivs, region, regions, vectorized, load, reads=None):
    ops, ety = region.ops, region.ety
    vec_ty, pad = region.vec_ty, region.pad

    def recur(child, into=region):
        return _eval(child, ivs, into, regions, vectorized, load, reads)

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
        value = load(node.buffer, ivs, region)
        if reads is not None:
            reads[key] = value
        return value

    if node.kind == "cast":
        # The operand is evaluated in *its* region -- a different element type,
        # a different arith table, and a different vector type -- and only the
        # conversion itself lands in this one.
        from .ops import _clamped_into, _conversion_op

        source = node.args[0].element_dtype()
        operand = recur(node.args[0], regions[source])
        # The same structural check ``cast`` used to accept this node. Asking
        # again here keeps the rule in one place rather than trusting a flag
        # threaded down from the call site.
        narrowing_ok = _clamped_into(node.args[0], region.dtype)
        build = _conversion_op(source, region.dtype, narrowing_ok=narrowing_ok)
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

    if node.kind == "fma":
        # a * b + c as one operation, so the product is not rounded before it
        # is added. Two separate arith ops round twice; the hand-written
        # vector_fma and axpy kernels both spell vector.fma for that reason.
        #
        # There is no integer counterpart and none is needed: integer multiply
        # and add are exact, so `a * b + c` already computes what an integer
        # fma would, and MLIR has no arith.fmai to lower to.
        if ops is _INT_OPS:
            raise NotImplementedError(
                "air.api.ops.fma is float-only: fusing the multiply and the "
                "add exists to avoid the intermediate *rounding*, and integer "
                "multiply-add has none to avoid. MLIR has no integer fma op "
                f"to lower to, so on a {ety} buffer write a * b + c instead, "
                "which computes the same thing"
            )
        # The scalar spelling would be math.fma, and it is refused rather than
        # emitted because it does not compile: AIE2 has no scalar fma
        # instruction, and `G_FMA` reaches the backend unlegalized. Measured on
        # npu1 and npu2, bf16 and f32, with and without bf16 emulation -- all
        # five fail with "unable to legalize instruction: (s16) = G_FMA".
        #
        # Emitting it anyway would repeat the math.tanh trap, where the
        # emitter's usual correctness-first fallback -- drop to a scalar loop
        # when the innermost dimension is not a multiple of the vector width --
        # is the *unsafe* direction, turning a working kernel into a build
        # failure whose error names an LLVM virtual register. Unlike tanh, this
        # is caught here, where the tile shape that caused it can be named.
        if not vectorized:
            raise NotImplementedError(
                "air.api.ops.fma has no scalar form on AIE2: there is no "
                "scalar fma instruction, so math.fma reaches the backend and "
                "fails to legalize. The emitter is on its scalar path here "
                "because the destination's innermost dimension is not a "
                "multiple of its vector width -- give the buffer a tile shape "
                "that is (air.alloc(..., vector=W) with shape[-1] % W == 0), "
                "or write a * b + c, which has a scalar form and rounds twice"
            )
        a, b, c = (recur(arg) for arg in node.args)
        return _result(vector_fma(a, b, c))

    if node.kind == "binary":
        op = ops.get(node.op)
        if op is None:
            if node.op in _BITWISE and ops is _FLOAT_OPS:
                raise NotImplementedError(
                    f"the bitwise operator '{node.op}' is integer-only: MLIR "
                    f"has arith.{node.op}i but no floating-point counterpart, "
                    f"so it cannot be applied to a {ety} buffer"
                )
            if node.op in _SHIFT and ops is _FLOAT_OPS:
                spelling = "<<" if node.op == "shl" else ">>"
                raise NotImplementedError(
                    f"the shift operator '{spelling}' is integer-only: MLIR "
                    f"has arith.shli/shrsi and no floating-point counterpart, "
                    f"so it cannot be applied to a {ety} buffer. Scaling a "
                    f"float by a power of two is a multiply"
                )
            raise NotImplementedError(
                f"elementwise operator '{node.op}' is not supported for dtype "
                f"{'float' if ops is _FLOAT_OPS else 'integer'}"
            )
        return _result(op(recur(node.args[0]), recur(node.args[1])))

    if node.kind == "reduce":
        # Reached only when a reduction is *nested*: emit_elementwise dispatches
        # a top-level one to _emit_reduce before the walk ever starts. It has no
        # emission here because it cannot have one -- a reduction collapses the
        # innermost axis, so its result has a different shape from the operands
        # around it, and the surrounding loop nest is built over a single shape.
        # Without this the walk fell through to the AssertionError below, which
        # named an internal node kind rather than the mistake.
        spelling = "reduce_max" if node.op == "max" else "reduce_add"
        raise NotImplementedError(
            f"air.api.ops.{spelling} cannot nest inside a larger expression: it "
            "collapses the innermost axis, so its result has a different shape "
            "from the operands around it, and one loop nest cannot span both. "
            "Assign the reduction first, then use the result:\n"
            f"    tmp[:] = ops.{spelling}(a[:])\n"
            "    out[:] = tmp[:] + b[:]"
        )

    raise AssertionError(f"unknown expression node kind {node.kind!r}")
