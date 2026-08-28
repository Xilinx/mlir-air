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

from ._index import Leaf
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


def _broadcast_shape(shapes):
    """numpy's broadcast of several shapes, or None if they do not agree.

    Right-aligned, each axis is the one non-1 extent among the operands, or 1
    when they are all 1. Unlike ``_broadcast_offset`` this is two-sided -- there
    is no destination to broadcast *to* yet, which is the situation inside a
    reduction, where the collapsed shape is whatever the operands together say
    it is.
    """
    rank = max(len(s) for s in shapes)
    out = []
    for axis in range(rank):
        extent = 1
        for shape in shapes:
            i = axis - (rank - len(shape))
            if i < 0:
                continue
            e = shape[i]
            if e == 1:
                continue
            if extent not in (1, e):
                return None
            extent = e
        out.append(extent)
    return tuple(out)


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


def _is_repack(node):
    """Is this bitcast one that changes the lane count?

    A same-width bitcast is a relabelling and behaves exactly like a cast. A
    repack -- 32 bytes read as 64 half-bytes -- changes how many elements a run
    of memory holds, so its operand is read at a different lane count from the
    rest of the nest and cannot share the per-dtype region table.
    """
    if node.kind != "bitcast":
        return False
    return node.args[0].element_dtype().bits != node.dtype.bits


def _pins_for(node, dst_shape):
    """Does any leaf under ``node`` need the shared zero index constant?

    Walks the tree rather than ``leaves()`` because a repacking bitcast's
    operand is not indexed like an ordinary leaf: its extent never matches the
    destination's, so ``_leaf_index`` always takes the pinned branch, which
    reaches for the constant whenever the region starts at 0.
    """
    if _is_repack(node):
        return True
    if node.kind == "buffer":
        return _pins_an_axis(node.buffer.shape, dst_shape)
    return any(_pins_for(a, dst_shape) for a in node.args)


def _zero_index_if(needed):
    """An index-typed 0 for the pinned axes of a broadcast, or None.

    Conditional so that a kernel that pins nothing emits no constant, and
    therefore the same IR it emitted before broadcasting existed.
    """
    return _result(arith.ConstantOp(IndexType.get(), 0)) if needed else None


def _axes_of(leaf):
    """``(sizes, dropped, base)`` for a leaf, whole buffer or region alike.

    A memref index has one entry per axis the *buffer* has, which is ``sizes``.
    The elementwise shape -- what broadcasting works on -- is ``sizes`` minus
    the axes an integer subscript took, and that is ``leaf.shape``. Keeping the
    two apart is the whole of integer indexing: ``gu[i, j]`` reads a scalar and
    still needs two indices to say which one.
    """
    sizes = getattr(leaf, "sizes", None)
    if sizes is None:
        # A whole buffer: every axis is walked, from zero.
        return list(leaf.shape), [False] * len(leaf.shape), None
    return list(sizes), list(leaf.dropped), leaf.base


def _leaf_index(leaf, dst_shape, ivs, zero):
    """The indices to read ``leaf`` at, inside a nest over ``dst_shape``.

    An axis that matches the destination's is walked with the destination's own
    induction variable; an axis of extent 1 against a wider destination is
    pinned at 0, which is what makes the read a broadcast. Axes the operand does
    not have contribute no index at all, and an axis an integer subscript took
    is never walked -- it contributes its offset and nothing else.

    The offset may be a herd coordinate or a loop variable rather than a
    constant. It materialises the same way every index in this DSL does: a
    constant folds, anything else becomes one ``affine.apply``.

    When the shapes are equal, nothing is dropped and the base is all zeros this
    returns ``ivs`` unchanged, which is what keeps every kernel that predates
    regions emitting byte-identical IR.
    """
    sizes, dropped, base = _axes_of(leaf)
    shape = [s for s, drop in zip(sizes, dropped) if not drop]
    offset = _broadcast_offset(shape, dst_shape)
    index = []
    kept = 0
    for i, extent in enumerate(sizes):
        start = 0 if base is None else base[i]
        if dropped[i]:
            # An integer subscript: this axis is not part of the element shape,
            # so no induction variable reaches it.
            index.append(_materialize_index(start))
            continue
        walks = extent == dst_shape[kept + offset]
        iv = ivs[kept + offset] if walks else None
        kept += 1
        if _is_zero(start):
            index.append(iv if walks else zero)
        elif not walks:
            index.append(_materialize_index(start))
        else:
            index.append(_result(arith.AddIOp(iv, _materialize_index(start))))
    return index


def _pinned_index(leaf, zero):
    """Every axis at the region's own start -- no induction variable anywhere.

    What a repacking bitcast's operand needs. It covers the same run of memory
    as the destination but counts it in different units, so the destination's
    induction variables are in the wrong units to index it; the emitter requires
    such an assignment to be a single trip precisely so that the start is the
    whole answer.
    """
    sizes, dropped, base = _axes_of(leaf)
    return [
        zero if base is None or _is_zero(base[i]) else _materialize_index(base[i])
        for i in range(len(sizes))
    ]


def _dst_index(dst, ivs):
    """Where to write, inside a nest over the destination's element shape.

    A whole buffer is written at the induction variables themselves, which is
    what every kernel that predates region assignment emits. A region shifts
    each axis by its own offset, and an axis an integer subscript took is
    written at that offset alone -- so ``out[i, j, k] = ...`` nests over
    nothing and stores at exactly ``[i, j, k]``.

    The destination is not broadcast, so this is ``_leaf_index`` with the
    element shape standing in for the destination's: every kept axis walks.
    """
    if getattr(dst, "sizes", None) is None:
        return ivs
    return _leaf_index(dst, dst.shape, ivs, None)


def _is_zero(start):
    """Is this offset the literal 0, needing no index arithmetic at all?"""
    if isinstance(start, int):
        return start == 0
    return start.as_const() == 0


def _materialize_index(start):
    """An offset as an index Value: a constant folds, anything else applies.

    A bare induction variable or coordinate -- one term, coefficient 1, no
    constant -- is passed straight through. ``IndexExpr.materialize`` would wrap
    it in an identity ``affine.apply``, which is correct but is one op per index
    per trip; a six-deep nest indexing three buffers is where that shows.

    The shortcut tests for a :class:`Leaf` specifically, not for a single term.
    A :class:`DerivedLeaf` -- ``x // k`` or ``x % k``, which is a leaf because
    neither operation is linear -- has no SSA value of its own to pass through;
    it exists precisely to be materialised, and goes the long way.
    """
    if isinstance(start, int):
        return _index_constant(start)
    if len(start.terms) == 1 and start.const == 0:
        ((leaf, coefficient),) = start.terms.items()
        if coefficient == 1 and isinstance(leaf, Leaf):
            return leaf.value
    value = start.materialize()
    return _index_constant(value) if isinstance(value, int) else value


def _index_constant(value):
    return _result(arith.ConstantOp(IndexType.get(), value))


def _base_of(leaf):
    """A slice leaf's starting offset per axis; None for a whole buffer."""
    return getattr(leaf, "base", None)


def _offset_key(offset):
    """A hashable identity for one offset, constant or not.

    An offset is an affine form ``sum(c_i * leaf_i) + k``, so two offsets are
    the same index exactly when their terms and constant agree. Keying on the
    ``IndexExpr`` object instead would read ``gu[i, :]`` twice for one region,
    and keying on ``as_const()`` would collapse ``gu[i, :]`` and ``gu[j, :]``
    onto a shared ``None``, which is the dangerous direction.

    The terms are taken as a ``frozenset`` of the mapping's own items, which
    leaves each leaf to say what makes it itself: a :class:`Leaf` is one
    coordinate and compares by identity, while a :class:`DerivedLeaf` compares
    structurally, so two separately built copies of ``(i + 1) % 4`` are one
    term. Substituting ``id()`` for that would split them and read twice.
    """
    if isinstance(offset, int):
        return ("k", offset)
    return (frozenset(offset.terms.items()), offset.const)


def _read_key(leaf):
    """What makes two leaves the same read.

    A buffer that appears twice in one expression is read once, and the same has
    to hold for a region: `gu[0, :] * 2 + gu[0, :]` writes two BufferSlice
    objects for one region, so keying on the slice's own identity would read it
    twice. Keying on the underlying buffer alone is the opposite mistake -- it
    would serve gu[1, :] the value already read for gu[0, :]. The region is what
    identifies the read: which buffer, from where, how much.
    """
    base = _base_of(leaf)
    buffer = getattr(leaf, "buffer", leaf)
    key = None if base is None else tuple(_offset_key(o) for o in base)
    return (id(buffer), key, tuple(leaf.shape))


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
    # A nested reduction is diagnosed here rather than at emission, because the
    # shape check below reaches it first and reports the collapsed axis as an
    # operand that does not broadcast -- which is true, and says nothing about
    # the actual mistake. `ops.cast(ops.reduce_add(row[:]), f32)` used to fail
    # with "destination has shape (1, 1) but operand has shape (1, 64)".
    if node.kind == "reduce":
        raise _nested_reduction(node)
    if _is_repack(node):
        # The operand covers the same run of memory as the destination but
        # counts it in different units, so it is its *reinterpreted* extent that
        # has to broadcast -- 32 bytes present as 64 half-bytes. Checking the
        # buffer's own shape would reject every correct use.
        leaf = node.args[0].buffer
        ratio = node.args[0].element_dtype().bits // node.dtype.bits
        presented = tuple(leaf.shape[:-1]) + (leaf.shape[-1] * ratio,)
        if _broadcast_offset(presented, dst.shape) is None:
            raise ValueError(
                f"shape mismatch in elementwise assignment: destination has "
                f"shape {dst.shape} but the bitcast operand {leaf.shape} of "
                f"{node.args[0].element_dtype()} reinterprets as {presented} of "
                f"{node.dtype}, which does not broadcast to it"
            )
        if leaf.dtype is not node.args[0].element_dtype():
            raise ValueError(
                f"dtype mismatch in elementwise assignment: the bitcast "
                f"reinterprets from {node.args[0].element_dtype()} but operand "
                f"is {leaf.dtype}"
            )
        if leaf.value is None:
            raise RuntimeError(
                "buffer used before allocation; air.alloc() must be called "
                "inside the herd body that uses it"
            )
        return

    if node.kind in ("cast", "bitcast"):
        source = node.args[0].element_dtype()
        what = "cast" if node.kind == "cast" else "bitcast"
        _check_region(
            node.args[0],
            dst,
            source,
            expected=(
                f"the {what} reinterprets from {source}"
                if what == "bitcast"
                else f"the cast converts from {source}"
            ),
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


def _nested_reduction(node):
    """The error for a reduction used anywhere but as the whole right-hand side."""
    spelling = "reduce_max" if node.op == "max" else "reduce_add"
    return NotImplementedError(
        f"air.api.ops.{spelling} cannot nest inside a larger expression: it "
        "collapses the innermost axis, so its result has a different shape "
        "from the operands around it, and one loop nest cannot span both. "
        "Assign the reduction first, then use the result:\n"
        f"    tmp[:] = ops.{spelling}(a[:])\n"
        "    out[:] = tmp[:] + b[:]"
    )


def _regions_in(node, dtype, out):
    """Element types appearing in ``node``, outermost first.

    Order is deliberate and not incidental: it fixes the order the padding
    constants are emitted in, so an expression with no cast in it produces
    byte-identical IR to the version of this emitter that knew only one type.
    """
    if dtype is not None and dtype not in out:
        out.append(dtype)
    if node.kind == "cast" or (node.kind == "bitcast" and not _is_repack(node)):
        _regions_in(node.args[0], node.args[0].element_dtype(), out)
        return out
    if node.kind == "bitcast":
        # A repack reads its operand at its own lane count -- 32 bytes where the
        # destination steps 64 half-bytes -- so its region cannot come from this
        # table, which holds one lane count for the whole nest. _eval builds it.
        return out
    for arg in node.args:
        _regions_in(arg, dtype, out)
    return out


def _resolve_switches(node, dtype):
    """Emit any ops.switch in the tree, before the loop nest is built.

    A choice does not depend on the nest's induction variables, so its
    scf.index_switch belongs above the loop -- one switch per assignment rather
    than one per trip, which is where the hand-written kernels put it. It also
    cannot be emitted any earlier than this: the element type it yields is the
    destination's, and ops.switch is written before the destination is known.
    """
    from .ops import _Switch

    if node.kind == "scalar" and isinstance(node.scalar, _Switch):
        node.scalar = node.scalar.materialize(dtype)
        return
    if node.kind == "cast":
        _resolve_switches(node.args[0], node.args[0].element_dtype() or dtype)
        return
    for arg in node.args:
        _resolve_switches(arg, dtype)


def emit_elementwise(dst, expr):
    """Emit ``dst[:] = expr`` as a loop nest over ``dst``'s shape."""
    _resolve_switches(expr, dst.dtype)
    # A reduction is the one right-hand side whose shape is not the
    # destination's, so it is dispatched before the elementwise shape check
    # rather than being taught to it.
    if expr.kind == "reduce":
        if expr.op == "argmax":
            _emit_argmax(dst, expr)
        else:
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

    _check_repack(expr, dst, shape, width)

    if vectorized:
        _emit_vector(dst, expr, width)
    else:
        _emit_scalar(dst, expr)


# ---------------------------------------------------------------------------
# Loop nest construction
# ---------------------------------------------------------------------------


def _has_repack(node):
    """Is there a lane-count-changing bitcast anywhere in this expression?"""
    return _is_repack(node) or any(_has_repack(a) for a in node.args)


def _check_repack(expr, dst, shape, width):
    """A repacking bitcast constrains the whole assignment, so say so up front.

    Raised here rather than where the read is built. The operand is indexed from
    its region's start alone -- there is no induction variable in the right
    units to advance it, the destination's are in the other type's -- so more
    than one trip would read the same run of memory every time. Emission has
    already opened the loop nest by the time the read happens, and raising from
    inside it leaves a block with no terminator and a trace too broken to print,
    which buries the real message under an MLIR verifier error.
    """
    if not _has_repack(expr):
        return
    if not shape or width <= 0 or shape[-1] % width:
        raise NotImplementedError(
            "air.api.ops.bitcast that changes the lane count needs the vector "
            f"path, and this assignment takes the scalar one: its destination "
            f"{shape} is not a multiple of its vector width {width}. "
            "Reinterpreting one element at a time is not the same operation -- "
            "the whole point is that a run of memory holds a different number "
            "of them"
        )
    trips = 1
    for extent in shape[:-1]:
        trips *= extent
    trips *= shape[-1] // width
    if trips != 1:
        raise NotImplementedError(
            f"air.api.ops.bitcast reinterprets a run of memory, so the "
            f"assignment it appears in has to cover exactly one vector. This "
            f"destination is {shape} at a width of {width}, which is {trips} "
            f"trips of the nest, and the reinterpreted operand is indexed from "
            f"its region's start alone -- it would read the same {width} "
            f"elements every trip. Write the loop yourself with air.sequential "
            f"and assign one vector per trip. Every axis counts, not just the "
            f"innermost: a {(4, width)} destination is four trips even though "
            f"its innermost extent is exactly one vector"
        )


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
    # The shape being reduced is every leaf broadcast together, exactly as an
    # elementwise assignment broadcasts them: a variance is
    # `reduce_add((x - mean) * (x - mean))` with mean a per-row scalar, and
    # numpy stretches it the same way. Taking the widest leaf instead would be
    # right only when one leaf already covers every axis -- [tm, 1] with [1, N]
    # broadcast to [tm, N] and neither operand is that shape.
    src_shape = _broadcast_shape([leaf.shape for leaf in leaves])
    if src_shape is None:
        raise ValueError(
            f"shape mismatch inside a reduction: operands have shapes "
            f"{[tuple(leaf.shape) for leaf in leaves]} and do not broadcast "
            f"together. Right-aligned, every axis must either agree or be 1"
        )
    for leaf in leaves:
        if _broadcast_offset(leaf.shape, src_shape) is None:
            raise ValueError(
                f"shape mismatch inside a reduction: operands have shapes "
                f"{src_shape} and {leaf.shape}, and the second does not "
                f"broadcast to the first. Right-aligned, every operand axis "
                f"must either match or be 1"
            )
        if leaf.value is None:
            raise RuntimeError(
                "buffer used before allocation; air.alloc() must be called "
                "inside the herd body that uses it"
            )
    # Each leaf is checked against the element type of *its own* region, not
    # against the destination's. A reduction that accumulates in a wider type
    # than it reads -- `reduce_add(ops.cast(row[:], f32))`, which is how a mean
    # is computed -- has bf16 leaves under an f32 destination, and comparing
    # every leaf to the destination refused exactly that.
    _check_reduce_regions(operand, dtype)

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

    axis = src_shape[-1]
    # How many lanes one step reads. The operand's own vector width, exactly as
    # an elementwise read of the same buffer would use -- so a [.., 768] row
    # allocated at width 16 is reduced in 48 steps of 16 rather than as one
    # 768-lane vector, which is what the backend refuses
    # (`G_EXTRACT_VECTOR_ELT <768 x s16>`). When the axis is one step, this is
    # the whole-axis read the emitter has always done, unchanged.
    # The step width comes from a leaf that actually spans the reduced axis. A
    # leaf pinned at extent 1 there is splatted, not stepped, so its own width
    # says nothing about how far each step should advance.
    spanning = [leaf for leaf in leaves if leaf.shape and leaf.shape[-1] == axis]
    lanes = _reduce_lanes(spanning[0], axis) if spanning else axis
    minor = _minor(len(src_shape))
    regions = {d: _Region(d, lanes, True) for d in _regions_in(operand, dtype, [])}
    kind = (_REDUCE_F if dtype.is_float else _REDUCE_I)[expr.op]

    # Index-typed 0 for the collapsed axis: it is both where the whole
    # row is read from and where the scalar result is stored.
    zero = _result(arith.ConstantOp(IndexType.get(), 0))
    bounds = [(0, extent, 1) for extent in src_shape[:-1]]

    # The same broadcast-aware read the elementwise path uses: a leaf with the
    # reduced axis's full extent is a vector read, and one of extent 1 there is
    # a single element splatted across the vector.
    def load(buf, at, region, packed=False):
        if packed:
            raise NotImplementedError(
                "air.api.ops.bitcast that changes the lane count cannot appear "
                "inside a reduction or an argmax: those walk the reduced axis "
                "themselves, and a reinterpretation changes how many elements "
                "that axis has. Assign the reinterpreted values to a buffer "
                "first, then reduce it"
            )
        index = _leaf_index(buf, src_shape, at, zero)
        if buf.shape and buf.shape[-1] == src_shape[-1]:
            return _result(
                transfer_read(
                    region.vec_ty,
                    buf.value,
                    index,
                    _minor(len(_axes_of(buf)[0])),
                    region.pad,
                    [True],
                )
            )
        return _result(broadcast(region.vec_ty, _result(memref_load(buf.value, index))))

    if lanes == axis:

        def body(ivs):
            reads = {}
            value = _eval(
                operand, ivs + [zero], regions[dtype], regions, True, load, reads
            )
            scalar = _result(reduction(dtype.mlir(), kind, value))
            memref_store(
                scalar, dst.value, _dst_index(dst, ivs + [zero] if keepdims else ivs)
            )

        _nest(bounds, body)
        return

    # Stepped. The partial sums are carried through an L1 scratch vector rather
    # than an scf.for iter_arg: a loop-carried vector is what LLVM splits into
    # sub-512-bit pieces the AIE2 backend will not legalize. Every hand-written
    # kernel this models does the same round-trip.
    from ._trace import current_herd

    scratch = current_herd().scratch(dtype, lanes)
    acc_region = regions[dtype]
    flat = _minor(1)

    def read_acc():
        return _result(
            transfer_read(
                acc_region.vec_ty, scratch.value, [zero], flat, acc_region.pad, [True]
            )
        )

    def write_acc(value):
        transfer_write(None, value, scratch.value, [zero], flat, [True])

    combine = (_FLOAT_OPS if dtype.is_float else _INT_OPS)[expr.op]

    def step_value(ivs, at):
        return _eval(operand, ivs + [at], regions[dtype], regions, True, load, {})

    def body(ivs):
        # The first step seeds the accumulator, rather than an identity vector
        # seeding it and the first step combining into that. An identity is the
        # obvious thing to write and is wrong for reduce_max: the identity there
        # is the type's minimum, not the zero _Region carries as its padding
        # value, so seeding with padding would floor every maximum at 0. Peeling
        # the first step needs no identity at all and is the same shape for both
        # reductions.
        write_acc(step_value(ivs, zero))

        for at in range_(lanes, axis, lanes):
            write_acc(_result(combine(read_acc(), step_value(ivs, at))))
            yield_([])

        scalar = _result(reduction(dtype.mlir(), kind, read_acc()))
        memref_store(
            scalar, dst.value, _dst_index(dst, ivs + [zero] if keepdims else ivs)
        )

    _nest(bounds, body)


def _reduce_lanes(leaf, axis):
    """How many lanes one step of a reduction reads.

    The operand's own vector width, which is what an elementwise read of the
    same buffer uses -- unless the axis is not a whole number of steps, in which
    case it is read in one go and the backend decides whether that width is
    legal. A width of 0 selects the scalar path elsewhere; here it means the
    same thing, so the whole axis is read at once and no stepping applies.
    """
    width = getattr(leaf, "vector_width", 0)
    if not width or width >= axis or axis % width:
        return axis
    return width


def _emit_argmax(dst, expr):
    """``dst[:] = ops.argmax(x[:])`` -- the index of the innermost maximum.

    A scalar loop, not a ``vector.reduction``: the running maximum and the index
    that produced it have to travel together, and no vector reduction carries an
    index. They ride in the loop's ``iter_args``, which is safe because they are
    scalars -- it is a loop-carried *vector* AIE2 refuses.

    The first element seeds the pair rather than an identity, for the same
    reason the stepped reduction peels its first step: the identity for a
    maximum is the type's minimum, and there is no need to name it.
    """
    from air.dialects.arith import CmpFOp, CmpIOp, IndexCastOp, SelectOp

    operand = expr.args[0]
    index_dtype = dst.dtype
    require_signless(index_dtype, "the destination of air.api.ops.argmax")
    if index_dtype.is_float:
        raise TypeError(
            f"air.api.ops.argmax writes an index, so its destination has to be "
            f"an integer buffer; this one is {index_dtype}. Use "
            "air.api.ops.reduce_max for the value itself."
        )

    leaves = operand.leaves()
    src_shape = _broadcast_shape([leaf.shape for leaf in leaves])
    if src_shape is None:
        raise ValueError(
            f"shape mismatch inside a reduction: operands have shapes "
            f"{[tuple(leaf.shape) for leaf in leaves]} and do not broadcast "
            f"together"
        )
    # The type the comparison happens in, which is the operand's *result* type
    # and not its leaves': `argmax(ops.cast(row[:], f32))` compares in f32 while
    # its leaves are bf16, and taking the leaf type would both evaluate the
    # walk in the wrong region and compare at the wrong width.
    value_dtype = operand.element_dtype() or leaves[0].dtype
    # Checked here for the same reason the other reductions check their
    # destination: without it an f16 or unsigned operand reaches arith.cmpf /
    # arith.cmpi, which have no form for either.
    require_computable(value_dtype, "the operand of air.api.ops.argmax")
    require_signless(value_dtype, "the operand of air.api.ops.argmax")
    _check_reduce_regions(operand, value_dtype)

    kept = tuple(list(src_shape[:-1]) + [1])
    dropped = tuple(src_shape[:-1])
    keepdims = tuple(dst.shape) == kept
    if not keepdims and tuple(dst.shape) != dropped:
        raise ValueError(
            f"shape mismatch in a reduction: reducing {src_shape} along its "
            f"innermost axis gives {kept} (keeping the axis) or {dropped} "
            f"(dropping it), but the destination is {tuple(dst.shape)}"
        )

    axis = src_shape[-1]
    regions = {d: _Region(d, 0, False) for d in _regions_in(operand, value_dtype, [])}
    zero = _result(arith.ConstantOp(IndexType.get(), 0))
    bounds = [(0, extent, 1) for extent in src_shape[:-1]]

    def load(buf, ivs, region, packed=False):
        if packed:
            raise NotImplementedError(
                "air.api.ops.bitcast that changes the lane count cannot appear "
                "inside a reduction or an argmax: those walk the reduced axis "
                "themselves, and a reinterpretation changes how many elements "
                "that axis has. Assign the reinterpreted values to a buffer "
                "first, then reduce it"
            )
        return _result(memref_load(buf.value, _leaf_index(buf, src_shape, ivs, zero)))

    def at(ivs, column):
        return _eval(
            operand, ivs + [column], regions[value_dtype], regions, False, load, {}
        )

    greater = CmpFOp if value_dtype.is_float else CmpIOp
    predicate = _CMP_F["gt"] if value_dtype.is_float else _CMP_I["gt"]

    def body(ivs):
        best = at(ivs, zero)
        first = _result(arith.ConstantOp(index_dtype.mlir(), 0))
        for column, (running, chosen), results in range_(
            1, axis, 1, iter_args=[best, first]
        ):
            candidate = at(ivs, column)
            wins = _result(greater(predicate, candidate, running))
            as_index = _result(IndexCastOp(index_dtype.mlir(), column))
            yield_(
                [
                    _result(SelectOp(wins, candidate, running)),
                    _result(SelectOp(wins, as_index, chosen)),
                ]
            )
        memref_store(results[1], dst.value, ivs + [zero] if keepdims else ivs)

    _nest(bounds, body)


def _check_reduce_regions(node, dtype):
    """Every leaf's element type against the region it sits in.

    The elementwise path spells this as ``_check_region``, which also checks
    shapes against the destination. A reduction's operand has a different shape
    from its destination by construction -- that is what a reduction is -- so
    only the type half applies here.
    """
    if node.kind == "cast":
        _check_reduce_regions(node.args[0], node.args[0].element_dtype())
        return
    if node.kind == "buffer":
        if dtype is not None and node.buffer.dtype is not dtype:
            raise ValueError(
                f"dtype mismatch inside a reduction: this part of the "
                f"expression is computed in {dtype} but the operand is "
                f"{node.buffer.dtype}. Reading in one type and accumulating in "
                f"another is spelled with ops.cast: "
                f"ops.reduce_add(ops.cast(row[:], {dtype}))"
            )
        return
    for arg in node.args:
        _check_reduce_regions(arg, dtype)


def _minor(rank):
    """A minor-identity permutation map: read a rank-N memref's innermost dim."""
    return AffineMapAttr.get(AffineMap.get(rank, 0, [AffineDimExpr.get(rank - 1)]))


def _emit_vector(dst, expr, width):
    shape = dst.shape
    # Read a rank-1 vector out of a rank-N memref along the innermost dim. The
    # rank is the *memref's*, which is the destination's element rank unless an
    # integer subscript dropped an axis -- gu[0, :] writes into a rank-2 memref
    # while its element shape is rank 1.
    minor = _minor(len(_axes_of(dst)[0]))
    # One lane count for the whole nest, taken from the destination. A cast does
    # not change how many elements a trip covers, only how wide each one is, so
    # the source is read at the destination's lane count and not at its own
    # dtype's default.
    regions = {d: _Region(d, width, True) for d in _regions_in(expr, dst.dtype, [])}

    bounds = [(0, extent, 1) for extent in shape[:-1]]
    bounds.append((0, shape[-1], width))

    # Hoisted above the nest, like the padding constants, so a broadcast costs
    # one constant for the whole kernel rather than one per trip.
    zero = _zero_index_if(_pins_for(expr, shape))

    def load(buf, ivs, region, packed=False):
        if packed:
            # A repacking bitcast's operand: fewer buffer elements than the
            # destination has, covering the same run of memory. `_leaf_index`
            # sees an extent that does not match the destination's and pins the
            # axis at the region's own base, which is the right index only while
            # the nest makes a single trip -- checked by `_check_repack` before
            # any of this was emitted.
            return _result(
                transfer_read(
                    region.vec_ty,
                    buf.value,
                    _pinned_index(buf, zero),
                    _minor(len(_axes_of(buf)[0])),
                    region.pad,
                    [True],
                )
            )
        index = _leaf_index(buf, shape, ivs, zero)
        if buf.shape and buf.shape[-1] == shape[-1]:
            # The operand has the destination's innermost extent, so the vector
            # read is the ordinary one -- at the operand's own rank, which is
            # the destination's unless leading axes were broadcast away. The
            # permutation map is over the *memref's* rank, which is wider than
            # the element shape when an integer subscript dropped an axis.
            return _result(
                transfer_read(
                    region.vec_ty,
                    buf.value,
                    index,
                    _minor(len(_axes_of(buf)[0])),
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
        transfer_write(None, value, dst.value, _dst_index(dst, ivs), minor, [True])

    _nest(bounds, body)


def _emit_scalar(dst, expr):
    shape = dst.shape
    regions = {d: _Region(d, 0, False) for d in _regions_in(expr, dst.dtype, [])}
    bounds = [(0, extent, 1) for extent in shape]

    zero = _zero_index_if(_pins_for(expr, shape))

    def load(buf, ivs, region, packed=False):
        if packed:
            raise NotImplementedError(
                "air.api.ops.bitcast that changes the lane count needs the "
                "vector path; this assignment fell back to a scalar loop"
            )
        return _result(memref_load(buf.value, _leaf_index(buf, shape, ivs, zero)))

    def body(ivs):
        value = _eval(expr, ivs, regions[dst.dtype], regions, False, load, {})
        memref_store(value, dst.value, _dst_index(dst, ivs))

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
        key = _read_key(node.buffer)
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
        build = _conversion_op(
            source, region.dtype, narrowing_ok=narrowing_ok, signed=node.signed
        )
        return _result(build(vec_ty if vectorized else ety, operand))

    if node.kind == "bitcast":
        from air.dialects.vector import bitcast as vector_bitcast

        source = node.args[0].element_dtype()
        if not _is_repack(node):
            # Same width: a relabelling, and the lane count is untouched, so it
            # rides the ordinary region machinery exactly as a cast does.
            operand = recur(node.args[0], regions[source])
            build = vector_bitcast if vectorized else arith.BitcastOp
            return _result(build(vec_ty if vectorized else ety, operand))

        # A repack. The operand is a buffer region (ops.bitcast enforces that),
        # read at the lane count its own type gives the same run of memory: 32
        # i8 where the destination steps 64 i4.
        if not vectorized:
            raise NotImplementedError(
                f"air.api.ops.bitcast from {source} to {node.dtype} needs the "
                f"vector path, and this assignment fell back to a scalar loop "
                f"-- its destination tile is not a multiple of its vector "
                f"width. Reinterpreting one element at a time is not the same "
                f"operation: the whole point is that a run of memory holds a "
                f"different number of them"
            )
        ratio = source.bits // node.dtype.bits
        lanes = vec_ty.shape[0]
        if lanes % ratio:
            raise ValueError(
                f"air.api.ops.bitcast from {source} to {node.dtype}: the "
                f"destination is {lanes} lanes wide, which is not a whole "
                f"number of {ratio}-element groups, so the reinterpreted run "
                f"does not line up with a vector"
            )
        leaf = node.args[0].buffer
        src_region = _Region(source, lanes // ratio, True)
        # Through `load`, not a transfer_read here: the destination shape and
        # the shared zero constant live in that closure, and so does the check
        # that the reinterpreted run lines up with a single trip of the nest.
        return _result(vector_bitcast(vec_ty, load(leaf, ivs, src_region, True)))

    if node.kind == "scalar":
        value = node.scalar
        from ._index import IndexExpr

        # An already-typed SSA value -- what ops.switch returns, having emitted
        # its scf.index_switch where it was written rather than inside this
        # loop. It is the element type already, so it only needs splatting.
        if not isinstance(value, (int, float, IndexExpr)):
            return _result(broadcast(vec_ty, value)) if vectorized else value

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
        raise _nested_reduction(node)

    raise AssertionError(f"unknown expression node kind {node.kind!r}")
