# ./python/air/api/ops.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Memory transfers and elementwise compute.

Re-exported by :mod:`air.api`, so importing the package is enough::

    from air import api as air

    air.ops.load(buf, A[window])

The transfer ops (``load``, ``store``) return a :class:`~air.api._value.Token`.
AIR builds its real asynchronous dependency graph from program order in the
``air-dependency`` pass, so a v1 token carries no SSA value -- it exists so that
``dependency=`` can be type-checked instead of silently ignored.

Elementwise compute ops (``maximum``, ``minimum``, ``relu``, ``tanh``, and the
``sigmoid``/``silu``/``gelu`` compositions built on them) build lazy expression
nodes instead, and return a :class:`~air.api._value.BufferExpr`.

``dot`` is a statement rather than an expression: it accumulates into a buffer
and returns a Token, matching the signature the API proposal specified. On rank-6
(micro-tiled) operands it becomes the blocked contraction the AIE2 matmul
intrinsic wants; see ``air.api._pack``.

The remaining compute ops from the wider API proposal (``reduce``, ``exp``,
``stack``, ``dequant``, ``atomic_add``) are not implemented. They raise rather
than returning a plausible-looking placeholder: a DSL that accepts an op it
cannot lower produces a kernel that runs and is silently wrong. ``exp`` is on
that list by choice rather than by oversight -- the activations here are composed
from ``tanh``, which has a checked lowering, so nothing has needed a vector
``exp`` yet.
"""

from ._value import Buffer, BufferExpr, BufferSlice, Tensor, TensorSlice, Token
from .types import require_computable, require_signless

__all__ = [
    "load",
    "store",
    "copy",
    "cast",
    "maximum",
    "minimum",
    # equal/not_equal/select were added with the comparison operators and were
    # never listed here, so `from air.api.ops import *` silently omitted them.
    # Corrected while adding fma rather than left as a trap for the next entry.
    "equal",
    "not_equal",
    "select",
    "fma",
    "reduce_add",
    "reduce_max",
    "relu",
    "tanh",
    "sigmoid",
    "silu",
    "gelu",
    "dot",
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


# A transfer endpoint is a whole buffer, a region of one, or a region of a
# global tensor. `Endpoint` normalises the three into (value, dtype, sizes,
# access pattern) so load and store can be written once for every level of the
# memory hierarchy -- L3 to L2, L2 to L1, L1 to L2, L2 to L3.
class _Endpoint:
    __slots__ = ("value", "dtype", "sizes", "pattern", "tensor", "what", "raw")

    def __init__(
        self, value, dtype, sizes, pattern, tensor=None, what="buffer", raw=None
    ):
        self.value = value
        self.dtype = dtype
        self.sizes = sizes
        # None means "the whole memref": dma_memcpy_nd prints that as [] [] [].
        self.pattern = pattern
        self.tensor = tensor
        self.what = what
        # The pattern before its offsets were materialised into SSA values, kept
        # so a packed destination can re-derive the pattern from it (see
        # _repack_source). Materialising is one-way.
        self.raw = raw


def _endpoint(obj, direction, role):
    if isinstance(obj, Buffer):
        if obj.value is None:
            raise RuntimeError("buffer used before allocation")
        # A micro-tiled buffer is contiguous, so it still transfers as a whole
        # memref; but it is *shaped* [.., N/n, M/m, m, n] while the other end
        # thinks in [.., M, N], so it reports the logical extents.
        sizes = obj.pack.lead + obj.pack.logical if obj.pack else obj.shape
        return _Endpoint(obj.value, obj.dtype, sizes, None, what="buffer")
    if isinstance(obj, BufferSlice):
        if obj.value is None:
            raise RuntimeError("buffer used before allocation")
        return _Endpoint(
            obj.value,
            obj.dtype,
            tuple(obj.logical_sizes),
            (obj.materialize_offsets(), list(obj.sizes), list(obj.strides)),
            what="buffer slice",
            raw=(list(obj.offsets), list(obj.sizes), list(obj.strides)),
        )
    if isinstance(obj, TensorSlice):
        return _Endpoint(
            obj.tensor.value,
            obj.dtype,
            tuple(obj.sizes),
            (obj.materialize_offsets(), list(obj.sizes), list(obj.strides)),
            tensor=obj.tensor,
            what="tensor slice",
            raw=(list(obj.offsets), list(obj.sizes), list(obj.strides)),
        )
    if isinstance(obj, Tensor):
        # A whole tensor, which channel put/get take routinely --
        # `ChannelPut("ChanIn", a)` is the commonest line in the hand-written
        # channel examples. There is no access pattern: the transfer is the
        # whole memref, which the op prints as [] [] [].
        #
        # This also widens load/store, which share this function: `ops.load(buf,
        # A)` is now the whole of A rather than an error. That is deliberate --
        # it is the same transfer as `ops.load(buf, A[:, :])` and reads better
        # -- and it stays safe because _check_pair still requires the shapes to
        # agree, and store still marks the tensor an output through
        # `_Endpoint.tensor` below.
        if obj.value is None:
            raise RuntimeError(
                "tensor used before the function was traced; air.tensor(...) "
                "declares an argument, and it is bound when the launch body runs"
            )
        return _Endpoint(
            obj.value, obj.dtype, obj.shape, None, tensor=obj, what="tensor"
        )
    raise TypeError(
        f"air.api.ops.{direction} expects its {role} to be a buffer from "
        f"air.alloc(), a region of one such as staged[tx, 0:n, :], or a tensor "
        f"slice such as A[i:i+n]; got {type(obj).__name__}"
    )


def _squeeze_leading_units(sizes, rank):
    """Drop leading unit dimensions until ``sizes`` has rank ``rank``.

    A staged L2 tile is indexed per core, so the natural access pattern carries
    a leading 1 -- ``staged[tx, j:j+m, :]`` is ``[1, m, k]`` into an ``[m, k]``
    L1 buffer, exactly as the hand-written matvec kernel writes it. Only leading
    ones are dropped, so a genuine shape mismatch still fails.
    """
    sizes = list(sizes)
    while len(sizes) > rank and sizes[0] == 1:
        sizes.pop(0)
    return tuple(sizes)


def _check_pair(dst, src, direction):
    """Shapes and dtypes must agree, up to leading unit dimensions."""
    d, s = tuple(dst.sizes), tuple(src.sizes)
    if d != s:
        rank = min(len(d), len(s))
        if _squeeze_leading_units(d, rank) != _squeeze_leading_units(s, rank):
            raise ValueError(
                f"transfer shape mismatch in air.api.ops.{direction}: the "
                f"destination {dst.what} is {d} but the source {src.what} is {s}"
            )
    if dst.dtype is not src.dtype:
        raise ValueError(
            f"transfer dtype mismatch in air.api.ops.{direction}: the "
            f"destination {dst.what} is {dst.dtype} but the source {src.what} "
            f"is {src.dtype}"
        )


def _emit_dma(dst, src):
    from air.dialects.air import dma_memcpy_nd

    kwargs = {}
    if dst.pattern is not None:
        offsets, sizes, strides = dst.pattern
        kwargs.update(dst_offsets=offsets, dst_sizes=sizes, dst_strides=strides)
    if src.pattern is not None:
        offsets, sizes, strides = src.pattern
        kwargs.update(src_offsets=offsets, src_sizes=sizes, src_strides=strides)
    return dma_memcpy_nd(dst.value, src.value, **kwargs)


def load(dst, src, pad_before=None, pad_after=None, dependency=None):
    """Fill a buffer from a tensor or from a coarser buffer.

    ``load(l1, A[i:i+n])`` is L3 to L1; ``load(l1, staged[tx, 0:n, :])`` is L2 to
    L1; ``load(l2, A[i:i+n])`` is L3 to L2. The destination is always the buffer
    being filled. A bare tensor means the whole of it, so ``load(l2, A)`` and
    ``load(l2, A[:, :])`` are the same transfer.
    """
    _check_dependency(dependency)
    _check_padding(pad_before, pad_after)

    if not isinstance(dst, Buffer):
        raise TypeError(
            f"air.api.ops.load fills a buffer, so its first argument must be one "
            f"from air.alloc(); got {type(dst).__name__}"
        )
    dst_ep = _endpoint(dst, "load", "destination")
    src_ep = _endpoint(src, "load", "source")
    if dst.pack is not None:
        src_ep = _repack_source(dst, src, src_ep)
    _check_pair(dst_ep, src_ep, "load")
    return Token(_emit_dma(dst_ep, src_ep))


def _repack_source(dst, src, src_ep):
    """Re-walk a flat source in micro-tile order, for a packed destination.

    Filling a micro-tiled A or B tile is the one transfer where the access
    pattern goes on the *other* side: the destination is contiguous (``[] [] []``
    in the emitted IR) and the flat source is read out of order, so that the DMA
    itself performs the pack. The pattern is derived from the destination's
    micro-tile, so the call site writes an ordinary logical slice::

        ops.load(l1_a, l2_a[tx, 0, :, kk : kk + tile_k])

    A packed source, by contrast, already carries its own pattern -- it is an
    unpack, and ``Buffer.__getitem__`` built it.
    """
    from ._pack import pack_pattern

    # A packed source is already an unpack and carries its own pattern.
    if _pack_of(src) is not None:
        return src_ep
    if dst.pack.role == "C":
        raise NotImplementedError(
            "air.api.ops.load into a micro-tiled C accumulator is not supported: "
            "C is written by ops.dot and drained with ops.store. Zero it with "
            "acc[:] = 0.0 rather than loading into it."
        )
    if src_ep.raw is None:
        raise TypeError(
            "air.api.ops.load into a micro-tiled buffer needs a source *region*, "
            f"not a whole buffer, so that the {dst.pack.role} pack can be "
            "derived; index the source, e.g. l2_a[tx, 0, :, k : k + tile_k]"
        )
    offsets, sizes, strides = src_ep.raw
    nlead = len(dst.pack.lead)
    # A rank-2 region is padded by pack_pattern when the destination's leading
    # dimensions are all 1 -- they are structural, required by block_matmul's
    # 6-D operands, and a flat staging buffer has no such axes to slice.
    flat_ok = len(sizes) == 2 and all(e == 1 for e in dst.pack.lead)
    if len(sizes) != nlead + 2 and not flat_ok:
        raise ValueError(
            f"air.api.ops.load into a micro-tiled {dst.pack.role} buffer needs a "
            f"source region of rank {nlead + 2} (or rank 2 when its leading "
            f"dimensions are all 1), with the two logical axes last; got rank "
            f"{len(sizes)}, {tuple(sizes)}"
        )
    p_off, p_sizes, p_strides = pack_pattern(dst.pack, sizes, strides, offsets)
    src_ep.pattern = ([o.materialize() for o in p_off], p_sizes, p_strides)
    return src_ep


def _pack_of(obj):
    if isinstance(obj, Buffer):
        return obj.pack
    if isinstance(obj, BufferSlice):
        return obj.buffer.pack
    return None


def store(src, dst, pad_before=None, pad_after=None, dependency=None):
    """Drain a buffer into a tensor or into a coarser buffer.

    ``store(l1, C[i:i+n])`` is L1 to L3; ``store(l1, staged[tx, :])`` is L1 to
    L2; ``store(l2, C[i:i+n])`` is L2 to L3. The source is always the buffer
    being drained. A bare tensor destination means the whole of it.
    """
    _check_dependency(dependency)
    _check_padding(pad_before, pad_after)

    if isinstance(src, Buffer) and src.pack is not None:
        # Draining a micro-tiled buffer whole would emit `[] [] []` -- a
        # contiguous read, which copies the tile still in micro-tile order and is
        # silently wrong. The unpack lives in the access pattern, so the source
        # has to be subscripted for one to exist.
        raise TypeError(
            "air.api.ops.store cannot drain a micro-tiled buffer whole: the "
            "unpack back to row-major order is the access pattern, and a whole "
            "buffer has none. Subscript it in logical coordinates, e.g. "
            f"ops.store(acc[{', '.join(['0'] * len(src.pack.lead))}, :, :], "
            "l2_c[...])."
        )
    if not isinstance(src, (Buffer, BufferSlice)):
        raise TypeError(
            f"air.api.ops.store drains a buffer, so its first argument must be "
            f"one from air.alloc() or a region of one; got {type(src).__name__}"
        )
    src_ep = _endpoint(src, "store", "source")
    dst_ep = _endpoint(dst, "store", "destination")
    _check_pair(dst_ep, src_ep, "store")

    # Being the destination of a store is what makes a tensor an output, which
    # in turn fixes the kernel's calling convention (inputs first, then
    # outputs). A store into L2 is staging, not an output, so it must not count.
    if dst_ep.tensor is not None:
        dst_ep.tensor.is_output = True

    return Token(_emit_dma(dst_ep, src_ep))


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


def cast(x, dtype):
    """Convert an elementwise expression to another element type.

        l1_out[:] = air.api.ops.cast(l1_in[:], air.api.i32)

    This is the one node whose operand has a different element type from its
    result, so everything below it is read, and computed on, in the source type:
    ``cast(a[:] * 2.0, i32)`` doubles in f32 and converts once, while
    ``cast(a[:], i32) * 2`` converts first and doubles in i32.

    Which ``arith`` op it becomes follows from the two types -- ``sitofp``,
    ``fptosi``, ``extf``, ``truncf`` or ``extsi``. Every one of those was run on
    npu1 hardware against an exact numpy reference before being allowed here,
    vectorised and scalar, because compiling is not the same claim as computing:
    of the pairs probed, four compiled cleanly and returned wrong values.

    Two results of that sweep are worth knowing before using this:

    * **A float result is rounded toward negative infinity, not to nearest.**
      Converting to bf16 differs from numpy's round-half-to-even by up to one
      ULP (2^-7 relative). That is not a property of this op -- a plain
      ``c[:] = a[:] + b[:]`` on bf16 buffers differs from numpy the same way --
      but an exact comparison against a numpy reference will fail on roughly
      half of all inputs, so compare with a tolerance.
    * **Narrowing between integer types is refused**, not supported-with-caveats.
      See ``_conversion_op`` below for the measurement.
    """
    from .types import DType

    if not isinstance(dtype, DType):
        raise TypeError(
            f"air.api.ops.cast needs an air.api element type as its second "
            f"argument (air.api.i32, air.api.f32, ...), got {dtype!r}"
        )
    expr = BufferExpr.coerce(x)
    if expr.kind == "compare":
        # A comparison evaluates to i1, not to an element type, so there is
        # nothing to convert *from* -- and a conversion op applied to a
        # vector<Wxi1> would be a verifier failure well downstream of the
        # mistake. Say what to do instead.
        raise TypeError(
            "air.api.ops.cast got a comparison, which is a predicate rather "
            "than a value: it evaluates to i1, not to an element type. Turn it "
            "into values first with air.api.ops.select(cond, a, b), or write "
            "the comparison against already-converted operands"
        )
    source = expr.element_dtype()
    if source is None:
        # Two different situations produce no source type, and they need
        # different messages because they need different fixes.
        if expr.leaves():
            # `cast(ops.select(a[:] > 0, 1, 2), i32)`. There *are* buffers, but
            # only in the predicate: ops.select allows that, and the values it
            # chooses between are both scalars, so the expression takes its
            # element type from whatever surrounds it. There is nothing to
            # convert from, and it cannot simply be adopted into the target
            # type either -- the predicate's own operands would then have to
            # be i32 as well, which is not what was written.
            raise TypeError(
                "air.api.ops.cast cannot convert this expression: its buffers "
                "appear only in a comparison, and the values it evaluates to "
                "are all scalars, so it has no element type of its own to "
                "convert from. Cast the operands instead -- for example "
                "ops.select(a[:] > 0, ops.cast(x[:], t), ops.cast(y[:], t))"
            )
        # `cast(1.0, i32)` -- no buffer at all, so the constant would simply be
        # built in the target type anyway. Write the constant you meant.
        raise TypeError(
            "air.api.ops.cast needs an expression containing at least one "
            f"buffer; {x!r} is a bare scalar, which is already built in "
            "whatever element type surrounds it"
        )
    if source is dtype:
        # Converting to the type it already has. numpy's astype does this too,
        # and it must not fall through to a builder: arith rejects a same-type
        # extf with "operand type and result type are cast incompatible".
        return expr
    require_signless(source, "air.api.ops.cast")
    require_signless(dtype, "air.api.ops.cast")
    _conversion_op(source, dtype)  # raises here, at the call site, not at emit
    return BufferExpr("cast", args=(expr,), dtype=dtype)


def _conversion_op(source, target):
    """The arith op converting ``source`` to ``target``, or a refusal.

    Shared by ``cast`` (so a bad pair is rejected where the user wrote it) and
    by the emitter (so it does not need a second copy of the rules).
    """
    from air.dialects import arith

    if source.is_float and target.is_float:
        if target.itemsize > source.itemsize:
            return arith.ExtFOp
        if target.itemsize < source.itemsize:
            return arith.TruncFOp
        # bf16 and f16 are both 2 bytes, so neither extf nor truncf applies:
        # they have different exponent widths and MLIR has no direct op.
        raise NotImplementedError(
            f"air.api.ops.cast cannot convert {source} to {target} directly: "
            f"they are the same width, so it is neither a widening nor a "
            f"narrowing, and arith has no op for it. Convert through "
            f"air.api.f32 in two steps"
        )
    if source.is_float:
        return arith.FPToSIOp
    if target.is_float:
        return arith.SIToFPOp
    if target.itemsize > source.itemsize:
        return arith.ExtSIOp
    # Narrowing int -> int. Refused on evidence rather than on principle:
    # measured on npu1, arith.trunci wraps on the scalar path (matching both
    # MLIR's own semantics and numpy) and *saturates* on the vector path --
    # i32 933033 came back as 32767 rather than 15529. Which path the emitter
    # takes depends on whether the tile is a multiple of the vector width,
    # which is not something the author of the cast is thinking about, so the
    # same source would compute two different things depending on the tile
    # size. In-range values agree, but nothing here can check that.
    raise NotImplementedError(
        f"air.api.ops.cast will not narrow {source} to {target}: on AIE the "
        f"vectorised arith.trunci saturates while the scalar one wraps, and "
        f"the emitter picks between them based on the tile size, so the result "
        f"would depend on a tile shape rather than on the program. Mask "
        f"explicitly if wrapping is what you want"
    )


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
    # The expression's own element type, not the first buffer under it. Those
    # agree everywhere except across a cast, where `relu(cast(a[:], i32))` has
    # an f32 leaf and an i32 result -- and it is the result that decides which
    # zero to build.
    dtype = expr.element_dtype()
    if dtype is None:
        # No value type of its own. That is a bare scalar, which relu cannot
        # shape -- or an expression whose buffers sit only in a predicate, as
        # in `relu(ops.select(a[:] > 0, 1, 2))`, which ops.select allows and
        # which the emitter shapes from those buffers. Fall back to a leaf,
        # which is the type the surrounding region will be built in anyway,
        # since every leaf in a region is required to match it.
        leaves = expr.leaves()
        if not leaves:
            raise ValueError("air.api.ops.relu needs a buffer operand, got a scalar")
        dtype = leaves[0].dtype
    return maximum(expr, 0.0 if dtype.is_float else 0)


# ---------------------------------------------------------------------------
# Comparison and select
# ---------------------------------------------------------------------------

# Float comparisons are the *ordered* predicates (OGE, not UGE): ordered means
# the result is false when either operand is NaN, which is what C's `>=` does
# and what the hand-written vector_select kernel asked for by name.
_COMPARISONS = {"lt", "le", "gt", "ge", "eq", "ne"}


def _comparison(name, key, a, b):
    for operand, pos in ((a, "first"), (b, "second")):
        if isinstance(operand, bool) or not isinstance(
            operand, (Buffer, BufferExpr, int, float)
        ):
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
    return BufferExpr("compare", op=key, args=(a, b))


def equal(a, b):
    """Elementwise ==. Lowers to arith.cmpf OEQ (float) / arith.cmpi eq (int).

    Spelled as a function rather than ``==`` on purpose -- see the note on
    ``BufferExpr.__eq__`` in ``_value.py``. The ordering comparisons (``<``,
    ``<=``, ``>``, ``>=``) *are* available as operators.
    """
    return _comparison("equal", "eq", a, b)


def not_equal(a, b):
    """Elementwise !=. Lowers to arith.cmpf ONE (float) / arith.cmpi ne (int)."""
    return _comparison("not_equal", "ne", a, b)


def select(cond, a, b):
    """Elementwise ``cond ? a : b``. Lowers to arith.select.

    ``cond`` must be a comparison -- ``x[:] >= y[:]``, or ``ops.equal(x, y)`` --
    because that is the only thing in this expression language whose result type
    is i1. The predicate is emitted as an arith.cmpf/cmpi feeding an
    arith.select, which is the pair the hand-written kernel spells out.

    Note that ``select(a >= b, a, b)`` is ``maximum(a, b)`` and lowers to two
    ops where ``ops.maximum`` lowers to one. Both spellings exist because the
    vector_select and vector_max examples exist to compare them; prefer
    ``ops.maximum`` unless the point is the select itself.
    """
    if isinstance(cond, bool):
        raise TypeError(
            "air.api.ops.select got a plain bool as its condition. `==` and "
            "`!=` on buffer expressions are Python identity comparisons, not "
            "elementwise ones, and evaluate to a bool before select ever sees "
            "them -- use air.api.ops.equal / not_equal for those, or one of "
            "the ordering operators <, <=, >, >= which do build a predicate"
        )
    if not isinstance(cond, BufferExpr) or cond.kind != "compare":
        raise TypeError(
            "air.api.ops.select expects a comparison as its condition (for "
            f"example x[:] >= y[:], or ops.equal(x[:], y[:])), got "
            f"{type(cond).__name__}"
        )
    for operand, pos in ((a, "second"), (b, "third")):
        if isinstance(operand, bool) or not isinstance(
            operand, (Buffer, BufferExpr, int, float)
        ):
            raise TypeError(
                f"air.api.ops.select expects a buffer slice or a numeric scalar "
                f"as its {pos} argument, got {type(operand).__name__}"
            )
    a, b = BufferExpr.coerce(a), BufferExpr.coerce(b)
    if not a.leaves() and not b.leaves() and not cond.leaves():
        raise ValueError(
            "air.api.ops.select needs at least one buffer operand; every "
            "argument is a scalar, which the emitter cannot shape"
        )
    return BufferExpr("select", op="select", args=(cond, a, b))


def fma(a, b, c):
    """Elementwise ``a * b + c`` as one operation. Lowers to vector.fma.
    Float only, and **vectorised only** -- see the constraints below.

    This is not a shorthand for ``a * b + c``: it is a different computation.
    Writing the product and the sum as two arith ops rounds twice, once when
    ``a * b`` is stored into an intermediate of the buffer's element type and
    again after the add. An fma rounds once, carrying the full-width product
    into the addition. On bf16, whose 8-bit significand loses the low half of
    nearly every product, that is a difference you can see in the output
    rather than a last-ulp curiosity.

    Two constraints, both measured rather than assumed, and both narrower than
    they first appear:

    * **There is no scalar form.** AIE2 has no scalar fma instruction, so
      math.fma reaches the backend and fails to legalize -- on npu1 and npu2,
      bf16 and f32, with and without bf16 emulation. The emitter refuses its
      own scalar fallback here rather than emitting one that cannot compile.
    * **f32 needs bf16 emulation.** ``vector.fma`` on ``vector<16xf32>`` is
      *explicitly marked illegal* by the aievec conversion, so a native f32
      kernel fails with "failed to legalize operation 'vector.fma'". Under
      ``bf16_emulation=True`` it lowers through bf16 and compiles.

    So this does **not** rescue f32 from the limitation the axpy conversion
    documents -- a chained f32 multiply-then-add does not legalize either
    (``<16 x s32> = G_FMUL``), and f32 has no vectorised route to a
    multiply-add by any spelling. What ops.fma buys is the single rounding on
    bf16, which is what the hand-written vector_fma kernel was written for.

    The third argument is the addend, matching C's ``fma(x, y, z)`` and MLIR's
    operand order; there is no operator spelling because Python has none.
    """
    for operand, pos in ((a, "first"), (b, "second"), (c, "third")):
        if not isinstance(operand, (Buffer, BufferExpr, int, float)):
            raise TypeError(
                f"air.api.ops.fma expects a buffer slice or a numeric scalar "
                f"as its {pos} argument, got {type(operand).__name__}"
            )
        # A comparison is the one BufferExpr whose result is i1 rather than the
        # element type. Only ops.select can consume one; feeding it to an fma
        # would build an arith op over a predicate and fail deep in the MLIR
        # verifier, naming an SSA value rather than the argument at fault.
        if isinstance(operand, BufferExpr) and operand.kind == "compare":
            raise TypeError(
                f"air.api.ops.fma got a comparison as its {pos} argument. A "
                "comparison is a predicate (i1), not a value, so it cannot be "
                "multiplied or added -- pass it through air.api.ops.select "
                "first to choose between two values"
            )
    a, b, c = BufferExpr.coerce(a), BufferExpr.coerce(b), BufferExpr.coerce(c)
    if not a.leaves() and not b.leaves() and not c.leaves():
        raise ValueError(
            "air.api.ops.fma needs at least one buffer operand; every argument "
            "is a scalar, which the emitter cannot shape"
        )
    return BufferExpr("fma", op="fma", args=(a, b, c))


def _reduce(name, key, x):
    if not isinstance(x, (Buffer, BufferExpr)):
        raise TypeError(
            f"air.api.ops.{name} expects a buffer slice, got {type(x).__name__}"
        )
    expr = BufferExpr.coerce(x)
    if expr.kind == "compare":
        raise TypeError(
            f"air.api.ops.{name} got a comparison, which is a predicate (i1) "
            "rather than a value. Pass it through air.api.ops.select first"
        )
    if expr.kind == "reduce":
        raise NotImplementedError(
            f"air.api.ops.{name} cannot reduce a reduction: the inner one has "
            "already collapsed the innermost dimension to 1, so the outer one "
            "would be a no-op over a single element. Reducing a second axis is "
            "not supported -- store the first result and reduce that"
        )
    if not expr.leaves():
        raise ValueError(
            f"air.api.ops.{name} needs a buffer operand, got a scalar; there "
            "is no shape to reduce over"
        )
    return BufferExpr("reduce", op=key, args=(expr,))


def reduce_add(x):
    """Sum along the innermost axis. Lowers to vector.reduction <add>.

    The destination keeps the operand's rank with its innermost dimension set
    to 1, which is numpy's ``keepdims=True`` and is what the hand-written
    kernels allocate (``[tile_m, n]`` in, ``[tile_m, 1]`` out)::

        acc = air.alloc([tile_m, 1], bf16, scope=h.private())
        acc[:] = air.ops.reduce_add(a[:])

    This is the only shape-changing operation in the expression language, so
    it has to be the whole right-hand side -- it cannot appear inside a larger
    expression, because the surrounding elementwise ops would be operating on
    two different shapes. Its *operand* may be any elementwise expression,
    which is what makes ``reduce_add(a[:] * b[:])`` a row-wise dot product.

    The whole innermost axis is read as a single vector, matching the
    predecessor kernels: there is no accumulation loop, and therefore no
    loop-carried vector accumulator, which is the construct ``ops.dot``
    documents as unlegalizable on AIE2. The cost is that the axis length has
    to be a vector width the target accepts.
    """
    return _reduce("reduce_add", "add", x)


def reduce_max(x):
    """Maximum along the innermost axis. Lowers to vector.reduction <maximumf>
    (float) or <maxsi> (signed integer). See :func:`reduce_add` for the shape
    rule and why the reduction must be the whole right-hand side.
    """
    return _reduce("reduce_max", "max", x)


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


# ---------------------------------------------------------------------------
# Matrix multiply-accumulate
# ---------------------------------------------------------------------------


# Rank dispatch, following numpy.dot / tl.dot rather than linalg's naming: `dot`
# is the contraction, and which linalg op carries it depends on the operand
# ranks. The rule is uniform -- contract a's last axis against b's first, and the
# accumulator keeps what is left of each -- so all four cases fall out of one
# shape computation.
_CONTRACTIONS = {
    (1, 1): ("dot", "(k,) . (k,) -> ()"),
    (1, 2): ("vecmat", "(k,) @ (k, n) -> (n,)"),
    (2, 1): ("matvec", "(m, k) @ (k,) -> (m,)"),
    (2, 2): ("matmul", "(m, k) @ (k, n) -> (m, n)"),
}


# The micro-tiled contraction. `linalg` has no named op for it -- the reference
# example defines it with the OpDSL, and so does this, from the same index
# expression. It is built lazily because the OpDSL decorator runs once and
# importing linalg.opdsl at module scope would pull it in for every program
# regardless of whether anything contracts.
_BLOCK_MATMUL = None


def _block_matmul_op():
    global _BLOCK_MATMUL
    if _BLOCK_MATMUL is not None:
        return _BLOCK_MATMUL

    import air.dialects.linalg.opdsl.lang as lang
    from air.dialects.linalg.opdsl.lang import (
        D,
        S,
        TensorDef,
        TypeFn,
        domain,
        linalg_structured_op,
    )

    @linalg_structured_op()
    def block_matmul(
        A=TensorDef(lang.TV.T1, S.a, S.c, S.f, S.d, S.g, S.i),
        B=TensorDef(lang.TV.T2, S.b, S.c, S.e, S.f, S.i, S.h),
        C=TensorDef(lang.TV.U, S.b, S.a, S.e, S.d, S.g, S.h, output=True),
    ):
        domain(D.a, D.b, D.c, D.d, D.e, D.f, D.g, D.h, D.i)
        C[D.b, D.a, D.e, D.d, D.g, D.h] += TypeFn.cast_signed(
            lang.TV.U, A[D.a, D.c, D.f, D.d, D.g, D.i]
        ) * TypeFn.cast_signed(lang.TV.U, B[D.b, D.c, D.e, D.f, D.i, D.h])

    _BLOCK_MATMUL = block_matmul
    return _BLOCK_MATMUL


def accumulator_subview(acc):
    """The calling core's slab of a micro-tiled accumulator.

    A herd-shared accumulator is one memref with a leading dimension per herd
    axis, and a core may only touch its own slab -- there is no choice to make,
    so the DSL emits the ``memref.subview`` itself rather than asking for
    coordinates it could only re-check. A per-core accumulator has leading
    dimensions of 1 and needs no subview in principle; one is emitted anyway,
    because that is the op the transform script's tile sizes are stated against.
    """
    from air.dialects.memref import subview

    from ._trace import current_herd

    nlead = len(acc.pack.lead)
    coords = current_herd()._coords
    if len(coords) > nlead:
        raise ValueError(
            f"the accumulator has {nlead} leading dimension(s) but the herd is "
            f"{len(coords)}-D; a herd-shared accumulator needs one leading "
            "dimension per herd axis so that every core has its own slab"
        )
    offsets = [c.materialize() for c in coords]
    offsets += [0] * (len(acc.shape) - len(offsets))
    sizes = [1] * nlead + list(acc.shape[nlead:])
    return subview(
        acc.value, offsets=offsets, sizes=sizes, strides=[1] * len(acc.shape)
    )


def _check_packed_operands(a, b, acc):
    """The three micro-tiles have to agree on m, k, n and on their extents."""
    packs = {"a": a.pack, "b": b.pack, "acc": acc.pack}
    missing = [n for n, p in packs.items() if p is None]
    if missing:
        raise TypeError(
            f"air.api.ops.dot got rank-6 operands, which means the micro-tiled "
            f"contraction, but {', '.join(missing)} "
            f"{'was' if len(missing) == 1 else 'were'} not allocated with a "
            "micro-tiled shape. Allocate all three from the same air.micro_tile, "
            "e.g. air.alloc(mm.a(tile_m, tile_k), bf16, scope=h.private())."
        )
    roles = {"a": "A", "b": "B", "acc": "C"}
    for name, want in roles.items():
        if packs[name].role != want:
            raise ValueError(
                f"air.api.ops.dot expects {name} to be a micro-tiled {want} "
                f"operand (mm.{want.lower()}(...)), but it was built as "
                f"{packs[name].role}"
            )
    micros = {n: p.micro for n, p in packs.items()}
    if len(set(micros.values())) != 1:
        raise ValueError(
            "air.api.ops.dot needs one micro-tile across all three operands, "
            f"got a={micros['a']!r}, b={micros['b']!r}, acc={micros['acc']!r}. "
            "The micro-tile is the shape of the hardware intrinsic; mixing them "
            "would silently contract the wrong elements."
        )
    (m_a, k_a), (k_b, n_b), (m_c, n_c) = (
        packs["a"].logical,
        packs["b"].logical,
        packs["acc"].logical,
    )
    if k_a != k_b:
        raise ValueError(
            f"air.api.ops.dot shape mismatch: a is {m_a}x{k_a} and b is "
            f"{k_b}x{n_b}; the contracting extents must agree"
        )
    if (m_c, n_c) != (m_a, n_b):
        raise ValueError(
            f"air.api.ops.dot shape mismatch: a @ b is {m_a}x{n_b} but acc is "
            f"{m_c}x{n_c}"
        )


def dot(a, b, acc=None, alpha=1.0, transpose_b=False, dependency=None, *, kernel=None):
    """``acc += a @ b`` over L1 tiles. Returns a :class:`Token`.

    This is a *statement*, not an expression, and the accumulator is a buffer
    rather than a value -- the signature the API proposal specified, and the
    right one for the hardware. Accumulating into memory is what
    ``linalg.matmul``'s ``outs=`` means, and it avoids a loop-carried SSA vector
    accumulator, which LLVM splits into sub-512-bit pieces the AIE2 backend
    cannot legalize.

    What is emitted is a plain ``linalg.matmul``. That is deliberate and is what
    every matmul example in this tree does: none of them hand-writes
    ``vector.contract``. Emitting linalg is the fork point that keeps both
    lowering strategies available -- ``transform.air.herd_vectorize`` for direct
    codegen, or ``lower_linalg_to_func`` to call an external kernel -- and
    hand-writing the vectorised form would forfeit the second.

    Mixed precision is expected: ``a``/``b`` are typically bf16 and ``acc`` f32.

    The operand ranks pick the contraction, as ``numpy.dot`` does -- ``dot`` here
    means the contraction, not specifically a vector inner product:

    ==============  ==========================  =================
    ranks           shapes                      linalg op
    ==============  ==========================  =================
    1-D x 1-D       ``(k,) . (k,) -> ()``       ``linalg.dot``
    1-D x 2-D       ``(k,) @ (k,n) -> (n,)``    ``linalg.vecmat``
    2-D x 1-D       ``(m,k) @ (k,) -> (m,)``    ``linalg.matvec``
    2-D x 2-D       ``(m,k) @ (k,n) -> (m,n)``  ``linalg.matmul``
    ==============  ==========================  =================

    One rule covers all four: ``a``'s last axis contracts against ``b``'s first,
    and ``acc`` keeps what is left of each.

    ``transpose_b=True`` says B is stored ``[n, k]`` rather than ``[k, n]``, so
    ``a``'s last axis contracts against b's last. It is emitted as a named
    ``linalg.matmul`` carrying the transpose in its indexing maps -- not the
    deprecated ``linalg.matmul_transpose_b``, and not a ``linalg.generic``.
    Nothing downstream needs to know: ``air-to-aie`` matches no named
    contraction, and both ``lower_linalg_to_func`` and
    ``transform.air.herd_vectorize`` go through the ``LinalgOp`` interface,
    which is indexing-map agnostic.

    Two limits worth stating. It is verified through the scalar path -- lowered
    by ``convert-linalg-to-loops`` and run on npu1 -- and *not* through the AIE2
    matmul intrinsic, which wants micro-tiled operands whose layout the DMA pack
    (``mm.b(...)``) already fixes. And under ``lower_linalg_to_func`` an external
    kernel built for ``[k, n]`` links happily against an ``[n, k]`` operand and
    computes silently wrong results, so pass ``kernel=`` to give the transposed
    form its own symbol.

    ``kernel=`` is keyword-only, and sits after ``dependency`` so that every
    existing positional binding is unchanged: inserting it earlier would have
    silently rebound a positionally-passed ``dependency`` to it. It names the
    external function this contraction should lower to
    under ``lower_linalg_to_func``, by setting linalg's ``library_call``
    attribute. Without it a micro-tiled contraction lowers to
    ``op_has_no_registered_library_name`` -- MLIR's placeholder for an op with no
    registered name, which the OpDSL emitter never overrides and which every
    hand-written kernel here therefore exports. Sharing one symbol means a
    kernel compiled for the wrong tile dimensions still links and computes
    silently wrong results, and two differently shaped contractions cannot
    coexist in one core. Naming the kernel makes both of those link errors::

        ops.dot(a, b, acc=acc, kernel="matmul_bf16_bf16_m32k16n32")
    """
    from air.dialects import linalg

    _check_dependency(dependency)
    for name, buf in (("a", a), ("b", b), ("acc", acc)):
        if buf is None:
            raise TypeError(f"air.api.ops.dot requires {name}=")
        if not isinstance(buf, Buffer):
            raise TypeError(
                f"air.api.ops.dot expects {name} to be an L1 buffer from "
                f"air.alloc(), got {type(buf).__name__}"
            )
        if buf.space != "L1":
            # A contraction runs on a core, and only L1 is core-local. An L2
            # buffer reaching here would emit a contraction over memtile memory,
            # which has DMA engines and no compute -- the same rule that makes
            # an elementwise read of an L2 buffer an error.
            raise TypeError(
                f"air.api.ops.dot expects {name} to be an L1 buffer, but it is "
                f"in {buf.space}: a memtile has DMA engines and no compute "
                "core, so it cannot be contracted over. Stage the tile into L1 "
                "with air.api.ops.load first."
            )
        if buf.value is None:
            raise RuntimeError(f"air.api.ops.dot: {name} used before allocation")
        # A named linalg contraction builds its region from arith ops --
        # arith.extsi and arith.muli for an integer operand -- so an unsigned
        # element type fails verification inside the op that was just emitted,
        # several frames from the call that caused it.
        require_signless(buf.dtype, f"air.api.ops.dot's {name} operand")
        require_computable(buf.dtype, f"air.api.ops.dot's {name} operand")

    if alpha != 1.0:
        raise NotImplementedError(
            "air.api.ops.dot(alpha=...) is not implemented yet; it needs a "
            "scaled linalg.generic rather than a named contraction"
        )

    ranks = (len(a.shape), len(b.shape))
    if ranks == (6, 6):
        if transpose_b:
            raise NotImplementedError(
                "air.api.ops.dot(transpose_b=True) is not implemented for "
                "micro-tiled operands; pack B transposed instead"
            )
        _check_packed_operands(a, b, acc)
        op = _block_matmul_op()(a.value, b.value, outs=[accumulator_subview(acc)])
        _set_library_call(kernel)
        return Token(op)

    if ranks not in _CONTRACTIONS:
        raise NotImplementedError(
            f"air.api.ops.dot contracts 1-D and 2-D tiles; got ranks {ranks} "
            f"(a is {a.shape}, b is {b.shape}). There is no named linalg op past "
            "one batch dimension, and a batch axis belongs in the herd grid if "
            "it is spatial or in air.sequential if it is temporal -- putting it "
            "inside the contraction hides the schedule."
        )
    op_name, signature = _CONTRACTIONS[ranks]

    if transpose_b and ranks != (2, 2):
        raise ValueError(
            f"air.api.ops.dot(transpose_b=True) is meaningless for {signature}; "
            "it transposes a matrix operand"
        )
    # a's last axis contracts against b's first -- or against b's last, when B
    # is stored transposed. acc keeps what is left of each.
    k_a = a.shape[-1]
    k_b = b.shape[-1] if transpose_b else b.shape[0]
    if k_a != k_b:
        raise ValueError(
            f"air.api.ops.dot shape mismatch for {signature}: a is {a.shape} and "
            f"b is {b.shape}; a's contracting dimension ({k_a}) must equal b's "
            f"({k_b})"
            + (" -- with transpose_b=True, b is [n, k]" if transpose_b else "")
        )
    expected = tuple(a.shape[:-1]) + (
        tuple(b.shape[:-1]) if transpose_b else tuple(b.shape[1:])
    )
    if tuple(acc.shape) != expected:
        raise ValueError(
            f"air.api.ops.dot shape mismatch for {signature}: a . b is "
            f"{expected} but acc is {tuple(acc.shape)}"
        )

    if transpose_b:
        # A named linalg.matmul carrying the transpose in its indexing maps,
        # rather than the deprecated linalg.matmul_transpose_b. Downstream this
        # is the same op: air-to-aie matches no named contraction, and both
        # `lower_linalg_to_func` and transform.air.herd_vectorize go through the
        # LinalgOp interface, which is indexing-map agnostic.
        op = linalg.matmul(
            a.value, b.value, outs=[acc.value], indexing_maps=_transpose_b_maps()
        )
    else:
        op = getattr(linalg, op_name)(a.value, b.value, outs=[acc.value])
    _set_library_call(kernel)
    return Token(op)


def _transpose_b_maps():
    """Indexing maps for ``acc[m, n] += a[m, k] * b[n, k]``.

    Raw ``AffineMap``s, not ``AffineMapAttr``s: the linalg wrapper's type hint
    says otherwise but it wraps them itself, and pre-wrapping raises.
    """
    from air.ir import AffineDimExpr, AffineMap

    m, n, k = (AffineDimExpr.get(i) for i in range(3))
    return [
        AffineMap.get(3, 0, [m, k]),
        AffineMap.get(3, 0, [n, k]),
        AffineMap.get(3, 0, [m, n]),
    ]


def _set_library_call(kernel):
    """Stamp ``library_call`` on the contraction just emitted.

    The OpDSL emitter hardcodes ``library_call=None`` (a standing TODO in
    upstream ``linalg/opdsl/lang/emitter.py``) and returns the op's *results*,
    which is an empty list for memref semantics -- so there is no handle to set
    the attribute on. The op is the last one written into the current block,
    which is where this reaches for it.
    """
    if kernel is None:
        return
    if not isinstance(kernel, str) or not kernel:
        raise TypeError(
            f"air.api.ops.dot(kernel=...) takes the external function's symbol "
            f"name, got {type(kernel).__name__}"
        )
    from air.ir import InsertionPoint, StringAttr

    block = InsertionPoint.current.block
    block.operations[len(block.operations) - 1].attributes["library_call"] = (
        StringAttr.get(kernel)
    )


def _unimplemented(name, needs):
    def stub(*args, **kwargs):
        raise NotImplementedError(
            f"air.api.ops.{name} is not implemented yet (needs {needs})"
        )

    stub.__name__ = name
    return stub


# Named so that a program using them fails loudly at the call site rather than
# at compile time with a confusing IR error.
# math.exp exists in the bindings, but nothing needs it yet and it has not
# been checked for an aievec lowering -- untested surface is worse than none.
exp = _unimplemented("exp", "a checked aievec lowering; use ops.tanh, which has one")
reduce = _unimplemented("reduce", "a reduction emitter")
stack = _unimplemented("stack", "multi-buffer concatenation")
dequant = _unimplemented("dequant", "BlockType support")
atomic_add = _unimplemented("atomic_add", "CacheDomain support")
