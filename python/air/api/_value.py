# ./python/air/api/_value.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Values the DSL manipulates: global tensors, L1 buffers, and slices of them.

Three rules shape this module:

  * A :class:`Tensor` is a host-visible L3 array and is never read or written
    directly -- only sliced, and handed to ``air.api.ops.load``/``store``.
  * A :class:`Buffer` is an L1 tile that lives inside a herd body. Reading it
    (``a_buf[:]``) yields a lazy :class:`BufferExpr` rather than emitting
    anything; only assignment (``c_buf[:] = expr``) emits code, so a whole
    expression tree is lowered as one vectorised loop.
  * Anything the API cannot express raises. There is no silent absorption of
    unsupported arithmetic -- that was the failure mode of the stub tracer this
    package replaces.
"""

from ._index import coerce_index

__all__ = ["Token", "Tensor", "TensorSlice", "Buffer", "BufferSlice", "BufferExpr"]


class Token:
    """Completion token returned by memory ops.

    AIR's own asynchrony is built by the ``air-dependency`` pass from the
    program order this tracer emits, so a v1 token carries no SSA value. It
    exists so that ``dependency=`` arguments can be *validated* -- passing a
    non-token is an error rather than being quietly ignored.
    """

    __slots__ = ("op",)

    def __init__(self, op=None):
        self.op = op

    def __repr__(self):
        return "air.api.Token()"


def _row_major_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _normalize_key(key, rank, what):
    """Expand a subscript into exactly ``rank`` slice/int entries."""
    if not isinstance(key, tuple):
        key = (key,)
    if Ellipsis in key:
        raise NotImplementedError(f"`...` indexing of {what} is not supported yet")
    if len(key) != rank:
        raise IndexError(
            f"{what} has rank {rank} but was indexed with {len(key)} "
            f"subscript{'s' if len(key) != 1 else ''}"
        )
    return key


def _permute(seq, axes):
    return [seq[a] for a in axes]


def _check_axes(axes, rank):
    """``transpose`` takes a full permutation, as numpy does."""
    if sorted(axes) != list(range(rank)):
        raise ValueError(
            f"transpose{tuple(axes)} is not a permutation of a rank-{rank} "
            f"view: give every axis exactly once, e.g. "
            f"transpose{tuple(range(rank))}. numpy has the same rule -- there "
            "is no partial transpose"
        )


def _carries_offset(offset):
    """True unless the offset is a compile-time zero.

    A runtime offset counts as carrying one: it may well be non-zero, and a
    reshape has no way to prove otherwise. This has to go through as_const()
    rather than comparing to 0 -- ``offset != 0`` on an IndexExpr builds a
    Condition for ``ops.branch``, not a bool, and Condition refuses ``bool()``, so
    the comparison would raise on every offset rather than answer the question.
    """
    value = offset.as_const() if hasattr(offset, "as_const") else offset
    return value != 0


def _reshape_pattern(sizes, strides, offsets, shape):
    """Re-express one walk at a different rank, or refuse.

    numpy's reshape silently copies when it cannot produce a view. Here a copy
    would be a hidden L2-to-L2 transfer nobody asked for, so this raises
    instead. What it can do is split and merge axes: input and output axes are
    matched into groups of equal element count, each group's input axes must be
    contiguous with one another, and the group's output strides are then laid
    out row-major from its innermost input stride.

    Offsets follow the same grouping. A group's total offset, in elements, is
    placed on its innermost output axis -- the one carrying the innermost
    stride -- which for the common case of splitting one axis is just that
    axis's own offset, unchanged.
    """
    shape = [int(e) for e in shape]
    for e in shape:
        if e < 1:
            raise ValueError(f"reshape extents must be positive, got {tuple(shape)}")
    n_in, n_out = 1, 1
    for e in sizes:
        n_in *= e
    for e in shape:
        n_out *= e
    if n_in != n_out:
        raise ValueError(
            f"cannot reshape a {tuple(sizes)} view ({n_in} elements) to "
            f"{tuple(shape)} ({n_out} elements)"
        )

    if list(sizes) == shape:
        return list(offsets), list(sizes), list(strides)

    out_sizes, out_strides, out_offsets = [], [], []
    i = j = 0
    while i < len(sizes) or j < len(shape):
        # Unit axes pair up positionally: a size-1 input axis maps onto a
        # size-1 output axis and keeps its own offset and stride. Pairing does
        # not consult the offset, because which axis an offset sits on is not
        # the reshape's to decide -- a staging buffer indexed per core carries a
        # herd coordinate here, and it has to come out on the axis it went in
        # on.
        if i < len(sizes) and j < len(shape) and sizes[i] == 1 and shape[j] == 1:
            out_sizes.append(1)
            out_strides.append(strides[i])
            out_offsets.append(offsets[i])
            i += 1
            j += 1
            continue
        # A unit input axis with no unit output axis left to pair with holds no
        # elements, so it can be dropped -- but only when it carries no offset,
        # since dropping a non-zero one would move the window.
        if i < len(sizes) and sizes[i] == 1 and not _carries_offset(offsets[i]):
            i += 1
            continue
        if j < len(shape) and shape[j] == 1:
            # A size-1 axis is never stepped, so its stride is arbitrary. Left
            # as None and filled in row-major below, which is the value the
            # hand-written examples carry there.
            out_sizes.append(1)
            out_strides.append(None)
            out_offsets.append(0)
            j += 1
            continue
        if i >= len(sizes) or j >= len(shape):
            raise ValueError(
                f"cannot reshape a {tuple(sizes)} view to {tuple(shape)} "
                "without copying"
            )
        # Grow a group on each side until the element counts agree.
        i0, j0 = i, j
        c_in, c_out = sizes[i], shape[j]
        i += 1
        j += 1
        while c_in != c_out:
            if c_in < c_out:
                if i >= len(sizes):
                    raise ValueError(
                        f"cannot reshape a {tuple(sizes)} view to "
                        f"{tuple(shape)} without copying"
                    )
                c_in *= sizes[i]
                i += 1
            else:
                if j >= len(shape):
                    raise ValueError(
                        f"cannot reshape a {tuple(sizes)} view to "
                        f"{tuple(shape)} without copying"
                    )
                c_out *= shape[j]
                j += 1
        # Trailing unit axes belong to the group they were split from, not to
        # whatever follows. [3, 48] -> [3, 1, 6, 8] splits the first axis into
        # (3, 1), and the offset then lands on the innermost of the pair --
        # which is what keeps a split from scattering across two code paths.
        while j < len(shape) and shape[j] == 1:
            j += 1
        # The group's input axes have to be contiguous with each other, or no
        # single stride describes the merged walk.
        for d in range(i0, i - 1):
            if strides[d] != strides[d + 1] * sizes[d + 1]:
                raise ValueError(
                    f"cannot reshape a {tuple(sizes)} view with strides "
                    f"{tuple(strides)} to {tuple(shape)} without copying: axes "
                    f"{d} and {d + 1} are not contiguous with each other"
                )
        inner = strides[i - 1]
        # Row-major within the group, innermost first.
        group = [0] * (j - j0)
        run = inner
        for d in range(j - 1, j0 - 1, -1):
            group[d - j0] = run
            run *= shape[d]
        out_sizes.extend(shape[j0:j])
        out_strides.extend(group)
        # The group's offset, in elements, goes back onto its innermost axis --
        # the one that kept the parent's stride.
        if i - i0 == 1:
            # A pure split, which is the common case. `inner` is this axis's own
            # stride, so the offset passes through unchanged and stays symbolic:
            # a region indexed by a herd coordinate or a loop variable reshapes
            # without needing its offset to be known at trace time.
            out_offsets.extend([0] * (j - j0 - 1) + [offsets[i0]])
        else:
            # Merging axes really does combine their offsets, and there is no
            # way to divide that by `inner` without the numbers.
            total = 0
            for d in range(i0, i):
                total += _as_int(offsets[d], "reshape") * strides[d]
            if total % inner:
                raise ValueError(
                    f"cannot reshape a {tuple(sizes)} view without copying: its "
                    f"offset does not land on a boundary of {tuple(shape)}"
                )
            out_offsets.extend([0] * (j - j0 - 1) + [total // inner])

    # Row-major continuation for the unit axes, right to left.
    run = 1
    for d in range(len(out_sizes) - 1, -1, -1):
        if out_strides[d] is None:
            out_strides[d] = run
        run = out_sizes[d] * out_strides[d]
    return out_offsets, out_sizes, out_strides


def _as_int(offset, what):
    value = offset.as_const() if hasattr(offset, "as_const") else offset
    if value is None:
        raise ValueError(
            f"{what} needs compile-time offsets; this region's offset is a "
            "runtime value. Subscript with constant bounds, or reshape before "
            "subscripting"
        )
    return int(value)


class _StridedView:
    """``reshape`` and ``transpose`` over an (offsets, sizes, strides) triple.

    Shared by :class:`TensorSlice` and :class:`BufferSlice`, which differ only
    in what they are a region *of*. Neither constructor moves anything: a view
    re-describes the same elements at a different rank or in a different order,
    so all three lists are rewritten together and the walk is unchanged.
    """

    __slots__ = ()

    def _respan(self, offsets, sizes, strides, logical_sizes):
        """Build another region of the same thing. Implemented by each slice."""
        raise NotImplementedError

    def reshape(self, *shape):
        """This region's elements at a different rank, as a view.

        Splitting an axis is how a tile is laid out in the blocks a matmul
        instruction consumes: a [32, 32] region becomes [8, 4, 4, 8] and the
        walk is unchanged, only re-described. Raises rather than copying when
        no view exists -- see :func:`_reshape_pattern`.
        """
        if len(shape) == 1 and not isinstance(shape[0], int):
            shape = tuple(shape[0])
        offsets, sizes, strides = _reshape_pattern(
            self.sizes, self.strides, self.offsets, shape
        )
        return self._respan([coerce_index(o) for o in offsets], sizes, strides, sizes)

    def __getitem__(self, key):
        """A sub-region of this region, subscripted like the original array.

        A view is a region like any other, so it subscripts like one -- which is
        what lets an offset be chosen *after* the walk has been re-described.
        conv2d_14x14's L2 gather is the case: the pattern it needs is a
        reshape and a transpose of the whole memtile buffer, and only then a
        pick of one (row, block) out of it.
        """
        key = _normalize_key(key, len(self.sizes), "region")
        offsets, sizes = [], []
        for dim, (sub, extent) in enumerate(zip(key, self.sizes)):
            offset, size = _resolve_subscript(sub, extent, dim)
            offsets.append(offset)
            sizes.append(size)
        # Added to this region's own offsets, not scaled: an offset is an index
        # along its axis, which the transfer multiplies by that axis's stride --
        # the same convention Tensor.__getitem__ and Buffer.__getitem__ use, and
        # the reason this view keeps the strides it already had.
        combined = [
            coerce_index(base + off) for base, off in zip(self.offsets, offsets)
        ]
        return self._respan(combined, sizes, list(self.strides), sizes)

    def transpose(self, *axes):
        """This region walked with its axes permuted.

        Nothing moves: offsets, sizes and strides are permuted together, so the
        descriptor visits the same elements in a different order. Takes a full
        permutation, as numpy does.
        """
        if len(axes) == 1 and not isinstance(axes[0], int):
            axes = tuple(axes[0])
        axes = [int(a) for a in axes]
        _check_axes(axes, len(self.sizes))
        return self._respan(
            _permute(self.offsets, axes),
            _permute(self.sizes, axes),
            _permute(self.strides, axes),
            _permute(self.logical_sizes, axes),
        )


class _Reshapable:
    """``reshape``/``transpose`` on the whole of a Tensor or a Buffer.

    Both are the same two lines -- take the whole thing as a region, then view
    it -- and the region type is the only difference, which ``_whole_view``
    supplies.
    """

    __slots__ = ()

    def _whole_view(self):
        raise NotImplementedError

    def reshape(self, *shape):
        """The whole array at a different rank, as a view. See _StridedView."""
        return self._whole_view().reshape(*shape)

    def transpose(self, *axes):
        """The whole array with its axes permuted. See _StridedView."""
        return self._whole_view().transpose(*axes)


class Tensor(_Reshapable):
    """A host-visible array in L3. Becomes a ``func.func`` argument."""

    def __init__(self, shape, dtype, name=None):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.name = name
        self.strides = _row_major_strides(self.shape)
        # Bound by the tracer when the enclosing function is created.
        self.value = None
        # Set when this tensor is the destination of an ops.store().
        self.is_output = False

    def __getitem__(self, key):
        key = _normalize_key(key, len(self.shape), "tensor")
        offsets, sizes = [], []
        for dim, (sub, extent) in enumerate(zip(key, self.shape)):
            offset, size = _resolve_subscript(sub, extent, dim)
            offsets.append(offset)
            sizes.append(size)
        return TensorSlice(self, offsets, sizes, list(self.strides))

    def _whole_view(self):
        """The whole tensor as a region, for reshape/transpose to work on.

        Built directly rather than through __getitem__ so that a rank-N tensor
        does not need a rank-N full subscript written out first.
        """
        return TensorSlice(
            self,
            [coerce_index(0) for _ in self.shape],
            list(self.shape),
            list(self.strides),
        )

    def __setitem__(self, key, value):
        raise TypeError(
            "cannot assign into a global tensor directly; use "
            "air.api.ops.store(buffer, tensor[...]) instead"
        )

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


def _resolve_subscript(sub, extent, dim):
    """Turn one subscript into an (offset, static size) pair."""
    if isinstance(sub, slice):
        if sub.step not in (None, 1):
            raise NotImplementedError(
                f"strided slicing (step={sub.step}) is not supported yet"
            )
        start = coerce_index(0 if sub.start is None else sub.start)
        stop = coerce_index(extent if sub.stop is None else sub.stop)
        size = (stop - start).as_const()
        if size is None:
            raise ValueError(
                f"slice size along dim {dim} is not a compile-time constant "
                f"({stop} - {start}); air.api needs static tile sizes"
            )
        if size <= 0:
            raise ValueError(f"empty slice along dim {dim} (size {size})")
        return start, size
    # A bare integer index selects one element and keeps the dimension at 1.
    return coerce_index(sub), 1


class TensorSlice(_StridedView):
    """An access pattern into a :class:`Tensor`: offsets, static sizes, strides."""

    __slots__ = ("tensor", "offsets", "sizes", "strides", "logical_sizes", "is_view")

    def __init__(
        self, tensor, offsets, sizes, strides, logical_sizes=None, is_view=False
    ):
        self.tensor = tensor
        self.offsets = offsets
        self.sizes = sizes
        self.strides = strides
        # Same two fields, and the same meaning, as on BufferSlice: the region's
        # element shape, and whether reshape/transpose produced it -- which is
        # what tells a transfer to check element counts instead of axes.
        self.logical_sizes = list(sizes if logical_sizes is None else logical_sizes)
        self.is_view = is_view

    def _respan(self, offsets, sizes, strides, logical_sizes):
        return TensorSlice(
            self.tensor, offsets, sizes, strides, logical_sizes, is_view=True
        )

    @property
    def dtype(self):
        return self.tensor.dtype

    def materialize_offsets(self):
        return [o.materialize() for o in self.offsets]

    def __repr__(self):
        return f"TensorSlice({self.tensor.name}, sizes={self.sizes})"


class Buffer(_Reshapable):
    """A tile allocated in a hardware scope: L1 (a core) or L2 (a memtile).

    A subscript designates a *region* of the buffer, and what that means depends
    on where it is used -- the same rule the API proposal's examples assume. A
    whole-tile subscript (``buf[:]``) in an expression is an elementwise read; a
    partial subscript (``buf[tx, 0:tm, :]``) is an access pattern for a DMA, and
    only ``ops.load``/``ops.store`` accept one.
    """

    def __init__(
        self,
        shape,
        dtype,
        scope=None,
        vector_width=None,
        value=None,
        space="L1",
    ):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.scope = scope
        # "L1" (core-local) or "L2" (memtile). Recorded rather than derived from
        # `scope` so that _value.py stays independent of the tracer.
        self.space = space
        self.vector_width = (
            dtype.default_vector_width if vector_width is None else int(vector_width)
        )
        # The memref SSA value produced by memref.alloc.
        self.value = value
        self.strides = _row_major_strides(self.shape)
        # The memref.dealloc that ended this buffer's life, if air.dealloc was
        # called on it. None means the tracer will place one itself, after the
        # last use it observes.
        self.released = None

    # -- reading: whole tile is a lazy expression, a region is a DMA slice ---

    def __getitem__(self, key):
        if self._is_whole(key):
            self._require_compute("read")
            return BufferExpr.leaf(self)
        key = _normalize_key(key, len(self.shape), "buffer")
        offsets, sizes = [], []
        for dim, (sub, extent) in enumerate(zip(key, self.shape)):
            offset, size = _resolve_subscript(sub, extent, dim)
            offsets.append(offset)
            sizes.append(size)
        return BufferSlice(self, offsets, sizes, list(self.strides))

    def _whole_view(self):
        """The whole buffer as a region, for reshape/transpose to work on.

        Built directly rather than through __getitem__, where a full-extent
        subscript is an elementwise read rather than a region.
        """
        return BufferSlice(
            self,
            [coerce_index(0) for _ in self.shape],
            list(self.shape),
            list(self.strides),
        )

    # -- writing: this is what triggers emission ----------------------------

    def __setitem__(self, key, value):
        if not self._is_whole(key):
            raise NotImplementedError(
                "partial assignment into a buffer is not supported yet; a "
                "partial subscript names a DMA region, so use "
                "air.api.ops.load(dst, src[...]) to fill one. air.api handles "
                "whole-tile elementwise assignment (buf[:] = ...) only"
            )
        self._require_compute("write")
        from ._emit import emit_elementwise

        emit_elementwise(self, BufferExpr.coerce(value))

    @staticmethod
    def _is_whole(key):
        key = key if isinstance(key, tuple) else (key,)
        return all(
            isinstance(k, slice)
            and k.start is None
            and k.stop is None
            and k.step is None
            for k in key
        )

    def _require_compute(self, what):
        """Elementwise access needs a core, and a memtile does not have one."""
        if self.space != "L1":
            raise TypeError(
                f"cannot {what} an {self.space} buffer elementwise: {self.space} "
                "is a memtile, which has DMA engines but no compute core. Stage "
                "the tile into an L1 buffer with air.api.ops.load first, and "
                "compute on that."
            )

    def __repr__(self):
        return f"Buffer(shape={self.shape}, dtype={self.dtype}, space={self.space})"


class BufferSlice(_StridedView):
    """An access pattern into a :class:`Buffer`, for use as a DMA endpoint."""

    __slots__ = ("buffer", "offsets", "sizes", "strides", "logical_sizes", "is_view")

    def __init__(
        self, buffer, offsets, sizes, strides, logical_sizes=None, is_view=False
    ):
        self.buffer = buffer
        self.offsets = offsets
        self.sizes = sizes
        self.strides = strides
        # For a micro-tiled buffer the access pattern's rank and the region's
        # rank differ -- a [1, 1, 32, 32] logical region is walked as a
        # [1, 1, 8, 4, 8, 4] pattern. Transfers are shape-checked against the
        # logical view, which is the one the two endpoints have in common.
        self.logical_sizes = list(sizes if logical_sizes is None else logical_sizes)
        # Set by reshape/transpose. A view re-describes the same elements at a
        # different rank or in a different order, so its shape no longer has to
        # match the other endpoint's axis for axis -- only the element count
        # does. Transfers relax their shape check for it, and only for it.
        self.is_view = is_view

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    def value(self):
        return self.buffer.value

    # -- reading a region elementwise ---------------------------------------
    #
    # A partial subscript is primarily a DMA access pattern, and that is still
    # what ops.load/store see. But numpy spells "the gate half of the packed
    # buffer" as `gu[0]`, and kernels pack precisely because DMA channels are
    # scarce -- an AIE2P tile has two S2MM, so swiglu carries gate and up in one
    # [2, N] buffer rather than two. Refusing to read a row of it forced a copy
    # the packing existed to avoid.
    #
    # Only a *plain* region qualifies. A reshape/transpose view re-describes the
    # elements at another rank or order, so an index into it is not an index
    # into the buffer; a dynamic offset has no constant to fold into the loop
    # nest. Both are rejected by name below rather than silently mis-indexed. A
    # stepped subscript never reaches here -- __getitem__ refuses step != 1
    # outright -- so a region built by subscripting always carries the buffer's
    # own strides.

    @property
    def shape(self):
        return tuple(self.sizes)

    @property
    def vector_width(self):
        return self.buffer.vector_width

    @property
    def base(self):
        """The region's starting index per axis, as Python ints."""
        return [o.as_const() for o in self.offsets]

    def _as_leaf(self):
        """This region as an elementwise leaf, or a TypeError saying why not."""
        self.buffer._require_compute("read")
        if self.is_view:
            raise TypeError(
                f"cannot read {self!r} elementwise: it is a reshaped or "
                "transposed view, which re-describes the same elements at a "
                "different rank or order, so an index into it is not an index "
                "into the buffer. Only a plain region -- buf[1, :] -- reads "
                "elementwise."
            )
        if any(b is None for b in self.base):
            raise TypeError(
                f"cannot read {self!r} elementwise: its offset depends on a "
                "coordinate or loop variable, and the loop nest is built at "
                "trace time from constant extents. Subscript with a constant "
                "-- gu[1, :] -- or move the region with air.api.ops.load."
            )
        return BufferExpr.leaf(self)

    def _arith(self, other, op, reflected=False):
        expr = self._as_leaf()
        other = BufferExpr.coerce(other)
        return op(other, expr) if reflected else op(expr, other)

    def __add__(self, o):
        return self._arith(o, lambda a, b: a + b)

    def __radd__(self, o):
        return self._arith(o, lambda a, b: a + b, reflected=True)

    def __sub__(self, o):
        return self._arith(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._arith(o, lambda a, b: a - b, reflected=True)

    def __mul__(self, o):
        return self._arith(o, lambda a, b: a * b)

    def __rmul__(self, o):
        return self._arith(o, lambda a, b: a * b, reflected=True)

    def __truediv__(self, o):
        return self._arith(o, lambda a, b: a / b)

    def __rtruediv__(self, o):
        return self._arith(o, lambda a, b: a / b, reflected=True)

    def __neg__(self):
        return -self._as_leaf()

    def materialize_offsets(self):
        return [o.materialize() for o in self.offsets]

    def _respan(self, offsets, sizes, strides, logical_sizes):
        return BufferSlice(
            self.buffer, offsets, sizes, strides, logical_sizes, is_view=True
        )

    def __repr__(self):
        return f"BufferSlice({self.buffer!r}, sizes={self.sizes})"


# How a binary node prints. The tree stores an internal key ("add"), not the
# source spelling, so without this a repr shows `(Buffer add Buffer)` -- which
# reads like a typo. Keys with no infix form fall back to a call spelling.
# Entries are listed for every key the emitter knows, including the comparison
# and bitwise ones, so the table does not need revisiting when those land.
_OP_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "and": "&",
    "or": "|",
    "xor": "^",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "eq": "==",
    "ne": "!=",
}

# Keys with no infix spelling print as the call the user actually wrote, so the
# internal key never reaches a repr either way: ops.maximum, not "max".
_OP_CALL_NAMES = {"max": "maximum", "min": "minimum"}


def _check_shift_amount(amount, value, op):
    """Reject a constant shift count that MLIR would turn into poison.

    ``>>`` matches Python on the part people actually reason about -- it is
    arithmetic, so a negative value floors rather than filling with zeros --
    but it does *not* match Python on the shift count, and the difference is
    silent. Python's ints are arbitrary precision: ``x >> 100`` is 0 or -1, and
    ``x >> -1`` raises. MLIR inherits LLVM's rule instead, where a shift of the
    operand's own width or more, or by a negative amount, is **poison**, and
    poison is not an error -- it is a value the optimiser may assume never
    happens, which is how it turns into a wrong answer several passes later
    rather than a diagnostic.

    So a constant out-of-range count is refused here, at the call site, which
    is the closest this can get to Python's own ValueError. A count that is not
    a compile-time constant cannot be checked and is documented rather than
    guarded -- see the shift operators on ``BufferExpr``.

    "Constant" has to include an IndexExpr that folds to one. Index arithmetic
    over herd coordinates is ordinary in a kernel body, and ``tx - tx + 32``
    reaches the emitter as a literal ``arith.constant 32`` -- indistinguishable,
    by the time it gets there, from having been written as ``32``. Reading it
    back has to go through ``as_const()``: an isinstance test alone sees every
    index expression as "runtime" and lets the folded ones through, and
    comparing to an int instead builds a Condition for ``ops.branch`` rather than
    answering.
    """
    if amount.kind != "scalar":
        return  # not a scalar operand at all: nothing to check
    count = amount.scalar
    if hasattr(count, "as_const"):
        count = count.as_const()  # None when it is genuinely runtime
    if not isinstance(count, (int, bool)):
        return  # runtime amount: nothing to check, see the docstring
    count = int(count)
    dtype = value.element_dtype()
    spelling = "<<" if op == "shl" else ">>"
    if count < 0:
        raise ValueError(
            f"negative shift count {count} in '{spelling}': Python raises here "
            f"and MLIR would make it poison, which is silent. Shift by a "
            f"non-negative amount, or use the opposite operator"
        )
    if dtype is not None and count >= dtype.itemsize * 8:
        raise ValueError(
            f"shift count {count} is not less than the width of {dtype} "
            f"({dtype.itemsize * 8} bits) in '{spelling}': Python would give "
            f"{'0 or -1' if op == 'shr' else 'a wider integer'} but MLIR makes "
            f"it poison, which is silent. Shift by less than the width"
        )


class BufferExpr:
    """A lazy elementwise expression over buffers and scalars.

    Nothing is emitted while the tree is built; ``Buffer.__setitem__`` walks it
    once and emits a single vectorised loop.
    """

    __slots__ = ("kind", "op", "args", "buffer", "scalar", "dtype")

    def __init__(self, kind, op=None, args=(), buffer=None, scalar=None, dtype=None):
        # "buffer" | "scalar" | "unary" | "binary" | "fma" evaluate to the
        # element type; "compare" evaluates to i1 and only ops.select consumes
        # it; "select" takes (compare, value, value) back to the element type;
        # "cast" evaluates to a *different* element type from its operand,
        # which is the only node for which that is true.
        #
        # "fma" is the second ternary kind. Unlike "select", whose first
        # argument is a predicate, all three of its arguments are value-typed.
        #
        # "reduce" is the only node whose *shape* differs from its operand's:
        # it collapses the innermost dimension to 1. Everything else here is
        # elementwise, which is why the emitter can check leaf shapes against
        # the destination, and why a reduce has to be the whole right-hand
        # side rather than nesting inside a larger expression.
        self.kind = kind
        self.op = op
        self.args = tuple(args)
        self.buffer = buffer
        self.scalar = scalar
        # Set on a "cast" node only: the element type its operand is converted
        # *to*. Every other node adopts the type of the region it sits in, which
        # is what ``element_dtype`` below reports.
        self.dtype = dtype

    @staticmethod
    def leaf(buffer):
        return BufferExpr("buffer", buffer=buffer)

    def element_dtype(self):
        """The element type this subtree evaluates to, or ``None`` if unknown.

        ``None`` means "no buffer decided it" -- a subtree of nothing but
        scalars, as in the fill ``acc[:] = 0.0``. Such a subtree adopts whatever
        type surrounds it, so the caller supplies one rather than reading it
        here.

        A ``cast`` node is where the answer stops depending on its operand: it
        reports its own target type and does not recurse. That is what makes an
        expression able to hold more than one element type at once -- everything
        below a cast is evaluated in the source type, everything above it in the
        target type.
        """
        if self.kind == "buffer":
            return self.buffer.dtype
        if self.kind == "cast":
            return self.dtype
        if self.kind == "scalar":
            return None
        if self.kind == "select":
            # Skip the predicate: a select evaluates to the type of the two
            # values it chooses between, and asking the comparison would report
            # the type of *its* operands. Those agree today, because one region
            # covers the whole tree -- but only by construction, not by rule.
            return self.args[1].element_dtype() or self.args[2].element_dtype()
        for arg in self.args:
            found = arg.element_dtype()
            if found is not None:
                return found
        return None

    @staticmethod
    def coerce(value):
        if isinstance(value, BufferExpr):
            return value
        if isinstance(value, Buffer):
            return BufferExpr.leaf(value)
        if isinstance(value, (int, float)):
            return BufferExpr("scalar", scalar=value)
        from ._index import IndexExpr

        if isinstance(value, IndexExpr):
            # A herd coordinate or loop variable used as a broadcast scalar:
            # `out[:] = in[:] + ty`. It folds to a Python int when constant and
            # otherwise materialises as an index Value the emitter casts to the
            # buffer's element type -- which is what the hand-written channel
            # examples spell as arith.index_cast(T.i32(), ty).
            return BufferExpr("scalar", scalar=value)
        if isinstance(value, BufferSlice):
            # A plain region reads elementwise; a view, a strided region or a
            # dynamic offset does not, and _as_leaf says which.
            return value._as_leaf()
        raise TypeError(
            f"cannot use {value!r} ({type(value).__name__}) in an elementwise "
            "expression; expected a buffer slice or a numeric scalar"
        )

    def _binary(self, other, op, reverse=False):
        other = BufferExpr.coerce(other)
        args = (other, self) if reverse else (self, other)
        if op in ("shl", "shr"):
            _check_shift_amount(args[1], args[0], op)
        return BufferExpr("binary", op=op, args=args)

    def __add__(self, o):
        return self._binary(o, "add")

    def __radd__(self, o):
        return self._binary(o, "add", reverse=True)

    def __sub__(self, o):
        return self._binary(o, "sub")

    def __rsub__(self, o):
        return self._binary(o, "sub", reverse=True)

    def __mul__(self, o):
        return self._binary(o, "mul")

    def __rmul__(self, o):
        return self._binary(o, "mul", reverse=True)

    def __truediv__(self, o):
        return self._binary(o, "div")

    def __rtruediv__(self, o):
        return self._binary(o, "div", reverse=True)

    def __neg__(self):
        return BufferExpr("scalar", scalar=0) - self

    # Comparisons build a *predicate* node, not a value node. Its result type is
    # i1 (vector<Wxi1> when vectorised), not the element type every other node
    # yields, so the only thing that can consume one is air.api.ops.select --
    # which is why there is no operator spelling for select itself.
    def _compare(self, other, op, reverse=False):
        other = BufferExpr.coerce(other)
        args = (other, self) if reverse else (self, other)
        return BufferExpr("compare", op=op, args=args)

    def __lt__(self, o):
        return self._compare(o, "lt")

    def __le__(self, o):
        return self._compare(o, "le")

    def __gt__(self, o):
        return self._compare(o, "gt")

    def __ge__(self, o):
        return self._compare(o, "ge")

    # __eq__ and __ne__ are deliberately NOT overloaded. Defining __eq__ sets
    # __hash__ to None, making every expression unhashable, and it changes what
    # `expr == expr` means for ordinary Python code that has nothing to do with
    # kernels. Equality comparisons are spelled air.api.ops.equal / not_equal
    # instead; ops.select rejects the plain bool that `a[:] == b[:]` produces,
    # naming those, so the difference cannot pass silently.
    # Bitwise operators are integer-only: MLIR has arith.andi/ori/xori and no
    # floating-point counterpart, so a float buffer reaching one of these is a
    # user error rather than something to coerce. The emitter rejects it by
    # name rather than falling through a generic "unsupported operator".
    def __and__(self, o):
        return self._binary(o, "and")

    def __rand__(self, o):
        return self._binary(o, "and", reverse=True)

    def __or__(self, o):
        return self._binary(o, "or")

    def __ror__(self, o):
        return self._binary(o, "or", reverse=True)

    def __xor__(self, o):
        return self._binary(o, "xor")

    def __rxor__(self, o):
        return self._binary(o, "xor", reverse=True)

    # Shifts are integer-only for the same reason and rejected the same way.
    #
    # `>>` is arithmetic (arith.shrsi). There is no signedness choice to make
    # here, rather than a convention being picked: arith requires signless
    # operands, so an unsigned buffer is refused before it reaches any operator
    # at all, and every buffer that can reach a shift is signed. Arithmetic is
    # also what Python does -- `-8 >> 1` is -4 in both, not a huge positive.
    #
    # The shift *count* is where the resemblance to Python stops. Python's ints
    # are arbitrary precision, so `x >> 100` is 0 and `x << 100` just gets
    # wider; MLIR makes a count of the width or more poison, and `<<` wraps.
    # A constant count out of range is refused by _check_shift_amount above. A
    # count computed at runtime -- `1 << a[:]`, or a shift read from a buffer --
    # cannot be checked, and is the one place where an out-of-range value still
    # reaches the backend as poison.
    def __lshift__(self, o):
        return self._binary(o, "shl")

    def __rlshift__(self, o):
        return self._binary(o, "shl", reverse=True)

    def __rshift__(self, o):
        return self._binary(o, "shr")

    def __rrshift__(self, o):
        return self._binary(o, "shr", reverse=True)

    def __bool__(self):
        # NumPy's guard, for NumPy's reason. `and`, `or` and `not` are the one
        # part of Python's operator surface a library cannot reach: there is no
        # dunder for them, they coerce the operand through __bool__, and they
        # short-circuit. Without this, `a[:] and b[:]` takes the default
        # object truthiness (always True), returns b[:], and emits nothing --
        # a kernel that silently computes half of what was written.
        #
        # The redirection is built from what this build actually has rather
        # than hardcoded. `&` and ops.select are landing in separate changes,
        # and an error that names surface the caller does not have is worse
        # than the silent bug it replaces -- while hedging every suggestion
        # with "if available" would make the useful case vague. Asking is
        # exact in both.
        suggestions = []
        if hasattr(type(self), "__and__"):
            suggestions.append("the elementwise operators `&`, `|`, `^` for logic")
        from . import ops

        if hasattr(ops, "select"):
            suggestions.append(
                "air.api.ops.select(cond, a, b) to choose between values"
            )
        advice = f" Use {' or '.join(suggestions)}." if suggestions else ""
        raise TypeError(
            "cannot use a buffer expression as a truth value: `and`, `or` and "
            "`not` would silently return one operand and emit no kernel code, "
            "because Python does not allow them to be overloaded." + advice
        )

    def leaves(self):
        """All buffer leaves, in traversal order."""
        if self.kind == "buffer":
            return [self.buffer]
        out = []
        for a in self.args:
            out.extend(a.leaves())
        return out

    def __repr__(self):
        if self.kind == "buffer":
            return repr(self.buffer)
        if self.kind == "scalar":
            return repr(self.scalar)
        if self.kind == "cast":
            return f"cast({self.args[0]!r}, {self.dtype!r})"
        if self.kind == "unary":
            return f"{self.op}({self.args[0]!r})"
        if self.kind == "select":
            c, a, b = self.args
            return f"select({c!r}, {a!r}, {b!r})"
        # The other ternary kind, and like select it needs its own branch: the
        # tail below is binary and indexes exactly two args.
        if self.kind == "fma":
            a, b, c = self.args
            return f"fma({a!r}, {b!r}, {c!r})"
        if self.kind == "reduce":
            return f"reduce_{self.op}({self.args[0]!r})"
        symbol = _OP_SYMBOLS.get(self.op)
        if symbol is None:
            # No infix spelling (maximum/minimum and anything added later):
            # show it as the call it is rather than inventing an operator.
            name = _OP_CALL_NAMES.get(self.op, self.op)
            return f"{name}({self.args[0]!r}, {self.args[1]!r})"
        return f"({self.args[0]!r} {symbol} {self.args[1]!r})"
