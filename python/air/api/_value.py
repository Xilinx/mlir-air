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


class Tensor:
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


class TensorSlice:
    """An access pattern into a :class:`Tensor`: offsets, static sizes, strides."""

    __slots__ = ("tensor", "offsets", "sizes", "strides")

    def __init__(self, tensor, offsets, sizes, strides):
        self.tensor = tensor
        self.offsets = offsets
        self.sizes = sizes
        self.strides = strides

    @property
    def dtype(self):
        return self.tensor.dtype

    def materialize_offsets(self):
        return [o.materialize() for o in self.offsets]

    def __repr__(self):
        return f"TensorSlice({self.tensor.name}, sizes={self.sizes})"


class Buffer:
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
        pack=None,
    ):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.scope = scope
        # A PackedShape when this tile is laid out in micro-tile order for the
        # AIE2 matmul intrinsic, else None. It does not change the memref -- the
        # buffer is contiguous either way -- but it tells ops.load/store how to
        # walk the flat side. See _pack.py.
        self.pack = pack
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
        if self.pack is not None:
            return self._packed_slice(key)
        key = _normalize_key(key, len(self.shape), "buffer")
        offsets, sizes = [], []
        for dim, (sub, extent) in enumerate(zip(key, self.shape)):
            offset, size = _resolve_subscript(sub, extent, dim)
            offsets.append(offset)
            sizes.append(size)
        return BufferSlice(self, offsets, sizes, list(self.strides))

    def _packed_slice(self, key):
        """Subscript a micro-tiled buffer in *logical* coordinates.

        The whole point of a packed layout is that the program keeps thinking in
        ``[M, N]`` while the memref is ``[N/n, M/m, m, n]``, so the subscript is
        the logical one: ``l1_c[tx, ty, :, :]`` on a herd-shared accumulator
        names one core's ``tile_m x tile_n`` slab. The rank-6 access pattern is
        derived from it -- see ``_pack.pack_pattern``.
        """
        from ._pack import pack_pattern

        logical = self.pack.lead + self.pack.logical
        key = _normalize_key(key, len(logical), "packed buffer")
        offsets, sizes = [], []
        for dim, (sub, extent) in enumerate(zip(key, logical)):
            offset, size = _resolve_subscript(sub, extent, dim)
            offsets.append(offset)
            sizes.append(size)
        pat_offsets, pat_sizes, pat_strides = pack_pattern(
            self.pack, sizes, self.strides, offsets
        )
        return BufferSlice(
            self, pat_offsets, pat_sizes, pat_strides, logical_sizes=sizes
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
        if self.pack is not None:
            return self._packed_fill(value)
        from ._emit import emit_elementwise

        emit_elementwise(self, BufferExpr.coerce(value))

    def _packed_fill(self, value):
        """``acc[:] = 0.0`` on a micro-tiled buffer, as one ``linalg.fill``.

        The generic emitter would build a loop nest over the *packed* shape --
        six nested loops and a scalar store per element, tens of thousands of
        them for a real tile, which is the documented cause of NPU timeouts.
        Zeroing an accumulator has no elementwise structure worth preserving, so
        it goes out as the single op the reference uses.
        """
        from air.dialects.arith import ConstantOp
        from air.dialects.linalg import fill

        from . import ops as _ops
        from .types import require_signless

        # linalg.fill's value comes from an arith.constant, which has no signful
        # form; a packed buffer is an accumulator anyway, and ops.dot has
        # already refused an unsigned one.
        require_signless(self.dtype, "a fill of a micro-tiled buffer")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise NotImplementedError(
                "only a scalar fill is supported on a micro-tiled buffer "
                f"(got {value!r}): its elements are not in row-major order, so "
                "an elementwise expression over it would not mean what it reads "
                "like. Compute into it with air.api.ops.dot instead."
            )
        scalar = float(value) if self.dtype.is_float else int(value)
        cst = ConstantOp(self.dtype.mlir(), scalar)
        return fill(cst, outs=[_ops.accumulator_subview(self)])

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


class BufferSlice:
    """An access pattern into a :class:`Buffer`, for use as a DMA endpoint."""

    __slots__ = ("buffer", "offsets", "sizes", "strides", "logical_sizes")

    def __init__(self, buffer, offsets, sizes, strides, logical_sizes=None):
        self.buffer = buffer
        self.offsets = offsets
        self.sizes = sizes
        self.strides = strides
        # For a micro-tiled buffer the access pattern's rank and the region's
        # rank differ -- a [1, 1, 32, 32] logical region is walked as a
        # [1, 1, 8, 4, 8, 4] pattern. Transfers are shape-checked against the
        # logical view, which is the one the two endpoints have in common.
        self.logical_sizes = list(sizes if logical_sizes is None else logical_sizes)

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    def value(self):
        return self.buffer.value

    def materialize_offsets(self):
        return [o.materialize() for o in self.offsets]

    def __repr__(self):
        return f"BufferSlice({self.buffer!r}, sizes={self.sizes})"


class BufferExpr:
    """A lazy elementwise expression over buffers and scalars.

    Nothing is emitted while the tree is built; ``Buffer.__setitem__`` walks it
    once and emits a single vectorised loop.
    """

    __slots__ = ("kind", "op", "args", "buffer", "scalar")

    def __init__(self, kind, op=None, args=(), buffer=None, scalar=None):
        # "buffer" | "scalar" | "unary" | "binary" evaluate to the element
        # type; "compare" evaluates to i1 and only ops.select consumes it;
        # "select" takes (compare, value, value) back to the element type.
        self.kind = kind
        self.op = op
        self.args = tuple(args)
        self.buffer = buffer
        self.scalar = scalar

    @staticmethod
    def leaf(buffer):
        return BufferExpr("buffer", buffer=buffer)

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
            raise TypeError(
                f"cannot use {value!r} in an elementwise expression: a partial "
                "subscript names a DMA region, not a value. Move the region into "
                "an L1 buffer with air.api.ops.load(dst, src[...]) and compute on "
                "that buffer"
            )
        raise TypeError(
            f"cannot use {value!r} ({type(value).__name__}) in an elementwise "
            "expression; expected a buffer slice or a numeric scalar"
        )

    def _binary(self, other, op, reverse=False):
        other = BufferExpr.coerce(other)
        args = (other, self) if reverse else (self, other)
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
        if self.kind == "unary":
            return f"{self.op}({self.args[0]!r})"
        if self.kind == "select":
            c, a, b = self.args
            return f"select({c!r}, {a!r}, {b!r})"
        return f"({self.args[0]!r} {self.op} {self.args[1]!r})"
