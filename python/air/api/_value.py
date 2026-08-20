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

__all__ = ["Token", "Tensor", "TensorSlice", "Buffer", "BufferExpr"]


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
    """A tile allocated in a hardware scope (L1 today)."""

    def __init__(self, shape, dtype, scope=None, vector_width=None, value=None):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.scope = scope
        self.vector_width = (
            dtype.default_vector_width if vector_width is None else int(vector_width)
        )
        # The memref SSA value produced by memref.alloc.
        self.value = value

    # -- reading: builds a lazy expression, emits nothing -------------------

    def __getitem__(self, key):
        self._require_whole(key, "read")
        return BufferExpr.leaf(self)

    # -- writing: this is what triggers emission ----------------------------

    def __setitem__(self, key, value):
        self._require_whole(key, "write")
        from ._emit import emit_elementwise

        emit_elementwise(self, BufferExpr.coerce(value))

    def _require_whole(self, key, what):
        """v1 only supports whole-tile elementwise access (``buf[:]``)."""
        key = key if isinstance(key, tuple) else (key,)
        whole = all(
            isinstance(k, slice)
            and k.start is None
            and k.stop is None
            and k.step is None
            for k in key
        )
        if not whole:
            raise NotImplementedError(
                f"partial {what} of an L1 buffer is not supported yet; "
                "air.api v1 handles whole-tile elementwise access (buf[:]) only"
            )

    def __repr__(self):
        return f"Buffer(shape={self.shape}, dtype={self.dtype})"


class BufferExpr:
    """A lazy elementwise expression over buffers and scalars.

    Nothing is emitted while the tree is built; ``Buffer.__setitem__`` walks it
    once and emits a single vectorised loop.
    """

    __slots__ = ("kind", "op", "args", "buffer", "scalar")

    def __init__(self, kind, op=None, args=(), buffer=None, scalar=None):
        self.kind = kind  # "buffer" | "scalar" | "unary" | "binary"
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
        return f"({self.args[0]!r} {self.op} {self.args[1]!r})"
