# ./python/air/api/_extern.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Calling a hand-written AIE kernel from a traced program.

    mv = air.extern("matvec_vectorized_bf16_bf16", link_with="mv.o",
                    scalars=[i32, i32, i32])
    mv(1, K, row, a_buf, b_buf, c_buf)

Every vectorised matmul-family example in this tree computes through a ``.cc``
kernel written against the AIE intrinsics (``aie::load_v``, ``aie::mac``,
``aie::accum``), because ``ops.dot`` on its own lowers through
``convert-linalg-to-loops`` and comes out scalar. This is the escape hatch to
that kernel, and it is deliberately general: what it emits is a ``func.call``,
not a contraction, so it serves the fill and dequantisation kernels those
examples also call.

The shape is taken from what the hand-written examples actually emit::

    func.func private @matvec_vectorized_bf16_bf16(i32, i32, i32,
        memref<1x8192xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>,
        memref<4xbf16, 2 : i32>)
        attributes {link_with = "mv.o", llvm.emit_c_interface}
    ...
    air.herd @herd_0 ... attributes {link_with = "mv.o"} {
      func.call @matvec_vectorized_bf16_bf16(%c1_i32, %c8192_i32, %2, ...)

Reading that IR settles the split between what is declared and what is
inferred: the declaration is *exactly* the types of the call's arguments, and
carries nothing the call site does not already have. Buffer types are therefore
derived from the :class:`Buffer` objects -- which is where all the verbosity
lives -- while scalar element types must be declared, because a Python ``0.0``
cannot distinguish the ``bf16`` a fill kernel wants from the ``f32`` it does
not, and ``1`` cannot pick between ``i32`` and ``index``. Guessing either would
be the silent-wrongness failure this package exists to avoid.
"""

from ._index import IndexExpr
from ._value import Buffer

__all__ = ["ExternKernel", "extern"]


class ExternKernel:
    """A private ``func.func`` in an object file, callable from a herd body."""

    def __init__(self, name, link_with=None, scalars=()):
        if not isinstance(name, str) or not name:
            raise TypeError("air.extern(name) takes the kernel's C symbol name")
        if not link_with:
            raise ValueError(
                "air.extern requires link_with=<file>, the compiled kernel aircc "
                'should link against, e.g. link_with="mv.o". It is the '
                "link_with attribute, stamped on both the declaration and the "
                "herd."
            )
        from .types import DType, require_computable, require_signless

        for s in scalars:
            if not isinstance(s, DType):
                raise TypeError(
                    f"air.extern(scalars=...) takes element types such as i32 or "
                    f"bf16, got {type(s).__name__}. Scalar types have to be "
                    "stated: a Python 0.0 does not say whether the kernel wants "
                    "bf16 or f32, and buffer types are inferred from the buffers "
                    "themselves."
                )
            # A scalar argument is materialised by arith.constant, so it has to
            # be signless even though the *buffer* arguments beside it need not
            # be. Caught at the declaration rather than at the first call, which
            # is where the constant would actually be built.
            require_signless(s, "an air.extern scalar argument")
            require_computable(s, "an air.extern scalar argument")
        self.name = name
        self.link_with = link_with
        self.scalars = list(scalars)
        # The declaration, materialised at the first call from the call's own
        # argument types, and reused after that.
        self._decl = None
        self._arg_types = None

    def __repr__(self):
        return f"air.api.extern({self.name!r}, link_with={self.link_with!r})"

    def __call__(self, *args):
        from air.ir import InsertionPoint, StringAttr, UnitAttr
        from air.dialects import arith
        from air.dialects.func import CallOp, FuncOp

        from ._trace import active_trace, current_herd

        herd = current_herd()
        trace = active_trace()

        n_scalars = sum(1 for a in args if not _is_memref_arg(a))
        if n_scalars != len(self.scalars):
            raise TypeError(
                f"{self.name} was declared with {len(self.scalars)} scalar "
                f"argument type(s) but called with {n_scalars}; air.extern needs "
                "one element type per non-buffer argument, in order"
            )

        values, types, scalars = [], [], iter(self.scalars)
        for pos, arg in enumerate(args):
            if _is_memref_arg(arg):
                if arg.value is None:
                    raise RuntimeError(
                        f"{self.name}: buffer argument {pos} used before allocation"
                    )
                value = _core_slab(arg) if isinstance(arg, Buffer) else _region(arg)
                values.append(value)
                types.append(value.type)
            else:
                dtype = next(scalars)
                values.append(_scalar_value(arg, dtype, self.name, pos, arith))
                types.append(dtype.mlir())

        if self._decl is None:
            # Declarations go at the top of the module, as the hand-written
            # examples emit them, so the IR reads the same way.
            with InsertionPoint.at_block_begin(trace.module.body):
                decl = FuncOp(self.name, (types, []), visibility="private")
                decl.attributes["link_with"] = StringAttr.get(self.link_with)
                decl.attributes["llvm.emit_c_interface"] = UnitAttr.get()
            self._decl, self._arg_types = decl, types
        elif [str(t) for t in types] != [str(t) for t in self._arg_types]:
            raise TypeError(
                f"{self.name} is already declared as "
                f"({', '.join(str(t) for t in self._arg_types)}) but is now "
                f"called with ({', '.join(str(t) for t in types)}). A C symbol "
                "has one signature; use a second air.extern for a second "
                "kernel, or make the buffer shapes agree."
            )

        # aircc links one object per herd, so the attribute is a single string.
        herd.require_object(self.link_with, self.name)
        return CallOp(self._decl, values)


def _is_memref_arg(arg):
    """Does ``arg`` become a memref operand rather than a scalar one?

    Rank decides, which is the same rule the rest of the DSL indexes by: an
    integer subscript drops an axis and a slice keeps it, so ``ctr[0]`` is a
    scalar and ``ctr[0:1]`` is a one-element region. A kernel wants the first as
    an ``i32`` operand and the second as a ``memref<1xi32>``, and until now both
    arrived as a memref.
    """
    from ._value import BufferSlice

    if isinstance(arg, Buffer):
        return True
    if isinstance(arg, BufferSlice):
        return arg.shape != ()
    # Anything else -- an expression such as ctr[0] + tx, an int, an IndexExpr
    # -- is a scalar. An expression has no memory of its own to pass.
    return False


def _region(region):
    """A sub-region of a buffer, as the ``memref.subview`` a kernel expects.

    ``buf[row, 0:n]`` is a DMA access pattern everywhere else in the DSL --
    ``ops.load`` and ``ops.store`` read its offsets/sizes/strides and build a
    transfer. A kernel cannot take an access pattern; it takes a memref. So
    here, and only here, the same subscript becomes a real ``memref.subview``.

    This is the third way a kernel is handed less than a whole buffer, and the
    other two are already narrower: ``ops.dot``'s shared accumulator and
    ``_core_slab`` below both emit a subview the caller does not get to choose,
    because there is exactly one slab a core may touch. Here the caller does
    choose, which is the point -- a cascade payload is a fixed-width slice of a
    wider L1 buffer, and which row it is comes from the loop.

    A *reshape* of the whole buffer is the one view that does correspond to a
    memref, and goes to :func:`_collapsed` below. A transpose, or a reshape of
    something narrower than the whole buffer, is refused: those re-describe the
    order a transfer walks memory and are not a sub-memref of anything.
    """
    from air.dialects.memref import subview

    if region.is_view:
        return _collapsed(region)
    offsets = [
        o.materialize() if hasattr(o, "materialize") else o for o in region.offsets
    ]
    return subview(
        region.value,
        offsets=offsets,
        sizes=list(region.sizes),
        strides=[1] * len(region.sizes),
    )


def _collapsed(region, _seen=None):
    """A whole buffer at a lower rank, as ``memref.collapse_shape``.

    The flash-attention kernels take their accumulator flat -- a ``[chunks, n]``
    L1 tile handed to ``zero_fill_g_bf16`` as ``[chunks * n]`` -- because the
    kernel walks it as one run and the rank only matters to the matmul that
    fills it. ``buf.reshape(n)`` is how the DSL already says "the same elements
    at a different rank", and for the whole of a contiguous buffer that is
    exactly what ``memref.collapse_shape`` names.

    Only a *grouping* of the existing axes is a collapse: each target extent has
    to be the product of a run of consecutive source axes. Splitting an axis --
    ``[4096]`` back to ``[64, 64]`` -- is ``memref.expand_shape``, a different
    op with different alignment rules, and it is refused rather than guessed at
    because nothing in this tree has wanted it.
    """
    from air.dialects.memref import collapse_shape

    buffer = region.buffer
    source = list(buffer.shape)
    target = list(region.sizes)

    def refuse(why):
        return TypeError(
            f"cannot pass this reshaped region to a kernel: {why}. A kernel "
            f"takes a memref, and the only reshape that is one is a collapse of "
            f"a whole contiguous buffer onto a grouping of its own axes -- "
            f"{source} to {target} is not that. Subscript the buffer and pass a "
            f"sub-region, or pass it whole"
        )

    if any(_as_int(o) != 0 for o in region.offsets):
        raise refuse("it starts at an offset, so it is not the whole buffer")
    if _product(source) != _product(target):
        raise refuse("it does not cover the whole buffer")
    if len(target) > len(source):
        raise refuse("it splits an axis, which is memref.expand_shape")
    # A permutation has the same extents as a collapse of the same rank, so the
    # grouping check below cannot tell the two apart -- it would accept
    # `g.transpose(1, 0)` and emit a collapse that hands the kernel the
    # *untransposed* buffer. The strides are what distinguish them: a view that
    # is still walking memory in order has row-major strides for its own sizes.
    if list(region.strides) != _row_major(target):
        raise refuse(
            "it walks the buffer in a different order (a transpose), and "
            "memref has no view that reorders elements -- only a copy does"
        )
    if target == source:
        # Nothing to collapse. Emitting the op anyway would be a no-op view of
        # the buffer, which verifies but reads as though something happened.
        return buffer.value

    # Group consecutive source axes until each product matches a target extent.
    groups, axis = [], 0
    for extent in target:
        run, acc = [], 1
        while acc < extent and axis < len(source):
            acc *= source[axis]
            run.append(axis)
            axis += 1
        if acc != extent or not run:
            raise refuse(
                f"extent {extent} is not the product of a run of consecutive "
                f"source axes"
            )
        groups.append(run)
    if axis != len(source):
        raise refuse("it leaves trailing axes unaccounted for")

    return collapse_shape(_collapsed_type(buffer, target), buffer.value, groups)


def _collapsed_type(buffer, shape):
    """The result memref type of collapsing ``buffer`` to ``shape``."""
    from air.ir import MemRefType

    memref = buffer.value.type
    return MemRefType.get(shape, memref.element_type, memory_space=memref.memory_space)


def _row_major(shape):
    """Contiguous strides for ``shape``, innermost first."""
    strides, acc = [1] * len(shape), 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return strides


def _product(extents):
    n = 1
    for e in extents:
        n *= e
    return n


def _as_int(offset):
    """An offset as a Python int, or None when it is only known at run time."""
    if isinstance(offset, int):
        return offset
    value = offset.as_const() if hasattr(offset, "as_const") else None
    return value


def _core_slab(buffer):
    """The value to pass for ``buffer``: its whole memref, or this core's slab.

    A ``<segment>.shared()`` buffer is one allocation spanning every core, with
    a leading dimension per herd axis, and a core may only touch its own slab.
    That is not a choice the caller gets to make -- there is exactly one slab a
    given core is allowed to write -- so the DSL narrows the argument itself
    rather than asking for coordinates it could only re-check. ``ops.fill`` and
    ``ops.dot`` already do this through the same helper, and a kernel reached by
    ``air.extern`` is the third way to write an accumulator: the two bfp16
    matmuls zero it, accumulate into it and narrow it, all through hand-written
    kernels, and each call passed a ``memref.subview`` written out by hand.

    Any other buffer is passed whole, which is every existing caller.
    """
    if getattr(buffer.scope, "kind", None) != "shared":
        return buffer.value
    from .ops import accumulator_subview

    return accumulator_subview(buffer)


def _scalar_value(arg, dtype, name, pos, arith):
    """Materialise a non-buffer argument as ``dtype``."""
    from ._value import BufferExpr, BufferSlice

    if isinstance(arg, (BufferExpr, BufferSlice)):
        # A value read out of a buffer. The counter tile the flash-attention
        # cores keep in L1 is the case: the causal mask takes its q-block index,
        # which is an element of that tile plus the tile coordinate.
        from ._emit import emit_scalar_value

        expr = BufferExpr.coerce(arg)
        source = expr.element_dtype()
        if source is not dtype:
            raise TypeError(
                f"{name}: argument {pos} reads {source} out of a buffer but was "
                f"declared {dtype}. A kernel scalar is passed as it is stored -- "
                f"convert it with air.api.ops.cast first if that is what you "
                f"meant, so the conversion is visible where it happens"
            )
        from .types import require_signless

        require_signless(dtype, "an air.extern scalar argument")
        return emit_scalar_value(expr, dtype)

    if isinstance(arg, IndexExpr):
        # A loop induction variable or tile coordinate. It folds to a Python int
        # when the expression is constant, and otherwise materialises as an
        # index Value that the kernel's i32 parameter needs a cast from -- which
        # is exactly what the hand-written matvec emits for its row offset.
        value = arg.materialize()
        if not isinstance(value, int):
            if dtype.is_float:
                raise TypeError(
                    f"{name}: argument {pos} is a tile coordinate or loop "
                    f"variable, which is an integer, but was declared {dtype}"
                )
            return arith.index_cast(dtype.mlir(), value)
        arg = value

    if dtype.is_float:
        if not isinstance(arg, (int, float)):
            raise TypeError(
                f"{name}: argument {pos} was declared {dtype} but got "
                f"{type(arg).__name__}"
            )
        return arith.ConstantOp(dtype.mlir(), float(arg)).result

    if isinstance(arg, float) and not float(arg).is_integer():
        raise ValueError(
            f"{name}: argument {pos} is {arg}, which is not an integer, but was "
            f"declared {dtype}"
        )
    if not hasattr(arg, "__index__") and not isinstance(arg, float):
        raise TypeError(
            f"{name}: argument {pos} was declared {dtype} but got "
            f"{type(arg).__name__}"
        )
    return arith.ConstantOp(dtype.mlir(), int(arg)).result


def extern(name, link_with=None, scalars=()):
    """Declare a hand-written kernel from ``link_with``; call it in a herd."""
    return ExternKernel(name, link_with=link_with, scalars=scalars)
