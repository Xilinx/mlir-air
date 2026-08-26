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

        n_scalars = sum(1 for a in args if not isinstance(a, Buffer))
        if n_scalars != len(self.scalars):
            raise TypeError(
                f"{self.name} was declared with {len(self.scalars)} scalar "
                f"argument type(s) but called with {n_scalars}; air.extern needs "
                "one element type per non-buffer argument, in order"
            )

        values, types, scalars = [], [], iter(self.scalars)
        for pos, arg in enumerate(args):
            if isinstance(arg, Buffer):
                if arg.value is None:
                    raise RuntimeError(
                        f"{self.name}: buffer argument {pos} used before allocation"
                    )
                values.append(arg.value)
                types.append(arg.value.type)
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


def _scalar_value(arg, dtype, name, pos, arith):
    """Materialise a non-buffer argument as ``dtype``."""
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
