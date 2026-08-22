# ./python/air/api/types.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Element types for the AIR Python API.

Each ``DType`` is a thin, context-free description of an element type. The
corresponding MLIR type is produced on demand by :meth:`DType.mlir`, which must
be called inside an active ``air.ir.Context`` -- it defers to the existing
``type_mapper`` in ``air/backend/xrt_runner.py`` so that the API and the XRT
runner can never disagree about how a numpy dtype maps onto the device.
"""

import numpy as np
from ml_dtypes import bfloat16

__all__ = [
    "DType",
    "bf16",
    "f16",
    "f32",
    "i8",
    "i16",
    "i32",
    "ui8",
    "ui16",
    "ui32",
    "dtype_of",
]
# `require_signless` is deliberately absent from __all__: it is the guard the
# emission paths call, not something a kernel author reaches for. All four
# callers import it by name, which __all__ does not govern.


class DType:
    """An element type, plus the vector width the emitter should default to.

    ``default_vector_width`` is the number of lanes used by the elementwise
    emitter when the user does not override it. A tile whose innermost dimension
    is not a multiple of the width falls back to a scalar loop.

    The widths are measured, not assumed. bf16 at 16 lanes (256-bit) compiles on
    both npu1 and npu2. f32 at 8 lanes (also 256-bit) does *not* -- it fails in
    the AIE backend with "unable to legalize instruction: <8 x s32> G_FADD" on
    both targets -- so f32 defaults to 16 lanes (512-bit), which does compile.
    The hand-written eltwise_add example documents 8 for f32 in its --help, but
    its npu1 config actually runs scalar (VECTOR_SIZE=0), so that advice is
    never exercised there.

    i32 is 32-bit like f32 and behaves the same way: 8 lanes (256-bit) fails in
    the AIE backend with "unable to legalize instruction: <8 x s32> G_ADD", so
    it also defaults to 16. Measured on npu1 by segment_unroll, which was the
    first example to vectorise an i32 elementwise body -- the previous value of
    8 was an extrapolation by element size and was wrong.

    Widths for f16, i8 and i16 are still extrapolated that way and have not been
    compiled; treat them as unverified. The unsigned widths are never reached at
    all -- see ``is_unsigned``.
    """

    def __init__(
        self, name, np_dtype, default_vector_width, is_float, is_unsigned=False
    ):
        self.name = name
        self.np_dtype = np_dtype
        self.default_vector_width = default_vector_width
        # Stated, not inferred: np.issubdtype(bfloat16, np.floating) is False
        # for the ml_dtypes extension types, which would silently select the
        # integer arithmetic ops for a bf16 kernel.
        self.is_float = is_float
        # MLIR's arith and linalg ops constrain their integer operands to be
        # *signless* (`SignlessIntegerLike`), while `type_mapper` maps numpy's
        # unsigned dtypes onto MLIR's signful `ui8`/`ui16`/`ui32`. So an
        # unsigned buffer can be declared, allocated, moved and handed to an
        # external kernel, but arithmetic on one does not verify -- see
        # `require_signless`, which every emission path that builds an arith or
        # linalg op calls.
        self.is_unsigned = is_unsigned

    @property
    def itemsize(self):
        return np.dtype(self.np_dtype).itemsize

    def mlir(self):
        """The MLIR element type. Requires an active Context."""
        # Imported lazily: air.backend pulls in the MLIR bindings, and importing
        # air.api must stay cheap for callers that only want the type objects.
        from air.backend.xrt_runner import type_mapper

        return type_mapper(self.np_dtype)

    def __repr__(self):
        return f"air.api.{self.name}"


bf16 = DType("bf16", bfloat16, 16, is_float=True)
f16 = DType("f16", np.float16, 16, is_float=True)
f32 = DType("f32", np.float32, 16, is_float=True)
i8 = DType("i8", np.int8, 32, is_float=False)
i16 = DType("i16", np.int16, 16, is_float=False)
i32 = DType("i32", np.int32, 16, is_float=False)

# Unsigned integers exist so that a kernel over `np.uint8` data can say so. The
# alternative -- declaring i8 and passing uint8 arrays -- type-checks nowhere
# and makes the emitted `func.func private @...` declaration disagree with the
# C prototype the object file was compiled from. The vector widths mirror their
# signed counterparts and are unreachable while arithmetic is refused; they are
# stated rather than left at 0 so that relaxing the refusal does not silently
# also change the width.
ui8 = DType("ui8", np.uint8, 32, is_float=False, is_unsigned=True)
ui16 = DType("ui16", np.uint16, 16, is_float=False, is_unsigned=True)
ui32 = DType("ui32", np.uint32, 8, is_float=False, is_unsigned=True)

_BY_NP = {d.np_dtype: d for d in (bf16, f16, f32, i8, i16, i32, ui8, ui16, ui32)}


def dtype_of(np_dtype):
    """Map a numpy dtype back onto the API's DType, for error messages.

    Accepts either the scalar type (``np.float32``) or a dtype instance
    (``np.dtype("float32")``); the table is keyed by the former.
    """
    return _BY_NP.get(np_dtype) or _BY_NP.get(getattr(np_dtype, "type", None))


def require_signless(dtype, what):
    """Refuse ``what`` on an unsigned element type, naming why it cannot work.

    MLIR spells signedness in the *operation*, not the type: ``arith.divsi`` and
    ``arith.divui`` both take signless operands. So the whole ``arith`` dialect
    -- and every named ``linalg`` contraction, whose region is built from it --
    is constrained to signless integers, and a ``ui8`` operand fails
    verification rather than being reinterpreted.

    This is not a limitation air.api could paper over by bitcasting to the
    signed sibling: that would silently change what ``max``, ``div`` and a
    widening contraction compute. Refuse, and name the signed type to declare
    instead.
    """
    if not dtype.is_unsigned:
        return
    raise NotImplementedError(
        f"{what} is not supported for {dtype}: MLIR's arith and linalg ops "
        f"require signless integer operands, and an unsigned numpy dtype maps "
        f"onto the signful MLIR type '{dtype.name}'. An unsigned buffer can be "
        f"allocated, moved with air.api.ops.load/store and air.channel, and "
        f"passed to an air.extern kernel -- which is what the uint8 examples in "
        f"this tree do. To compute on it in the DSL, declare it "
        f"air.api.{dtype.name[1:]} instead and interpret the data yourself."
    )
