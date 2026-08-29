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
    "i4",
    "i8",
    "i16",
    "i32",
    "ui8",
    "ui16",
    "ui32",
    "dtype_of",
]
# `require_signless` and `require_computable` are deliberately absent from
# __all__: they are the guards the emission paths call, not something a kernel
# author reaches for. All four callers import them by name, which __all__ does
# not govern.


class DType:
    """An element type, plus the vector width the emitter should default to.

    ``default_vector_width`` is the number of lanes used by the elementwise
    emitter when the user does not override it. A tile whose innermost dimension
    is not a multiple of the width falls back to a scalar loop.

    The widths are measured, not assumed. bf16 at 16 lanes (256-bit) compiles on
    both npu1 and npu2; npu2 also legalizes 32 lanes (512-bit), which is what
    ``wide_vector_width`` and ``width_for`` select there. f32 at 8 lanes (also
    256-bit) does *not* -- it fails in the AIE backend with "unable to legalize
    instruction: <8 x s32> G_FADD" on both targets -- so f32 defaults to 16
    lanes (512-bit), which does compile.
    The hand-written eltwise_add example documents 8 for f32 in its --help, but
    its npu1 config actually runs scalar (VECTOR_SIZE=0), so that advice is
    never exercised there.

    i32 is 32-bit like f32 and behaves the same way: 8 lanes (256-bit) fails in
    the AIE backend with "unable to legalize instruction: <8 x s32> G_ADD", so
    it also defaults to 16. Measured on npu1 by segment_unroll, which was the
    first example to vectorise an i32 elementwise body -- the previous value of
    8 was an extrapolation by element size and was wrong.

    Widths for i8 and i16 are still extrapolated that way and have not been
    compiled; treat them as unverified.

    The *unsigned* widths are genuinely dead: an unsigned buffer is forced onto
    the scalar path even for a plain copy, because ``vector.transfer_read``
    takes a padding value and that padding value is an ``arith.constant``. See
    ``is_unsigned``.

    f16 is not like that, and it would be wrong to lump the two together.
    Arithmetic on f16 is refused (see ``computes``) but a plain copy is still
    allowed and still vectorises, so f16's width *is* used -- an ``f16`` tile
    copy emits ``vector<16xf16>``. That 16 is measured rather than
    extrapolated: an f16 copy at 16 lanes is exact on 2048 of 2048 elements on
    npu1, which is unsurprising, since a copy moves bit patterns and never asks
    the core to interpret them as floats.
    """

    def __init__(
        self,
        name,
        np_dtype,
        default_vector_width,
        is_float,
        wide_vector_width=None,
        is_unsigned=False,
        computes=True,
        bits=None,
        allocatable=True,
    ):
        self.name = name
        self.np_dtype = np_dtype
        self.default_vector_width = default_vector_width
        # Lanes on a target whose backend legalizes a wider vector for this
        # type. None means the one width is all there is.
        self.wide_vector_width = wide_vector_width
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
        # False when the AIE core has no instruction for this type, so a buffer
        # of it can be declared, allocated and moved but not computed on. Same
        # shape of restriction as `is_unsigned` above, enforced at the same four
        # call sites -- see `require_computable`.
        self.computes = computes
        # Width in bits. Everything but i4 is a whole number of bytes and takes
        # it from numpy; i4 cannot, because numpy has no sub-byte storage and
        # reports an itemsize of 1 for it. Widening and narrowing are decided on
        # `bits`, not on `itemsize`, so that i4 -> i8 reads as the widening it
        # is rather than as a same-size conversion with no op.
        self.bits = bits if bits is not None else np.dtype(np_dtype).itemsize * 8
        # False for a type a buffer cannot hold. i4 is the only one: a DMA moves
        # whole bytes, the L1 budget is in bytes, and nothing in this tree wants
        # a nibble-addressed memref. It exists to name the *result* of
        # ops.bitcast -- half-bytes read out of a byte buffer -- and lives only
        # inside one expression.
        self.allocatable = allocatable

    @property
    def itemsize(self):
        if self.np_dtype is None:
            # A sub-byte type has no honest answer: numpy would store an i4 in a
            # whole byte and report 1, which is the number that made i4 -> i8
            # look like a same-size conversion in the first place. Callers that
            # mean "how wide" want `bits`; the ones that mean "how much memory"
            # only ever see allocatable types.
            raise TypeError(
                f"{self} has no byte size: it is {self.bits} bits and no buffer "
                f"can hold one. Use .bits for the width"
            )
        return np.dtype(self.np_dtype).itemsize

    def width_for(self, target):
        """Default lanes on ``target``, which is not the same on every one.

        npu2's backend legalizes a 512-bit bf16 vector and npu1's does not, so a
        single number has to be npu1's or be wrong somewhere. It was npu1's, and
        every bf16 elementwise loop on npu2 ran at half the width the hardware
        offers -- measured at 7% end to end on the int4 decode block, whose
        hand-written predecessor used 512-bit throughout.
        """
        if self.wide_vector_width is not None and target == "npu2":
            return self.wide_vector_width
        return self.default_vector_width

    def mlir(self):
        """The MLIR element type. Requires an active Context."""
        # Imported lazily: air.backend pulls in the MLIR bindings, and importing
        # air.api must stay cheap for callers that only want the type objects.
        from air.backend.xrt_runner import type_mapper

        if self.bits % 8:
            # type_mapper is keyed on numpy dtypes and cross-checks the MLIR
            # width against numpy's itemsize, which a sub-byte type fails by
            # construction: numpy stores an int4 in a whole byte. Build it
            # directly.
            from air.ir import IntegerType

            return IntegerType.get_signless(self.bits)
        return type_mapper(self.np_dtype)

    def __repr__(self):
        return f"air.api.{self.name}"


bf16 = DType("bf16", bfloat16, 16, is_float=True, wide_vector_width=32)
# AIE2 and AIE2P have no fp16 instruction, scalar or vector -- bf16 is the
# 16-bit float the hardware implements. Nothing in the toolchain refuses an f16
# kernel, though: it compiles, runs, and returns the result of having read the
# f16 bit patterns as bf16. Measured on npu1, `c[:] = a[:] + b[:]` over f16
# buffers is wrong on 2048 of 2048 elements -- f16 512.5 is 0x6001, and 0x6001
# read as bf16 is 1.0078125 x 2^65, which is what came back. Declared anyway so
# the type can still describe f16 *data* being moved, which does work.
f16 = DType("f16", np.float16, 16, is_float=True, computes=False)
f32 = DType("f32", np.float32, 16, is_float=True)
# Half a byte. Not a type a buffer can hold -- a DMA moves whole bytes and the
# L1 budget is counted in them -- but the type a *packed* buffer's contents are,
# and so the type `ops.bitcast` names when it reinterprets bytes as the pairs of
# quantised weights inside them. numpy has no sub-byte storage, so ml_dtypes'
# int4 stands in for the data type while `bits=4` carries the width that
# actually matters.
# `np_dtype=None`: there is deliberately no numpy type behind this one. numpy
# has no sub-byte storage, and ml_dtypes' int4 -- the obvious stand-in -- would
# make importing air.api fail on any older ml_dtypes that predates it, for a
# type most kernels never name. Nothing needs it: `mlir()` builds the MLIR type
# from `bits`, `type_mapper` is never reached, and `itemsize` is only ever asked
# of a type a buffer can hold.
i4 = DType("i4", None, 64, is_float=False, bits=4, allocatable=False)
i8 = DType("i8", np.int8, 32, is_float=False)
i16 = DType("i16", np.int16, 16, is_float=False)
i32 = DType("i32", np.int32, 16, is_float=False)

# Unsigned integers exist so that a kernel over `np.uint8` data can say so. The
# alternative -- declaring i8 and passing uint8 arrays -- type-checks nowhere
# and makes the emitted `func.func private @...` declaration disagree with the
# C prototype the object file was compiled from.
#
# The vector width is *taken from* the signed counterpart rather than repeated,
# so the two cannot drift. They were repeated as literals when these types
# landed, and i32's 8 -> 16 correction in this PR immediately left ui32 holding
# the width now known not to legalize -- which is the silent divergence that
# stating them was meant to prevent. Unreachable while arithmetic on unsigned is
# refused, and stated anyway so that relaxing that refusal does not also quietly
# change the width.
ui8 = DType("ui8", np.uint8, i8.default_vector_width, is_float=False, is_unsigned=True)
ui16 = DType(
    "ui16", np.uint16, i16.default_vector_width, is_float=False, is_unsigned=True
)
ui32 = DType(
    "ui32", np.uint32, i32.default_vector_width, is_float=False, is_unsigned=True
)

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


def require_computable(dtype, what):
    """Refuse ``what`` on an element type the AIE core cannot compute on.

    The same shape of restriction as :func:`require_signless`, for a different
    reason, and enforced at the same four call sites: a buffer of such a type
    can be declared, allocated, moved with ``ops.load``/``store`` and
    ``air.channel``, and handed to an ``air.extern`` kernel, but no arith or
    linalg op may be built over it.

    Today that is f16 alone. It matters because the failure it replaces is
    silent: AIE2 has no fp16 instruction, and rather than refusing an f16
    kernel the toolchain compiles one that reads the f16 bit patterns as bf16.
    Nothing raises, nothing warns, and the numbers come back wrong -- which is
    the failure mode this package exists to eliminate.
    """
    if dtype.computes:
        return
    raise NotImplementedError(
        f"{what} is not supported for {dtype}: neither NPU generation has an "
        f"fp16 instruction, scalar or vector, so there is nothing to lower it "
        f"to. This is not caught downstream -- the backend reinterprets the "
        f"f16 bits as bf16 and returns wrong numbers with no error at all, so "
        f"it is refused here instead. Use air.api.bf16, which is the 16-bit "
        f"float the hardware implements, or air.api.f32. An air.api.f16 buffer "
        f"can still be declared and moved, so f16 *data* can be transferred "
        f"and passed to an air.extern kernel."
    )
