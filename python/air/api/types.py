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

__all__ = ["DType", "bf16", "f16", "f32", "i8", "i16", "i32", "dtype_of"]


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

    Widths for f16, i8, i16 and i32 are extrapolated by element size and have
    not been compiled; treat them as unverified.
    """

    def __init__(self, name, np_dtype, default_vector_width, is_float):
        self.name = name
        self.np_dtype = np_dtype
        self.default_vector_width = default_vector_width
        # Stated, not inferred: np.issubdtype(bfloat16, np.floating) is False
        # for the ml_dtypes extension types, which would silently select the
        # integer arithmetic ops for a bf16 kernel.
        self.is_float = is_float

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
i32 = DType("i32", np.int32, 8, is_float=False)

_BY_NP = {d.np_dtype: d for d in (bf16, f16, f32, i8, i16, i32)}


def dtype_of(np_dtype):
    """Map a numpy dtype back onto the API's DType, for error messages.

    Accepts either the scalar type (``np.float32``) or a dtype instance
    (``np.dtype("float32")``); the table is keyed by the former.
    """
    return _BY_NP.get(np_dtype) or _BY_NP.get(getattr(np_dtype, "type", None))
