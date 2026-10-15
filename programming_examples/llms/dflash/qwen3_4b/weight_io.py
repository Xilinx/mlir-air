# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bounded-memory upload of a cache's unchanged packed W.npy payload."""

import math
import zipfile
import numpy as np


def upload_weights(bo, path, elements, chunk_bytes=32 << 20):
    """Stream original bytes and check every uploaded host-mapped word.

    zipfile validates its CRC at EOF. No dequantization or repacking occurs.
    The caller retains responsibility for BO synchronization before execution.
    """
    with zipfile.ZipFile(path) as archive, archive.open("W.npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }.get(version)
        if reader is None:
            raise ValueError(("Unsupported npy version", version))
        shape, fortran, dtype = reader(stream)
        if fortran or dtype.itemsize != 2 or math.prod(shape) != elements:
            raise ValueError(
                ("Packed weight shape/dtype mismatch", shape, fortran, dtype, elements)
            )
        mapped = np.frombuffer(bo.map(), np.uint16, count=elements)
        offset = 0
        while offset < elements * 2:
            raw = stream.read(min(chunk_bytes, elements * 2 - offset))
            if not raw or len(raw) % 2:
                raise ValueError("Truncated packed weight cache")
            words = np.frombuffer(raw, np.uint16)
            bo.write(words.view(np.int16), offset)
            if not np.array_equal(
                mapped[offset // 2 : offset // 2 + words.size], words
            ):
                raise ValueError(("Weight BO upload mismatch", offset))
            offset += len(raw)
        if stream.read(1):
            raise ValueError("Unexpected trailing weight payload")
