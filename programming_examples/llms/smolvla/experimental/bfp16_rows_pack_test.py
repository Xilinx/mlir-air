# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys, time
import types
from pathlib import Path

# programming_examples/ is published as the air_examples package rather than
# put on sys.path: every directory under it would otherwise become a
# top-level module name and shadow an installed package that shares it.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[3])
]
import numpy as np
from ml_dtypes import bfloat16
from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
    pack_b_bfp16ebs8,
)
from bfp16_rows_pack import pack_rows_bfp16ebs8


def ref(rows):  # rows [R,8] f32
    B = rows.T.astype(bfloat16)
    return pack_b_bfp16ebs8(B, 8, 8).reshape(-1, 9)


rng = np.random.default_rng(1)
allbf = np.arange(65536, dtype=np.uint16).view(bfloat16).astype(np.float32)
allbf = allbf[~np.isnan(allbf)]
allbf = allbf[
    : len(allbf) // 8 * 8
]  # NaN payloads are not preserved by the reference's f32->bf16 cast
cases = {
    "all 65536 bf16 patterns (shuffled rows)": rng.permutation(allbf).reshape(-1, 8),
    "all patterns, 3 shuffles": np.concatenate(
        [rng.permutation(allbf) for _ in range(3)]
    ).reshape(-1, 8),
    "randn": rng.standard_normal((200000, 8)).astype(bfloat16).astype(np.float32),
    "wide exponent spread": (
        rng.standard_normal((200000, 8)) * 2.0 ** rng.integers(-60, 60, (200000, 8))
    )
    .astype(bfloat16)
    .astype(np.float32),
    "zeros/-0 mixed with big": np.where(
        rng.random((100000, 8)) < 0.5,
        rng.choice([0.0, -0.0], (100000, 8)),
        1e30 * rng.standard_normal((100000, 8)),
    )
    .astype(bfloat16)
    .astype(np.float32),
}
ok = True
for name, x in cases.items():
    a, b = pack_rows_bfp16ebs8(x), ref(x)
    same = np.array_equal(a, b)
    ok &= same
    print(
        f"{name:45s} rows {len(x):7d}  identical: {same}"
        + ("" if same else f"  mismatching rows {np.any(a!=b,axis=1).sum()}")
    )
x = rng.standard_normal((312960, 8)).astype(bfloat16).astype(np.float32)
for f, nm in ((pack_rows_bfp16ebs8, "new"), (ref, "pack_b_bfp16ebs8 (+ .T/bf16 cast)")):
    ts = []
    for _ in range(6):
        t = time.perf_counter()
        f(x)
        ts.append((time.perf_counter() - t) * 1e3)
    print(f"{nm:40s} {np.median(ts[1:]):6.1f} ms for 312960 rows")
print("ALL IDENTICAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
