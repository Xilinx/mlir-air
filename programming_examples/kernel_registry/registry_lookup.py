# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Programmatic access to the kernel_registry machine-readable config (the
``details/*.json`` files).

These JSON files are the single source of truth for each kernel's best measured
tile config + method + performance, per shape, on NPU2. The matching ``.md``
pages mirror them for humans; model code (e.g. llama) calls the lookups here so
tile sizes are never hand-copied (which caused drift / stale-config bugs).

Currently provides GEMM lookups; other kernels add their own as their JSON lands.
"""

import json
from pathlib import Path

_DETAILS = Path(__file__).resolve().parent / "details"

# output_dtype -> registry JSON filename
_GEMM_JSON = {
    "bf16": "GEMM_bf16_in_bf16_out.json",
    "f32": "GEMM_bf16_in_fp32_out.json",
}

_cache = {}


def _load(filename):
    if filename not in _cache:
        path = _DETAILS / filename
        if not path.exists():
            raise FileNotFoundError(f"registry JSON not found: {path}")
        _cache[filename] = json.loads(path.read_text())
    return _cache[filename]


def gemm_config(M, K, N, output_dtype="bf16", precision="high"):
    """Best measured GEMM config for one shape + contract, from the registry JSON.

    Args:
        M, K, N: GEMM dims, C[M,N] = A[M,K] @ B[K,N].
        output_dtype: "bf16" or "f32" (selects which registry page).
        precision: "high" (FP32-accumulate, GPU standard ~9.3e-3) or "low"
            (bf16-out only; per-L2-tile truncation, faster but 1.0-1.9e-2).

    Returns dict:
        {"method": str,
         "tile": {"tile_m":.., "tile_k_l2":.., "tile_k_l1":.., "tile_n":..},
         "gflops": float, "mean_rel_L1": float}
        - method names: f32 page -> external/direct; bf16 page -> fused-cast/drain/direct.
        - tile is a named dict (self-describing; copied from the JSON verbatim).

    Raises:
        KeyError if (M,K,N) is not in the registry for this dtype, or the
        requested precision tier has no measured entry. Message tells you to run
        a sweep + add the shape to the JSON (no silent fallback to a guessed config).
    """
    if output_dtype not in _GEMM_JSON:
        raise ValueError(
            f"gemm_config: output_dtype must be one of {sorted(_GEMM_JSON)}, got {output_dtype!r}"
        )
    data = _load(_GEMM_JSON[output_dtype])
    for s in data["shapes"]:
        if (s["M"], s["K"], s["N"]) == (M, K, N):
            best = s.get("best", {})
            if precision not in best:
                raise KeyError(
                    f"gemm_config: shape {M}x{K}x{N} (out={output_dtype}) has no "
                    f"'{precision}'-precision best in {_GEMM_JSON[output_dtype]} "
                    f"(available: {sorted(best)}). Run a sweep for this tier and add it."
                )
            method = best[precision]
            m = s["methods"][method]
            return {
                "method": method,
                "tile": dict(m["tile"]),
                "gflops": m["gflops"],
                "mean_rel_L1": m["mean_rel_L1"],
            }
    raise KeyError(
        f"gemm_config: shape {M}x{K}x{N} (out={output_dtype}) not in registry "
        f"{_GEMM_JSON[output_dtype]}. Measured shapes: "
        f"{[(s['M'], s['K'], s['N']) for s in data['shapes']]}. "
        f"Run a sweep for this shape (matrix_multiplication/bf16_in_{'bf16' if output_dtype=='bf16' else 'fp32'}_out) "
        f"and add it to the JSON before using it."
    )


def gemm_config_method(M, K, N, output_dtype, method, precision="high"):
    """Like gemm_config, but return a SPECIFIC method's measured config (not the
    best). Use when a caller forces a method (e.g. all-drain A/B comparison) but
    still wants the registry's tiles for it. Raises KeyError if the shape or the
    requested method isn't in the registry for this dtype."""
    if output_dtype not in _GEMM_JSON:
        raise ValueError(
            f"gemm_config_method: output_dtype must be one of {sorted(_GEMM_JSON)}, got {output_dtype!r}"
        )
    data = _load(_GEMM_JSON[output_dtype])
    for s in data["shapes"]:
        if (s["M"], s["K"], s["N"]) == (M, K, N):
            if method not in s["methods"]:
                raise KeyError(
                    f"gemm_config_method: shape {M}x{K}x{N} (out={output_dtype}) has no "
                    f"method '{method}' in {_GEMM_JSON[output_dtype]} "
                    f"(available: {sorted(s['methods'])})."
                )
            m = s["methods"][method]
            return {
                "method": method,
                "tile": dict(m["tile"]),
                "gflops": m["gflops"],
                "mean_rel_L1": m["mean_rel_L1"],
            }
    raise KeyError(
        f"gemm_config_method: shape {M}x{K}x{N} (out={output_dtype}) not in registry "
        f"{_GEMM_JSON[output_dtype]}. Run a sweep and add it."
    )
