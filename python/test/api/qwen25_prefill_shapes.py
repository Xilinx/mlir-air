# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""Qwen2.5-3B prefill: mixed drain widths and cached runtime argument layouts.

Build the actual modules without a device or weights. The short Q projection
and K/V used to share a symbol with incompatible types. Loading cached short
ELFs also reconstructed the 2048-token scratch ABI instead of their own.
"""

import sys
from pathlib import Path
from unittest.mock import patch

root = Path(sys.argv[1]) / "programming_examples"
sys.path[:0] = [str(root / "llms/qwen25_3b"), str(root / "llms"), str(root)]
import qwen25_3b_prefill as prefill
from qwen25_3b_weights import LlamaConfig
from shared.infra import external_kernels

config = LlamaConfig()


class ModuleCache:
    def __init__(self):
        self.artifacts = {}
        self.cache_dir = "unused"

    def compile_and_cache(self, name, module, backend):
        assert module.operation.verify(), name
        self.artifacts[name] = module

    def _save_manifest(self):
        pass


# Build real modules at each registry transition and validate their runtime ABI.
for seq in (512, 1024, 2048, 4096):
    cache = ModuleCache()
    recipes = []
    with (
        patch.object(
            external_kernels,
            "compile_gemm_mm",
            side_effect=lambda **kw: recipes.append(kw),
        ),
        patch.object(external_kernels, "compile_rope"),
        patch.object(external_kernels, "compile_silu_and_mul"),
    ):
        prefill.compile_all_kernels(cache, config, seq, cpu_attn=True)

    # Every external GEMM reference must have a compiled object of the matching
    # tile shape. In particular the short Q and K/V drain objects must differ.
    q = prefill._gemm_spec(seq, 2048, 2048, "high")
    kv = prefill._gemm_spec(seq, 2048, 256, "high")
    assert q["obj"] != kv["obj"]
    assert q["sym_suffix"] != kv["sym_suffix"]
    for spec in (q, kv, prefill._gemm_spec(seq, 11008, 2048, "high")):
        matches = [r for r in recipes if r["out_name"] == spec["obj"]]
        assert len(matches) == 1
        assert tuple(matches[0][k] for k in ("tile_m", "tile_n", "tile_k_l1")) == tuple(
            spec[k] for k in ("tile_m", "tile_n", "tile_k_l1")
        )
        assert matches[0]["sym_suffix"] == spec["sym_suffix"]

    runtime_scratch = prefill._resolve_scratch_for(seq, config)
    expected = (
        ([None, None, None], [None], [5])
        if seq < 2048
        else ([19, None, None], [7], [5])
    )
    assert runtime_scratch == expected
    for name, base, scratch in zip(
        ("rms_qkv_bias_rope", "o_res_norm", "down_add"), (19, 7, 5), runtime_scratch
    ):
        fn = next(
            op
            for op in cache.artifacts[name].body.operations
            if op.operation.name == "func.func"
            and str(op.attributes["sym_name"]) == f'"{name}"'
        )
        assert len(fn.type.inputs) == base + sum(x is not None for x in scratch)
    # A cached short engine must keep its own ABI even after a larger build.
    assert prefill._resolve_scratch_for(512, config) == (
        [None, None, None],
        [None],
        [5],
    )
    print(f"PASS seq={seq}: compiled modules, external recipes, cached scratch ABI")

# CHECK: PASS seq=512:
# CHECK: PASS seq=1024:
# CHECK: PASS seq=2048:
# CHECK: PASS seq=4096:
