# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Load compiled prefill caches and BF16 arrays for affine-Q4 sessions."""

import json
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16


def read_bf16(path):
    return np.load(path, mmap_mode="r", allow_pickle=False).view(bfloat16)


def load_cache(path, *, resident=False):
    from .infra.cache import KernelCache
    from air.backend.xrt import XRTCompileArtifact

    class PrefillCache(KernelCache):
        def load_and_run(self, name, backend_kwargs, *inputs, **kwargs):
            if resident:
                aliases = {
                    "rms_gemms_rope": {0: "hidden", 8: "v", 11: "q", 12: "k"},
                    "flash_attn": {0: "q", 1: "k", 2: "v", 3: "attention"},
                    "o_ffn": {0: "attention", 3: "hidden", 14: "hidden"},
                }
                if name in aliases:
                    inter = set(kwargs.get("intermediate_indices") or ())
                    if name == "rms_gemms_rope":
                        # Only layer zero receives new request embeddings from
                        # the host. Every later input is the preceding FFN output.
                        if kwargs.get("bo_key") != "rms_gemms_rope_L0":
                            inter.add(0)
                    elif name == "flash_attn":
                        inter.update((0, 1, 2, 3))
                    else:
                        inter.update((0, 3, 14))
                    kwargs.update(
                        shared_alias=aliases[name],
                        intermediate_indices=inter,
                        shared_nonstatic=True,
                    )
            return super().load_and_run(name, backend_kwargs, *inputs, **kwargs)

    cache = PrefillCache(cache_dir=str(path))
    info = json.loads((path / cache.MANIFEST_FILE).read_text())
    if info["_toolchain"] != cache._toolchain_id():
        raise RuntimeError(f"build toolchain changed: {path}")
    for name, entry in info["entries"].items():
        binary = path / Path(entry["output_binary"]).name
        insts = path / Path(entry["insts"]).name if entry["insts"] else None
        if not binary.is_file():
            raise FileNotFoundError(binary)
        if insts is not None and not insts.is_file():
            raise FileNotFoundError(insts)
        kernel = entry["kernel"]
        if kernel == "MLIR_AIE" and binary.suffix == ".elf":
            # Legacy manifests did not store the outer runtime symbol.
            import re

            (symbol,) = re.findall(
                r"func\.func @(\w+)\(", (path.parent / f"{name}.mlir").read_text()
            )
            kernel = "main:" + symbol
        cache.artifacts[name] = XRTCompileArtifact(
            str(binary), kernel, str(insts) if insts else None
        )
    return cache
