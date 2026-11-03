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


def fused_prefill_model(model):
    """Whether `model` prefills on the shared fused prefill (fused_prefill/affine.py)."""
    from .deployment import spec

    return spec(model).get("prefill") == "fused"


def _fused_prefill_key(model):
    """Identity of a fused prefill build: its sources, model shape and toolchain."""
    import hashlib
    import os

    from .fused_prefill.models import MODELS

    here = Path(__file__).resolve().parent / "fused_prefill"
    mm = here.parents[2] / "matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc"
    h = hashlib.sha256(repr(MODELS[model]).encode())
    for p in sorted(here.rglob("*")) + [mm]:
        if p.is_file() and p.suffix in (".py", ".cc", ".c", ".h"):
            h.update(str(p.relative_to(here.parents[2])).encode() + p.read_bytes())
    for var in ("PEANO_INSTALL_DIR", "MLIR_AIE_INSTALL_DIR"):
        h.update(os.environ.get(var, "").encode())
    import air

    h.update(str(Path(air.__path__[0]).resolve()).encode())
    return h.hexdigest()


def build_fused_prefill(model, out, jobs=4):
    """Build `model`'s fused prefill into `out` once: an existing build of the
    same sources, shape and toolchain is kept."""
    from .decode_build import run
    from .deployment import save_json

    out = Path(out).resolve()
    key = _fused_prefill_key(model)
    stamp = out / "build.json"
    if (out / "manifest.json").is_file() and stamp.is_file():
        if json.loads(stamp.read_text()).get("key") == key:
            print(f"fused prefill up to date: {out}", flush=True)
            return
    out.mkdir(parents=True, exist_ok=True)
    stamp.unlink(missing_ok=True)
    main = Path(__file__).resolve().parent / "fused_prefill/dense_main.py"
    record = run(
        ["python3", main, "build", model, out, "-j", str(jobs)],
        out,
        out / "build.log",
    )
    save_json(stamp, dict(key=key, build=record))


class _ChunkRows:
    """[layers, rows, cols] of per-layer lists of row chunks, as seed_kv
    indexes it: [layer, :n, c0:c1] copies only that column block."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __getitem__(self, key):
        layer, rows, cols = key
        return np.concatenate([c[:, cols] for c in self.chunks[layer]])[rows]


class FusedPrefillSession:
    """The prefill of an affine-Q4 session on the shared fused prefill. The
    session provides self.decode (seed_kv), self.capacity and self.vocab, and
    calls open_prefill() while loading."""

    def open_prefill(self, model, directory, q4_root):
        from .fused_prefill import dense

        self.prefiller = dense.load(model, directory, q4_root, context=self.capacity)
        # one position stays free for the first decode step
        self.max_prompt = min(self.prefiller.max_len, self.capacity - 1)
        self.prefill_dispatches = 0
        self._pending_kv = None
        self.position = None

    def _ids(self, prompt_ids, limit):
        ids = np.asarray(prompt_ids, dtype=np.int64)
        if (
            ids.ndim != 1
            or not 0 < len(ids) <= limit
            or np.any((ids < 0) | (ids >= self.vocab))
        ):
            raise ValueError(f"expected 1 to {limit} token IDs below {self.vocab}")
        return ids.tolist()

    def prefill(self, prompt_ids):
        """Last-row logits of the prompt; its KV rows go to the decode in
        finish_prefill(), after the first token is chosen."""
        if self.closed:
            raise RuntimeError("session is closed")
        ids = self._ids(prompt_ids, self.max_prompt)
        self.position = None
        self.request_number += 1
        logits = self.prefiller.prefill(ids)
        self.prefill_dispatches += 1
        # the prefiller keeps this prompt's KV rows until its next prefill
        self._pending_kv = len(ids)
        self.position = len(ids)
        return np.asarray(logits, np.float32)

    def prefill_logits(self, prompt_ids, logits_from):
        """[n - logits_from, vocab] logits of the prompt's rows from logits_from
        on. No decode follows."""
        if self.closed:
            raise RuntimeError("session is closed")
        ids = self._ids(prompt_ids, self.prefiller.max_len)
        self.position = None
        self._pending_kv = None
        rows = self.prefiller.prefill(ids, logits_from=int(logits_from))
        self.prefill_dispatches += 1
        return np.asarray(rows, np.float32)

    def finish_prefill(self):
        """The physical KV handoff, after the first-token boundary."""
        if self._pending_kv is not None:
            if self.record_state:
                keys, values = self.prefiller.kv_bf16()
                self.last_prefill_keys, self.last_prefill_values = keys, values
            else:
                kv = self.prefiller.kv
                keys, values = (_ChunkRows([a[i] for a in kv]) for i in range(2))
            self.decode.seed_kv(keys, values, self._pending_kv)
            self._pending_kv = None

    def prefill_identity(self, directory):
        from .deployment import file_hash

        directory = Path(directory)
        return {
            f"fused_prefill/{p.name}": {
                "path": str(p.resolve()),
                "sha256": file_hash(p),
            }
            for pattern in ("*.xclbin", "*.txn.h", "*.o", "*.so", "manifest.json")
            for p in sorted(directory.glob(pattern))
        }
