# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Use the standard Q4NX NPU prefill, capturing DFlash's layer-boundary taps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from ml_dtypes import bfloat16

from paths import EXAMPLES, HERE
from model_paths import default_prefill_cache

# Register the namespace explicitly before importing shared example modules.
# Do not rely on registration as a side effect of the native prefill import.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(EXAMPLES)
]

sys.path.insert(0, str(HERE.parents[1] / "llama31_8b_q4nx"))
from llama31_8b_q4nx_prefill import LlamaQ4nxPrefill


from common.prefill_cache import NPU2Cache


class NpuPrefill(LlamaQ4nxPrefill):
    """Same prefill arithmetic and kernels as llama31_8b_q4nx.

    The shared runner already returns each layer's residual output on the host.
    Copy selected boundaries before its transient buffers are reused. Both KV
    and taps contain only real prompt rows; padded rows never enter DFlash.
    BF16 weights are mapped from a separately prepared disk cache, with one
    reusable device weight set per kernel. This avoids keeping full host and
    NPU copies of all 32 layers resident alongside the decode models.
    Host argument scratch is transient per layer; device BOs are reused.
    """

    def __init__(
        self,
        *,
        seq_len=2048,
        cache_dir=None,
        tap_slots=(2, 9, 16, 23, 30),
        weight_cache=None,
    ):
        if seq_len not in (2048, 4096):
            raise ValueError(
                "The Llama-3.1-8B prefill GEMM registry supports 2048 or 4096 padded rows"
            )
        if os.environ.get("Q4NX_CPU_ATTN") == "1":
            raise ValueError("NPU prefill requires NPU attention; unset Q4NX_CPU_ATTN")
        self.tap_slots = tuple(int(slot) for slot in tap_slots)
        if not self.tap_slots or len(set(self.tap_slots)) != len(self.tap_slots):
            raise ValueError("Tap boundaries must be nonempty and unique")
        if any(slot < 1 or slot > 32 for slot in self.tap_slots):
            raise ValueError(
                "Tap boundaries must refer to outputs after 1 through 32 layers"
            )
        self.boundaries = {}
        cache_dir = Path(cache_dir or HERE / f"build/prefill_C{seq_len}").resolve()
        self.weight_cache = Path(weight_cache or default_prefill_cache()).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "model": "llama31-8b-q4nx",
            "seq_len": seq_len,
            "layers": 32,
            "cpu_attention": False,
            "target_device": "npu2",
        }
        identity_path = cache_dir / "dflash_prefill_config.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError(
                    "NPU prefill cache geometry differs; select a separate cache directory"
                )
        elif (cache_dir / "manifest.json").exists():
            raise ValueError(
                "Existing prefill cache has no DFlash geometry identity; select a fresh cache directory"
            )
        else:
            identity_path.write_text(json.dumps(identity, indent=2) + "\n")
        # Shared builders create .o and air_project files in CWD. Keep all of
        # those files inside this example's build directory, even for auto-build.
        previous = Path.cwd()
        try:
            os.chdir(cache_dir)
            super().__init__(
                seq_len=seq_len,
                cache_dir=str(cache_dir),
                cache=NPU2Cache(cache_dir=str(cache_dir)),
                cache_host_args=False,
            )
        finally:
            os.chdir(previous)

    def load_weights(self, model=None):
        from model_paths import TARGET_MODEL, resolve_target_model
        from prefill_weights import MappedWeights

        self._weights = MappedWeights(
            self.weight_cache, resolve_target_model(model or TARGET_MODEL)
        )
        self._embed = self._weights.embed_table
        self._final_norm = np.asarray(self._weights.final_norm, np.float32)
        self._lm_head = self._weights.lm_head
        self._lm_parts = [self._lm_head[p * 16384 : (p + 1) * 16384] for p in range(8)]
        print(
            "[q4nx_prefill] mapped BF16 cache; one reusable weight BO set per kernel",
            flush=True,
        )

    def share_embedding(self, shared_embedding):
        shared = np.asarray(shared_embedding)
        if shared.shape != self._embed.shape or shared.dtype != self._embed.dtype:
            raise ValueError("Shared target embedding has incompatible shape/dtype")
        for first in range(0, len(shared), 256):
            if not np.array_equal(
                shared[first : first + 256], self._embed[first : first + 256]
            ):
                raise ValueError("Prefill/decode embeddings differ")
        # This model's LM head is untied: only share the embedding table.
        self._embed = self._weights.embed_table = shared

    def _lm_head_npu(self, hidden_bf16):
        try:
            return super()._lm_head_npu(hidden_bf16)
        finally:
            self._weights.release_head()

    def _run_layer(self, x, layer, real_length):
        try:
            output = super()._run_layer(x, layer, real_length)
        finally:
            self._weights.release_layer(layer)
        if layer + 1 in self.tap_slots:
            self.boundaries[layer + 1] = np.asarray(
                output[:real_length], bfloat16
            ).copy()
        return output

    def prepare(self, input_ids):
        if not 0 < len(input_ids) <= self.seq:
            raise ValueError(f"NPU prefill needs 1 through {self.seq} prompt tokens")
        started = time.perf_counter()
        self.clear_context()
        self.boundaries.clear()
        logits = self.prefill(input_ids)
        k = np.stack([self.kv_view(i)[0] for i in range(self.n_layers)])
        v = np.stack([self.kv_view(i)[1] for i in range(self.n_layers)])
        taps = np.stack([self.boundaries[slot] for slot in self.tap_slots], axis=1)
        result = {
            "input_ids": list(input_ids),
            "k": k,
            "v": v,
            "taps": taps.reshape(len(input_ids), -1),
            "first": int(np.asarray(logits).argmax()),
            "logits": np.asarray(logits, np.float32).copy(),
        }
        return result, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    prefill = NpuPrefill(seq_len=args.seq_len, cache_dir=args.cache_dir)
    print(
        f"NPU prefill compiled: {len(prefill.cache.artifacts)} kernels, C{prefill.seq}",
        flush=True,
    )


if __name__ == "__main__":
    main()
