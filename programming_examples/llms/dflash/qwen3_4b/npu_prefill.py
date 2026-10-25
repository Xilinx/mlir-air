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

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "qwen3_4b_q4nx"))
from qwen3_4b_q4nx_prefill import Qwen3Q4nxPrefill


class NpuPrefill(Qwen3Q4nxPrefill):
    """Same prefill arithmetic and kernels as qwen3_4b_q4nx.

    The shared runner already returns each layer's residual output on the host.
    Copy selected boundaries before its transient buffers are reused. Both KV
    and taps contain only real prompt rows; padded rows never enter DFlash.
    """

    def __init__(self, *, seq_len=2048, cache_dir=None, tap_slots=(2, 10, 18, 26, 34)):
        if seq_len not in (2048, 4096):
            raise ValueError(
                "The Qwen3-4B prefill GEMM registry supports 2048 or 4096 padded rows"
            )
        if os.environ.get("Q4NX_CPU_ATTN") == "1":
            raise ValueError("NPU prefill requires NPU attention; unset Q4NX_CPU_ATTN")
        self.tap_slots = tuple(int(slot) for slot in tap_slots)
        if not self.tap_slots or len(set(self.tap_slots)) != len(self.tap_slots):
            raise ValueError("Tap boundaries must be nonempty and unique")
        if any(slot < 1 or slot > 36 for slot in self.tap_slots):
            raise ValueError(
                "Tap boundaries must refer to outputs after 1 through 36 layers"
            )
        self.boundaries = {}
        cache_dir = Path(cache_dir or HERE / f"build/prefill_C{seq_len}").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "model": "qwen3-4b-q4nx",
            "seq_len": seq_len,
            "layers": 36,
            "cpu_attention": False,
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
            super().__init__(seq_len=seq_len, cache_dir=str(cache_dir))
        finally:
            os.chdir(previous)

    def load_weights(self, model=None, *, shared_embedding=None):
        from model_paths import TARGET_MODEL, resolve_target_model

        super().load_weights(model=resolve_target_model(model or TARGET_MODEL))
        if shared_embedding is not None:
            self.share_embedding(shared_embedding)

    def _preload_lm_head_gemv(self):
        # The shared prefill loader calls this after uploading all block weights.
        # Release duplicate host MLP storage before allocating LM-head buffers,
        # while keeping the same per-layer resident BOs and exact weight values.
        self._reuse_resident_mlp_weights()
        super()._preload_lm_head_gemv()

    def share_embedding(self, shared_embedding):
        shared = np.asarray(shared_embedding)
        if shared.shape != self._embed.shape or shared.dtype != self._embed.dtype:
            raise ValueError("Shared target embedding has incompatible shape/dtype")
        # Verify equality in bounded chunks before sharing immutable storage.
        for first in range(0, len(shared), 256):
            if not np.array_equal(
                shared[first : first + 256], self._embed[first : first + 256]
            ):
                raise ValueError("Prefill/decode embeddings differ")
        self._embed = self._lm_head = shared
        self._weights.embed_table = self._weights.lm_head = shared

    def _reuse_resident_mlp_weights(self):
        """Share already-uploaded static weights, preserving per-layer BO keys.

        The upstream prefill calls place each projection weight at argument 1.
        KernelCache keeps these BOs alive across hardware-context eviction and
        skips static uploads after preload. The kernels never overwrite them.
        This removes a second host copy of gate/up/down (~5 GiB for 36 layers).
        """
        if not getattr(self._weights, "_prefill_weights_preloaded", False):
            raise ValueError("Prefill weights must be uploaded before sharing")
        for layer, weights in enumerate(self._weights.layers):
            for kernel, field in (
                ("gate", "w_gate"),
                ("up", "w_up"),
                ("down_add", "w_down"),
            ):
                original = getattr(weights, field)
                bo = self.cache._cached_bos[f"{kernel}_L{layer}"][1]
                if bo.size() != original.nbytes or not original.flags.c_contiguous:
                    raise ValueError(f"Unexpected {kernel} weight buffer layout")
                view = np.frombuffer(
                    bo.map(), original.dtype, count=original.size
                ).reshape(original.shape)
                for first in range(0, len(original), 128):
                    if not np.array_equal(
                        view[first : first + 128], original[first : first + 128]
                    ):
                        raise ValueError(
                            f"Resident {kernel} weights differ at layer {layer}"
                        )
                view.flags.writeable = False
                setattr(weights, field, view)

    def _run_layer(self, x, layer, real_length):
        output = super()._run_layer(x, layer, real_length)
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
        k, v = self.kv_stack()
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
