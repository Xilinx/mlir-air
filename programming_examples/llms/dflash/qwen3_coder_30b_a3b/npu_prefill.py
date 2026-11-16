# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NPU prefill on the target's own verify pass, 8 prompt tokens per dispatch,
capturing DFlash's layer-boundary taps. It needs no build of its own and
shares the loaded target's weights."""

import time

import model_spec
from target_runtime import loaded_target


class NpuPrefill:
    def __init__(self, *, seq_len=2048, cache_dir=None, tap_slots=None):
        if seq_len > model_spec.CAPACITY:
            raise ValueError(f"The target's cache holds {model_spec.CAPACITY}")
        expected = [layer + 1 for layer in model_spec.TAP_LAYERS]
        if tap_slots is not None and list(tap_slots) != expected:
            raise ValueError(f"Tap boundaries {tap_slots}, the target taps {expected}")
        self.seq = seq_len

    def load_weights(self, model=None):
        pass

    def share_embedding(self, shared_embedding):
        pass

    def prepare(self, input_ids):
        if not 0 < len(input_ids) <= self.seq:
            raise ValueError(f"NPU prefill needs 1 through {self.seq} prompt tokens")
        target = loaded_target()
        if target is None:
            raise RuntimeError("The NPU prefill runs on the target; load it first")
        started = time.perf_counter()
        first, k, v, taps = target.prefill(list(input_ids))
        result = {
            "input_ids": list(input_ids),
            "k": k,
            "v": v,
            "taps": taps,
            "first": int(first),
        }
        return result, time.perf_counter() - started
