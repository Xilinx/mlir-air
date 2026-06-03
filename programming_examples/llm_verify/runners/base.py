# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runner protocol.

Each LLM example in `programming_examples/` contributes a `verify_adapter.py`
module exposing a `build_runner(config, max_seq, tokenizer, **kwargs)`
callable that returns an object satisfying this protocol. The shared
`verify_runner.py` driver imports the adapter by dotted path
(`--runner=<module.path>`), calls `build_runner(...)`, and then drives it
with `prefill(...)` and `decode_step(...)` exactly the way `npu_runner.py`
did for the bf16 baseline.

This is a structural Protocol (no inheritance required). An adapter
implementation can be any class; it just needs the two methods below.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from runners._records import DecodeStepRecord, PrefillRecord


class Runner(Protocol):
    """One LLM execution backend (NPU production driver, HF reference, ...)."""

    name: str

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        """Run the model's prefill over `prompt_tokens` (1D int64 array of
        token IDs). Returns a PrefillRecord with at minimum
        `logits_at_pred` and `top1_token` populated. In lite mode
        `layer_intermediates` and `final_hidden_normed` may be empty;
        non-lite (diagnosis) callers expect both populated.

        Implementations should mirror what their model's production
        driver does (padding, layer loop, LM head). Any mismatch with
        production = the verifier validates a different code path than
        what `make run` exercises.
        """
        ...

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        """Run one decode step starting from `input_token` at position
        `current_pos`. Must internally manage / update the runner's KV
        cache (populated by the most recent `prefill` call). Returns
        a DecodeStepRecord with `lm_head_logits` (vocab-sized) and
        `top1_token`.
        """
        ...
