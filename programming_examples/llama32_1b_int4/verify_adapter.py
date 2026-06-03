# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the int4-AWQ Llama-3.2-1B example.

Wraps the production `llama32_1b_int4_inference` driver into a Runner for
the shared verify framework. Mirrors `llama32_1b/verify_adapter.py` but
loads AWQ weights (`load_weights_awq`) and uses the int4 decode kernels.

The HF reference still runs in bf16 — AutoAWQ is not in CI deps, and the
NPU int4 path's correctness target is "matches bf16 dequant" which the
top-K-set inclusion gate already measures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_PROG_EXAMPLES = _THIS_DIR.parent
_LLAMA_BF16 = _PROG_EXAMPLES / "llama32_1b"
_LLM_VERIFY = _PROG_EXAMPLES / "llm_verify"
for _p in (str(_PROG_EXAMPLES), str(_LLAMA_BF16), str(_LLM_VERIFY), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from llama_kernel_builder.cache import KernelCache  # noqa: E402
from llama32_1b_weights import LlamaConfig, generate_rope_lut  # noqa: E402
from llama32_1b_cpu_helpers import rms_norm  # noqa: E402
from llama32_1b_int4_weights import load_weights_awq  # noqa: E402
from llama32_1b_int4_inference import (  # noqa: E402
    _multi_launch_dir,
    prepare_runtime,
    run_npu_prefill,
    run_npu_decode_step,
)
from llama32_1b_int4_decode import compile_decode_kernels  # noqa: E402
from llama32_1b_prefill import (
    compile_all_kernels as compile_prefill_kernels,
)  # noqa: E402
from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

# Default AWQ checkpoint exposed by AMD; un-gated, no HF_TOKEN required to
# fetch the AWQ weights themselves. The tokenizer behind it (the upstream
# meta-llama/Llama-3.2-1B-Instruct tokenizer) IS gated, so `make verify`
# still needs HF_TOKEN.
_DEFAULT_AWQ_MODEL = "amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead"

MODEL_CHOICES = {
    "instruct": _DEFAULT_AWQ_MODEL,
}
DEFAULT_MODEL = "instruct"


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


# HF reference is the un-quantized upstream Meta checkpoint. The AWQ
# weights are a particular quantization of it; the verify gate measures
# NPU int4 (with AWQ weights) vs HF bf16 (with the source weights),
# which is a looser comparison than NPU-int4 vs HF-AutoAWQ-dequant but
# avoids dragging the autoawq package into CI deps.
_HF_REF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def hf_reference(npu_model_name: str) -> str:
    return _HF_REF_MODEL


def build_config():
    return LlamaConfig()


def build_runner(
    model_name: str,
    config,
    max_seq: int,
    tokenizer,
    *,
    npu_attn: bool = True,
    lite_mode: bool = False,
):
    """Load AWQ weights, compile NPU kernels (bf16 prefill + int4 decode),
    return an `Int4NpuRunner`."""
    weights = load_weights_awq(model_name, config=config)
    return Int4NpuRunner(
        weights=weights,
        config=config,
        max_seq=max_seq,
        tokenizer=tokenizer,
        npu_attn=npu_attn,
        lite_mode=lite_mode,
    )


class Int4NpuRunner:
    """Adapter over the int4-AWQ production NPU prefill + decode functions.

    Prefill is NPU bf16 (on dequantized AWQ weights) since the int4 prefill
    path is currently kernel-bound; decode is NPU int4 (the ELFs landed
    with this verify subsystem move).
    """

    name = "npu_int4"

    def __init__(
        self,
        weights,
        config,
        max_seq: int,
        tokenizer,
        npu_attn: bool = True,
        lite_mode: bool = False,
    ):
        self.weights = weights
        self.config = config
        self.max_seq = max_seq
        self.npu_attn = npu_attn
        self.cpu_attn = not npu_attn
        self.lite_mode = lite_mode
        self._tokenizer = tokenizer

        self.rope_lut_bf16 = generate_rope_lut(config=config, seq_len=max_seq).astype(
            bfloat16
        )

        # Same multi_launch_builder collision as in inference.py — flush
        # the cached package + repin sys.path[0] around each compile phase.
        self.prefill_cache = KernelCache(verbose=False)
        with _multi_launch_dir(str(_LLAMA_BF16)):
            compile_prefill_kernels(
                self.prefill_cache, config, seq_len=max_seq, cpu_attn=self.cpu_attn
            )
        self.decode_cache = KernelCache(verbose=False)
        with _multi_launch_dir(str(_THIS_DIR)):
            compile_decode_kernels(self.decode_cache, config)

        prepare_runtime(
            self.prefill_cache,
            self.decode_cache,
            weights,
            config,
            max_seq,
            self.rope_lut_bf16,
        )

        self.k_cache = None
        self.v_cache = None

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        # Pad to max_seq (mirrors run_once in production driver). int4
        # production driver uses the same shape so the test path is
        # identical to make run.
        eos = self._tokenizer.eos_token_id
        if len(prompt_tokens) < self.max_seq:
            padded = list(prompt_tokens) + [eos] * (self.max_seq - len(prompt_tokens))
        else:
            padded = list(prompt_tokens)[: self.max_seq]
        prefill_token, logits_row, k_cache, v_cache, prompt_len = run_npu_prefill(
            padded,
            self.weights,
            self.config,
            self.prefill_cache,
            self.decode_cache,
            self.rope_lut_bf16,
            self.max_seq,
            tokenizer=self._tokenizer,
            cpu_attn=self.cpu_attn,
            quiet=True,
        )
        self.k_cache = k_cache
        self.v_cache = v_cache

        if self.lite_mode:
            empty = np.empty((0,), dtype=np.float32)
            return PrefillRecord(
                layer_intermediates=[],
                final_hidden_normed=empty,
                logits_at_pred=logits_row,
                top1_token=prefill_token,
            )
        # Diagnosis path: not implemented for int4 yet — the int4 prefill
        # block doesn't expose per-layer intermediates. Lite mode is the
        # only verified path; return empties so the diagnosis comparator
        # at least surfaces the missing data clearly.
        empty = np.empty((0,), dtype=np.float32)
        return PrefillRecord(
            layer_intermediates=[{} for _ in range(self.config.n_layers)],
            final_hidden_normed=empty,
            logits_at_pred=logits_row,
            top1_token=prefill_token,
        )

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        x = self.weights.embed_table[input_token].astype(bfloat16)
        next_token, logits = run_npu_decode_step(
            x,
            self.weights,
            self.config,
            self.decode_cache,
            self.rope_lut_bf16,
            self.k_cache,
            self.v_cache,
            current_pos,
        )
        return DecodeStepRecord(
            lm_head_logits=logits,
            top1_token=next_token,
        )
