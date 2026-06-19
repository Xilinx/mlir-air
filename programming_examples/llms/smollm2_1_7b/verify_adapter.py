# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the bf16 SmolLM2-1.7B example.

Wraps the production `llama32_1b_inference` driver into a Runner that
satisfies `verify/runners/base.Runner`. The shared verify framework
(see `programming_examples/llms/verify/verify_runner.py`) imports this
module via `--runner=smollm2_1_7b.verify_adapter` and calls `build_runner`.

Nothing here is reachable from production code; it exists only so the
verify gate can exercise the exact same NPU code path that `make run`
uses.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Ensure llms/, this dir, llms/verify/, and the llama32_1b reference (whose
# production prefill/decode/inference we inherit bit-for-bit) are importable.
_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
_LLAMA_REF = _LLMS_DIR / "llama32_1b"
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_LLAMA_REF), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from shared.infra.cache import KernelCache  # noqa: E402

# SmolLM2 is pure MHA (kv_dim == emb_dim). The shared llama32_1b_prefill
# run_transformer_block / preload_prefill_weights are registry-driven and
# MHA-safe (per-shape gemm_registry_config supplies the fused-cast f32 C-scratch
# args), so they're used directly — no fork or monkeypatch.
from llama32_1b_prefill import (  # noqa: E402
    compile_all_kernels as compile_prefill_kernels,
    run_transformer_block as run_prefill_block,
)

from llama32_1b_decode import compile_decode_kernels  # noqa: E402
from llama32_1b_inference import (  # noqa: E402
    prepare_runtime,
    run_npu_prefill,
    run_npu_decode_step,
)
from smollm2_1_7b_weights import (  # noqa: E402
    LlamaConfig,
    load_weights,
    generate_rope_lut,
)
from smollm2_1_7b_cpu_helpers import rms_norm  # noqa: E402
from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

# CLI --model choice -> HF id. Both Llamas use the same architecture; only
# the weights and chat template differ.
MODEL_CHOICES = {
    "base": "HuggingFaceTB/SmolLM2-1.7B",
    "instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}
DEFAULT_MODEL = "base"


def resolve_model(model_choice_or_id: str) -> str:
    """`--model` accepts either a `MODEL_CHOICES` key (base/instruct) or a
    raw HF model id / local path. Return the HF id."""
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """HF reference checkpoint for `npu_model_name`. bf16 baseline is its
    own reference (verifies NPU bf16 vs HF bf16 on the same checkpoint)."""
    return npu_model_name


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
    """Load bf16 weights, compile NPU kernels, return a `NpuRunner`."""
    weights = load_weights(model_name, config=config)
    return NpuRunner(
        weights=weights,
        config=config,
        max_seq=max_seq,
        tokenizer=tokenizer,
        npu_attn=npu_attn,
        lite_mode=lite_mode,
    )


class NpuRunner:
    """Adapter over the bf16 production NPU prefill + decode functions."""

    name = "npu_bf16"

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

        # Per-model cache dirs (absolute, CWD-independent) so verify runs of
        # different models never share the default shared/infra/
        # kernel_cache/ and pick up each other's stale ELFs.
        _cache_root = _THIS_DIR / "verify_kernel_cache"
        self.prefill_cache = KernelCache(str(_cache_root / "prefill"), verbose=False)
        compile_prefill_kernels(
            self.prefill_cache,
            config,
            seq_len=max_seq,
            cpu_attn=self.cpu_attn,
        )
        self.decode_cache = KernelCache(str(_cache_root / "decode"), verbose=False)
        compile_decode_kernels(self.decode_cache, config)

        prepare_runtime(
            self.prefill_cache,
            self.decode_cache,
            weights,
            config,
            max_seq,
            self.rope_lut_bf16,
        )

        # Repopulated by prefill(); read by decode_step() within the same
        # verify run.
        self.k_cache = None
        self.v_cache = None

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        # Mirror production's eos-pad-to-max_seq before run_npu_prefill so
        # the verify path hits the same kernel shape make run does.
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
            profile=False,
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

        # Diagnosis-only: re-run the prefill layer loop to capture per-layer
        # ffn_out + final post-norm hidden state. ~3-5s extra; diagnosis is
        # single-prompt so the overhead doesn't matter.
        cfg = self.config
        if len(prompt_tokens) < self.max_seq:
            pad = np.zeros(self.max_seq - len(prompt_tokens), dtype=prompt_tokens.dtype)
            padded_diag = np.concatenate([prompt_tokens, pad])
        else:
            padded_diag = prompt_tokens[: self.max_seq]
        embed = self.weights.embed_table[padded_diag].astype(np.float32)
        x = embed.astype(bfloat16)
        layer_intermediates: list[dict[str, np.ndarray]] = []
        for li in range(cfg.n_layers):
            x, ints = run_prefill_block(
                x,
                self.weights.layers[li],
                self.rope_lut_bf16,
                cfg,
                self.prefill_cache,
                layer_idx=li,
                cpu_attn=self.cpu_attn,
                verbose=False,
            )
            fo_full = np.asarray(ints["ffn_out"])
            layer_intermediates.append({"ffn_out": fo_full[:prompt_len]})

        x_full_f32 = np.asarray(x, dtype=np.float32)[:prompt_len]
        x_full_normed = rms_norm(x_full_f32, self.weights.final_norm)

        return PrefillRecord(
            layer_intermediates=layer_intermediates,
            final_hidden_normed=x_full_normed.astype(np.float32),
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
