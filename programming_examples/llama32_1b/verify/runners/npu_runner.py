# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""NPU runner — thin adapter over the production prefill / decode functions.

Delegates the actual work to:
  - llama32_1b_inference.prepare_runtime  (runtime setup)
  - llama32_1b_inference.run_npu_prefill  (prefill + KV cache extract + LM head)
  - llama32_1b_inference.run_npu_decode_step (one decode step + LM head)
  - llama32_1b_prefill.compile_all_kernels / decode.compile_decode_kernels

The runner holds the stateful pieces (kernel caches + KV cache) across calls;
the actual NPU compute path is identical to what `make run` exercises. Any
change to the production functions is automatically picked up by `make verify`.

Two modes:
  - lite_mode=True  (used by `make verify`): prefill returns logits + chosen
    token only; layer_intermediates is left empty.
  - lite_mode=False (used by `make diagnosis`): also collects per-layer
    ffn_out + the post-final-norm hidden state for the L15 probe. The
    layer-intermediate collection runs OUTSIDE the production path — it
    re-invokes run_transformer_block layer-by-layer with the same inputs,
    capturing the dict each block returns. This is a diagnosis-only side
    channel; verify never touches it.
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from llama_kernel_builder.cache import KernelCache
from llama32_1b_prefill import (
    compile_all_kernels as compile_prefill_kernels,
    run_transformer_block as run_prefill_block,
)
from llama32_1b_decode import compile_decode_kernels
from llama32_1b_inference import (
    prepare_runtime,
    run_npu_prefill,
    run_npu_decode_step,
)
from llama32_1b_weights import generate_rope_lut
from llama32_1b_cpu_helpers import rms_norm

from runners._records import PrefillRecord, DecodeStepRecord


class NpuRunner:
    name = "npu"

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
        # tokenizer is needed only to give run_npu_prefill an EOS-token-id
        # for padding the (raw) prompt to max_seq. Verify orchestrator passes
        # the same tokenizer it uses to encode prompts, so pad-token ID
        # matches the prompt's tokenization.
        self._tokenizer = tokenizer

        self.rope_lut_bf16 = generate_rope_lut(config=config, seq_len=max_seq).astype(
            bfloat16
        )

        # Compile prefill + decode kernels (same ones production compiles).
        self.prefill_cache = KernelCache(verbose=False)
        compile_prefill_kernels(
            self.prefill_cache,
            config,
            seq_len=max_seq,
            cpu_attn=self.cpu_attn,
        )
        self.decode_cache = KernelCache(verbose=False)
        compile_decode_kernels(self.decode_cache, config)

        # Production prepare_runtime: weight pre-transpose, per-layer index
        # tagging, BO preloading.
        prepare_runtime(
            self.prefill_cache,
            self.decode_cache,
            weights,
            config,
            max_seq,
            self.rope_lut_bf16,
        )

        # KV cache state lives across decode_step calls within one prefill.
        # prefill() repopulates this from run_npu_prefill's return.
        self.k_cache = None
        self.v_cache = None

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        # Production-side run_once pre-pads the prompt to the kernel's
        # compiled seq_len (= self.max_seq) with eos_token_id before calling
        # run_npu_prefill. Mirror that here so the verify path hits exactly
        # the same code with exactly the same shape.
        eos = self._tokenizer.eos_token_id
        if len(prompt_tokens) < self.max_seq:
            padded = list(prompt_tokens) + [eos] * (self.max_seq - len(prompt_tokens))
        else:
            padded = list(prompt_tokens)[: self.max_seq]
        # Production path — exact same code make run uses.
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
        # Persist KV cache for subsequent decode_step calls in this run.
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

        # ---- Diagnosis-only side channel: re-run the prefill layer loop
        # to capture per-layer ffn_out + the post-final-norm hidden state.
        # This is duplicate compute (~3-5 s extra) but only happens in
        # diagnosis mode, which is single-prompt by design.
        cfg = self.config
        if len(prompt_tokens) < self.max_seq:
            pad = np.zeros(self.max_seq - len(prompt_tokens), dtype=prompt_tokens.dtype)
            padded = np.concatenate([prompt_tokens, pad])
        else:
            padded = prompt_tokens[: self.max_seq]
        embed = self.weights.embed_table[padded].astype(np.float32)
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

        # Post-final-norm hidden — the value the LM-head GEMV sees.
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
