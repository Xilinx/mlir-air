# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the int4-AWQ Llama-3.2-1B example.

Wraps the production `llama32_1b_int4_inference` driver into a Runner for
the shared verify framework. Mirrors `llama32_1b/verify_adapter.py` but
loads AWQ weights (`load_weights_awq`) and uses the int4 decode kernels.

HF reference is built from the AMD AWQ checkpoint's config alone (no
weight download) with AWQ-dequantized weights patched in. That isolates
the verify gate to NPU drift only — both sides see exactly the same bf16
tensor values for every Linear weight — and keeps the verify path
ungated (no HF_TOKEN needed). No autoawq dep — we use our own
`awq_repacker.dequant_to_bf16`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_LLAMA_BF16 = _LLMS_DIR / "llama32_1b"
_VERIFY = _LLMS_DIR / "verify"
for _p in (str(_LLMS_DIR), str(_LLAMA_BF16), str(_VERIFY), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from shared.infra.cache import KernelCache  # noqa: E402
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

# Default AWQ checkpoint exposed by AMD; un-gated, no HF_TOKEN needed.
# `build_hf_model` reuses this checkpoint's config to construct the HF
# reference architecture, so the entire verify path is ungated.
_DEFAULT_AWQ_MODEL = "amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead"

MODEL_CHOICES = {
    "instruct": _DEFAULT_AWQ_MODEL,
}
DEFAULT_MODEL = "instruct"


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """HF reference model name; same as the AWQ checkpoint so the
    tokenizer and HF arch config are loaded from a single ungated repo."""
    return npu_model_name


def build_config():
    return LlamaConfig()


# Both build_runner() and build_hf_model() need the AWQ-loaded LlamaWeights.
# Cache once per model_name so we don't read safetensors twice in one verify
# run (the framework calls build_runner first and build_hf_model second).
_WEIGHTS_CACHE: dict = {}


def _get_or_load_weights(model_name: str, config):
    key = (model_name, id(config))
    if key not in _WEIGHTS_CACHE:
        _WEIGHTS_CACHE[key] = load_weights_awq(model_name, config=config)
    return _WEIGHTS_CACHE[key]


def build_hf_model(npu_model_name: str, hf_ref_model: str, config):
    """Construct an HF reference model with AWQ-dequantized weights.

    Load the architecture config from the AMD AWQ checkpoint (ungated),
    construct an empty LlamaForCausalLM from that config, then overwrite
    every Linear and layernorm with the AWQ-dequant bf16 from
    `load_weights_awq`. No remote weight download — only the config.json
    is fetched, which is already cached by the prefill driver's run.

    Tightens the verify gate from (quant_error + NPU_drift) down to
    (NPU_drift) since both sides see exactly the same bf16 tensor values.
    HF stores Linear weights as (out_features, in_features) but our
    dequant is (in_features, out_features), hence the transpose.
    """
    import torch
    from transformers import AutoConfig, LlamaForCausalLM

    weights = _get_or_load_weights(npu_model_name, config)
    print(f"[verify_adapter] building HF arch from {hf_ref_model} config...")
    hf_cfg = AutoConfig.from_pretrained(hf_ref_model)
    if hasattr(hf_cfg, "quantization_config"):
        delattr(hf_cfg, "quantization_config")
    hf_cfg.torch_dtype = torch.bfloat16
    model = LlamaForCausalLM(hf_cfg).to(torch.bfloat16)

    def _to_bf16_tensor(arr):
        # numpy bfloat16 -> torch bfloat16 via int16 bit-reinterpret. The
        # standard `torch.from_numpy(bf16_array)` errors out because numpy
        # has no native bfloat16 dtype.
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)

    n_layers = config.n_layers
    for li in range(n_layers):
        lw = weights.layers[li]
        hf_layer = model.model.layers[li]
        # Linear weights: HF (out, in); our dequant (in, out) => transpose.
        hf_layer.self_attn.q_proj.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(lw.wq).T)
        )
        hf_layer.self_attn.k_proj.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(lw.wk).T)
        )
        hf_layer.self_attn.v_proj.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(lw.wv).T)
        )
        hf_layer.self_attn.o_proj.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(lw.wo).T)
        )
        hf_layer.mlp.gate_proj.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(lw.w_gate).T)
        )
        hf_layer.mlp.up_proj.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(lw.w_up).T)
        )
        hf_layer.mlp.down_proj.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(lw.w_down).T)
        )
        # Layernorms (un-quantized in AWQ checkpoint).
        hf_layer.input_layernorm.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(lw.attn_norm))
        )
        hf_layer.post_attention_layernorm.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(lw.ffn_norm))
        )

    # Embedding, final norm, lm_head from AMD checkpoint.
    model.model.embed_tokens.weight.data = _to_bf16_tensor(
        np.ascontiguousarray(np.asarray(weights.embed_table))
    )
    model.model.norm.weight.data = _to_bf16_tensor(
        np.ascontiguousarray(np.asarray(weights.final_norm))
    )
    # Llama-3.2 ties lm_head to embed_tokens. The patched embed_tokens
    # already carries the AMD checkpoint values; lm_head's shared weight
    # is the same tensor, so no separate patch is needed for tied models.
    if not getattr(model.config, "tie_word_embeddings", False):
        model.lm_head.weight.data = _to_bf16_tensor(
            np.ascontiguousarray(np.asarray(weights.lm_head))
        )

    print(f"[verify_adapter] HF reference model patched with AWQ-dequant weights")
    return model


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
    weights = _get_or_load_weights(model_name, config)
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
        # Per-model cache dirs (absolute, CWD-independent) so verify runs of
        # different models never share the default shared/infra/
        # kernel_cache/ and pick up each other's stale ELFs.
        _cache_root = _THIS_DIR / "verify_kernel_cache"
        self.prefill_cache = KernelCache(str(_cache_root / "prefill"), verbose=False)
        with _multi_launch_dir(str(_LLAMA_BF16)):
            compile_prefill_kernels(
                self.prefill_cache, config, seq_len=max_seq, cpu_attn=self.cpu_attn
            )
        self.decode_cache = KernelCache(str(_cache_root / "decode"), verbose=False)
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
