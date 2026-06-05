# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end prefill + verification for int4-AWQ LLAMA-3.2-1B on NPU2.

Both backends consume the *same* AWQ HF checkpoint (e.g.
amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead):

* `--prefill-dtype=int4` (default): AWQ qweight/qzeros/scales are
  bit-shuffled (no re-quantization) into the packed BO layout the int4
  GEMM kernel consumes. Two int4 multi-launch stitchers run all 16
  transformer blocks. Lower compute throughput (kernel-bound), low
  weight memory.

* `--prefill-dtype=bf16`: same AWQ tensors are dequantized to dense bf16
  projections at load and routed through the bf16 prefill stitchers
  from `../llama32_1b/`. ~9x faster compute per layer, same AWQ-quality
  numerics, 2x weight memory. Recommended for prefill; decode work that
  truly benefits from int4 (DMA-bound) lives in a separate driver.

Reference path: the dequantized AWQ tensors are loaded into a vanilla
`LlamaForCausalLM` (no quantization_config) and HF runs prefill in bf16
on CPU. Top-k token-level inclusion check on the predicted position.

Usage:
    python3 llama32_1b_int4_prefill.py \\
        --model amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead \\
        --prefill-dtype bf16 --prompt "The capital of France is" \\
        --n-layers 16 --topk 10
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_PROG_EXAMPLES = str(_THIS_DIR.parent)
_LLAMA_BF16 = str(_THIS_DIR.parent / "llama32_1b")
for p in (_PROG_EXAMPLES, _LLAMA_BF16, str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llama32_1b_weights import LlamaConfig, generate_rope_lut
from llama32_1b_cpu_helpers import rms_norm, attention_reference
from llama_kernel_builder import cache as _cache_mod
from llama_kernel_builder.cache import KernelCache, Profiler
from llama_kernel_builder.external_kernels import (
    _compile_kernel,
    _PROJ_ROOT,
    compile_rope,
    compile_silu_and_mul,
    compile_attn_npu2,
)
from awq_pack import load_awq_weights

# ---------------------------------------------------------------------------
# air_project preparation for the int4 stitchers
# ---------------------------------------------------------------------------
#
# The bf16 prefill driver compiles `mv_int4_bf16.cc` with GEMV defines
# (DIM_M=8, DIM_K=2048) for the decode-side int4 GEMV kernels. The same
# source file produces the GEMM entry under a different set of defines
# (DIM_M=16, DIM_N=16, DIM_K_CHUNK=128, GS=128, BFP16 emulation) — the
# two are mutually exclusive (guarded by `#if DIM_M >= 16 && DIM_N >= 16`).
# The int4 stitchers (rms_gemms_rope_int4, o_ffn_int4) link against the
# GEMM variant. Monkey-patch `prepare_air_project` to wipe + repopulate
# air_project/ with the right `mv_int4_bf16.o` plus `rope.o` and
# `silu_and_mul.o` (the other externals the stitchers reference).


import shutil


def _compile_mv_int4_bf16_matmul(tile_m=16, tile_n=16, k_chunk=128, gs=128):
    """Compile mv_int4_bf16.cc for the int4 GEMM (matmul) entry.

    Produces `mv_int4_bf16_matmul.o` (config-tagged) and stages it as
    the canonical `mv_int4_bf16.o`. Decode-side `compile_mv_int4_bf16`
    produces `mv_int4_bf16_gemv.o` from the same .cc with DIM_M=8 and
    similarly stages it; the two .o variants coexist on disk so neither
    invalidates the other's cache.
    """
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "int4_awq" / "mv_int4_bf16.cc"
    _compile_kernel(
        src,
        "mv_int4_bf16_matmul.o",
        extra_flags=[
            f"-DDIM_M={tile_m}",
            f"-DDIM_N={tile_n}",
            f"-DDIM_K_CHUNK={k_chunk}",
            f"-DDIM_GS={gs}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
    )
    shutil.copy2("mv_int4_bf16_matmul.o", "mv_int4_bf16.o")


_INT4_TILE_N = 16  # overridden by --tile-n


def _prepare_air_project_int4():
    """Replacement for cache.prepare_air_project: wipe + repopulate
    air_project/ with `mv_int4_bf16.o` (GEMM flags), `rope.o`, and
    `silu_and_mul.o`."""
    air_proj = Path("air_project")
    if air_proj.exists():
        shutil.rmtree(air_proj)
    air_proj.mkdir(parents=True, exist_ok=True)

    _compile_mv_int4_bf16_matmul(tile_n=_INT4_TILE_N)
    compile_rope()
    compile_silu_and_mul()
    compile_attn_npu2()

    # attn.o is an alias the flash_attn link_with attribute uses.
    if Path("attn_npu2.o").exists() and not Path("attn.o").exists():
        shutil.copy2("attn_npu2.o", "attn.o")

    for obj_name in [
        "mv_int4_bf16.o",
        "rope.o",
        "silu_and_mul.o",
        "attn_npu2.o",
        "attn.o",
    ]:
        src = Path(obj_name)
        if src.exists():
            shutil.copy2(src, air_proj / obj_name)


# The monkey-patch is applied conditionally in main() — only the int4
# prefill path needs the GEMM-flavored mv_int4_bf16.o. The bf16 prefill
# path uses the stock cache.prepare_air_project (no int4 .o needed).


# stack_size=16384: int4 GEMM kernel needs it; default 1024 silently
# overflows (corrupts later sub-launches' compute).
# runtime_loop_tiling_sizes=[2,2]: int4 GEMM uses tile_n=16 (mmul constraint)
# vs bf16's tile_n=128, giving 8x more launch_n iters. Tile the runtime
# loop so the shim DMA BD chain doesn't exhaust the BD pool at seq>=512.
RMS_GEMMS_ROPE_INT4_BACKEND = {
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "rms_gemms_rope_int4",
    "stack_size": 16384,
    "runtime_loop_tiling_sizes": [2, 2],
}
O_FFN_INT4_BACKEND = {
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "o_ffn_int4",
    "stack_size": 16384,
    "runtime_loop_tiling_sizes": [2, 2],
}
# Attention runs on the same bf16 q/k/v regardless of upstream quant —
# the bf16 flash_attn ELF works unchanged on int4-produced tensors.
FLASH_ATTN_BACKEND = {
    "omit_while_true_loop": False,
    "omit_pingpong": "all",
    "runtime_loop_tiling_sizes": [1, 1],
    "output_format": "elf",
    "instance_name": "attention_bf16",
}


# ---------------------------------------------------------------------------
# Per-layer NPU int4 prefill
# ---------------------------------------------------------------------------


def _run_layer_int4(
    x_bf16,
    layer,
    layer_packed,
    rope_lut_bf16,
    config,
    cache,
    layer_idx,
    return_intermediates=False,
    cpu_attn=False,
):
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    rope_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    rope_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    # ---- rms_gemms_rope_int4 (13 args)
    rms_args = [
        np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim),
        np.asarray(layer.attn_norm, dtype=bfloat16).reshape(emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        layer_packed["wq"],
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        layer_packed["wk"],
        np.zeros((seq_len, kv_dim), dtype=bfloat16),
        layer_packed["wv"],
        np.zeros((seq_len, kv_dim), dtype=bfloat16),
        rope_q,
        rope_k,
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        np.zeros((seq_len, kv_dim), dtype=bfloat16),
    ]
    output_idx = [2, 4, 6, 8, 11, 12] if return_intermediates else [8, 11, 12]
    results = cache.load_and_run(
        "rms_gemms_rope_int4",
        {"verbose": cache.verbose, **RMS_GEMMS_ROPE_INT4_BACKEND},
        *rms_args,
        output_indices=output_idx,
        static_input_indices={1, 3, 5, 7, 9, 10},
        intermediate_indices={2, 4, 6, 8, 11, 12},
        bo_key=f"rms_int4_L{layer_idx}",
    )
    if return_intermediates:
        normed = results[2].reshape(seq_len, emb_dim)
        q = results[4].reshape(seq_len, emb_dim)
        k = results[6].reshape(seq_len, kv_dim)
        v = results[8].reshape(seq_len, kv_dim)
        q_roped = results[11].reshape(seq_len, n_heads * head_dim)
        k_roped = results[12].reshape(seq_len, n_kv_heads * head_dim)
    else:
        v = results[8].reshape(seq_len, kv_dim)
        q_roped = results[11].reshape(seq_len, n_heads * head_dim)
        k_roped = results[12].reshape(seq_len, n_kv_heads * head_dim)

    # ---- GQA attention. bf16 flash_attn ELF is q/k/v-dtype-agnostic so the
    # same ELF runs regardless of int4 quant upstream.
    if cpu_attn:
        with cache.profiler.time_cpu("prefill_cpu_attention"):
            attn_out = attention_reference(
                q_roped.astype(np.float32),
                k_roped.astype(np.float32),
                v.astype(np.float32),
                n_heads,
                n_kv_heads,
            ).astype(bfloat16)
    else:
        attn_buf = np.zeros((seq_len, n_heads * head_dim), dtype=bfloat16)
        res = cache.load_and_run(
            "flash_attn",
            FLASH_ATTN_BACKEND,
            np.ascontiguousarray(q_roped),
            np.ascontiguousarray(k_roped),
            np.ascontiguousarray(v),
            attn_buf,
            output_indices=[3],
            bo_key=f"flash_attn_L{layer_idx}",
        )
        attn_out = res[3].reshape(seq_len, n_heads * head_dim)

    # ---- o_ffn_int4 (15 args)
    n_total = seq_len * emb_dim
    offn_args = [
        np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim),
        layer_packed["wo"],
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        np.asarray(layer.ffn_norm, dtype=bfloat16).reshape(emb_dim),
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        layer_packed["w_gate"],
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),
        layer_packed["w_up"],
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),
        np.zeros((seq_len, hidden_dim), dtype=bfloat16),
        layer_packed["w_down"],
        np.zeros((seq_len, emb_dim), dtype=bfloat16),
        np.zeros(n_total, dtype=bfloat16),
    ]
    if return_intermediates:
        # Quick stop after the first stitcher so callers can probe q/k/v.
        return None, {
            "normed": normed,
            "q": q,
            "k": k,
            "v": v,
            "q_roped": q_roped,
            "k_roped": k_roped,
            "attn_out": attn_out,
        }

    results = cache.load_and_run(
        "o_ffn_int4",
        {"verbose": cache.verbose, **O_FFN_INT4_BACKEND},
        *offn_args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
        bo_key=f"offn_int4_L{layer_idx}",
    )
    return results[14].reshape(seq_len, emb_dim)


# ---------------------------------------------------------------------------
# CPU reference: same flow as _run_layer_int4 but with numpy ops on the
# dequantized weights. If this matches NPU we've validated the *stitching
# logic* and the divergence vs HF is something else (e.g. attention impl,
# RoPE convention, prediction position handling); if it differs from NPU
# the bug is in the NPU kernels themselves.
# ---------------------------------------------------------------------------


def _cpu_rmsnorm(x, weight, eps=1e-5):
    x_f32 = x.astype(np.float32)
    w = weight.astype(np.float32)
    rms = np.sqrt((x_f32 * x_f32).mean(axis=-1, keepdims=True) + eps)
    return (x_f32 / rms * w).astype(bfloat16)


def _cpu_rope(x_2d, lut_per_row):
    """Half-split RoPE matching rope_halfsplit.cc and HF transformers:
    out[i]      = x[i]*cos[i] - x[i+half]*sin[i]
    out[i+half] = x[i+half]*cos[i] + x[i]*sin[i]
    lut layout: concatenated [cos_0..cos_{half-1}, sin_0..sin_{half-1}]."""
    x = x_2d.astype(np.float32)
    lut = lut_per_row.astype(np.float32)
    rows, dim = x.shape
    half = dim // 2
    cos = lut[:, :half]
    sin = lut[:, half:]
    x1 = x[:, :half]
    x2 = x[:, half:]
    out = np.empty_like(x)
    out[:, :half] = x1 * cos - x2 * sin
    out[:, half:] = x2 * cos + x1 * sin
    return out.astype(bfloat16)


def _run_layer_cpu_int4(x_bf16, layer, rope_lut_bf16, config):
    seq_len = x_bf16.shape[0]
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    kv_dim = n_kv_heads * head_dim

    normed = _cpu_rmsnorm(x_bf16, layer.attn_norm)
    q = (normed.astype(np.float32) @ layer.wq.astype(np.float32)).astype(bfloat16)
    k = (normed.astype(np.float32) @ layer.wk.astype(np.float32)).astype(bfloat16)
    v = (normed.astype(np.float32) @ layer.wv.astype(np.float32)).astype(bfloat16)

    # Per-head RoPE: each head's [head_dim] slice gets the same per-row LUT.
    q_per_head = q.reshape(seq_len, n_heads, head_dim)
    k_per_head = k.reshape(seq_len, n_kv_heads, head_dim)
    q_rope = np.empty_like(q_per_head)
    k_rope = np.empty_like(k_per_head)
    for h in range(n_heads):
        q_rope[:, h, :] = _cpu_rope(q_per_head[:, h, :], rope_lut_bf16[:seq_len])
    for h in range(n_kv_heads):
        k_rope[:, h, :] = _cpu_rope(k_per_head[:, h, :], rope_lut_bf16[:seq_len])
    q_roped = q_rope.reshape(seq_len, n_heads * head_dim)
    k_roped = k_rope.reshape(seq_len, n_kv_heads * head_dim)

    attn_out = attention_reference(
        q_roped.astype(np.float32),
        k_roped.astype(np.float32),
        v.astype(np.float32),
        n_heads,
        n_kv_heads,
    ).astype(bfloat16)

    proj = (attn_out.astype(np.float32) @ layer.wo.astype(np.float32)).astype(bfloat16)
    res1 = (proj.astype(np.float32) + x_bf16.astype(np.float32)).astype(bfloat16)
    normed2 = _cpu_rmsnorm(res1, layer.ffn_norm)
    gate = normed2.astype(np.float32) @ layer.w_gate.astype(np.float32)
    up = normed2.astype(np.float32) @ layer.w_up.astype(np.float32)
    swiglu = (gate / (1.0 + np.exp(-gate)) * up).astype(bfloat16)
    down = (swiglu.astype(np.float32) @ layer.w_down.astype(np.float32)).astype(
        bfloat16
    )
    out = (down.astype(np.float32) + res1.astype(np.float32)).astype(bfloat16)
    return out


# ---------------------------------------------------------------------------
# HF reference — vanilla bf16 LlamaForCausalLM loaded with our dequant'd weights
# ---------------------------------------------------------------------------


def _build_hf_model(model_path, weights_bf16, n_layers=None):
    """Build a vanilla (non-quantized) Llama-3.2-1B in bf16, populate it
    with the dequantized AWQ weights, and return (model, tokenizer).
    Per-prompt forward passes can then call `_hf_forward(model, tok, ...)`.
    """
    import torch
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        LlamaForCausalLM,
    )

    print("  building HF vanilla bf16 model from config...")
    hf_cfg = AutoConfig.from_pretrained(model_path)
    if hasattr(hf_cfg, "quantization_config"):
        delattr(hf_cfg, "quantization_config")
    hf_cfg.torch_dtype = torch.bfloat16
    model = LlamaForCausalLM(hf_cfg).to(torch.bfloat16)
    model.eval()

    def _to_torch_bf16_T(arr):
        a = np.ascontiguousarray(arr.view(np.int16).T)
        return torch.from_numpy(a).view(torch.bfloat16)

    def _to_torch_bf16(arr):
        a = np.ascontiguousarray(arr.view(np.int16))
        return torch.from_numpy(a).view(torch.bfloat16)

    sd = {
        "model.embed_tokens.weight": _to_torch_bf16(weights_bf16.embed_table),
        "model.norm.weight": _to_torch_bf16(weights_bf16.final_norm),
        "lm_head.weight": _to_torch_bf16(weights_bf16.embed_table),  # tied
    }
    for li, lw in enumerate(weights_bf16.layers):
        b = f"model.layers.{li}"
        sd[f"{b}.input_layernorm.weight"] = _to_torch_bf16(lw.attn_norm)
        sd[f"{b}.post_attention_layernorm.weight"] = _to_torch_bf16(lw.ffn_norm)
        sd[f"{b}.self_attn.q_proj.weight"] = _to_torch_bf16_T(lw.wq)
        sd[f"{b}.self_attn.k_proj.weight"] = _to_torch_bf16_T(lw.wk)
        sd[f"{b}.self_attn.v_proj.weight"] = _to_torch_bf16_T(lw.wv)
        sd[f"{b}.self_attn.o_proj.weight"] = _to_torch_bf16_T(lw.wo)
        sd[f"{b}.mlp.gate_proj.weight"] = _to_torch_bf16_T(lw.w_gate)
        sd[f"{b}.mlp.up_proj.weight"] = _to_torch_bf16_T(lw.w_up)
        sd[f"{b}.mlp.down_proj.weight"] = _to_torch_bf16_T(lw.w_down)

    print("  loading state_dict...")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    if missing:
        nontrivial = [m for m in missing if m != "lm_head.weight"]
        if nontrivial:
            print(f"  WARNING: {len(nontrivial)} missing keys (e.g. {nontrivial[:3]})")

    if n_layers is not None and n_layers < len(model.model.layers):
        import torch.nn as nn

        model.model.layers = nn.ModuleList(model.model.layers[:n_layers])

    tok = AutoTokenizer.from_pretrained(model_path)
    return model, tok


def _hf_forward(model, tok, prompt, want_hidden_states=False):
    """Run one HF forward pass. Returns (logits[vocab] f32, token_ids list,
    hidden_states or None). `hidden_states[i]` is the bf16-as-f32 array of
    shape (seq_len, hidden_dim) after layer i (per HF v5.3 convention,
    [0] is post-embedding and [n_layers] is post-final-RMSNorm)."""
    import torch

    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=want_hidden_states)
    logits = out.logits[0, -1].to(torch.float32).numpy()
    hs = None
    if want_hidden_states:
        hs = [t[0].to(torch.float32).numpy() for t in out.hidden_states]
    return logits, ids[0].tolist(), hs


def _hf_reference_logits(model_path, weights_bf16, prompt, config, n_layers=None):
    """Convenience wrapper preserving the older single-shot API: builds the
    HF model + tokenizer once and runs one forward. Returns
    (logits[vocab] f32, token_ids list, tokenizer)."""
    model, tok = _build_hf_model(model_path, weights_bf16, n_layers=n_layers)
    logits, ids, _ = _hf_forward(model, tok, prompt)
    print(f"  HF prefill ({len(ids)} tokens, {len(model.model.layers)} layers)...")
    return logits, ids, tok


# ---------------------------------------------------------------------------
# Diagnosis lens (per-layer ffn_out NPU vs HF) + prompts-file loader
# ---------------------------------------------------------------------------


def _load_prompts_file(path):
    """Read a prompts file (one prompt per line, '#' comments + blanks
    ignored). Same format as ../verify/prompts/{base,instruct}.txt."""
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            out.append(ln)
    return out


def _layer_cosine(a, b):
    """Cosine sim + MAE between two arrays (any matching shape)."""
    af = np.asarray(a, dtype=np.float32).flatten()
    bf = np.asarray(b, dtype=np.float32).flatten()
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    cos = float(np.dot(af, bf) / (na * nb + 1e-30))
    mae = float(np.abs(af - bf).mean())
    return cos, mae, float(na), float(nb)


def _run_diagnosis(
    args,
    weights_bf16,
    layers_packed,
    config,
    rope_lut_bf16,
    cache,
    hf_model,
    tok,
    prompt,
):
    """Per-layer ffn_out diff (NPU vs HF bf16). Mirrors the bf16 sibling's
    `make diagnosis` behavior. Informational only — no PASS/FAIL gate."""
    seq_len = args.seq_len
    emb_dim = config.emb_dim

    print(f"[diagnosis] prompt: {prompt!r}")
    print("[diagnosis] HF prefill (with output_hidden_states=True)...")
    _, token_ids, hf_hs = _hf_forward(hf_model, tok, prompt, want_hidden_states=True)
    prompt_len = len(token_ids)
    if prompt_len > seq_len:
        raise SystemExit(f"prompt_len={prompt_len} > seq_len={seq_len}")
    pred_pos = prompt_len - 1

    embed = weights_bf16.embed_table
    x_bf16 = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    for i, tid in enumerate(token_ids):
        x_bf16[i] = embed[tid]

    npu_layer_outs = []
    print(f"[diagnosis] NPU prefill (capturing per-layer ffn_out)...")
    if args.prefill_dtype == "bf16":
        sys.path.insert(0, _LLAMA_BF16)
        from llama32_1b_prefill import (
            run_transformer_block,
            preload_prefill_weights,
        )

        preload_prefill_weights(weights_bf16, config, cache, seq_len, rope_lut_bf16)

    for li in range(args.n_layers):
        if args.prefill_dtype == "int4":
            x_bf16 = _run_layer_int4(
                x_bf16,
                weights_bf16.layers[li],
                layers_packed[li],
                rope_lut_bf16,
                config,
                cache,
                li,
                cpu_attn=args.cpu_attn,
            )
        else:
            x_bf16, _ = run_transformer_block(
                x_bf16,
                weights_bf16.layers[li],
                rope_lut_bf16,
                config,
                cache,
                layer_idx=li,
                cpu_attn=args.cpu_attn,
                verbose=False,
            )
        # x_bf16 has shape (seq_len, emb_dim). Take only the prompt region
        # for a fair compare against HF (which only sees prompt_len tokens).
        npu_layer_outs.append(np.asarray(x_bf16[:prompt_len], dtype=bfloat16))

    # HF v5.3 convention: hidden_states[0] = post-embedding, hidden_states[i+1]
    # = block output for layers 0..n-2, hidden_states[n_layers] = post-final-
    # RMSNorm. So for the LAST layer we apply final_norm to NPU output too,
    # else the cosine collapses (post-norm vs pre-norm have wildly different
    # magnitudes). Mirrors the bf16 sibling's _run_diagnosis behavior.
    print("\n[diagnosis] per-layer ffn_out cosine (NPU vs HF bf16):")
    print(f"  {'layer':>5} {'cos':>9} {'MAE':>9} {'||NPU||':>10} {'||HF||':>10}  note")
    print(f"  {'-'*5} {'-'*9} {'-'*9} {'-'*10} {'-'*10}  {'-'*20}")
    for li in range(args.n_layers):
        npu = npu_layer_outs[li]
        if li == args.n_layers - 1:
            # Apply final RMSNorm to NPU and compare to HF post-norm.
            npu_f32 = np.asarray(npu, dtype=np.float32)
            npu_normed = rms_norm(npu_f32, weights_bf16.final_norm)
            hf = hf_hs[args.n_layers][:prompt_len]
            cos, mae, na, nb = _layer_cosine(npu_normed, hf)
            note = "(post-final-norm)"
        else:
            hf = hf_hs[li + 1][:prompt_len]
            cos, mae, na, nb = _layer_cosine(npu, hf)
            note = ""
        print(f"  {li:>5d} {cos:>+9.4f} {mae:>9.4f} {na:>10.2f} {nb:>10.2f}  {note}")

    print("\n[diagnosis] done. (informational — no PASS/FAIL gate)")


def _verify_one_prompt(
    args,
    weights_bf16,
    layers_packed,
    config,
    rope_lut_bf16,
    cache,
    hf_model,
    tok,
    prompt,
):
    """Single-prompt prefill + top-K vs HF. Returns dict with overlap,
    argmax_match for caller aggregation."""
    seq_len = args.seq_len
    emb_dim = config.emb_dim

    hf_logits, token_ids, _ = _hf_forward(hf_model, tok, prompt)
    prompt_len = len(token_ids)
    if prompt_len > seq_len:
        raise SystemExit(f"prompt_len={prompt_len} > seq_len={seq_len}")
    pred_pos = prompt_len - 1
    print(f"  prompt: {prompt!r} (len={prompt_len})")

    hf_top = np.argsort(-hf_logits)[: args.topk]

    embed = weights_bf16.embed_table
    x_bf16 = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    for i, tid in enumerate(token_ids):
        x_bf16[i] = embed[tid]

    if args.prefill_dtype == "bf16":
        sys.path.insert(0, _LLAMA_BF16)
        from llama32_1b_prefill import (
            run_transformer_block,
            preload_prefill_weights,
        )

        preload_prefill_weights(weights_bf16, config, cache, seq_len, rope_lut_bf16)

    t0 = time.time()
    for li in range(args.n_layers):
        t_layer = cache.profiler.start_layer()
        if args.prefill_dtype == "int4":
            x_bf16 = _run_layer_int4(
                x_bf16,
                weights_bf16.layers[li],
                layers_packed[li],
                rope_lut_bf16,
                config,
                cache,
                li,
                cpu_attn=args.cpu_attn,
            )
        else:
            x_bf16, _ = run_transformer_block(
                x_bf16,
                weights_bf16.layers[li],
                rope_lut_bf16,
                config,
                cache,
                layer_idx=li,
                cpu_attn=args.cpu_attn,
                verbose=args.verbose,
            )
        cache.profiler.end_layer(li, t_layer)

    last_row = np.asarray(x_bf16, dtype=np.float32)[pred_pos]
    normed = rms_norm(last_row[np.newaxis, :], weights_bf16.final_norm).flatten()
    npu_logits = (
        normed.astype(np.float32) @ weights_bf16.lm_head.astype(np.float32).T
    ).astype(np.float32)
    npu_top = np.argsort(-npu_logits)[: args.topk]
    overlap = len(set(hf_top.tolist()) & set(npu_top.tolist()))
    argmax_match = int(hf_top[0]) == int(npu_top[0])

    print(f"  HF argmax   : id={int(hf_top[0]):>6d} {tok.decode([int(hf_top[0])])!r}")
    print(
        f"  NPU argmax  : id={int(npu_top[0]):>6d} {tok.decode([int(npu_top[0])])!r}  "
        f"match={argmax_match}"
    )
    print(
        f"  Top-{args.topk} overlap: {overlap}/{args.topk}  "
        f"({time.time()-t0:.1f}s NPU prefill)"
    )
    return {
        "prompt": prompt,
        "overlap": overlap,
        "argmax_match": argmax_match,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead",
        help="HF model ID (or local dir) of an AWQ uint4 g128 Llama checkpoint.",
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--gs", type=int, default=128)
    ap.add_argument("--cache-dir", default=str(_THIS_DIR / "verify_kernel_cache"))
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="Build + compile the two int4 ELFs and exit.",
    )
    ap.add_argument(
        "--run-only", action="store_true", help="Use cached ELFs; skip compile."
    )
    ap.add_argument(
        "--skip-npu",
        action="store_true",
        help="Skip NPU prefill; just print HF reference top-K.",
    )
    ap.add_argument(
        "--probe-stitcher1",
        action="store_true",
        help="Run only rms_gemms_rope_int4 on layer 0 and diff "
        "q/k/v against numpy reference on dequantized weights.",
    )
    ap.add_argument(
        "--cpu-int4",
        action="store_true",
        help="Also run a CPU-numpy prefill on the dequantized "
        "AWQ weights and print its top-K. Useful for isolating "
        "whether divergence vs HF is in our stitching logic vs "
        "in the NPU kernels themselves.",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-layer / per-kernel timing instrumentation.",
    )
    ap.add_argument(
        "--tile-n",
        type=int,
        default=16,
        help="int4 GEMM tile_n (16/32/64). Larger reduces launch_n "
        "iter count; AIE2P caps the kernel at 64.",
    )
    ap.add_argument(
        "--cpu-attn",
        action="store_true",
        help="Use numpy GQA attention instead of NPU flash_attn. "
        "Default is NPU (the bf16 flash_attn ELF is "
        "q/k/v-dtype-agnostic).",
    )
    ap.add_argument(
        "--prefill-dtype",
        choices=["int4", "bf16"],
        default="int4",
        help="Which prefill GEMM ELFs to run. 'int4' uses the "
        "int4 stitchers (low-memory). 'bf16' dequants the AWQ "
        "weights once at load and runs the bf16 prefill stitchers "
        "(3-6x faster compute; same AWQ-quality output). Decode "
        "is unaffected — that path still benefits from int4.",
    )
    ap.add_argument(
        "--min-overlap",
        type=int,
        default=0,
        help="If >0, print '[verify] PASS' iff top-K overlap vs "
        "HF reaches this threshold (and argmax matches); else "
        "'[verify] FAIL'. Used by run_npu2_verify.lit.",
    )
    ap.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to a prompts file (one prompt per line, '#' "
        "comments allowed). When set, runs each prompt and "
        "aggregates pass/fail. Used by `make verify-full`.",
    )
    ap.add_argument(
        "--diagnosis",
        action="store_true",
        help="Diagnosis lens: collect per-layer ffn_out from "
        "NPU prefill and diff against HF bf16 reference's "
        "per-layer hidden_states. Informational, no PASS/FAIL "
        "gate. Used by `make diagnosis`.",
    )
    args = ap.parse_args()
    global _INT4_TILE_N
    _INT4_TILE_N = args.tile_n

    # int4 prefill needs the GEMM-flavored mv_int4_bf16.o under air_project/;
    # bf16 prefill uses the stock external-kernel set (rope/silu/attn).
    if args.prefill_dtype == "int4":
        _cache_mod.prepare_air_project = _prepare_air_project_int4

    config = LlamaConfig()
    seq_len = args.seq_len
    emb_dim = config.emb_dim
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    kv_dim = n_kv_heads * head_dim

    cache = KernelCache(
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        profiler=Profiler(enabled=args.profile),
    )

    # ---- Load AWQ checkpoint once (both reference + NPU consume this).
    print(f"Loading AWQ checkpoint: {args.model}")
    t0 = time.time()
    weights_bf16, layers_packed = load_awq_weights(
        args.model,
        config=config,
        gs=args.gs,
        n_tile=args.tile_n,
        k_chunk=128,
        seq_len=seq_len,
    )
    print(f"  loaded + dequant + packed in {time.time()-t0:.1f}s")

    # Skip-if-cached: avoid rebuilding ELFs that already exist on disk.
    # Compile times at seq=2048 are minutes per kernel, so this matters.
    # kernel_sym must match the actual ELF symbol (= XRTBackend's
    # `instance_name` arg); it is NOT always equal to the cache key (e.g.
    # flash_attn ships as `attention_bf16`).
    def _need(name, kernel_sym=None):
        if kernel_sym is None:
            kernel_sym = name
        elf = Path(args.cache_dir) / f"{name}.elf"
        if elf.exists():
            print(f"  using cached {name}.elf ({elf.stat().st_size//1024} KB)")
            from air.backend.xrt import XRTCompileArtifact

            cache.artifacts[name] = XRTCompileArtifact(
                str(elf), f"main:{kernel_sym}", None
            )
            return False
        return True

    # ---- Compile (or load) the prefill stitcher ELFs.
    if not args.skip_npu and not args.run_only:
        # awq_pack inserts the bf16 sibling dir (which also has a
        # `multi_launch_builder` package) at sys.path[0] for shared
        # scaffolding cross-link — re-prepend our dir so our int4 stitchers
        # win the namespace race here.
        if sys.path[0] != str(_THIS_DIR):
            sys.path.insert(0, str(_THIS_DIR))

        if args.prefill_dtype == "int4":
            from multi_launch_builder.rms_gemms_rope_int4_multi import (
                build_rms_gemms_rope_int4_module,
            )
            from multi_launch_builder.o_ffn_int4_multi import (
                build_o_ffn_int4_module,
            )

            if _need("rms_gemms_rope_int4"):
                print("\nCompiling rms_gemms_rope_int4...")
                cache.compile_and_cache(
                    "rms_gemms_rope_int4",
                    build_rms_gemms_rope_int4_module(
                        seq_len=seq_len,
                        emb_dim=emb_dim,
                        kv_dim=kv_dim,
                        n_heads=config.n_heads,
                        n_kv_heads=n_kv_heads,
                        head_dim=head_dim,
                        gs=args.gs,
                        tile_n=args.tile_n,
                    ),
                    {"verbose": args.verbose, **RMS_GEMMS_ROPE_INT4_BACKEND},
                )
            if _need("o_ffn_int4"):
                print("Compiling o_ffn_int4...")
                cache.compile_and_cache(
                    "o_ffn_int4",
                    build_o_ffn_int4_module(
                        seq_len=seq_len,
                        emb_dim=emb_dim,
                        hidden_dim=hidden_dim,
                        gs=args.gs,
                        tile_n=args.tile_n,
                    ),
                    {"verbose": args.verbose, **O_FFN_INT4_BACKEND},
                )
        else:
            # bf16 prefill: dequantized AWQ weights through the bf16 stitchers
            # (~3-6x faster compute per layer; see bisection findings).
            sys.path.insert(0, _LLAMA_BF16)
            from llama_kernel_builder.backend_presets import (
                RMS_GEMMS_ROPE_BACKEND,
                O_FFN_BACKEND,
            )

            if _need("rms_gemms_rope"):
                print("\nCompiling rms_gemms_rope (bf16)...")
                from multi_launch_builder.rms_gemms_rope_multi import (
                    build_rms_gemms_rope_module,
                )

                cache.compile_and_cache(
                    "rms_gemms_rope",
                    build_rms_gemms_rope_module(
                        seq_len,
                        emb_dim,
                        kv_dim,
                        config.n_heads,
                        n_kv_heads,
                        head_dim,
                    ),
                    {"verbose": args.verbose, **RMS_GEMMS_ROPE_BACKEND},
                )
            if _need("o_ffn"):
                print("Compiling o_ffn (bf16)...")
                from multi_launch_builder.o_ffn_multi import build_o_ffn_module

                cache.compile_and_cache(
                    "o_ffn",
                    build_o_ffn_module(seq_len, emb_dim, hidden_dim),
                    {"verbose": args.verbose, **O_FFN_BACKEND},
                )

        if not args.cpu_attn and _need("flash_attn", kernel_sym="attention_bf16"):
            print("Compiling flash_attn (bf16 ELF)...")
            sys.path.insert(0, str(_PROJ_ROOT))
            from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
                build_module as build_attn,
            )

            lkp = head_dim
            enable_shared = lkp == head_dim
            cache.compile_and_cache(
                "flash_attn",
                build_attn(
                    lk=seq_len,
                    lkp=lkp,
                    lq=seq_len,
                    lqp=256,
                    dk=head_dim,
                    dv=head_dim,
                    num_q_tiles=4,
                    num_cascade_stages=4,
                    num_heads=config.n_heads,
                    num_kv_heads=n_kv_heads,
                    causal=True,
                ),
                {
                    "verbose": args.verbose,
                    "omit_while_true_loop": not enable_shared,
                    **{
                        k: v
                        for k, v in FLASH_ATTN_BACKEND.items()
                        if k != "omit_while_true_loop"
                    },
                },
            )
        cache._save_manifest()
    elif not args.skip_npu and args.run_only:
        if not cache.load_manifest():
            raise SystemExit(f"--run-only but no manifest at {args.cache_dir}")

    if args.compile_only:
        # Marker matched by run_npu2_compile.lit.
        print("Compilation passed.")
        return

    # ---- Multi-prompt (`make verify-full`) and diagnosis (`make diagnosis`)
    # paths build the HF model once and dispatch to the per-prompt helpers.
    if args.prompts_file or args.diagnosis:
        rope_lut_bf16 = generate_rope_lut(config, seq_len=seq_len)
        print(f"\nBuilding HF bf16 reference model (one-time)...")
        hf_model, tok = _build_hf_model(
            args.model,
            weights_bf16,
            n_layers=args.n_layers,
        )

        if args.diagnosis:
            _run_diagnosis(
                args,
                weights_bf16,
                layers_packed,
                config,
                rope_lut_bf16,
                cache,
                hf_model,
                tok,
                args.prompt,
            )
            return

        prompts = _load_prompts_file(args.prompts_file)
        if not prompts:
            raise SystemExit(
                f"--prompts-file {args.prompts_file} has no usable prompts"
            )
        print(
            f"\n=== verify-full: {len(prompts)} prompt(s), "
            f"--prefill-dtype={args.prefill_dtype}, --n-layers={args.n_layers} ==="
        )
        results = []
        for pi, prompt in enumerate(prompts):
            print(f"\n--- prompt {pi+1}/{len(prompts)} ---")
            results.append(
                _verify_one_prompt(
                    args,
                    weights_bf16,
                    layers_packed,
                    config,
                    rope_lut_bf16,
                    cache,
                    hf_model,
                    tok,
                    prompt,
                )
            )

        n = len(results)
        n_argmax = sum(r["argmax_match"] for r in results)
        avg_overlap = sum(r["overlap"] for r in results) / n
        print(f"\n=== verify-full summary ===")
        print(f"  prompts                : {n}")
        print(f"  argmax matches         : {n_argmax}/{n}")
        print(f"  avg top-{args.topk} overlap : {avg_overlap:.2f}/{args.topk}")
        if args.min_overlap > 0:
            passes = sum(
                1
                for r in results
                if r["overlap"] >= args.min_overlap and r["argmax_match"]
            )
            verdict = "PASS" if passes == n else "FAIL"
            print(
                f"\n[verify] {verdict} "
                f"({passes}/{n} prompts met overlap>={args.min_overlap} "
                f"AND argmax match)"
            )
        return

    # ---- Single-prompt path (legacy `make verify`, `make run`, --probe-stitcher1).
    print(f"\nRunning HF bf16 reference (same dequant'd weights)...")
    hf_logits, token_ids, tok = _hf_reference_logits(
        args.model,
        weights_bf16,
        args.prompt,
        config,
        n_layers=args.n_layers,
    )
    prompt_len = len(token_ids)
    pred_pos = prompt_len - 1
    print(f"  prompt_len = {prompt_len}  pred_pos = {pred_pos}")
    print(f"  prompt: {args.prompt!r}")
    if prompt_len > seq_len:
        raise SystemExit(f"prompt_len={prompt_len} > seq_len={seq_len}")

    hf_top = np.argsort(-hf_logits)[: args.topk]
    print(f"\nHF top-{args.topk}:")
    for r, tid in enumerate(hf_top):
        text = tok.decode([int(tid)])
        print(
            f"  #{r+1:2d}  id={int(tid):>6d}  logit={float(hf_logits[tid]):+9.3f}  {text!r}"
        )

    # ---- Optional CPU-int4 reference path (same dequant weights, numpy ops).
    if args.cpu_int4:
        print(
            f"\nRunning {args.n_layers}-layer CPU int4 prefill (numpy on "
            f"dequantized AWQ weights)..."
        )
        cpu_rope = generate_rope_lut(config, seq_len=seq_len)
        cpu_x = np.zeros((seq_len, emb_dim), dtype=bfloat16)
        for i, tid in enumerate(token_ids):
            cpu_x[i] = weights_bf16.embed_table[tid]
        for li in range(args.n_layers):
            cpu_x = _run_layer_cpu_int4(
                cpu_x,
                weights_bf16.layers[li],
                cpu_rope,
                config,
            )
        cpu_last = np.asarray(cpu_x, dtype=np.float32)[pred_pos]
        cpu_normed = rms_norm(
            cpu_last[np.newaxis, :], weights_bf16.final_norm
        ).flatten()
        cpu_logits = (
            cpu_normed.astype(np.float32) @ weights_bf16.lm_head.astype(np.float32).T
        ).astype(np.float32)
        cpu_top = np.argsort(-cpu_logits)[: args.topk]
        cpu_in_hf = set(hf_top.tolist())
        print(f"\nCPU int4 top-{args.topk}:")
        for r, tid in enumerate(cpu_top):
            text = tok.decode([int(tid)])
            mark = "*" if int(tid) in cpu_in_hf else " "
            print(
                f"  #{r+1:2d}{mark} id={int(tid):>6d}  logit={float(cpu_logits[tid]):+9.3f}  {text!r}"
            )
        cpu_overlap = len(set(cpu_top.tolist()) & cpu_in_hf)
        print(f"\nCPU int4 vs HF top-{args.topk} overlap: {cpu_overlap}/{args.topk}")
        cpu_union = sorted(set(cpu_top.tolist()) | set(hf_top.tolist()))
        a, b = hf_logits[cpu_union], cpu_logits[cpu_union]
        if a.std() > 0 and b.std() > 0:
            print(
                f"CPU int4 vs HF logit Pearson r: {float(np.corrcoef(a, b)[0,1]):.4f}"
            )

    if args.skip_npu:
        return

    # ---- Build initial x from embedding table, pad to seq_len with zeros.
    embed = weights_bf16.embed_table
    x_bf16 = np.zeros((seq_len, emb_dim), dtype=bfloat16)
    for i, tid in enumerate(token_ids):
        x_bf16[i] = embed[tid]

    rope_lut_bf16 = generate_rope_lut(config, seq_len=seq_len)

    if args.probe_stitcher1:
        print("\n=== Probing rms_gemms_rope_int4 on layer 0 ===")
        # OPTIONAL: zero the input to see if NPU produces zero output
        # (RMSNorm of zero = zero → GEMM of zero = zero).
        if os.environ.get("ZERO_INPUT") == "1":
            print("  (ZERO_INPUT=1: feeding all-zero x_in for sanity)")
            x_bf16 = np.zeros_like(x_bf16)
        if os.environ.get("ZERO_WV") == "1":
            print("  (ZERO_WV=1: zeroing wv packed BO to see if NPU output changes)")
            layers_packed[0]["wv"] = np.zeros_like(layers_packed[0]["wv"])
        _, npu_int = _run_layer_int4(
            x_bf16,
            weights_bf16.layers[0],
            layers_packed[0],
            rope_lut_bf16,
            config,
            cache,
            0,
            return_intermediates=True,
            cpu_attn=args.cpu_attn,
        )
        # CPU reference: same dequantized weights, same numpy ops.
        normed = _cpu_rmsnorm(x_bf16, weights_bf16.layers[0].attn_norm)
        cpu_q = normed.astype(np.float32) @ weights_bf16.layers[0].wq.astype(np.float32)
        cpu_k = normed.astype(np.float32) @ weights_bf16.layers[0].wk.astype(np.float32)
        cpu_v = (
            normed.astype(np.float32) @ weights_bf16.layers[0].wv.astype(np.float32)
        ).astype(bfloat16)
        # Half-split RoPE on CPU (match NPU kernel convention)
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        head_dim = config.head_dim
        lut_f32 = rope_lut_bf16[:seq_len].astype(np.float32)  # (seq, head_dim)
        half = head_dim // 2
        cos = lut_f32[:, :half]
        sin = lut_f32[:, half:]

        def hs_rope(x, n_h):
            x = x.reshape(seq_len, n_h, head_dim).astype(np.float32)
            x1 = x[..., :half]
            x2 = x[..., half:]
            r1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
            r2 = x2 * cos[:, None, :] + x1 * sin[:, None, :]
            return (
                np.concatenate([r1, r2], axis=-1)
                .reshape(seq_len, n_h * head_dim)
                .astype(bfloat16)
            )

        cpu_q_roped = hs_rope(cpu_q, n_heads)
        cpu_k_roped = hs_rope(cpu_k, n_kv_heads)

        def _diff(name, a, b):
            af = a.astype(np.float32).flatten()
            bf = b.astype(np.float32).flatten()
            cos_ab = float(
                np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30)
            )
            mae = float(np.abs(af - bf).mean())
            print(
                f"  {name}: shape={a.shape}  ||NPU||={np.linalg.norm(af):.2f}  "
                f"||CPU||={np.linalg.norm(bf):.2f}  cos={cos_ab:+.4f}  MAE={mae:.4f}  "
                f"NPU[:5]={af[:5].round(3)}  CPU[:5]={bf[:5].round(3)}"
            )

        _diff("normed", npu_int["normed"], normed)
        _diff("Q", npu_int["q"], cpu_q.astype(bfloat16))
        _diff("K", npu_int["k"], cpu_k.astype(bfloat16))
        _diff("V", npu_int["v"], cpu_v)
        _diff("Q_roped", npu_int["q_roped"], cpu_q_roped)
        _diff("K_roped", npu_int["k_roped"], cpu_k_roped)
        return

    # ---- N-layer NPU prefill.
    print(
        f"\nRunning {args.n_layers}-layer NPU prefill "
        f"(prefill-dtype={args.prefill_dtype})..."
    )
    t0 = time.time()

    if args.prefill_dtype == "bf16":
        # Reuse the bf16 prefill driver. weights_bf16.layers[li] is the
        # dequantized AWQ weights — same numerical values the int4 path
        # consumes, just laid out as plain bf16 matrices.
        sys.path.insert(0, _LLAMA_BF16)
        from llama32_1b_prefill import run_transformer_block, preload_prefill_weights

        preload_prefill_weights(
            weights_bf16,
            config,
            cache,
            seq_len,
            rope_lut_bf16,
        )

    for li in range(args.n_layers):
        t_layer = cache.profiler.start_layer()
        if args.prefill_dtype == "int4":
            x_bf16 = _run_layer_int4(
                x_bf16,
                weights_bf16.layers[li],
                layers_packed[li],
                rope_lut_bf16,
                config,
                cache,
                li,
                cpu_attn=args.cpu_attn,
            )
        else:
            x_bf16, _ = run_transformer_block(
                x_bf16,
                weights_bf16.layers[li],
                rope_lut_bf16,
                config,
                cache,
                layer_idx=li,
                cpu_attn=args.cpu_attn,
                verbose=args.verbose,
            )
        cache.profiler.end_layer(li, t_layer)
        print(
            f"  layer {li+1}/{args.n_layers} done "
            f"({time.time()-t0:.1f}s, ||x||="
            f"{np.linalg.norm(x_bf16.astype(np.float32)):.3f})"
        )

    # ---- Final RMSNorm + lm_head on prediction position (CPU).
    print("\nFinal RMSNorm + LM-head on pred position (CPU)...")
    last_row = np.asarray(x_bf16, dtype=np.float32)[pred_pos]
    normed = rms_norm(last_row[np.newaxis, :], weights_bf16.final_norm).flatten()
    npu_logits = (
        normed.astype(np.float32) @ weights_bf16.lm_head.astype(np.float32).T
    ).astype(np.float32)

    # ---- Compare top-K.
    npu_label = f"NPU {args.prefill_dtype}"
    print(f"\n{'='*60}")
    print(f"Top-{args.topk} comparison ({npu_label} prefill vs HF bf16 dequant)")
    print(f"{'='*60}")
    npu_top = np.argsort(-npu_logits)[: args.topk]
    overlap = len(set(hf_top.tolist()) & set(npu_top.tolist()))

    print(f"\n{npu_label} top-{args.topk}:")
    for r, tid in enumerate(npu_top):
        text = tok.decode([int(tid)])
        mark = "*" if int(tid) in set(hf_top.tolist()) else " "
        print(
            f"  #{r+1:2d}{mark} id={int(tid):>6d}  logit={float(npu_logits[tid]):+9.3f}  {text!r}"
        )

    print(f"\nTop-{args.topk} overlap   : {overlap}/{args.topk}")
    print(f"HF argmax       : id={int(hf_top[0])}   {tok.decode([int(hf_top[0])])!r}")
    print(
        f"{npu_label} argmax : id={int(npu_top[0])}   {tok.decode([int(npu_top[0])])!r}"
    )
    print(f"Argmax match    : {int(hf_top[0]) == int(npu_top[0])}")

    union = sorted(set(hf_top.tolist()) | set(npu_top.tolist()))
    hf_v = hf_logits[union]
    npu_v = npu_logits[union]
    if hf_v.std() > 0 and npu_v.std() > 0:
        corr = float(np.corrcoef(hf_v, npu_v)[0, 1])
        print(f"Top-{args.topk}-union NPU-vs-HF logit Pearson r: {corr:.4f}")

    if args.cpu_int4:
        # NPU vs CPU int4 isolates the NPU kernel correctness (same weights,
        # same dequant, same numpy attention placeholder — only the
        # rms+gemm+rope and o+ffn paths differ).
        npu_cpu_union = sorted(set(npu_top.tolist()) | set(cpu_top.tolist()))
        a, b = cpu_logits[npu_cpu_union], npu_logits[npu_cpu_union]
        npu_cpu_overlap = len(set(npu_top.tolist()) & set(cpu_top.tolist()))
        print(
            f"Top-{args.topk} {npu_label}-vs-CPU-int4 overlap: {npu_cpu_overlap}/{args.topk}"
        )
        if a.std() > 0 and b.std() > 0:
            print(
                f"{npu_label}-vs-CPU-int4 logit Pearson r: {float(np.corrcoef(a,b)[0,1]):.4f}"
            )

    if cache.profiler.enabled:
        cache.profiler.report()

    if args.min_overlap > 0:
        argmax_match = int(hf_top[0]) == int(npu_top[0])
        verdict = "PASS" if (overlap >= args.min_overlap and argmax_match) else "FAIL"
        print(
            f"\n[verify] {verdict} "
            f"(overlap={overlap}/{args.topk} threshold={args.min_overlap} "
            f"argmax_match={argmax_match})"
        )


if __name__ == "__main__":
    main()
