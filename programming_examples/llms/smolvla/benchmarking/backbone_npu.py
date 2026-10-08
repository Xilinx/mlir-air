# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the SmolVLA backbone's transformer layers on NPU2 via llama32_1b's
already-built, already-fused prefill machinery (rms_gemms_rope + flash_attn/
CPU-attn + o_ffn per layer), instead of a hand-stitched raw-GEMM prototype.

Backbone's `text_model` is confirmed to BE `transformers.models.llama.modeling_llama
.LlamaModel` (bit-for-bit, not "llama-like"). But lerobot's SmolVLMWithExpertModel.forward_attn_layer does
NOT call LlamaModel.forward(): it reaches into layer.self_attn.{q,k,v,o}_proj and
layer.mlp directly and re-implements RoPE/masking/residuals itself, with its own
conventions that differ from the HF config:
  - RoPE base = 10000 (apply_rope's hardcoded default), NOT text_config.rope_theta
    (100000) -- confirmed by reading every apply_rope call site, none override it.
  - position_ids are NOT arange(seq_len): they REPEAT (multi-camera prefix tokens
    share position ids), so the RoPE LUT must be gathered by the real position_id
    values, not indexed by sequence position.
  - attention_mask is a real (seq, seq) bool prefix mask (confirmed non-causal,
    non-symmetric by probing it), not causal -- llama32_1b's attention_reference
    hardcodes a causal mask, so it is monkeypatched here with a masked variant.
  - RMSNorm eps=1e-5 matches llama32_1b_cpu_helpers' default; no change needed.

Usage: python backbone_npu.py [--layers 16] [--cpu-attn] [--profile]
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_SMOLVLA = Path(__file__).resolve().parent.parent
_LLMS = _SMOLVLA.parent
_LLAMA = _LLMS / "llama32_1b"
for p in (str(_SMOLVLA), str(_LLAMA)):
    if p not in sys.path:
        sys.path.insert(0, p)

# programming_examples/ is published as the air_examples package rather than
# put on sys.path: every directory under it would otherwise become a
# top-level module name and shadow an installed package that shares it.
import types

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(_LLMS.parent)
]

os.environ.setdefault("SMOLVLA_CPU_BIND", "1")
os.environ.setdefault("SMOLVLA_CPU_THREADS", "8")
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES", "cores")

import llama32_1b_prefill as prefill  # noqa: E402
from llama32_1b_weights import LlamaConfig, LlamaWeights, LayerWeights  # noqa: E402

BACKBONE_CONFIG = LlamaConfig(
    n_layers=16,
    emb_dim=960,
    n_heads=15,
    head_dim=64,
    n_kv_heads=5,
    hidden_dim=2560,
    vocab_size=1,  # unused: backbone forward never touches embed/lm_head here
    rope_base=10000.0,  # apply_rope's hardcoded default, NOT text_config.rope_theta=100000
)

SEQ_REAL = 241
SEQ_PAD = 256  # 241 padded to a multiple of 64 (llama32_1b_prefill's fused-cast GEMM requirement)


def extract_backbone_weights(policy) -> LlamaWeights:
    """Pull the REAL loaded weights straight out of the live nn.Module (not a
    fresh safetensors download) -- guarantees an exact match with the CPU
    reference for correctness comparison."""
    import torch

    tm = policy.model.vlm_with_expert.get_vlm_model().text_model
    layers = []
    for layer in tm.layers:
        t = lambda w: np.ascontiguousarray(
            w.detach().to(torch.float32).numpy().T
        ).astype(bfloat16)
        norm = lambda w: w.detach().to(torch.float32).numpy().astype(bfloat16)
        layers.append(
            LayerWeights(
                attn_norm=norm(layer.input_layernorm.weight),
                wq=t(layer.self_attn.q_proj.weight),
                wk=t(layer.self_attn.k_proj.weight),
                wv=t(layer.self_attn.v_proj.weight),
                wo=t(layer.self_attn.o_proj.weight),
                ffn_norm=norm(layer.post_attention_layernorm.weight),
                w_gate=t(layer.mlp.gate_proj.weight),
                w_up=t(layer.mlp.up_proj.weight),
                w_down=t(layer.mlp.down_proj.weight),
            )
        )
    return LlamaWeights(embed_table=None, layers=layers, final_norm=None, lm_head=None)


def build_rope_lut_gathered(
    position_ids: np.ndarray, config: LlamaConfig
) -> np.ndarray:
    """LUT rows gathered by the REAL position_id values (which repeat), not by
    sequence index -- matches apply_rope's `radians = positions / timescale`
    exactly (verified algebraically identical to standard rotate_half RoPE)."""
    head_dim = config.head_dim
    half = head_dim // 2
    max_pos = int(position_ids.max()) + 1
    dim_indices = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (config.rope_base ** (dim_indices / head_dim))
    positions = np.arange(max_pos, dtype=np.float64)
    angles = np.outer(positions, inv_freq)
    base_lut = np.empty((max_pos, head_dim), dtype=np.float64)
    base_lut[:, :half] = np.cos(angles)
    base_lut[:, half:] = np.sin(angles)
    return base_lut[position_ids].astype(bfloat16)


def masked_attention_reference(q, k, v, n_heads, n_kv_heads, mask_bool):
    """Same structure as llama32_1b_cpu_helpers.attention_reference, but with
    an explicit (seq, seq) bool mask (True=attend) instead of a hardcoded
    causal mask -- matches eager_attention_forward's `torch.where(mask, w, big_neg)`."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    seq_len = q.shape[0]
    head_dim = q.shape[1] // n_heads
    group_size = n_heads // n_kv_heads

    q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(head_dim)
    big_neg = np.finfo(np.float32).min
    add_mask = np.where(mask_bool, 0.0, big_neg).astype(np.float32)

    out_heads = np.empty((n_heads, seq_len, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_idx = h // group_size
        scores = q[h] @ k[kv_idx].T * scale + add_mask
        m = scores.max(axis=-1, keepdims=True)
        p = np.exp(scores - m)
        probs = p / p.sum(axis=-1, keepdims=True)
        out_heads[h] = probs @ v[kv_idx]
    return out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)


_MASK_HOLDER = {"mask": None}
# runtime_loop_tiling_sizes of the fused rms_gemms_rope / o_ffn ELFs, shared by
# compile and load (main() overrides from --rgr-tiling / --offn-tiling).
_TILING = {"rgr": [2, 2], "offn": [2, 2]}
# Additive bf16 (seq, seq) mask for the on-NPU FlashAttention ELF; None keeps
# attention on the CPU (main() sets it from --npu-attn).
_NPU_ATTN = {"mask": None}
# The FA grid is [1, n_heads]: tiling the head dim breaks the K/V DMA strides
# (13-element stride, not 4-byte aligned) at [x, 5]/[x, 6] and silently
# miscompiles at [x, 3]; the M dim has trip count 1, so [2, 1] == [1, 1].
_FA_BACKEND = {
    "verbose": False,
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "attention_bf16",
    "runtime_loop_tiling_sizes": [2, 1],
}
# Whole-layer ELF: empty global tiling, so each launch's air.shim_dma_tile_sizes applies.
_LAYER_BACKEND = {
    "verbose": False,
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "layer",
    "runtime_loop_tiling_sizes": [],
}


# GEMMs taking bfp16ebs8 weights -> (tile_n, tile_k_l2, tile_k_l1) (main() fills from --bfp16).
_BFP16 = {}
_BFP16_TILES = {
    "qkv": (80, 480, 160),
    "o": (80, 480, 160),
    "gu": (128, 480, 160, 8),
    "dn": (80, 256, 128),
}


# gemm_engine O+FFN tiles (--offn-engine): one tile_n / tile_k_l1 for O, GateUp and Down.
_ENGINE = {"on": False, "qkv": False, "tile_n": 80, "tile_k_l1": 160, "herd": 4}


def _engine_offn_weights(lw, emb, hidden):
    """wo, w_gateup (ffn_norm folded in, SwiGLU-permuted) and w_down packed for the engine."""
    from gemm_engine import permute_gate_up
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        pack_b_bfp16ebs8,
    )

    tn, tk1 = _ENGINE["tile_n"], _ENGINE["tile_k_l1"]
    f32 = np.float32
    nw = np.asarray(lw.ffn_norm, dtype=bfloat16).astype(f32).reshape(emb, 1)
    wg = (nw * np.asarray(lw.w_gate, dtype=bfloat16).astype(f32)).astype(bfloat16)
    wu = (nw * np.asarray(lw.w_up, dtype=bfloat16).astype(f32)).astype(bfloat16)

    def pack(w):
        return pack_b_bfp16ebs8(
            np.ascontiguousarray(np.asarray(w, dtype=bfloat16)), tn, tk1
        )

    return (
        pack(np.asarray(lw.wo).reshape(emb, emb)),
        pack(permute_gate_up(wg, wu, tn, tn * _ENGINE["herd"])),
        pack(np.asarray(lw.w_down).reshape(hidden, emb)),
    )


def _engine_qkv_args(lw, rope_lut_bf16, config, seq_len):
    """w_qkv (attn_norm folded in, q/k heads pair-interleaved) packed for the
    engine, and its RoPE table."""
    from gemm_engine import qkv_col_perm, rope_table
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        pack_b_bfp16ebs8,
    )

    f32 = np.float32
    nh, nkv, hd = config.n_heads, config.n_kv_heads, config.head_dim
    w = np.concatenate([lw.wq, lw.wk, lw.wv], axis=1).astype(bfloat16).astype(f32)
    nw = np.asarray(lw.attn_norm, dtype=bfloat16).astype(f32)[:, None]
    wp = (nw * w).astype(bfloat16)[:, qkv_col_perm(nh, nkv, hd)]
    table = rope_table(
        np.asarray(rope_lut_bf16[:seq_len], f32), nh, nkv, hd, nkv * hd
    ).astype(bfloat16)
    return (
        pack_b_bfp16ebs8(
            np.ascontiguousarray(wp), _ENGINE["tile_n"], _ENGINE["tile_k_l1"]
        ),
        table,
    )


def run_layer_engine(
    x_bf16, layer_weights, rope_lut_bf16, config, cache, layer_idx=0, with_kv=False
):
    """One layer as the 3-launch engine ELF (layer_fused.build_engine_layer_module).
    with_kv=True also returns the qkv buffer (q/k pair-interleaved; a view into a
    shared BO, overwritten by the next call)."""
    from layer_fused import (
        ENG_LAYER_INTERMEDIATE,
        ENG_LAYER_OUT,
        ENG_LAYER_QKV,
        ENG_LAYER_STATIC,
    )

    seq_len = x_bf16.shape[0]
    emb, hidden = config.emb_dim, config.hidden_dim
    kv = config.n_kv_heads * config.head_dim
    _arg_cache = getattr(run_layer_engine, "_arg_cache", {})
    run_layer_engine._arg_cache = _arg_cache
    key = f"elayer_L{layer_idx}"
    if key not in _arg_cache:
        lw = layer_weights

        def z(*shape):
            return np.zeros(shape, dtype=bfloat16)

        wqkv, table = _engine_qkv_args(lw, rope_lut_bf16, config, seq_len)
        wo, wgu, wdn = _engine_offn_weights(lw, emb, hidden)
        _arg_cache[key] = [
            None,
            wqkv,
            table,
            z(seq_len, emb + 2 * kv),
            _NPU_ATTN["mask"],
            z(seq_len, emb),
            wo,
            z(seq_len, emb),
            wgu,
            z(seq_len, hidden),
            wdn,
            z(seq_len, emb),
        ]
    args = _arg_cache[key]
    args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb)
    results = cache.load_and_run(
        "layer",
        _LAYER_BACKEND,
        *args,
        output_indices=[ENG_LAYER_OUT, ENG_LAYER_QKV] if with_kv else [ENG_LAYER_OUT],
        static_input_indices=ENG_LAYER_STATIC,
        intermediate_indices=ENG_LAYER_INTERMEDIATE,
        bo_key=key,
        shared_nonstatic=True,
    )
    out = results[ENG_LAYER_OUT].reshape(seq_len, emb)
    if with_kv:
        return out, results[ENG_LAYER_QKV].reshape(seq_len, emb + 2 * kv)
    return out


def _weight(key, w):
    """bf16 [K, N] weight -> what the GEMM `key` consumes (packed bfp16ebs8 if enabled)."""
    w = np.ascontiguousarray(np.asarray(w, dtype=bfloat16))
    if key not in _BFP16:
        return w
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        pack_b_bfp16ebs8,
    )

    tn, _, tk1 = _BFP16[key][:3]
    return pack_b_bfp16ebs8(w, tn, tk1)


def additive_attn_mask(mask_bool):
    """bool (True = attend) -> the additive bf16 mask attn_npu2_seqfirst's
    attn_mask=True expects: 0 where kept, bf16 lowest (0xff7f) where dropped.
    Lowest rather than -inf keeps fully masked rows finite, reproducing the
    float32-min reference (such a row averages every key)."""
    m = np.zeros(mask_bool.shape, np.uint16)
    m[~mask_bool] = 0xFF7F
    return m.view(bfloat16)


def _patched_attention_reference(q, k, v, n_heads, n_kv_heads):
    return masked_attention_reference(
        q, k, v, n_heads, n_kv_heads, _MASK_HOLDER["mask"]
    )


def capture_real_backbone_io(policy, n_layers_to_capture):
    """Hook forward_attn_layer (gives real hidden-in/mask/position_ids per
    layer) and o_proj/mlp (gives the pieces needed to reconstruct the exact
    per-layer output, since lerobot inlines the residual adds in plain Python
    rather than calling a single hookable `layer.forward`)."""
    import torch
    from smolvla_inference import build_oracle_batch, fixed_noise, run_hybrid_forward

    vwe = policy.model.vlm_with_expert
    orig_fal = vwe.forward_attn_layer
    per_layer = {}

    def hooked_fal(
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        *rest,
        **kw,
    ):
        if layer_idx < n_layers_to_capture and layer_idx not in per_layer:
            per_layer[layer_idx] = {
                "hidden_in": inputs_embeds[0].detach().clone(),
                "position_ids": position_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
            }
        return orig_fal(
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            *rest,
            **kw,
        )

    vwe.forward_attn_layer = hooked_fal

    # o_proj / mlp hooks to reconstruct the real per-layer output (see forward()
    # lines ~480-491: out_emb = o_proj(att_out) + hidden_in; after_first_residual
    # = out_emb; out_emb = mlp(post_attention_layernorm(out_emb)); out_emb += after_first_residual).
    tm = policy.model.vlm_with_expert.get_vlm_model().text_model
    o_outs, mlp_outs = {}, {}
    o_hooks, mlp_hooks = [], []
    call_idx = {"o": 0, "mlp": 0}

    def make_o_hook(idx):
        def hook(module, inp, out):
            if call_idx["o"] == idx:
                o_outs[idx] = out.detach().clone()
            call_idx["o"] += 1

        return hook

    def make_mlp_hook(idx):
        def hook(module, inp, out):
            if call_idx["mlp"] == idx:
                mlp_outs[idx] = out.detach().clone()
            call_idx["mlp"] += 1

        return hook

    for i in range(n_layers_to_capture):
        o_hooks.append(
            tm.layers[i].self_attn.o_proj.register_forward_hook(make_o_hook(i))
        )
        mlp_hooks.append(tm.layers[i].mlp.register_forward_hook(make_mlp_hook(i)))

    policy_eval = policy
    batch = build_oracle_batch(policy_eval, n_cameras=3)
    noise = fixed_noise(policy_eval)
    run_hybrid_forward(batch, policy=policy_eval, noise=noise, npu_vision=False)

    vwe.forward_attn_layer = orig_fal
    for h in o_hooks + mlp_hooks:
        h.remove()

    real_outputs = {}
    for i in range(n_layers_to_capture):
        hidden_in = per_layer[i]["hidden_in"]
        after_first_residual = o_outs[i] + hidden_in
        real_outputs[i] = (
            (after_first_residual + mlp_outs[i]).squeeze(0).to(torch.float32).numpy()
        )
        per_layer[i]["hidden_in"] = hidden_in.squeeze(0).to(torch.float32).numpy()
        per_layer[i]["position_ids"] = per_layer[i]["position_ids"].squeeze(0).numpy()
        per_layer[i]["attention_mask"] = (
            per_layer[i]["attention_mask"].squeeze(0).numpy()
        )

    return per_layer, real_outputs


def compile_backbone_kernels(
    cache,
    config,
    seq_len,
    herd_m_override=None,
    fused_gu=False,
    gu_tile_n=80,
    fused_qkv=False,
    qkv_tile_n=80,
    gu_bstationary=False,
    qkv_bstationary=False,
    od_bstationary=False,
    od_tile_n=80,
    o_bstationary=None,
    dn_bstationary=None,
    dn_herd_m=None,
    dn_tile_m=32,
    dn_tile_n=None,
    offn_dup=(),
    gu_swiglu=False,
    npu_attn=False,
    fused_layer=False,
    fa_opt="-O2",
    layers_per_call=1,
    fa_his=1,
    fa_qb=False,
    offn_engine=False,
    qkv_engine=False,
):
    """Replacement for llama32_1b_prefill.compile_all_kernels: that function
    hardcodes mm.o pre-compiles at tile_n=128 (llama32_1b's own registry
    tile_n), which would SILENTLY produce wrong results here -- backbone's
    registry rows resolve to tile_n=80 (Q/K/V/O/Down) and tile_n=128
    (Gate/Up), and compile_gemm_mm bakes tile_n as a compile-time DIM_N macro
    into the object; reusing a wrongly-baked mm_m32.o under the same symbol
    name is not a link error, it is a silent correctness bug. Verified via a
    probe (not guessed) exactly which 3 (tile_m, tile_n, tile_k_l1, sym_suffix)
    combinations the registry + disambiguate_by_tile_n actually produce for
    this config at seq_len=256:
        rms_gemms_rope Q/K/V : drain, tile_n=80,  sym_suffix "_m32"      -> mm_m32.o
        o_ffn O/Down         : drain, tile_n=80,  sym_suffix "_m32_n80"  -> mm_m32_n80.o
        o_ffn Gate/Up        : drain, tile_n=128, sym_suffix "_m32_n128"-> mm_m32_n128.o
    """
    from air_examples.llms.shared.infra.external_kernels import compile_gemm_mm
    from air_examples.llms.shared.builders.rms_gemms_rope_multi import (
        build_rms_gemms_rope_module,
    )
    from air_examples.llms.shared.builders.o_ffn_multi import build_o_ffn_module

    compile_gemm_mm(
        tile_m=32, tile_n=80, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o"
    )
    compile_gemm_mm(
        tile_m=32,
        tile_n=80,
        tile_k_l1=32,
        sym_suffix="_m32_n80",
        out_name="mm_m32_n80.o",
    )
    compile_gemm_mm(
        tile_m=32,
        tile_n=128,
        tile_k_l1=32,
        sym_suffix="_m32_n128",
        out_name="mm_m32_n128.o",
    )

    gemm_herd_m = herd_m_override or next(
        h for h in (8, 4, 2, 1) if seq_len % (64 * h) == 0
    )

    if fused_qkv:
        from rms_gemms_rope_fused_qkv import build_rms_gemms_rope_module_fused_qkv

        rgr_mod = build_rms_gemms_rope_module_fused_qkv(
            seq_len,
            config.emb_dim,
            config.n_kv_heads * config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            herd_m=gemm_herd_m,
            qkv_tile_n=qkv_tile_n,
            b_stationary=qkv_bstationary,
            bfp16=_BFP16.get("qkv"),
        )
        cache.compile_and_cache(
            "rms_gemms_rope",
            rgr_mod,
            {
                "verbose": cache.verbose,
                "omit_while_true_loop": False,
                "output_format": "elf",
                "instance_name": "rms_gemms_rope_fused_qkv",
                "runtime_loop_tiling_sizes": _TILING["rgr"],
            },
        )
    else:
        cache.compile_and_cache(
            "rms_gemms_rope",
            build_rms_gemms_rope_module(
                seq_len,
                config.emb_dim,
                config.n_kv_heads * config.head_dim,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                herd_m=gemm_herd_m,
            ),
            {"verbose": cache.verbose, **prefill._rms_gemms_rope_run_backend()},
        )
    o_ffn_backend = {
        "verbose": cache.verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn",
        "runtime_loop_tiling_sizes": [2, 2],
    }
    fused_offn_backend = {
        **o_ffn_backend,
        "instance_name": "o_ffn_fused_gu",
        "runtime_loop_tiling_sizes": _TILING["offn"],
    }
    if fused_gu:
        from o_ffn_fused_gu import build_o_ffn_module_fused_gu

        offn_mod = build_o_ffn_module_fused_gu(
            seq_len,
            config.emb_dim,
            config.hidden_dim,
            herd_m=gemm_herd_m,
            gu_tile_n=gu_tile_n,
            gu_b_stationary=gu_bstationary,
            od_b_stationary=od_bstationary,
            od_tile_n=od_tile_n,
            o_b_stationary=o_bstationary,
            dn_b_stationary=dn_bstationary,
            dn_herd_m=dn_herd_m,
            dn_tile_m=dn_tile_m,
            dn_tile_n=dn_tile_n,
            dup=offn_dup,
            gu_swiglu=gu_swiglu,
            bfp16={k: v for k, v in _BFP16.items() if k in ("o", "gu", "dn")},
        )
        cache.compile_and_cache("o_ffn", offn_mod, fused_offn_backend)
    else:
        cache.compile_and_cache(
            "o_ffn",
            build_o_ffn_module(
                seq_len, config.emb_dim, config.hidden_dim, herd_m=gemm_herd_m
            ),
            o_ffn_backend,
        )
    if npu_attn:
        from air_examples.flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
            build_module,
        )
        import air_examples.llms.shared.infra.external_kernels as ek

        hd = config.head_dim
        kv_dim = config.n_kv_heads * hd
        peano_flags = ek._PEANO_FLAGS
        ek._PEANO_FLAGS = [fa_opt if f == "-O2" else f for f in peano_flags]
        try:
            # bfp16=True: the plain bf16 microkernel NaNs at this shape.
            ek.compile_attn_npu2(head_dim=hd, bfp16=True, force=True)
        finally:
            ek._PEANO_FLAGS = peano_flags
        # V is read in place from the fused QKV GEMM output (its last kv_dim columns).
        fa_mod = build_module(
            lk=seq_len,
            lkp=hd,
            lq=seq_len,
            lqp=seq_len,
            dk=hd,
            dv=hd,
            num_q_tiles=seq_len // hd,
            num_cascade_stages=seq_len // hd,
            num_heads=config.n_heads,
            num_kv_heads=config.n_kv_heads,
            num_heads_per_unroll=1,
            causal=False,
            attn_mask=True,
            v_cols=config.emb_dim + 2 * kv_dim,
            heads_in_segment=fa_his,
            q_bcast=fa_qb,
        )
        cache.compile_and_cache(
            "flash_attn", fa_mod, {**_FA_BACKEND, "verbose": cache.verbose}
        )
    if fused_layer:
        from layer_fused import OFFN_ENGINE_ORDER, build_layer_module

        offn_tiling = _TILING["offn"]
        if offn_engine:
            from gemm_engine import Job, build_gemm_engine, compile_mm_engine

            E, H = config.emb_dim, config.hidden_dim
            tn, tk1, herd = _ENGINE["tile_n"], _ENGINE["tile_k_l1"], _ENGINE["herd"]
            compile_mm_engine(32, tn, tk1, "_eng", "mm_engine.o", rms_k=E)
            offn_mod = build_gemm_engine(
                seq_len,
                [
                    Job("attn", "wo", "res1", E, E, residual="x"),
                    Job("res1", "wgu", "sw", E, 2 * H, rms=True, swiglu=True),
                    Job("sw", "wdn", "out", H, E, residual="res1"),
                ],
                32,
                tn,
                tk1,
                tn * herd,
                herd,
                herd,
                "_eng",
                "mm_engine.o",
                arg_order=OFFN_ENGINE_ORDER,
            )
            offn_tiling = []
        if qkv_engine:
            from layer_fused import QKV_ENGINE_ORDER, build_engine_layer_module
            from air_examples.flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
                build_module,
            )

            hd, E = config.head_dim, config.emb_dim
            qkv_eng = build_gemm_engine(
                seq_len,
                [
                    Job(
                        "x",
                        "wqkv",
                        "qkv",
                        E,
                        E + 2 * config.n_kv_heads * hd,
                        rms=True,
                        rope="rope",
                    )
                ],
                32,
                tn,
                tk1,
                tn * herd,
                herd,
                herd,
                "_eng",
                "mm_engine.o",
                arg_order=QKV_ENGINE_ORDER,
            )
            fa_fused = build_module(
                lk=seq_len,
                lkp=hd,
                lq=seq_len,
                lqp=seq_len,
                dk=hd,
                dv=hd,
                num_q_tiles=seq_len // hd,
                num_cascade_stages=seq_len // hd,
                num_heads=config.n_heads,
                num_kv_heads=config.n_kv_heads,
                num_heads_per_unroll=1,
                causal=False,
                attn_mask=True,
                fused_qkv=True,
                heads_in_segment=fa_his,
                q_bcast=fa_qb,
            )
            layer_mod = build_engine_layer_module(
                str(qkv_eng),
                str(fa_fused),
                str(offn_mod),
                _FA_BACKEND["runtime_loop_tiling_sizes"],
            )
        else:
            layer_mod = build_layer_module(
                str(rgr_mod),
                str(fa_mod),
                str(offn_mod),
                {
                    "rgr": _TILING["rgr"],
                    "fa": _FA_BACKEND["runtime_loop_tiling_sizes"],
                    "offn": offn_tiling,
                },
                offn_engine=offn_engine,
            )
        cache.compile_and_cache(
            "layer", layer_mod, {**_LAYER_BACKEND, "verbose": cache.verbose}
        )
        if layers_per_call > 1:
            from layer_fused import build_multi_layer_module

            cache.compile_and_cache(
                f"layers{layers_per_call}",
                build_multi_layer_module(
                    str(rgr_mod),
                    str(fa_mod),
                    str(offn_mod),
                    {
                        "rgr": _TILING["rgr"],
                        "fa": _FA_BACKEND["runtime_loop_tiling_sizes"],
                        "offn": _TILING["offn"],
                    },
                    layers_per_call,
                ),
                {**_LAYER_BACKEND, "instance_name": "layers", "verbose": cache.verbose},
            )
    cache._save_manifest()
    print(f"Backbone kernels compiled and cached to {cache.cache_dir}/")


def run_transformer_block_custom(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    fused_qkv=False,
    fused_gu=False,
    gu_swiglu_half=0,
):
    """Same as llama32_1b_prefill.run_transformer_block, but either or both
    halves can use a fused-launch variant:
      fused_qkv: rms_gemms_rope_fused_qkv's 9-arg/4-launch RMS+QKV+RoPE
                 (Q+K+V in one GEMM; V has no RoPE, sliced host-side from the
                 fused GEMM's wide output) instead of the original 13-arg/6-launch.
      fused_gu:  o_ffn_fused_gu's 13-arg/7-launch O+FFN (Gate+Up in one GEMM)
                 instead of the original 15-arg/8-launch.
    Both False reproduces prefill.run_transformer_block's own arg layout
    exactly (kept here rather than delegating, so one function handles every
    combination without re-deriving the unfused arg lists twice).
    """
    seq_len = x_bf16.shape[0]
    emb_dim, n_heads, n_kv_heads, head_dim, hidden_dim = (
        config.emb_dim,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.hidden_dim,
    )
    kv_dim = n_kv_heads * head_dim
    _arg_cache = getattr(run_transformer_block_custom, "_arg_cache", {})
    run_transformer_block_custom._arg_cache = _arg_cache

    # ---- RMS + QKV + RoPE ----
    if fused_qkv:
        qkv_n = emb_dim + 2 * kv_dim
        _rms_key = f"rms_gemms_rope_qkv_L{layer_idx}"
        if _rms_key not in _arg_cache:
            w_qkv = _weight(
                "qkv",
                np.concatenate(
                    [layer_weights.wq, layer_weights.wk, layer_weights.wv], axis=1
                ),
            )
            _rms_args = [
                None,
                np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                w_qkv,
                np.zeros((seq_len, qkv_n), dtype=bfloat16),
                np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten(),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten(),
                np.zeros((seq_len, kv_dim), dtype=bfloat16),
            ]
            _arg_cache[_rms_key] = _rms_args
        cached_args = _arg_cache[_rms_key]
        cached_args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)

        results = cache.load_and_run(
            "rms_gemms_rope",
            {
                "verbose": False,
                "omit_while_true_loop": False,
                "output_format": "elf",
                "instance_name": "rms_gemms_rope_fused_qkv",
                "runtime_loop_tiling_sizes": _TILING["rgr"],
            },
            *cached_args,
            output_indices=[4, 6, 8],
            static_input_indices={1, 3, 5, 7},
            intermediate_indices={2, 4, 6, 8},
            bo_key=_rms_key,
            shared_nonstatic=True,
        )
        qkv_buf = results[4].reshape(seq_len, qkv_n)
        v = qkv_buf[:, emb_dim + kv_dim : emb_dim + 2 * kv_dim]
        q_roped = results[6].reshape(seq_len, emb_dim)
        k_roped = results[8].reshape(seq_len, kv_dim)
    else:
        _rms_key = f"rms_gemms_rope_L{layer_idx}"
        if _rms_key not in _arg_cache:
            _rms_args = [
                None,
                np.asarray(layer_weights.attn_norm, dtype=bfloat16).reshape(emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.wq, dtype=bfloat16).reshape(emb_dim, emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),
                np.zeros((seq_len, kv_dim), dtype=bfloat16),
                np.asarray(layer_weights.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),
                np.zeros((seq_len, kv_dim), dtype=bfloat16),
                np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten(),
                np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten(),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.zeros((seq_len, kv_dim), dtype=bfloat16),
            ]
            _scratch_arrays, _scratch_inter = prefill._rms_scratch_specs(
                seq_len, emb_dim, kv_dim
            )
            _rms_args.extend(_scratch_arrays)
            _arg_cache[_rms_key] = (_rms_args, _scratch_inter)
        cached_args, _scratch_inter = _arg_cache[_rms_key]
        cached_args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb_dim)

        _rms_inter = {2, 4, 6, 8, 11, 12} | _scratch_inter
        results = cache.load_and_run(
            "rms_gemms_rope",
            prefill._rms_gemms_rope_run_backend(),
            *cached_args,
            output_indices=[8, 11, 12],
            static_input_indices={1, 3, 5, 7, 9, 10},
            intermediate_indices=_rms_inter,
            bo_key=_rms_key,
            shared_nonstatic=True,
        )
        v = results[8].reshape(seq_len, kv_dim)
        q_roped = results[11].reshape(seq_len, n_heads * head_dim)
        k_roped = results[12].reshape(seq_len, n_kv_heads * head_dim)

    if _NPU_ATTN["mask"] is not None:
        _fa_key = "flash_attn"
        if _fa_key not in _arg_cache:
            _arg_cache[_fa_key] = [
                None,
                None,
                None,
                _NPU_ATTN["mask"],
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
            ]
        fa_args = _arg_cache[_fa_key]
        fa_args[0], fa_args[1], fa_args[2] = q_roped, k_roped, qkv_buf
        attn_out = cache.load_and_run(
            "flash_attn",
            _FA_BACKEND,
            *fa_args,
            output_indices=[4],
            static_input_indices={3},
            intermediate_indices={4},
            bo_key=_fa_key,
            shared_nonstatic=True,
        )[4].reshape(seq_len, emb_dim)
    else:
        with cache.profiler.time_cpu("prefill_cpu_attention"):
            attn_out = prefill.attention_reference(
                q_roped.astype(np.float32),
                k_roped.astype(np.float32),
                v.astype(np.float32),
                n_heads,
                n_kv_heads,
            ).astype(bfloat16)

    # ---- O + Residual + FFN ----
    if fused_gu:
        gu_n = 2 * hidden_dim
        _offn_key = f"o_ffn_fused_gu_L{layer_idx}"
        if _offn_key not in _arg_cache:
            if gu_swiglu_half:
                from o_ffn_fused_gu import interleave_gate_up

                w_gateup = interleave_gate_up(
                    np.asarray(layer_weights.w_gate),
                    np.asarray(layer_weights.w_up),
                    gu_swiglu_half,
                )
            else:
                w_gateup = np.concatenate(
                    [layer_weights.w_gate, layer_weights.w_up], axis=1
                )
            w_gateup = _weight("gu", w_gateup)
            offn_args = [
                None,
                _weight("o", np.asarray(layer_weights.wo).reshape(emb_dim, emb_dim)),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                None,
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                w_gateup,
                np.zeros((seq_len, gu_n), dtype=bfloat16),
                np.zeros((seq_len, hidden_dim), dtype=bfloat16),
                _weight(
                    "dn", np.asarray(layer_weights.w_down).reshape(hidden_dim, emb_dim)
                ),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.zeros(seq_len * emb_dim, dtype=bfloat16),
            ]
            _arg_cache[_offn_key] = offn_args
        cached_args = _arg_cache[_offn_key]
        cached_args[0] = np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim)
        cached_args[3] = x_bf16.reshape(seq_len, emb_dim).astype(bfloat16, copy=False)

        _out_idx = 12
        _inter = {2, 4, 6, 8, 9, 11, 12}
        results = cache.load_and_run(
            "o_ffn",
            {
                "verbose": False,
                "omit_while_true_loop": False,
                "output_format": "elf",
                "instance_name": "o_ffn_fused_gu",
                "runtime_loop_tiling_sizes": _TILING["offn"],
            },
            *cached_args,
            output_indices=[_out_idx],
            static_input_indices={1, 5, 7, 10},
            intermediate_indices=_inter,
            bo_key=_offn_key,
            shared_nonstatic=True,
        )
        return results[_out_idx].reshape(seq_len, emb_dim)
    else:
        _offn_key = f"o_ffn_L{layer_idx}"
        if _offn_key not in _arg_cache:
            offn_args = [
                None,
                np.asarray(layer_weights.wo, dtype=bfloat16).reshape(emb_dim, emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                None,
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.ffn_norm, dtype=bfloat16).reshape(emb_dim),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.asarray(layer_weights.w_gate, dtype=bfloat16).reshape(
                    emb_dim, hidden_dim
                ),
                np.zeros((seq_len, hidden_dim), dtype=bfloat16),
                np.asarray(layer_weights.w_up, dtype=bfloat16).reshape(
                    emb_dim, hidden_dim
                ),
                np.zeros((seq_len, hidden_dim), dtype=bfloat16),
                np.zeros((seq_len, hidden_dim), dtype=bfloat16),
                np.asarray(layer_weights.w_down, dtype=bfloat16).reshape(
                    hidden_dim, emb_dim
                ),
                np.zeros((seq_len, emb_dim), dtype=bfloat16),
                np.zeros(seq_len * emb_dim, dtype=bfloat16),
            ]
            offn_args.extend(
                np.zeros(shape, dtype=np.float32)
                for shape in prefill._o_ffn_scratch_plan(seq_len, emb_dim, hidden_dim)[
                    0
                ]
            )
            _arg_cache[_offn_key] = offn_args
        cached_args = _arg_cache[_offn_key]
        cached_args[0] = np.asarray(attn_out, dtype=bfloat16).reshape(seq_len, emb_dim)
        cached_args[3] = x_bf16.reshape(seq_len, emb_dim).astype(bfloat16, copy=False)

        _out_idx = 14
        _inter = {2, 4, 6, 8, 10, 11, 13, 14} | prefill._o_ffn_scratch_plan(
            seq_len, emb_dim, hidden_dim
        )[1]
        results = cache.load_and_run(
            "o_ffn",
            prefill._o_ffn_run_backend(),
            *cached_args,
            output_indices=[_out_idx],
            static_input_indices={1, 5, 7, 9, 12},
            intermediate_indices=_inter,
            bo_key=_offn_key,
            shared_nonstatic=True,
        )
        return results[_out_idx].reshape(seq_len, emb_dim)


def run_layer_fused(
    x_bf16,
    layer_weights,
    rope_lut_bf16,
    config,
    cache,
    layer_idx=0,
    gu_swiglu_half=0,
    with_kv=False,
):
    """One backbone layer as ONE dispatch of the stitched `layer` ELF
    (layer_fused.py): RMS+QKV+RoPE, masked FlashAttention, O+FFN. with_kv=True also
    returns the qkv (V in its last kv columns) and roped-K buffers; they are views
    into shared BOs, overwritten by the next call."""
    from layer_fused import LAYER_INTERMEDIATE, LAYER_OUT, LAYER_STATIC
    from o_ffn_fused_gu import interleave_gate_up

    seq_len = x_bf16.shape[0]
    emb, nh, nkv, hidden = (
        config.emb_dim,
        config.n_heads,
        config.n_kv_heads,
        config.hidden_dim,
    )
    kv = nkv * config.head_dim
    _arg_cache = getattr(run_layer_fused, "_arg_cache", {})
    run_layer_fused._arg_cache = _arg_cache
    key = f"layer_L{layer_idx}"
    if key not in _arg_cache:
        lw = layer_weights
        w_qkv = _weight("qkv", np.concatenate([lw.wq, lw.wk, lw.wv], axis=1))
        w_gateup = _weight(
            "gu",
            interleave_gate_up(
                np.asarray(lw.w_gate), np.asarray(lw.w_up), gu_swiglu_half
            ),
        )

        def z(*shape):
            return np.zeros(shape, dtype=bfloat16)

        args = [
            None,
            np.asarray(lw.attn_norm, dtype=bfloat16).reshape(emb),
            z(seq_len, emb),
            w_qkv,
            z(seq_len, emb + 2 * kv),
            np.repeat(rope_lut_bf16[:seq_len], nh, axis=0).flatten(),
            z(seq_len, emb),
            np.repeat(rope_lut_bf16[:seq_len], nkv, axis=0).flatten(),
            z(seq_len, kv),
            _NPU_ATTN["mask"],
            z(seq_len, emb),
            _weight("o", np.asarray(lw.wo).reshape(emb, emb)),
            z(seq_len, emb),
            z(seq_len, emb),
            np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb),
            z(seq_len, emb),
            w_gateup,
            z(seq_len, hidden),
            _weight("dn", np.asarray(lw.w_down).reshape(hidden, emb)),
            z(seq_len, emb),
            z(seq_len * emb),
        ]
        if _ENGINE["on"]:
            args[11], args[16], args[18] = _engine_offn_weights(lw, emb, hidden)
            args[20] = z(seq_len, emb)
        _arg_cache[key] = args
    args = _arg_cache[key]
    args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb)
    results = cache.load_and_run(
        "layer",
        _LAYER_BACKEND,
        *args,
        output_indices=[LAYER_OUT, 4, 8] if with_kv else [LAYER_OUT],
        static_input_indices=LAYER_STATIC,
        intermediate_indices=LAYER_INTERMEDIATE,
        bo_key=key,
        shared_nonstatic=True,
    )
    out = results[LAYER_OUT].reshape(seq_len, emb)
    if with_kv:
        return (
            out,
            results[4].reshape(seq_len, emb + 2 * kv),
            results[8].reshape(seq_len, kv),
        )
    return out


def run_layers_fused(
    x_bf16,
    first_layer,
    n_layers,
    all_weights,
    rope_lut_bf16,
    config,
    cache,
    gu_swiglu_half=0,
):
    """Layers first_layer .. first_layer+n_layers-1 as ONE dispatch of the
    `layers{n}` ELF (layer_fused.build_multi_layer_module)."""
    from layer_fused import MULTI_PER_LAYER, MULTI_SHARED, multi_layer_arg

    seq_len, emb = x_bf16.shape[0], config.emb_dim
    _arg_cache = getattr(run_layers_fused, "_arg_cache", {})
    run_layers_fused._arg_cache = _arg_cache
    key = f"layers{n_layers}_L{first_layer}"
    out_idx = n_layers % 2
    if key not in _arg_cache:
        per = []
        for i in range(first_layer, first_layer + n_layers):
            # run_layer_fused builds (and caches) the per-layer arg list; reuse its arrays.
            if f"layer_L{i}" not in getattr(run_layer_fused, "_arg_cache", {}):
                run_layer_fused(
                    x_bf16,
                    all_weights[i],
                    rope_lut_bf16,
                    config,
                    cache,
                    layer_idx=i,
                    gu_swiglu_half=gu_swiglu_half,
                )
            per.append(run_layer_fused._arg_cache[f"layer_L{i}"])
        args = [None] * (2 + len(MULTI_SHARED) + n_layers * len(MULTI_PER_LAYER))
        args[1] = np.zeros((seq_len, emb), dtype=bfloat16)
        for i in range(n_layers):
            for a in MULTI_SHARED + MULTI_PER_LAYER:
                args[multi_layer_arg(i, a)] = per[i][a]
        static = {multi_layer_arg(0, a) for a in (5, 7, 9)} | {
            multi_layer_arg(i, a) for i in range(n_layers) for a in MULTI_PER_LAYER
        }
        _arg_cache[key] = (args, static, set(range(len(args))) - static - {0})
    args, static, inter = _arg_cache[key]
    args[0] = np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb)
    results = cache.load_and_run(
        f"layers{n_layers}",
        {**_LAYER_BACKEND, "instance_name": "layers"},
        *args,
        output_indices=[out_idx],
        static_input_indices=static,
        intermediate_indices=inter - {out_idx},
        bo_key=key,
        shared_nonstatic=True,
    )
    return results[out_idx].reshape(seq_len, emb)


def pad_seq(x, seq_pad, fill=0.0):
    out = np.full((seq_pad,) + x.shape[1:], fill, dtype=x.dtype)
    out[: x.shape[0]] = x
    return out


def pad_mask(mask_bool, seq_pad):
    """(seq_real, seq_real) -> (seq_pad, seq_pad): padding rows/cols blocked
    from real tokens (False), each padding row can see itself (diagonal True)
    so its own softmax stays finite -- its output is discarded regardless."""
    seq_real = mask_bool.shape[0]
    out = np.zeros((seq_pad, seq_pad), dtype=bool)
    out[:seq_real, :seq_real] = mask_bool
    for i in range(seq_real, seq_pad):
        out[i, i] = True
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--cpu-attn", action="store_true", default=True)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--herd-m", type=int, default=None)
    ap.add_argument(
        "--fused-gu",
        action="store_true",
        help="Gate+Up fused into 1 GEMM (7 launches vs 8)",
    )
    ap.add_argument("--gu-tile-n", type=int, default=80)
    ap.add_argument(
        "--fused-qkv",
        action="store_true",
        help="Q+K+V fused into 1 GEMM (4 launches vs 6)",
    )
    ap.add_argument("--qkv-tile-n", type=int, default=80)
    ap.add_argument("--gu-bstationary", action="store_true")
    ap.add_argument("--qkv-bstationary", action="store_true")
    ap.add_argument(
        "--od-bstationary",
        action="store_true",
        help="O/Down GEMMs bypass registry, full-K + B-stationary",
    )
    ap.add_argument("--od-tile-n", type=int, default=48)
    ap.add_argument(
        "--o-bstationary",
        action="store_true",
        help="O GEMM only: full-K + B-stationary",
    )
    ap.add_argument(
        "--dn-bstationary",
        action="store_true",
        help="Down GEMM only: full-K + B-stationary",
    )
    ap.add_argument(
        "--dn-herd-m",
        type=int,
        default=None,
        help="herd_m for the B-stationary Down GEMM",
    )
    ap.add_argument("--dn-tile-m", type=int, default=32)
    ap.add_argument("--dn-tile-n", type=int, default=None, help="default: --od-tile-n")
    ap.add_argument(
        "--gu-swiglu",
        action="store_true",
        help="fold SwiGLU into the fused GateUp GEMM's drain (needs --fused-gu)",
    )
    ap.add_argument(
        "--rgr-tiling",
        default="2,2",
        help="runtime_loop_tiling_sizes of fused rms_gemms_rope",
    )
    ap.add_argument(
        "--offn-tiling", default="2,2", help="runtime_loop_tiling_sizes of fused o_ffn"
    )
    ap.add_argument(
        "--offn-elf",
        default="",
        help="replace the compiled o_ffn.elf with this prebuilt ELF",
    )
    ap.add_argument(
        "--rgr-elf",
        default="",
        help="replace the compiled rms_gemms_rope.elf with this prebuilt ELF",
    )
    ap.add_argument(
        "--npu-attn",
        action="store_true",
        help="masked FlashAttention as a third ELF instead of CPU attention "
        "(needs --fused-qkv)",
    )
    ap.add_argument(
        "--fused-layer",
        action="store_true",
        help="whole layer (RMS+QKV+RoPE, FA, O+FFN) as ONE stitched ELF "
        "(needs --fused-qkv --fused-gu --gu-swiglu --npu-attn)",
    )
    ap.add_argument(
        "--bfp16",
        default="",
        help="comma list of GEMMs (qkv,o,gu,dn) taking bfp16ebs8 weights; "
        "optional tiles as key:tile_n:tile_k_l2:tile_k_l1[:columns] (o/gu/dn: N spread "
        "over that many array columns)",
    )
    ap.add_argument(
        "--layers-per-call",
        type=int,
        default=1,
        help="with --fused-layer: stitch this many layers into one ELF (one XRT run)",
    )
    ap.add_argument(
        "--fa-opt",
        default="-O2",
        help="Peano optimization level for attn_npu2.o (e.g. -Os)",
    )
    ap.add_argument(
        "--fa-his",
        type=int,
        default=1,
        help="FA heads_in_segment: heads looped inside the segment per launch iteration",
    )
    ap.add_argument(
        "--fa-qb",
        action="store_true",
        help="FA q_bcast: Q on its own per-column channel",
    )
    ap.add_argument(
        "--qkv-engine",
        action="store_true",
        help="with --offn-engine: rms+QKV+RoPE as one gemm_engine launch too (3-launch layer)",
    )
    ap.add_argument(
        "--offn-engine",
        action="store_true",
        help="with --fused-layer: O+FFN as one gemm_engine launch instead of six",
    )
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument(
        "--save-out", default="", help="np.save every layer's NPU output (float32) here"
    )
    ap.add_argument(
        "--dup",
        default="",
        help="timing probe: comma list of o_ffn slice prefixes to run twice "
        "(og,ra,rm,gu,sw,dg,fa)",
    )
    args = ap.parse_args()
    offn_dup = tuple(p for p in args.dup.split(",") if p)
    o_bst = args.od_bstationary or args.o_bstationary
    dn_bst = args.od_bstationary or args.dn_bstationary

    hm_tag = f"_hm{args.herd_m}" if args.herd_m else ""
    od_tag = (f"o{args.od_tile_n}" if o_bst else "") + (
        f"dn{args.dn_tile_m}x{args.dn_tile_n or args.od_tile_n}hm{args.dn_herd_m or 'a'}"
        if dn_bst
        else ""
    )
    dup_tag = f"dup{'-'.join(offn_dup)}" if offn_dup else ""
    assert not args.gu_swiglu or args.fused_gu, "--gu-swiglu needs --fused-gu"
    assert not args.npu_attn or args.fused_qkv, "--npu-attn needs --fused-qkv"
    assert not args.fused_layer or (
        args.fused_qkv and args.fused_gu and args.gu_swiglu and args.npu_attn
    ), "--fused-layer needs --fused-qkv --fused-gu --gu-swiglu --npu-attn"
    for spec in filter(None, args.bfp16.split(",")):
        k, *tiles = spec.split(":")
        assert k in _BFP16_TILES, f"--bfp16: unknown GEMM {k}"
        _BFP16[k] = tuple(map(int, tiles)) if tiles else _BFP16_TILES[k]
    assert "qkv" not in _BFP16 or args.fused_qkv, "--bfp16 qkv needs --fused-qkv"
    assert len(_BFP16.get("qkv", ())) <= 3, "--bfp16 qkv takes no column count"
    assert (
        not (_BFP16.keys() - {"qkv"}) or args.fused_gu
    ), "--bfp16 o/gu/dn needs --fused-gu"
    lpc = args.layers_per_call
    assert lpc == 1 or (
        args.fused_layer and args.layers % lpc == 0
    ), "--layers-per-call needs --fused-layer and must divide --layers"
    sw_half = (_BFP16["gu"][0] if "gu" in _BFP16 else args.gu_tile_n) // 2
    bfp_tag = "".join(f"_b{k}{'x'.join(map(str, v))}" for k, v in _BFP16.items())
    if args.fa_opt != "-O2":
        bfp_tag += f"_fa{args.fa_opt.lstrip('-')}"
    if args.fa_his > 1:
        bfp_tag += f"_his{args.fa_his}"
    if args.fa_qb:
        bfp_tag += "_qb"
    assert not args.offn_engine or (
        args.fused_layer and lpc == 1
    ), "--offn-engine needs --fused-layer, 1 layer/call"
    _ENGINE["on"] = args.offn_engine
    _ENGINE["qkv"] = args.qkv_engine
    assert not args.qkv_engine or args.offn_engine, "--qkv-engine needs --offn-engine"
    if args.offn_engine:
        bfp_tag += "_offneng"
    if args.qkv_engine:
        bfp_tag += "_qkveng"
    sw_tag = "sw" if args.gu_swiglu else ""
    gu_tag = (
        f"_fgu{args.gu_tile_n}{'bst' if args.gu_bstationary else ''}{sw_tag}{od_tag}{dup_tag}"
        if args.fused_gu
        else ""
    )
    qkv_tag = (
        f"_fqkv{args.qkv_tile_n}{'bst' if args.qkv_bstationary else ''}"
        if args.fused_qkv
        else ""
    )
    _TILING["rgr"] = [int(t) for t in args.rgr_tiling.split(",")]
    _TILING["offn"] = [int(t) for t in args.offn_tiling.split(",")]
    tl_tag = "".join(
        f"_{k}t{'x'.join(map(str, v))}" for k, v in _TILING.items() if v != [2, 2]
    )
    cache_dir = str(
        Path(__file__).resolve().parent
        / "build"
        / f"backbone_npu_cache{hm_tag}{gu_tag}{qkv_tag}{tl_tag}{bfp_tag}"
    )
    from air_examples.llms.shared.infra.cache import KernelCache, Profiler

    cache = KernelCache(cache_dir, verbose=False, profiler=Profiler(enabled=True))
    print(
        f"Compiling kernels (seq_len=256, herd_m={args.herd_m or 'auto'}, "
        f"fused_gu={args.fused_gu}, fused_qkv={args.fused_qkv}) -> {cache_dir}"
    )
    compile_backbone_kernels(
        cache,
        BACKBONE_CONFIG,
        SEQ_PAD,
        herd_m_override=args.herd_m,
        fused_gu=args.fused_gu,
        gu_tile_n=args.gu_tile_n,
        fused_qkv=args.fused_qkv,
        qkv_tile_n=args.qkv_tile_n,
        gu_bstationary=args.gu_bstationary,
        qkv_bstationary=args.qkv_bstationary,
        od_tile_n=args.od_tile_n,
        o_bstationary=o_bst,
        dn_bstationary=dn_bst,
        dn_herd_m=args.dn_herd_m,
        dn_tile_m=args.dn_tile_m,
        dn_tile_n=args.dn_tile_n,
        offn_dup=offn_dup,
        gu_swiglu=args.gu_swiglu,
        npu_attn=args.npu_attn,
        fused_layer=args.fused_layer,
        fa_opt=args.fa_opt,
        layers_per_call=args.layers_per_call,
        fa_his=args.fa_his,
        fa_qb=args.fa_qb,
        offn_engine=args.offn_engine,
        qkv_engine=args.qkv_engine,
    )
    for name, override in (("o_ffn", args.offn_elf), ("rms_gemms_rope", args.rgr_elf)):
        if override:
            import shutil

            shutil.copy2(override, cache.artifacts[name].output_binary)
            print(f"  {name}: using prebuilt {override}")
    if args.compile_only:
        return

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval()
    print("Extracting real backbone weights...")
    weights = extract_backbone_weights(policy)

    print(f"Capturing real per-layer I/O for {args.layers} layer(s)...")
    per_layer, real_outputs = capture_real_backbone_io(policy, args.layers)

    layer0 = per_layer[0]
    print(
        f"hidden_in: {layer0['hidden_in'].shape}, position_ids: {layer0['position_ids'].shape}, "
        f"mask: {layer0['attention_mask'].shape}"
    )

    prefill.attention_reference = _patched_attention_reference

    mask_padded = pad_mask(layer0["attention_mask"].astype(bool), SEQ_PAD)
    _MASK_HOLDER["mask"] = mask_padded
    if args.npu_attn:
        _NPU_ATTN["mask"] = additive_attn_mask(mask_padded)

    x = pad_seq(layer0["hidden_in"].astype(bfloat16), SEQ_PAD)
    rope_lut_real = build_rope_lut_gathered(layer0["position_ids"], BACKBONE_CONFIG)
    # Pad positions: repeat the last real position id's LUT row for padding rows
    # (their output is discarded, but rms_gemms_rope indexes rope_lut[:seq_len]
    # contiguously so it needs SEQ_PAD rows).
    rope_lut_padded = pad_seq(rope_lut_real, SEQ_PAD, fill=0.0)
    rope_lut_padded[SEQ_REAL:] = rope_lut_real[-1]

    def run_one_layer(xx, i, verbose=False):
        if args.qkv_engine:
            return run_layer_engine(
                xx,
                weights.layers[i],
                rope_lut_padded,
                BACKBONE_CONFIG,
                cache,
                layer_idx=i,
            )
        if args.fused_layer:
            return run_layer_fused(
                xx,
                weights.layers[i],
                rope_lut_padded,
                BACKBONE_CONFIG,
                cache,
                layer_idx=i,
                gu_swiglu_half=sw_half,
            )
        if args.fused_gu or args.fused_qkv:
            return run_transformer_block_custom(
                xx,
                weights.layers[i],
                rope_lut_padded,
                BACKBONE_CONFIG,
                cache,
                layer_idx=i,
                fused_qkv=args.fused_qkv,
                fused_gu=args.fused_gu,
                gu_swiglu_half=sw_half if args.gu_swiglu else 0,
            )
        out, _inter = prefill.run_transformer_block(
            xx,
            weights.layers[i],
            rope_lut_padded,
            BACKBONE_CONFIG,
            cache,
            layer_idx=i,
            cpu_attn=True,
            verbose=verbose,
        )
        return out

    def _cos(a, b):
        a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    # run_one_layer returns views into shared_nonstatic BOs that the next call
    # overwrites, so every kept output must be copied.
    npu_outputs, npu_isolated = {}, {}
    for i in range(args.layers):
        out = np.array(run_one_layer(x, i, verbose=(i == 0)), dtype=np.float32)
        npu_outputs[i] = out
        x = out.astype(bfloat16)
    for i in range(args.layers):
        xi = pad_seq(per_layer[i]["hidden_in"].astype(bfloat16), SEQ_PAD)
        npu_isolated[i] = np.array(run_one_layer(xi, i), dtype=np.float32)

    def run_group(xx, g):
        return run_layers_fused(
            xx,
            g * lpc,
            lpc,
            weights.layers,
            rope_lut_padded,
            BACKBONE_CONFIG,
            cache,
            gu_swiglu_half=sw_half,
        )

    if lpc > 1:
        xg = pad_seq(layer0["hidden_in"].astype(bfloat16), SEQ_PAD)
        for g in range(args.layers // lpc):
            out = np.array(run_group(xg, g), dtype=np.float32)
            last = (g + 1) * lpc - 1
            print(
                f"  {lpc}-layer call {g}: L{last} bit-identical to per-layer ELF: "
                f"{np.array_equal(out, npu_outputs[last])}  cos {_cos(out, npu_outputs[last]):.6f}"
            )
            xg = out.astype(bfloat16)

    if args.save_out:
        np.save(
            args.save_out,
            np.stack(
                [
                    np.stack([npu_outputs[i], npu_isolated[i]])
                    for i in range(args.layers)
                ]
            ),
        )

    print(
        "\nPer-layer cosine vs real lerobot backbone (chained = NPU output feeds next layer; "
        "isolated = real input per layer):"
    )
    for i in range(args.layers):
        real = real_outputs[i].astype(np.float32)
        print(
            f"  L{i:2d}  chained {_cos(npu_outputs[i][:SEQ_REAL], real):.6f}  "
            f"isolated {_cos(npu_isolated[i][:SEQ_REAL], real):.6f}"
        )
    npu0 = npu_outputs[0][:SEQ_REAL]
    real0 = real_outputs[0].astype(np.float32)
    print(
        f"\nLayer 0 cosine (NPU-path vs real lerobot backbone): {_cos(npu0, real0):.6f}"
    )
    print(
        f"  npu0 mean/std: {npu0.mean():.4f}/{npu0.std():.4f}  real0 mean/std: {real0.mean():.4f}/{real0.std():.4f}"
    )

    if args.profile:
        print(f"\nProfiling {args.layers} layers, {args.reps} reps...")
        x = pad_seq(layer0["hidden_in"].astype(bfloat16), SEQ_PAD)
        cache.profiler.kernel_breakdowns.clear()
        cache.profiler.cpu_times.clear()
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            xx = x
            if lpc > 1:
                for g in range(args.layers // lpc):
                    xx = run_group(xx, g)
            else:
                for i in range(args.layers):
                    xx = run_one_layer(xx, i)
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        print(
            f"{args.layers}-layer NPU (npu_attn={args.npu_attn}) wall: median {times[len(times)//2]:.2f} ms, "
            f"min {times[0]:.2f}, max {times[-1]:.2f}"
        )
        print("CPU baseline: full fill 43.9 ms")

        print("\n--- Per-ELF breakdown (avg per invocation, all reps) ---")
        for name, entries in sorted(cache.profiler.kernel_breakdowns.items()):
            n = len(entries)
            avg_w = sum(e["write_ms"] for e in entries) / n
            avg_k = sum(e["kernel_ms"] for e in entries) / n
            avg_r = sum(e["read_ms"] for e in entries) / n
            print(
                f"  {name:20s} write={avg_w:7.3f}ms  device={avg_k:7.3f}ms  read={avg_r:7.3f}ms"
                f"  total={avg_w+avg_k+avg_r:7.3f}ms  (x{n} calls)"
            )
            ks = sorted(e["kernel_ms"] for e in entries)
            print(
                f"  {'':20s} device min {ks[0]:.3f}  p10 {ks[n // 10]:.3f}  median {ks[n // 2]:.3f}  "
                f"p90 {ks[9 * n // 10]:.3f}  max {ks[-1]:.3f}"
            )
            if name.startswith("layers"):
                print(
                    f"  {'':20s} device per layer = {avg_k / lpc:7.3f}ms  (median {ks[n // 2] / lpc:.3f})"
                )
        if cache.profiler.cpu_times:
            print("\n--- CPU-side ops (attention fallback etc.) ---")
            for name, ts in sorted(cache.profiler.cpu_times.items()):
                print(f"  {name:20s} avg={sum(ts)/len(ts)*1000:7.3f}ms  (x{len(ts)})")


if __name__ == "__main__":
    main()
