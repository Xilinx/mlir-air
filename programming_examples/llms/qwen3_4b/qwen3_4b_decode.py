# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-4B Decode on MLIR-AIR (NPU2).

Single-token autoregressive generation with KV cache. Mirrors the Phase-2
prefill (qwen3_4b_prefill.py) deltas, applied to the M=1 GEMV path:

  1. QKV BIAS on host (Qwen2 family, attention_bias=True; NOT QK-norm — that
     is Qwen3). After the Q/K/V GEMV projection, a per-channel bias is added on
     the HOST: q=(q_dim,)+bq, k=(kv_dim,)+bk, v=(kv_dim,)+bv, BEFORE RoPE for
     q/k (HF Qwen2 order: proj -> +bias -> RoPE(Q,K) -> attention; V is
     bias-added and used directly). RoPE linearity is irrelevant here — we just
     do the cheap single-token elementwise add between the (bias-free) GEMV
     and the standalone RoPE ELFs.

  2. Dims: emb=2560, q_dim=4096 (32*128), kv_dim=1024 (8*128),
     hidden=9728, head_dim=128. o_proj is DECOUPLED (q_dim=4096 != emb=2560).
     The fused llama/qwen3 `o_gemv_ffn` cascade (`matvec_swiglu_rms`)
     double-buffers the full reduction K in L1 and overflows 64 KB L1 at
     hidden=9728, so it is NOT used. Instead decode uses STANDALONE per-projection
     GEMV ELFs (the plain `matvec` builder, 3-arg A@B) for O / Gate / Up / Down,
     and does the residual add / FFN RMSNorm / SwiGLU on the host (single-token,
     cheap, exact). Down (K=9728) also runs on the NPU as its own standalone ELF
     (A staged in L2, B streamed L3->L1; the standalone is NOT blocked by the L1
     overflow the fused cascade hits).

  3. LM-head vocab = 151936. Per-partition GEMV broadcasts the K=emb input
     vector with a push_queue repeat_count ~= n_part/32 - 1, capped at [0:255]
     → n_part <= 8192. Use 19 partitions × 8192 (19*8192 = 155648 >= 151936;
     last partition zero-padded, logits truncated to vocab on host). Same as
     Qwen3-0.6B (shared vocab 151936).

Decode attention is CPU (decode_attention_cpu); single-token attention is
trivial on host (head_dim=128 FA risk is irrelevant at M=1).

Decode GEMV tile configs (M×K : tile_m m_input herd_m):
  Q     4096×2560 : 8 8 8
  K/V   1024×2560 : 8 8 8
  O     2560×4096 : 4 4 8   (DECOUPLED; tile_m=4 to fit L2 at K=4096)
  Gate/Up 9728×2560 : 8 8 8
  Down  2560×9728 : 2 2 8   (NPU standalone, omit_pingpong + own down_mv.o;
        m_input=2 NOT 1 so push_queue repeat=M/(herd_m*m_input)-1=159 <=255)

MEASURED (2026-06-27): with O/Gate/Up/Down on NPU, the residual + FFN RMSNorm +
SwiGLU left on host are ~0.045ms/layer of M=1 numpy (microbench at emb=2560/
hidden=9728). Moving them to a standalone NPU ELF would add a ~0.7-1.4ms dispatch
per op to replace ~0.02ms numpy — a large regression — and folding into the GEMV
epilogue is the matvec_swiglu_rms cascade that overflows L1 here. Glue stays on
host by design; the decode cost is now the NPU GEMVs (real compute).
"""

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_PROG_EXAMPLES = str(Path(__file__).resolve().parent.parent.parent)
if _PROG_EXAMPLES not in sys.path:
    sys.path.insert(0, _PROG_EXAMPLES)
_LLMS_DIR = str(Path(__file__).resolve().parent.parent)
if _LLMS_DIR not in sys.path:
    sys.path.insert(0, _LLMS_DIR)

from qwen3_4b_weights import LlamaConfig
from qwen3_4b_cpu_helpers import rms_norm


def build_rms_qkv_qknorm_rope_gemv_module(config):
    """Fused decode ELF: RMSNorm + Q/K/V GEMV + per-head QK-norm + RoPE (M=1).

    The GEMV builders are generic (not registry-coupled), so the shared decode
    builder works directly for qwen3_4b's dims (emb=2560)."""
    from shared.builders.rms_qkv_qknorm_rope_multi import (
        build_rms_qkv_qknorm_rope_gemv_module as _build,
    )

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    return _build(emb_dim, q_dim, kv_dim, n_heads, n_kv_heads, head_dim, qknorm_eps=1e-6)


def _rms_qkv_qknorm_rope_gemv_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_qknorm_rope_gemv",
    }


# LM-head decode partitioning. vocab=151936. Per-partition GEMV broadcasts the
# K=emb_dim input vector with a push_queue repeat_count ~= n_part/32 - 1, capped
# at [0:255] → n_part <= 8192. 19 * 8192 = 155648 >= 151936; the final partition
# carries zero-padded rows (logits truncated to vocab on host).
_LM_N_PARTITIONS = 19
_LM_N_PART = 8192  # % 64 == 0; n_part/32 - 1 = 255 (at the repeat-count limit)

# Decode GEMV tile configs (tile_m, m_input, herd_m). Qwen3-4B dims:
#   Q     M=4096 K=2560 ; K/V M=1024 K=2560 ; O M=2560 K=4096 (DECOUPLED) ;
#   Gate/Up M=9728 K=2560 ; Down M=2560 K=9728 (HOST).
# matvec L2 budget: A_l2 = tile_m*herd_m*K*2 bytes must be <= 512 KiB. With
# herd_m=8 that caps tile_m*K <= 32768. K=2560 -> tile_m<=12 (use 8). The O
# GEMV has the larger K=q_dim=4096 -> tile_m*4096<=32768 -> tile_m<=8, but 8
# lands exactly at the L2 ceiling (A=524288 + C overflows), so O uses tile_m=4.
_GEMV_QO = (8, 8, 8)        # Q proj (M=4096 K=2560)
_GEMV_KV = (8, 8, 8)        # K/V proj (M=1024 K=2560)
_GEMV_O = (4, 4, 8)         # O proj (M=2560 K=4096 DECOUPLED; tile_m=4 to fit L2, m_input=4 | tile_m)
_GEMV_GATEUP = (8, 8, 8)    # Gate/Up proj (M=9728 K=2560)
_GEMV_DOWN = (2, 2, 8)      # Down proj (M=2560 K=9728). m_input=2 (NOT 1): the B
                            # vector is streamed once per (launch × tile_m/m_input)
                            # and hoisted into one push_queue whose repeat_count =
                            # M/(herd_m*m_input)-1 must be <=255. m_input=1 gives
                            # 2560/8-1=319 (OVERFLOW); m_input=2 gives 159. Mirrors
                            # llama32_3b down (2,2,8). omit_pingpong keeps L1 in budget.


# ---------------------------------------------------------------------------
# Builder 1: standalone single-projection GEMV ELF (plain matvec, A@B).
#   Used for O / Gate / Up / Down — the residual / RMSNorm / SwiGLU around
#   them is done on the host (single token, cheap, exact). Avoids the cascade
#   builders' K % 512 == 0 requirement which qwen2.5 dims violate.
# ---------------------------------------------------------------------------


def build_gemv_module(m, k, tile_m, m_input, herd_m=8, name="gemv", link_with="mv.o"):
    """Standalone GEMV ELF: C[m] = A[m,k] @ B[k]. 3-arg func.

    Func args: %arg0 A (m,k)  %arg1 B (k,)  %arg2 C (m,)

    The raw `matvec` builder names its func @matvec_bf16; for ELF output the
    backend's instance_name must match the module's func name (else the loaded
    kernel symbol `main:<instance_name>` is not found). We stitch the single
    GEMV slice through stitch_elf so the public func is renamed to `name`,
    matching the per-projection instance_name in the backend kwargs.
    """
    _mv_dir = os.path.join(
        _PROG_EXAMPLES, "matrix_vector_multiplication", "bf16"
    )
    if _mv_dir not in sys.path:
        sys.path.insert(0, _mv_dir)
    from matvec import build_module as build_gemv
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg

    gemv_ir = str(build_gemv(m, k, tile_m, m_input, herd_m, bfloat16, bfloat16,
                             link_with=link_with))
    base_args = [
        FuncArg("%arg0", f"memref<{m}x{k}xbf16>"),
        FuncArg("%arg1", f"memref<{k}xbf16>"),
        FuncArg("%arg2", f"memref<{m}xbf16>"),
    ]
    # GEMV func args: {0: weight (MxK), 1: input (K,), 2: output (M,)}.
    slices = [
        KernelSlice(
            gemv_ir, "g", {0: 0, 1: 1, 2: 2},
            extern_syms={"@matvec_vectorized_bf16_bf16", "@linalg_fill_bf16"},
        )
    ]
    return stitch_elf(name, base_args, slices)


# ---------------------------------------------------------------------------
# Builder 2: LM-head GEMV (19 partitions, n_part=8192 for vocab 151936).
# ---------------------------------------------------------------------------


def build_lm_head_gemv_qwen_module(emb_dim):
    from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module

    return build_lm_head_gemv_module(
        emb_dim=emb_dim,
        n_partitions=_LM_N_PARTITIONS,
        n_part=_LM_N_PART,
        tile_m=8,
        m_input=4,
        herd_m=8,
    )


# ---------------------------------------------------------------------------
# Backend kwargs
# ---------------------------------------------------------------------------


def _gemv_backend(verbose=False, name="gemv", omit_pingpong=False):
    bk = {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
    }
    # Large-K Down GEMV (K=9728): the B vector is ~19 KB; ping-pong would
    # double-buffer it (4×19 KB > 64 KB L1) and AIE allocation fails. Disable
    # ping-pong so the standalone Down ELF fits L1 (mirrors qwen25_3b/llama32_3b).
    if omit_pingpong:
        bk["omit_pingpong"] = "all"
    return bk


def _lm_gemv_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "lm_head_gemv",
    }


# ---------------------------------------------------------------------------
# Decode kernel compilation
# ---------------------------------------------------------------------------


def compile_decode_kernels(cache, config, verbose=False):
    """Compile the Qwen3 decode kernels."""
    from shared.infra.external_kernels import compile_mv, compile_rope

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim

    print(f"\n{'='*60}\nCompiling Qwen3 decode kernels...\n{'='*60}\n")

    # External .o: GEMV (mv.o), RoPE (rope.o).
    compile_mv()
    compile_rope()

    print("\n--- rms_qkv_qknorm_rope_gemv (FUSED: RMSNorm+QKV+QK-norm+RoPE, 8 launches) ---")
    cache.compile_and_cache(
        "rms_qkv_qknorm_rope_gemv",
        build_rms_qkv_qknorm_rope_gemv_module(config),
        _rms_qkv_qknorm_rope_gemv_backend(verbose),
    )

    # Standalone projection GEMVs. O proj is DECOUPLED (M=emb, K=q_dim).
    print(f"\n--- o_gemv (O proj GEMV, {emb_dim}x{q_dim}) ---")
    o_tm, o_mi, o_hm = _GEMV_O
    cache.compile_and_cache(
        "o_gemv", build_gemv_module(emb_dim, q_dim, o_tm, o_mi, o_hm, name="o_gemv"),
        _gemv_backend(verbose, "o_gemv"),
    )
    print(f"\n--- gate_gemv (Gate proj GEMV, {hidden_dim}x{emb_dim}) ---")
    g_tm, g_mi, g_hm = _GEMV_GATEUP
    cache.compile_and_cache(
        "gate_gemv", build_gemv_module(hidden_dim, emb_dim, g_tm, g_mi, g_hm, name="gate_gemv"),
        _gemv_backend(verbose, "gate_gemv"),
    )
    print(f"\n--- up_gemv (Up proj GEMV, {hidden_dim}x{emb_dim}) ---")
    cache.compile_and_cache(
        "up_gemv", build_gemv_module(hidden_dim, emb_dim, g_tm, g_mi, g_hm, name="up_gemv"),
        _gemv_backend(verbose, "up_gemv"),
    )
    # Down proj GEMV (M=emb=2560, K=hidden=9728). The STANDALONE matvec stages A
    # in L2 and streams B L3->L1, so K=9728 compiles + runs (only the *fused*
    # swiglu cascade overflowed L1). Needs its own mv.o with DIM_M_OUTPUT=tile_m
    # (the shared mv.o uses 8); link down_mv.o. omit_pingpong so the large B fits.
    import shutil as _shutil
    d_tm, d_mi, d_hm = _GEMV_DOWN
    compile_mv(tile_m=d_tm)            # writes mv.o with DIM_M_OUTPUT=d_tm
    _shutil.copy2("mv.o", "down_mv.o")
    compile_mv()                       # restore mv.o (DIM_M_OUTPUT=8) for o/gate/up
    print(f"\n--- down_gemv (Down proj GEMV, {emb_dim}x{hidden_dim}) ---")
    cache.compile_and_cache(
        "down_gemv",
        build_gemv_module(emb_dim, hidden_dim, d_tm, d_mi, d_hm,
                          name="down_gemv", link_with="down_mv.o"),
        _gemv_backend(verbose, "down_gemv", omit_pingpong=True),
    )

    print(f"\n--- lm_head_gemv ({_LM_N_PARTITIONS}-partition, vocab 151936) ---")
    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_qwen_module(emb_dim),
        _lm_gemv_backend(verbose),
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


# ---------------------------------------------------------------------------
# CPU decode attention (with KV cache)
# ---------------------------------------------------------------------------


def decode_attention_cpu(q, k_cache, v_cache, current_pos, n_heads, n_kv_heads, head_dim):
    """Single-query GQA attention with KV cache.

    Args:
        q: (q_dim,) — RoPE'd (and bias-added) query vector for the current token.
        k_cache: (n_kv_heads, max_seq, head_dim) — cached keys (after bias+RoPE).
        v_cache: (n_kv_heads, max_seq, head_dim) — cached values (after bias).
        current_pos: current token position (0-indexed).
    Returns:
        attn_out: (q_dim,) bfloat16.
    """
    group_size = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    seq_len = current_pos + 1

    q_heads = q.astype(np.float32).reshape(n_heads, head_dim)
    k_cached = k_cache[:, :seq_len, :].astype(np.float32)
    v_cached = v_cache[:, :seq_len, :].astype(np.float32)

    out = np.zeros((n_heads, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_h = h // group_size
        scores = (q_heads[h] @ k_cached[kv_h].T) * scale
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        out[h] = probs @ v_cached[kv_h]

    return out.reshape(-1).astype(bfloat16)


# ---------------------------------------------------------------------------
# Single decode transformer block
# ---------------------------------------------------------------------------


def run_decode_block(
    x_bf16,
    layer_weights,
    cache,
    config,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    rope_lut_bf16,
    verbose=False,
):
    """Run one Qwen3 transformer block for a single decode token.

    Stages: rms_qkv_qknorm_rope_gemv (NPU: RMSNorm + Q/K/V GEMV + per-head
    QK-norm + RoPE) -> KV-cache write -> CPU attention -> O GEMV (NPU)
    -> residual (host) -> FFN RMSNorm (host) -> Gate/Up GEMV (NPU)
    -> SwiGLU (host) -> Down GEMV (NPU) -> FFN residual (host).
    """
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = getattr(layer_weights, "_layer_idx", None)
    _suffix = f"_L{layer_idx}" if layer_idx is not None else None
    inter = {}

    x_in = x_bf16.flatten().astype(bfloat16)

    # RoPE LUT for this position (position-dependent — NOT static).
    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]  # (1, head_dim)
    lut_q = np.tile(rope_lut_pos, (n_heads, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (n_kv_heads, 1)).flatten().astype(bfloat16)

    # --- One ELF = RMSNorm + Q/K/V GEMV + per-head QK-norm + RoPE ---
    res = cache.load_and_run(
        "rms_qkv_qknorm_rope_gemv",
        _rms_qkv_qknorm_rope_gemv_backend(verbose),
        x_in,                                              # 0 x_in
        layer_weights.attn_norm.reshape(emb_dim).astype(bfloat16),  # 1 norm_w (static)
        np.zeros(emb_dim, dtype=bfloat16),                 # 2 normed
        layer_weights._wq_t,                               # 3 wq (static)
        np.zeros(q_dim, dtype=bfloat16),                   # 4 q
        layer_weights._wk_t,                               # 5 wk (static)
        np.zeros(kv_dim, dtype=bfloat16),                  # 6 k
        layer_weights._wv_t,                               # 7 wv (static)
        np.zeros(kv_dim, dtype=bfloat16),                  # 8 v
        np.asarray(layer_weights.q_norm, bfloat16).reshape(head_dim),  # 9 q_norm (static)
        np.asarray(layer_weights.k_norm, bfloat16).reshape(head_dim),  # 10 k_norm (static)
        np.zeros(q_dim, dtype=bfloat16),                   # 11 q_n
        np.zeros(kv_dim, dtype=bfloat16),                  # 12 k_n
        lut_q,                                             # 13 lut_q (DYNAMIC)
        lut_k,                                             # 14 lut_k (DYNAMIC)
        np.zeros(q_dim, dtype=bfloat16),                   # 15 q_roped
        np.zeros(kv_dim, dtype=bfloat16),                  # 16 k_roped
        output_indices=[8, 15, 16],
        static_input_indices={1, 3, 5, 7, 9, 10},
        intermediate_indices={2, 4, 6, 8, 11, 12, 15, 16},
        bo_key=f"rms_qkv_qknorm_rope_gemv{_suffix}" if _suffix else None,
    )
    v = res[8].astype(bfloat16)
    q_roped = res[15].astype(bfloat16)
    k_roped = res[16].astype(bfloat16)
    inter["v"] = v
    inter["q_roped"] = q_roped
    inter["k_roped"] = k_roped

    # --- Update KV cache (K after qk-norm+RoPE, V raw projection) ---
    k_cache_layer[:, current_pos, :] = k_roped.reshape(n_kv_heads, head_dim)
    v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

    # --- CPU attention ---
    with cache.profiler.time_cpu("decode_attention_cpu"):
        attn_out = decode_attention_cpu(
            q_roped, k_cache_layer, v_cache_layer, current_pos,
            n_heads, n_kv_heads, head_dim,
        )
    inter["attn_out"] = attn_out

    # --- Stage E: O proj GEMV + residual + FFN (RMSNorm/SwiGLU on host) ---
    ro = cache.load_and_run(
        "o_gemv",
        _gemv_backend(verbose, "o_gemv"),
        layer_weights._wo_t,                        # arg0 wo (static) (emb, q_dim)
        np.ascontiguousarray(attn_out),             # arg1 attn_out (q_dim,)
        np.zeros(emb_dim, dtype=bfloat16),          # arg2 proj (emb,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"o_gemv{_suffix}" if _suffix else None,
    )
    proj = ro[2].astype(np.float32)

    # residual 1 (host)
    res1 = proj + x_in.astype(np.float32)
    inter["res1"] = res1.astype(bfloat16)

    # FFN RMSNorm (host, eps=1e-6)
    normed2 = rms_norm(res1.reshape(1, emb_dim), layer_weights.ffn_norm, eps=1e-6).reshape(emb_dim).astype(bfloat16)

    rg = cache.load_and_run(
        "gate_gemv",
        _gemv_backend(verbose, "gate_gemv"),
        layer_weights._wgate_t,                     # arg0 w_gate (static) (hidden, emb)
        np.ascontiguousarray(normed2),              # arg1 normed2 (emb,)
        np.zeros(hidden_dim, dtype=bfloat16),       # arg2 gate (hidden,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"gate_gemv{_suffix}" if _suffix else None,
    )
    gate = rg[2].astype(np.float32)
    ru = cache.load_and_run(
        "up_gemv",
        _gemv_backend(verbose, "up_gemv"),
        layer_weights._wup_t,                       # arg0 w_up (static) (hidden, emb)
        np.ascontiguousarray(normed2),              # arg1 normed2 (emb,)
        np.zeros(hidden_dim, dtype=bfloat16),       # arg2 up (hidden,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"up_gemv{_suffix}" if _suffix else None,
    )
    up = ru[2].astype(np.float32)
    # SwiGLU (host)
    swig = ((gate / (1.0 + np.exp(-gate))) * up).astype(bfloat16)
    # Down projection: M=emb=2560, K=hidden=9728. The STANDALONE matvec stages
    # A in L2 and streams B L3->L1, so K=9728 compiles + runs on NPU (only the
    # *fused* swiglu cascade overflowed L1). Runs as its own down_gemv ELF,
    # like o/gate/up.
    rd = cache.load_and_run(
        "down_gemv",
        _gemv_backend(verbose, "down_gemv", omit_pingpong=True),
        layer_weights._wdown_t,                     # arg0 w_down (static) (emb, hidden)
        np.ascontiguousarray(swig),                 # arg1 swig (hidden,)
        np.zeros(emb_dim, dtype=bfloat16),          # arg2 down (emb,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"down_gemv{_suffix}" if _suffix else None,
    )
    down = rd[2].astype(np.float32)

    out = (down + res1).astype(bfloat16)
    inter["ffn_out"] = out
    return out, inter
