# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LLAMA-3.2-3B Decode on MLIR-AIR (NPU2).

Pure-Llama decode. Identical kernel sequence to Llama-3.2-1B EXCEPT for one
hardware constraint at emb_dim=3072:

  The fused decode `o_gemv_ffn` ELF's SwiGLU stage (`matvec_swiglu_rms`) holds
  the full reduction dimension K=emb_dim in an L1 buffer and streams it through
  the L2->L1 cascade. At K=emb_dim<=2048 (every prior model) this compiles; at
  K=emb_dim=3072 the lowered `aie.dma_bd` exceeds the AIE2P BD size limit
  ("Size may not exceed 1023") and aircc rejects the ELF. This is a kernel-
  builder limitation (the K-dimension DMAs are not tiled below the BD ceiling),
  independent of the model wiring.

  The FUSED cascade stays off-NPU (BD-1023 limit). But each projection GEMV
  ALONE is not blocked by BD-1023, so the split is:
    - rms_gemv_rope (RMSNorm + Q/K/V GEMV + RoPE Q/K, hd=128) -> NPU
    - attention with KV cache -> CPU (as in llama32_1b decode)
    - O-proj GEMV (M=emb=3072, K=q_dim=3072) -> NPU (standalone ELF)
    - residual + FFN RMSNorm + SwiGLU -> CPU (cheap single-token glue)
    - Gate/Up GEMV (M=hidden=8192, K=emb=3072) -> NPU (standalone ELF)
    - Down GEMV (M=emb=3072, K=hidden=8192) -> NPU (standalone ELF)
    - LM-head GEMV (vocab 128256, 8 partitions) -> NPU

  O / Gate / Up / Down each run as standalone GEMV ELFs (A staged in L2, B
  streamed L3->L1), mirroring qwen25_3b decode. Only the FUSED swiglu_rms cascade
  overflowed the BD-1023 limit; the standalone per-projection matvec does not. The
  residual add / FFN RMSNorm / SwiGLU stay on host (single-token, cheap, exact),
  computed bit-for-bit identically to the FFN reference so Phase 3 stays correct.

  This keeps Phase 3 numerically correct (the NPU GEMVs are fp32-accumulate,
  matching the f32 ref bit-closely) and exercises the real decode loop + KV cache
  + NPU Q/K/V + NPU O/Gate/Up/Down + NPU LM-head.

  MEASURED (profile N_TOKENS=8): with O/Gate/Up/Down on NPU, the
  `decode_o_ffn_cpu` profiler block reads ~5.0ms/layer, but that block WRAPS the
  four NPU GEMV calls (o 0.72 + gate 1.38 + up 1.36 + down 1.42 ≈ 4.9ms NPU run
  time). The PURE host glue inside it — FFN RMSNorm + SwiGLU + residual add at
  M=1 — is only ~0.045ms/layer (RMSNorm 0.023 + SwiGLU 0.021 + residual 0.001,
  numpy microbench at emb=3072/hidden=8192). Moving that glue to a standalone
  NPU ELF would ADD a ~0.7–1.4ms dispatch to replace ~0.02ms of numpy — a 30–70×
  REGRESSION — and folding it into the GEMV epilogue is exactly the
  matvec_swiglu_rms cascade that overflows L1 at emb=3072 (the reason for this
  split). So the remaining glue stays on host BY DESIGN; the decode bottleneck is
  now the NPU GEMVs themselves (legitimate compute), not host glue.
"""

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_PROG = _LLMS_DIR.parent
_LLAMA1B = _LLMS_DIR / "llama32_1b"
for _p in (str(_PROG), str(_LLMS_DIR), str(_LLAMA1B), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.infra.backend_presets import RGR_BACKEND, LM_GEMV_BACKEND

# Reuse the CPU decode attention from llama32_1b verbatim (head_dim agnostic).
from llama32_1b_decode import decode_attention_cpu  # noqa: F401

# Down proj GEMV on NPU. The Down GEMV (M=emb=3072, K=hidden=8192) compiles +
# runs as a STANDALONE matvec (A staged in L2, B streamed L3→L1; verified
# bit-identical to f32 ref at tile_m=2/m_input=2/herd_m=8). The fused o_gemv_ffn
# cascade's SwiGLU stage held the full K=emb in L1 and overflowed the AIE2P BD
# size limit at emb=3072 — that is why O+FFN runs as standalone GEMVs rather than
# the fused cascade.
_GEMV_DOWN = (2, 2, 8)  # 3072×8192 (Down proj; K=8192 → L2 caps tile_m=2)

# O / Gate / Up proj GEMVs on NPU. Standalone matvec ELFs (A in L2, B streamed
# L3->L1), mirroring qwen25_3b. The residual / FFN RMSNorm / SwiGLU glue stays on
# host (single-token, cheap, exact, bit-for-bit with the FFN reference).
# Tile configs (tile_m, m_input, herd_m). O proj is SQUARE (M=emb=3072, K=q_dim
# =3072). Gate/Up are M=hidden=8192, K=emb=3072. matvec L2 budget:
# tile_m*herd_m*K*2 <= 512 KiB; herd_m=8 -> tile_m*K <= 32768. K=3072 -> tile_m<=10.
_GEMV_O = (8, 8, 8)  # O proj (M=3072 K=3072)
_GEMV_GATEUP = (8, 8, 8)  # Gate/Up proj (M=8192 K=3072)


def _gemv_backend(verbose=False, name="gemv", omit_pingpong=False):
    bk = {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": name,
    }
    # Large-K Down GEMV (K=8192): the B vector is 16 KB; ping-pong double-buffers
    # it (4×16 KB = 64 KB, plus stack > 64 KB L1) and AIE allocation fails.
    # Disable ping-pong so the standalone Down ELF fits L1 (same reason the fused
    # OGF turns ping-pong off for its K=8192 down).
    if omit_pingpong:
        bk["omit_pingpong"] = "all"
    return bk


def build_gemv_module(m, k, tile_m, m_input, herd_m=8, name="gemv", link_with="mv.o"):
    """Standalone GEMV ELF: C[m] = A[m,k] @ B[k]. 3-arg func (mirrors qwen25_3b).

    Func args: %arg0 A (m,k)  %arg1 B (k,)  %arg2 C (m,). Stitched through
    stitch_elf so the public func is renamed to `name`, matching the
    instance_name in the backend kwargs.
    """
    _mv_dir = os.path.join(_PROG, "matrix_vector_multiplication", "bf16")
    if _mv_dir not in sys.path:
        sys.path.insert(0, _mv_dir)
    from matvec import build_module as build_gemv
    from shared.infra.stitching import stitch_elf, KernelSlice, FuncArg

    gemv_ir = str(
        build_gemv(
            m, k, tile_m, m_input, herd_m, bfloat16, bfloat16, link_with=link_with
        )
    )
    base_args = [
        FuncArg("%arg0", f"memref<{m}x{k}xbf16>"),
        FuncArg("%arg1", f"memref<{k}xbf16>"),
        FuncArg("%arg2", f"memref<{m}xbf16>"),
    ]
    slices = [
        KernelSlice(
            gemv_ir,
            "g",
            {0: 0, 1: 1, 2: 2},
            extern_syms={"@matvec_vectorized_bf16_bf16", "@linalg_fill_bf16"},
        )
    ]
    return stitch_elf(name, base_args, slices)


def compile_decode_kernels(cache, config):
    """Compile the decode NPU kernels that DO build at emb_dim=3072:
    rms_gemv_rope (Q/K/V + RoPE) and lm_head_gemv. The fused o_gemv_ffn is
    intentionally NOT compiled (its SwiGLU stage overflows the BD limit at
    K=3072); that stage runs on CPU in run_decode_block."""
    from shared.infra.external_kernels import compile_all_external_kernels

    compile_all_external_kernels(head_dim=config.head_dim)

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}")
    print(
        "Compiling decode kernels (rms_gemv_rope + lm_head_gemv; "
        "o_gemv_ffn -> CPU at emb=3072)..."
    )
    print(f"{'='*60}\n")

    from shared.builders.rms_gemv_rope_multi import build_rms_gemv_rope_module

    cache.compile_and_cache(
        "rms_gemv_rope",
        build_rms_gemv_rope_module(emb_dim, kv_dim, n_heads, n_kv_heads, head_dim),
        {"verbose": cache.verbose, **RGR_BACKEND},
    )

    from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module

    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_module(emb_dim),
        {"verbose": cache.verbose, **LM_GEMV_BACKEND},
    )

    q_dim = n_heads * head_dim
    hidden_dim = config.hidden_dim

    # O proj GEMV (M=emb=3072, K=q_dim=3072). tile_m=8 -> shared mv.o
    # (DIM_M_OUTPUT=8) works; no dedicated .o needed.
    from shared.infra.external_kernels import compile_mv

    compile_mv()  # ensure shared mv.o is DIM_M_OUTPUT=8
    print(f"\n--- o_gemv (O proj GEMV, {emb_dim}x{q_dim}) ---")
    o_tm, o_mi, o_hm = _GEMV_O
    cache.compile_and_cache(
        "o_gemv",
        build_gemv_module(emb_dim, q_dim, o_tm, o_mi, o_hm, name="o_gemv"),
        _gemv_backend(cache.verbose, "o_gemv"),
    )
    print(f"\n--- gate_gemv (Gate proj GEMV, {hidden_dim}x{emb_dim}) ---")
    g_tm, g_mi, g_hm = _GEMV_GATEUP
    cache.compile_and_cache(
        "gate_gemv",
        build_gemv_module(hidden_dim, emb_dim, g_tm, g_mi, g_hm, name="gate_gemv"),
        _gemv_backend(cache.verbose, "gate_gemv"),
    )
    print(f"\n--- up_gemv (Up proj GEMV, {hidden_dim}x{emb_dim}) ---")
    cache.compile_and_cache(
        "up_gemv",
        build_gemv_module(hidden_dim, emb_dim, g_tm, g_mi, g_hm, name="up_gemv"),
        _gemv_backend(cache.verbose, "up_gemv"),
    )

    # Standalone Down GEMV (M=emb=3072, K=hidden=8192). Needs its own mv.o
    # with DIM_M_OUTPUT=tile_m (the shared mv.o uses 8); link down_mv.o.
    import shutil as _shutil

    d_tm, d_mi, d_hm = _GEMV_DOWN
    compile_mv(tile_m=d_tm)
    _shutil.copy2("mv.o", "down_mv.o")
    compile_mv()  # restore mv.o (DIM_M_OUTPUT=8) for the other external kernels
    print(f"\n--- down_gemv (Down proj GEMV, {emb_dim}x{hidden_dim}) ---")
    cache.compile_and_cache(
        "down_gemv",
        build_gemv_module(
            emb_dim,
            hidden_dim,
            d_tm,
            d_mi,
            d_hm,
            name="down_gemv",
            link_with="down_mv.o",
        ),
        _gemv_backend(cache.verbose, "down_gemv", omit_pingpong=True),
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} decode kernels compiled.")


def _o_ffn_cpu(layer_weights, attn_out, x_residual, config, cache, verbose=False):
    """O-proj + residual + FFN(SwiGLU) + down + residual for one decode token.

    O / Gate / Up / Down projections run on NPU as standalone GEMV ELFs (the big
    decode win). The residual add / FFN RMSNorm / SwiGLU stay on host —
    single-token, cheap, and computed bit-for-bit identically to the FFN reference
    so Phase 3 cannot regress. The NPU GEMVs are bf16-input / fp32-accumulate,
    matching the f32 ref bit-closely.
    Weights:
      _wo_t   : (emb, q_dim)   _wgate_t/_wup_t: (hidden, emb)
      _wdown_t: (emb, hidden)  ffn_norm: (emb,)
    Returns (emb,) bf16.
    """
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim

    layer_idx = getattr(layer_weights, "_layer_idx", None)
    _suffix = f"_L{layer_idx}" if layer_idx is not None else None

    # --- O proj (NPU GEMV): res1 = wo @ attn_out + x_residual ---
    ro = cache.load_and_run(
        "o_gemv",
        _gemv_backend(verbose, "o_gemv"),
        layer_weights._wo_t,  # arg0 wo (static) (emb, q_dim)
        np.ascontiguousarray(attn_out).astype(bfloat16),  # arg1 attn_out (q_dim,)
        np.zeros(emb_dim, dtype=bfloat16),  # arg2 proj (emb,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"o_gemv{_suffix}" if _suffix else None,
    )
    proj = ro[2].astype(np.float32)
    res1 = proj + np.asarray(x_residual, dtype=np.float32)

    # --- FFN RMSNorm (host glue, eps=1e-5; bit-for-bit with the reference) ---
    rstd = 1.0 / np.sqrt((res1 * res1).mean() + 1e-5)
    ffn_norm_w = np.asarray(layer_weights.ffn_norm, dtype=np.float32).reshape(emb_dim)
    normed = (res1 * rstd) * ffn_norm_w
    normed_bf16 = normed.astype(bfloat16)

    # --- Gate / Up proj (NPU GEMV) ---
    rg = cache.load_and_run(
        "gate_gemv",
        _gemv_backend(verbose, "gate_gemv"),
        layer_weights._wgate_t,  # arg0 w_gate (static) (hidden, emb)
        np.ascontiguousarray(normed_bf16),  # arg1 normed (emb,)
        np.zeros(hidden_dim, dtype=bfloat16),  # arg2 gate (hidden,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"gate_gemv{_suffix}" if _suffix else None,
    )
    gate = rg[2].astype(np.float32)
    ru = cache.load_and_run(
        "up_gemv",
        _gemv_backend(verbose, "up_gemv"),
        layer_weights._wup_t,  # arg0 w_up (static) (hidden, emb)
        np.ascontiguousarray(normed_bf16),  # arg1 normed (emb,)
        np.zeros(hidden_dim, dtype=bfloat16),  # arg2 up (hidden,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"up_gemv{_suffix}" if _suffix else None,
    )
    up = ru[2].astype(np.float32)

    # --- SwiGLU (host glue) ---
    swiglu = (gate * 0.5 * (np.tanh(gate / 2.0) + 1.0)) * up
    swiglu_bf16 = swiglu.astype(bfloat16)

    # --- Down proj (NPU GEMV): down = _wdown_t (emb, hidden) @ swiglu (hidden,) ---
    rd = cache.load_and_run(
        "down_gemv",
        _gemv_backend(verbose, "down_gemv", omit_pingpong=True),
        layer_weights._wdown_t,  # arg0 w_down (static) (emb, hidden)
        np.ascontiguousarray(swiglu_bf16),  # arg1 swiglu (hidden,)
        np.zeros(emb_dim, dtype=bfloat16),  # arg2 down (emb,)
        output_indices=[2],
        static_input_indices={0},
        intermediate_indices={2},
        bo_key=f"down_gemv{_suffix}" if _suffix else None,
    )
    down = rd[2].astype(np.float32)
    return (down + res1).astype(bfloat16)


def run_decode_block(
    x_bf16,
    layer_weights,
    cache,
    config,
    k_cache_layer,
    v_cache_layer,
    current_pos,
    rope_lut_bf16,
):
    """One decode transformer block: NPU rms_gemv_rope -> CPU attention ->
    CPU O+FFN -> output. Returns (emb,) bf16."""
    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    kv_dim = n_kv_heads * head_dim

    layer_idx = getattr(layer_weights, "_layer_idx", None)

    # --- Call 1: rms_gemv_rope (NPU) : RMSNorm + Q/K/V GEMV + RoPE Q/K ---
    x_in = x_bf16.flatten().astype(bfloat16)
    w_norm = layer_weights.attn_norm.reshape(emb_dim).astype(bfloat16)
    normed_buf = np.zeros(emb_dim, dtype=bfloat16)
    wq = layer_weights._wq_t
    q_buf = np.zeros(emb_dim, dtype=bfloat16)
    wk = layer_weights._wk_t
    k_buf = np.zeros(kv_dim, dtype=bfloat16)
    wv = layer_weights._wv_t
    v_buf = np.zeros(kv_dim, dtype=bfloat16)

    rope_lut_pos = rope_lut_bf16[current_pos : current_pos + 1]
    lut_q = np.tile(rope_lut_pos, (n_heads, 1)).flatten().astype(bfloat16)
    lut_k = np.tile(rope_lut_pos, (n_kv_heads, 1)).flatten().astype(bfloat16)
    q_roped_buf = np.zeros(emb_dim, dtype=bfloat16)
    k_roped_buf = np.zeros(kv_dim, dtype=bfloat16)

    bk = f"rms_gemv_rope_L{layer_idx}" if layer_idx is not None else None
    results = cache.load_and_run(
        "rms_gemv_rope",
        RGR_BACKEND,
        x_in,
        w_norm,
        normed_buf,
        wq,
        q_buf,
        wk,
        k_buf,
        wv,
        v_buf,
        lut_q,
        lut_k,
        q_roped_buf,
        k_roped_buf,
        output_indices=[8, 11, 12],
        static_input_indices={1, 3, 5, 7},
        intermediate_indices={2, 4, 6, 8, 11, 12},
        bo_key=bk,
    )
    v = results[8].astype(bfloat16)
    q_roped = results[11].reshape(n_heads, head_dim).astype(bfloat16)
    k_roped = results[12].reshape(n_kv_heads, head_dim).astype(bfloat16)

    # KV cache update
    k_cache_layer[:, current_pos, :] = k_roped
    v_cache_layer[:, current_pos, :] = v.reshape(n_kv_heads, head_dim)

    # --- CPU attention (KV cache) ---
    with cache.profiler.time_cpu("decode_attention_cpu"):
        attn_out = decode_attention_cpu(
            q_roped.flatten(),
            k_cache_layer,
            v_cache_layer,
            current_pos,
            n_heads,
            n_kv_heads,
            head_dim,
        )

    # --- O/Gate/Up/Down GEMVs on NPU; residual + FFN RMSNorm + SwiGLU host glue ---
    with cache.profiler.time_cpu("decode_o_ffn_cpu"):
        x_residual = x_bf16.flatten().astype(bfloat16)
        output = _o_ffn_cpu(layer_weights, attn_out, x_residual, config, cache=cache)

    return output.astype(bfloat16)
