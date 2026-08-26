# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LFM2-1.2B Q4_0 prefill on NPU2 -- the hybrid conv/attention layer stack.

LFM2 has TWO block types, selected by `config.is_attn_layer(idx)`
(`full_attn_idxs = [2, 5, 8, 10, 12, 14]`):

    ATTENTION (6 layers)          SHORTCONV (10 layers)
    -------------------------     --------------------------------
    RMSNorm(operator_norm)        RMSNorm(operator_norm)
    Q/K/V proj                    in_proj  (emb -> 3*conv_dim)
    QK-norm (per-head)            split -> B, C, x
    RoPE                          h = B * x
    FlashAttention (seq-first)    h = causal_depthwise_conv1d(h)
                                  y = C * h
    -------- both then: --------
    out/O proj -> +residual -> RMSNorm(ffn_norm) -> SwiGLU FFN -> +residual

Three structural facts make this cheap to build:

1. **The attention half is Llama-3.2-1B's, plus QK-norm.** emb 2048, head_dim
   64, GQA 32Q:8KV, hidden 8192 -- every shape matches, so the seq-first
   FlashAttention path is reused verbatim. Only the front ELF differs:
   `rms_gemms_rope` (llama) -> `rms_qkv_qknorm_rope`.

2. **The ShortConv block's `out_proj` is 2048x2048x2048 -- the same shape as
   llama's O proj.** So the SAME `o_ffn` ELF serves BOTH block types: it takes
   the ShortConv output where the attention block passes attention output, and
   runs out_proj + residual + ffn_norm + SwiGLU + down + residual identically.
   The conv block needs only four extra leaf launches in front of it.

3. **Q4_0 is a HOST-side codec here.** Like `llama32_1b_q4nx_prefill.py`, the
   weights are dequantized to bf16 once at load and the NPU runs bf16 ops
   against resident weight BOs. Quantization affects the VALUES, not the
   dataflow -- so the prefill graph is the bf16 one unchanged, and the only
   q4nx-specific code is the loader. (The fused DECODE is different: it streams
   4-bit weights and dequantizes on chip.)

## What this feeds

`prefill()` fills the two device-resident regions the fused decode continues
from, and they are not the same shape:

  * attention layers -> a per-layer **K/V cache**;
  * ShortConv layers -> a per-layer **carried conv state**, which is the last
    `conv_L_cache - 1` rows of the PRE-convolution gated signal `h`, not of the
    layer output. Causality is a left PAD, not a mask, so the state is exactly
    the pad the next token's convolution needs.

Both are handed to the decode by `lfm2_1_2b_q4nx_inference.py`.

Usage:
    python3 lfm2_1_2b_q4nx_prefill.py                 # Paris first-token gate
    python3 lfm2_1_2b_q4nx_prefill.py --bench-l 2048  # warm TTFT benchmark
"""

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_PROG_EXAMPLES = _LLMS_DIR.parent
for _p in (str(_PROG_EXAMPLES), str(_LLMS_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lfm2_1_2b_q4nx_weights import (  # noqa: E402
    Lfm2Q4nxConfig as Lfm2Config,
    generate_rope_lut,
    load_weights,
)

K_TAPS = 3
HALO = K_TAPS - 1

# Set by compile_all_kernels: which of (q, k, v) GEMMs the registry resolved to
# fused-cast, hence need an f32 C-scratch arg appended to the fused ELF call.
_FUSED_SCRATCH_FOR = None

# Per-layer ELF argument lists, built once and then mutated in place. Weight
# args are `static_input_indices` so their BOs upload once; rebuilding the
# numpy wrappers on every layer call is pure host churn.
_ARG_CACHE = {}

_ATTN_BACKEND = {
    "omit_while_true_loop": False,
    "omit_pingpong": "all",
    "runtime_loop_tiling_sizes": [1, 1],
    "output_format": "elf",
    "instance_name": "attention_bf16",
}


def _rms_qkv_qknorm_rope_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "rms_qkv_qknorm_rope",
    }


def _o_ffn_backend(verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "o_ffn",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _leaf_backend(name, verbose=False):
    return {
        "verbose": verbose,
        "omit_while_true_loop": False,
        "output_format": "xclbin",
        "instance_name": name,
    }


def _gate_mul_backend(verbose=False):
    """gate_mul is an air.api DSL kernel: it must be RUN with the same
    target_device + loop tiling it was COMPILED with, or the backend rebuilds
    a different xclbin than the cached one."""
    d = _leaf_backend("gate_mul", verbose)
    tgt = globals().get("_GATE_MUL_TARGET")
    if tgt is not None:
        d["target_device"] = tgt
    d["runtime_loop_tiling_sizes"] = [4, 4]
    return d


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def compile_all_kernels(cache, config, seq_len, verbose=False, cpu_attn=False):
    """Compile every ELF both LFM2 block types need, into `cache`."""
    global _FUSED_SCRATCH_FOR

    emb_dim = config.emb_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    hidden_dim = config.hidden_dim
    conv_dim = config.conv_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    print(f"\n{'='*60}\nCompiling LFM2 prefill kernels (seq_len={seq_len})\n{'='*60}")

    from shared.infra.external_kernels import compile_gemm_mm, compile_rope

    # External microkernels FIRST — aircc picks them up from cwd.
    compile_gemm_mm(
        tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o"
    )
    compile_gemm_mm(
        tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o"
    )
    compile_rope()
    _compile_conv1d_o()

    if seq_len % 64 != 0:
        raise ValueError(f"seq_len={seq_len} must be a multiple of 64")
    herd_m = next(h for h in (8, 4, 2, 1) if seq_len % (64 * h) == 0)

    # --- ATTENTION front: RMSNorm + QKV + QK-norm + RoPE -------------------
    print("\n--- rms_qkv_qknorm_rope (RMSNorm+QKV+QK-norm+RoPE) ---")
    from shared.builders.rms_qkv_qknorm_rope_multi import (
        build_rms_qkv_qknorm_rope_module,
    )

    fused_mod, fused_scratch = build_rms_qkv_qknorm_rope_module(
        seq_len,
        emb_dim,
        q_dim,
        kv_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        qknorm_eps=config.norm_eps,
    )
    _FUSED_SCRATCH_FOR = fused_scratch
    cache.compile_and_cache(
        "rms_qkv_qknorm_rope", fused_mod, _rms_qkv_qknorm_rope_backend(verbose)
    )

    # --- shared tail: O/out proj + residual + FFN --------------------------
    print("\n--- o_ffn (out proj + residual + FFN) — shared by BOTH block types ---")
    from shared.builders.o_ffn_multi import build_o_ffn_module

    cache.compile_and_cache(
        "o_ffn",
        build_o_ffn_module(seq_len, emb_dim, hidden_dim, herd_m=herd_m),
        _o_ffn_backend(verbose),
    )

    # --- FlashAttention (seq-first; head_dim=64 needs no host transpose) ---
    if not cpu_attn:
        print("\n--- flash_attn (seq-first FA, head_dim=64) ---")
        from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
            build_module as build_attn,
        )

        cache.compile_and_cache(
            "flash_attn",
            build_attn(
                lk=seq_len,
                lkp=head_dim,
                lq=seq_len,
                lqp=256,
                dk=head_dim,
                dv=head_dim,
                num_q_tiles=4,
                num_cascade_stages=4,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                causal=True,
            ),
            {"verbose": verbose, **_ATTN_BACKEND},
        )
    else:
        print("\n--- Skipping flash_attn (CPU attention fallback) ---")

    # --- CONV block leaves -------------------------------------------------
    print("\n--- conv block leaves (rms_op, in_proj, gate_mul, conv1d) ---")
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from shared.builders.gemm_builder import _build_gemm_module, gemm_registry_config
    from eltwise_mul.eltwise_mul import build_eltwise_mul
    from conv1d_depthwise.conv1d_depthwise import build_module as build_conv1d

    cache.compile_and_cache(
        "rms_op",
        build_rms(seq_len, emb_dim, bfloat16, herd_x=8),
        _leaf_backend("rms_op", verbose),
    )

    # Tiles + method come from the registry per shape, never hardcoded.
    # 2048x2048x6144 is the Qwen3-1.7B Gate/Up row: fused-cast, 6729 GFLOP/s.
    m, k, n = seq_len, emb_dim, 3 * conv_dim
    spec = gemm_registry_config(m, k, n, output_dtype="bf16", precision="high")

    # ------------------------------------------------------------------
    # in_proj runs on the registry's OTHER high-precision bf16-out tier,
    # **drain** (tile_m=32, 1 launch, no f32 scratch), NOT the fused-cast the
    # registry nominates. Identical accuracy — both are FP32-accumulate + a
    # single epilogue cast, the 9.3e-3 GPU-standard tier — but slower.
    #
    # Fused-cast does not place as a standalone ELF at this shape. Measured,
    # across two different builders and three herd shapes:
    #   gemm_builder._build_gemm_module, herd_n=4 -> memtile:
    #       "Failed to allocate buffer buf184, size 131072"
    #       (f32 C-scratch = herd_n*tile_m*tile_n*4 = 128 KB)
    #   gemm_builder._build_gemm_module, herd_n=2 -> L1: "Not all requested
    #       buffers fit", 16x8x8x8xf32 = 32 KB accumulator
    #   matrix_multiplication.bf16_in_bf16_out.build_module_gemm_cast,
    #       herd_m=8/herd_n=4 -> same 32 KB L1 failure
    # The 32 KB L1 C-tile is intrinsic to tile_m=64 x tile_n=128 x f32, so it
    # is not a herd-tuning problem: fused-cast at these tiles needs to be
    # sharing an L1 budget with the launches it is fused into.
    #
    # PHASE 4 OPPORTUNITY (not taken here): fusing the conv operator into one
    # multi-launch ELF would make fused-cast available again and cut in_proj
    # from ~28 ms to ~7.7 ms, i.e. ~200 ms off prefill across the 10 conv
    # layers. That is the single largest remaining prefill win. It needs a
    # conv-operator builder in the o_ffn_multi style, which is new builder
    # work rather than a config change.
    from shared.builders.gemm_builder import gemm_method_spec

    drain = gemm_method_spec("drain")
    globals()["_IN_PROJ_SPEC"] = drain
    print(f"  in_proj GEMM (registry={spec['method']} -> drain standalone) {m}x{k}x{n}")
    cache.compile_and_cache(
        "in_proj",
        _build_gemm_module(
            m,
            k,
            n,
            drain["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
            herd_m,
            4,
            **dict(drain["build_kwargs"]),
        ),
        _leaf_backend("in_proj", verbose),
    )

    # eltwise_mul is an `air.api` DSL example, so it differs from the raw
    # module_builder kernels in two ways:
    #  1. it wants the AIR api type (air.api.types.bf16), not the ml_dtypes
    #     class — it reads `dtype.itemsize` to size the L1 tile budget, and
    #     ml_dtypes.bfloat16 exposes that as a getset_descriptor;
    #  2. build_eltwise_mul returns a LAUNCH, not a module. The module comes
    #     from `launch.build(target=...)`, and the backend then needs
    #     `target_device=launch.target` so it compiles for the generation the
    #     herd was sized against.
    from air.api.types import bf16 as _api_bf16

    _gm_launch = build_eltwise_mul((seq_len, conv_dim), dtype=_api_bf16)
    _gm_module = _gm_launch.build(target="auto")
    globals()["_GATE_MUL_TARGET"] = _gm_launch.target
    cache.compile_and_cache(
        "gate_mul",
        _gm_module,
        {
            **_leaf_backend("gate_mul", verbose),
            "target_device": _gm_launch.target,
            "runtime_loop_tiling_sizes": [4, 4],
        },
    )

    # herd 8x1 / tile_s=8 is the measured-best Conv1D config. herd_x=2 and
    # herd_y>1 are rejected by asserts inside the builder — both are SILENT
    # failure modes (see kernel_registry/details/Conv1D_bf16.md).
    cache.compile_and_cache(
        "conv1d",
        build_conv1d(seq_len, conv_dim, 8, bfloat16, herd_x=8, herd_y=1),
        _leaf_backend("conv1d", verbose),
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} kernels compiled to {cache.cache_dir}/")


def _compile_conv1d_o():
    """Compile conv1d_depthwise.cc -> conv1d_depthwise.o into the cwd."""
    import os
    import subprocess

    peano = os.environ.get("PEANO_INSTALL_DIR")
    aieopt = os.environ.get("MLIR_AIE_INSTALL_DIR") or os.environ.get("AIEOPT_DIR")
    if not peano:
        raise RuntimeError("PEANO_INSTALL_DIR is not set")
    src = _PROG_EXAMPLES / "conv1d_depthwise" / "conv1d_depthwise.cc"
    subprocess.run(
        [
            f"{peano}/bin/clang++",
            "-O2",
            "-std=c++20",
            "--target=aie2p-none-unknown-elf",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-DNDEBUG",
            "-I",
            f"{aieopt}/include",
            "-D__AIE_API_AIE_ADF_HPP__",
            "-include",
            "aie_kernels/aie_kernel_utils.h",
            "-c",
            str(src),
            "-o",
            "conv1d_depthwise.o",
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# ELF call helpers
# ---------------------------------------------------------------------------


def _call_qkv_qknorm_rope(
    cache, lw, config, seq_len, lut_q, lut_k, layer_idx, x_in, verbose=False
):
    """One rms_qkv_qknorm_rope ELF call. Returns (v, q_roped, k_roped)."""
    emb_dim = config.emb_dim
    q_dim = config.n_heads * config.head_dim
    kv_dim = config.n_kv_heads * config.head_dim
    aw = lw.attn

    args = [
        np.asarray(x_in, dtype=bfloat16).reshape(seq_len, emb_dim),  # 0
        np.asarray(lw.operator_norm, dtype=bfloat16).reshape(emb_dim),  # 1
        np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2
        np.asarray(aw.wq, dtype=bfloat16).reshape(emb_dim, q_dim),  # 3
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 4
        np.asarray(aw.wk, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 5
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 6
        np.asarray(aw.wv, dtype=bfloat16).reshape(emb_dim, kv_dim),  # 7
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 8 v
        np.asarray(aw.q_norm, dtype=bfloat16).reshape(config.head_dim),  # 9
        np.asarray(aw.k_norm, dtype=bfloat16).reshape(config.head_dim),  # 10
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 11
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 12
        lut_q,  # 13
        lut_k,  # 14
        np.zeros((seq_len, q_dim), dtype=bfloat16),  # 15 q_roped
        np.zeros((seq_len, kv_dim), dtype=bfloat16),  # 16 k_roped
    ]
    inter = {2, 4, 6, 8, 11, 12, 15, 16}
    nxt = 17
    for sc, cols in zip(_FUSED_SCRATCH_FOR or [], (q_dim, kv_dim, kv_dim)):
        if sc is not None:
            args.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(nxt)
            nxt += 1
    return cache.load_and_run(
        "rms_qkv_qknorm_rope",
        _rms_qkv_qknorm_rope_backend(verbose),
        *args,
        output_indices=[8, 15, 16],
        static_input_indices={1, 3, 5, 7, 9, 10, 13, 14},
        intermediate_indices=inter,
        bo_key=f"rms_qkv_qknorm_rope_L{layer_idx}",
        shared_nonstatic=True,
    )


def _call_o_ffn(
    cache, op_out, residual_in, lw, config, seq_len, layer_idx, verbose=False
):
    """out/O proj + residual + ffn_norm + SwiGLU + down + residual.

    Shared by both block types: `wo` is the attention O proj on attention
    layers and the ShortConv `out_proj` on conv layers — same 2048x2048x2048
    shape either way.
    """
    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    n_total = seq_len * emb_dim
    wo = lw.attn.wo if lw.is_attn else lw.conv.w_out

    # Build the 19-arg list ONCE per layer and mutate only the two dynamic
    # slots (0 = operator output, 3 = residual) on later calls. The weight
    # arrays are `static_input_indices`, so their BOs upload once; rebuilding
    # the numpy wrappers every call was pure host churn (~12 MB/layer).
    key = f"o_ffn_L{layer_idx}"
    cached = _ARG_CACHE.get(key)
    if cached is not None:
        cached[0] = np.asarray(op_out, dtype=bfloat16).reshape(seq_len, emb_dim)
        cached[3] = np.asarray(residual_in, dtype=bfloat16).reshape(seq_len, emb_dim)
        args = cached
    else:
        args = [
            np.asarray(op_out, dtype=bfloat16).reshape(seq_len, emb_dim),  # 0
            np.asarray(wo, dtype=bfloat16).reshape(emb_dim, emb_dim),  # 1
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 2
            np.asarray(residual_in, dtype=bfloat16).reshape(seq_len, emb_dim),  # 3
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 4
            np.asarray(lw.ffn_norm, dtype=bfloat16).reshape(emb_dim),  # 5
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 6
            np.asarray(lw.w_gate, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # 7
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 8
            np.asarray(lw.w_up, dtype=bfloat16).reshape(emb_dim, hidden_dim),  # 9
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 10
            np.zeros((seq_len, hidden_dim), dtype=bfloat16),  # 11
            np.asarray(lw.w_down, dtype=bfloat16).reshape(hidden_dim, emb_dim),  # 12
            np.zeros((seq_len, emb_dim), dtype=bfloat16),  # 13
            np.zeros(n_total, dtype=bfloat16),  # 14 out
        ]
        # 15..: one f32 C-scratch per fused-cast GEMM, per the registry.
        args.extend(_o_ffn_scratch_specs(seq_len, emb_dim, hidden_dim)[0])
        _ARG_CACHE[key] = args
    results = cache.load_and_run(
        "o_ffn",
        _o_ffn_backend(verbose),
        *args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14}
        | _o_ffn_scratch_specs(seq_len, emb_dim, hidden_dim)[1],
        bo_key=f"o_ffn_L{layer_idx}",
        shared_nonstatic=True,
    )
    return results[14].reshape(seq_len, emb_dim)


# ---------------------------------------------------------------------------
# Block runners
# ---------------------------------------------------------------------------


def run_conv_block(
    x_bf16, lw, config, cache, layer_idx=0, conv_state=None, verbose=False
):
    """Run one Lfm2ShortConv layer on NPU.

    Returns (out_bf16, intermediates, new_conv_state).
    """
    seq_len, emb = x_bf16.shape
    cdim = config.conv_dim
    cw = lw.conv
    ints = {}

    if verbose:
        print(f"  Layer {layer_idx} [CONV]")

    # 1. RMSNorm(operator_norm)
    r = cache.load_and_run(
        "rms_op",
        _leaf_backend("rms_op", verbose),
        np.asarray(x_bf16, dtype=bfloat16).reshape(seq_len, emb),
        np.asarray(lw.operator_norm, dtype=bfloat16).reshape(emb),
        np.zeros((seq_len, emb), dtype=bfloat16),
    )
    xn = r[-1].reshape(seq_len, emb)

    # 2. in_proj -> (seq, 3*conv_dim). Column order is B | C | v (v LAST).
    r = cache.load_and_run(
        "in_proj",
        _leaf_backend("in_proj", verbose),
        *_gemm_args(xn, cw.w_in, seq_len, emb, 3 * cdim),
    )
    bcx = r[-1].reshape(seq_len, 3 * cdim)
    B = np.ascontiguousarray(bcx[:, 0:cdim])
    C = np.ascontiguousarray(bcx[:, cdim : 2 * cdim])
    v = np.ascontiguousarray(bcx[:, 2 * cdim : 3 * cdim])

    # 3. gate 1: h = B * v
    r = cache.load_and_run(
        "gate_mul",
        _gate_mul_backend(verbose),
        B,
        v,
        np.zeros((seq_len, cdim), dtype=bfloat16),
    )
    h = r[-1].reshape(seq_len, cdim)

    # 4. causal depthwise conv. Causality is the PAD, not a mask: prepend the
    #    2-row conv state (zeros at sequence start) and hand the kernel
    #    (seq+2, C); it returns (seq, C). HF stores w_conv channel-major
    #    (C, 3); the kernel wants tap-major (3, C) for unit-stride loads.
    x_pad = np.zeros((seq_len + HALO, cdim), dtype=bfloat16)
    if conv_state is not None:
        x_pad[:HALO] = conv_state
    x_pad[HALO:] = h
    w_tap = np.ascontiguousarray(np.asarray(cw.w_conv, dtype=bfloat16).T)
    r = cache.load_and_run(
        "conv1d",
        _leaf_backend("conv1d", verbose),
        x_pad,
        w_tap,
        np.zeros((seq_len, cdim), dtype=bfloat16),
    )
    conv_out = r[-1].reshape(seq_len, cdim)
    new_state = np.ascontiguousarray(h[-HALO:]).copy()
    # The caller needs the PRE-conv gated signal to slice the carried state at
    # the real end of the prompt: `new_state` above is the state at the end of
    # the PADDED sequence, which is only the same thing when the prompt fills
    # seq_len exactly.
    ints["h_preconv"] = h

    # 5. gate 2: y = C * conv_out
    r = cache.load_and_run(
        "gate_mul",
        _gate_mul_backend(verbose),
        C,
        conv_out,
        np.zeros((seq_len, cdim), dtype=bfloat16),
    )
    y = r[-1].reshape(seq_len, cdim)

    # 6. shared tail
    out = _call_o_ffn(cache, y, x_bf16, lw, config, seq_len, layer_idx, verbose)
    ints["ffn_out"] = out
    return out, ints, new_state


def _o_ffn_scratch_specs(seq_len, emb_dim, hidden_dim):
    """Registry-driven f32 C-scratch args for o_ffn's four GEMMs, in builder
    order (O, Gate, Up, Down). Returns (list_of_scratch_arrays, set_of_indices).

    Same contract as _gemm_args below -- the registry spec says which GEMMs
    take an f32 C scratch, never a guess -- and it MUST mirror
    build_o_ffn_module's own alloc_gemm_scratch call (same shapes, same order,
    base index 15). All four are fused-cast at seq_len 2048, which is why the
    list used to be four hardcoded arrays, but not at every length: at 512 and
    1024 the registry measures drain faster for Gate/Up, the ELF then declares
    two scratch args instead of four, and the hardcoded list overran it
    (`set_arg(16) >= size 16`).
    """
    from shared.builders.gemm_builder import gemm_registry_config

    o_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    g_spec = gemm_registry_config(seq_len, emb_dim, hidden_dim, "bf16", "high")
    d_spec = gemm_registry_config(seq_len, hidden_dim, emb_dim, "bf16", "high")
    arrays, inter = [], set()
    nxt = 15
    for spec, cols in (
        (o_spec, emb_dim),
        (g_spec, hidden_dim),  # gate
        (g_spec, hidden_dim),  # up (same shape/spec as gate)
        (d_spec, emb_dim),  # down
    ):
        if spec["needs_f32_scratch"]:
            arrays.append(np.zeros((seq_len, cols), dtype=np.float32))
            inter.add(nxt)
            nxt += 1
    return arrays, inter


def _gemm_args(a, b, m, k, n):
    """Arg list for a standalone bf16-out GEMM ELF.

    Fused-cast GEMMs write an f32 C scratch and then run a separate on-chip
    cast launch, so they take a 4th arg. The registry spec says which —
    `needs_f32_scratch`, not a guess at the method name.
    """
    spec = globals().get("_IN_PROJ_SPEC") or {}
    args = [
        np.asarray(a, dtype=bfloat16).reshape(m, k),
        np.asarray(b, dtype=bfloat16).reshape(k, n),
        np.zeros((m, n), dtype=bfloat16),
    ]
    if spec.get("needs_f32_scratch"):
        args.append(np.zeros((m, n), dtype=np.float32))
    return args


def run_attn_block(
    x_bf16, lw, rope_lut_bf16, config, cache, layer_idx=0, cpu_attn=False, verbose=False
):
    """Run one Lfm2Attention layer on NPU."""
    seq_len, emb = x_bf16.shape
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    q_dim = n_heads * head_dim
    ints = {}

    if verbose:
        print(f"  Layer {layer_idx} [ATTN]")

    lut_q = np.repeat(rope_lut_bf16[:seq_len], n_heads, axis=0).flatten()
    lut_k = np.repeat(rope_lut_bf16[:seq_len], n_kv_heads, axis=0).flatten()

    res = _call_qkv_qknorm_rope(
        cache, lw, config, seq_len, lut_q, lut_k, layer_idx, x_bf16, verbose
    )
    v = res[8].reshape(seq_len, n_kv_heads * head_dim)
    q_roped = res[15].reshape(seq_len, q_dim)
    k_roped = res[16].reshape(seq_len, n_kv_heads * head_dim)
    ints["q_roped"], ints["k_roped"], ints["v"] = q_roped, k_roped, v

    if cpu_attn:
        from lfm2_1_2b_q4nx_cpu_attn import attention_reference

        attn_out = attention_reference(
            q_roped.astype(np.float32),
            k_roped.astype(np.float32),
            v.astype(np.float32),
            n_heads,
            n_kv_heads,
        ).astype(bfloat16)
    else:
        r = cache.load_and_run(
            "flash_attn",
            _ATTN_BACKEND,
            np.ascontiguousarray(q_roped),
            np.ascontiguousarray(k_roped),
            np.ascontiguousarray(v),
            np.zeros((seq_len, q_dim), dtype=bfloat16),
        )
        attn_out = r[-1].reshape(seq_len, q_dim)

    out = _call_o_ffn(cache, attn_out, x_bf16, lw, config, seq_len, layer_idx, verbose)
    ints["ffn_out"] = out
    return out, ints


def run_transformer_block(
    x_bf16,
    lw,
    rope_lut_bf16,
    config,
    cache,
    layer_idx,
    conv_state=None,
    cpu_attn=False,
    verbose=False,
):
    """Dispatch to the right block type for `layer_idx`.

    Drives off `config.is_attn_layer` rather than any assumed stride — LFM2's
    schedule is irregular (gaps of 2,2,1,1,1,0 conv layers between attention
    layers), so a modulo would silently mis-type layers.
    """
    if config.is_attn_layer(layer_idx):
        out, ints = run_attn_block(
            x_bf16,
            lw,
            rope_lut_bf16,
            config,
            cache,
            layer_idx,
            cpu_attn=cpu_attn,
            verbose=verbose,
        )
        return out, ints, None
    return run_conv_block(
        x_bf16, lw, config, cache, layer_idx, conv_state=conv_state, verbose=verbose
    )


# ---------------------------------------------------------------------------
# Driver-facing wrapper
# ---------------------------------------------------------------------------

MODEL_DEFAULT = os.environ.get("LFM2_MODEL_SOURCE") or "LiquidAI/LFM2-1.2B"
# "The capital of France is" -- the cheapest end-to-end weight-integrity check
# there is. If the Q4_0 pack, the layer schedule or any GEMM's operand order is
# wrong, the argmax is not " Paris".
PARIS_PROMPT = "The capital of France is"


class Lfm2Q4nxPrefill:
    """Batched NPU prefill for LFM2-1.2B with Q4_0 weights.

    Same contract as `llama32_1b_q4nx_prefill.LlamaQ4nxPrefill`, so the shared
    inference driver and verify adapter treat the two the same -- plus
    `get_conv_state()`, which LFM2 needs and a pure transformer does not.
    """

    def __init__(self, seq_len=2048, cache_dir=None, verbose=False, cpu_attn=False):
        self.config = Lfm2Config()
        self.seq_len = int(seq_len)
        self.verbose = verbose
        self.cpu_attn = cpu_attn
        self.weights = None
        self._rope_lut = None
        # Per-layer captures the fused decode continues from. Both are indexed
        # by MODEL layer, so a layer has an entry in exactly one of them.
        self.k_cache = {}
        self.v_cache = {}
        self.conv_state = {}
        self._ctx_len = 0

        from shared.infra.cache import KernelCache

        cache_dir = (
            cache_dir
            or os.environ.get("LFM2_CACHE_DIR")
            or os.path.join(os.getcwd(), f"kernel_cache_{self.seq_len}")
        )
        self.cache = KernelCache(str(cache_dir), verbose=verbose)

    # -- setup ------------------------------------------------------------
    def compile(self):
        """Build every ELF both block types need. No weights, no dispatch."""
        compile_all_kernels(
            self.cache,
            self.config,
            self.seq_len,
            verbose=self.verbose,
            cpu_attn=self.cpu_attn,
        )

    def load_weights(self, model=None):
        model = model or MODEL_DEFAULT
        print(f"[lfm2_prefill] loading + Q4_0-quantizing weights ({model})", flush=True)
        self.weights = load_weights(model, config=self.config, verbose=self.verbose)
        self._rope_lut = generate_rope_lut(self.config, seq_len=self.seq_len)
        return self.weights

    # -- run --------------------------------------------------------------
    def prefill(self, ids):
        """Embed `ids`, run all 16 layers, return the LAST position's logits.

        The prompt is right-padded to `seq_len`. That is safe for BOTH block
        types and for the same reason: attention is causally masked, and the
        ShortConv is a causal (left-padded) convolution, so no position <= the
        real prompt length can see a padded one.
        """
        if self.weights is None:
            self.load_weights()
        cfg, S = self.config, self.seq_len
        n = len(ids)
        if n > S:
            raise ValueError(f"prompt of {n} tokens exceeds seq_len={S}")

        x = np.zeros((S, cfg.emb_dim), dtype=bfloat16)
        x[:n] = self.weights.embed_table[np.asarray(ids, dtype=np.int64)]

        for li in range(cfg.n_layers):
            lw = self.weights.layers[li]
            out, ints, new_state = run_transformer_block(
                x,
                lw,
                self._rope_lut,
                cfg,
                self.cache,
                li,
                conv_state=None,
                cpu_attn=self.cpu_attn,
                verbose=self.verbose,
            )
            if cfg.is_attn_layer(li):
                # Keep only the REAL positions: the decode appends from n
                # onward, so a padded row left in the cache would be attended
                # to as though it were context.
                self.k_cache[li] = np.ascontiguousarray(ints["k_roped"][:n]).copy()
                self.v_cache[li] = np.ascontiguousarray(ints["v"][:n]).copy()
            else:
                # The carried state is the last conv_L_cache-1 rows of the
                # PRE-conv gated signal at the REAL end of the prompt (row
                # n-1), NOT at the padded end (row S-1) that run_conv_block
                # returns. Those coincide only when the prompt fills seq_len
                # exactly, so slice it here.
                #
                # Getting this wrong is invisible in the prefill's own output
                # -- the logits at row n-1 are right either way -- and shows up
                # only as the first few DECODED tokens being wrong, which is a
                # far more expensive place to find it.
                h_pre = ints["h_preconv"]
                pad = max(0, HALO - n)
                st = np.zeros((HALO, cfg.conv_dim), dtype=h_pre.dtype)
                st[pad:] = h_pre[max(0, n - HALO) : n]
                self.conv_state[li] = np.ascontiguousarray(st).copy()
            x = out

        self._ctx_len = n
        h = np.asarray(x[n - 1], dtype=np.float32)
        h = h / np.sqrt((h * h).mean() + cfg.norm_eps)
        h = h * np.asarray(self.weights.final_norm, dtype=np.float32)
        # LM head on the HOST. It is one GEMV over the tied embedding table
        # (2048 x 65536) and runs once per prompt, so it is a small slice of
        # TTFT; the per-token LM head, which runs every token, is on the NPU
        # inside the fused decode.
        head = self.weights.lm_head
        if head is None:
            head = self.weights.embed_table  # tied
        return (np.asarray(head, dtype=np.float32) @ h).astype(np.float32)

    # -- accessors the inference driver and verify adapter use -------------
    def kv_view(self, layer_idx):
        return self.k_cache.get(layer_idx), self.v_cache.get(layer_idx)

    def get_k_cache(self, layer_idx, idx=None):
        k = self.k_cache[layer_idx]
        return k if idx is None else k[idx]

    def get_v_cache(self, layer_idx, idx=None):
        v = self.v_cache[layer_idx]
        return v if idx is None else v[idx]

    def get_conv_state(self, layer_idx):
        """[conv_L_cache-1, conv_dim] carried state for a ShortConv layer."""
        return self.conv_state[layer_idx]

    def clear_context(self):
        self.k_cache.clear()
        self.v_cache.clear()
        self.conv_state.clear()
        self._ctx_len = 0

    def get_current_context_length(self):
        return self._ctx_len

    def set_context_length(self, L):
        self._ctx_len = int(L)


# ---------------------------------------------------------------------------
# CLI: Paris weight-integrity gate, and the warm-TTFT benchmark
# ---------------------------------------------------------------------------


def _main():
    import argparse
    import time

    ap = argparse.ArgumentParser(description="LFM2-1.2B Q4_0 prefill")
    ap.add_argument("--model", default=MODEL_DEFAULT, help="HF repo id or local dir")
    ap.add_argument(
        "--seq-len", type=int, default=int(os.environ.get("LFM2_SEQ_LEN", "2048"))
    )
    ap.add_argument("--cache-dir", default=os.environ.get("LFM2_CACHE_DIR") or None)
    ap.add_argument(
        "--bench-l",
        type=int,
        default=int(os.environ.get("LFM2_BENCH_L", "0")),
        help="warm-TTFT benchmark at this context length (0 = run the Paris gate)",
    )
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--cpu-attn", action="store_true", help="bring-up: CPU FA")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seq_len = args.bench_l or args.seq_len
    pf = Lfm2Q4nxPrefill(
        seq_len=seq_len,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        cpu_attn=args.cpu_attn,
    )
    pf.compile()
    if args.compile_only:
        print("Compilation passed.", flush=True)
        return 0

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    pf.load_weights(args.model)

    if args.bench_l:
        ids = [tok.bos_token_id or 1] * args.bench_l
        print(f"[bench] warmup prefill L={args.bench_l}...", flush=True)
        pf.prefill(ids)  # first call uploads the weight BOs
        print(f"[bench] timed prefill L={args.bench_l}...", flush=True)
        t0 = time.perf_counter()
        pf.prefill(ids)
        wall = time.perf_counter() - t0
        # Same label extract_perf.py parses for every other model, so this
        # example's row on the published benchmark is filled by the same regex.
        print(f"Time to first token (TTFT): {wall:.3f}s", flush=True)
        print(f"Warm time to first token (TTFT): {wall:.3f}s", flush=True)
        return 0

    ids = tok(PARIS_PROMPT, return_tensors=None)["input_ids"]
    print(f"[lfm2_prefill] prefill prompt N={len(ids)} ...", flush=True)
    logits = pf.prefill(ids)
    top = int(np.argmax(logits))
    text = tok.decode([top])
    ok = text.strip().lower() == "paris"
    print(f"[lfm2_prefill] first-token argmax={top} ({text!r})", flush=True)
    print("[lfm2_prefill] *** PARIS ***" if ok else "[lfm2_prefill] MISS", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
