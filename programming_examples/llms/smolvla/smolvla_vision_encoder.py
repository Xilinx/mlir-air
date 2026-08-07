# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolVLA Vision-Encoder (SigLIP ViT) 12-layer prefill on MLIR-AIR (NPU2).

The NPU driver for the SigLIP ViT (bidirectional
12-layer encoder) instead of the causal language backbone. Correctness-first
(A3-5 Step 2-3): the four heavy ops run on NPU — projection/MLP GEMMs, affine
LayerNorm, GELU-tanh, and non-causal FlashAttention — while the cheap glue
(bias-adds, residual adds, im2col patch-embed) stays on host numpy. Fusing those
host ops onto the device and cutting dispatches is A3-6 (a SEPARATE perf phase);
nothing here is optimized.

SigLIP differences from the backbone (all handled below):
  - Every Linear has a BIAS (q/k/v/out proj + fc1/fc2) — added on host after GEMM.
  - Norm is affine LayerNorm (gamma + beta, eps 1e-6), not RMSNorm. The affine
    `layer_norm` ELF takes a packed [2N] weight||bias buffer.
  - Bidirectional MHA (12 heads, no GQA, no mask, scale 1/8) → the registry FA
    ELF with causal=False. FA applies 1/sqrt(dk)=1/8 internally, matching
    SigLIP's attn_scale exactly, so Q is NOT pre-scaled.
  - GELU-tanh MLP activation (not SwiGLU) → the `gelu` 1D elementwise ELF.

NPU kernels driven (all validated at these exact shapes in A3-1..A3-4, registry):
  gemm_qkvo : 1024x768x768   (q/k/v/o projections, drain tile_m32/tn96)
  gemm_fc1  : 1024x768x3072  (MLP fc1, drain tile_m32/tn128)
  gemm_fc2  : 1024x3072x768  (MLP fc2, drain tile_m32/tn96)
  layer_norm: 1024x768 affine (ln1, ln2, post_layernorm), herd_x=8
  gelu      : N=1024*3072 GELU-tanh, herd 8x2
  flash_attn: 1024/1024 12q/12kv MHA, head_dim=64, non-causal, hpu=2 (full array)
  gemm_connector : 64x12288x960 (A3-5 Step 4 modality projection, drain
                   tile_m16/tn80, herd 4x4 — the registry's per-shape override)
"""

import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Add parent directory to path for kernel imports (programming_examples/).
_PROG_EXAMPLES = str(Path(__file__).resolve().parent.parent.parent)
if _PROG_EXAMPLES not in sys.path:
    sys.path.insert(0, _PROG_EXAMPLES)
# Also add llms/ for sibling LLM packages (shared.infra, shared.builders).
_LLMS_DIR = str(Path(__file__).resolve().parent.parent)
if _LLMS_DIR not in sys.path:
    sys.path.insert(0, _LLMS_DIR)

from smolvla_vision_weights import SigLIPVisionConfig
from smolvla_cpu_helpers import im2col_patch_embed
from shared.infra.cache import KernelCache, Profiler  # noqa: F401 (re-exported)

# ---------------------------------------------------------------------------
# Per-kernel backend presets (match each kernel's own standalone example)
# ---------------------------------------------------------------------------


def _gemm_backend():
    # Standalone external-mm.o GEMM ELF. instance_name MUST equal the module
    # func name (`matmul_bf16`), NOT the cache key.
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "matmul_bf16",
        "runtime_loop_tiling_sizes": [4, 4],
    }


def _ln_backend():
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "layer_norm",
        "runtime_loop_tiling_sizes": [4, 4],
    }


def _gelu_backend():
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "gelu",
        "runtime_loop_tiling_sizes": [4, 4],
    }


_ATTN_BACKEND_KWARGS = {
    "verbose": False,
    "omit_while_true_loop": False,
    "omit_pingpong": "all",
    "runtime_loop_tiling_sizes": [1, 1],
    "output_format": "elf",
    "instance_name": "attention_bf16",
}


# --- A3-6b fused-ELF backends (Lever 2 + Lever 3) ---
# Both fused ViT ELFs (vit_ln_qkv, vit_o_ffn) are stitched from drain GEMMs +
# affine LN + on-device bias-adds + residual adds + GELU. Drain herds need
# runtime_loop_tiling_sizes=[2,2] for BD-ID recycling (same as the backbone o_ffn).


def _vit_ln_qkv_backend():
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "vit_ln_qkv",
        "runtime_loop_tiling_sizes": [2, 2],
    }


def _vit_o_ffn_backend():
    return {
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "vit_o_ffn",
        "runtime_loop_tiling_sizes": [2, 2],
    }


# Vision GEMM tiles (registry, precision="high" → drain). All three resolve to
# tile_m=32 drain; kept as explicit constants so compile-time mm.o bakes match
# the build_module tile args (registry-validated in A3-1, commit 3fc1bb12).
#   gemm_qkvo 1024x768x768 : tile_m32 tk_l2 384 tk_l1 64 tn 96
#   gemm_fc1  1024x768x3072: tile_m32 tk_l2 256 tk_l1 32 tn 128
#   gemm_fc2  1024x3072x768: tile_m32 tk_l2 384 tk_l1 64 tn 96
_GEMM_SHAPES = {
    "gemm_qkvo": dict(
        m=1024, k=768, n=768, tile_m=32, tile_k_l2=384, tile_k_l1=64, tile_n=96
    ),
    "gemm_fc1": dict(
        m=1024, k=768, n=3072, tile_m=32, tile_k_l2=256, tile_k_l1=32, tile_n=128
    ),
    "gemm_fc2": dict(
        m=1024, k=3072, n=768, tile_m=32, tile_k_l2=384, tile_k_l1=64, tile_n=96
    ),
}

# A3-5 Step 4: the connector (modality projection) GEMM, 64x12288x960.
# Registry row (drain, precision="high"): tile_m=16, tile_k_l2=384, tile_k_l1=32,
# tile_n=240, with a per-shape HERD OVERRIDE of 4x4 — M=64 is only 4 tile_m rows,
# so the usual 8-row herd cannot be filled (build_module asserts
# m % (tile_m*herd_m) == 0). tile_m=16 also differs from the _m32 drain default,
# so this ELF links its OWN symbol-suffixed microkernel (mm_m16_n240.o) and can
# never collide with the encoder's mm_m32_n{96,128}.o (see the fused-ELF mm.o
# gotcha in smolvla_vision_builders._force_tile_n_suffix).
#
# tile_n=240 (was 80): measured on NPU2 2026-07-27, the tile_n sweep at herd 4x4
# is strongly non-flat — 16:3352us, 48:1347us, 80:890us, 240:667-713us. 240 is
# 1.33x faster at BIT-IDENTICAL accuracy (both mean_rel_L1=9.450e-3, abs_err max
# 3.662e-4). The old 80 came from the generic "N not 512-aligned -> shrink
# TILE_N" habit, which is the wrong instinct here: this GEMM is WEIGHT-DMA-BOUND
# (B = 23.6 MB = 93% of traffic, independent of M), so fewer/larger N-tiles
# amortize the weight stream better. 16 tiles is the hard ceiling AND the right
# choice — padding M 64->128 to fill the full 8x4 herd was measured SLOWER in
# wall-clock (734us vs 667-713us). See kernel_registry GEMM detail page.
_CONNECTOR_GEMM = dict(
    m=64,
    k=12288,
    n=960,
    tile_m=16,
    tile_k_l2=384,
    tile_k_l1=32,
    tile_n=240,
    herd_m=4,
    herd_n=4,
    sym_suffix="_m16_n240",
    obj="mm_m16_n240.o",
)
_PIXEL_SHUFFLE_FACTOR = 4


# ---------------------------------------------------------------------------
# Kernel compilation
# ---------------------------------------------------------------------------


def _compile_flash_attn(cache, config, seq_len, fa_bfp16):
    """Compile the non-causal FlashAttention ELF (shared by fused + unfused)."""
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as build_attn,
    )
    from shared.infra.external_kernels import compile_attn_npu2

    n_heads = config.n_heads
    n_kv_heads = config.n_heads  # MHA
    head_dim = config.head_dim
    num_heads_per_unroll = 2
    num_q_tiles = 4
    assert n_heads % num_heads_per_unroll == 0
    assert num_heads_per_unroll * num_q_tiles <= 8
    print(
        f"  Compiling flash_attn: non-causal, seq={seq_len}, {n_heads}Q/{n_kv_heads}KV, "
        f"head_dim={head_dim}, hpu={num_heads_per_unroll}"
    )
    attn_mod = build_attn(
        lk=seq_len,
        lkp=head_dim,
        lq=seq_len,
        lqp=256,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=num_q_tiles,
        num_cascade_stages=4,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        causal=False,
        num_heads_per_unroll=num_heads_per_unroll,
    )
    compile_attn_npu2(head_dim=head_dim, bfp16=fa_bfp16, force=True)
    print(f"    (FA microkernel BFP16={fa_bfp16})")
    cache.compile_and_cache(
        "flash_attn", attn_mod, {**_ATTN_BACKEND_KWARGS, "verbose": cache.verbose}
    )


def _compile_connector_gemm(cache):
    """Compile the connector projection GEMM ELF (64x12288x960, drain 4x4 herd).

    Registry-validated shape; see `_CONNECTOR_GEMM` for why it needs its own
    tile_m=16 / herd 4x4 / symbol-suffixed microkernel. The pixel-shuffle that
    produces its (64, 12288) input is a pure host reshape (no arithmetic), so
    the connector's only math — the projection — runs on NPU.
    """
    from shared.infra.external_kernels import compile_gemm_mm
    from matrix_multiplication.bf16_in_bf16_out.run import build_module as build_gemm

    s = _CONNECTOR_GEMM
    print(
        f"  Compiling gemm_connector: {s['m']}x{s['k']}x{s['n']} "
        f"(drain tile_m={s['tile_m']} tile_n={s['tile_n']} "
        f"herd {s['herd_m']}x{s['herd_n']})"
    )
    compile_gemm_mm(
        tile_m=s["tile_m"],
        tile_n=s["tile_n"],
        tile_k_l1=s["tile_k_l1"],
        sym_suffix=s["sym_suffix"],
        out_name=s["obj"],
    )
    mod = build_gemm(
        s["m"],
        s["k"],
        s["n"],
        s["tile_m"],
        s["tile_k_l2"],
        s["tile_k_l1"],
        s["tile_n"],
        s["herd_m"],
        s["herd_n"],
        bfloat16,
        bfloat16,
        arch="aie2p",
        emit_external_call=True,
        sym_suffix=s["sym_suffix"],
        link_with_name=s["obj"],
    )
    cache.compile_and_cache(
        "gemm_connector", mod, {"verbose": cache.verbose, **_gemm_backend()}
    )


def _compile_fused_kernels(cache, config, seq_len, fa_bfp16, with_connector=True):
    """A3-6b Lever 2+3: compile the two fused ViT multi-launch ELFs + FA.

    vit_ln_qkv  : LN1 + Q/K/V GEMM + Q/K/V bias-add       (7 launches, 1 ELF)
    flash_attn  : non-causal MHA                          (1 launch,  registry ELF)
    vit_o_ffn   : O GEMM + O bias + residual + LN2 + fc1 + fc1 bias + GELU
                  + fc2 + fc2 bias + residual             (10 launches, 1 ELF)

    All the per-Linear bias-adds and both residual adds — host f32 glue in the
    unfused driver — run on-device inside these two ELFs. => 3 NPU dispatches per
    layer (was 10) and NO host bias/residual round-trips.

    Order matters: compile_gemm_mm bakes DIM_N into each mm_*.o at compile time,
    then compile_and_cache -> prepare_air_project stages the current CWD .o's into
    air_project/. Both ELFs use tile_n-keyed drain objects (mm_m32_n96 for
    qkvo/o/fc2, mm_m32_n128 for fc1); compile every distinct one first so both
    ELFs link the correctly-baked objects (see smolvla_vision_builders._force_tile_n_suffix).
    """
    from shared.infra.external_kernels import compile_gemm_mm
    from shared.builders.gemm_builder import (
        gemm_registry_config,
        disambiguate_by_tile_n,
    )
    from smolvla_vision_builders import build_vit_ln_qkv_module, build_vit_o_ffn_module

    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    n_heads = config.n_heads
    head_dim = config.head_dim

    # Compile every distinct tile_n-keyed drain mm.o the two ELFs link.
    o_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    g_spec = gemm_registry_config(seq_len, emb_dim, hidden_dim, "bf16", "high")
    d_spec = gemm_registry_config(seq_len, hidden_dim, emb_dim, "bf16", "high")
    o_spec, g_spec, d_spec = disambiguate_by_tile_n([o_spec, g_spec, d_spec])
    _needed = {}
    for s in (o_spec, g_spec, d_spec):
        _needed[s["sym_suffix"]] = s
    for s in _needed.values():
        print(
            f"  Compiling {s['obj']} (drain tile_m={s['tile_m']} tile_n={s['tile_n']})"
        )
        compile_gemm_mm(
            tile_m=s["tile_m"],
            tile_n=s["tile_n"],
            tile_k_l1=s["tile_k_l1"],
            sym_suffix=s["sym_suffix"],
            out_name=s["obj"],
        )

    print("  Compiling vit_ln_qkv (7-launch fused ELF)...")
    cache.compile_and_cache(
        "vit_ln_qkv",
        build_vit_ln_qkv_module(seq_len, emb_dim, n_heads, head_dim),
        {"verbose": cache.verbose, **_vit_ln_qkv_backend()},
    )

    print("  Compiling vit_o_ffn (10-launch fused ELF)...")
    cache.compile_and_cache(
        "vit_o_ffn",
        build_vit_o_ffn_module(seq_len, emb_dim, hidden_dim),
        {"verbose": cache.verbose, **_vit_o_ffn_backend()},
    )

    # Standalone affine LayerNorm ELF — used ONCE at the end of the stack for
    # post_layernorm (kept on NPU, identical to the unfused path so post_ln
    # cosine vs oracle is unchanged). Not in the per-layer hot loop.
    from layer_norm.layer_norm import build_module as build_layer_norm

    print(f"  Compiling layer_norm: {seq_len}x{emb_dim} affine (post_ln, herd_x=8)")
    cache.compile_and_cache(
        "layer_norm",
        build_layer_norm(seq_len, emb_dim, bfloat16, herd_x=8),
        {"verbose": cache.verbose, **_ln_backend()},
    )

    _compile_flash_attn(cache, config, seq_len, fa_bfp16)

    if with_connector:
        _compile_connector_gemm(cache)

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} vision kernels compiled to {cache.cache_dir}/")
    if cache.profiler.enabled:
        total = sum(cache.profiler.compile_times.values())
        print(f"Total compilation time: {total:.1f}s")


def compile_all_kernels(
    cache, config, seq_len=1024, fa_bfp16=True, fused=True, with_connector=True
):
    """Pre-compile every unique vision-encoder kernel config to the cache.

    with_connector: also compile the `gemm_connector` ELF (A3-5 Step 4, the
        64x12288x960 modality projection). Needed for
        `run_vit_encoder(do_connector=True)`; pass False to save ~1 ELF of
        compile time when only the encoder's post_ln is wanted.

    fused: if True (A3-6b default), compile the two fused multi-launch ELFs
        (vit_ln_qkv + vit_o_ffn) plus flash_attn — 3 ELFs, 3 dispatches/layer.
        If False, compile the 6 unfused per-op ELFs (10 dispatches/layer, the
        frozen smolvla-vision-unfused-v1 path) for A/B comparison + diagnosis.

    fa_bfp16: FlashAttention microkernel (attn_npu2.o) block-float mode.
        BFP16=True (default) is the only WORKING FA build at this shape: BFP16=False
        (native aie2p bf16 8x8x8 mmul) was tried to remove the systematic
        attention bias but the FA kernel's L1 tiling is sized for the BFP16 mmul
        and the native mmul overflows/mismatches → runtime hang
        (ERT_CMD_STATE_TIMEOUT). Kept as a flag purely to document the experiment.

    Order matters for the GEMM ELFs: `compile_gemm_mm` writes the DIM-baked
    `mm.o` into CWD, then `compile_and_cache` → `prepare_air_project` wipes
    air_project/ fresh and copies the current mm.o into it. So each GEMM ELF's
    mm.o is compiled immediately before its compile_and_cache (mirrors
    the backbone port's attention compile path). The LayerNorm /
    GELU / FA ELFs don't link mm.o (a stale copy staged into their air_project
    is harmless).
    """
    from shared.infra.external_kernels import compile_gemm_mm
    from matrix_multiplication.bf16_in_bf16_out.run import build_module as build_gemm
    from layer_norm.layer_norm import build_module as build_layer_norm
    from gelu.gelu import build_module as build_gelu
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as build_attn,
    )

    emb_dim = config.emb_dim
    hidden_dim = config.hidden_dim
    n_heads = config.n_heads
    n_kv_heads = config.n_heads  # MHA, no GQA
    head_dim = config.head_dim

    print(f"\n{'='*60}")
    print(
        f"Compiling vision-encoder kernels (seq_len={seq_len}, "
        f"{'FUSED 3-ELF' if fused else 'UNFUSED 6-ELF'})..."
    )
    print(f"{'='*60}\n")

    if fused:
        _compile_fused_kernels(
            cache, config, seq_len, fa_bfp16, with_connector=with_connector
        )
        return

    # --- 1. Projection / MLP GEMMs (one ELF per distinct shape) ---
    for name, s in _GEMM_SHAPES.items():
        print(
            f"  Compiling {name}: {s['m']}x{s['k']}x{s['n']} (drain tile_m={s['tile_m']} tile_n={s['tile_n']})"
        )
        compile_gemm_mm(
            tile_m=s["tile_m"],
            tile_n=s["tile_n"],
            tile_k_l1=s["tile_k_l1"],
            sym_suffix="",
            out_name="mm.o",
        )
        mod = build_gemm(
            s["m"],
            s["k"],
            s["n"],
            s["tile_m"],
            s["tile_k_l2"],
            s["tile_k_l1"],
            s["tile_n"],
            8,  # herd_m
            4,  # herd_n
            bfloat16,
            bfloat16,
            arch="aie2p",
            emit_external_call=True,
        )
        cache.compile_and_cache(
            name, mod, {"verbose": cache.verbose, **_gemm_backend()}
        )

    # --- 2. Affine LayerNorm (ln1, ln2, post_layernorm all 1024x768) ---
    print(f"  Compiling layer_norm: {seq_len}x{emb_dim} affine (herd_x=8)")
    ln_mod = build_layer_norm(seq_len, emb_dim, bfloat16, herd_x=8)
    cache.compile_and_cache(
        "layer_norm", ln_mod, {"verbose": cache.verbose, **_ln_backend()}
    )

    # --- 3. GELU-tanh (1D, N = seq_len * hidden_dim) ---
    gelu_n = seq_len * hidden_dim
    print(f"  Compiling gelu: N={gelu_n} (tile_n=4096, herd 8x2)")
    gelu_mod = build_gelu(gelu_n, 4096, bfloat16, herd_x=8, herd_y=2)
    cache.compile_and_cache(
        "gelu", gelu_mod, {"verbose": cache.verbose, **_gelu_backend()}
    )

    # --- 4. Non-causal FlashAttention (12q/12kv MHA, head_dim=64) ---
    # 12 even heads → num_heads_per_unroll=2 divides evenly → FILLS the full 8x4
    # array (A3-4: 1.58x placement win vs hpu=1). Scale 1/sqrt(dk)=1/8 is applied
    # inside the kernel, matching SigLIP attn_scale — Q is NOT pre-scaled.
    num_heads_per_unroll = 2
    num_q_tiles = 4
    assert n_heads % num_heads_per_unroll == 0
    assert num_heads_per_unroll * num_q_tiles <= 8
    print(
        f"  Compiling flash_attn: non-causal, seq={seq_len}, {n_heads}Q/{n_kv_heads}KV, "
        f"head_dim={head_dim}, hpu={num_heads_per_unroll}"
    )
    attn_mod = build_attn(
        lk=seq_len,
        lkp=head_dim,
        lq=seq_len,
        lqp=256,
        dk=head_dim,
        dv=head_dim,
        num_q_tiles=num_q_tiles,
        num_cascade_stages=4,
        num_heads=n_heads,
        num_kv_heads=n_kv_heads,
        causal=False,
        num_heads_per_unroll=num_heads_per_unroll,
    )
    # Pre-build attn_npu2.o with the chosen precision mode and force=True. The
    # compile_and_cache below calls prepare_air_project → compile_all_external_kernels,
    # which rebuilds attn_npu2.o only if absent (force=False) — so this force=True
    # build wins and its .o is the one linked into the FA ELF. fa_bfp16=False gives
    # the native aie2p bf16 mmul (no systematic block-float attention bias).
    from shared.infra.external_kernels import compile_attn_npu2

    compile_attn_npu2(head_dim=head_dim, bfp16=fa_bfp16, force=True)
    print(f"    (FA microkernel BFP16={fa_bfp16})")
    cache.compile_and_cache(
        "flash_attn", attn_mod, {**_ATTN_BACKEND_KWARGS, "verbose": cache.verbose}
    )

    # --- 5. Connector projection GEMM (A3-5 Step 4) ---
    if with_connector:
        _compile_connector_gemm(cache)

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} vision kernels compiled to {cache.cache_dir}/")
    if cache.profiler.enabled:
        total = sum(cache.profiler.compile_times.values())
        print(f"Total compilation time: {total:.1f}s")


# ---------------------------------------------------------------------------
# Per-kernel NPU run helpers
# ---------------------------------------------------------------------------


def _run_gemm(cache, name, A, B, M, N, bo_key):
    """y = A @ B on NPU (bf16-in, bf16-out). A:(M,K), B:(K,N). B is the weight
    (static, written once per bo_key)."""
    A = np.ascontiguousarray(np.asarray(A, dtype=bfloat16)).reshape(-1)
    B = np.ascontiguousarray(np.asarray(B, dtype=bfloat16)).reshape(-1)
    C = np.zeros(M * N, dtype=bfloat16)
    res = cache.load_and_run(
        name,
        _gemm_backend(),
        A,
        B,
        C,
        output_indices=[2],
        static_input_indices={1},
        intermediate_indices={2},  # C is kernel-overwritten; skip host upload
        bo_key=bo_key,
    )
    return res[2].reshape(M, N)


def _run_layer_norm(cache, x, weight, bias, M, N, bo_key):
    """Affine LayerNorm on NPU. weight/bias:(N,) packed into a flat [2N] buffer
    ([0:N]=weight, [N:2N]=bias) — the kernel reads it as one DMA."""
    x = np.ascontiguousarray(np.asarray(x, dtype=bfloat16)).reshape(-1)
    param = np.concatenate(
        [np.asarray(weight, dtype=bfloat16), np.asarray(bias, dtype=bfloat16)]
    ).astype(bfloat16)
    out = np.zeros(M * N, dtype=bfloat16)
    res = cache.load_and_run(
        "layer_norm",
        _ln_backend(),
        x,
        param,
        out,
        output_indices=[2],
        static_input_indices={1},
        intermediate_indices={2},  # out is kernel-overwritten; skip host upload
        bo_key=bo_key,
    )
    return res[2].reshape(M, N)


def _run_gelu(cache, x, M, N, bo_key):
    """GELU-tanh elementwise on NPU. x:(M,N) → flat [M*N] 1D kernel."""
    x = np.ascontiguousarray(np.asarray(x, dtype=bfloat16)).reshape(-1)
    out = np.zeros(M * N, dtype=bfloat16)
    res = cache.load_and_run(
        "gelu",
        _gelu_backend(),
        x,
        out,
        output_indices=[1],
        intermediate_indices={1},  # out is kernel-overwritten; skip host upload
        bo_key=bo_key,
    )
    return res[1].reshape(M, N)


def _run_connector(cache, post_ln, connector_w, config, bo_key="gemm_connector"):
    """Connector / modality projection on NPU (A3-5 Step 4).

    post_ln (1024, 768) -> pixel_shuffle (host reshape, space-to-depth factor 4,
    bit-exact vs HF) -> (64, 12288) -> GEMM against connector_w (12288, 960),
    no bias -> (64, 960).

    Returns the RAW connector output. lerobot's `embed_prefix` multiplies it by
    sqrt(960) AFTER `embed_image` returns, so this must NOT pre-apply that scale
    (the oracle keeps both: `connector` raw and `connector_scaled`).
    """
    from smolvla_cpu_helpers import pixel_shuffle

    shuffled = pixel_shuffle(
        np.asarray(post_ln, dtype=np.float32), scale_factor=_PIXEL_SHUFFLE_FACTOR
    )  # (64, 12288)
    s = _CONNECTOR_GEMM
    assert shuffled.shape == (s["m"], s["k"]), (shuffled.shape, (s["m"], s["k"]))
    out = _run_gemm(
        cache, "gemm_connector", shuffled, connector_w, s["m"], s["n"], bo_key=bo_key
    )
    return np.asarray(out, dtype=np.float32)


def _run_flash_attention(cache, q, k, v, config, seq_len):
    """Non-causal MHA on NPU via FlashAttention. q/k/v:(seq, n_heads*head_dim)
    seq-first (head h occupies columns [h*hd:(h+1)*hd]). Returns (seq, emb)."""
    n_heads = config.n_heads
    head_dim = config.head_dim
    q_attn = np.ascontiguousarray(np.asarray(q, dtype=bfloat16))
    k_attn = np.ascontiguousarray(np.asarray(k, dtype=bfloat16))
    v_attn = np.ascontiguousarray(np.asarray(v, dtype=bfloat16))
    out = np.zeros((seq_len, n_heads * head_dim), dtype=bfloat16)
    res = cache.load_and_run(
        "flash_attn",
        _ATTN_BACKEND_KWARGS,
        q_attn,
        k_attn,
        v_attn,
        out,
        output_indices=[3],
        intermediate_indices={3},  # out is kernel-overwritten; skip host upload
        bo_key="flash_attn",
    )
    return res[3].reshape(seq_len, n_heads * head_dim)


# ---------------------------------------------------------------------------
# A3-6b fused block runner (Lever 1+2+3): 3 dispatches/layer, static weight BOs
# ---------------------------------------------------------------------------


def run_vit_block_fused(
    x_bf16, lw, config, cache, layer_idx=0, verbose=False, attn_mode="flash"
):
    """Execute one SigLIP encoder layer via the two fused ELFs + FA (A3-6b).

    3 NPU dispatches: vit_ln_qkv -> flash_attn -> vit_o_ffn. Per-layer weight
    BOs (LN params, wq/wk/wv/wo, biases, fc1/fc2) are pre-loaded once via
    static_input_indices + bo_key=f"...L{layer_idx}" and skipped on re-upload;
    every scratch/output buffer is intermediate (kernel-overwritten, no host
    upload). ALL bias-adds + both residuals run on-device inside the ELFs — no
    host f32 glue. Mirrors the per-layer runner the sibling LLM ports use.

    attn_mode: "flash" (default) = FlashAttention ELF. "cpu" = host MHA (diag).
    """
    seq_len = x_bf16.shape[0]
    emb = config.emb_dim
    hidden = config.hidden_dim
    n_heads = config.n_heads
    head_dim = config.head_dim

    _cache = getattr(run_vit_block_fused, "_arg_cache", {})
    run_vit_block_fused._arg_cache = _cache

    def _pack(w, b):
        return np.concatenate(
            [np.asarray(w, dtype=bfloat16), np.asarray(b, dtype=bfloat16)]
        ).astype(bfloat16)

    def z2(cols):
        return np.zeros((seq_len, cols), dtype=bfloat16)

    # ---- 1. vit_ln_qkv: LN1 + Q/K/V GEMM + Q/K/V bias -> q_b, k_b, v_b ----
    ln_key = f"vit_ln_qkv_L{layer_idx}"
    if ln_key not in _cache:
        _cache[ln_key] = [
            None,  # arg0 x_in (dynamic)
            _pack(lw.ln1_w, lw.ln1_b),  # arg1 LN1 param
            z2(emb),  # arg2 normed
            np.asarray(lw.wq, dtype=bfloat16).reshape(emb, emb),  # arg3
            z2(emb),  # arg4 q_raw
            np.asarray(lw.wk, dtype=bfloat16).reshape(emb, emb),  # arg5
            z2(emb),  # arg6 k_raw
            np.asarray(lw.wv, dtype=bfloat16).reshape(emb, emb),  # arg7
            z2(emb),  # arg8 v_raw
            np.asarray(lw.bq, dtype=bfloat16).reshape(emb),  # arg9
            np.asarray(lw.bk, dtype=bfloat16).reshape(emb),  # arg10
            np.asarray(lw.bv, dtype=bfloat16).reshape(emb),  # arg11
            z2(emb),  # arg12 q_b (out)
            z2(emb),  # arg13 k_b (out)
            z2(emb),  # arg14 v_b (out)
        ]
    ln_args = _cache[ln_key]
    ln_args[0] = np.ascontiguousarray(np.asarray(x_bf16, dtype=bfloat16)).reshape(-1)
    res = cache.load_and_run(
        "vit_ln_qkv",
        _vit_ln_qkv_backend(),
        *ln_args,
        output_indices=[12, 13, 14],
        static_input_indices={1, 3, 5, 7, 9, 10, 11},  # LN param + weights + biases
        intermediate_indices={2, 4, 6, 8, 12, 13, 14},  # scratch + outputs
        bo_key=ln_key,
    )
    q = res[12].reshape(seq_len, emb)
    k = res[13].reshape(seq_len, emb)
    v = res[14].reshape(seq_len, emb)

    # ---- 2. Attention ----
    if attn_mode == "cpu":
        from smolvla_cpu_helpers import mha_bidirectional

        attn = mha_bidirectional(
            q.astype(np.float32),
            k.astype(np.float32),
            v.astype(np.float32),
            n_heads,
            head_dim,
            config.attn_scale,
        ).astype(bfloat16)
    else:
        attn = _run_flash_attention(cache, q, k, v, config, seq_len)

    # ---- 3. vit_o_ffn: O + bias + residual + LN2 + fc1 + bias + GELU + fc2
    #        + bias + residual -> output ----
    offn_key = f"vit_o_ffn_L{layer_idx}"
    if offn_key not in _cache:
        _cache[offn_key] = [
            None,  # arg0 attn (dynamic)
            np.asarray(lw.wo, dtype=bfloat16).reshape(emb, emb),  # arg1
            z2(emb),  # arg2 o_raw
            np.asarray(lw.bo, dtype=bfloat16).reshape(emb),  # arg3 bo
            z2(emb),  # arg4 o_b
            None,  # arg5 x_res (dynamic = block input)
            z2(emb),  # arg6 res1
            _pack(lw.ln2_w, lw.ln2_b),  # arg7 LN2 param
            z2(emb),  # arg8 normed2
            np.asarray(lw.w_fc1, dtype=bfloat16).reshape(emb, hidden),  # arg9
            z2(hidden),  # arg10 fc1_raw
            np.asarray(lw.b_fc1, dtype=bfloat16).reshape(hidden),  # arg11
            z2(hidden),  # arg12 fc1_b
            z2(hidden),  # arg13 gelu_out
            np.asarray(lw.w_fc2, dtype=bfloat16).reshape(hidden, emb),  # arg14
            z2(emb),  # arg15 fc2_raw
            np.asarray(lw.b_fc2, dtype=bfloat16).reshape(emb),  # arg16
            z2(emb),  # arg17 fc2_b
            z2(emb),  # arg18 output
        ]
    offn_args = _cache[offn_key]
    offn_args[0] = np.ascontiguousarray(np.asarray(attn, dtype=bfloat16)).reshape(-1)
    offn_args[5] = np.ascontiguousarray(np.asarray(x_bf16, dtype=bfloat16)).reshape(-1)
    res = cache.load_and_run(
        "vit_o_ffn",
        _vit_o_ffn_backend(),
        *offn_args,
        output_indices=[18],
        static_input_indices={1, 3, 7, 9, 11, 14, 16},  # wo,bo,LN2,fc1,bfc1,fc2,bfc2
        intermediate_indices={2, 4, 6, 8, 10, 12, 13, 15, 17, 18},  # scratch+out
        bo_key=offn_key,
    )
    return res[18].reshape(seq_len, emb)


# ---------------------------------------------------------------------------
# One SigLIP encoder block (pre-norm) — UNFUSED reference (10 dispatches/layer)
# ---------------------------------------------------------------------------


def run_vit_block(
    x_bf16, lw, config, cache, layer_idx=0, verbose=False, attn_mode="flash"
):
    """Execute one SigLIP encoder layer on NPU. x_bf16:(seq, emb). Returns
    output bf16 (seq, emb).

    Heavy ops (LayerNorm, q/k/v/o GEMM, FA, fc1/fc2 GEMM, GELU) run on NPU; the
    bias-adds (every Linear has a bias) and the two residual adds run on host in
    f32 — a correctness-first shortcut (A3-6 fuses them). See module docstring.

    attn_mode: "flash" (default) = non-causal FlashAttention ELF on NPU.
        "cpu" = mha_bidirectional on host (diagnostic / documented fallback).
    """
    seq_len = x_bf16.shape[0]
    emb = config.emb_dim
    hidden = config.hidden_dim

    if verbose:
        print(
            f"  ViT layer {layer_idx}: LN1 -> qkv -> FA -> o -> res -> LN2 -> fc1 -> gelu -> fc2 -> res"
        )

    # --- Attention block (pre-norm) ---
    h = _run_layer_norm(
        cache, x_bf16, lw.ln1_w, lw.ln1_b, seq_len, emb, bo_key=f"ln1_L{layer_idx}"
    )

    q = _run_gemm(cache, "gemm_qkvo", h, lw.wq, seq_len, emb, bo_key=f"wq_L{layer_idx}")
    k = _run_gemm(cache, "gemm_qkvo", h, lw.wk, seq_len, emb, bo_key=f"wk_L{layer_idx}")
    v = _run_gemm(cache, "gemm_qkvo", h, lw.wv, seq_len, emb, bo_key=f"wv_L{layer_idx}")
    # Host bias-add (every SigLIP proj has a bias), cast back to bf16 for FA.
    q = (q.astype(np.float32) + lw.bq.astype(np.float32)).astype(bfloat16)
    k = (k.astype(np.float32) + lw.bk.astype(np.float32)).astype(bfloat16)
    v = (v.astype(np.float32) + lw.bv.astype(np.float32)).astype(bfloat16)

    if attn_mode == "cpu":
        from smolvla_cpu_helpers import mha_bidirectional

        attn = mha_bidirectional(
            q.astype(np.float32),
            k.astype(np.float32),
            v.astype(np.float32),
            config.n_heads,
            config.head_dim,
            config.attn_scale,
        ).astype(bfloat16)
    else:
        attn = _run_flash_attention(cache, q, k, v, config, seq_len)

    o = _run_gemm(
        cache, "gemm_qkvo", attn, lw.wo, seq_len, emb, bo_key=f"wo_L{layer_idx}"
    )
    o = o.astype(np.float32) + lw.bo.astype(np.float32)

    # Residual (host f32).
    x = x_bf16.astype(np.float32) + o
    x_bf16 = x.astype(bfloat16)

    # --- MLP block (pre-norm) ---
    h = _run_layer_norm(
        cache, x_bf16, lw.ln2_w, lw.ln2_b, seq_len, emb, bo_key=f"ln2_L{layer_idx}"
    )

    h1 = _run_gemm(
        cache, "gemm_fc1", h, lw.w_fc1, seq_len, hidden, bo_key=f"fc1_L{layer_idx}"
    )
    h1 = (h1.astype(np.float32) + lw.b_fc1.astype(np.float32)).astype(bfloat16)

    g = _run_gelu(cache, h1, seq_len, hidden, bo_key=f"gelu_L{layer_idx}")

    h2 = _run_gemm(
        cache, "gemm_fc2", g, lw.w_fc2, seq_len, emb, bo_key=f"fc2_L{layer_idx}"
    )
    h2 = h2.astype(np.float32) + lw.b_fc2.astype(np.float32)

    # Residual (host f32).
    x = x.astype(np.float32) + h2
    return x.astype(bfloat16)


# ---------------------------------------------------------------------------
# Full 12-layer encoder
# ---------------------------------------------------------------------------


def run_vit_encoder(
    patch_embed_or_pixel,
    weights,
    config,
    cache,
    return_per_layer=False,
    do_connector=False,
    verbose=False,
    attn_mode="flash",
    fused=True,
):
    """Run the full 12-layer SigLIP ViT encoder on NPU.

    Args:
        patch_embed_or_pixel: either pixel_values (3, 512, 512) — in which case
            the host im2col patch-embed + position-embedding add produces the
            (1024, 768) patch embedding — or a precomputed patch_embed (1024,
            768) fed directly.
        weights: VisionWeights.
        config: SigLIPVisionConfig.
        cache: KernelCache with vision kernels pre-compiled.
        return_per_layer: if True, collect each layer's output.
        do_connector: if True (A3-5 Step 4), also run the connector — host
            pixel-shuffle (1024,768) -> (64,12288) then the `gemm_connector` ELF
            -> (64, 960). Requires compile_all_kernels(with_connector=True).
            The returned `connector` is the RAW projection; lerobot applies the
            sqrt(960) scale afterwards in embed_prefix (do not double-apply).

    Returns:
        dict with:
            post_ln: (1024, 768) f32 — LayerNorm(post_layernorm) of the last layer.
            layer_hidden: list of 12 (1024, 768) bf16 [if return_per_layer].
            connector: (64, 960) f32 [if do_connector].
    """
    inp = np.asarray(patch_embed_or_pixel)
    if inp.ndim == 3:
        # pixel_values (3, H, W) → host im2col patch-embed + pos-embed add.
        x = im2col_patch_embed(
            inp,
            weights.patch_w,
            weights.patch_b,
            weights.pos_embed,
            config.patch_size,
        )  # (1024, 768) f32
    else:
        x = inp.astype(np.float32)
    x_bf16 = x.astype(bfloat16)
    seq_len, emb = x_bf16.shape

    _block = run_vit_block_fused if fused else run_vit_block
    per_layer = []
    for layer_idx, lw in enumerate(weights.layers):
        if verbose:
            print(f"\n--- ViT layer {layer_idx}/{len(weights.layers) - 1} ---")
        x_bf16 = _block(
            x_bf16,
            lw,
            config,
            cache,
            layer_idx=layer_idx,
            verbose=verbose,
            attn_mode=attn_mode,
        )
        if return_per_layer:
            per_layer.append(x_bf16)

    # post_layernorm on NPU (affine LayerNorm, same ELF).
    post_ln = _run_layer_norm(
        cache,
        x_bf16,
        weights.post_ln_w,
        weights.post_ln_b,
        seq_len,
        emb,
        bo_key="post_ln",
    ).astype(np.float32)

    result = {"post_ln": post_ln}
    if return_per_layer:
        result["layer_hidden"] = per_layer
    if do_connector:
        result["connector"] = _run_connector(
            cache, post_ln, weights.connector_w, config
        )
    return result


# This module is a library: the single CLI entry point is smolvla_inference.py
# (`--compile-only` calls compile_all_kernels above), mirroring the siblings.
