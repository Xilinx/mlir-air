# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Gemma3-4B (text) Q4NX prefill in mlir-air.
#
# Runs the prefill on the AMD NPU2: Q4NX weights, host dequant Q4NX->bf16 at
# load, then RMSNorm / Q-K-V GEMM / per-head QK-norm / dual-theta RoPE / GQA
# flash attention / GELU-tanh GLU / all seven projections ON THE NPU with
# RESIDENT weight BOs, and the tied LM head as an on-device GEMV.
#
# This is the Gemma sibling of llama32_1b_q4nx_prefill.py and is structured
# like qwen3_4b_prefill.py (the closest architecture: same emb_dim=2560, QK-norm,
# decoupled O proj, and an FFN too wide for a merged O+FFN ELF).
#
# The five Gemma deltas vs Llama, and where each lands:
#   1. FOUR norms/layer. Gemma norms the attention and MLP outputs as well as
#      their inputs (o_proj -> post_attn_norm -> +residual, and
#      down_proj -> post_ffn_norm -> +residual), so the O and Down tails carry an
#      extra weighted_rms_norm slice vs the Llama/Qwen shape. eps=1e-6.
#   2. GELU-tanh GLU instead of SwiGLU -> the gelu_and_mul ELF.
#   3. Per-head QK-norm -> the shared rms_qkv_qknorm_rope 8-launch ELF.
#   4. Dual-theta RoPE + a 1024 sliding window on 5 of every 6 layers -> two
#      RoPE LUTs and two flash-attention ELFs, selected per layer by
#      gemma3_4b_q4nx_weights.is_global_layer.
#   5. head_dim=256 -> the head-first FA path at its own tiling (see
#      shared/infra/fa_headfirst._FA_TILING); vocab 262208 -> 17 LM-head
#      partitions.
#
# Gate: first prompt token argmax 9079 (" Paris") for "The capital of France is".
#
# Weight source (env-overridable):
#   Q4NX_MODEL_SOURCE : the model.q4nx bundle -- HF repo id or a local dir/file.
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_PROG = str(_HERE.parent.parent)  # programming_examples
_LLMS = str(_HERE.parent)  # llms
for _p in (_PROG, _LLMS, str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gemma3_4b_q4nx_weights import (  # noqa: E402
    D,
    DH,
    DQ,
    DK,
    INTER,
    NUM_LAYERS,
    N_Q_HEADS,
    N_KV_HEADS,
    RMS_EPS,
    ROPE_GLOBAL_LINEAR_FACTOR,
    ROPE_GLOBAL_THETA,
    ROPE_LOCAL_THETA,
    SLIDING_WINDOW,
    VOCAB,
    _bf,
    is_global_layer,
)

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma3-4B-NPU2")

# <bos> + "The capital of France is" (Gemma3 tokenizer) -> 9079 " Paris".
PROMPT = [2, 818, 5279, 529, 7001, 563]
EXPECT_FIRST = 9079

# On-device LM head GEMV. VOCAB=262208 is not a multiple of the 16384-row
# partition the Llama example uses, so pad to 17 partitions (17*16384=278528).
_LM_N_PART = 16384
_LM_N_PARTITIONS = 17

# Cache keys of the two per-layer attention ELFs (global = plain causal,
# local = the same kernel with the 1024 sliding-window mask).
_FA_GLOBAL = "flash_attn_global"
_FA_LOCAL = "flash_attn_local"


def _elf_backend(instance_name, tiling=(2, 2)):
    return {
        "verbose": False,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": instance_name,
        "runtime_loop_tiling_sizes": list(tiling),
    }


_RMS_QKV_BACKEND = _elf_backend("rms_qkv_qknorm_rope")
_ONORM_BACKEND = _elf_backend("o_norm_res_norm")
_GATE_BACKEND = _elf_backend("gate")
_UP_BACKEND = _elf_backend("up")
_DOWN_BACKEND = _elf_backend("down_norm_add")
# The GELU ELF's instance name must match the top func name in build_module_2d.
_GELU_BACKEND = {
    "verbose": False,
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "gelu_and_mul_2d",
}


# ---------------------------------------------------------------------------
# eps: every Gemma RMSNorm (the 4 layer norms, the QK norms and the final norm)
# uses 1e-6, but weighted_rms_norm bakes a module-level EPS=1e-5 at build time.
# Override it around each build, the pattern rms_qkv_qknorm_rope_multi uses.
# ---------------------------------------------------------------------------


class _rms_eps:
    """Context manager: build weighted_rms_norm slices at Gemma's eps."""

    def __init__(self, eps=RMS_EPS):
        self.eps = eps

    def __enter__(self):
        import weighted_rms_norm.weighted_rms_norm as wrn

        self._wrn = wrn
        self._saved = wrn.EPS
        wrn.EPS = self.eps

    def __exit__(self, *exc):
        self._wrn.EPS = self._saved
        return False


def gemm_spec(m, k, n, precision="high"):
    """Per-GEMM build recipe for one Gemma3-4B shape, from the kernel registry.

    All seven Gemma prefill shapes at M=2048 resolve to high-precision
    fused-cast (see kernel_registry/details/GEMM_bf16_in_bf16_out.json)."""
    from shared.builders.gemm_builder import gemm_registry_config

    return gemm_registry_config(m, k, n, "bf16", precision)


def _build_gemm_ir(m, k, n, spec, herd_m=8, herd_n=4):
    from shared.builders.gemm_builder import _build_gemm_module

    return str(
        _build_gemm_module(
            m,
            k,
            n,
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
            herd_m,
            herd_n,
            **dict(spec["build_kwargs"]),
        )
    )


def _gemm_externs(spec):
    sfx = spec["sym_suffix"]
    return {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }


def _gemm_amap(inp, w, out, sc):
    """arg_map for one GEMM slice: fused-cast writes f32 scratch then casts."""
    return {0: inp, 1: w, 2: sc, 3: out} if sc is not None else {0: inp, 1: w, 2: out}


# ---------------------------------------------------------------------------
# ELF builders
# ---------------------------------------------------------------------------


def build_o_norm_res_norm_module(seq_len, herd_m=8, herd_n=4):
    """O proj (DECOUPLED) + post-attention norm + residual + pre-FFN norm.

    The Gemma attention tail: unlike Llama/Qwen (O -> +residual -> ffn_norm),
    Gemma normalizes the projection BEFORE adding the residual.

      %arg0 attn_out    (seq, DQ)      DECOUPLED (n_heads*head_dim = 2048)
      %arg1 wo          (DQ, D)        static
      %arg2 proj        (seq, D)
      %arg3 post_attn_w (D,)           static
      %arg4 proj_n      (seq, D)
      %arg5 x_resid     (seq, D)
      %arg6 res1        (seq, D)       OUTPUT (feeds down_norm_add)
      %arg7 pre_ffn_w   (D,)           static
      %arg8 normed2     (seq, D)       OUTPUT (feeds gate/up)
      [+ f32 C-scratch tail for the fused-cast O GEMM]
    """
    from shared.infra.stitching import (
        _wrap_ir_in_launch,
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
        build_residual_add_2d_ir,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    o_spec = gemm_spec(seq_len, DQ, D)
    print(f"  [1/4] O GEMM ({o_spec['method']}) {seq_len}x{DQ}x{D} (DECOUPLED)...")
    o_ir = _build_gemm_ir(seq_len, DQ, D, o_spec, herd_m, herd_n)
    with _rms_eps():
        print(f"  [2/4] post-attention RMSNorm (eps={RMS_EPS:g})...")
        post_attn_ir = _wrap_ir_in_launch(
            str(build_rms(seq_len, D, bfloat16, 16, herd_x=8))
        )
        print("  [3/4] Residual Add...")
        add_ir = build_residual_add_2d_ir(seq_len, D)
        print(f"  [4/4] pre-FFN RMSNorm (eps={RMS_EPS:g})...")
        pre_ffn_ir = _wrap_ir_in_launch(
            str(build_rms(seq_len, D, bfloat16, 16, herd_x=8))
        )

    scratch_args, scratch_for = alloc_gemm_scratch([(o_spec, seq_len, D)], 9)

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{DQ}xbf16>"),
        FuncArg("%arg1", f"memref<{DQ}x{D}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg3", f"memref<{D}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg5", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg7", f"memref<{D}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{D}xbf16>"),
    ]
    slices = [
        KernelSlice(
            o_ir,
            "og",
            _gemm_amap(0, 1, 2, scratch_for[0]),
            extern_syms=_gemm_externs(o_spec),
        ),
        KernelSlice(post_attn_ir, "pa", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(add_ir, "ra", {0: 4, 1: 5, 2: 6}, private_from=False),
        KernelSlice(pre_ffn_ir, "pf", {0: 6, 1: 7, 2: 8}, private_from=False),
    ]
    module = stitch_elf(
        "o_norm_res_norm",
        base_args,
        slices,
        scratch_args=scratch_args,
        debug_dump_path="/tmp/debug_gemma_o_norm_res_norm.mlir",
    )
    print(f"  o_norm_res_norm module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def _build_single_gemm_elf(name, sym, seq_len, k_dim, n_dim, herd_m=8, herd_n=4):
    """Standalone single-GEMM ELF (Gate or Up): arg0 in, arg1 weight, arg2 out."""
    from shared.infra.stitching import (
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
    )

    g_spec = gemm_spec(seq_len, k_dim, n_dim)
    print(
        f"  [{name}] GEMM ({g_spec['method']}) {seq_len}x{k_dim}x{n_dim} "
        f"(tk_l2={g_spec['tile_k_l2']}, tn={g_spec['tile_n']})..."
    )
    gemm_ir = _build_gemm_ir(seq_len, k_dim, n_dim, g_spec, herd_m, herd_n)
    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{k_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{k_dim}x{n_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{n_dim}xbf16>"),
    ]
    scratch_args, scratch_for = alloc_gemm_scratch([(g_spec, seq_len, n_dim)], 3)
    slices = [
        KernelSlice(
            gemm_ir,
            sym,
            _gemm_amap(0, 1, 2, scratch_for[0]),
            extern_syms=_gemm_externs(g_spec),
        )
    ]
    module = stitch_elf(name, base_args, slices, scratch_args=scratch_args)
    print(f"  {name} module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def _gelu_tile_n(seq_len, hidden_dim, herd_x=8):
    """Largest tile_n in the placeable band that divides the per-tile span.

    Same L1/BD sweet spot as the SwiGLU sibling (qwen3_4b_prefill._swiglu_tile_n):
    too large overflows L1, too small exhausts the BD pool."""
    span = (seq_len * hidden_dim) // herd_x
    for t in range(5120, 1024, -64):
        if span % t == 0:
            return t
    raise RuntimeError(f"No GELU tile_n for seq={seq_len} hidden={hidden_dim}")


def build_gelu_mul_module(seq_len, hidden_dim, herd_x=8, herd_y=1):
    """Standalone NPU GeGLU ELF: gelu_tanh(gate) * up -> (seq, hidden)."""
    from gelu_and_mul.gelu_and_mul import build_module_2d as build_gelu

    tile_n = _gelu_tile_n(seq_len, hidden_dim, herd_x)
    print(
        f"  [gelu_mul] GELU-tanh GLU {seq_len}x{hidden_dim} (tile_n={tile_n}, "
        f"iters={seq_len * hidden_dim // (tile_n * herd_x)})..."
    )
    module = build_gelu(seq_len, hidden_dim, tile_n, bfloat16, herd_x, herd_y)
    print(f"  gelu_mul module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def build_down_norm_add_module(seq_len, herd_m=8, herd_n=4):
    """Down proj + post-FFN norm + residual add (the Gemma MLP tail).

    %arg0 act         (seq, INTER)   gelu_tanh(gate)*up
    %arg1 w_down      (INTER, D)     static
    %arg2 down        (seq, D)
    %arg3 post_ffn_w  (D,)           static
    %arg4 down_n      (seq, D)
    %arg5 res1        (seq, D)
    %arg6 output      (seq*D,)       OUTPUT (1D)
    [+ f32 C-scratch tail for the fused-cast Down GEMM]
    """
    from shared.infra.stitching import (
        _wrap_ir_in_launch,
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
        build_add_2d_to_1d_ir,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    n_total = seq_len * D
    d_spec = gemm_spec(seq_len, INTER, D)
    print(f"  [1/3] Down GEMM ({d_spec['method']}) {seq_len}x{INTER}x{D}...")
    down_ir = _build_gemm_ir(seq_len, INTER, D, d_spec, herd_m, herd_n)
    with _rms_eps():
        print(f"  [2/3] post-FFN RMSNorm (eps={RMS_EPS:g})...")
        post_ffn_ir = _wrap_ir_in_launch(
            str(build_rms(seq_len, D, bfloat16, 16, herd_x=8))
        )
    print("  [3/3] FFN Add (2D -> 1D)...")
    add_ir = build_add_2d_to_1d_ir(seq_len, D)

    scratch_args, scratch_for = alloc_gemm_scratch([(d_spec, seq_len, D)], 7)

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{INTER}xbf16>"),
        FuncArg("%arg1", f"memref<{INTER}x{D}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg3", f"memref<{D}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg5", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg6", f"memref<{n_total}xbf16>"),
    ]
    slices = [
        KernelSlice(
            down_ir,
            "dg",
            _gemm_amap(0, 1, 2, scratch_for[0]),
            extern_syms=_gemm_externs(d_spec),
            private_from=True,
        ),
        KernelSlice(post_ffn_ir, "pf2", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(add_ir, "fa", {0: 4, 1: 5, 2: 6}, private_from=False),
    ]
    module = stitch_elf(
        "down_norm_add",
        base_args,
        slices,
        scratch_args=scratch_args,
        debug_dump_path="/tmp/debug_gemma_down_norm_add.mlir",
    )
    print(f"  down_norm_add module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def build_rms_qkv_qknorm_rope(seq_len):
    """The shared 8-launch attention-input ELF at Gemma's shapes and eps."""
    from shared.builders.rms_qkv_qknorm_rope_multi import (
        build_rms_qkv_qknorm_rope_module as _build,
    )

    with _rms_eps():
        return _build(
            seq_len,
            D,
            DQ,
            DK,
            N_Q_HEADS,
            N_KV_HEADS,
            DH,
            qknorm_eps=RMS_EPS,
        )


# ---------------------------------------------------------------------------
# RoPE LUTs (dual theta)
# ---------------------------------------------------------------------------


def rope_luts(seq_len):
    """(local, global) half-split RoPE cos/sin LUTs, each [seq_len, DH].

    Matches gemma3_4b_q4nx_weights.generate_rope_lut but batched over positions:
    row p = [cos(p*inv_freq) (DH/2) ++ sin(p*inv_freq) (DH/2)]. Global layers use
    theta=1e6 with positions divided by ROPE_GLOBAL_LINEAR_FACTOR."""
    half = DH // 2
    pos = np.arange(seq_len, dtype=np.float64)
    out = []
    for theta, lf in (
        (ROPE_LOCAL_THETA, 1.0),
        (ROPE_GLOBAL_THETA, ROPE_GLOBAL_LINEAR_FACTOR),
    ):
        inv = 1.0 / (theta ** (np.arange(half, dtype=np.float64) / half))
        ang = (pos / lf)[:, None] * inv[None, :]
        out.append(np.concatenate([np.cos(ang), np.sin(ang)], axis=-1).astype(bfloat16))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_all_kernels(cache, seq_len, verbose=False):
    """Compile the 8 prefill ELFs. Returns the scratch-index map."""
    print(
        f"\n{'='*60}\nCompiling Gemma3-4B prefill kernels (seq_len={seq_len})...\n{'='*60}\n"
    )

    from shared.infra.external_kernels import (
        compile_gemm_mm,
        compile_rope,
        compile_gelu_and_mul,
    )

    # External microkernels, compiled FIRST so prepare_air_project copies them
    # into air_project/ for every ELF that links them.
    compile_gemm_mm(
        tile_m=32, tile_n=128, tile_k_l1=32, sym_suffix="_m32", out_name="mm_m32.o"
    )
    compile_gemm_mm(
        tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="_m64", out_name="mm_m64.o"
    )
    compile_rope()  # rope_halfsplit.cc; head_dim is a runtime arg
    compile_gelu_and_mul()

    scratch = {}

    print("\n--- rms_qkv_qknorm_rope (RMSNorm + QKV + QK-norm + RoPE, 8 launches) ---")
    mod, scratch["rms_qkv"] = build_rms_qkv_qknorm_rope(seq_len)
    cache.compile_and_cache("rms_qkv_qknorm_rope", mod, _RMS_QKV_BACKEND)

    print("\n--- o_norm_res_norm (O + post-attn norm + residual + pre-FFN norm) ---")
    mod, scratch["o_norm"] = build_o_norm_res_norm_module(seq_len)
    cache.compile_and_cache("o_norm_res_norm", mod, _ONORM_BACKEND)

    print("\n--- gate (Gate GEMM) ---")
    mod, scratch["gate"] = _build_single_gemm_elf("gate", "gg", seq_len, D, INTER)
    cache.compile_and_cache("gate", mod, _GATE_BACKEND)

    print("\n--- up (Up GEMM) ---")
    mod, scratch["up"] = _build_single_gemm_elf("up", "ug", seq_len, D, INTER)
    cache.compile_and_cache("up", mod, _UP_BACKEND)

    print("\n--- gelu_mul (GELU-tanh GLU) ---")
    cache.compile_and_cache(
        "gelu_mul", build_gelu_mul_module(seq_len, INTER), _GELU_BACKEND
    )

    print("\n--- down_norm_add (Down + post-FFN norm + residual) ---")
    mod, scratch["down"] = build_down_norm_add_module(seq_len)
    cache.compile_and_cache("down_norm_add", mod, _DOWN_BACKEND)

    # Two attention ELFs: Gemma alternates 5 sliding-window layers to 1 global.
    from shared.infra.fa_headfirst import compile_headfirst_fa

    print(f"\n--- {_FA_GLOBAL} (head-first FA, head_dim={DH}, causal) ---")
    compile_headfirst_fa(
        cache,
        seq_len,
        N_Q_HEADS,
        N_KV_HEADS,
        DH,
        verbose,
        name=_FA_GLOBAL,
        causal_skip=True,
    )
    print(
        f"\n--- {_FA_LOCAL} (head-first FA, head_dim={DH}, window={SLIDING_WINDOW}) ---"
    )
    compile_headfirst_fa(
        cache,
        seq_len,
        N_Q_HEADS,
        N_KV_HEADS,
        DH,
        verbose,
        window=SLIDING_WINDOW,
        name=_FA_LOCAL,
        causal_skip=True,
    )

    print(f"\n--- lm_head_gemv ({_LM_N_PARTITIONS} x {_LM_N_PART}, K={D}) ---")
    from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module
    from shared.infra.backend_presets import LM_GEMV_BACKEND

    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_module(D, n_partitions=_LM_N_PARTITIONS, n_part=_LM_N_PART),
        {"verbose": verbose, **dict(LM_GEMV_BACKEND)},
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} kernels compiled to {cache.cache_dir}/")
    return scratch


def resolve_scratch(seq_len):
    """Recompute the scratch-arg index map without building any IR (run-only)."""

    def _alloc(specs, base):
        out, nxt = [], base
        for s in specs:
            if s["needs_f32_scratch"]:
                out.append(nxt)
                nxt += 1
            else:
                out.append(None)
        return out

    return {
        # rms_qkv_qknorm_rope: Q/K/V specs, scratch tail starts at arg 17.
        "rms_qkv": _alloc(
            [
                gemm_spec(seq_len, D, DQ),
                gemm_spec(seq_len, D, DK),
                gemm_spec(seq_len, D, DK),
            ],
            17,
        ),
        "o_norm": _alloc([gemm_spec(seq_len, DQ, D)], 9),
        "gate": _alloc([gemm_spec(seq_len, D, INTER)], 3),
        "up": _alloc([gemm_spec(seq_len, D, INTER)], 3),
        "down": _alloc([gemm_spec(seq_len, INTER, D)], 7),
    }


# ---------------------------------------------------------------------------
# The prefill model
# ---------------------------------------------------------------------------


class GemmaQ4nxPrefill:
    """AIR realization of the Gemma3-4B Q4NX causal-LM prefill interface."""

    def __init__(
        self, seq_len=2048, n_layers=NUM_LAYERS, cache_dir=None, verbose=False
    ):
        from shared.infra.cache import KernelCache

        self.seq = seq_len
        self.n_layers = n_layers
        self.MAX_L = seq_len
        self.current_context_length = 0
        # seq_len-specific cache: the ELFs are compiled for this padded context
        # length but KernelCache keys on kernel NAME only, so a shorter build
        # would otherwise be silently reused (-> all-zero logits past its length).
        cache_dir = cache_dir or str(_HERE / f"_q4nx_cache_seq{seq_len}")
        self.cache = KernelCache(cache_dir=cache_dir, verbose=verbose)
        self._verbose = verbose

        self._lut_local, self._lut_global = rope_luts(seq_len)

        force = os.environ.get("Q4NX_FORCE_COMPILE") == "1"
        if not force:
            self.cache.load_manifest()
        cached = set() if force else set(self.cache.artifacts)
        needed = {
            "rms_qkv_qknorm_rope",
            "o_norm_res_norm",
            "gate",
            "up",
            "gelu_mul",
            "down_norm_add",
            _FA_GLOBAL,
            _FA_LOCAL,
            "lm_head_gemv",
        }
        if not needed.issubset(cached):
            self.scratch = compile_all_kernels(self.cache, seq_len, verbose)
        else:
            print(
                "[gemma_prefill] using cached prefill ELFs (skip compile)", flush=True
            )
            self.scratch = resolve_scratch(seq_len)

        from shared.infra.backend_presets import LM_GEMV_BACKEND

        self._lm_backend = dict(LM_GEMV_BACKEND)

        # Per-layer KV cache (roped K + raw V), [MAX_L, n_kv_heads*head_dim].
        self.kv_k = [np.zeros((self.MAX_L, DK), bfloat16) for _ in range(n_layers)]
        self.kv_v = [np.zeros((self.MAX_L, DK), bfloat16) for _ in range(n_layers)]
        self._w = None
        self._rms = None
        self._qk = None
        self._embed = None
        self._final_norm = None
        self._lm_head = None
        self._dev_t = 0.0
        self._op_t = {}

    def _dev(self, fn, *a, tag="?"):
        import time

        t = time.time()
        r = fn(*a)
        dt = time.time() - t
        self._dev_t += dt
        self._op_t[tag] = self._op_t.get(tag, 0.0) + dt
        return r

    # ---- causal_lm interface ----
    def load_weights(self, model=None):
        """Load the Q4NX transformer weights + embed / norms / tied lm_head from
        the self-contained `model.q4nx` bundle, dequantizing Q4NX->bf16 on the
        host at load, then pre-load them into resident per-layer BOs."""
        from gemma3_4b_q4nx_weights import Q4nxModel

        model = model or os.environ.get("Q4NX_MODEL_SOURCE", MODEL_DEFAULT)
        print(f"[gemma_prefill] loading weights from model.q4nx ({model})", flush=True)
        qm = Q4nxModel(model)
        self._w = [qm.layer_weights(k) for k in range(self.n_layers)]
        self._rms = [qm.layer_rms(k) for k in range(self.n_layers)]
        self._qk = [qm.layer_qk_norm(k) for k in range(self.n_layers)]
        self._embed, self._final_norm, self._lm_head = qm.embed_norm_lmhead()
        self._preload()

    # ---- per-ELF calls (single owner of each arg layout) ----
    def _lut_for(self, layer_idx):
        """Per-head-repeated (lut_q, lut_k) for a layer's RoPE theta."""
        lut = self._lut_global if is_global_layer(layer_idx) else self._lut_local
        return (
            np.repeat(lut[: self.seq], N_Q_HEADS, axis=0).flatten(),
            np.repeat(lut[: self.seq], N_KV_HEADS, axis=0).flatten(),
        )

    def _call_rms_qkv(self, k, x_in):
        seq = self.seq
        Wt = self._w[k]
        n_in = self._rms[k][0]
        qn, kn = self._qk[k]
        lut_q, lut_k = self._lut_for(k)
        args = [
            np.asarray(x_in, bfloat16).reshape(seq, D),  # 0 x_in (dynamic)
            np.asarray(n_in, bfloat16).reshape(D),  # 1 input_layernorm
            np.zeros((seq, D), bfloat16),  # 2 normed
            np.asarray(Wt["q"], bfloat16).reshape(D, DQ),  # 3
            np.zeros((seq, DQ), bfloat16),  # 4 q
            np.asarray(Wt["k"], bfloat16).reshape(D, DK),  # 5
            np.zeros((seq, DK), bfloat16),  # 6 k
            np.asarray(Wt["v"], bfloat16).reshape(D, DK),  # 7
            np.zeros((seq, DK), bfloat16),  # 8 v (out)
            np.asarray(qn, bfloat16).reshape(DH),  # 9 q_norm
            np.asarray(kn, bfloat16).reshape(DH),  # 10 k_norm
            np.zeros((seq, DQ), bfloat16),  # 11 q_n
            np.zeros((seq, DK), bfloat16),  # 12 k_n
            lut_q,  # 13
            lut_k,  # 14
            np.zeros((seq, DQ), bfloat16),  # 15 q_roped (out)
            np.zeros((seq, DK), bfloat16),  # 16 k_roped (out)
        ]
        inter = {2, 4, 6, 8, 11, 12, 15, 16}
        for sc, cols in zip(self.scratch["rms_qkv"], (DQ, DK, DK)):
            if sc is not None:
                args.append(np.zeros((seq, cols), np.float32))
                inter.add(sc)
        return self.cache.load_and_run(
            "rms_qkv_qknorm_rope",
            _RMS_QKV_BACKEND,
            *args,
            output_indices=[8, 15, 16],
            static_input_indices={1, 3, 5, 7, 9, 10, 13, 14},
            intermediate_indices=inter,
            bo_key=f"rms_qkv_L{k}",
            shared_nonstatic=True,
        )

    def _call_o_norm(self, k, attn_out, x_resid):
        seq = self.seq
        _n_in, n_pa, n_pf, _n_pf2 = self._rms[k]
        args = [
            np.asarray(attn_out, bfloat16).reshape(seq, DQ),  # 0
            np.asarray(self._w[k]["o"], bfloat16).reshape(DQ, D),  # 1 wo
            np.zeros((seq, D), bfloat16),  # 2 proj
            np.asarray(n_pa, bfloat16).reshape(D),  # 3 post_attention_layernorm
            np.zeros((seq, D), bfloat16),  # 4 proj_n
            np.asarray(x_resid, bfloat16).reshape(seq, D),  # 5 residual
            np.zeros((seq, D), bfloat16),  # 6 res1 (out)
            np.asarray(n_pf, bfloat16).reshape(D),  # 7 pre_feedforward_layernorm
            np.zeros((seq, D), bfloat16),  # 8 normed2 (out)
        ]
        inter = {2, 4, 6, 8}
        for sc in self.scratch["o_norm"]:
            if sc is not None:
                args.append(np.zeros((seq, D), np.float32))
                inter.add(sc)
        return self.cache.load_and_run(
            "o_norm_res_norm",
            _ONORM_BACKEND,
            *args,
            output_indices=[6, 8],
            static_input_indices={1, 3, 7},
            intermediate_indices=inter,
            bo_key=f"o_norm_L{k}",
            shared_nonstatic=True,
        )

    def _call_ffn_gemm(self, name, backend, k, wkey, normed2):
        seq = self.seq
        args = [
            np.asarray(normed2, bfloat16).reshape(seq, D),
            np.asarray(self._w[k][wkey], bfloat16).reshape(D, INTER),
            np.zeros((seq, INTER), bfloat16),
        ]
        inter = {2}
        for sc in self.scratch[name]:
            if sc is not None:
                args.append(np.zeros((seq, INTER), np.float32))
                inter.add(sc)
        return self.cache.load_and_run(
            name,
            backend,
            *args,
            output_indices=[2],
            static_input_indices={1},
            intermediate_indices=inter,
            bo_key=f"{name}_L{k}",
            shared_nonstatic=True,
        )

    def _call_gelu_mul(self, k, gate, up):
        seq = self.seq
        return self.cache.load_and_run(
            "gelu_mul",
            _GELU_BACKEND,
            np.asarray(gate, bfloat16).reshape(seq, INTER),
            np.asarray(up, bfloat16).reshape(seq, INTER),
            np.zeros((seq, INTER), bfloat16),
            output_indices=[2],
            intermediate_indices={2},
            bo_key=f"gelu_mul_L{k}",
            shared_nonstatic=True,
        )

    def _call_down_norm_add(self, k, act, res1):
        seq = self.seq
        n_pf2 = self._rms[k][3]
        args = [
            np.asarray(act, bfloat16).reshape(seq, INTER),  # 0
            np.asarray(self._w[k]["down"], bfloat16).reshape(INTER, D),  # 1
            np.zeros((seq, D), bfloat16),  # 2 down
            np.asarray(n_pf2, bfloat16).reshape(D),  # 3 post_feedforward_layernorm
            np.zeros((seq, D), bfloat16),  # 4 down_n
            np.asarray(res1, bfloat16).reshape(seq, D),  # 5 res1
            np.zeros(seq * D, bfloat16),  # 6 output (out)
        ]
        inter = {2, 4, 6}
        for sc in self.scratch["down"]:
            if sc is not None:
                args.append(np.zeros((seq, D), np.float32))
                inter.add(sc)
        return self.cache.load_and_run(
            "down_norm_add",
            _DOWN_BACKEND,
            *args,
            output_indices=[6],
            static_input_indices={1, 3},
            intermediate_indices=inter,
            bo_key=f"down_norm_add_L{k}",
            shared_nonstatic=True,
        )

    def _preload(self):
        """Write every layer's weights into per-layer resident BOs once, using
        the SAME arg layouts the prefill uses (static_input_indices then skips
        the weight writes on every subsequent call)."""
        print(
            "[gemma_prefill] pre-loading layer weights (per-layer BOs)...", flush=True
        )
        prof = self.cache.profiler.enabled
        self.cache.profiler.enabled = False
        seq = self.seq
        z_d = np.zeros((seq, D), bfloat16)
        z_i = np.zeros((seq, INTER), bfloat16)
        z_q = np.zeros((seq, DQ), bfloat16)
        for k in range(self.n_layers):
            self._call_rms_qkv(k, z_d)
            self._call_o_norm(k, z_q, z_d)
            self._call_ffn_gemm("gate", _GATE_BACKEND, k, "gate", z_d)
            self._call_ffn_gemm("up", _UP_BACKEND, k, "up", z_d)
            self._call_gelu_mul(k, z_i, z_i)
            self._call_down_norm_add(k, z_i, z_d)
        self.cache.profiler.enabled = prof
        self._preload_lm_head_gemv()
        w_mb = (
            self.n_layers
            * (
                D * DQ * 2
                + D * DK * 2 * 2
                + DQ * D * 2
                + D * INTER * 2 * 2
                + INTER * D * 2
            )
            // 1024
            // 1024
        )
        print(f"  Pre-loaded {self.n_layers} layers ({w_mb}MB)", flush=True)

    def _preload_lm_head_gemv(self):
        """Build the padded bf16 lm_head partitions [16384, D] and write them
        into resident BOs once (static; skipped thereafter)."""
        self._lm_parts = []
        for p in range(_LM_N_PARTITIONS):
            n0 = p * _LM_N_PART
            n1 = min(n0 + _LM_N_PART, VOCAB)
            w = np.zeros((_LM_N_PART, D), bfloat16)
            if n1 > n0:
                w[: n1 - n0] = np.asarray(self._lm_head[n0:n1], bfloat16)
            self._lm_parts.append(w)
        self._lm_head_npu(np.zeros(D, bfloat16))  # allocate + write the weight BOs

    def _lm_head_npu(self, hidden_bf16):
        """On-device logits from one bf16 hidden row [D] -> [VOCAB]."""
        lm_inputs = [np.ascontiguousarray(hidden_bf16, bfloat16)]
        for p in range(_LM_N_PARTITIONS):
            lm_inputs.append(self._lm_parts[p])
            lm_inputs.append(np.zeros(_LM_N_PART, bfloat16))
        res = self.cache.load_and_run(
            "lm_head_gemv",
            self._lm_backend,
            *lm_inputs,
            output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
            static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
            intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
        )
        return np.concatenate(res, axis=0)[:VOCAB]

    def _run_layer(self, x, k, N):
        """One Gemma decoder layer fully on-device. Captures roped-K + raw-V."""
        from shared.infra.fa_headfirst import npu_fa_headfirst

        seq = self.seq
        res = self._dev(self._call_rms_qkv, k, x, tag="rms_qkv")
        v = res[8].reshape(seq, DK)
        q_roped = res[15].reshape(seq, DQ)
        k_roped = res[16].reshape(seq, DK)
        self.kv_k[k][:N] = np.asarray(k_roped, bfloat16)[:N]
        self.kv_v[k][:N] = np.asarray(v, bfloat16)[:N]

        # Alternating attention: global (plain causal) every 6th layer, sliding
        # window elsewhere. Same kernel, different compiled mask.
        fa_name = _FA_GLOBAL if is_global_layer(k) else _FA_LOCAL
        attn_out = self._dev(
            npu_fa_headfirst,
            self.cache,
            np.ascontiguousarray(q_roped),
            np.ascontiguousarray(k_roped),
            np.ascontiguousarray(v),
            N_Q_HEADS,
            N_KV_HEADS,
            DH,
            seq,
            self._verbose,
            fa_name,
            tag="attn",
        )

        ores = self._dev(self._call_o_norm, k, attn_out, x, tag="o_norm")
        res1 = ores[6].reshape(seq, D)
        normed2 = ores[8].reshape(seq, D)

        gate = self._dev(
            self._call_ffn_gemm, "gate", _GATE_BACKEND, k, "gate", normed2, tag="gate"
        )[2].reshape(seq, INTER)
        up = self._dev(
            self._call_ffn_gemm, "up", _UP_BACKEND, k, "up", normed2, tag="up"
        )[2].reshape(seq, INTER)
        act = self._dev(self._call_gelu_mul, k, gate, up, tag="gelu")[2].reshape(
            seq, INTER
        )
        out = self._dev(self._call_down_norm_add, k, act, res1, tag="down")[6]
        return out.reshape(seq, D)

    def prefill(self, ids):
        assert self._w is not None, "call load_weights() first"
        N = len(ids)
        assert N <= self.seq, (N, self.seq)
        base = self.current_context_length
        # The q4nx bundle's embed_tokens is ALREADY scaled by sqrt(hidden_size)
        # (Gemma's normalizer), so gather as-is -- do NOT re-apply EMBED_SCALE.
        x = np.zeros((self.seq, D), bfloat16)
        x[:N] = _bf(np.stack([self._embed[t] for t in ids]))
        for k in range(self.n_layers):
            x = self._run_layer(x, k, N)
        self.current_context_length = base + N
        # Final RMSNorm on the single prediction row (host, <1ms), then NPU LM head.
        xf = x[N - 1].astype(np.float32)
        xn = xf / np.sqrt((xf * xf).mean() + RMS_EPS) * self._final_norm
        return self._dev(self._lm_head_npu, _bf(xn), tag="lm_head")

    # ---- KV cache (causal_lm) ----
    def get_k_cache(self, layer_idx, idx):
        return self.kv_k[layer_idx][idx]

    def get_v_cache(self, layer_idx, idx):
        return self.kv_v[layer_idx][idx]

    def kv_view(self, layer_idx):
        """(roped_K, raw_V) for the filled context [0:ctx] -> decode handoff."""
        c = self.current_context_length
        return self.kv_k[layer_idx][:c], self.kv_v[layer_idx][:c]

    def kv_stack(self):
        """(Kc, Vc) as [n_layers, ctx, DK] -- the FusedDecoder.seed_kv layout."""
        c = self.current_context_length
        return (
            np.stack([self.kv_k[k][:c] for k in range(self.n_layers)]).astype(
                np.float32
            ),
            np.stack([self.kv_v[k][:c] for k in range(self.n_layers)]).astype(
                np.float32
            ),
        )

    def clear_context(self):
        self.current_context_length = 0
        for k in range(self.n_layers):
            self.kv_k[k][:] = 0
            self.kv_v[k][:] = 0

    def get_current_context_length(self):
        return self.current_context_length

    def set_context_length(self, L):
        self.current_context_length = L


def _main():
    ap = argparse.ArgumentParser(description="Gemma3-4B Q4NX prefill on NPU2")
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="build/cache the prefill ELFs and exit (no weights, no NPU dispatch)",
    )
    ap.add_argument(
        "--n-layers", type=int, default=int(os.environ.get("NLAYERS", str(NUM_LAYERS)))
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=int(os.environ.get("Q4NX_SEQ_LEN", "2048")),
        help="padded prefill length",
    )
    ap.add_argument("--cache-dir", default=os.environ.get("Q4NX_CACHE_DIR") or None)
    ap.add_argument(
        "--bench-l",
        type=int,
        default=int(os.environ.get("Q4NX_BENCH_L", "0")),
        help="warm TTFT benchmark at this context length",
    )
    ap.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help=f"weight source: HF repo id (model.q4nx) or a local dir/file "
        f"(default: {MODEL_DEFAULT})",
    )
    args = ap.parse_args()

    print(
        f"[gemma_prefill] constructing seq_len={args.seq_len} (compiling engines)...",
        flush=True,
    )
    model = GemmaQ4nxPrefill(
        seq_len=args.seq_len, n_layers=args.n_layers, cache_dir=args.cache_dir
    )
    if args.compile_only:
        print("Compilation passed.", flush=True)
        return 0

    print("[gemma_prefill] loading Q4NX weights (host dequant)...", flush=True)
    model.load_weights(model=args.model)
    print(f"[gemma_prefill] prefill prompt N={len(PROMPT)} ...", flush=True)
    logits = model.prefill(PROMPT)
    top = int(np.asarray(logits).argmax())
    print(
        f"[gemma_prefill] first-token argmax={top} (expect {EXPECT_FIRST} ' Paris')",
        flush=True,
    )
    ok = top == EXPECT_FIRST
    print("[gemma_prefill] *** PARIS ***" if ok else "[gemma_prefill] MISS", flush=True)

    if args.bench_l:
        import time

        model.clear_context()
        ids = [int(t % VOCAB) for t in range(args.bench_l)]  # synthetic (timing only)
        print(f"[bench] warmup prefill L={args.bench_l}...", flush=True)
        model.prefill(ids)
        model.clear_context()
        model._dev_t = 0.0
        model._op_t.clear()
        print(f"[bench] timed prefill L={args.bench_l}...", flush=True)
        t0 = time.time()
        model.prefill(ids)
        wall = time.time() - t0
        npu = model._dev_t
        print(
            f"\n[bench] L={args.bench_l}: WALL={wall*1000:.0f}ms {args.bench_l/wall:.0f} tok/s prefill  |  "
            f"NPU-dispatch={npu*1000:.0f}ms {args.bench_l/npu:.0f} tok/s  |  host={(wall-npu)*1000:.0f}ms",
            flush=True,
        )
        print(
            f"[gemma_prefill] Inference: prompt_len={args.bench_l}, n_tokens=0",
            flush=True,
        )
        print(f"Time to first token (TTFT): {wall:.3f}s", flush=True)
        print(
            "[bench] per-op NPU: "
            + "  ".join(
                f"{k}={v*1000:.0f}ms"
                for k, v in sorted(model._op_t.items(), key=lambda x: -x[1])
            ),
            flush=True,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())
