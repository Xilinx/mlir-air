# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Gemma4-E2B (text) Q4NX prefill in mlir-air.
#
# Runs the prefill on the AMD NPU2: Q4NX weights, host dequant Q4NX->bf16 at
# load, then RMSNorm / Q-K-V GEMM / per-head QK-norm / weightless V-norm /
# RoPE / MQA flash attention / GELU-tanh GLU / the per-layer-embedding branch /
# all projections ON THE NPU with RESIDENT weight BOs, and the LM head as an
# on-device GEMV.
#
# Structured like gemma3_4b_q4nx_prefill.py, the closest sibling and the only
# other Gemma here. Like it, this is a SELF-OWNER -- there is no bf16 Gemma4 to
# delegate compile_all_kernels/run_transformer_block to, the way the thin q4nx
# wrappers (phi4_mini, qwen3_8b, ...) delegate to a bf16 owner.
#
# The five Gemma4 deltas vs Gemma3-4B, and where each lands:
#   1. TWO ATTENTION CLASSES WITH DIFFERENT head_dim. 28 sliding layers at
#      head_dim=256 / window=512 / theta=1e4, and 7 full layers at head_dim=512 /
#      theta=1e6 / partial rotary. head_dim is not a model constant here, so
#      every attention-shaped ELF is built twice (see _CLS).
#   2. KV SHARING. Layers >= FIRST_KV_SHARED_LAYER (15) do not project their own
#      K/V; they read the last own-KV layer OF THE SAME CLASS. Those same layers
#      carry the double-wide 12288 FFN, so the FFN ELFs are built twice too.
#   3. MQA (N_KV_HEADS=1) plus a WEIGHTLESS value_norm on V, which no other
#      model on the shared rms_qkv builder has (hence its value_norm flag).
#   4. PLE. A third sub-layer after the FFN: gelu_tanh(o2 @ inp_gate) * pli
#      projected back to D, normed, and added. Its input comes from the token
#      EMBEDDINGS, not from the layer's own hidden state.
#   5. ATTN_SCALE=1.0 (not 1/sqrt(head_dim)) and a tanh softcap on the logits.
#
# Gate: first prompt token argmax for "The capital of France is" is ' Paris',
# the same check `make paris` runs against the CPU reference.
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

from gemma4_e2b_q4nx_weights import (  # noqa: E402
    ATTN_SCALE,
    D,
    DH_GLOBAL,
    DH_SLIDING,
    EMBED_SCALE,
    FINAL_LOGIT_SOFTCAP,
    FIRST_KV_SHARED_LAYER,
    INTER,
    NUM_LAYERS,
    N_Q_HEADS,
    N_KV_HEADS,
    PLE_INPUT_SCALE,
    PLE_MODEL_PROJ_SCALE,
    PLI_D,
    RMS_EPS,
    ROPE_GLOBAL_THETA,
    ROPE_SLIDING_THETA,
    SLIDING_WINDOW,
    VOCAB,
    _gelu_tanh,
    _rmsnorm,
    head_dim,
    is_sliding,
    kv_source_layer,
    owns_kv,
)

MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma4-E2B-IT-NPU2")

# <bos> + "The capital of France is". Kept in sync with run_reference.py's
# `paris` smoke test; the expected token id is resolved from the tokenizer
# rather than hardcoded, because this bundle ships a 262144-entry vocab whose
# " Paris" id is not stable across the tokenizer revisions on the hub.
PROMPT_TEXT = "The capital of France is"
EXPECT_TEXT = "Paris"

# On-device LM head GEMV. 262144 = 16 * 16384 exactly, so unlike Gemma3-4B
# (262208 -> 17 padded partitions) nothing is wasted here.
_LM_N_PART = 16384
_LM_N_PARTITIONS = VOCAB // _LM_N_PART

# The two attention classes. Every attention-shaped ELF is built once per class
# and selected per layer by is_sliding(). "swa" = sliding window, "full" = the
# every-fifth layer that sees the whole context.
_CLS = ("swa", "full")
# The two FFN widths. Narrow is layers < FIRST_KV_SHARED_LAYER, wide is the rest
# -- the SAME predicate as KV sharing, asserted against the bundle in
# _check_class_map() rather than trusted.
_WID = ("n", "w")

INTER_NARROW = INTER  # 6144
INTER_WIDE = 2 * INTER  # 12288


def _cls(k):
    return "swa" if is_sliding(k) else "full"


def _wid(k):
    return "n" if owns_kv(k) else "w"


def dims(k):
    """(head_dim, q_dim, kv_dim) for a layer."""
    dh = head_dim(k)
    return dh, N_Q_HEADS * dh, N_KV_HEADS * dh


def cls_dims(cls):
    dh = DH_SLIDING if cls == "swa" else DH_GLOBAL
    return dh, N_Q_HEADS * dh, N_KV_HEADS * dh


def wid_inter(w):
    return INTER_NARROW if w == "n" else INTER_WIDE


def _elf_backend(instance_name, tiling=(2, 2)):
    return {
        "verbose": False,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": instance_name,
        "runtime_loop_tiling_sizes": list(tiling),
    }


def _gelu_backend(name):
    # The GELU ELF's instance name must match the top func name in build_module_2d.
    return {
        "verbose": False,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "gelu_and_mul_2d",
    }


# Kernel cache keys. One per (stage, class-or-width) so a layer never picks up
# an ELF compiled for the other class -- which would be silent: the shapes of
# the wrong-class ELF are self-consistent, it just computes a different model.
def K_RMS_Q(c):
    return f"rms_q_{c}"


def K_K(c):
    return f"k_{c}"


def K_V(c):
    return f"v_{c}"


def K_ONORM(c):
    return f"o_norm_{c}"


def K_FA(c):
    return f"flash_attn_{c}"


def K_GATE(w):
    return f"gate_{w}"


def K_UP(w):
    return f"up_{w}"


def K_GELU(w):
    return f"gelu_mul_{w}"


def K_DOWN(w):
    return f"down_{w}"


K_PLE_GEMM = "ple_gemm"  # (seq, D) -> (seq, PLI_D); used by BOTH PLE GEMMs
K_PLE_GELU = "gelu_mul_ple"  # gelu_tanh(g) * pli at PLI_D
K_PLE_PROJ = "ple_proj"  # (seq, PLI_D) -> D, norm, + residual
K_LM = "lm_head_gemv"


# ---------------------------------------------------------------------------
# eps: every Gemma4 RMSNorm uses 1e-6, but weighted_rms_norm bakes a
# module-level EPS=1e-5 at build time. Override it around each build.
# ---------------------------------------------------------------------------


class _rms_eps:
    """Context manager: build weighted_rms_norm slices at Gemma4's eps."""

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


# ---------------------------------------------------------------------------
# GEMM specs
#
# D=1536 puts every Gemma4 prefill shape OUTSIDE the measured kernel registry
# (which has no K=1536 entry), so unlike Gemma3-4B this model cannot call
# gemm_registry_config. It supplies its own spec fn, the hook the builder
# already exposes for exactly this (qwen3_4b uses it for emb=2560).
#
# The rule below is READ OFF the registry rather than invented: fused-cast above
# the documented M*K*N >= 4e9 cross-over and drain below it, tile_k_l2 = min(256,
# K), tile_k_l1 = 32, tile_n = 128 (64 when N is not a multiple of 128). Checked
# against all 159 scored registry entries: it reproduces the recorded best
# method on 41 of 41 at M=2048, the regime this prefill runs in. The 10
# disagreements are all at other M, so a non-2048 --seq-len extrapolates the
# rule rather than matching a measurement. Tiles affect throughput, not results.
# ---------------------------------------------------------------------------


def gemm_spec(m, k, n, precision="high"):
    """Per-GEMM build recipe for one Gemma4 shape."""
    from shared.builders.gemm_builder import _spec_with_tiles

    method = "fused-cast" if m * k * n >= 4e9 else "drain"
    tile_m = 64 if method == "fused-cast" else 32
    # The weight walk emits tile_k_l2 * n as its K-tile DMA stride, and NPU2
    # caps a BD stride at 1048576. The double-wide FFN (n=12288) blows that at
    # tile_k_l2=256 -- "'aie.dma_bd' op Stride 2 exceeds the [1:1048576] range",
    # 3145728 -- so it drops to 64, the largest multiple of 32 dividing K=1536
    # that fits. Applied where it is OBSERVED to bite, not from the formula:
    # n=6144 at tile_k_l2=256 exceeds the same arithmetic and compiles anyway,
    # because the emitted BD factors differently there.
    tile_k_l2 = 64 if n >= 12288 else min(256, k)
    return _spec_with_tiles(
        method,
        dict(
            tile_m=tile_m,
            tile_k_l2=tile_k_l2,
            tile_k_l1=32,
            # Always 128 wide. A narrow N is given FEWER HERD COLUMNS instead of
            # a smaller tile (see _build_gemm_ir): at N=256 the 4-column,
            # 64-wide shape is NONDETERMINISTIC on device -- the same binary and
            # inputs returned partial NaN on 2 of 4 repeats, and clean results
            # on the other 2 -- while 2 columns x 128 is stable over repeats and
            # lands at cos 0.999968 against the CPU reference. N=512 and wider
            # were stable either way.
            tile_n=128,
        ),
    )


def gemm_herd_n(n, tile_n):
    """Herd columns for an N-wide GEMM: one per tile, capped at the herd's 4."""
    return max(1, min(4, n // tile_n))


def _build_gemm_ir(m, k, n, spec, herd_m=8, herd_n=None):
    from shared.builders.gemm_builder import _build_gemm_module

    if herd_n is None:
        herd_n = gemm_herd_n(n, spec["tile_n"])
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


def _wT(w, k_dim, n_dim):
    """A bundle weight as the (K, N) bf16 operand the GEMM wants.

    Q4nxModel.dequant returns every projection as (out, in) -- the HF layout,
    which forward_prompt consumes as `x @ w.T`. The GEMMs here take (in, out),
    so this is a TRANSPOSE, not a reshape: reshaping a (4096, 1536) array to
    (1536, 4096) is silent and wrong.

    Called once per weight at load time, not per dispatch -- see _transpose_all.
    """
    a = np.ascontiguousarray(np.asarray(w, bfloat16).T)
    assert a.shape == (k_dim, n_dim), (a.shape, (k_dim, n_dim))
    return a


def _gemm_amap(inp, w, out, sc):
    """arg_map for one GEMM slice: fused-cast writes f32 scratch then casts."""
    return {0: inp, 1: w, 2: sc, 3: out} if sc is not None else {0: inp, 1: w, 2: out}


# ---------------------------------------------------------------------------
# ELF builders
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The attention input, SPLIT into two ELFs.
#
# The shared 8-launch builder (rms_qkv_qknorm_rope_multi) stitches Q, K and V
# into ONE ELF. Gemma4 cannot use it: MORE THAN ONE GEMM PER ELF MISCOMPILES
# here, silently, as partial NaN. Measured on device at seq=2048, layer 0, with
# every arm repeated:
#   Q + K/V in one ELF (mixed methods)  -> Q cos 0.99993, K/V all-NaN
#   all three GEMMs one method          -> K/V cos 0.9995, Q partial-NaN
#   K and V alone together in one ELF   -> NaN on 2 of 4 repeats, clean on 2
#   one GEMM per ELF                    -> clean on 4 of 4, every class
# So it is not the method and not the shape -- a second GEMM slice in the same
# ELF is enough. Q, K and V therefore get one ELF EACH. Every other ELF in this
# file already holds exactly one GEMM; keep it that way.
#
# The split pays for itself anyway: the 20 KV-shared layers now skip the K and V
# dispatches outright instead of running them against zero weights.
# ---------------------------------------------------------------------------


def _rms_q_kv_slices(seq_len, cls):
    from shared.builders.rms_qkv_qknorm_rope_multi import _build_qknorm_2d
    from shared.builders.rms_gemms_rope_multi import _build_rope_2d

    return _build_qknorm_2d, _build_rope_2d


def build_rms_q(seq_len, cls, herd_m=8, herd_n=4):
    """RMSNorm + Q GEMM + per-head QK-norm + RoPE.

    %arg0 x_in    (seq, D)
    %arg1 norm_w  (D,)       static
    %arg2 normed  (seq, D)   OUTPUT -- the K/V ELF's input
    %arg3 wq      (D, dq)    static
    %arg4 q       (seq, dq)
    %arg5 q_norm  (dh,)      static (carries the sqrt(head_dim) score scale)
    %arg6 q_n     (seq, dq)
    %arg7 lut_q   (seq*dq,)  static
    %arg8 q_roped (seq, dq)  OUTPUT
    [+ f32 C-scratch tail for the fused-cast Q GEMM]
    """
    from shared.infra.stitching import (
        _wrap_ir_in_launch,
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    qknorm_2d, rope_2d = _rms_q_kv_slices(seq_len, cls)
    dh, dq, _dkv = cls_dims(cls)
    q_spec = gemm_spec(seq_len, D, dq)

    with _rms_eps():
        print(f"  [1/4] RMSNorm (eps={RMS_EPS:g})...")
        rms_ir = _wrap_ir_in_launch(str(build_rms(seq_len, D, bfloat16, 16, herd_x=8)))
    print(f"  [2/4] Q GEMM ({q_spec['method']}) {seq_len}x{D}x{dq}...")
    q_ir = _build_gemm_ir(seq_len, D, dq, q_spec, herd_m)
    print(f"  [3/4] QK-norm Q (dim={dh} eps={RMS_EPS:g})...")
    qn_ir = str(_build_qknorm_2d_at(qknorm_2d, seq_len, dq, dh))
    print(f"  [4/4] RoPE Q (dim={dh})...")
    rq_ir = str(rope_2d(seq_len, dq, dh, bfloat16, 8))

    scratch_args, scratch_for = alloc_gemm_scratch([(q_spec, seq_len, dq)], 9)
    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg1", f"memref<{D}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg3", f"memref<{D}x{dq}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{dq}xbf16>"),
        FuncArg("%arg5", f"memref<{dh}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{dq}xbf16>"),
        FuncArg("%arg7", f"memref<{seq_len * dq}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{dq}xbf16>"),
    ]
    slices = [
        KernelSlice(
            rms_ir, "r", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}
        ),
        KernelSlice(
            q_ir,
            "q",
            _gemm_amap(2, 3, 4, scratch_for[0]),
            extern_syms=_gemm_externs(q_spec),
        ),
        KernelSlice(qn_ir, "qn", {0: 4, 1: 5, 2: 6}, private_from=False),
        KernelSlice(rq_ir, "rq", {0: 6, 1: 7, 2: 8}, extern_syms={"@rope"}),
    ]
    module = stitch_elf(K_RMS_Q(cls), base_args, slices, scratch_args=scratch_args)
    print(f"  {K_RMS_Q(cls)} module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def build_k(seq_len, cls, herd_m=8, herd_n=4):
    """K GEMM + per-head QK-norm + RoPE. Only the own-KV layers dispatch it.

    %arg0 normed   (seq, D)    from the Q ELF
    %arg1 wk       (D, dkv)    static
    %arg2 k        (seq, dkv)
    %arg3 k_norm   (dh,)       static
    %arg4 k_n      (seq, dkv)
    %arg5 lut_k    (seq*dkv,)  static
    %arg6 k_roped  (seq, dkv)  OUTPUT
    [+ f32 C-scratch tail]
    """
    from shared.infra.stitching import (
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
    )

    qknorm_2d, rope_2d = _rms_q_kv_slices(seq_len, cls)
    dh, _dq, dkv = cls_dims(cls)
    spec = gemm_spec(seq_len, D, dkv)
    print(f"  [1/3] K GEMM ({spec['method']}) {seq_len}x{D}x{dkv}...")
    k_ir = _build_gemm_ir(seq_len, D, dkv, spec, herd_m)
    print(f"  [2/3] QK-norm K (dim={dh})...")
    kn_ir = str(_build_qknorm_2d_at(qknorm_2d, seq_len, dkv, dh))
    print(f"  [3/3] RoPE K (dim={dh})...")
    rk_ir = str(rope_2d(seq_len, dkv, dh, bfloat16, 8))

    scratch_args, scratch_for = alloc_gemm_scratch([(spec, seq_len, dkv)], 7)
    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg1", f"memref<{D}x{dkv}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{dkv}xbf16>"),
        FuncArg("%arg3", f"memref<{dh}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{dkv}xbf16>"),
        FuncArg("%arg5", f"memref<{seq_len * dkv}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{dkv}xbf16>"),
    ]
    slices = [
        KernelSlice(
            k_ir,
            "k",
            _gemm_amap(0, 1, 2, scratch_for[0]),
            extern_syms=_gemm_externs(spec),
        ),
        KernelSlice(kn_ir, "kn", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(rk_ir, "rk", {0: 4, 1: 5, 2: 6}, extern_syms={"@rope"}),
    ]
    module = stitch_elf(K_K(cls), base_args, slices, scratch_args=scratch_args)
    print(f"  {K_K(cls)} module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def build_v(seq_len, cls, herd_m=8, herd_n=4):
    """V GEMM + weightless value_norm. V is not roped.

    %arg0 normed  (seq, D)   from the Q ELF
    %arg1 wv      (D, dkv)   static
    %arg2 v       (seq, dkv)
    %arg3 v_norm  (dh,)      static -- ALL ONES: Gemma4's value_norm is
                             weightless (`_rmsnorm(v, None)`), which is the
                             per-head norm slice with a unit weight
    %arg4 v_n     (seq, dkv) OUTPUT
    [+ f32 C-scratch tail]
    """
    from shared.infra.stitching import (
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
    )

    qknorm_2d, _rope_2d = _rms_q_kv_slices(seq_len, cls)
    dh, _dq, dkv = cls_dims(cls)
    spec = gemm_spec(seq_len, D, dkv)
    print(f"  [1/2] V GEMM ({spec['method']}) {seq_len}x{D}x{dkv}...")
    v_ir = _build_gemm_ir(seq_len, D, dkv, spec, herd_m)
    print(f"  [2/2] V-norm (dim={dh}, weightless)...")
    vn_ir = str(_build_qknorm_2d_at(qknorm_2d, seq_len, dkv, dh))

    scratch_args, scratch_for = alloc_gemm_scratch([(spec, seq_len, dkv)], 5)
    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg1", f"memref<{D}x{dkv}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{dkv}xbf16>"),
        FuncArg("%arg3", f"memref<{dh}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{dkv}xbf16>"),
    ]
    slices = [
        KernelSlice(
            v_ir,
            "v",
            _gemm_amap(0, 1, 2, scratch_for[0]),
            extern_syms=_gemm_externs(spec),
        ),
        KernelSlice(vn_ir, "vn", {0: 2, 1: 3, 2: 4}, private_from=False),
    ]
    module = stitch_elf(K_V(cls), base_args, slices, scratch_args=scratch_args)
    print(f"  {K_V(cls)} module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def _build_qknorm_2d_at(qknorm_2d, seq_len, cols, dh):
    """The shared per-head RMSNorm slice at Gemma4's eps."""
    with _rms_eps():
        return qknorm_2d(seq_len, cols, dh, bfloat16, RMS_EPS, 8)


def build_o_norm_res_norm_module(seq_len, cls, herd_m=8, herd_n=4):
    """O proj (DECOUPLED) + post-attention norm + residual + pre-FFN norm.

    The Gemma attention tail, identical in shape to Gemma3-4B's: the projection
    is normalized BEFORE the residual add, unlike Llama/Qwen.

      %arg0 attn_out    (seq, dq)   DECOUPLED (n_heads*head_dim, class-dependent)
      %arg1 wo          (dq, D)     static
      %arg2 proj        (seq, D)
      %arg3 post_attn_w (D,)        static
      %arg4 proj_n      (seq, D)
      %arg5 x_resid     (seq, D)
      %arg6 res1        (seq, D)    OUTPUT (feeds down_norm_add)
      %arg7 pre_ffn_w   (D,)        static
      %arg8 normed2     (seq, D)    OUTPUT (feeds gate/up)
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

    _dh, dq, _dkv = cls_dims(cls)
    o_spec = gemm_spec(seq_len, dq, D)
    print(f"  [1/4] O GEMM ({o_spec['method']}) {seq_len}x{dq}x{D} (DECOUPLED)...")
    o_ir = _build_gemm_ir(seq_len, dq, D, o_spec, herd_m)
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
        FuncArg("%arg0", f"memref<{seq_len}x{dq}xbf16>"),
        FuncArg("%arg1", f"memref<{dq}x{D}xbf16>"),
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
    module = stitch_elf(K_ONORM(cls), base_args, slices, scratch_args=scratch_args)
    print(f"  o_norm_{cls} module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


def _build_single_gemm_elf(name, sym, seq_len, k_dim, n_dim, herd_m=8, herd_n=4):
    """Standalone single-GEMM ELF: arg0 in, arg1 weight, arg2 out."""
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
    gemm_ir = _build_gemm_ir(seq_len, k_dim, n_dim, g_spec, herd_m)
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

    Same L1/BD sweet spot as the Gemma3-4B sibling: too large overflows L1, too
    small exhausts the BD pool."""
    span = (seq_len * hidden_dim) // herd_x
    for t in range(5120, 1024, -64):
        if span % t == 0:
            return t
    raise RuntimeError(f"No GELU tile_n for seq={seq_len} hidden={hidden_dim}")


def build_gelu_mul_module(seq_len, hidden_dim, herd_x=8, herd_y=1):
    """Standalone NPU GeGLU ELF: gelu_tanh(a) * b -> (seq, hidden_dim)."""
    from gelu_and_mul.gelu_and_mul import build_module_2d as build_gelu

    tile_n = _gelu_tile_n(seq_len, hidden_dim, herd_x)
    print(f"  [gelu_mul] GELU-tanh GLU {seq_len}x{hidden_dim} (tile_n={tile_n})...")
    module = build_gelu(seq_len, hidden_dim, tile_n, bfloat16, herd_x, herd_y)
    print(f"  gelu_mul module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def build_gemm_norm_add_module(name, seq_len, k_dim, out_1d, herd_m=8, herd_n=4):
    """GEMM + RMSNorm + residual add -- the tail shared by the FFN and the PLE.

    Gemma4 ends both its FFN and its per-layer-embedding branch the same way:
    project back to D, normalize the projection, then add the residual. The only
    differences are k_dim (INTER vs PLI_D) and whether the consumer wants the
    result flat.

    %arg0 act     (seq, k_dim)   the branch's activation
    %arg1 w       (k_dim, D)     static
    %arg2 proj    (seq, D)
    %arg3 norm_w  (D,)           static
    %arg4 proj_n  (seq, D)
    %arg5 resid   (seq, D)
    %arg6 output  (seq*D,) if out_1d else (seq, D)   OUTPUT
    [+ f32 C-scratch tail for the fused-cast GEMM]
    """
    from shared.infra.stitching import (
        _wrap_ir_in_launch,
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
        build_add_2d_to_1d_ir,
        build_residual_add_2d_ir,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    d_spec = gemm_spec(seq_len, k_dim, D)
    print(f"  [1/3] {name} GEMM ({d_spec['method']}) {seq_len}x{k_dim}x{D}...")
    down_ir = _build_gemm_ir(seq_len, k_dim, D, d_spec, herd_m)
    with _rms_eps():
        print(f"  [2/3] post RMSNorm (eps={RMS_EPS:g})...")
        post_ir = _wrap_ir_in_launch(str(build_rms(seq_len, D, bfloat16, 16, herd_x=8)))
    print(f"  [3/3] residual add ({'2D -> 1D' if out_1d else '2D'})...")
    add_ir = (
        build_add_2d_to_1d_ir(seq_len, D)
        if out_1d
        else build_residual_add_2d_ir(seq_len, D)
    )

    scratch_args, scratch_for = alloc_gemm_scratch([(d_spec, seq_len, D)], 7)
    out_ty = f"memref<{seq_len * D}xbf16>" if out_1d else f"memref<{seq_len}x{D}xbf16>"
    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{k_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{k_dim}x{D}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg3", f"memref<{D}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg5", f"memref<{seq_len}x{D}xbf16>"),
        FuncArg("%arg6", out_ty),
    ]
    slices = [
        KernelSlice(
            down_ir,
            "dg",
            _gemm_amap(0, 1, 2, scratch_for[0]),
            extern_syms=_gemm_externs(d_spec),
            private_from=True,
        ),
        KernelSlice(post_ir, "pn", {0: 2, 1: 3, 2: 4}, private_from=False),
        KernelSlice(add_ir, "ad", {0: 4, 1: 5, 2: 6}, private_from=False),
    ]
    module = stitch_elf(name, base_args, slices, scratch_args=scratch_args)
    print(f"  {name} module: {len(str(module).splitlines())} lines, parsed OK")
    return module, scratch_for


# ---------------------------------------------------------------------------
# RoPE LUTs
# ---------------------------------------------------------------------------


def rope_luts(seq_len, rope_freqs):
    """{class: [seq_len, head_dim]} half-split RoPE cos/sin LUTs.

    Batched over positions from gemma4_e2b_q4nx_weights.rope_lut, which owns the
    convention: row p = [cos(p*inv) (dh/2) ++ sin(p*inv) (dh/2)], pairing dim i
    with i + dh/2. PARTIAL ROTARY IS NOT A SEPARATE CODE PATH -- the full layers
    divide inv_freq by the bundle's rope_freqs table, whose 1e30 tail zeroes the
    unrotated frequencies (cos=1, sin=0 = identity). Passing rope_freqs=None
    there would silently rotate all 512 dims.
    """
    out = {}
    pos = np.arange(seq_len, dtype=np.float64)
    for cls in _CLS:
        dh = DH_SLIDING if cls == "swa" else DH_GLOBAL
        half = dh // 2
        theta = ROPE_SLIDING_THETA if cls == "swa" else ROPE_GLOBAL_THETA
        inv = 1.0 / (theta ** (np.arange(half, dtype=np.float64) * 2.0 / dh))
        if cls == "full":
            if rope_freqs is None:
                raise RuntimeError(
                    "the bundle has no rope_freqs.weight; the full-attention "
                    "layers need it for partial rotary (see rope_lut)"
                )
            inv = inv / np.asarray(rope_freqs[:half], np.float64)
        ang = pos[:, None] * inv[None, :]
        out[cls] = np.concatenate([np.cos(ang), np.sin(ang)], axis=-1).astype(bfloat16)
    return out


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

_NEEDED = (
    [K_RMS_Q(c) for c in _CLS]
    + [K_K(c) for c in _CLS]
    + [K_V(c) for c in _CLS]
    + [K_ONORM(c) for c in _CLS]
    + [K_FA(c) for c in _CLS]
    + [K_GATE(w) for w in _WID]
    + [K_UP(w) for w in _WID]
    + [K_GELU(w) for w in _WID]
    + [K_DOWN(w) for w in _WID]
    + [K_PLE_GEMM, K_PLE_GELU, K_PLE_PROJ, K_LM]
)


def compile_all_kernels(cache, seq_len, verbose=False):
    """Compile the prefill ELFs. Returns the scratch-index map."""
    print(
        f"\n{'='*60}\nCompiling Gemma4-E2B prefill kernels (seq_len={seq_len}, "
        f"{len(_NEEDED)} ELFs)...\n{'='*60}\n"
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

    for c in _CLS:
        dh, dq, dkv = cls_dims(c)
        print(f"\n--- {K_RMS_Q(c)} (RMSNorm + Q GEMM + QK-norm + RoPE, dh={dh}) ---")
        mod, scratch[K_RMS_Q(c)] = build_rms_q(seq_len, c)
        cache.compile_and_cache(K_RMS_Q(c), mod, _elf_backend(K_RMS_Q(c)))

        print(f"\n--- {K_K(c)} (K GEMM + QK-norm + RoPE, dh={dh}) ---")
        mod, scratch[K_K(c)] = build_k(seq_len, c)
        cache.compile_and_cache(K_K(c), mod, _elf_backend(K_K(c)))

        print(f"\n--- {K_V(c)} (V GEMM + weightless value_norm, dh={dh}) ---")
        mod, scratch[K_V(c)] = build_v(seq_len, c)
        cache.compile_and_cache(K_V(c), mod, _elf_backend(K_V(c)))

        print(f"\n--- {K_ONORM(c)} (O + post-attn norm + residual + pre-FFN norm) ---")
        mod, scratch[K_ONORM(c)] = build_o_norm_res_norm_module(seq_len, c)
        cache.compile_and_cache(K_ONORM(c), mod, _elf_backend(K_ONORM(c)))

    for w in _WID:
        inter = wid_inter(w)
        print(f"\n--- {K_GATE(w)} (Gate GEMM, INTER={inter}) ---")
        mod, scratch[K_GATE(w)] = _build_single_gemm_elf(
            K_GATE(w), "gg", seq_len, D, inter
        )
        cache.compile_and_cache(K_GATE(w), mod, _elf_backend(K_GATE(w)))

        print(f"\n--- {K_UP(w)} (Up GEMM, INTER={inter}) ---")
        mod, scratch[K_UP(w)] = _build_single_gemm_elf(K_UP(w), "ug", seq_len, D, inter)
        cache.compile_and_cache(K_UP(w), mod, _elf_backend(K_UP(w)))

        print(f"\n--- {K_GELU(w)} (GELU-tanh GLU, INTER={inter}) ---")
        cache.compile_and_cache(
            K_GELU(w),
            build_gelu_mul_module(seq_len, inter),
            _gelu_backend(K_GELU(w)),
        )

        print(f"\n--- {K_DOWN(w)} (Down + post-FFN norm + residual) ---")
        mod, scratch[K_DOWN(w)] = build_gemm_norm_add_module(
            K_DOWN(w), seq_len, inter, out_1d=True
        )
        cache.compile_and_cache(K_DOWN(w), mod, _elf_backend(K_DOWN(w)))

    # --- PLE. One set of ELFs for all 35 layers: the branch is PLI_D-wide
    # regardless of attention class or FFN width.
    print(f"\n--- {K_PLE_GEMM} (D -> PLI_D GEMM; inp_gate AND model_proj) ---")
    mod, scratch[K_PLE_GEMM] = _build_single_gemm_elf(
        K_PLE_GEMM, "pg", seq_len, D, PLI_D
    )
    cache.compile_and_cache(K_PLE_GEMM, mod, _elf_backend(K_PLE_GEMM))

    print(f"\n--- {K_PLE_GELU} (gelu_tanh(gate) * per-layer input, {PLI_D}) ---")
    cache.compile_and_cache(
        K_PLE_GELU, build_gelu_mul_module(seq_len, PLI_D), _gelu_backend(K_PLE_GELU)
    )

    print(f"\n--- {K_PLE_PROJ} (PLI_D -> D + post_layernorm + residual) ---")
    mod, scratch[K_PLE_PROJ] = build_gemm_norm_add_module(
        K_PLE_PROJ, seq_len, PLI_D, out_1d=False
    )
    cache.compile_and_cache(K_PLE_PROJ, mod, _elf_backend(K_PLE_PROJ))

    # --- Attention. One ELF per class: the sliding layers carry the window
    # mask AND head_dim=256, the full layers plain causal at head_dim=512.
    from shared.infra.fa_headfirst import compile_headfirst_fa

    for c in _CLS:
        dh = cls_dims(c)[0]
        win = SLIDING_WINDOW if c == "swa" else None
        print(f"\n--- {K_FA(c)} (head-first FA, head_dim={dh}, window={win}) ---")
        compile_headfirst_fa(
            cache,
            seq_len,
            N_Q_HEADS,
            N_KV_HEADS,
            dh,
            verbose,
            window=win,
            name=K_FA(c),
            causal_skip=True,
        )

    print(f"\n--- {K_LM} ({_LM_N_PARTITIONS} x {_LM_N_PART}, K={D}) ---")
    from shared.builders.lm_head_gemv_multi import build_lm_head_gemv_module
    from shared.infra.backend_presets import LM_GEMV_BACKEND

    cache.compile_and_cache(
        K_LM,
        build_lm_head_gemv_module(D, n_partitions=_LM_N_PARTITIONS, n_part=_LM_N_PART),
        {"verbose": verbose, **dict(LM_GEMV_BACKEND)},
    )

    cache._save_manifest()
    print(f"\nAll {len(cache.artifacts)} kernels compiled to {cache.cache_dir}/")
    return scratch


# Device-resident handoffs down the FFN and PLE chains. Each name is one buffer
# that a producing ELF writes and the next consuming ELF reads in place, so the
# value never round-trips through the host. The consumer's index must also be
# listed as an intermediate so its host->device write is skipped.
_A_NORMED2 = "ffn_normed2"  # o_norm -> gate, up
_A_RES1 = "ffn_res1"  # o_norm -> down
_A_GATE = "ffn_gate"  # gate -> gelu_mul
_A_UP = "ffn_up"  # up -> gelu_mul
_A_ACT = "ffn_act"  # gelu_mul -> down


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

    sc = {}
    for c in _CLS:
        _dh, dq, dkv = cls_dims(c)
        sc[K_RMS_Q(c)] = _alloc([gemm_spec(seq_len, D, dq)], 9)
        sc[K_K(c)] = _alloc([gemm_spec(seq_len, D, dkv)], 7)
        sc[K_V(c)] = _alloc([gemm_spec(seq_len, D, dkv)], 5)
        sc[K_ONORM(c)] = _alloc([gemm_spec(seq_len, dq, D)], 9)
    for w in _WID:
        inter = wid_inter(w)
        sc[K_GATE(w)] = _alloc([gemm_spec(seq_len, D, inter)], 3)
        sc[K_UP(w)] = _alloc([gemm_spec(seq_len, D, inter)], 3)
        sc[K_DOWN(w)] = _alloc([gemm_spec(seq_len, inter, D)], 7)
    sc[K_PLE_GEMM] = _alloc([gemm_spec(seq_len, D, PLI_D)], 3)
    sc[K_PLE_PROJ] = _alloc([gemm_spec(seq_len, PLI_D, D)], 7)
    return sc


# ---------------------------------------------------------------------------
# The prefill model
# ---------------------------------------------------------------------------


class Gemma4Q4nxPrefill:
    """AIR realization of the Gemma4-E2B Q4NX causal-LM prefill interface."""

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
        # KernelCache keys on kernel NAME, and the manifest validates only the
        # toolchain -- so a cache built at another seq_len would be reused here
        # and dispatched with differently sized buffers. The default path
        # carries the length, but --cache-dir / Q4NX_CACHE_DIR is taken
        # verbatim, so stamp the length and refuse a mismatch.
        self._seq_stamp = Path(cache_dir) / ".seq_len"
        self._verbose = verbose

        force = os.environ.get("Q4NX_FORCE_COMPILE") == "1"
        if self._seq_stamp.is_file():
            was = self._seq_stamp.read_text().strip()
            if was != str(seq_len):
                raise SystemExit(
                    f"cache {cache_dir} was built for seq_len={was}, not "
                    f"{seq_len}. Point --cache-dir elsewhere or remove it."
                )
        if not force:
            self.cache.load_manifest()
        cached = set() if force else set(self.cache.artifacts)
        if not set(_NEEDED).issubset(cached):
            self.scratch = compile_all_kernels(self.cache, seq_len, verbose)
        else:
            print("[g4_prefill] using cached prefill ELFs (skip compile)", flush=True)
            self.scratch = resolve_scratch(seq_len)
        self._seq_stamp.parent.mkdir(parents=True, exist_ok=True)
        self._seq_stamp.write_text(str(seq_len))

        from shared.infra.backend_presets import LM_GEMV_BACKEND

        self._lm_backend = dict(LM_GEMV_BACKEND)

        # Per-layer KV cache (roped K + V-normed V). Only the layers that OWN a
        # cache allocate one; the other 20 read their source layer's.
        self.kv_k, self.kv_v = {}, {}
        for k in range(n_layers):
            if owns_kv(k):
                dkv = dims(k)[2]
                self.kv_k[k] = np.zeros((self.MAX_L, dkv), bfloat16)
                self.kv_v[k] = np.zeros((self.MAX_L, dkv), bfloat16)
        self._w = None
        self._nm = None
        self._ple = None
        self._embed = None
        self._g = None
        self._luts = None
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
        """Load the Q4NX weights from the self-contained `model.q4nx` bundle,
        dequantizing Q4NX->bf16 on the host at load, then pre-load them into
        resident per-layer BOs."""
        from gemma4_e2b_q4nx_weights import Q4nxModel

        model = model or os.environ.get("Q4NX_MODEL_SOURCE", MODEL_DEFAULT)
        print(f"[g4_prefill] loading weights from model.q4nx ({model})", flush=True)
        self._qm = qm = Q4nxModel(model)
        self._check_class_map(qm)
        self._w = [
            self._transpose_all(qm.layer_weights(k), k) for k in range(self.n_layers)
        ]
        self._nm = [qm.layer_norms(k) for k in range(self.n_layers)]
        self._ple = [self._transpose_ple(qm.layer_ple(k)) for k in range(self.n_layers)]
        self._g = qm.globals()
        self._luts = rope_luts(self.seq, qm.rope_freqs())
        self._preload()

    def _transpose_all(self, w, k):
        """Every projection of one layer, transposed to (K, N) bf16 once.

        The dispatch path used to call _wT on each weight on every layer of
        every prefill, which re-materialized a contiguous transpose of up to
        12288x1536 per call even though static_input_indices means the device
        write is skipped. Doing it here also HALVES the host footprint: the
        float32 arrays Q4nxModel.dequant returns are dropped on return.
        """
        _dh, dq, dkv = dims(k)
        inter = wid_inter(_wid(k))
        out = {
            "q": _wT(w["q"], D, dq),
            "o": _wT(w["o"], dq, D),
            "gate": _wT(w["gate"], D, inter),
            "up": _wT(w["up"], D, inter),
            "down": _wT(w["down"], inter, D),
        }
        if owns_kv(k):
            out["k"] = _wT(w["k"], D, dkv)
            out["v"] = _wT(w["v"], D, dkv)
        return out

    def _transpose_ple(self, pw):
        """The PLE matrices as the (K, N) operands their GEMMs want."""
        return {
            "inp_gate": _wT(pw["inp_gate"], D, PLI_D),
            "model_proj": _wT(pw["model_proj"], D, PLI_D),
            "per_layer_projection": _wT(pw["per_layer_projection"], PLI_D, D),
        }

    def _check_class_map(self, qm):
        """The bundle, not this file, decides which layers are double-wide.

        owns_kv() is reused as the FFN-width predicate, which is only valid
        because the two coincide in this model. Assert it instead of trusting
        it: a mismatch would pick an ELF of the wrong width and produce a
        shape error deep inside a dispatch rather than here.
        """
        for k in range(self.n_layers):
            want = wid_inter(_wid(k))
            got = qm.mlp_inter(k)
            if got != want:
                raise RuntimeError(
                    f"layer {k}: bundle mlp_inter={got} but the class map says "
                    f"{want} (owns_kv={owns_kv(k)}). The FFN-width and KV-sharing "
                    f"boundaries are assumed identical here and are not."
                )

    # ---- per-ELF calls (single owner of each arg layout) ----
    def _lut_for(self, k):
        """Per-head-repeated (lut_q, lut_k) for a layer's RoPE class."""
        lut = self._luts[_cls(k)][: self.seq]
        return (
            np.repeat(lut, N_Q_HEADS, axis=0).flatten(),
            np.repeat(lut, N_KV_HEADS, axis=0).flatten(),
        )

    def _q_norm_scaled(self, k):
        """q_norm with sqrt(head_dim) folded in.

        The head-first FA kernel bakes in a 1/sqrt(head_dim) score scale, but
        Gemma4's config scaling is ATTN_SCALE=1.0. Folding sqrt(head_dim) into
        the QK-norm weight cancels it exactly: the norm runs in f32 and RoPE is
        a rotation, so the scale commutes through both. Scaling Q afterwards
        instead would round the product to bf16 a second time.
        """
        dh = dims(k)[0]
        return np.asarray(self._nm[k]["q_norm"], np.float32) * (
            np.sqrt(dh) * ATTN_SCALE
        )

    def _call_rms_q(self, k, x_in):
        """RMSNorm + Q GEMM + QK-norm + RoPE. Returns arg2 normed, arg8 q_roped."""
        seq = self.seq
        dh, dq, _dkv = dims(k)
        nm = self._nm[k]
        lut_q, _lut_k = self._lut_for(k)
        name = K_RMS_Q(_cls(k))
        args = [
            np.asarray(x_in, bfloat16).reshape(seq, D),  # 0 x_in (dynamic)
            np.asarray(nm["input"], bfloat16).reshape(D),  # 1 input_layernorm
            np.zeros((seq, D), bfloat16),  # 2 normed (out -> the K/V ELF)
            self._w[k]["q"],  # 3
            np.zeros((seq, dq), bfloat16),  # 4 q
            np.asarray(self._q_norm_scaled(k), bfloat16).reshape(dh),  # 5 q_norm
            np.zeros((seq, dq), bfloat16),  # 6 q_n
            lut_q,  # 7
            np.zeros((seq, dq), bfloat16),  # 8 q_roped (out)
        ]
        inter = {4, 6, 8}
        for sc in self.scratch[name]:
            if sc is not None:
                args.append(np.zeros((seq, dq), np.float32))
                inter.add(sc)
        return self.cache.load_and_run(
            name,
            _elf_backend(name),
            *args,
            output_indices=[2, 8],
            static_input_indices={1, 3, 5, 7},
            intermediate_indices=inter,
            bo_key=f"rms_q_L{k}",
            shared_nonstatic=True,
        )

    def _call_k(self, k, normed):
        """K GEMM + QK-norm + RoPE. Returns arg6 k_roped."""
        seq = self.seq
        dh, _dq, dkv = dims(k)
        nm = self._nm[k]
        _lut_q, lut_k = self._lut_for(k)
        name = K_K(_cls(k))
        args = [
            np.asarray(normed, bfloat16).reshape(seq, D),  # 0 normed (a real input)
            self._w[k]["k"],  # 1
            np.zeros((seq, dkv), bfloat16),  # 2 k
            np.asarray(nm["k_norm"], bfloat16).reshape(dh),  # 3
            np.zeros((seq, dkv), bfloat16),  # 4 k_n
            lut_k,  # 5
            np.zeros((seq, dkv), bfloat16),  # 6 k_roped (out)
        ]
        inter = {2, 4, 6}
        for sc in self.scratch[name]:
            if sc is not None:
                args.append(np.zeros((seq, dkv), np.float32))
                inter.add(sc)
        return self.cache.load_and_run(
            name,
            _elf_backend(name),
            *args,
            output_indices=[6],
            static_input_indices={1, 3, 5},
            intermediate_indices=inter,
            bo_key=f"k_L{k}",
            shared_nonstatic=True,
        )

    def _call_v(self, k, normed):
        """V GEMM + weightless value_norm. Returns arg4 v_n."""
        seq = self.seq
        dh, _dq, dkv = dims(k)
        name = K_V(_cls(k))
        args = [
            np.asarray(normed, bfloat16).reshape(seq, D),  # 0 normed
            self._w[k]["v"],  # 1
            np.zeros((seq, dkv), bfloat16),  # 2 v
            np.ones(dh, bfloat16),  # 3 value_norm -- WEIGHTLESS
            np.zeros((seq, dkv), bfloat16),  # 4 v_n (out)
        ]
        inter = {2, 4}
        for sc in self.scratch[name]:
            if sc is not None:
                args.append(np.zeros((seq, dkv), np.float32))
                inter.add(sc)
        return self.cache.load_and_run(
            name,
            _elf_backend(name),
            *args,
            output_indices=[4],
            static_input_indices={1, 3},
            intermediate_indices=inter,
            bo_key=f"v_L{k}",
            shared_nonstatic=True,
        )

    def _call_o_norm(self, k, attn_out, x_resid):
        seq = self.seq
        _dh, dq, _dkv = dims(k)
        nm = self._nm[k]
        args = [
            np.asarray(attn_out, bfloat16).reshape(seq, dq),  # 0
            self._w[k]["o"],  # 1 wo
            np.zeros((seq, D), bfloat16),  # 2 proj
            np.asarray(nm["post_attn"], bfloat16).reshape(D),  # 3
            np.zeros((seq, D), bfloat16),  # 4 proj_n
            np.asarray(x_resid, bfloat16).reshape(seq, D),  # 5 residual
            np.zeros((seq, D), bfloat16),  # 6 res1 (out)
            np.asarray(nm["pre_ffn"], bfloat16).reshape(D),  # 7
            np.zeros((seq, D), bfloat16),  # 8 normed2 (out)
        ]
        inter = {2, 4, 6, 8}
        for sc in self.scratch[K_ONORM(_cls(k))]:
            if sc is not None:
                args.append(np.zeros((seq, D), np.float32))
                inter.add(sc)
        return self.cache.load_and_run(
            K_ONORM(_cls(k)),
            _elf_backend(K_ONORM(_cls(k))),
            *args,
            output_indices=[6, 8],
            shared_alias={6: _A_RES1, 8: _A_NORMED2},
            static_input_indices={1, 3, 7},
            intermediate_indices=inter,
            bo_key=f"o_norm_L{k}",
            shared_nonstatic=True,
        )

    def _call_ffn_gemm(self, name, k, wkey, normed2, out_alias):
        seq = self.seq
        inter = wid_inter(_wid(k))
        args = [
            np.asarray(normed2, bfloat16).reshape(seq, D),
            self._w[k][wkey],
            np.zeros((seq, inter), bfloat16),
        ]
        idx = {0, 2}
        for sc in self.scratch[name]:
            if sc is not None:
                args.append(np.zeros((seq, inter), np.float32))
                idx.add(sc)
        return self.cache.load_and_run(
            name,
            _elf_backend(name),
            *args,
            output_indices=[2],
            static_input_indices={1},
            intermediate_indices=idx,
            bo_key=f"{name}_L{k}",
            shared_nonstatic=True,
            shared_alias={0: _A_NORMED2, 2: out_alias},
        )

    def _call_gelu_mul(self, name, k, hidden, a, b, alias):
        seq = self.seq
        return self.cache.load_and_run(
            name,
            _gelu_backend(name),
            np.asarray(a, bfloat16).reshape(seq, hidden),
            np.asarray(b, bfloat16).reshape(seq, hidden),
            np.zeros((seq, hidden), bfloat16),
            output_indices=[2],
            # An index listed here has its host->device write SKIPPED, which is
            # only correct when shared_alias supplies it on device. The PLE
            # branch passes no alias (its gate comes from a GEMM and its second
            # operand from the host), so only the output is an intermediate
            # there -- listing the inputs made the kernel read a zero buffer.
            intermediate_indices={2} | {i for i in (0, 1) if (alias or {}).get(i)},
            bo_key=f"{name}_L{k}",
            shared_nonstatic=True,
            shared_alias=alias,
        )

    def _call_gemm_norm_add(self, name, k, k_dim, act, w, norm_w, resid, out_1d, alias):
        seq = self.seq
        args = [
            np.asarray(act, bfloat16).reshape(seq, k_dim),  # 0
            np.asarray(w, bfloat16).reshape(k_dim, D),  # 1 (already (K, N))
            np.zeros((seq, D), bfloat16),  # 2 proj
            np.asarray(norm_w, bfloat16).reshape(D),  # 3
            np.zeros((seq, D), bfloat16),  # 4 proj_n
            np.asarray(resid, bfloat16).reshape(seq, D),  # 5
            np.zeros(seq * D if out_1d else (seq, D), bfloat16),  # 6 out
        ]
        # See _call_gelu_mul: only alias-supplied inputs may be intermediates.
        idx = {2, 4, 6} | {i for i in (0, 5) if (alias or {}).get(i)}
        for sc in self.scratch[name]:
            if sc is not None:
                args.append(np.zeros((seq, D), np.float32))
                idx.add(sc)
        return self.cache.load_and_run(
            name,
            _elf_backend(name),
            *args,
            output_indices=[6],
            static_input_indices={1, 3},
            intermediate_indices=idx,
            bo_key=f"{name}_L{k}",
            shared_nonstatic=True,
            shared_alias=alias,
        )

    def _call_ple_gemm(self, k, x, w, tag):
        """The (seq, D) -> (seq, PLI_D) GEMM, shared by the two PLE projections.

        `tag` separates their resident weight BOs: inp_gate runs per layer during
        prefill, model_proj runs per layer once per prompt, and they must not
        land on the same BO.
        """
        seq = self.seq
        args = [
            np.asarray(x, bfloat16).reshape(seq, D),
            np.asarray(w, bfloat16).reshape(D, PLI_D),
            np.zeros((seq, PLI_D), bfloat16),
        ]
        idx = {2}  # arg0 is a real host input -- see _call_gelu_mul
        for sc in self.scratch[K_PLE_GEMM]:
            if sc is not None:
                args.append(np.zeros((seq, PLI_D), np.float32))
                idx.add(sc)
        return self.cache.load_and_run(
            K_PLE_GEMM,
            _elf_backend(K_PLE_GEMM),
            *args,
            output_indices=[2],
            static_input_indices={1},
            intermediate_indices=idx,
            bo_key=f"ple_{tag}_L{k}",
            shared_nonstatic=True,
        )

    def _preload(self):
        """Write every layer's weights into per-layer resident BOs once, using
        the SAME arg layouts the prefill uses (static_input_indices then skips
        the weight writes on every subsequent call)."""
        print("[g4_prefill] pre-loading layer weights (per-layer BOs)...", flush=True)
        prof = self.cache.profiler.enabled
        self.cache.profiler.enabled = False
        seq = self.seq
        z_d = np.zeros((seq, D), bfloat16)
        z_p = np.zeros((seq, PLI_D), bfloat16)
        for k in range(self.n_layers):
            w, nm, pw = self._w[k], self._nm[k], self._ple[k]
            inter = wid_inter(_wid(k))
            z_i = np.zeros((seq, inter), bfloat16)
            z_q = np.zeros((seq, dims(k)[1]), bfloat16)
            self._call_rms_q(k, z_d)
            if owns_kv(k):
                self._call_k(k, z_d)
                self._call_v(k, z_d)
            self._call_o_norm(k, z_q, z_d)
            self._call_ffn_gemm(K_GATE(_wid(k)), k, "gate", z_d, _A_GATE)
            self._call_ffn_gemm(K_UP(_wid(k)), k, "up", z_d, _A_UP)
            self._call_gelu_mul(
                K_GELU(_wid(k)),
                k,
                inter,
                z_i,
                z_i,
                {0: _A_GATE, 1: _A_UP, 2: _A_ACT},
            )
            self._call_gemm_norm_add(
                K_DOWN(_wid(k)),
                k,
                inter,
                z_i,
                w["down"],
                nm["post_ffn"],
                z_d,
                True,
                {0: _A_ACT, 5: _A_RES1},
            )
            self._call_ple_gemm(k, z_d, pw["inp_gate"], "gate")
            self._call_ple_gemm(k, z_d, pw["model_proj"], "mp")
            self._call_gelu_mul(K_PLE_GELU, k, PLI_D, z_p, z_p, None)
            self._call_gemm_norm_add(
                K_PLE_PROJ,
                k,
                PLI_D,
                z_p,
                pw["per_layer_projection"],
                nm["post_ple"],
                z_d,
                False,
                None,
            )
        self.cache.profiler.enabled = prof
        self._preload_lm_head_gemv()
        print(f"  Pre-loaded {self.n_layers} layers", flush=True)

    def _preload_lm_head_gemv(self):
        """Build the bf16 lm_head partitions [16384, D] and write them into
        resident BOs once (static; skipped thereafter).

        lm_head here is its OWN 4-bit matrix, not tied to embed_tokens, and is
        1.6 GB dequantized in f32 -- so it is walked in partition-sized chunks
        straight into bf16 rather than materialized whole.
        """
        self._lm_parts = [
            np.asarray(
                self._qm.lm_head_rows(p * _LM_N_PART, (p + 1) * _LM_N_PART), bfloat16
            )
            for p in range(_LM_N_PARTITIONS)
        ]
        self._lm_head_npu(np.zeros(D, bfloat16))  # allocate + write the weight BOs

    def _lm_head_npu(self, hidden_bf16):
        """On-device logits from one bf16 hidden row [D] -> [VOCAB]."""
        lm_inputs = [np.ascontiguousarray(hidden_bf16, bfloat16)]
        for p in range(_LM_N_PARTITIONS):
            lm_inputs.append(self._lm_parts[p])
            lm_inputs.append(np.zeros(_LM_N_PART, bfloat16))
        res = self.cache.load_and_run(
            K_LM,
            self._lm_backend,
            *lm_inputs,
            output_indices=[2 + 2 * p for p in range(_LM_N_PARTITIONS)],
            static_input_indices={1 + 2 * p for p in range(_LM_N_PARTITIONS)},
            intermediate_indices={2 + 2 * p for p in range(_LM_N_PARTITIONS)},
        )
        return np.concatenate(res, axis=0)[:VOCAB]

    def _per_layer_inputs(self, ids, embeds):
        """The PLE input for every layer, [seq, NUM_LAYERS, PLI_D].

        Computed ONCE from the token EMBEDDINGS -- not per layer from that
        layer's hidden state -- exactly as gemma4_e2b_q4nx_weights.per_layer_inputs
        does. The 35 (seq, D) x (D, PLI_D) projections run on the NPU (the same
        ple_gemm ELF the inp_gate branch uses); the norm, table add and scale are
        elementwise and stay on the host.
        """
        seq = self.seq
        tbl = self._qm.embed_rows("model.per_layer_token_embd.weight", ids)
        tbl = tbl.reshape(len(ids), NUM_LAYERS, PLI_D)
        norm_w = self._g["ple_proj_norm"]
        emb = np.zeros((seq, D), bfloat16)
        emb[: len(ids)] = np.asarray(embeds, bfloat16)
        out = np.zeros((seq, NUM_LAYERS, PLI_D), np.float32)
        for L in range(self.n_layers):
            proj = self._dev(
                self._call_ple_gemm,
                L,
                emb,
                self._ple[L]["model_proj"],
                "mp",
                tag="ple_mp",
            )[2].reshape(seq, PLI_D)
            proj = _rmsnorm(np.asarray(proj, np.float32) * PLE_MODEL_PROJ_SCALE, norm_w)
            out[: len(ids), L, :] = (proj[: len(ids)] + tbl[:, L, :]) * PLE_INPUT_SCALE
        return out

    def _run_layer(self, x, k, pli):
        """One Gemma4 decoder layer on device: attention, FFN, then the PLE tail."""
        from shared.infra.fa_headfirst import npu_fa_headfirst

        seq = self.seq
        dh, dq, dkv = dims(k)
        c, w = _cls(k), _wid(k)
        inter = wid_inter(w)

        res = self._dev(self._call_rms_q, k, x, tag="rms_q")
        q_roped = res[8].reshape(seq, dq)
        if owns_kv(k):
            normed = res[2].reshape(seq, D)
            kk = self._dev(self._call_k, k, normed, tag="k")
            vv = self._dev(self._call_v, k, normed, tag="v")
            self.kv_k[k][:] = np.asarray(kk[6].reshape(seq, dkv), bfloat16)
            self.kv_v[k][:] = np.asarray(vv[4].reshape(seq, dkv), bfloat16)
        src = kv_source_layer(k)

        attn_out = self._dev(
            npu_fa_headfirst,
            self.cache,
            np.ascontiguousarray(q_roped),
            np.ascontiguousarray(self.kv_k[src]),
            np.ascontiguousarray(self.kv_v[src]),
            N_Q_HEADS,
            N_KV_HEADS,
            dh,
            seq,
            self._verbose,
            K_FA(c),
            tag="attn",
        )

        ores = self._dev(self._call_o_norm, k, attn_out, x, tag="o_norm")
        res1 = ores[6].reshape(seq, D)
        normed2 = ores[8].reshape(seq, D)

        gate = self._dev(
            self._call_ffn_gemm, K_GATE(w), k, "gate", normed2, _A_GATE, tag="gate"
        )[2].reshape(seq, inter)
        up = self._dev(self._call_ffn_gemm, K_UP(w), k, "up", normed2, _A_UP, tag="up")[
            2
        ].reshape(seq, inter)
        act = self._dev(
            self._call_gelu_mul,
            K_GELU(w),
            k,
            inter,
            gate,
            up,
            {0: _A_GATE, 1: _A_UP, 2: _A_ACT},
            tag="gelu",
        )[2].reshape(seq, inter)
        o2 = self._dev(
            self._call_gemm_norm_add,
            K_DOWN(w),
            k,
            inter,
            act,
            self._w[k]["down"],
            self._nm[k]["post_ffn"],
            res1,
            True,
            {0: _A_ACT, 5: _A_RES1},
            tag="down",
        )[6].reshape(seq, D)

        # ---- per-layer embedding branch ----
        pw = self._ple[k]
        g = self._dev(
            self._call_ple_gemm,
            k,
            o2,
            pw["inp_gate"],
            "gate",
            tag="ple_gate",
        )[2].reshape(seq, PLI_D)
        gated = self._dev(
            self._call_gelu_mul, K_PLE_GELU, k, PLI_D, g, pli, None, tag="ple_gelu"
        )[2].reshape(seq, PLI_D)
        o3 = self._dev(
            self._call_gemm_norm_add,
            K_PLE_PROJ,
            k,
            PLI_D,
            gated,
            pw["per_layer_projection"],
            self._nm[k]["post_ple"],
            o2,
            False,
            None,
            tag="ple_proj",
        )[6].reshape(seq, D)

        # layer_output_scale. A scalar on the RESIDUAL stream, so it cannot be
        # folded into the next layer's input_layernorm (RMSNorm is scale
        # invariant -- the normed path would not see it, but the residual does).
        return np.asarray(
            np.asarray(o3, np.float32) * self._nm[k]["out_scale"], bfloat16
        )

    def prefill(self, ids):
        assert self._w is not None, "call load_weights() first"
        N = len(ids)
        assert N <= self.seq, (N, self.seq)
        base = self.current_context_length
        # The q4nx bundle's embed_tokens is ALREADY scaled by sqrt(hidden_size)
        # (Gemma's normalizer), so gather as-is -- EMBED_SCALE is 1.
        emb = self._qm.embed_rows("model.embed_tokens.weight", ids) * EMBED_SCALE
        pli_all = self._per_layer_inputs(ids, emb)

        x = np.zeros((self.seq, D), bfloat16)
        x[:N] = np.asarray(emb, bfloat16)
        for k in range(self.n_layers):
            x = self._run_layer(x, k, np.asarray(pli_all[:, k, :], bfloat16))
        self.current_context_length = base + N

        # Final RMSNorm on the single prediction row (host, <1ms), then NPU LM
        # head, then Gemma4's tanh logit softcap.
        xn = _rmsnorm(np.asarray(x[N - 1], np.float32), self._g["final_norm"])
        logits = np.asarray(self._lm_head_npu(np.asarray(xn, bfloat16)), np.float32)
        if FINAL_LOGIT_SOFTCAP:
            logits = FINAL_LOGIT_SOFTCAP * np.tanh(logits / FINAL_LOGIT_SOFTCAP)
        return logits

    # ---- KV cache (causal_lm) ----
    def get_k_cache(self, layer_idx, idx):
        return self.kv_k[kv_source_layer(layer_idx)][idx]

    def get_v_cache(self, layer_idx, idx):
        return self.kv_v[kv_source_layer(layer_idx)][idx]

    def kv_view(self, layer_idx):
        """(roped_K, V-normed V) for the filled context -> decode handoff."""
        c = self.current_context_length
        src = kv_source_layer(layer_idx)
        return self.kv_k[src][:c], self.kv_v[src][:c]

    def kv_stack(self):
        """Per-layer (K, V) lists for the filled context.

        NOT a single stacked array, unlike the other q4nx prefills: head_dim is
        256 on the sliding layers and 512 on the full ones, so the per-layer
        slices are not the same shape and cannot be stacked.
        """
        c = self.current_context_length
        ks, vs = [], []
        for k in range(self.n_layers):
            src = kv_source_layer(k)
            ks.append(self.kv_k[src][:c].astype(np.float32))
            vs.append(self.kv_v[src][:c].astype(np.float32))
        return ks, vs

    def clear_context(self):
        self.current_context_length = 0
        for k in self.kv_k:
            self.kv_k[k][:] = 0
            self.kv_v[k][:] = 0

    def get_current_context_length(self):
        return self.current_context_length

    def set_context_length(self, L):
        self.current_context_length = L


def _main():
    ap = argparse.ArgumentParser(description="Gemma4-E2B Q4NX prefill on NPU2")
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
        "--gate",
        action="store_true",
        help="also score the full logit vector against the CPU reference",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=float(os.environ.get("PREFILL_TOL", "0.99")),
        help="cosine floor for --gate",
    )
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
        f"[g4_prefill] constructing seq_len={args.seq_len} (compiling engines)...",
        flush=True,
    )
    model = Gemma4Q4nxPrefill(
        seq_len=args.seq_len, n_layers=args.n_layers, cache_dir=args.cache_dir
    )
    if args.compile_only:
        print("Compilation passed.", flush=True)
        return 0

    print("[g4_prefill] loading Q4NX weights (host dequant)...", flush=True)
    model.load_weights(model=args.model)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    # The bos id is prepended by hand, exactly as run_reference.py's `paris`
    # does -- this tokenizer does not add it, and without it the two paths are
    # scoring different prompts.
    ids = [2] + tok(PROMPT_TEXT)["input_ids"]
    print(f"[g4_prefill] prefill prompt N={len(ids)} ...", flush=True)
    logits = np.asarray(model.prefill(ids), np.float32)
    top = int(logits.argmax())
    text = tok.decode([top]).strip()
    print(f"[g4_prefill] first-token argmax={top} {text!r} (expect {EXPECT_TEXT!r})")
    paris = text == EXPECT_TEXT
    print("[g4_prefill] *** PARIS ***" if paris else "[g4_prefill] MISS", flush=True)
    # Only --gate makes a verdict. profile-prefill runs this same path for its
    # TTFT numbers and is documented latency-only, so a missed argmax there must
    # not become a non-zero exit and fail a sweep point.
    ok = paris if args.gate else True

    if args.gate and args.n_layers != NUM_LAYERS:
        raise SystemExit(
            f"--gate compares against forward_prompt, which always runs all "
            f"{NUM_LAYERS} layers; --n-layers={args.n_layers} would score two "
            f"different networks. Drop --n-layers or drop --gate."
        )
    if args.gate:
        # argmax alone is a weak gate on this model: a doubled embedding scale
        # and a per-layer-embedding read from the wrong tensor BOTH still
        # predicted ' Paris' here. Score the whole logit vector against the CPU
        # reference the example already ships.
        import gemma4_e2b_q4nx_weights as W

        print(
            "[g4_prefill] scoring vs the CPU reference (slow, streams the "
            "weights layer by layer)...",
            flush=True,
        )
        ref = np.asarray(W.forward_prompt(model._qm, ids)[0], np.float32)
        cos = float(
            (ref * logits).sum() / (np.linalg.norm(ref) * np.linalg.norm(logits))
        )
        ref_top = int(ref.argmax())
        print(
            f"[g4_prefill] logit cosine vs CPU reference = {cos:.6f} "
            f"(floor {args.tol}); reference argmax={ref_top}"
        )
        if top != ref_top:
            print(f"[g4_prefill] FAIL: argmax {top} != reference {ref_top}")
            ok = False
        # Reject non-finite BEFORE the floor. `nan < tol` is False, so a NaN
        # cosine would otherwise pass the floor silently and leave the verdict
        # resting on the argmax alone -- and NaN is this design's characteristic
        # failure (see the one-GEMM-per-ELF note above).
        if not np.isfinite(cos):
            print(f"[g4_prefill] FAIL: cosine is {cos} (non-finite logits?)")
            ok = False
        elif cos < args.tol:
            print(f"[g4_prefill] FAIL: cosine {cos:.6f} below floor {args.tol}")
            ok = False
        print("[g4_prefill] GATE PASS" if ok else "[g4_prefill] GATE FAIL", flush=True)

    if args.bench_l:
        import time

        model.clear_context()
        ids_b = [int(t % VOCAB) for t in range(args.bench_l)]  # synthetic (timing only)
        print(f"[bench] warmup prefill L={args.bench_l}...", flush=True)
        model.prefill(ids_b)
        model.clear_context()
        model._dev_t = 0.0
        model._op_t.clear()
        print(f"[bench] timed prefill L={args.bench_l}...", flush=True)
        t0 = time.time()
        model.prefill(ids_b)
        wall = time.time() - t0
        npu = model._dev_t
        print(
            f"\n[bench] L={args.bench_l}: WALL={wall*1000:.0f}ms "
            f"{args.bench_l/wall:.0f} tok/s prefill  |  NPU-dispatch={npu*1000:.0f}ms "
            f"{args.bench_l/npu:.0f} tok/s  |  host={(wall-npu)*1000:.0f}ms",
            flush=True,
        )
        print(f"[g4_prefill] Inference: prompt_len={args.bench_l}, n_tokens=0")
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
