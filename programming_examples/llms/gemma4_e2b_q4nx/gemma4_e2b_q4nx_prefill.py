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


def _mul_backend(name):
    # The plain-multiply ELF's instance name is the eltwise_mul launch's own.
    return {
        "verbose": False,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "eltwise_mul",
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


# The wide GATE GEMM (n=12288) runs as two n=6144 halves writing into one
# contiguous output. Only the halves reach tile_k_l2=256: at n=12288 the BD
# stride cap (1048576, in 32-bit WORDS, so bf16 doubles the element limit to
# 2097152) pins tile_k_l2 to <=170, and the wider tile is what pays.
#
# Up splits too, now that the cast launch can write a column window. Both
# halves want the fused-cast path; forcing them onto drain is slower, which is
# what made an earlier split of up look flat.
#
# One GEMM per ELF, never two slices -- see the miscompile note above.


def _ffn_halves(w):
    """Column windows for one FFN GEMM: [(n, n_out, offset, tag)]."""
    inter = wid_inter(w)
    if w != "w":
        return [(inter, None, 0, "")]
    half = inter // 2
    return [(half, inter, 0, "_lo"), (half, inter, half, "_hi")]


def _ffn_weights(w, wid):
    """gate/up as the (K, N) operands their GEMMs want, split when wide.

    A split layer stores ONLY the halves: each must be its own contiguous array
    (a column slice keeps the 12288 row stride, which is the whole thing the
    split exists to get under), and keeping the full copies too would cost
    ~37 MB per matrix per layer.
    """
    inter = wid_inter(wid)
    out = {}
    for key in ("gate", "up"):
        full = _wT(w[key], D, inter)
        halves = _ffn_halves(wid)
        if len(halves) == 1:
            out[key] = full
            continue
        for n_h, _n_out, off, tag in halves:
            out[key + tag] = np.ascontiguousarray(full[:, off : off + n_h])
    return out


def K_FFN(w):
    """The whole FFN branch in one ELF: gate, up, multiply, down, norm, add.

    One air.launch is one PDI, and launches sharing a func have their PDI loads
    CHAINED, so this costs one XRT run instead of six. Measured: a chained
    launch is 0.332 ms against 0.487 for a standalone run.
    """
    return f"ffn_{w}"


K_PLE = "ple"  # the whole per-layer-embedding branch, one ELF like K_FFN
K_PLE_MP = "ple_mp_all"  # (seq, D) -> (seq, PLE_MP_N); ALL layers' model_proj
K_LM = "lm_head_gemv"

# Round-groups for the full-attention layers' causal K/V staircase. More groups
# skip more of the upper triangle but each costs a launch; past four the launch
# overhead outweighs the blocks saved.
_FA_CAUSAL_GROUPS = 4

# The cache keys on artifact NAME and validates only the toolchain, so a kernel
# whose layout or meaning changed is reused under its old name. Bump this.
# rev 2: the FFN's six ELFs became one ffn_<wid>.
# rev 3: the PLE's three became one ple. A rev-2 cache would keep its three
# orphans alongside the new ELF rather than being rejected.
_KERNEL_REV = 3

# The model_proj branch projects the SAME token embeddings through every
# layer's matrix, so FastFlowLM's `pli_down_proj` is one GEMM of width
# num_hidden_layers * PLI_D (gemma4e_prefill.cpp, pre_pass) rather than
# NUM_LAYERS narrow ones. N must be a multiple of tile_n * herd_n = 512, so the
# concatenation is padded by one layer's worth and the tail is discarded.
PLE_MP_N = -(-NUM_LAYERS * PLI_D // 512) * 512
PLE_MP_TK_L2 = 64


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


# The GELU-epilogue GEMMs are pinned to drain (the activation belongs on the
# GEMM's own 32 cores, not the 8-column cast launch), and drain defaults to
# tile_m=32 -- which is slower, because inbound bytes per MAC go as
# (tile_m+tile_n)/(tile_m*tile_n). tile_m=64 needs tile_n<=96 to fit L1
# (at 128 it is 72 KB and the chunked drain that exists for exactly this
# overruns the memtile's 48 BD blocks).
_DRAIN_WIDE_TILE = {"tile_m": 64, "tile_n": 96}


def _wide_drain(spec):
    """Re-point a drain spec at tile_m=64/tile_n=96 and its own mm.o.

    DIM_M and DIM_N are compile-time in mm.o, so this variant cannot share
    "_m32"/mm_m32.o with the tile_m=32 drain.
    """
    tag = f"_m{_DRAIN_WIDE_TILE['tile_m']}n{_DRAIN_WIDE_TILE['tile_n']}"
    return _retag(dict(spec, **_DRAIN_WIDE_TILE), tag)


# Suffix for the PLE projection's copy of the tile_m=32 drain kernel. Two GEMMs
# in one ELF cannot share a symbol unless their herd_n matches: herd_n sets the
# outer stride of the f32 accumulator subview, which is part of the extern's
# type. The PLE's gate is 2 columns wide and its projection 4, so the shared
# "_m32" decl would be emitted twice with different types.
_PLE_PROJ_TAG = "_m32c4"


def _retag(spec, tag):
    """Point a spec at its own symbol suffix and mm.o, tiling unchanged."""
    spec = dict(spec, sym_suffix=tag, obj=f"mm{tag}.o")
    spec["build_kwargs"] = dict(
        spec["build_kwargs"], sym_suffix=tag, link_with_name=f"mm{tag}.o"
    )
    return spec


def gemm_spec(m, k, n, precision="high", force_method=None, tile_k_l2=None):
    """Per-GEMM build recipe for one Gemma4 shape.

    `force_method` overrides the size rule. The gate GEMM uses it to take the
    DRAIN path: that is the only one whose cast runs inside the GEMM's own herd
    (32 cores), which is where FastFlowLM puts its GELU -- fused-cast's separate
    cast launch is capped at 8 columns by the shim budget.
    """
    from shared.builders.gemm_builder import _spec_with_tiles

    method = force_method or ("fused-cast" if m * k * n >= 4e9 else "drain")
    tile_m = 64 if method == "fused-cast" else 32
    # The weight walk emits tile_k_l2 * n as its K-tile DMA stride, and NPU2
    # caps a BD stride at 1048576. The double-wide FFN (n=12288) blows that at
    # tile_k_l2=256 -- "'aie.dma_bd' op Stride 2 exceeds the [1:1048576] range",
    # 3145728 -- so it drops to 64, the largest multiple of 32 dividing K=1536
    # that fits. Applied where it is OBSERVED to bite, not from the formula:
    # n=6144 at tile_k_l2=256 exceeds the same arithmetic and compiles anyway,
    # because the emitted BD factors differently there.
    # n=9216 (the batched model_proj) blows it too, hence the explicit override
    # -- the knob barely moves the dispatch, so lowering it where a width needs
    # it costs nothing worth protecting.
    if tile_k_l2 is None:
        tile_k_l2 = 64 if n >= 12288 else min(256, k)
    spec = _spec_with_tiles(
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
    # n % (tile_n * herd_n) must stay exact, so only widths divisible by 384.
    if method == "drain" and force_method == "drain" and n % 384 == 0:
        spec = _wide_drain(spec)
    return spec


def gemm_herd_n(n, tile_n):
    """Herd columns for an N-wide GEMM: one per tile, capped at the herd's 4."""
    return max(1, min(4, n // tile_n))


def _build_gemm_ir(
    m,
    k,
    n,
    spec,
    herd_m=8,
    herd_n=None,
    epilogue_gelu=False,
    n_out=None,
    n_out_offset=0,
):
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
            epilogue_gelu=epilogue_gelu,
            n_out=n_out,
            n_out_offset=n_out_offset,
            **dict(spec["build_kwargs"]),
        )
    )


def _gemm_externs(spec, epilogue_gelu=False):
    sfx = spec["sym_suffix"]
    syms = {
        "@matmul_bf16",
        "@op_has_no_registered_library_name" + sfx,
        "@zero_f32_mn" + sfx,
        "@f32_to_bf16_mn" + sfx,
    }
    if epilogue_gelu:
        syms.add("@f32_to_bf16_gelu_mn" + sfx)
    return syms


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


def _build_single_gemm_elf(
    name,
    sym,
    seq_len,
    k_dim,
    n_dim,
    herd_m=8,
    herd_n=4,
    epilogue_gelu=False,
    tile_k_l2=None,
    n_out=None,
    n_out_offset=0,
):
    """Standalone single-GEMM ELF: arg0 in, arg1 weight, arg2 out.

    `epilogue_gelu` applies GELU-tanh in the GEMM's own cast launch, which is
    FastFlowLM's `Gemm::GeLU` output mode: the activation is a pointwise
    function of an accumulator that launch already holds, so it costs no
    operand and no DMA, and the GeGLU pass left behind is a plain multiply --
    which the registry measures at 55.7 GB/s against gelu-and-mul's 13.1.
    """
    from shared.infra.stitching import (
        stitch_elf,
        KernelSlice,
        FuncArg,
        alloc_gemm_scratch,
    )

    # n_out: this GEMM writes its n_dim columns into an n_out-wide output at
    # n_out_offset, so a wide FFN can run as narrower halves that still land
    # contiguous. Both methods can do it; the fused cast is the faster one.
    g_spec = gemm_spec(
        seq_len,
        k_dim,
        n_dim,
        force_method="drain" if epilogue_gelu else None,
        tile_k_l2=tile_k_l2,
    )
    print(
        f"  [{name}] GEMM ({g_spec['method']}) {seq_len}x{k_dim}x{n_dim} "
        f"(tk_l2={g_spec['tile_k_l2']}, tn={g_spec['tile_n']})..."
    )
    gemm_ir = _build_gemm_ir(
        seq_len,
        k_dim,
        n_dim,
        g_spec,
        herd_m,
        epilogue_gelu=epilogue_gelu,
        n_out=n_out,
        n_out_offset=n_out_offset,
    )
    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{k_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{k_dim}x{n_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{n_out or n_dim}xbf16>"),
    ]
    scratch_args, scratch_for = alloc_gemm_scratch([(g_spec, seq_len, n_dim)], 3)
    slices = [
        KernelSlice(
            gemm_ir,
            sym,
            _gemm_amap(0, 1, 2, scratch_for[0]),
            extern_syms=_gemm_externs(g_spec, epilogue_gelu),
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


def build_mul_module(seq_len, hidden_dim, herd_x=8, herd_y=1):
    """Standalone NPU elementwise multiply ELF: gate * up -> (seq, hidden_dim).

    What replaced the GeGLU pass once the gate GEMM grew a GELU epilogue. The
    split is FastFlowLM's: its gate GEMM carries the activation as an RTP output
    mode and `mlp_block` does only the multiply. Worth doing rather than leaving
    the fused kernel in place -- the kernel registry measures gelu-and-mul at
    13.1 GB/s and a plain multiply at 55.7, both on the 8 tiles the 3-stream
    shim budget allows, so the tanh was the whole difference.
    """
    from eltwise_mul.eltwise_mul import build_eltwise_mul

    tile = _gelu_tile_n(seq_len, hidden_dim, herd_x)
    print(f"  [mul] {seq_len}x{hidden_dim} (tile={tile})...")
    module = build_eltwise_mul(
        [seq_len * hidden_dim], tile=tile, herd_shape=(herd_x * herd_y,)
    ).build(target="npu2")
    print(f"  mul module: {len(str(module).splitlines())} lines, parsed OK")
    return module


def ffn_specs(seq_len, wid):
    """(gate spec, up spec, down spec) for one FFN width.

    The gate is forced to drain because its GELU rides in that method's own
    cast; the up GEMM keeps whatever the size rule picks, which above the
    cross-over is fused-cast and therefore needs an f32 scratch arg.
    """
    n_h = _ffn_halves(wid)[0][0]
    return (
        gemm_spec(seq_len, D, n_h, force_method="drain"),
        gemm_spec(seq_len, D, n_h),
        gemm_spec(seq_len, wid_inter(wid), D),
    )


def ffn_args(seq_len, wid):
    """The K_FFN signature: (base args, scratch args, index map, scratch map).

    Shared by the builder and the dispatch so the two cannot drift. Arg order
    follows the dataflow -- normed2 -> gate/up -> act -> proj -> proj_n -> out
    -- with the fused-cast f32 scratch on the tail, as everywhere else.
    """
    from shared.infra.stitching import FuncArg, alloc_gemm_scratch

    inter = wid_inter(wid)
    halves = _ffn_halves(wid)
    g_spec, u_spec, d_spec = ffn_specs(seq_len, wid)
    args = [FuncArg("%arg0", f"memref<{seq_len}x{D}xbf16>")]
    idx = {"normed2": 0, "gate_w": [], "up_w": []}
    for key in ("gate_w", "up_w"):
        for n_h, *_ in halves:
            idx[key].append(len(args))
            args.append(FuncArg(f"%arg{len(args)}", f"memref<{D}x{n_h}xbf16>"))
    for nm, ty in (
        ("gate", f"memref<{seq_len}x{inter}xbf16>"),
        ("up", f"memref<{seq_len}x{inter}xbf16>"),
        ("act", f"memref<{seq_len * inter}xbf16>"),
        ("down_w", f"memref<{inter}x{D}xbf16>"),
        ("proj", f"memref<{seq_len}x{D}xbf16>"),
        ("norm_w", f"memref<{D}xbf16>"),
        ("proj_n", f"memref<{seq_len}x{D}xbf16>"),
        ("resid", f"memref<{seq_len}x{D}xbf16>"),
        ("out", f"memref<{seq_len * D}xbf16>"),
    ):
        idx[nm] = len(args)
        args.append(FuncArg(f"%arg{len(args)}", ty))
    # Same order as the slices below: gate halves, up halves, down.
    n_h = halves[0][0]
    scratch_args, scratch_for = alloc_gemm_scratch(
        [(g_spec, seq_len, n_h)] * len(halves)
        + [(u_spec, seq_len, n_h)] * len(halves)
        + [(d_spec, seq_len, D)],
        len(args),
    )
    return args, scratch_args, idx, scratch_for


def build_ffn_module(seq_len, wid, herd_m=8):
    """Gemma4's whole FFN branch as ONE ELF: the six dispatches it replaces are
    six air.launches whose PDI loads the compiler chains.

    gate (GELU epilogue) -> up -> multiply -> down -> post-FFN norm -> residual.
    The wide layers run gate and up as two column halves writing one buffer, so
    a wide ELF carries eight launches and a narrow one six.
    """
    from shared.infra.stitching import (
        _wrap_ir_in_launch,
        stitch_elf,
        KernelSlice,
        build_add_2d_to_1d_ir,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    inter = wid_inter(wid)
    halves = _ffn_halves(wid)
    args, scratch_args, ix, scratch_for = ffn_args(seq_len, wid)
    g_spec, u_spec, d_spec = ffn_specs(seq_len, wid)
    name = K_FFN(wid)

    print(
        f"  [1/4] {name} gate ({g_spec['method']}) + up ({u_spec['method']}) "
        f"{seq_len}x{D}x{inter} as {len(halves)} half/halves..."
    )
    slices, s_at = [], 0
    for tgt, spec, gelu, wkey in (
        ("gate", g_spec, True, "gate_w"),
        ("up", u_spec, False, "up_w"),
    ):
        for i, (n_h, n_out, off, _tag) in enumerate(halves):
            ir = _build_gemm_ir(
                seq_len,
                D,
                n_h,
                spec,
                herd_m,
                epilogue_gelu=gelu,
                n_out=n_out,
                n_out_offset=off,
            )
            slices.append(
                KernelSlice(
                    ir,
                    f"{tgt}{i}",
                    _gemm_amap(ix["normed2"], ix[wkey][i], ix[tgt], scratch_for[s_at]),
                    extern_syms=_gemm_externs(spec, gelu),
                    # Every GEMM slice contributes its private decls: the gate
                    # declares the GELU cast and the up the plain one, and
                    # stitch_elf unions them (identical text collapses, a
                    # genuine type clash fails loudly).
                    private_from=True,
                )
            )
            s_at += 1

    print(f"  [2/4] gate*up multiply {seq_len}x{inter}...")
    slices.append(
        KernelSlice(
            str(build_mul_module(seq_len, inter)),
            "ml",
            {2: ix["act"]},
            arg_aliases={0: "%gate_flat", 1: "%up_flat"},
        )
    )

    print(f"  [3/4] down GEMM ({d_spec['method']}) {seq_len}x{inter}x{D}...")
    d_map = _gemm_amap(0, ix["down_w"], ix["proj"], scratch_for[s_at])
    d_map.pop(0)  # the activation comes from the prelude view, not an arg
    slices.append(
        KernelSlice(
            _build_gemm_ir(seq_len, inter, D, d_spec, herd_m),
            "dg",
            d_map,
            arg_aliases={0: "%act_2d"},
            extern_syms=_gemm_externs(d_spec),
        )
    )

    with _rms_eps():
        print(f"  [4/4] post-FFN RMSNorm (eps={RMS_EPS:g}) + residual...")
        post_ir = _wrap_ir_in_launch(str(build_rms(seq_len, D, bfloat16, 16, herd_x=8)))
    slices.append(
        KernelSlice(post_ir, "pn", {0: ix["proj"], 1: ix["norm_w"], 2: ix["proj_n"]})
    )
    slices.append(
        KernelSlice(
            build_add_2d_to_1d_ir(seq_len, D),
            "ad",
            {0: ix["proj_n"], 1: ix["resid"], 2: ix["out"]},
        )
    )

    # The GEMMs write [seq, inter] and the multiply reads [seq*inter]; a view
    # in the prelude joins them without retiling either kernel. It has to be
    # reinterpret_cast: the DMA lowering accepts subview/view/cast/
    # reinterpret_cast rooted at a block argument and rejects collapse_shape
    # and expand_shape outright.
    flat = f"memref<{seq_len * inter}xbf16>"
    two_d = f"memref<{seq_len}x{inter}xbf16>"
    prelude = "\n".join(
        [
            f"    %{nm} = memref.reinterpret_cast %arg{ix[src]} to offset: [0], "
            f"sizes: {sizes}, strides: {strides} : {frm} to {to}"
            for nm, src, sizes, strides, frm, to in (
                ("gate_flat", "gate", f"[{seq_len * inter}]", "[1]", two_d, flat),
                ("up_flat", "up", f"[{seq_len * inter}]", "[1]", two_d, flat),
                (
                    "act_2d",
                    "act",
                    f"[{seq_len}, {inter}]",
                    f"[{inter}, 1]",
                    flat,
                    two_d,
                ),
            )
        ]
    )
    module = stitch_elf(name, args, slices, scratch_args=scratch_args, prelude=prelude)
    print(
        f"  {name} module: {len(str(module).splitlines())} lines, "
        f"{len(args) + len(scratch_args)} args, {len(slices)} slices, parsed OK"
    )
    return module, scratch_for


def _ple_specs(seq_len):
    """(gate, projection) GEMM specs for the PLE branch, shared by the two
    callers so the projection's retag cannot drift between them."""
    return (
        gemm_spec(seq_len, D, PLI_D, force_method="drain"),
        _retag(gemm_spec(seq_len, PLI_D, D), _PLE_PROJ_TAG),
    )


def ple_args(seq_len):
    """The K_PLE signature: (base args, scratch args, index map, scratch map).

    `x` is both the branch input and its residual, so it appears once and two
    slices read it.
    """
    from shared.infra.stitching import FuncArg, alloc_gemm_scratch

    g_spec, p_spec = _ple_specs(seq_len)
    args, idx = [], {}
    for nm, ty in (
        ("x", f"memref<{seq_len}x{D}xbf16>"),
        ("gate_w", f"memref<{D}x{PLI_D}xbf16>"),
        ("g", f"memref<{seq_len}x{PLI_D}xbf16>"),
        ("pli", f"memref<{seq_len * PLI_D}xbf16>"),
        ("gated", f"memref<{seq_len * PLI_D}xbf16>"),
        ("proj_w", f"memref<{PLI_D}x{D}xbf16>"),
        ("proj", f"memref<{seq_len}x{D}xbf16>"),
        ("norm_w", f"memref<{D}xbf16>"),
        ("proj_n", f"memref<{seq_len}x{D}xbf16>"),
        ("out", f"memref<{seq_len}x{D}xbf16>"),
    ):
        idx[nm] = len(args)
        args.append(FuncArg(f"%arg{len(args)}", ty))
    scratch_args, scratch_for = alloc_gemm_scratch(
        [(g_spec, seq_len, PLI_D), (p_spec, seq_len, D)], len(args)
    )
    return args, scratch_args, idx, scratch_for


def build_ple_module(seq_len, herd_m=8):
    """Gemma4's per-layer-embedding branch as ONE ELF.

    inp_gate GEMM (GELU epilogue) -> multiply by the layer's PLE input ->
    per_layer_projection -> post_ple norm -> residual. Kept out of K_FFN
    because folding it in would need a third retagged copy of the drain kernel:
    the PLE widths give a different herd_n again, and herd_n is part of the
    cast extern's type.
    """
    from shared.infra.stitching import (
        _wrap_ir_in_launch,
        stitch_elf,
        KernelSlice,
        build_residual_add_2d_ir,
    )
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    args, scratch_args, ix, scratch_for = ple_args(seq_len)
    g_spec, p_spec = _ple_specs(seq_len)

    print(f"  [1/4] inp_gate GEMM ({g_spec['method']}) {seq_len}x{D}x{PLI_D} + GELU...")
    slices = [
        KernelSlice(
            _build_gemm_ir(seq_len, D, PLI_D, g_spec, herd_m, epilogue_gelu=True),
            "pg",
            _gemm_amap(ix["x"], ix["gate_w"], ix["g"], scratch_for[0]),
            extern_syms=_gemm_externs(g_spec, True),
        )
    ]

    print(f"  [2/4] gate * per-layer input {seq_len}x{PLI_D}...")
    slices.append(
        KernelSlice(
            str(build_mul_module(seq_len, PLI_D)),
            "ml",
            {1: ix["pli"], 2: ix["gated"]},
            arg_aliases={0: "%g_flat"},
        )
    )

    print(f"  [3/4] per_layer_projection ({p_spec['method']}) {seq_len}x{PLI_D}x{D}...")
    p_map = _gemm_amap(0, ix["proj_w"], ix["proj"], scratch_for[1])
    p_map.pop(0)
    slices.append(
        KernelSlice(
            _build_gemm_ir(seq_len, PLI_D, D, p_spec, herd_m),
            "pp",
            p_map,
            arg_aliases={0: "%gated_2d"},
            extern_syms=_gemm_externs(p_spec),
        )
    )

    with _rms_eps():
        print(f"  [4/4] post_ple RMSNorm (eps={RMS_EPS:g}) + residual...")
        post_ir = _wrap_ir_in_launch(str(build_rms(seq_len, D, bfloat16, 16, herd_x=8)))
    slices.append(
        KernelSlice(post_ir, "pn", {0: ix["proj"], 1: ix["norm_w"], 2: ix["proj_n"]})
    )
    slices.append(
        KernelSlice(
            build_residual_add_2d_ir(seq_len, D),
            "ad",
            {0: ix["proj_n"], 1: ix["x"], 2: ix["out"]},
        )
    )

    flat = f"memref<{seq_len * PLI_D}xbf16>"
    two_d = f"memref<{seq_len}x{PLI_D}xbf16>"
    prelude = "\n".join(
        [
            f"    %g_flat = memref.reinterpret_cast %arg{ix['g']} to offset: [0], "
            f"sizes: [{seq_len * PLI_D}], strides: [1] : {two_d} to {flat}",
            f"    %gated_2d = memref.reinterpret_cast %arg{ix['gated']} to "
            f"offset: [0], sizes: [{seq_len}, {PLI_D}], strides: [{PLI_D}, 1] "
            f": {flat} to {two_d}",
        ]
    )
    module = stitch_elf(K_PLE, args, slices, scratch_args=scratch_args, prelude=prelude)
    print(
        f"  {K_PLE} module: {len(str(module).splitlines())} lines, "
        f"{len(args) + len(scratch_args)} args, {len(slices)} slices, parsed OK"
    )
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
    + [K_FFN(w) for w in _WID]
    + [K_PLE, K_PLE_MP, K_LM]
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
    _wt = _DRAIN_WIDE_TILE
    _wtag = f"_m{_wt['tile_m']}n{_wt['tile_n']}"
    compile_gemm_mm(
        tile_m=_wt["tile_m"],
        tile_n=_wt["tile_n"],
        tile_k_l1=32,
        sym_suffix=_wtag,
        out_name=f"mm{_wtag}.o",
    )
    # Same tiling as mm_m32.o under another name -- see _PLE_PROJ_TAG.
    compile_gemm_mm(
        tile_m=32,
        tile_n=128,
        tile_k_l1=32,
        sym_suffix=_PLE_PROJ_TAG,
        out_name=f"mm{_PLE_PROJ_TAG}.o",
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
        print(
            f"\n--- {K_FFN(w)} (gate + up + multiply + down + norm + residual, "
            f"INTER={wid_inter(w)}) ---"
        )
        mod, scratch[K_FFN(w)] = build_ffn_module(seq_len, w)
        cache.compile_and_cache(K_FFN(w), mod, _elf_backend(K_FFN(w)))

    # --- PLE. One set of ELFs for all 35 layers: the branch is PLI_D-wide
    # regardless of attention class or FFN width.
    print(f"\n--- {K_PLE} (inp_gate + multiply + projection + norm + residual) ---")
    mod, scratch[K_PLE] = build_ple_module(seq_len)
    cache.compile_and_cache(K_PLE, mod, _elf_backend(K_PLE))

    print(f"\n--- {K_PLE_MP} (D -> {PLE_MP_N}; all layers' model_proj at once) ---")
    mod, scratch[K_PLE_MP] = _build_single_gemm_elf(
        K_PLE_MP, "pm", seq_len, D, PLE_MP_N, tile_k_l2=PLE_MP_TK_L2
    )
    cache.compile_and_cache(K_PLE_MP, mod, _elf_backend(K_PLE_MP))

    # --- Attention. One ELF per class: the sliding layers carry the window
    # mask AND head_dim=256, the full layers plain causal at head_dim=512.
    from shared.infra.fa_headfirst import compile_headfirst_fa
    from shared.infra.fa_headspatial import (
        compile_headspatial_fa,
        hs_tiling,
        supports,
    )

    for c in _CLS:
        dh = cls_dims(c)[0]
        win = SLIDING_WINDOW if c == "swa" else None
        # Head-spatial where it fits: under MQA one K/V broadcast serves four
        # resident heads, so K and V cross L3 twice per layer instead of eight
        # times.
        hs = supports(dh, N_KV_HEADS)
        kind = "head-spatial" if hs else "head-first"
        print(f"\n--- {K_FA(c)} ({kind} FA, head_dim={dh}, window={win}) ---")
        # The staircase splits the round axis, so the group count has to
        # divide it; take the largest divisor rather than refuse the build.
        extra = {}
        if hs and win is None:
            n_rounds = seq_len // hs_tiling(dh)[1]
            extra["causal_groups"] = next(
                g for g in range(_FA_CAUSAL_GROUPS, 0, -1) if n_rounds % g == 0
            )
        # causal_skip is the head-first path's per-block elision; the
        # head-spatial one skips in the DMA instead and has no such knob.
        if not hs:
            extra["causal_skip"] = True
        (compile_headspatial_fa if hs else compile_headfirst_fa)(
            cache,
            seq_len,
            N_Q_HEADS,
            N_KV_HEADS,
            dh,
            verbose,
            window=win,
            name=K_FA(c),
            **extra,
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
_A_NORMED2 = "ffn_normed2"  # o_norm -> ffn
_A_RES1 = "ffn_res1"  # o_norm -> ffn
# gate/up/act used to need names here too; they are now plain intermediates
# INSIDE the ffn ELF, so they never cross a dispatch boundary at all.


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
        # ffn_args owns this numbering for both the builder and the dispatch,
        # so the run-only path re-derives it rather than re-stating it.
        sc[K_FFN(w)] = ffn_args(seq_len, w)[3]
    sc[K_PLE] = ple_args(seq_len)[3]
    sc[K_PLE_MP] = _alloc([gemm_spec(seq_len, D, PLE_MP_N, tile_k_l2=PLE_MP_TK_L2)], 3)
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
        stamp = f"{seq_len}/rev{_KERNEL_REV}"
        if self._seq_stamp.is_file():
            was = self._seq_stamp.read_text().strip()
            if was != stamp:
                raise SystemExit(
                    f"cache {cache_dir} was built for {was}, not {stamp}. "
                    f"Point --cache-dir elsewhere or remove it."
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
        self._seq_stamp.write_text(stamp)

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
        self._mp_w = None
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
            **_ffn_weights(w, _wid(k)),
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

    def _call_ffn(self, k, normed2, resid):
        """The whole FFN branch in one dispatch. Returns (seq*D,) flat.

        Everything between normed2 and the output -- gate, up, act, proj,
        proj_n -- is an intermediate: the host neither writes nor reads it, and
        the six ELFs this replaced used `shared_alias` to get the same effect
        across dispatch boundaries.
        """
        seq, w = self.seq, _wid(k)
        inter = wid_inter(w)
        halves = _ffn_halves(w)
        name = K_FFN(w)
        _args, _sa, ix, scratch_for = ffn_args(seq, w)
        vals = {
            ix["normed2"]: np.asarray(normed2, bfloat16).reshape(seq, D),
            ix["gate"]: np.zeros((seq, inter), bfloat16),
            ix["up"]: np.zeros((seq, inter), bfloat16),
            ix["act"]: np.zeros(seq * inter, bfloat16),
            ix["down_w"]: self._w[k]["down"],
            ix["proj"]: np.zeros((seq, D), bfloat16),
            ix["norm_w"]: np.asarray(self._nm[k]["post_ffn"], bfloat16),
            ix["proj_n"]: np.zeros((seq, D), bfloat16),
            ix["resid"]: np.asarray(resid, bfloat16).reshape(seq, D),
            ix["out"]: np.zeros(seq * D, bfloat16),
        }
        for key, tag in (("gate_w", "gate"), ("up_w", "up")):
            for i, (_n, _no, _of, t) in enumerate(halves):
                vals[ix[key][i]] = self._w[k][tag + t]
        static = {ix["down_w"], ix["norm_w"]} | set(ix["gate_w"]) | set(ix["up_w"])
        inter_idx = {ix["gate"], ix["up"], ix["act"], ix["proj"], ix["proj_n"]}
        # A split half accumulates only its own columns, so its f32 scratch is
        # narrower than the shared bf16 buffer it writes into.
        n_h = halves[0][0]
        widths = [n_h] * (2 * len(halves)) + [D]
        for sc, wdt in zip(scratch_for, widths):
            if sc is not None:
                vals[sc] = np.zeros((seq, wdt), np.float32)
                inter_idx.add(sc)
        return self.cache.load_and_run(
            name,
            _elf_backend(name),
            *[vals[i] for i in sorted(vals)],
            output_indices=[ix["out"]],
            static_input_indices=static,
            intermediate_indices=inter_idx,
            bo_key=f"{name}_L{k}",
            shared_nonstatic=True,
            shared_alias={ix["normed2"]: _A_NORMED2, ix["resid"]: _A_RES1},
        )

    def _ple_mp_weight(self):
        """Every layer's model_proj side by side: (D, PLE_MP_N), built once."""
        if getattr(self, "_mp_w", None) is None:
            w = np.zeros((D, PLE_MP_N), bfloat16)
            for L in range(self.n_layers):
                w[:, L * PLI_D : (L + 1) * PLI_D] = self._ple[L]["model_proj"]
            self._mp_w = w
        return self._mp_w

    def _call_ple_mp_all(self, x):
        seq = self.seq
        args = [
            np.asarray(x, bfloat16).reshape(seq, D),
            self._ple_mp_weight(),
            np.zeros((seq, PLE_MP_N), bfloat16),
        ]
        idx = {2}
        for sc in self.scratch[K_PLE_MP]:
            if sc is not None:
                args.append(np.zeros((seq, PLE_MP_N), np.float32))
                idx.add(sc)
        return self.cache.load_and_run(
            K_PLE_MP,
            _elf_backend(K_PLE_MP),
            *args,
            output_indices=[2],
            static_input_indices={1},
            intermediate_indices=idx,
            bo_key="ple_mp_all",
            shared_nonstatic=True,
        )

    def _call_ple(self, k, x, pli):
        """The whole per-layer-embedding branch in one dispatch.

        `x` is both the branch input and its residual -- the ELF reads one arg
        twice rather than being handed the same buffer as two.
        """
        seq = self.seq
        pw = self._ple[k]
        _a, _sa, ix, scratch_for = ple_args(seq)
        vals = {
            ix["x"]: np.asarray(x, bfloat16).reshape(seq, D),
            ix["gate_w"]: np.asarray(pw["inp_gate"], bfloat16).reshape(D, PLI_D),
            ix["g"]: np.zeros((seq, PLI_D), bfloat16),
            ix["pli"]: np.asarray(pli, bfloat16).reshape(seq * PLI_D),
            ix["gated"]: np.zeros(seq * PLI_D, bfloat16),
            ix["proj_w"]: pw["per_layer_projection"],
            ix["proj"]: np.zeros((seq, D), bfloat16),
            ix["norm_w"]: np.asarray(self._nm[k]["post_ple"], bfloat16),
            ix["proj_n"]: np.zeros((seq, D), bfloat16),
            ix["out"]: np.zeros((seq, D), bfloat16),
        }
        inter_idx = {ix["g"], ix["gated"], ix["proj"], ix["proj_n"]}
        for sc, cols in zip(scratch_for, (PLI_D, D)):
            if sc is not None:
                vals[sc] = np.zeros((seq, cols), np.float32)
                inter_idx.add(sc)
        return self.cache.load_and_run(
            K_PLE,
            _elf_backend(K_PLE),
            *[vals[i] for i in sorted(vals)],
            output_indices=[ix["out"]],
            static_input_indices={ix["gate_w"], ix["proj_w"], ix["norm_w"]},
            intermediate_indices=inter_idx,
            bo_key=f"{K_PLE}_L{k}",
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
            self._call_ffn(k, z_d, z_d)
            self._call_ple(k, z_d, z_p)
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
        allp = self._dev(self._call_ple_mp_all, emb, tag="ple_mp")[2].reshape(
            seq, PLE_MP_N
        )
        for L in range(self.n_layers):
            proj = allp[:, L * PLI_D : (L + 1) * PLI_D]
            proj = _rmsnorm(np.asarray(proj, np.float32) * PLE_MODEL_PROJ_SCALE, norm_w)
            out[: len(ids), L, :] = (proj[: len(ids)] + tbl[:, L, :]) * PLE_INPUT_SCALE
        return out

    def _run_layer(self, x, k, pli):
        """One Gemma4 decoder layer on device: attention, FFN, then the PLE tail."""
        from shared.infra.fa_headfirst import npu_fa_headfirst
        from shared.infra.fa_headspatial import npu_fa_headspatial, supports

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
            npu_fa_headspatial if supports(dh, N_KV_HEADS) else npu_fa_headfirst,
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

        o2 = self._dev(self._call_ffn, k, normed2, res1, tag="ffn")[
            ffn_args(seq, w)[2]["out"]
        ].reshape(seq, D)

        # ---- per-layer embedding branch ----
        o3 = self._dev(self._call_ple, k, o2, pli, tag="ple")[
            ple_args(seq)[2]["out"]
        ].reshape(seq, D)

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
