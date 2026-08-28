# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RMSNorm + QKV GEMMs + RoPE Q+K — 6-launch multi-launch ELF.

Merges rms_attn_gemms (4 launches) + rope_qk (2 launches) into a single
AIR function with 6 sequential air.launch operations:
  1. RMSNorm      [8,1]   x_in x norm_w -> normed
  2. Q GEMM       [8,4]   normed x wq -> q
  3. K GEMM       [8,4]   normed x wk -> k
  4. V GEMM       [8,4]   normed x wv -> v
  5. RoPE Q       [8,1]   q(2D->1D) x lut_q -> q_roped(1D->2D)
  6. RoPE K       [8,1]   k(2D->1D) x lut_k -> k_roped(1D->2D)

13 func args (6 launches). Q/K GEMM outputs are 2D memrefs shared with
RoPE launches that use memref.collapse_shape inside the launch body.

Usage:
    python3 rms_gemms_rope_multi.py -p           # print combined MLIR
    python3 rms_gemms_rope_multi.py              # compile + run + validate
"""

import argparse
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from air.ir import *
from air.dialects.affine import apply as affine_apply
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp, CallOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

from air import api as air
from air.api import ops
from air.api.types import bf16, f32, i32


def _api_dtype(np_dtype):
    """The air.api dtype for a numpy dtype, for builders that take np dtypes.

    The llms/ builders are all called with ml_dtypes.bfloat16; the map is
    explicit rather than a getattr so an unsupported dtype names itself here
    instead of failing later inside the emitter.
    """
    for np_t, api_t in ((bfloat16, bf16), (np.float32, f32)):
        if np_dtype is np_t:
            return api_t
    raise TypeError(f"no air.api dtype for {np_dtype!r}")


from shared.infra.stitching import (
    _wrap_ir_in_launch,
    stitch_elf,
    KernelSlice,
    FuncArg,
    alloc_gemm_scratch,
)

range_ = for_


# ---------------------------------------------------------------------------
# 2D RoPE launch builder (accepts 2D in/out, collapses to 1D inside launch)
# ---------------------------------------------------------------------------


def _build_rope_2d(
    outer_rows, outer_cols, embed_dim, np_dtype, herd_x, rope_dim=None, target="npu2"
):
    """Build a RoPE launch with 2D in/out args (for GEMM type compatibility).

    The outer 2D shape (outer_rows, outer_cols) matches the GEMM output type;
    the RoPE herd walks the same memory as flat ``embed_dim``-wide rows.

    Func signature:
      (in_2d: [outer_rows, outer_cols], lut_1d: [total], out_2d: [outer_rows, outer_cols])

    Args:
        outer_rows: 2D func arg rows (e.g. seq_len=2048)
        outer_cols: 2D func arg cols (e.g. emb_dim=2048 or kv_dim=512)
        embed_dim:  RoPE column width per row (head_dim=64)
        herd_x:     Number of tiles for row-parallel
        rope_dim:   Rotated width; None/==embed_dim -> full rotary (calls `rope`).
                    Less than embed_dim -> PARTIAL rotary (Phi-4: 96 of 128), which
                    calls `rope_partial` and passes the tail through. The LUT row
                    stays embed_dim wide -- [cos|sin|unused] -- so every DMA shape
                    and row offset below is identical either way.
        target:     NPU generation to build for. These are prefill builders, so
                    npu2; it is a parameter because build() needs one and the
                    caller str()s the result straight into a stitched module.

    The predecessor spelled the row offset as a hand-built AffineMap over three
    symbols and the flattening as an explicit memref.collapse_shape at launch
    scope. Here the offset is Python arithmetic on the tile coordinates and the
    flattening is ``.reshape(total)`` -- a view, so it moves nothing and emits
    no op; the 2-D operand simply carries a rank-1 access pattern. Both spellings
    compile to a byte-identical air.insts.bin.
    """
    assert embed_dim % 16 == 0, "embed_dim must be divisible by 16"
    total = outer_rows * outer_cols
    assert total % embed_dim == 0
    rope_rows = total // embed_dim  # actual RoPE rows (n_heads * seq_len)
    herd_y = 1
    total_tiles = herd_x * herd_y
    assert rope_rows % total_tiles == 0
    rows_per_tile = rope_rows // total_tiles

    partial = rope_dim is not None and rope_dim != embed_dim
    if partial:
        assert 0 < rope_dim < embed_dim and rope_dim % 32 == 0, (
            f"rope_dim {rope_dim} must be a positive multiple of 32 below "
            f"embed_dim {embed_dim} (each half must vectorize by 16)"
        )

    dtype = _api_dtype(np_dtype)
    IN = air.tensor([outer_rows, outer_cols], dtype)
    LUT = air.tensor([total], dtype)
    OUT = air.tensor([outer_rows, outer_cols], dtype)

    rope = air.extern(
        "rope_partial" if partial else "rope",
        link_with="rope.o",
        scalars=[i32, i32] if partial else [i32],
    )

    with air.launch(name="rope_2d") as launch:

        @launch.body
        def _():
            with air.segment(name="rope_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(herd_x), range(herd_y)],
                        name="rope_herd",
                        shape=(herd_x, herd_y),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            l1_in = air.alloc([embed_dim], dtype, scope=h.private())
                            l1_lut = air.alloc([embed_dim], dtype, scope=h.private())
                            l1_out = air.alloc([embed_dim], dtype, scope=h.private())

                            in_flat = IN.reshape(total)
                            out_flat = OUT.reshape(total)

                            for local_row in air.sequential(0, rows_per_tile):
                                row = (
                                    local_row + (tx * herd_y + ty) * rows_per_tile
                                ) * embed_dim

                                ops.load(l1_in, in_flat[row : row + embed_dim])
                                ops.load(l1_lut, LUT[row : row + embed_dim])
                                if partial:
                                    rope(l1_in, l1_lut, l1_out, embed_dim, rope_dim)
                                else:
                                    rope(l1_in, l1_lut, l1_out, embed_dim)
                                ops.store(l1_out, out_flat[row : row + embed_dim])

    return launch.build(target=target)


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------


def build_rms_gemms_rope_module(
    seq_len=2048,
    emb_dim=2048,
    kv_dim=512,
    n_heads=32,
    n_kv_heads=8,
    head_dim=64,
    herd_m=8,
    herd_n=4,
    # RoPE config
    rope_herd_x=8,
    rope_dim=None,  # partial rotary width (Phi-4: 96); None = full head_dim
    print_kernels=False,
):
    """Build 6-launch module: RMSNorm + Q/K/V GEMMs + RoPE Q + RoPE K.

    Returns:
        Module with func @rms_gemms_rope and 13 memref args:
            %arg0:  x_in        (seq_len, emb_dim)       input
            %arg1:  norm_w      (emb_dim,)               RMSNorm weight
            %arg2:  normed      (seq_len, emb_dim)       RMSNorm output
            %arg3:  wq          (emb_dim, emb_dim)       Q weight
            %arg4:  q           (seq_len, emb_dim)       Q GEMM output (2D)
            %arg5:  wk          (emb_dim, kv_dim)        K weight
            %arg6:  k           (seq_len, kv_dim)        K GEMM output (2D)
            %arg7:  wv          (emb_dim, kv_dim)        V weight
            %arg8:  v           (seq_len, kv_dim)        V GEMM output
            %arg9:  lut_q       (q_total,)               RoPE Q LUT (1D)
            %arg10: lut_k       (k_total,)               RoPE K LUT (1D)
            %arg11: q_roped     (seq_len, emb_dim)       RoPE Q output (2D)
            %arg12: k_roped     (seq_len, kv_dim)        RoPE K output (2D)
    """
    from shared.builders.gemm_builder import _build_gemm_module, gemm_registry_config
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms

    # Per-GEMM config from the kernel_registry JSON (single source of truth): method
    # (fused-cast vs drain) AND all tiles are looked up per shape, never hardcoded.
    # Q (large) resolves to fused-cast (_m64), K/V (small) to drain (_m32); both
    # co-link in one ELF (distinct symbols + mm_*.o; each air.launch reconfigures
    # L1/L2 so launch buffers don't accumulate). Adapts automatically to other models.
    q_spec = gemm_registry_config(seq_len, emb_dim, emb_dim, "bf16", "high")
    k_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")
    v_spec = gemm_registry_config(seq_len, emb_dim, kv_dim, "bf16", "high")

    def _gemm_kw_and_tiles(spec):
        return (
            dict(spec["build_kwargs"]),
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
        )

    q_total = seq_len * emb_dim  # = n_heads * seq_len * head_dim
    k_total = seq_len * kv_dim  # = n_kv_heads * seq_len * head_dim

    # RoPE rows: the LUT has one row per (position, head) pair in seq-first order
    # Q: n_heads * seq_len rows of head_dim
    # K: n_kv_heads * seq_len rows of head_dim
    rope_q_rows = n_heads * seq_len  # 65536
    rope_k_rows = n_kv_heads * seq_len  # 16384

    # ---- Build sub-kernels ----

    # 1. RMSNorm (bare herd → wrap in launch+segment)
    print("  [1/6] RMSNorm...")
    rms_ir = _wrap_ir_in_launch(
        str(build_rms(seq_len, emb_dim, bfloat16, 16, herd_x=8))
    )

    # 2-4. Q/K/V GEMMs — method + ALL tiles come from the registry spec per shape.
    _q_kw, _q_tm, _q_k2, _q_k1, _q_tn = _gemm_kw_and_tiles(q_spec)
    _k_kw, _k_tm, _k_k2, _k_k1, _k_tn = _gemm_kw_and_tiles(k_spec)
    _v_kw, _v_tm, _v_k2, _v_k1, _v_tn = _gemm_kw_and_tiles(v_spec)
    _qm = q_spec["method"]
    _km = k_spec["method"]
    _vm = v_spec["method"]
    print(f"  [2/6] Q GEMM ({_qm})...")
    q_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            emb_dim,
            _q_tm,
            _q_k2,
            _q_k1,
            _q_tn,
            herd_m,
            herd_n,
            **_q_kw,
        )
    )
    print(f"  [3/6] K GEMM ({_km})...")
    k_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            _k_tm,
            _k_k2,
            _k_k1,
            _k_tn,
            herd_m,
            herd_n,
            **_k_kw,
        )
    )
    print(f"  [4/6] V GEMM ({_vm})...")
    v_ir = str(
        _build_gemm_module(
            seq_len,
            emb_dim,
            kv_dim,
            _v_tm,
            _v_k2,
            _v_k1,
            _v_tn,
            herd_m,
            herd_n,
            **_v_kw,
        )
    )

    # 5-6. RoPE Q/K (2D in/out with collapse_shape inside launch)
    # Outer 2D shape matches GEMM output type; inner processing uses head_dim
    print(
        f"  [5/6] RoPE Q (outer={seq_len}x{emb_dim}, embed_dim={head_dim}, herd_x={rope_herd_x})..."
    )
    _rope_sym = "@rope_partial" if (rope_dim and rope_dim != head_dim) else "@rope"
    rope_q_ir = str(
        _build_rope_2d(
            seq_len, emb_dim, head_dim, bfloat16, rope_herd_x, rope_dim=rope_dim
        )
    )

    print(
        f"  [6/6] RoPE K (outer={seq_len}x{kv_dim}, embed_dim={head_dim}, herd_x={rope_herd_x})..."
    )
    rope_k_ir = str(
        _build_rope_2d(
            seq_len, kv_dim, head_dim, bfloat16, rope_herd_x, rope_dim=rope_dim
        )
    )

    if print_kernels:
        for name, ir in [
            ("RMSNorm", rms_ir),
            ("Q GEMM", q_ir),
            ("K GEMM", k_ir),
            ("V GEMM", v_ir),
            ("RoPE Q", rope_q_ir),
            ("RoPE K", rope_k_ir),
        ]:
            print(f"\n{'='*60}")
            print(f"  Sub-kernel: {name} ({len(ir.splitlines())} lines)")
            print(f"{'='*60}")
            print(ir)

    # ---- Stitch (declarative via stitch_elf) ----
    # Base bf16 signature args 0..12. Each fused-cast GEMM appends one f32
    # C-scratch tail arg (13+), allocated registry-driven by alloc_gemm_scratch
    # in Q,K,V order; drain GEMMs get none. GQA -> 1 scratch (Q), MHA -> 3.
    #   RMSNorm:  {0->0, 1->1, 2->2}       (x_in, norm_w, normed)
    #   Q GEMM:   normed=2, wq=3, q=4   (+ q_f32 scratch if fused)
    #   K GEMM:   normed=2, wk=5, k=6   (+ k_f32 scratch if fused)
    #   V GEMM:   normed=2, wv=7, v=8   (+ v_f32 scratch if fused)
    #   RoPE Q:   {0->4, 1->9, 2->11}      (q[2D], lut_q[1D], q_roped[2D])
    #   RoPE K:   {0->6, 1->10, 2->12}     (k[2D], lut_k[1D], k_roped[2D])
    scratch_args, scratch_for = alloc_gemm_scratch(
        [
            (q_spec, seq_len, emb_dim),
            (k_spec, seq_len, kv_dim),
            (v_spec, seq_len, kv_dim),
        ],
        base_arg_count=13,
    )

    def _gemm_arg_map(in_idx, w_idx, out_idx, sc):
        if sc is not None:  # fused-cast: {0:in, 1:w, 2:Cf32-scratch, 3:bf16-out}
            return {0: in_idx, 1: w_idx, 2: sc, 3: out_idx}
        return {0: in_idx, 1: w_idx, 2: out_idx}  # drain: {0:in, 1:w, 2:bf16-out}

    # Per-GEMM externs: each fused/drain GEMM uses suffixed mm.o symbols
    # (_m64 / _m32) so both variants co-link in one ELF.
    def _gemm_externs(spec):
        sfx = spec["sym_suffix"]
        return {
            "@op_has_no_registered_library_name" + sfx,
            "@zero_f32_mn" + sfx,
            "@f32_to_bf16_mn" + sfx,
        }

    base_args = [
        FuncArg("%arg0", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg1", f"memref<{emb_dim}xbf16>"),
        FuncArg("%arg2", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg3", f"memref<{emb_dim}x{emb_dim}xbf16>"),
        FuncArg("%arg4", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg5", f"memref<{emb_dim}x{kv_dim}xbf16>"),
        FuncArg("%arg6", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg7", f"memref<{emb_dim}x{kv_dim}xbf16>"),
        FuncArg("%arg8", f"memref<{seq_len}x{kv_dim}xbf16>"),
        FuncArg("%arg9", f"memref<{q_total}xbf16>"),
        FuncArg("%arg10", f"memref<{k_total}xbf16>"),
        FuncArg("%arg11", f"memref<{seq_len}x{emb_dim}xbf16>"),
        FuncArg("%arg12", f"memref<{seq_len}x{kv_dim}xbf16>"),
    ]

    slices = [
        KernelSlice(
            rms_ir, "r", {0: 0, 1: 1, 2: 2}, extern_syms={"@zero_vectorized_bf16"}
        ),
        KernelSlice(
            q_ir,
            "q",
            _gemm_arg_map(2, 3, 4, scratch_for[0]),
            extern_syms={"@matmul_bf16"} | _gemm_externs(q_spec),
        ),
        KernelSlice(
            k_ir,
            "k",
            _gemm_arg_map(2, 5, 6, scratch_for[1]),
            extern_syms={"@matmul_bf16"} | _gemm_externs(k_spec),
        ),
        KernelSlice(
            v_ir,
            "v",
            _gemm_arg_map(2, 7, 8, scratch_for[2]),
            extern_syms={"@matmul_bf16"} | _gemm_externs(v_spec),
        ),
        # The rope leaf symbol must not be prefix-renamed; partial rotary calls a
        # different one (@rope_partial), so the pin has to follow rope_dim.
        KernelSlice(rope_q_ir, "rq", {0: 4, 1: 9, 2: 11}, extern_syms={_rope_sym}),
        KernelSlice(rope_k_ir, "rk", {0: 6, 1: 10, 2: 12}, extern_syms={_rope_sym}),
    ]

    module = stitch_elf(
        "rms_gemms_rope",
        base_args,
        slices,
        scratch_args=scratch_args,
    )
    print(f"  Module: {len(str(module).splitlines())} lines, parsed OK")
    return module


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def _rms_norm_ref(x, weight, eps=1e-5):
    """CPU RMSNorm reference."""
    x_f32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f32**2, axis=-1, keepdims=True) + eps)
    return (x_f32 / rms * weight.astype(np.float32)).astype(bfloat16)


def _rope_ref(x_2d, lut_2d):
    """CPU RoPE reference."""
    x = x_2d.astype(np.float32)
    lut = lut_2d.astype(np.float32)
    out = np.empty_like(x)
    out[:, 0::2] = x[:, 0::2] * lut[:, 0::2] - x[:, 1::2] * lut[:, 1::2]
    out[:, 1::2] = x[:, 0::2] * lut[:, 1::2] + x[:, 1::2] * lut[:, 0::2]
    return out.astype(bfloat16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEQ_LEN = 2048
    EMB_DIM = 2048
    KV_DIM = 512
    N_HEADS = 32
    N_KV_HEADS = 8
    HEAD_DIM = 64

    parser = argparse.ArgumentParser(
        description="RMSNorm + QKV GEMMs + RoPE QK multi-launch test"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
        help="Print combined MLIR and exit",
    )
    parser.add_argument("--print-kernels", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
    )
    args = parser.parse_args()

    print(
        f"RMS+QKV+RoPE Multi-Launch: seq={SEQ_LEN}, emb={EMB_DIM}, "
        f"kv={KV_DIM}, heads={N_HEADS}/{N_KV_HEADS}, dk={HEAD_DIM}"
    )

    module = build_rms_gemms_rope_module(
        seq_len=SEQ_LEN,
        emb_dim=EMB_DIM,
        kv_dim=KV_DIM,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        print_kernels=args.print_kernels,
    )

    if args.print_module_only:
        print(module)
        sys.exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rms_gemms_rope",
        )
        module_function = backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    # ---- compile-and-run: build test data, run, verify ----
    np.random.seed(42)

    # Inputs
    x_in = np.random.uniform(-1.0, 1.0, (SEQ_LEN, EMB_DIM)).astype(bfloat16)
    norm_w = np.random.uniform(0.5, 1.5, (EMB_DIM,)).astype(bfloat16)
    wq = np.random.uniform(-0.1, 0.1, (EMB_DIM, EMB_DIM)).astype(bfloat16)
    wk = np.random.uniform(-0.1, 0.1, (EMB_DIM, KV_DIM)).astype(bfloat16)
    wv = np.random.uniform(-0.1, 0.1, (EMB_DIM, KV_DIM)).astype(bfloat16)

    # RoPE LUTs (seq-first: repeated per head)
    from rope_lut.rope_lut import generate_lut

    base_lut = generate_lut(SEQ_LEN, HEAD_DIM, bfloat16)  # (SEQ_LEN, HEAD_DIM)
    lut_q = np.repeat(base_lut, N_HEADS, axis=0)  # (N_HEADS*SEQ_LEN, HEAD_DIM)
    lut_k = np.repeat(base_lut, N_KV_HEADS, axis=0)  # (N_KV_HEADS*SEQ_LEN, HEAD_DIM)

    # CPU reference
    print("Computing CPU reference...")
    normed_ref = _rms_norm_ref(x_in, norm_w)
    q_ref = (normed_ref.astype(np.float32) @ wq.astype(np.float32)).astype(bfloat16)
    k_ref = (normed_ref.astype(np.float32) @ wk.astype(np.float32)).astype(bfloat16)
    v_ref = (normed_ref.astype(np.float32) @ wv.astype(np.float32)).astype(bfloat16)

    # Apply RoPE to Q and K in seq-first layout
    q_2d = q_ref.reshape(SEQ_LEN, N_HEADS, HEAD_DIM)  # (seq, heads, dk)
    q_flat = q_2d.reshape(SEQ_LEN * N_HEADS, HEAD_DIM)  # seq-first order
    q_roped_ref = _rope_ref(q_flat, lut_q.reshape(-1, HEAD_DIM))
    q_roped_ref = q_roped_ref.reshape(SEQ_LEN, EMB_DIM)  # back to (seq, emb)

    k_2d = k_ref.reshape(SEQ_LEN, N_KV_HEADS, HEAD_DIM)
    k_flat = k_2d.reshape(SEQ_LEN * N_KV_HEADS, HEAD_DIM)
    k_roped_ref = _rope_ref(k_flat, lut_k.reshape(-1, HEAD_DIM))
    k_roped_ref = k_roped_ref.reshape(SEQ_LEN, KV_DIM)

    # Output buffers (zeroed)
    normed_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    q_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    k_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)
    v_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)
    q_roped_buf = np.zeros((SEQ_LEN, EMB_DIM), dtype=bfloat16)
    k_roped_buf = np.zeros((SEQ_LEN, KV_DIM), dtype=bfloat16)

    # Func signature: 13 args
    # (x_in, norm_w, normed, wq, q, wk, k, wv, v, lut_q, lut_k, q_roped, k_roped)
    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="rms_gemms_rope",
    )

    # XRTRunner convention: inputs = first N func args, expected_outputs = last M args.
    # Func has 13 args total. Last 2 (arg11=q_roped, arg12=k_roped) are outputs.
    # First 11 (arg0-arg10) are inputs (including zeroed intermediate buffers).
    exit(
        runner.run_test(
            module,
            inputs=[
                x_in,  # arg0
                norm_w,  # arg1
                normed_buf,  # arg2 (intermediate, zeroed)
                wq,  # arg3
                q_buf,  # arg4 (intermediate, zeroed)
                wk,  # arg5
                k_buf,  # arg6 (intermediate, zeroed)
                wv,  # arg7
                v_buf,  # arg8 (intermediate, zeroed)
                lut_q.flatten(),  # arg9
                lut_k.flatten(),  # arg10
            ],
            expected_outputs=[
                q_roped_ref,  # arg11
                k_roped_ref,  # arg12
            ],
            rtol=0.2,
            atol=0.5,
            min_correlation=0.99,
        )
    )
