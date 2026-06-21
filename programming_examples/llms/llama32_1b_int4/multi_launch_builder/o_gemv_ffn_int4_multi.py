# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""o_gemv_ffn_int4 — full-int4 ELF2 for the LLAMA decode block.

3-launch ELF derived from the bf16 baseline o_gemv_ffn_multi.py. ALL three
weight matrices (wo, gate+up, wdown) are int4-AWQ packed BOs:

  Stage 1 (matvec_int4_packed_add):  res1 = dequant(wo) @ attn_out + x_residual
                                     into arg6[0]
  Stage 2 (matvec_int4_swiglu_rms):  swiglu = silu(dequant(gateup) @ rms_norm(arg6)) * up
  Stage 3 (matvec_int4_packed_add):  output = dequant(wdown) @ swiglu + arg6[0]

ABI (15 args; arg0/arg7/arg12 are packed int4 BOs):

    arg0:  memref<tq x tile_bytes xi8>          wo_packed         STATIC
    arg1:  memref<emb xbf16>                    attn_out          INPUT
    arg2:  memref<emb xbf16>                    (dead)
    arg3:  memref<emb xbf16>                    x_residual        INPUT
    arg4:  memref<emb xbf16>                    (dead)
    arg5:  memref<emb xbf16>                    (dead)
    arg6:  memref<2 x emb xbf16>                packed RMS input  STATIC
                                                  (row 0 = res1, row 1 = ffn_norm_w)
    arg7:  memref<tg x tile_bytes xi8>          gate/up_packed    STATIC
                                                  (interleaved gate/up packed)
    arg8:  memref<hidden xbf16>                 (dead)
    arg9:  memref<hidden x emb xbf16>           (dead)
    arg10: memref<hidden xbf16>                 (dead)
    arg11: memref<hidden xbf16>                 swiglu            INTERMEDIATE
    arg12: memref<td x tile_bytes xi8>          wdown_packed      STATIC
    arg13: memref<emb xbf16>                    (dead)
    arg14: memref<emb xbf16>                    output            OUTPUT

K_CHUNK for all three int4 stages is fixed at 2048 → all three stages
link the same mv_int4_bf16.o (DIM_K=2048, DIM_M=8). Stage 3 (K=hidden=8192)
splits into K_div=4 inner iterations; stages 1 and 2 do a single K chunk.
"""

import argparse
import os
import re
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "matrix_vector_multiplication",
        "int4_awq",
    ),
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "decode_ffn_swiglu",
    ),
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matvec_int4_packed_add
from matvec_int4_packed_add import build_module as build_int4_gemv_add
import matvec_int4_swiglu_rms
from matvec_int4_swiglu_rms import build_module as build_int4_swiglu_rms
from matvec_int4_packed import pack_inputs

from shared.infra.stitching import (
    _extract_between_func_and_return,
    _extract_affine_maps,
    _extract_private_funcs,
    _extract_channel_decls,
    _rename_all_with_externs,
    _fix_launch_func_args,
)
from air.ir import Module, Context
from air.backend.xrt import XRTBackend

# All three stages share mv_int4_bf16.o — keep these symbols un-prefixed
# across the stitched launches so all three calls resolve to the same .o
# entry points.
_EXTERNS = {
    "@matvec_int4_bf16_packed",
    "@zero_vectorized_bf16",
    "@partial_plus_r_bf16",
}


def _packed_dims(M, K, GS, M_TILE, K_CHUNK, N_CORES, M_PER_LAUNCH):
    n_gpc = K_CHUNK // GS
    tile_bytes = M_TILE * (K_CHUNK // 2) + n_gpc * M_TILE * 2 + n_gpc * M_TILE
    M_per_core_per_launch = M_PER_LAUNCH // N_CORES
    M_div = M_per_core_per_launch // M_TILE
    K_div = K // K_CHUNK
    N_LAUNCHES = M // M_PER_LAUNCH
    total_tiles = N_LAUNCHES * N_CORES * M_div * K_div
    return total_tiles, tile_bytes


def build_o_gemv_ffn_int4_module(
    emb_dim=2048,
    hidden_dim=8192,
    gs=128,
    m_tile=8,
    k_chunk=2048,
    n_cores=8,
):
    """Build full-int4 3-launch ELF2."""
    assert emb_dim % k_chunk == 0 and hidden_dim % k_chunk == 0
    # Stage 2's int4 module requires K == K_CHUNK; emb_dim must equal k_chunk.
    assert (
        emb_dim == k_chunk
    ), f"Stage 2 int4 swiglu_rms requires emb_dim ({emb_dim}) == k_chunk ({k_chunk})"

    matvec_int4_packed_add.KERNEL_OBJ_NAME = "mv_int4_bf16.o"
    matvec_int4_swiglu_rms.KERNEL_OBJ_NAME = "mv_int4_bf16.o"

    # Stage 1: O proj (M=emb, K=emb), int4 GEMV+R.
    stage1 = build_int4_gemv_add(
        emb_dim,
        emb_dim,
        GS=gs,
        M_TILE=m_tile,
        K_CHUNK=k_chunk,
        N_CORES=n_cores,
    )

    # Stage 2: FFN gate+up + RMS + SwiGLU, int4 (M=2*hidden, K=emb).
    stage2 = build_int4_swiglu_rms(
        2 * hidden_dim,
        emb_dim,
        GS=gs,
        M_TILE=m_tile,
        K_CHUNK=k_chunk,
        N_CORES=n_cores,
    )

    # Stage 3: Down proj (M=emb, K=hidden), int4 GEMV+R.
    stage3 = build_int4_gemv_add(
        emb_dim,
        hidden_dim,
        GS=gs,
        M_TILE=m_tile,
        K_CHUNK=k_chunk,
        N_CORES=n_cores,
    )

    # Packed BO dims for the three int4 args.
    tq, tile_bytes_o = _packed_dims(
        emb_dim, emb_dim, gs, m_tile, k_chunk, n_cores, emb_dim
    )
    tg, tile_bytes_g = _packed_dims(
        2 * hidden_dim, emb_dim, gs, m_tile, k_chunk, n_cores, 2 * hidden_dim
    )
    td, tile_bytes_d = _packed_dims(
        emb_dim, hidden_dim, gs, m_tile, k_chunk, n_cores, emb_dim
    )
    assert (
        tile_bytes_o == tile_bytes_d == tile_bytes_g
    ), "All three int4 stages must share tile_bytes (same K_CHUNK/GS/M_TILE)"
    tile_bytes = tile_bytes_o

    def _slice(ir, prefix, arg_map, arg_aliases=None):
        body = _extract_between_func_and_return(ir)
        maps = _extract_affine_maps(ir)
        chans = _extract_channel_decls(ir)
        privs = _extract_private_funcs(ir)
        body = _rename_all_with_externs(body, prefix, _EXTERNS)
        maps = [_rename_all_with_externs(m, prefix, _EXTERNS) for m in maps]
        chans = [_rename_all_with_externs(c, prefix, _EXTERNS) for c in chans]
        privs = [_rename_all_with_externs(p, prefix, _EXTERNS) for p in privs]
        body = _fix_launch_func_args(
            body,
            prefix,
            arg_map=arg_map,
            arg_aliases=arg_aliases,
        )
        return body, maps, chans, privs

    # Stage 1 — matvec_int4_packed_add local (PACKED=0, B=1, R=2, D=3):
    #   wo_packed (arg0) @ attn_out (arg1) + x_residual (arg3) → arg6[0]
    s1_body, s1_maps, s1_chans, s1_privs = _slice(
        str(stage1),
        "s1",
        arg_map={0: 0, 1: 1, 2: 3},
        arg_aliases={3: "%arg6_row0"},
    )
    # Stage 2 — matvec_int4_swiglu_rms local (PACKED=0, RMS=1, D=2):
    #   gateup_packed (arg7), packed_rms native 2D (arg6), swiglu (arg11)
    s2_body, s2_maps, s2_chans, s2_privs = _slice(
        str(stage2),
        "s2",
        arg_map={0: 7, 1: 6, 2: 11},
    )
    # Stage 3 — matvec_int4_packed_add local (PACKED=0, B=1, R=2, D=3):
    #   wdown_packed (arg12) @ swiglu (arg11) + arg6[0] → output (arg14)
    s3_body, s3_maps, s3_chans, s3_privs = _slice(
        str(stage3),
        "s3",
        arg_map={0: 12, 1: 11, 3: 14},
        arg_aliases={2: "%arg6_row0"},
    )

    # Dedup private decls by symbol identity (all three stages declare the
    # same int4 externs; keep one copy).
    seen = set()
    all_privs = []
    for p in s1_privs + s2_privs + s3_privs:
        m = re.search(r"@(\w+)", p)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            all_privs.append(p.strip())

    seen_chans = set()
    all_chans = []
    for c in s1_chans + s2_chans + s3_chans:
        cs = c.strip()
        if cs not in seen_chans:
            seen_chans.add(cs)
            all_chans.append(cs)

    all_maps = s1_maps + s2_maps + s3_maps

    maps_block = "\n".join(all_maps)
    chans_block = "\n".join("  " + c for c in all_chans)
    privs_block = "\n".join("  " + p for p in all_privs)
    combined = f"""{maps_block}
module {{
{chans_block}
{privs_block}
  func.func @o_gemv_ffn_int4(
    %arg0: memref<{tq}x{tile_bytes}xi8>,
    %arg1: memref<{emb_dim}xbf16>,
    %arg2: memref<{emb_dim}xbf16>,
    %arg3: memref<{emb_dim}xbf16>,
    %arg4: memref<{emb_dim}xbf16>,
    %arg5: memref<{emb_dim}xbf16>,
    %arg6: memref<2x{emb_dim}xbf16>,
    %arg7: memref<{tg}x{tile_bytes}xi8>,
    %arg8: memref<{hidden_dim}xbf16>,
    %arg9: memref<{hidden_dim}x{emb_dim}xbf16>,
    %arg10: memref<{hidden_dim}xbf16>,
    %arg11: memref<{hidden_dim}xbf16>,
    %arg12: memref<{td}x{tile_bytes}xi8>,
    %arg13: memref<{emb_dim}xbf16>,
    %arg14: memref<{emb_dim}xbf16>
  ) {{
    %arg6_row0_strided = memref.subview %arg6[0, 0] [1, {emb_dim}] [1, 1]
        : memref<2x{emb_dim}xbf16> to memref<{emb_dim}xbf16, strided<[1]>>
    %arg6_row0 = memref.cast %arg6_row0_strided
        : memref<{emb_dim}xbf16, strided<[1]>> to memref<{emb_dim}xbf16>
{s1_body}
{s2_body}
{s3_body}
    return
  }}
}}
"""
    with Context() as ctx:
        return Module.parse(combined, ctx)


def o_gemv_ffn_int4_reference(
    wo, attn_out, x_residual, ffn_norm_w, wgate, wup, wdown, eps=1e-5
):
    """CPU bf16 reference (caller dequantizes int4 weights upstream)."""
    res1 = wo.astype(np.float32) @ attn_out.astype(np.float32) + x_residual.astype(
        np.float32
    )
    rstd = 1.0 / np.sqrt((res1 * res1).mean() + eps)
    normed = (res1 * rstd) * ffn_norm_w.astype(np.float32)
    normed_bf16 = normed.astype(bfloat16).astype(np.float32)
    gate = wgate.astype(np.float32) @ normed_bf16
    up = wup.astype(np.float32) @ normed_bf16
    swiglu = (gate * 0.5 * (np.tanh(gate / 2.0) + 1.0)) * up
    swiglu_bf16 = swiglu.astype(bfloat16).astype(np.float32)
    output = (wdown.astype(np.float32) @ swiglu_bf16 + res1).astype(bfloat16)
    return output


def _gen_int4_weights(M, K, gs, seed):
    """uint4 packed weights + scales + zeros + dequantized bf16 (for CPU ref)."""
    rng = np.random.default_rng(seed)
    A_q_unp = rng.integers(0, 16, size=(M, K), dtype=np.uint8)
    A_q = (A_q_unp[:, 0::2] | (A_q_unp[:, 1::2] << 4)).astype(np.uint8)
    n_groups = K // gs
    A_s = rng.uniform(0.005, 0.02, size=(n_groups, M)).astype(bfloat16)
    A_z = rng.integers(7, 9, size=(n_groups, M), dtype=np.uint8)
    # Vectorized dequant.
    A_q_i = A_q.astype(np.int32)
    low = A_q_i & 0x0F
    high = (A_q_i >> 4) & 0x0F
    nibs = np.empty((M, K), dtype=np.int32)
    nibs[:, 0::2] = low
    nibs[:, 1::2] = high
    s_per_kk = np.repeat(A_s.astype(np.float32), gs, axis=0)
    z_per_kk = np.repeat(A_z.astype(np.int32), gs, axis=0)
    dequant_f32 = (nibs - z_per_kk.T) * s_per_kk.T
    dequant_bf16 = dequant_f32.astype(bfloat16)
    return A_q, A_s, A_z, dequant_bf16


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="o_gemv_ffn_int4_multi.py",
        description="Full int4-AWQ ELF2 (stages 1+2+3 all int4).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--emb-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--gs", type=int, default=128)
    parser.add_argument("--m-tile", type=int, default=8, dest="m_tile")
    parser.add_argument("--k-chunk", type=int, default=2048, dest="k_chunk")
    parser.add_argument("--n-cores", type=int, default=8, dest="n_cores")
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format", type=str, choices=["xclbin", "elf"], default="elf"
    )
    args = parser.parse_args()

    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    print(
        f"O GEMV + FFN full-int4 3-launch: "
        f"emb_dim={emb_dim}, hidden_dim={hidden_dim}, k_chunk={args.k_chunk}"
    )

    module = build_o_gemv_ffn_int4_module(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        gs=args.gs,
        m_tile=args.m_tile,
        k_chunk=args.k_chunk,
        n_cores=args.n_cores,
    )
    if args.print_module_only:
        print(module)
        sys.exit(0)

    if args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="o_gemv_ffn_int4",
            use_lock_race_condition_fix=False,
            stack_size=4096,
        )
        backend.compile(module)
        backend.unload()
        print("Compile-only done.")
        sys.exit(0)

    np.random.seed(42)
    print("Generating int4 wo + dequantizing for CPU ref...")
    wo_q, wo_s, wo_z, wo_bf16 = _gen_int4_weights(emb_dim, emb_dim, args.gs, 100)
    # Stage 2 packed gate/up: rows 2i = gate[i], rows 2i+1 = up[i].
    # Generate as ONE interleaved [2*hidden, emb] matrix, then split into
    # gate/up bf16 references for the CPU ref.
    print("Generating int4 gate/up (interleaved) + dequantizing for CPU ref...")
    gu_q, gu_s, gu_z, gu_bf16 = _gen_int4_weights(2 * hidden_dim, emb_dim, args.gs, 150)
    gate_bf16 = gu_bf16[0::2]
    up_bf16 = gu_bf16[1::2]
    print("Generating int4 wdown + dequantizing for CPU ref...")
    wd_q, wd_s, wd_z, wd_bf16 = _gen_int4_weights(emb_dim, hidden_dim, args.gs, 200)

    attn_out = np.random.randn(emb_dim).astype(bfloat16)
    x_residual = np.random.randn(emb_dim).astype(bfloat16)
    ffn_norm_w = (np.random.randn(emb_dim) * 0.1 + 1.0).astype(bfloat16)
    packed_rms = np.empty((2, emb_dim), dtype=bfloat16)
    packed_rms[0] = 0.0
    packed_rms[1] = ffn_norm_w
    swiglu_buf = np.zeros(hidden_dim, dtype=bfloat16)

    wo_packed = pack_inputs(
        wo_q,
        wo_s,
        wo_z,
        emb_dim,
        emb_dim,
        args.gs,
        args.m_tile,
        args.k_chunk,
        args.n_cores,
        emb_dim,
    )
    gu_packed = pack_inputs(
        gu_q,
        gu_s,
        gu_z,
        2 * hidden_dim,
        emb_dim,
        args.gs,
        args.m_tile,
        args.k_chunk,
        args.n_cores,
        2 * hidden_dim,
    )
    wd_packed = pack_inputs(
        wd_q,
        wd_s,
        wd_z,
        emb_dim,
        hidden_dim,
        args.gs,
        args.m_tile,
        args.k_chunk,
        args.n_cores,
        emb_dim,
    )

    expected = o_gemv_ffn_int4_reference(
        wo_bf16,
        attn_out,
        x_residual,
        ffn_norm_w,
        gate_bf16,
        up_bf16,
        wd_bf16,
    )

    z_emb = np.zeros(emb_dim, dtype=bfloat16)
    z_hidden = np.zeros(hidden_dim, dtype=bfloat16)
    z_hidden_emb = np.zeros((hidden_dim, emb_dim), dtype=bfloat16)

    from air.backend.xrt_runner import XRTRunner

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="o_gemv_ffn_int4",
        use_lock_race_condition_fix=False,
        stack_size=4096,
    )
    sys.exit(
        runner.run_test(
            module,
            inputs=[
                wo_packed,  # arg0
                attn_out,  # arg1
                z_emb,  # arg2 dead
                x_residual,  # arg3
                z_emb,  # arg4 dead
                z_emb,  # arg5 dead
                packed_rms,  # arg6
                gu_packed,  # arg7 (int4 interleaved gate/up)
                z_hidden,  # arg8 dead
                z_hidden_emb,  # arg9 dead
                z_hidden,  # arg10 dead
                swiglu_buf,  # arg11
                wd_packed,  # arg12
                z_emb,  # arg13 dead
            ],
            expected_outputs=[expected],
            rtol=0.2,
            atol=2.0,
            min_correlation=0.99,
        )
    )
