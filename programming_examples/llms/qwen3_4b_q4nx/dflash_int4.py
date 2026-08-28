# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# int4-AWQ quantization + GEMM launches for the DFlash drafter's two non-decode
# projections: `fc` (tap fusion) and the per-layer context k/v.
#
# WHY int4 AND NOT bf16. Both projections first ran as bf16 GEMMs and both gate
# clean (docs/DFlashFeasibility.md section 3.3), but bf16 costs 65 MB for fc and
# 52 MB for the context k/v against a draft pass whose 5 Q4 decode layers are
# ~315 MB -- a 37% surcharge, and the k/v half is worse than it looks because
# those are the SAME tensors the decode streams in Q4 within the same call.
#
# WHY int4-AWQ AND NOT THE SUPERKERNEL'S OWN Q4NX. fc cannot be a fused_decode
# phase: `FULL4` is `NPH == 4 and ...` and `GLU_PHASE = 2 if NPH == 4 else -1`,
# so a fifth phase disables both the residual path and the GLU path. Making the
# phase geometry per-wave instead of per-build is a change to the most
# load-bearing table in that builder. The int4-AWQ GEMM
# (matrix_multiplication/int4_awq/matmul_int4_packed.py) is an existing,
# exercised alternative that needs none of that.
#
# WHY NOT THE int4 GEMV. fc applies one weight to every context row; a GEMV
# re-streams it per row, so at ctx=8 int4 GEMV moves 8 x 18 MB = 144 MB and
# loses to a single bf16 GEMM pass. The GEMM form streams once.
#
# QUANTIZATION CONVENTION is AWQ's, which is NOT the q4k-cascade convention
# qwen3_4b_q4nx_requant.py uses for the decode weights:
#     q4k (decode):  w = q*scale + min,   per 32 columns
#     AWQ  (here):   w = (q - z)*scale,   per `gs` columns, z an integer
# Mixing them silently produces plausible-looking garbage, so the two live in
# different files and nothing is shared between them.

import numpy as np
from ml_dtypes import bfloat16

GS = 128  # AWQ group size along K
TILE_M = 16
TILE_N = 16
TILE_K_L1 = 128
HERD_N = 4
# herd_m=1, not the usual 8: the L2 A stage is herd_m * tile_m * K * 2 bytes,
# which at K=12800 is 400 KB for herd_m=1 and overflows the 512 KB memtile at 2.
HERD_M = 1

# What the int4 GEMM calls into mv_int4_bf16.o by name. stitch_elf prefixes
# every symbol a slice defines, so these must be declared extern or the
# prefixed call has no callee -- read off the emitted IR rather than guessed,
# because the bf16 GEMM's set (zero_f32_mn / f32_to_bf16_mn with a _m32 suffix)
# is nearly but not quite the same.
EXTERN_SYMS = {
    "@matmul_int4_bf16_packed_f32",
    "@zero_vectorized_f32_mn",
    "@f32_to_bf16_mn",
}


def awq_quantize(W, gs=GS):
    """[N, K] float -> (W_q [N, K/2] u8, W_s [K/gs, N] bf16, W_z [K/gs, N] u8).

    AWQ's convention: w ~= (q - z) * s, with q and z 4-bit unsigned and one
    (s, z) pair per `gs` consecutive INPUT columns. `pack_inputs` wants the
    scales and zeros group-major, hence the transpose at the end.
    """
    N, K = W.shape
    assert K % gs == 0, (K, gs)
    G = K // gs
    Wg = np.asarray(W, np.float32).reshape(N, G, gs)
    mn = Wg.min(2)
    mx = Wg.max(2)
    s = (mx - mn) / 15.0
    s = np.where(s <= 0, 1.0, s).astype(np.float32)
    z = np.clip(np.rint(-mn / s), 0, 15).astype(np.uint8)
    q = np.clip(np.rint(Wg / s[..., None]) + z[..., None].astype(np.float32), 0, 15)
    q = q.astype(np.uint8).reshape(N, K)
    # Two nibbles per byte along K, low nibble first -- the order the kernel's
    # dequant unpacks.
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, np.asarray(s.T, bfloat16), np.ascontiguousarray(z.T)


def awq_dequantize(W_q, W_s, W_z, gs=GS):
    """Inverse of awq_quantize, for the reference. [N, K] float32."""
    N, Kh = W_q.shape
    K = Kh * 2
    q = np.zeros((N, K), np.uint8)
    q[:, 0::2] = W_q & 0x0F
    q[:, 1::2] = W_q >> 4
    s = np.asarray(W_s, np.float32).T.reshape(N, K // gs, 1)
    z = np.asarray(W_z, np.float32).T.reshape(N, K // gs, 1)
    return ((q.reshape(N, K // gs, gs).astype(np.float32) - z) * s).reshape(N, K)


# tile_k_l2 MUST EQUAL K. `matmul_int4_packed.build_module` puts the herd inside
# the K-outer loop and the herd body zeroes its L1 accumulator and drains to L2
# on every iteration, so with more than one K-outer step the L2 C tile ends up
# holding the LAST chunk's partial product instead of the sum. Measured on
# device (M=64, N=128, herd 2x4): tile_k_l2 == K gives 7.2-7.4e-03 at K=128, 256
# and 1280; K=256 with tile_k_l2=128 gives 1.12 and K=1280 with tile_k_l2=128
# gives 1.03. Every lit test in matrix_multiplication/int4_awq uses
# K=128/tile_k_l2=128, so the K-outer path is never exercised there.
#
# The practical consequence is that K is bounded by L2: at herd_m=1, tile_m=16,
# tile_n=16, herd_n=4 the stage is K*32 bytes of A plus K*33.5 bytes of B, so
# K=6400 (406 KB) fits a 512 KB memtile and K=12800 does not. fc splits its K
# accordingly -- see dflash_int4_fc_builder.py.
TILE_K_L2 = None  # no default: every caller must pass tile_k_l2 == its own K


def pack_for_device(W_q, W_s, W_z, M, K, N):
    """[N/tile_n, K/tile_k_l1, tile_bytes] u8 -- the B operand's L3 layout.

    The packing granularity is (tile_n, tile_k_l1), NOT (tile_n*herd_n,
    tile_k_l2): build_module derives N_div/K_div from the L1 tiles, and
    `packed_tile_bytes` is called with them.
    """
    from matmul_int4_packed import pack_inputs

    return pack_inputs(W_q, W_s, W_z, M, K, N, GS, TILE_N, TILE_K_L1)


def build_int4_gemm_ir(m, k, n, tile_k_l2=None, herd_m=HERD_M):
    """One int4-AWQ GEMM launch, as MLIR text. `tile_k_l2` must equal `k`."""
    from gemm_builder import _build_int4_gemm_module

    tile_k_l2 = k if tile_k_l2 is None else tile_k_l2
    assert tile_k_l2 == k, (
        f"tile_k_l2 ({tile_k_l2}) != k ({k}): the K-outer path drops every "
        f"chunk but the last. Split K across launches instead."
    )
    return str(
        _build_int4_gemm_module(
            m, k, n, GS, TILE_M, tile_k_l2, TILE_K_L1, TILE_N, herd_m, HERD_N
        )
    )


def packed_shape(k, n):
    from matmul_int4_packed import packed_tile_bytes

    return (n // TILE_N, k // TILE_K_L1, packed_tile_bytes(TILE_N, TILE_K_L1, GS)[3])


def compile_int4_gemm_kernel(tile_m=TILE_M, tile_n=TILE_N, tile_k_l1=TILE_K_L1, gs=GS):
    """Build mv_int4_bf16.o for the GEMM, NOT the GEMV.

    `shared.infra.external_kernels.compile_mv_int4_bf16` builds the GEMV
    variant: it passes -DDIM_K and no -DDIM_N at all, where the GEMM's own
    Makefile (matrix_multiplication/int4_awq) passes -DDIM_N and -DDIM_K_CHUNK.
    Linking the GEMV object into the GEMM compiles, loads and runs -- and
    returns NaNs and uncorrelated rows, because the kernel's tile constants
    disagree with the IR's. Mirrors that Makefile line for line.
    """
    from pathlib import Path

    from shared.infra.external_kernels import _compile_kernel

    src = (
        Path(__file__).resolve().parent.parent.parent
        / "matrix_vector_multiplication"
        / "int4_awq"
        / "mv_int4_bf16.cc"
    )
    _compile_kernel(
        src,
        "mv_int4_bf16.o",
        extra_flags=[
            f"-DDIM_M={tile_m}",
            f"-DDIM_N={tile_n}",
            f"-DDIM_K_CHUNK={tile_k_l1}",
            f"-DDIM_GS={gs}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
        force=True,
    )


def paths():
    """int4 GEMM builder + packer onto sys.path."""
    import sys
    from pathlib import Path

    pe = Path(__file__).resolve().parent.parent.parent
    for p in (
        pe / "llms" / "llama32_1b_int4",
        pe / "matrix_multiplication" / "int4_awq",
        pe / "llms",
        pe,
    ):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return pe


def self_check():
    """Round-trip the real fc through AWQ and report the error it introduces."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen3_4b_draft_weights import DraftWeights

    W = DraftWeights().fc()
    q, s, z = awq_quantize(W)
    back = awq_dequantize(q, s, z)
    rel = np.abs(back - W).max() / max(np.abs(W).max(), 1e-9)
    step = float(np.asarray(s, np.float32).max()) / max(np.abs(W).max(), 1e-9)
    print(f"[int4] fc {W.shape} -> q {q.shape} u8, s {s.shape} bf16, z {z.shape} u8")
    print(
        f"[int4] AWQ round-trip max rel {rel:.4e}, one step {step:.4e}"
        f"   {'OK' if rel <= 2 * step else 'TOO LARGE'}"
    )
    print(
        f"[int4] fc bytes: bf16 {W.size*2/1e6:.1f} MB -> int4 "
        f"{(q.size + s.size*2 + z.size)/1e6:.1f} MB"
    )
    return 0 if rel <= 2 * step else 1


if __name__ == "__main__":
    import sys

    sys.exit(self_check())
