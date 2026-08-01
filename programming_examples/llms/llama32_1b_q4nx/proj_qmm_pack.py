# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# numpy Q4NX block packer + dequant reference for the proj_qmm kernel. Byte
# layout of one quantized block (npu_quantize_block):
#   bf16  scales[8 groups][32 rows]   (offset 0,    512 B)
#   bf16  mins[8 groups][32 rows]     (offset 512,  512 B)
#   uint8 qs[2 rowgrp][256 col][8]    (offset 1024, 4096 B)   pair = (w1<<4)|w0
# One block = 32 rows x 256 cols = 5120 B = 2560 bf16.
# Dequant convention matches the proj_qmm kernel (_qmm_q4k_bf16): w = q*scale + min.
import numpy as np
from ml_dtypes import bfloat16

ROW_BLOCK = 32
COL_BLOCK = 256
GROUP = 32
PARALLEL = 16
N_GROUPS = COL_BLOCK // GROUP  # 8
ROW_GROUPS = ROW_BLOCK // PARALLEL  # 2
BLOCK_BF16 = (ROW_BLOCK * COL_BLOCK) // 2 // 2 + 2 * (N_GROUPS * ROW_BLOCK)  # 2560


def pack_q4k_block(q, scale, mn):
    """Pack one 32x256 block.
    q:     uint8 [ROW_BLOCK, COL_BLOCK] nibbles 0..15
    scale: float [ROW_BLOCK, N_GROUPS]   (per row, per 32-col group)
    mn:    float [ROW_BLOCK, N_GROUPS]
    returns: int16 view of the 5120-byte block (2560 bf16), ready for a BO.
    """
    assert q.shape == (ROW_BLOCK, COL_BLOCK)
    assert scale.shape == (ROW_BLOCK, N_GROUPS)
    buf = bytearray(ROW_BLOCK * COL_BLOCK // 2 + 4 * N_GROUPS * ROW_BLOCK)
    # scales[group][row], mins[group][row] as bf16
    sc = np.zeros((N_GROUPS, ROW_BLOCK), dtype=bfloat16)
    mi = np.zeros((N_GROUPS, ROW_BLOCK), dtype=bfloat16)
    for g in range(N_GROUPS):
        for r in range(ROW_BLOCK):
            sc[g, r] = bfloat16(scale[r, g])
            mi[g, r] = bfloat16(mn[r, g])
    off = 0
    buf[off : off + 2 * N_GROUPS * ROW_BLOCK] = sc.tobytes()
    off += 2 * N_GROUPS * ROW_BLOCK
    buf[off : off + 2 * N_GROUPS * ROW_BLOCK] = mi.tobytes()
    off += 2 * N_GROUPS * ROW_BLOCK
    # qs[rowgrp(2)][col(256)][pair(8)] = (w1<<4)|w0 ; w0 even row, w1 odd row
    qs = np.zeros((ROW_GROUPS, COL_BLOCK, PARALLEL // 2), dtype=np.uint8)
    for g in range(ROW_GROUPS):
        for h in range(COL_BLOCK):
            for kk in range(PARALLEL // 2):
                w0 = q[g * PARALLEL + kk * 2, h] & 0xF
                w1 = q[g * PARALLEL + kk * 2 + 1, h] & 0xF
                qs[g, h, kk] = (w1 << 4) | w0
    buf[off : off + qs.size] = qs.tobytes()
    arr = np.frombuffer(bytes(buf), dtype=np.int16).copy()
    assert arr.size == BLOCK_BF16, (arr.size, BLOCK_BF16)
    return arr


def dequant_block(q, scale, mn):
    """w[r,c] = q[r,c]*scale[r,c//32] + mn[r,c//32]  (kernel convention)."""
    w = np.zeros((ROW_BLOCK, COL_BLOCK), dtype=np.float32)
    for r in range(ROW_BLOCK):
        for c in range(COL_BLOCK):
            g = c // GROUP
            w[r, c] = float(q[r, c]) * float(scale[r, g]) + float(mn[r, g])
    return w


def ref_gemv_block(q, scale, mn, x):
    """y[r] = sum_c (q*scale+min)[r,c] * x[c], bf16-rounded inputs."""
    w = dequant_block(q, scale, mn)
    xf = x.astype(np.float32)
    return (w @ xf).astype(np.float32)


def pack_q4k_matrix(q, scale, mn):
    """Pack an [M, K] q4 weight matrix into (M/32) x (K/256) q4k blocks,
    concatenated in (row-block i, col-block j) order. Returns int16 [n_blocks*2560].
    q: uint8 [M, K]; scale/mn: float [M, K/32].
    """
    M, K = q.shape
    assert M % ROW_BLOCK == 0 and K % COL_BLOCK == 0
    nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
    blocks = []
    for i in range(nbi):
        for j in range(nbj):
            rs, cs = i * ROW_BLOCK, j * COL_BLOCK
            gs = j * (COL_BLOCK // GROUP)
            blocks.append(
                pack_q4k_block(
                    q[rs : rs + ROW_BLOCK, cs : cs + COL_BLOCK],
                    scale[rs : rs + ROW_BLOCK, gs : gs + N_GROUPS],
                    mn[rs : rs + ROW_BLOCK, gs : gs + N_GROUPS],
                )
            )
    return np.concatenate(blocks)


def pack_q4k_cascade(q, scale, mn, NCX, NCY, core_major=False, iter_major=False):
    """Pack an [M, K] q4 weight matrix in the memtile-cascade STREAM order
    [cx][i][j][cy], where each element is one 32x256 q4k block (BLOCK_BF16).

    Mirrors the merge's pack_inputs_v2 (cx,outer,kc,cy) ordering so a per-col
    memtile can ChannelGet NCY consecutive blocks per (cx,i,j) step and fan one
    to each cy core. Output assemble emits in (i, cy) order per column, so the
    block for (cx,i,j,cy) is matrix row-block gi = cx*(NCY*nbi_pc) + i*NCY + cy
    (col cx owns a CONTIGUOUS row span; within it, row-iter i then cy). This
    makes the per-col assemble write one contiguous, aligned shim S2MM BD whose
    natural order maps to identity DDR rows.

    core_major=True (for the id-demux one-packet-per-core path): each CORE (flat
    index c = cx*NCY + cy) owns a CONTIGUOUS natural row span
    [c*nbi_pc : (c+1)*nbi_pc], i.e. gi = (cx*NCY+cy)*nbi_pc + i. With a
    contiguous per-core gather (core c's whole packet placed at c*m_pc), the
    concatenated main buffer is still natural QKV[M] -- so the rope is unchanged
    -- while each core emits ONE contiguous packet (one header), a core-major
    layout where every Y BD is a plain linear transfer.

    Returns int16 [NCX*nbi_pc*nbj*NCY * BLOCK_BF16] = [nbi*nbj * BLOCK_BF16].
    """
    M, K = q.shape
    assert M % ROW_BLOCK == 0 and K % COL_BLOCK == 0
    nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
    n_cores = NCX * NCY
    assert nbi % n_cores == 0, "row-blocks must split evenly across cores"
    nbi_pc = nbi // n_cores
    blocks = []
    for cx in range(NCX):
        for i in range(nbi_pc):
            for j in range(nbj):
                for cy in range(NCY):
                    if iter_major:
                        # iteration-major: rb = i*(NCX*NCY) + cx*NCY + cy.
                        # iteration i owns a contiguous 16-row-block span, so the
                        # natural QKV[M] emerges (Q=i0..3, K=i4, V=i5) and the
                        # rope split sees K at rows 2048..2559 directly.
                        gi = i * (NCX * NCY) + cx * NCY + cy
                    elif core_major:
                        gi = (cx * NCY + cy) * nbi_pc + i  # contiguous per core
                    else:
                        gi = cx * (NCY * nbi_pc) + i * NCY + cy  # global row-block
                    rs, cs = gi * ROW_BLOCK, j * COL_BLOCK
                    gs = j * (COL_BLOCK // GROUP)
                    blocks.append(
                        pack_q4k_block(
                            q[rs : rs + ROW_BLOCK, cs : cs + COL_BLOCK],
                            scale[rs : rs + ROW_BLOCK, gs : gs + N_GROUPS],
                            mn[rs : rs + ROW_BLOCK, gs : gs + N_GROUPS],
                        )
                    )
    return np.concatenate(blocks)


def dequant_matrix(q, scale, mn):
    """w[r,c] = q[r,c]*scale[r,c//32] + mn[r,c//32] for full [M,K]."""
    M, K = q.shape
    w = np.zeros((M, K), dtype=np.float32)
    for r in range(M):
        for c in range(K):
            g = c // GROUP
            w[r, c] = float(q[r, c]) * float(scale[r, g]) + float(mn[r, g])
    return w


def ref_gemv_matrix(q, scale, mn, x):
    """y[r] = sum_c (q*scale+min)[r,c] * x[c] for full [M,K] @ [K]."""
    return (dequant_matrix(q, scale, mn) @ x.astype(np.float32)).astype(np.float32)
