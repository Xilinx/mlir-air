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
    assert mn.shape == (ROW_BLOCK, N_GROUPS)
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


def pack_q4k_cascade(
    q, scale, mn, NCX, NCY, core_major=False, iter_major=False, dual_chan=False
):
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

    dual_chan=True (the two-MM2S-per-column weight feed): the column's stream is
    split SPATIALLY, by cascade pair -- the low half of the rows (cy 0..NCY/2-1) is
    emitted for every fan step, then the high half. This is FLM's layout: its
    mem_C_1 takes shim ch0 on S2MM4 (writing w_buffer[0:5120], drained by MM2S0/1 to
    rows 2/3) and shim ch1 on S2MM5 (w_buffer[5120:10240], drained by MM2S2/3 to
    rows 4/5), with two INDEPENDENT lock cycles. Each channel therefore owns a
    disjoint set of cores and never has to be ordered against the other; each also
    reads one contiguous DDR run, so both stay single 1D shim BDs (a strided feed
    cannot: the innermost dim exceeds the AIE2 per-dim wrap limit, and only a
    contiguous 1D BD gets the wide buffer_length register).

    Do NOT split temporally (even/odd fan steps) instead: that makes every core's
    MM2S BD chain alternate between the two channels' buffers, which couples the two
    shim channels at every step and deadlocks on device.

    Returns int16 [NCX*nbi_pc*nbj*NCY * BLOCK_BF16] = [nbi*nbj * BLOCK_BF16].
    """
    M, K = q.shape
    assert M % ROW_BLOCK == 0 and K % COL_BLOCK == 0
    nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
    n_cores = NCX * NCY
    assert nbi % n_cores == 0, "row-blocks must split evenly across cores"
    nbi_pc = nbi // n_cores
    steps = [(i, j) for i in range(nbi_pc) for j in range(nbj)]
    # Per-column emission order: (step, cy) by default; dual_chan hoists the row
    # half (the shim channel) outermost so each channel's share is contiguous.
    if dual_chan:
        assert NCY % 2 == 0, f"dual_chan needs an even NCY (got {NCY})"
        order = [
            (i, j, cy)
            for h in range(2)
            for (i, j) in steps
            for cy in range(h * (NCY // 2), (h + 1) * (NCY // 2))
        ]
    else:
        order = [(i, j, cy) for (i, j) in steps for cy in range(NCY)]
    blocks = []
    for cx in range(NCX):
        for i, j, cy in order:
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
