# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# =========================== HOW TO RUN (recipe) ============================
# Builds the AIR decode layer, aiecc's an xclbin, and runs it on NPU2, printing
# the numeric check + xrt state. ~13 min end-to-end.
#
#   ENV (Peano, NOT Chess; build/bin, NOT install/bin). $AIR = your mlir-air checkout:
#     export PEANO=<llvm-aie install>   # e.g. .../site-packages/llvm-aie
#     export PEANO_INSTALL_DIR=$PEANO
#     export PATH=$AIR/build/bin:$AIR/mlir-aie/build/bin:/opt/xilinx/xrt/bin:$PATH
#     export PYTHONPATH=$AIR/build/python:$AIR/mlir-aie/build/python:\
#                 /opt/xilinx/xrt/python:$PYTHONPATH
#     source /opt/xilinx/xrt/setup.sh
#   RUN (native pyxrt harness, builds + runs + checks):
#     cd programming_examples/attn && python3.13 q4nx_decode.py
#   PASS = "state=<...COMPLETED: 4>" + "res2(...) cos=1.0000" + "PASS".
#   (rebuild the C++/Python shared libs first if the compiler changed:
#    ninja -C $AIR/build)
#
# The design is the static the reference-faithful LOOPCLOSE full decode (proj->attn->o-proj
# closed loop, separate post-attn rmsnorm); it is NOT parametrized. Remaining inputs:
#     DECODE_GOLDEN   (env)   dir of real Llama-3.2-1B golden dumps -> real weights/input/rms/rope
#     DECODE_GOLDEN_L (env)   KV context length (sets ATTN_L); MULTIBLK = ATTN_L>1
#   MULTIBLK (L>1) appends this token's roped K/V into the DDR cache on-chip
#   (KV_APPEND, = the reference _receive_kv_cache) then reads the whole cache back for attention.
#     the reference chaining ABI: layer output (res2) is written IN-PLACE to arg0 (hidden BO)
#   ABI (5 BOs, opcode 3): arg0=x/hidden (IN, and OUT in-place), arg1=proj_w(DDR),
#     arg2=rms_w(+rope LUT), arg3=(free; the reference rope_rms slot), arg4=kv_cache(DDR).
#
#   the reference-HOST-STACK harness (drives OUR xclbin through the reference's npu_app, the-reference-harness
#   generic_decoding_layer/bringup/bringup_gen.cpp):
#     1) dump inputs from this script: DECODE_BRINGUP_DUMP=/tmp/mb_dump python3.13 q4nx_decode.py
#     2) aiecc air_project/input_with_addresses.mlir -> /tmp/T.{xclbin,insts.bin}
#        (aiecc.py --get-xclbin --get-npu-insts
#         --xclbin-kernel-name=MLIR_AIE --peano=$PEANO)
#     3) build harness: bringup/build_gen.sh ; run (LD_LIBRARY_PATH=xrt/lib:../host_common):
#        INPLACE=1 BAKED_INSTS=/tmp/T.insts.bin ./bringup_gen.exe /tmp/mb_dump /tmp/T.xclbin
#        env: INPLACE=1 (output in arg0), ZERO_SLOT=1 + DUMP_KVC=1 (clean KV-append verify vs
#        kv_ref.bin), PROBE=6 (multi-dispatch: re-run 6x, expect cos(vs first res2)=1.0).
#
#   REAL-NUMERICS validation (the reference golden parity):
#     DECODE_GOLDEN=<run/golden*_dir> [DECODE_GOLDEN_L=1] python3.13 q4nx_decode.py
#       feeds REAL Llama-3.2-1B layer-0 weights/input/rmsnorm (repacked from the reference's
#       in_proj_w.bin via llama32_1b_q4nx_weights.py) + the reference's separate post_attention_layernorm
#       weight (POST_RMS), and compares the device layer output to a validated numpy
#       reference (llama32_1b_q4nx_weights.forward_layer, cos~1.0 on the KV/proj/rope path) and
#       to the reference's out_hidden.bin when valid. DECODE_GOLDEN_L=1 uses the iwa on-chip-KV
#       path (pos0: attention o=v, RoPE identity=real). Regenerate a golden dir with
#       the reference's host: DUMP_GOLDEN=<dir> DUMP_TOKEN=<i> the-reference-harness
#       --skip-reference (NOTE: the reference's out_hidden dump is NaN for tokens >0 -- a dump
#       artifact; the model is healthy -- so use the numpy reference for pos>0).
#
#   SCOPE / LIMITS (this is a single-decode-LAYER dataflow+ABI prototype, NOT a
#   deployable model): identity RoPE LUT (arg3 rope_rms slot unused), SYNTHETIC KV
#   oracle numerics (res2 cos is self-consistent, NOT PyTorch-Llama parity), ONE
#   layer (not 16), no LM head / embedding / sampling / tokenizer, KV block-count
#   baked (L in (16,32]). The runtime instruction stream is aiecc-BAKED and
#   DEVICE-SPECIFIC (targets THIS xclbin's tiles/RTP-addrs/shim channels) -- the reference's
#   native decoding_layer::_gen_sequence targets the reference's device and will NOT drive
#   this xclbin; a per-device generator (bringup/our_gen_sequence.hpp, using the reference's
#   npu_sequence API against OUR symbols) is the compatible route.
# ===========================================================================
#
# Clean, faithful AIR builder for the the reference (the reference) Llama-3.2-1B decode-layer
# PROJ subsystem. Single path, no config flags. Mirrors the proven hand-written
# reproducer q4nx_decode_repro/full_stripped.mlir EXACTLY for the proj cores +
# header-driven packet id-demux output:
#
#   16 proj cores (cols 0,1,6,7 x rows 2..5) form 8 CASCADE PAIRS (lead row 2/4 +
#   partner row 3/5 per col). Each pair shares two L1 y-buffers (memref<80> =
#   16 hdr region + 2*32 payload) on the LEAD tile. The lead writes its row
#   (proj_qmm_flush_row i=0 -> @16) and emits a 2-row packet (offset 14, size
#   66); the partner writes its row (i=1 -> @48) cross-tile into the SAME lead
#   buffer. Neither writes the routing header at @14 -- the compiler stores it
#   from the `dest` the lead's put names. Output flows are PACKET channels
#   carrying that header; the group memtile does the asymmetric one-header
#   gather (258 = hdr + 4*64), the main memtile a 2-slot daisy chain (514), and
#   ONE egress demuxes by it.
#
#   The proj core is the reproducer's persistent phase loop: for ph in 0..NPH with
#   scf.index_switch selecting I2 (row-pair iters) / J2 (col-block pairs) / pkt id
#   per phase -- NO repeat_count (count-free next_bd rings); Python-unrolled phases
#   are forbidden. X is the 256-element ping-pong ring (proj_qmm_acc256), 2*J2 gets
#   per row-block; the X memtile re-feeds the resident X via a count-free ring.
#
# This file is built up incrementally but stays a SINGLE clean path. The prior
# flag-heavy stage-2 build is preserved as q4nx_decode_BACKUP_stage2_circuit_golden.py.
import argparse
import numpy as np
from ml_dtypes import bfloat16

from air.ir import (
    ArrayAttr,
    BF16Type,
    F32Type,
    FlatSymbolRefAttr,
    IndexType,
    InsertionPoint,
    IntegerType,
    IntegerAttr,
    MemRefType,
    StringAttr,
    UnitAttr,
)
from air.dialects.air import (
    Channel,
    ChannelGet,
    ChannelPut,
    MemorySpace,
    T,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects.air import channel as channel_decl
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects import arith
from air.dialects.scf import (
    for_,
    yield_,
    index_switch,
    ParallelOp,
    ReduceOp,
    IfOp,
    ForOp,
)
from air.backend.xrt import XRTBackend

# An AIE2/AIE2P lock counter is 6 bits (AIETargetModel getMaxLockValue() == 0x3F),
# and the refeed count below becomes a lock init that nothing range-checks -- a
# larger one truncates on the device and the re-broadcast deadlocks.
MAX_REFEED = 0x3F


def refeed(n, emit):
    """Re-send ONE resident buffer n times: an n-trip scf.for around a single
    air.channel.put. The body holds nothing but the put and no operand depends
    on the induction variable, so this is a re-broadcast, not n productions --
    air-annotate-refeed recognizes the shape, collapses the loop, and derives
    the count for the lock init. n <= 1 emits the bare put.

    n > MAX_REFEED is split into equal lock-legal groups: the consumer still sees
    n puts, but no single collapsed loop asks for a lock the hardware cannot hold.
    One group (n <= 63) is every model shipped so far, and emits the same IR as
    before this split existed."""
    if n <= 1:
        emit()
        return
    ngrp = -(-n // MAX_REFEED)
    for g in range(ngrp):
        cnt = n // ngrp + (1 if g < n % ngrp else 0)
        if cnt <= 1:
            emit()
            continue
        c0 = arith.ConstantOp(IndexType.get(), 0).result
        cn = arith.ConstantOp(IndexType.get(), cnt).result
        c1 = arith.ConstantOp(IndexType.get(), 1).result
        for _rf in for_(c0, cn, c1):
            emit()
            yield_([])


def parallel_(n):
    """Spatial scf.parallel over [0, n) — the canonical form for a channel fan
    over a BUNDLE INDEX (@chan[iv]). air-to-aie spatially unrolls it to one
    physical endpoint per index (scf.for over a bundle index is a verifier
    error: bundle indices must be spatial, not temporal)."""
    c0 = arith.ConstantOp(IndexType.get(), 0).result
    cn = arith.ConstantOp(IndexType.get(), n).result
    c1 = arith.ConstantOp(IndexType.get(), 1).result
    par = ParallelOp(
        results_=[], lowerBound=[c0], upperBound=[cn], step=[c1], initVals=[]
    )
    blk = par.regions[0].blocks.append(IndexType.get())
    with InsertionPoint(blk):
        yield blk.arguments[0]
        ReduceOp([], 0)


from proj_qmm_pack import (
    ROW_BLOCK,  # 32
    COL_BLOCK,  # 256
    GROUP,  # 32
    BLOCK_BF16,  # 2560 (one packed q4k block)
    pack_q4k_cascade,
)

# ============================ model config ==================================
# Incremental model parametrization: the base per-model DIMENSIONS live in this
# table; every constant below DERIVES from these names, so a model is defined by
# its entry alone. DECODE_MODEL selects it (default llama-3.2-1b). The llama entry
# reproduces the original hardcoded values BYTE-IDENTICALLY (no-op).
import contextlib
import json as _json
import math as _math
import os as _os

_MODELS = {
    "llama-3.2-1b": dict(
        K=2048,  # model dim
        M=3072,  # QKV proj output rows (DQ+DK+DV = 2048+512+512)
        DH_A=64,  # head dim
        KV_PER_CU=2,  # kv heads per attention CU (8 kv / 4 CU)
        N_ATTN_CU=4,  # attention compute units (4 CU = 32 q heads)
        NPH=4,  # proj phases (QKV, o, gate-up, down)
        I2P=[3, 2, 16, 2],  # per-phase row-pair iters/core
        J2P=[4, 4, 4, 16],  # per-phase col-block pairs (2*J2 = NBJ = K/COL_BLOCK)
        DEST=["rope", "rms", "glu", "rms"],  # phase -> egress consumer
        GQA_SEG=4,  # GQA q-heads-per-group padding segment (ATTN_IMPL 2x4x1)
        PAIR_ROWS=2,  # proj egress: 2 = lead/partner shared-L1 pairing (FLM llama)
        N_NORMS=2,  # pre-norms only (input, post_attention) -- standard pre-norm
        HAS_QK_NORM=False,  # rope_w = cos/sin(DH) only
        VOCAB_SIZE=128256,
        UNI_DEC=16,  # decode waves (layers) in the unified sequence
        # lm-head waves (vocab chunks). 7 = the FLOOR: UNI_LM*VOCAB_CHUNK_I2 is
        # pinned at VOCAB_FULL_ROWBLKS/ROW_BLOCK = 126, and 18 is the largest legal
        # chunk (see the VOCAB_CHUNK_I2 derivation below), so 126/18 = 7 is the
        # fewest lm-head waves this model can be built with. Each wave is a
        # host-armed barrier, hence "fewest" is the direction to want.
        UNI_LM=7,  # lm-head waves (vocab chunks) in the unified sequence
    ),
    # Gemma3-4B (text): mirrors the ONE built FLM reference (gemma_npu_bin).
    # PAIR_ROWS=1 -> FLM-gemma NON-PAIRED proj egress (each CT emits 1 block ->
    # memtile 4-way gather), which natively handles D=2560 (odd 5 blocks/tile).
    # Per-phase block/col counts (PAIR_ROWS=1): I2P = NBI_PH/(NCX*NCY) blocks/tile
    #   NBI_PH = [M, D, 2*INTER, D]/ROW_BLOCK = [128,80,640,80] -> /16 = [8,5,40,5]
    #   J2P = NBJ/2, NBJ = [K,DQ,K,INTER]/COL_BLOCK = [10,8,10,40] -> [5,4,5,20]
    "gemma3-4b": dict(
        K=2560,
        M=4096,  # DQ+DK+DV = 2048+1024+1024
        DH_A=256,
        KV_PER_CU=1,  # 4 kv / 4 CU
        N_ATTN_CU=4,
        NPH=4,
        I2P=[8, 5, 40, 5],  # blocks/tile per phase (non-paired)
        J2P=[5, 4, 5, 20],  # col-block pairs per phase (2*J2 = NBJ = Kphase/COL_BLOCK)
        DEST=["rope", "rms", "glu", "rms"],
        GQA_SEG=4,  # ATTN_IMPL_1x4x1
        PAIR_ROWS=1,  # NON-PAIRED egress (FLM gemma)
        # Gemma3 sandwich norm: 4 norms/layer (input, post_attention, pre_feedforward,
        # post_feedforward). The 2 "post" norms are applied to the SUBLAYER OUTPUT
        # (o-proj / down) BEFORE the residual add: x = x + post_norm(sublayer(pre_norm(x))).
        N_NORMS=4,
        HAS_QK_NORM=True,  # rope_w = [cos/sin(DH), q_norm(DH), k_norm(DH)] = 3*DH
        VOCAB_SIZE=262208,
        UNI_DEC=34,  # 34 decoder layers
        # LM-head vocab chunking: VOCAB_SIZE_PADDED_FULL = ceil(262208/2560)*2560 =
        # 263680 -> 8240 rowblocks = 16*515, 515=5*103. VOCAB_ROWBLKS = 16*VOCAB_I2
        # (PAIR_ROWS=1) must divide 8240 -> VOCAB_I2 in {5,103}. VOCAB_I2=5 keeps the
        # per-dispatch op-count/BDs small (RNDS=5, 80 rowblocks/chunk) -> 103 chunks.
        # The driver MUST set VOCAB_CHUNK_I2=5 (env) to match this UNI_LM.
        UNI_LM=103,  # vocab chunks per LM head (VOCAB_CHUNK_I2=5)
    ),
    # Llama-3.2-3B: same topology as the 1B entry, only dimensions differ
    # (FLM's 1B and 3B layer.mlir are byte-structurally identical). head_dim
    # doubles to 128 and GQA goes 4 q/kv-group -> 3, which the GQA_SEG=4
    # padding absorbs (Q_HEADS_PADDED_PER_CU stays 8, so SSZ_BLK stays 192).
    #   I2P = [M, D, 2*INTER, D]/(ROW_BLOCK*NCX*NCY*PAIR_ROWS) = [5,3,16,3]
    #   J2P = [K, K, K, INTER]/(2*COL_BLOCK)                   = [6,6,6,16]
    "llama-3.2-3b": dict(
        K=3072,
        M=5120,  # DQ+DK+DV = 3072+1024+1024
        DH_A=128,
        KV_PER_CU=2,  # 8 kv / 4 CU
        N_ATTN_CU=4,
        NPH=4,
        I2P=[5, 3, 16, 3],
        J2P=[6, 6, 6, 16],
        DEST=["rope", "rms", "glu", "rms"],
        GQA_SEG=4,  # ATTN_IMPL_2x4x1
        PAIR_ROWS=2,
        N_NORMS=2,
        HAS_QK_NORM=False,
        VOCAB_SIZE=128256,
        UNI_DEC=28,
        # LM-head vocab chunking: the vocab rms relays logits in whole-K blocks
        # (VOCAB_RNDS*PAYLOAD // K rounds), so (K/PAYLOAD) must DIVIDE
        # VOCAB_RNDS = VOCAB_I2*PAIR_ROWS or the round count floor-truncates ->
        # too few xnorm broadcasts + a short logit drain -> the vocab wave
        # DEADLOCKS. K/PAYLOAD = 6 here, so the 1B default 14 fails; VOCAB_I2=9
        # gives RNDS=18 (6 | 18) and 288 rowblocks/chunk, dividing the 4032
        # full-vocab rowblocks into 14 chunks. The driver MUST set
        # VOCAB_CHUNK_I2=9 (env) to match this UNI_LM.
        UNI_LM=14,  # vocab chunks per LM head (VOCAB_CHUNK_I2=9)
    ),
    # Phi-4-mini-instruct. Attention topology and per-phase block counts are
    # IDENTICAL to llama-3.2-3b (K=3072, M=5120, 8 kv heads, 2x4x1, DH=128), so
    # I2P/J2P carry over unchanged. Two things differ:
    #   - PARTIAL ROTARY. partial_rotary_factor=0.75 -> RoPE covers the leading
    #     96 of 128 head dims, the rest passes through. ROPE_DIM shrinks the
    #     cos/sin LUT to 96 and selects rope.cc's partial path (FLM ships a
    #     separate rope_phi4.cc for the same thing).
    #   - VOCAB 200064, which re-derives the chunking below.
    "phi4-mini": dict(
        K=3072,
        M=5120,  # DQ+DK+DV = 3072+1024+1024
        DH_A=128,
        ROPE_DIM=96,  # partial rotary (0.75 * 128)
        KV_PER_CU=2,  # 8 kv / 4 CU
        N_ATTN_CU=4,
        NPH=4,
        I2P=[5, 3, 16, 3],
        J2P=[6, 6, 6, 16],
        DEST=["rope", "rms", "glu", "rms"],
        GQA_SEG=4,  # ATTN_IMPL_2x4x1
        PAIR_ROWS=2,
        N_NORMS=2,
        HAS_QK_NORM=False,
        VOCAB_SIZE=200064,
        UNI_DEC=32,
        # VOCAB_SIZE_PADDED_FULL = ceil(200064/3072)*3072 = 202752 -> 6336
        # rowblocks = 198 * (NCX*NCY*PAIR_ROWS), so VOCAB_I2 must divide 198.
        # K/PAYLOAD = 6 must divide VOCAB_I2*PAIR_ROWS, i.e. 3 | VOCAB_I2, and the
        # tested envelope caps 2*VOCAB_I2 <= 63. That leaves {3,6,9,18}; 18 is the
        # largest, i.e. the fewest host-armed waves: 198/18 = 11.
        # The driver MUST set VOCAB_CHUNK_I2=18 (env) to match this UNI_LM.
        UNI_LM=11,  # vocab chunks per LM head (VOCAB_CHUNK_I2=18)
    ),
    # Qwen3-8B: same paired-egress topology as the llama entries (ATTN_IMPL_2x4x1,
    # DH=128, PAIR_ROWS=2), plus Qwen3's per-head QK-norm. Only the dims grow.
    #   I2P = [M, D, 2*INTER, D]/(ROW_BLOCK*NCX*NCY*PAIR_ROWS) = [6,4,24,4]
    #   J2P = [K, K, K, INTER]/(2*COL_BLOCK)                   = [8,8,8,24]
    "qwen3-8b": dict(
        K=4096,
        M=6144,  # DQ+DK+DV = 4096+1024+1024
        DH_A=128,
        KV_PER_CU=2,  # 8 kv / 4 CU
        N_ATTN_CU=4,
        NPH=4,
        I2P=[6, 4, 24, 4],
        J2P=[8, 8, 8, 24],
        DEST=["rope", "rms", "glu", "rms"],
        GQA_SEG=4,  # ATTN_IMPL_2x4x1
        PAIR_ROWS=2,
        N_NORMS=2,  # standard pre-norm (input, post_attention)
        HAS_QK_NORM=True,  # Qwen3: rope_w = [cos/sin(DH), q_norm(DH), k_norm(DH)]
        VOCAB_SIZE=151936,
        UNI_DEC=36,  # 36 decoder layers
        # LM-head vocab chunking: VOCAB_SIZE_PADDED_FULL = ceil(151936/4096)*4096 =
        # 155648 -> 4864 rowblocks, so UNI_LM*VOCAB_CHUNK_I2 = 4864/32 = 152 = 8*19.
        # K/PAYLOAD = 4096/512 = 8 must divide VOCAB_RNDS = VOCAB_I2*PAIR_ROWS, so
        # VOCAB_I2 must be a multiple of 4: {4,8,76,152}. 8 is the largest at or
        # below the 18-chunk ceiling -> 19 waves, 256 rowblocks/chunk.
        # The driver MUST set VOCAB_CHUNK_I2=8 (env) to match this UNI_LM.
        UNI_LM=19,  # vocab chunks per LM head (VOCAB_CHUNK_I2=8)
    ),
    # Qwen2.5-7B-Instruct: the first HAS_QKV_BIAS model on this engine (q/k/v_proj
    # carry a bias, added in-place before RoPE from a slab at rope_w+DH -- see
    # ROPE_W_LEN). D=3584 is 7 col-blocks wide, which is odd in units of the paired
    # egress, so PAIR_ROWS=1 (gemma-style non-paired) is forced: PAIR_ROWS=2 gives
    # I2P[0] = 4608/1024 = 4.5. 28 q heads / 4 kv is 7 per group, padded to 8 by
    # GQA_SEG (ATTN_IMPL_1x8x1), same padding gemma uses for 3 -> 4.
    #   I2P = [M, D, 2*INTER, D]/(ROW_BLOCK*NCX*NCY*PAIR_ROWS) = [9,7,74,7]
    #   J2P = [K, DQ, K, INTER]/(2*COL_BLOCK)                  = [7,7,7,37]
    "qwen2.5-7b": dict(
        K=3584,
        M=4608,  # DQ+DK+DV = 3584+512+512
        DH_A=128,
        KV_PER_CU=1,  # 4 kv / 4 CU
        N_ATTN_CU=4,
        NPH=4,
        I2P=[9, 7, 74, 7],  # blocks/tile per phase (non-paired)
        J2P=[7, 7, 7, 37],  # col-block pairs per phase
        DEST=["rope", "rms", "glu", "rms"],
        GQA_SEG=8,  # ATTN_IMPL_1x8x1
        PAIR_ROWS=1,  # NON-PAIRED egress (D=3584 is odd in paired units)
        N_NORMS=2,  # standard pre-norm (input, post_attention)
        HAS_QKV_BIAS=True,  # rope_w = [cos/sin(DH), q_bias|k_bias|v_bias(M)]
        VOCAB_SIZE=152064,
        UNI_DEC=28,  # 28 decoder layers
        # LM-head vocab chunking: VOCAB_SIZE_PADDED_FULL = ceil(152064/3584)*3584 =
        # 154112 -> 4816 rowblocks. VOCAB_ROWBLKS = 16*VOCAB_I2 (PAIR_ROWS=1) must
        # divide 4816 -> UNI_LM*VOCAB_I2 = 301 = 7*43, so VOCAB_I2 in {1,7,43,301}.
        # K/PAYLOAD = 3584/512 = 7 must divide VOCAB_RNDS = VOCAB_I2*PAIR_ROWS,
        # dropping 1; the 2*VOCAB_I2 <= 63 envelope drops 43 and 301. VOCAB_I2=7 is
        # therefore the ONLY legal chunk (RNDS=7 = exactly one relay round, as in
        # gemma) -> 43 waves, 112 rowblocks/chunk.
        # The driver MUST set VOCAB_CHUNK_I2=7 (env) to match this UNI_LM.
        UNI_LM=43,  # vocab chunks per LM head (VOCAB_CHUNK_I2=7)
    ),
    # Qwen3-4B: the DFlash target (see docs/DFlashFeasibility.md). Qwen3 QK-norm
    # like qwen3-8b, but with the DECOUPLED q dim the bf16 llms/qwen3_4b example
    # already handles: n_heads*head_dim = 4096 != hidden 2560, so the o-proj
    # contracts 4096 -> 2560 and J2P[1] is DQ/512, not K/512.
    #
    # PAIR_ROWS=1 is forced, as it is for qwen2.5-7b: the paired egress needs
    # every phase output divisible by ROW_BLOCK*NCX*NCY*PAIR_ROWS = 1024, and the
    # o/down phases emit K=2560 -> 2.5. Non-paired (divisor 512) is exact.
    #   I2P = [M, K, 2*INTER, K]/(ROW_BLOCK*NCX*NCY*PAIR_ROWS)
    #       = [6144, 2560, 19456, 2560]/512 = [12, 5, 38, 5]
    #   J2P = [K, DQ, K, INTER]/(2*COL_BLOCK)
    #       = [2560, 4096, 2560, 9728]/512 = [5, 8, 5, 19]
    "qwen3-4b": dict(
        K=2560,
        M=6144,  # DQ+DK+DV = 4096+1024+1024
        DH_A=128,
        KV_PER_CU=2,  # 8 kv / 4 CU
        N_ATTN_CU=4,
        NPH=4,
        I2P=[12, 5, 38, 5],  # blocks/tile per phase (non-paired)
        J2P=[5, 8, 5, 19],
        DEST=["rope", "rms", "glu", "rms"],
        GQA_SEG=4,  # 32 q / 8 kv = 4 per group, no padding needed
        PAIR_ROWS=1,  # NON-PAIRED egress (K=2560 is odd in paired units)
        N_NORMS=2,  # standard pre-norm (input, post_attention)
        HAS_QK_NORM=True,  # Qwen3: rope_w = [cos/sin(DH), q_norm(DH), k_norm(DH)]
        VOCAB_SIZE=151936,
        UNI_DEC=36,  # 36 decoder layers
        # LM-head vocab chunking: VOCAB_SIZE_PADDED_FULL = ceil(151936/2560)*2560
        # = 153600 -> 4800 rowblocks. VOCAB_ROWBLKS = 16*VOCAB_I2 (PAIR_ROWS=1)
        # must divide 4800, so UNI_LM*VOCAB_CHUNK_I2 = 300. K/PAYLOAD = 2560/512
        # = 5 must divide VOCAB_RNDS = VOCAB_I2, so VOCAB_I2 is a multiple of 5:
        # {5,10,15,20,25,30} once the tested 2*VOCAB_I2 <= 63 envelope is applied.
        # 30 is the largest, i.e. the fewest host-armed waves: 300/30 = 10.
        # The driver MUST set VOCAB_CHUNK_I2=30 (env) to match this UNI_LM.
        UNI_LM=10,  # vocab chunks per LM head (VOCAB_CHUNK_I2=30)
    ),
    # Qwen3-4B DFlash DRAFTER (z-lab/Qwen3-4B-DFlash-b16). Byte-identical to the
    # qwen3-4b entry above except UNI_DEC: the drafter is 5 Qwen3-4B layers, so
    # every per-layer constant carries over and only the layer count changes.
    # That is the "one engine, two configurations" claim in
    # docs/DFlashFeasibility.md, made concrete.
    #
    # Not modelled here (they sit outside the per-layer loop): the mask-token
    # embedding, the fc linear that fuses the 5 target hidden-state taps
    # (12800 -> 2560), and hidden_norm.
    "qwen3-4b-draft": dict(
        K=2560,
        M=6144,
        DH_A=128,
        KV_PER_CU=2,
        N_ATTN_CU=4,
        NPH=4,
        I2P=[12, 5, 38, 5],
        J2P=[5, 8, 5, 19],
        DEST=["rope", "rms", "glu", "rms"],
        GQA_SEG=4,
        PAIR_ROWS=1,
        N_NORMS=2,
        HAS_QK_NORM=True,
        VOCAB_SIZE=151936,
        UNI_DEC=5,  # <-- the only difference: 5 drafter layers, not 36
        UNI_LM=10,  # tied to the target's head; VOCAB_CHUNK_I2=30 as above
    ),
    # Llama-3.1-8B: same attention topology as 1B/3B (2x4x1, 8 kv heads, DH=128),
    # so the per-CU KV geometry is unchanged; only the proj/FFN widths grow. Like
    # qwen3-8b this needs DECODE_WGROUP (32 layers of K=4096 weights are 3.6 GiB
    # of layer slabs, and the lm-head pushes the total past the 4 GiB one-BO
    # ceiling) and a lowered DECODE_STACK. Unlike qwen3-8b there is no QK-norm,
    # and the LM head is NOT tied to the embedding -- a weight-loader concern
    # only, the device sequence is identical.
    #   I2P = [M, D, 2*INTER, D]/(ROW_BLOCK*NCX*NCY*PAIR_ROWS) = [6,4,28,4]
    #   J2P = [K, K, K, INTER]/(2*COL_BLOCK)                   = [8,8,8,28]
    "llama-3.1-8b": dict(
        K=4096,
        M=6144,  # DQ+DK+DV = 4096+1024+1024
        DH_A=128,
        KV_PER_CU=2,  # 8 kv / 4 CU
        N_ATTN_CU=4,
        NPH=4,
        I2P=[6, 4, 28, 4],
        J2P=[8, 8, 8, 28],
        DEST=["rope", "rms", "glu", "rms"],
        GQA_SEG=4,  # ATTN_IMPL_2x4x1
        PAIR_ROWS=2,
        N_NORMS=2,  # standard pre-norm (input, post_attention)
        VOCAB_SIZE=128256,
        UNI_DEC=32,  # 32 decoder layers
        # LM-head vocab chunking (same rule as the other entries):
        # VOCAB_SIZE_PADDED_FULL = ceil(128256/4096)*4096 = 131072 -> 4096
        # rowblocks, so UNI_LM*VOCAB_CHUNK_I2 = 4096/32 = 128. K/PAYLOAD =
        # 4096/512 = 8 must divide VOCAB_RNDS = VOCAB_I2*PAIR_ROWS, so VOCAB_I2
        # must be a multiple of 4: {4,8,16,32}. The tested 2*VOCAB_I2 <= 63
        # envelope rules out 32, so 16 is the largest -> 8 waves.
        # The driver MUST set VOCAB_CHUNK_I2=16 (env) to match this UNI_LM.
        UNI_LM=8,  # vocab chunks per LM head (VOCAB_CHUNK_I2=16)
    ),
}
MODEL_NAME = _os.environ.get("DECODE_MODEL", "llama-3.2-1b")
MODEL = _MODELS[MODEL_NAME]

# Derived model geometry (attention head layout). All values are byte-identical to
# the original hardcoded Llama literals; used to migrate the raw 64/512 attention
# dims off Llama-specific constants. (llama / gemma3-4b):
DH = MODEL["DH_A"]  # head dim (64 / 256)
# rope weight buffer: cos/sin(DH) for llama; qk-norm models (gemma) prepend q/k
# per-head RMSNorm weights -> rope_w = [cos/sin, q_norm, k_norm] = 3*DH. The rope
# kernel reads segments 2,3 behind #ifdef HAS_QK_NORM (already present).
HAS_QK_NORM = MODEL.get("HAS_QK_NORM", False)
# Rotary width: DH for full rotary, less for a partial-rotary model (Phi-4 ropes
# 96 of 128 and copies the tail), which also shrinks the cos/sin LUT to ROPE_DIM.
# Must agree with PARTIAL_ROPE_DIM in the model's C++ header -- the kernel reads
# sin at rope_w + ROPE_DIM/2. Partial rotary + qk-norm is rejected there.
ROPE_DIM = MODEL.get("ROPE_DIM", DH)
# Qwen2.5 has q/k/v_proj biases; the kernel adds them in-place before RoPE from a
# [q(DQ)|k(DK)|v(DV)] slab it reads at rope_w+DH (rope.cc add_q_k_v_bias), so the
# rope weight buffer carries the DH cos/sin LUT plus M = DQ+DK+DV bias elements.
HAS_QKV_BIAS = MODEL.get("HAS_QKV_BIAS", False)
ROPE_W_LEN = (
    (DH + MODEL["M"]) if HAS_QKV_BIAS else (3 * DH) if HAS_QK_NORM else ROPE_DIM
)  # 64 / 768 / 96 / 4736
# Does rope_w DIFFER PER LAYER? Llama's is a single per-position cos/sin LUT shared
# by every layer, so one slab suffices; qk-norm (gemma/qwen3) and q/k/v-bias
# (qwen2.5) both append per-layer weights, so the RMS BO needs UNI_DEC slabs and
# the feed has to index the current wave's. Getting this wrong DEADLOCKS: the host
# writes UNI_DEC slabs and puts final_norm after them, so a device that sized the
# region for one slab reads final_norm from inside the rope region.
ROPE_W_PER_LAYER = HAS_QK_NORM or HAS_QKV_BIAS
NUM_KV_HEADS = MODEL["N_ATTN_CU"] * MODEL["KV_PER_CU"]  # 8 / 4
NUM_Q_HEADS = (MODEL["M"] - 2 * NUM_KV_HEADS * DH) // DH  # 32 / 8
Q_HEADS_PER_CU = NUM_Q_HEADS // MODEL["N_ATTN_CU"]  # 8 / 2
DQ = NUM_Q_HEADS * DH  # q width 2048 / 2048
DK = NUM_KV_HEADS * DH  # k width 512 / 1024
DV = DK  # v width
DQ_PER_CU = Q_HEADS_PER_CU * DH  # per-CU q (=o) width 512 / 512
DK_PER_CU = MODEL["KV_PER_CU"] * DH  # per-CU k (=v) width 128 / 256
# GQA padding (mirrors kernels/model_spec.h): q-heads-per-group padded up to the
# attn-impl segment; the per-CU q/o/y buffers + score buffer are sized on the
# PADDED head count. (llama: no padding -> 512/192 ; gemma: 2->4 pad -> 1024/128)
Q_HEADS_PER_GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 4 / 2
_GSEG = MODEL["GQA_SEG"]
Q_HEADS_PER_GROUP_PADDED = ((Q_HEADS_PER_GROUP + _GSEG - 1) // _GSEG) * _GSEG  # 4 / 4
Q_HEADS_PADDED_PER_CU = MODEL["KV_PER_CU"] * Q_HEADS_PER_GROUP_PADDED  # 8 / 4
DQ_PADDED_PER_CU = Q_HEADS_PADDED_PER_CU * DH  # per-CU q/o/y buffer 512 / 1024
# Padded total Q the rope kernel emits: per kv-head it writes Q_HEADS_PER_GROUP real
# heads + ATTN_GROUPS_PADDING zero heads (GQA-segment alignment). llama pad=0 ->
# DQ_PADDED==DQ; gemma pad=2 -> 2x DQ. The rope Q buffer + broadcast memtile must be
# this size or the rope overflows them.
DQ_PADDED = MODEL["N_ATTN_CU"] * DQ_PADDED_PER_CU  # 2048 / 4096
SSZ_BLK = ((Q_HEADS_PADDED_PER_CU * 16 + 16 + 63) // 64) * 64  # score buffer 192 / 128

# ============================ faithful config ===============================
# Reproducer: Llama-3.2-1B decode layer, single token. QKV proj = q(2048)+k(512)+
# v(512) = 3072 rows out of K=2048 model dim. 16 proj cores at the reference columns
# 0,1,6,7, rows 2..5.
M = MODEL["M"]  # QKV proj output rows
K = MODEL["K"]  # model dim (proj contraction)
NCX = 4  # proj columns
NCY = 4  # proj rows (2..5)
PCOL = [0, 1, 6, 7]  # physical proj columns
NBI = M // ROW_BLOCK  # 96 output row-blocks
NBJ = K // COL_BLOCK  # 8 col-blocks (256-wide each)

# Cascade pairs: per col cx, two pairs pp; lead cy=2*pp (rows 2/4), partner
# cy=2*pp+1 (rows 3/5). 2 GROUPS of 4 leads (group g = cols {2g,2g+1}); group
# memtiles at phys cols 0 and 6; main memtile at phys col 1.
# Emitters per column: paired (PAIR_ROWS=2, FLM llama) = NCY//2 lead/partner pairs;
# non-paired (PAIR_ROWS=1, FLM gemma) = NCY independent tiles. Each emitter ships
# PAIR_PAY = PAIR_ROWS*ROW_BLOCK payload (2 blocks lead+partner / 1 block per tile).
PAIRS_PC = NCY // MODEL["PAIR_ROWS"]  # emitters per column (2 paired / 4 non-paired)
N_PAIRS = NCX * PAIRS_PC  # emitters total (8 paired / 16 non-paired)
# Group memtiles. Llama (paired): 2 groups of 2 columns each. Gemma (non-paired):
# PER-COLUMN gather (N_GRP=NCX) so each proj-col memtile gathers only ITS OWN cores
# packet-merged onto one S2MM -- matches FLM gemma's mem_C_1 4-gather and keeps the
# packet-switch arbiter (AMSel, <=4 master-selects) within capacity (8 cross-column
# leads on one S2MM overflow the arbiter -> aiecc pathfinder crash).
N_GRP = NCX if MODEL["PAIR_ROWS"] == 1 else 2  # group memtiles
LEADS_PER_GRP = N_PAIRS // N_GRP  # emitters per group (4 paired / 8 non-paired)
GRP_PCOL = (
    list(PCOL) if MODEL["PAIR_ROWS"] == 1 else [0, 6]
)  # phys cols of group memtiles
# Main/assemble memtile column. Llama: col 1 (a proj col; paired egress leaves room).
# Gemma (per-column egress): the proj col memtiles each need 6 S2MM (inW on 4/5 +
# 4 separate per-lead egress on 0-3, mirroring FLM mem_C_1 -- NO packet-merge), so the
# hub must NOT sit on a proj col. Put it on col 2, the X-broadcast memtile (5 free S2MM)
# -- exactly FLM, whose hub mem_1_1 IS its X-broadcast memtile.
# W_DUAL_CHAN=1: drive each proj column's weight stream on BOTH of its shim MM2S
# channels (@inW0c{cx} and @inW1c{cx}) instead of ch0 only. Decode at batch 1 is
# ~92% weight streaming, and the reference feeds 2 MM2S per weight column while we
# feed 1; the per-column bandwidth gap (10.2 vs 14.4 GB/s on the SAME physical
# columns) says the column is not saturated by one channel.
#
# The split is SPATIAL, by cascade pair: channel 0 carries the low half of the
# column's rows (cy 0..NCY/2-1) for every fan step, channel 1 the high half. Each
# channel therefore feeds a DISJOINT set of cores through its own fan ring, so the
# two run concurrently without ever being ordered against each other -- this is
# FLM's layout (mem_C_1 takes shim ch0 on S2MM4 and ch1 on S2MM5, two independent
# lock cycles). Each channel also reads one contiguous DDR run, so both stay single
# 1D shim BDs and keep the air.coalesced_shim_feed cross-channel phase barrier.
#
# Do NOT split temporally (even/odd fan steps) instead: that makes every core's
# MM2S BD chain alternate between the two channels' buffers, coupling the channels
# at every step -- measured to deadlock on device. A temporal feed also cannot be a
# single shim task, since the 10240-element fan step exceeds the AIE2 per-dim wrap
# limit and only a contiguous 1D BD gets the wide buffer_length register.
#
# Requires the host weight array packed with pack_q4k_cascade(dual_chan=True).
# Exported so the weight packers (llms/*_q4nx requant) key their cascade order and
# their cache off the same flag as the build.
W_DUAL_CHAN = int(_os.environ.get("W_DUAL_CHAN", "1"))


def _wname(ci, cx):
    """Weight channel name: the single @inW bundle when W_DUAL_CHAN is off, else
    the per-column, shim-col-pinned channel for shim channel ci of column cx."""
    return f"inW{ci}c{cx}" if W_DUAL_CHAN else "inW"


MAIN_PCOL = 2 if MODEL["PAIR_ROWS"] == 1 else 1  # phys col of the main memtile
# Faithful X-feed (reproducer core_2_2): the rms producer core
# (tile_2_2, col2) normalizes raw X once and re-feeds it via an output-lock release
# of N (= REFEED) into a 512 x_buffer that broadcasts 256-blocks to the 16 proj
# cores. (An older note here claimed col-1 congestion forced that x_buffer onto
# col 2, away from the reference's mem_1_1. That is STALE: with W_DUAL_CHAN the
# x_buffer sits on mem_1_1 exactly like the reference, and it has to -- see
# XMT_PCOL below.)
RMS_PCOL = 2  # rms producer core column
# X-broadcast memtile column. W_DUAL_CHAN follows FLM and puts it on the MAIN/hub
# memtile (mem_1_1) instead of col 2: FLM has NO memtile in column 2 at all -- that
# column holds only the shim rms/rope feeds and the rms/rope cores -- and broadcasts
# X from mem_1_1 DMA:4 to all 16 proj cores. Keeping the X memtile on col 2 while
# also doubling the weight flows makes the pathfinder fail outright (it cannot even
# route the one-hop rms->X xnorm packet flow tile_2_2 DMA1 -> mem_2_1 DMA0), because
# col 2 would carry the shim feeds, both cores, AND a 16-way broadcast hub.
# Overridable so the floorplan move can be A/B-tested independently of the
# channel split (XMT_PCOL=1 with W_DUAL_CHAN=0 isolates the placement effect).
XMT_PCOL = int(_os.environ.get("XMT_PCOL", MAIN_PCOL if W_DUAL_CHAN else RMS_PCOL))
# Column of the glu-down memtile, the third producer converging on @xnorm (the
# other two are the o-proj memtile on col 5 and the rms core itself). Distinct
# from col 5 either way, so the convergence never merges o+down onto one MM2S
# ring. Gemma needs it ADJACENT to the X memtile: its per-column egress puts a
# group memtile on every proj column AND lands the hub on the X column (
# MAIN_PCOL == XMT_PCOL == 2), so a down->X route from col 4 crosses switches
# that are already carrying the hub traffic and the pathfinder cannot complete
# all three sources of the merge -- mlir-aie reports it as an incomplete
# packet-flow routing (it used to drop the source silently). Llama's hub is on
# col 1, clear of the X column, and routes fine from col 4.
DOWN_PCOL = int(_os.environ.get("DOWN_PCOL", 3 if MODEL["PAIR_ROWS"] == 1 else 4))

# Phases (reproducer I2=[3,2,16,2], J2=[4,4,4,16], pkt=[1,4,8,4]). Phase 0 = QKV
# (id1), phase 1 = o-proj (id4), phase 2 = gate-up MLP (id8), phase 3 = DOWN (id4).
# Phases 0-2 contract over K=MODEL_DIM=2048 reading the rmsnorm'd token X; gate-up's
# 16384 output is consumed on-chip by the GLU tile (silu(gate)*up -> 8192), and that
# 8192 is fed back ON-CHIP as the DOWN phase X (K=INTERMEDIATE=8192) -> layer output
# 2048. Per-phase K differs: ph0-2 K=2048 (NBJ=8), ph3 K=8192 (NBJ=32).
import os as _os

# Tokens per superkernel call. Read HERE rather than beside the rest of the
# batching constants further down, because the attention block count needs it
# and that is settled with the model geometry. See the block comment at
# BATCH_MAX_PROJ for what the value means and why 8 is the ceiling.
BATCH = int(_os.environ.get("DECODE_BATCH", "1"))

# DECODE_PROBE: tap mid-layer buffers out to the (otherwise unused) Y BO, so a hung
# dispatch says HOW FAR it got. This exists because nothing else does.
#
# A hung batched dispatch leaves exactly two pieces of evidence -- the DDR KV cache
# (written by the rope append, early) and X (written by the layer-out drain, last) --
# and everything in between is one opaque region. The two obvious ways to split it
# both fail on this floorplan, and both cost a build to learn:
# air.preserve_shim_dma_order is a GLOBAL order, so hoisting a drain to report
# earlier starves the whole sequence; and a fresh shim endpoint between the append
# and the readback does not route. What DOES work is tapping a buffer that ALREADY
# EXISTS, on a tile that already has a route to a shim, into a BO nothing else writes.
#
# A bit mask, so several taps can share one build -- each one costs ~5 minutes:
#
#   1  Q  the q memtile, after rope's B q rows have landed and BEFORE the fan
#         -> rope finished, and the fan is what is blocked
#   2  O  the o-gather memtile, ONE PUT PER TOKEN as that token's four CU
#         outputs land -> how many tokens of the block got through attention
#   4  D  the down memtile, one put per GLU slice
#         -> o-proj (ph1), the ph2 norm, gate-up and the GLU core all completed
#
# PER TOKEN, not once at the end, and that is most of the value. Four of the six
# faults so far were "got N of B tokens through", and a tap that fires only on
# completion cannot tell that from "got none". A partially-received shim BD still
# wrote the bytes it did receive, so the prefix that lands IS the count.
#
# ONE CHANNEL PER SOURCE TILE, for a reason that cost two builds: a fresh shim
# endpoint is not free. probeA from the attention CU took shim_noc_tile_3_0, moved
# layerOut to column 0, and the cascade left the rms core's @xnorm with no route to
# the X memtile. So Q and O share one channel out of mem_tile_5_1 -- they are both
# on it, in program order -- and D has its own out of mem_tile_4_1.
#
# Y is the right target: HOST_DRAIN is [dest 0] and dest 0 is loop-closed on chip,
# so in this config NOTHING writes Y, batch_equiv.py --smoke already reads it back,
# and it takes no traffic off the KV cache's RAW barrier. Off by default, and the
# batch-1 no-op gate is what keeps it that way.
PROBE = int(_os.environ.get("DECODE_PROBE", "0"))
PROBE_Q, PROBE_O, PROBE_D = 1, 2, 4

# DECODE_ACC_STOP: send an INTERMEDIATE residual out on layerOut instead of the
# layer output. The bisector for a wrong layer output, and the one that works
# where the memtile taps do not.
#
#   0  (default)  layer output = x + o-proj + down
#   1             layer output = x                 -- pure passthrough
#   2             layer output = x + o-proj        -- through attention and ph1
#
# The gets are still issued at every setting, so every channel stays balanced
# and the shim task is the same task in the same place -- which is the whole
# trick. layerOut is the ONLY thing the layer sends across the shim, and it is
# ordered last; moving the PUT earlier in the core's program deadlocks (the
# later weight feeds are behind that shim task and the core has stopped feeding
# them), while dropping an ADD leaves the program shape alone and only changes
# what the buffer holds.
ACC_STOP = int(_os.environ.get("DECODE_ACC_STOP", "0"))

# Faithful decode: real attention (QKV -> rope -> flash attn -> o-proj X), mirroring
# q4nx_decode_repro/full_decode_faithful.mlir.
# the reference's fixed attention geometry: 4 CUs, each = 8 q heads + 2 kv heads (= 32 q heads,
# 8 kv heads). CU placement (col, qk_row, kv_row) below uses cols 3,4.
N_ATTN_CU = MODEL["N_ATTN_CU"]  # fixed the reference dimension (4 CUs = 32 q heads)
# Loop-close (faithful): the gathered attention o feeds o-proj's X (ph1), closing the
# proj->attn->o-proj loop. The X feed carries 4 phase sources in order: ph0
# rmsnorm(input), ph1 attn-o, ph2 rmsnorm(x+o-proj), ph3 GLU.
# Multi-block (real-L) attention: ATTN_L = KV context length (number of cached
# positions this token attends to). ATTN_L=1 -> the proven single-block (o==v)
# path; ATTN_L>1 -> reproducer model: in-core block loop over ATTN_ROUNDS=(L+15)/16
# blocks + online softmax + whole-cache-linear DDR KV readback (the reference _move_kv_cache;
# NOT per-block-strided, which FAILed 0.94). Compile-time rounds first (build per L);
# runtime RTP-L + the reference-sequence driving layered on after numeric validation.
ATTN_L = 32  # KV context length
# REAL-NUMERICS validation against the reference golden dumps (run/golden*/). When
# DECODE_GOLDEN=<dir> is set, the harness feeds REAL Llama-3.2-1B layer-0 weights,
# input, and rmsnorm weights (repacked from the reference's in_proj_w.bin via
# llama32_1b_q4nx_weights.py) and compares the device layer output to a validated numpy
# reference (+ the reference's out_hidden.bin when valid). DECODE_GOLDEN_L sets the context
# length (pos0 -> L=1 = iwa on-chip-KV path, attention o=v, RoPE irrelevant).
DECODE_GOLDEN = _os.environ.get("DECODE_GOLDEN", "")
if DECODE_GOLDEN:
    ATTN_L = int(_os.environ.get("DECODE_GOLDEN_L", "1"))
# Feed the reference's separate post_attention_layernorm weight to the on-chip 2nd rmsnorm
# (the reference uses a distinct post_attention_layernorm; required for real the reference parity).
POST_RMS = bool(DECODE_GOLDEN)
# Norms per layer: 2 = pre-norm (llama input+post_attention); 4 = Gemma sandwich
# (input, post_attention, pre_feedforward, post_feedforward). The 2 extra Gemma
# norms are applied to the sublayer OUTPUT before the residual add. Falls back to
# 1 when POST_RMS is off (debug configs), keeping RMS_LAYER byte-identical.
N_NORMS = MODEL["N_NORMS"] if POST_RMS else 1
# A block of B tokens occupies B CONSECUTIVE cache positions, so the LAST
# token's context is ATTN_L + B - 1 and every block count -- the shim readback,
# the memtile dequeue, the cores' loops -- is sized for that. Each token then
# reads the same number of KV blocks and masks with its OWN L: attn_qk_blk and
# attn_kv_blk already return early on a block past L, which is what lets one
# uniform count serve a batch whose contexts differ. Sizing per token instead
# would desync the shim's push from the core's consume.
ATTN_L_BLK = ATTN_L + BATCH - 1
ATTN_ROUNDS = (ATTN_L_BLK + 15) // 16
MULTIBLK = True  # fixed config: decode is always multi-block; the L=1 single-token path
# (attn_qk_p1/attn_kv_p1) was removed. ATTN_L (=DECODE_GOLDEN_L) stays a real parameter
# (context length: 2048/2047 chatbot, 32 for the run_paris_gen gate).
# DECODE_ATTN_LL=1: link the attn_qk/attn_kv kernels as LLVM IR (.ll) instead of .o,
# so they can be llvm-linked+inlined INTO the core (kernels built alwaysinline via
# -DDECODE_INLINE_ATTN). This uses upstream mlir-aie's func-level inline-kernel API:
# the kernel func.func declaration carries link_with = "<name>.ll" together with
# link_with_mode = "merge", which aiecc's aie-assign-core-link-files pass routes
# into the core's link_merge_files -> llvm-link merges the alwaysinline body into
# the core module before opt/llc (no surviving func.call, no object link).
# air-to-aie copies the decl's discardable attrs onto the lowered AIE func.func,
# so setting link_with_mode here is all that is needed. Default .o = object-linked.
_ATTN_EXT = ".ll"  # fixed config: inline-attn merge-mode (.ll) is the only decode path
_ATTN_MERGE = _ATTN_EXT == ".ll"  # emit link_with_mode="merge" for the inline path


def _set_attn_link(op, base):
    """Attach the kernel link_with (+ link_with_mode="merge" for the .ll inline path)."""
    op.attributes["link_with"] = StringAttr.get(base + _ATTN_EXT)
    if _ATTN_MERGE:
        op.attributes["link_with_mode"] = StringAttr.get("merge")


# The attention block loop is a compile-time ATTN_ROUNDS (=ceil(ATTN_MAXL/16)) loop; the kernel
# masks/skips blocks beyond the runtime RTP-L so one ATTN_MAXL build serves every L. That loop is
# single-buffered: air-label-scf-for-to-ping-pong declines it because the running max and
# accumulator are live across blocks and the score buffer is shared with the kv tile.
# DECODE_RB_ROUNDS overrides the shim KV-readback nd-DMA outer block count (default ATTN_ROUNDS).
# Used to (a) locate the readback-count word in insts.bin by diffing two builds, and (b) let the
# host patch it to ceil(L/16) per token so the shim pushes exactly what the runtime core consumes.
RB_ROUNDS = int(_os.environ.get("DECODE_RB_ROUNDS", str((ATTN_L_BLK + 15) // 16)))
# DECODE_DYNSEQ=1: take the context length as a runtime scalar instead of baking it
# in. It becomes a launch operand that drives BOTH the shim readback's block count
# and the attention herd's RTP-L, so the shim pushes exactly what the cores consume
# at whatever L the host dispatches -- one build serving every context, with the KV
# traffic of the actual context rather than of ATTN_MAXL. The instruction stream can
# no longer be a frozen insts.bin, so the build also emits a TXN builder the host
# calls per token (air.backend.txn_builder). Off by default: the staircase templates
# remain the shipping path until this is measured across all four decoders.
DYNSEQ = int(_os.environ.get("DECODE_DYNSEQ", "0"))
# The four bindings move together: the shim's push count, the memtile's dequeue
# count and the cores' trip count must agree, and the append has to land on the
# position the cores are about to read. Named separately only because each one
# reads better at its use.
DYNSEQ_RB = DYNSEQ_APPEND = DYNSEQ_RTP = DYNSEQ_MEM = bool(DYNSEQ)
# DECODE_MASK_MODE_RTP=1: carry the attention MASK MODE in the RTP-L scalar the
# host already writes, so one device program serves both DFlash passes.
#
# batch_attn_mask.py establishes that a per-query triangular mask is nothing but
# a per-query VALUE of L: token t of a block gets L+t keys (causal verify) or the
# whole block (bidirectional draft). Everything else about the two passes -- the
# tile program, the herds, the channels, the wave loop -- is identical, because
# the layer loop is rolled and the device is wave-invariant. So the ONLY thing
# standing between "one xclbin runs both DFlash passes" and "two xclbins with a
# PDI reload between them" is which of two integers the core adds per token.
#
# Rather than add a second RTP (a new launch operand threaded launch -> segment
# -> herd, and a second word for the instruction stream to patch), the mode
# rides in bit 30 of the L the host already sends. Real context lengths are
# under 2^30 by six orders of magnitude, so the bit is free:
#
#     L_rt = L                 causal, per-token step 1   (verify)
#     L_rt = L | 1<<30         bidirectional, step 0      (draft)
#
# L keeps ONE meaning in both modes -- token 0's context length -- so only the
# per-token mask value moves. The four other consumers of L (the shim readback
# count, the memtile dequeue count, the core trip count, the KV append slot) all
# already size themselves off the LAST token, L + BATCH - 1, which is exactly
# what every token sees in bidirectional mode. They need the mode bit stripped
# and nothing else.
#
# Decoded with divui/muli/subi only -- no andi, which this builder does not use
# anywhere else and which is not worth being the first to rely on.
MASK_MODE_RTP = bool(int(_os.environ.get("DECODE_MASK_MODE_RTP", "0")))
MASK_MODE_BIT = 1 << 30
# DECODE_MASK_BIDIR=1: bake the bidirectional mask in at BUILD time instead of
# selecting it per dispatch.
#
# The RTP form above needs DECODE_DYNSEQ, because that is the only configuration
# where L reaches the cores as a runtime scalar at all -- and DYNSEQ does not
# currently build at BATCH>1: `airrt.dma_memcpy_nd` on @inKV_K fails to legalize
# with a dynamic dim-1 length and a dynamic dim-3 offset together. That is a
# pre-existing gap (a build with MASK_MODE_RTP off fails identically), so it
# gates the runtime-selectable form but says nothing about whether the MASK is
# right. This knob separates those two questions: it exercises exactly the same
# core arithmetic with the step folded to a constant, so the numerics can be
# gated on hardware now and the RTP selection re-tested once DYNSEQ reaches
# batch 8.
MASK_BIDIR = bool(int(_os.environ.get("DECODE_MASK_BIDIR", "0")))


def _split_mask_mode(Lh):
    """(L_base, per_token_step) from an RTP-L that may carry the mode bit.

    The step comes back as a PYTHON int when it is known at build time -- 1 for
    the causal staircase, 0 for a baked bidirectional block -- and as an SSA
    value only under MASK_MODE_RTP. Callers branch on that so the default build
    emits exactly the arithmetic it emitted before: it has to stay
    byte-identical, because every gate in docs/DFlashFeasibility.md was measured
    against it.
    """
    if not MASK_MODE_RTP:
        return Lh, (0 if MASK_BIDIR else 1)
    # `i32` at the top of build() is function-local; this helper is called from
    # inside the builder but defined outside it, so it makes its own.
    _i32 = IntegerType.get_signless(32)
    _bit = arith.ConstantOp(IntegerAttr.get(_i32, MASK_MODE_BIT), None).result
    # Lh < 2*MASK_MODE_BIT, so the quotient is exactly the mode bit: 0 or 1.
    _bid = arith.divui(Lh, _bit)
    _base = arith.subi(Lh, arith.muli(_bid, _bit))
    _one = arith.ConstantOp(IntegerAttr.get(_i32, 1), None).result
    return _base, arith.subi(_one, _bid)


# DECODE_COALESCE=0: turn off the cross-wave shim-feed coalescing, for A/B.
COALESCE = int(_os.environ.get("DECODE_COALESCE", "1"))
# Core stack. At K=4096 (qwen3-8b) the seven K-wide L1 activation buffers leave
# under 8 KiB, so that geometry lowers it; every other model keeps 10240.
STACK_SIZE = int(_os.environ.get("DECODE_STACK", "10240"))
# DECODE_KV_SPLIT=1: decouple the attention K and V memtile rings (mirror the reference mem_3_1:
# separate k_mem_buffer / v_mem_buffer, filled by SEPARATE S2MM = inKV_K / inKV_V, so
# the qk core's K supply is NOT lock-chained to the kv core's V drain). Default off
# (shared per-CU [K|V] buffer, byte-identical baseline). Fixes the ~4.9->~2.5 us/block
# attention slope: our shared buffer serializes K/V drains + adds a backward qk<-kv edge
# that breaks the pipeline; the reference's independent K/V rings couple the cores only by score.
# Packs 2 CUs per group buffer (16x256) so it fits the same 4 KV shim channels.
KV_SPLIT = True  # fixed config: decoupled K/V memtile rings
# DDR KV-cache shapes (the reference full-faithful append+readback) for MULTIBLK. Per CU = 2 kv
# heads x DH=64 = 128 (one K or V region). All-CU region width DK_TOT_A; per-token
# K++V = KVSZ_TOK; cache padded to ATTN_MAXL = ATTN_ROUNDS*16 positions.
KV_PER_CU = MODEL["KV_PER_CU"]
DH_A = MODEL["DH_A"]
KVPC_DH = KV_PER_CU * DH_A  # 128
DK_TOT_A = N_ATTN_CU * KVPC_DH  # all-CU K (or V) width
KVSZ_TOK = 2 * DK_TOT_A  # per-token K ++ V (all heads)
ATTN_MAXL = ATTN_ROUNDS * 16  # padded context (compile-time block count)
APPEND_OFF = (ATTN_L - 1) * KVSZ_TOK  # this token's slot in the cache
# kvappend_bd.check_bounds, as a build-time property. The block writes positions
# ATTN_L-1 .. ATTN_L+BATCH-2, and overrunning does not fault: position ATTN_MAXL
# of one group's region IS position 0 of the next group's, so a block that
# crosses the end silently corrupts live KV for a real attention CU. Sizing
# ATTN_ROUNDS from ATTN_L_BLK makes this hold by construction; the assert is
# here so that stays true if it is ever sized from something else.
assert ATTN_L - 1 + BATCH <= ATTN_MAXL, (
    f"a block of {BATCH} at position {ATTN_L - 1} runs past the "
    f"{ATTN_MAXL}-position cache and would overwrite the next KV region"
)
# the reference-faithful on-device KV append: the rope core writes this token's roped-K/raw-V
# into the DDR cache (appendK/appendV S2MM -> KVC at slot L-1 = the reference _receive_kv_cache),
# then the whole cache is read back for the block-loop attention (the reference _move_kv_cache).
# The append->readback RAW on the shared cache is ordered in the runtime sequence by
# air-annotate-append-barrier, which derives it from the shared L3 memref (= the
# reference's dma_wait). Only for MULTIBLK (L>1); L=1 uses the trivial on-chip-KV path.
KV_APPEND = MULTIBLK
# the reference layer-chaining ABI: the layer output (res2 = new hidden states) is written
# IN-PLACE into arg0 (the hidden_states BO), so layer N's output == layer N+1's input
# in the same buffer -- matching the reference's decoding_layer (output S2MM back to x_arg_id,
# no separate output arg). Frees arg3 (== the reference's rope_rms slot).
# Reference 4-CU layout: attn cols 3,4 (CU0,1 col3 / CU2,3 col4), adjacent to q/o on
# mem_5_1 (col5). kv on mem_3_1/mem_4_1. (col4 freed by GLU->col5 relayout.)
ATTN_CU_LOC = [(3, 2, 3), (3, 4, 5), (4, 2, 3), (4, 4, 5)][:N_ATTN_CU]
# The columns the attention CUs sit in. Both reserve their memtile's MM2S 0 for
# the q-broadcast transit, so anything else landing on one of them has to start
# at channel 1 (see the transposer's air.memtile_dma_channel_min).
_ATTN_COLS = {loc[0] for loc in ATTN_CU_LOC}
# Group CUs by column. rope k/v fans to one packet channel PER COLUMN (the reference
# routes k/v as per-destination packets). A single channel feeding memtiles on 2 cols
# deadlocks: its FIFO interleaves the 2 cols' gets, so one col blocks on the other's
# puts. Per-col channels keep each col's k/v puts contiguous/in-order.
ATTN_COL_GROUPS = []  # [(col, [cu_idx,...]), ...] in CU order
for _c, _loc in enumerate(ATTN_CU_LOC):
    if ATTN_COL_GROUPS and ATTN_COL_GROUPS[-1][0] == _loc[0]:
        ATTN_COL_GROUPS[-1][1].append(_c)
    else:
        ATTN_COL_GROUPS.append((_loc[0], [_c]))
ATTN_CU_GROUP = {c: gi for gi, (_, cus) in enumerate(ATTN_COL_GROUPS) for c in cus}
# DECODE_KV_REGION=1 (requires KV_SPLIT): store the DDR KV cache REGION-MAJOR
# (quadrant layout, = the reference _receive_kv_cache / _move_kv_cache) instead of per-token
# interleaved [tok][K|V]. Regions per layer, each ATTN_MAXL*REGION_W contiguous:
#   [ K_grp0 | K_grp1 | ... | V_grp0 | V_grp1 | ... ]  (== the reference K03,K47,V03,V47).
# Per-token per-group width REGION_W = len(cus)*KVPC_DH (=256 for 2-CU groups).
# WHY: the interleaved layout makes a K-only (or V-only) readback STRIDED-WITH-HOLES
# (skip the token's V bytes) -> non-coalescible -> ~1 shim task/token (~4100 @L2k).
# Region-major makes each group's K (resp V) a single CONTIGUOUS span, so the whole
# readback collapses to 4 contiguous coalesced BDs streamed concurrently on the 2
# inKV_K/inKV_V channels -- exactly the reference's 4 npu_dma_memcpy_nd. The append instead
# scatters this token's K/V into the group regions (constant few strided writes/token).
KV_REGION = True  # fixed config: region-major DDR KV quadrants + fire-and-free readback (50 tok/s)
NGRP = len(ATTN_COL_GROUPS)
# Uniform group width (all groups same #CUs in the reference 4-CU/2-group layout).
REGION_W = len(ATTN_COL_GROUPS[0][1]) * KVPC_DH  # 256
REGION_STRIDE = ATTN_MAXL * REGION_W  # per-group region span (one K or V region)


def _kreg_off(gi):
    return gi * REGION_STRIDE  # base of group gi's K region (within a layer slab)


def _vreg_off(gi):
    return (NGRP + gi) * REGION_STRIDE  # base of group gi's V region


NPH = MODEL["NPH"]
I2P = MODEL["I2P"]  # row-pair iters per phase
J2P = MODEL["J2P"]  # col-block pairs (2*J2 = NBJ = K/COL_BLOCK)
# Which egress consumer each proj phase sends to, BY NAME. This used to be a
# list of packet ids that the kernel hardcoded too; the ids are now allocated by
# air-annotate-packet-ids, so all the design states is where a phase's output
# goes. Repeats are meaningful -- down shares o-proj's consumer.
#
# The ordinal a name maps to is its position in first-appearance order, which is
# also the @outY broadcast index the receiving gets sit at. Naming the consumer and
# deriving the index keeps the routing number out of the source entirely.
DEST_NAMES = MODEL["DEST"]
DEMUX = list(dict.fromkeys(DEST_NAMES))  # ordinal order: ["rope", "rms", "glu"]
NDEST = len(DEMUX)
DEST = [DEMUX.index(d) for d in DEST_NAMES]  # phase -> ordinal
DOWN_PHASE = NPH - 1
NBJ_PH = [2 * J2P[p] for p in range(NPH)]  # per-phase col-blocks: [8,8,8,32]
KPH = [NBJ_PH[p] * COL_BLOCK for p in range(NPH)]  # per-phase K: [2048,2048,2048,8192]

# Output wire layout (reproducer y_0_2_0 memref<80>, group 258, main 514).
HDR = 2  # wire header words (the compiler stores the routing id at elem 14)
PAIR_ROWS = MODEL["PAIR_ROWS"]  # 2 = lead/partner shared-L1 pairing (FLM llama)
PAIR_PAY = PAIR_ROWS * ROW_BLOCK  # 64
GRP_ROWS = HDR + LEADS_PER_GRP * PAIR_PAY  # 258
MAIN_ROWS = GRP_ROWS + (N_GRP - 1) * LEADS_PER_GRP * PAIR_PAY  # 514
PAYLOAD = N_PAIRS * PAIR_PAY  # 512 payload elems per round (16 rows)

# A 512-bit vector move, in bf16 elements. Compute-tile buffers are packed end
# to end by mlir-aie and only 32-byte aligned on AIE2p, so any buffer whose SIZE
# is not a multiple of this misaligns whatever the allocator puts after it --
# and a misaligned 512-bit access does not fault on AIE2, it silently shifts.
# See ypair_mm_l1 for the one that bit, and l1_align.py for the gate.
L1_VEC_BF16 = 32  # 64 bytes

# ===== LM-head (IS_ATTN=0) vocab projection =====================================
# the reference-faithful LM head: an RTP-guarded MODE of the SAME proj cores + rms core on
# the SAME xclbin (mirrors the reference llama: lm_head = layer_app_manager->create_app(),
# gen_lm_head_seq). IS_ATTN[0] RTP (our _arm herd operand): ==1 -> the 4-phase
# decode layer; ==0 -> a single vocab projection phase. The vocab GEMV is
# structurally the QKV phase (same proj_qmm_acc256, K=MODEL_DIM) with I2 scaled to
# cover VOCAB_SIZE_PADDED rows, emitting on the RMS_DEST id (=id4, pkt_id_to_rms_norm)
# -- the exact route the reference proj kernel uses -- so NO new proj-side flow. The
# rms core (mode 0) does final rmsnorm(x)->feed proj X, then forwards the vocab
# chunks it gets back (id4) out to shim as logits (see rms_residual.cc:211).
VOCAB_SIZE = MODEL["VOCAB_SIZE"]  # llama-3.2-1b (models/llama3.2-1b.h)
MODEL_DIM = K  # 2048
# FULL vocab (host side): the whole LM-head output.
VOCAB_SIZE_PADDED_FULL = (
    (VOCAB_SIZE + MODEL_DIM - 1) // MODEL_DIM
) * MODEL_DIM  # 129024
VOCAB_FULL_ROWBLKS = VOCAB_SIZE_PADDED_FULL // ROW_BLOCK  # 4032
# DEVICE CHUNK: the LM head is computed in N_VOCAB_CHUNKS separate dispatches on ONE
# persistent chunk-sized xclbin (mirrors the reference's gen_lm_head_seq re-dispatch). A single
# full-vocab dispatch is NOT buildable: 8064 launch inW puts kill air-to-aie, and a
# per-round drain exhausts shim BD IDs. VOCAB_I2 = per-dispatch row-pair iters/core;
# keep it small so the feed op-count + shim BDs + refeed lock all fit.
#
# The value is NOT free -- four constraints fix the legal set, and within it we want
# the LARGEST chunk, because UNI_LM = 126/VOCAB_I2 is the number of lm-head waves and
# every wave is a host-armed barrier (the herd lock + RTP re-dispatch that gate all 27
# cores). Measured cost of a wave: ~20 us, from a constant-work sweep at
# (VOCAB_I2,UNI_LM) = (18,7)/(14,9)/(6,21) -> 18.13/18.18/18.40 ms/token at ctx 2k.
#
#   1. UNI_LM * VOCAB_I2 == VOCAB_FULL_ROWBLKS/ROW_BLOCK == 126   (covers the vocab;
#      so every legal pair streams the identical 164 MB -- only the barrier count moves)
#   2. VOCAB_I2 divides 126        (the assert below: a chunk must divide the vocab)
#   3. VOCAB_I2 even               (the vocab relay drains whole-K blocks, so
#      K/PAYLOAD = 4 must divide VOCAB_RNDS = VOCAB_I2*PAIR_ROWS; odd values
#      floor-truncate the round count -> DEADLOCK. See the 3B entry's note.)
#   4. 2*VOCAB_I2 <= 63            (HISTORICAL: held when the whole VOCAB_RNDS xnorm
#      count sat in ONE producer credit lock and the AIE-ML lock is 7-bit, max +63;
#      larger made AcquireGreaterEqual(N) unsatisfiable -> DEADLOCK. The re-broadcast
#      now carries only K/PAYLOAD=4 per outer trip, so the credit no longer scales
#      with VOCAB_I2. Kept as the tested envelope -- larger I2 is untested on device.)
#
# Even divisors of 126 are {2,6,14,18,42,126}; (4) rules out 42 and 126, leaving
# {2,6,14,18} -> UNI_LM {63,21,9,7}. 18 is therefore the largest legal chunk and 7 the
# minimum wave count. 18 -> RNDS 36, 576 rowblocks/chunk, 4032/576 = 7 dispatches.
#
# Was 14/9. 18/7 is numerically IDENTICAL (the 64-token greedy id sequence at ctx 2k
# hashes the same) and ~0.05 ms/token faster -- consistent with 2 waves x ~20 us, but
# that is BELOW run-to-run noise at n=4, so treat the win as principled rather than
# measured. The wave-cost slope itself is only resolvable over the wider 23->37 range.
VOCAB_I2 = int(_os.environ.get("VOCAB_CHUNK_I2", "18"))
VOCAB_ROWBLKS = VOCAB_I2 * (NCX * NCY) * PAIR_ROWS  # rowblocks per chunk/dispatch
VOCAB_SIZE_PADDED = VOCAB_ROWBLKS * ROW_BLOCK  # logits per chunk (device drain size)
assert VOCAB_FULL_ROWBLKS % VOCAB_ROWBLKS == 0, "chunk must divide the full vocab"
N_VOCAB_CHUNKS = VOCAB_FULL_ROWBLKS // VOCAB_ROWBLKS  # host dispatches (9)
VOCAB_J2 = J2P[0]  # 4 (K=MODEL_DIM=2048 -> NBJ=8 col-blocks)
# ---- proj b_col_reduce_add cache (see kernels/proj_qmm.cc proj_qmm_acc256_c) ----
# The +min-term reduction of the activation depends only on the col-block j, not
# on the row-block i, so it is computed on the first row-block of a projection
# and reused. PROJ_RC_CACHE=0 restores the recompute-every-block path
# (proj_qmm_acc256) byte-identically, for A/B.
PROJ_RC_CACHE = int(_os.environ.get("PROJ_RC_CACHE", "1"))
# Debug bisect: keep the cache plumbing (buffer, slot index, extra args) but fill
# EVERY row-block, i.e. compute exactly what the uncached path computes. Correct
# output here isolates a broken reuse assumption; wrong output isolates broken
# plumbing. Costs the full recompute, so it is a diagnostic only.
PROJ_RC_FILL_ALL = int(_os.environ.get("PROJ_RC_FILL_ALL", "0"))
# One slot of COL_BLOCK/32 bf16 per col-block, sized for the WIDEST projection
# (2*J2 col-blocks; llama-1B down-proj K=8192 -> 32 -> 256 bf16 = 512 B/core).
# Same size as the reference's b_col_reduce_add[INTERMEDIATE_SIZE/GROUP_SIZE].
RCACHE_LEN = 2 * max(J2P + [VOCAB_J2]) * (COL_BLOCK // 32)
VOCAB_RNDS = (
    VOCAB_I2 * PAIR_ROWS
)  # egress PAYLOAD-rounds per chunk (VOCAB_SIZE_PADDED/512)
VOCAB_W_BLOCKS = VOCAB_ROWBLKS * NBJ  # packed q4k blocks per chunk (vocab weights)
# per-col vocab weight-fan blocks (matches PER_COL_PH form: nbi_pc*NCY*nbj); the
# launch feeds per_col_v//NCY fan-steps of wstep, cx-unrolled (count-free relay).
VOCAB_PER_COL = VOCAB_I2 * PAIR_ROWS * NCY * NBJ  # blocks/col per chunk
# LM_HEAD=1 builds the vocab-mode sequence (IS_ATTN=0); default 0 = decode.
LM_HEAD = int(_os.environ.get("LM_HEAD", "0"))


# ===== UNIFIED single-launch decode+lm_head (one PDI, no multi-launch) =====
# UNIFIED=1: ONE air.launch in for_(0, UNI_DEC+UNI_LM); per-wave arm =
# (iv<UNI_DEC)?1:0 drives the herds' on-core index_switch AND a launch-scope
# index_switch selecting decode vs vocab host feeds. Concatenated args for the
# first folding test (separate ELF args come after folding is proven).
UNIFIED = 1  # fixed config: single-launch unified decode + lm_head
UNI_DEC = int(_os.environ.get("UNI_DEC_OVERRIDE", MODEL["UNI_DEC"]))  # decode waves
# lm-head waves in the unified sequence. Overridable so the wave count can be varied
# while the LM-head WORK is held constant: UNI_LM * VOCAB_I2 == VOCAB_FULL_ROWBLKS/32
# (=126) always, so (UNI_LM=9,VOCAB_I2=14) and (UNI_LM=21,VOCAB_I2=6) stream the exact
# same 164 MB of vocab weights -- only the number of host-armed barriers differs. Used
# to measure what a wave barrier costs; must stay consistent with N_VOCAB_CHUNKS.
UNI_LM = int(_os.environ.get("UNI_LM", MODEL["UNI_LM"]))
assert UNI_LM == N_VOCAB_CHUNKS, (
    f"UNI_LM={UNI_LM} must equal N_VOCAB_CHUNKS={N_VOCAB_CHUNKS} "
    f"(VOCAB_CHUNK_I2={VOCAB_I2}); their product covers the padded vocab"
)
# EXTRA WAVES: launch iterations that are neither a decode layer nor an LM-head
# chunk. A third arm, on the same terms as the vocab one -- ONE phase, scalars
# only (see the proj core's `_sel`: an index_switch over the DATAFLOW would push
# the tile past 16 BDs, so an arm may only ever select I2/J2/dest, never a
# channel op). Their weights come from their OWN BO, appended after the existing
# args exactly as W_SPLIT's groups are, so x/w/rms/y/kvc binding positions do
# not move.
#
# Deliberately generic: the caller says what shape each wave is and where its
# slab starts, and nothing here knows what the waves compute. The DFlash
# pre-pass is the first user (programming_examples/llms/qwen3_4b_q4nx/
# dflash_prepass_waves.py builds this list); it is a projection like any other,
# and it is not this file's business that it belongs to a different model.
#
#   DECODE_EXTRA_WAVES='[{"name":"fc","i2":5,"j2":25,"iter_lo":0,"w_off":0,
#                         "x_src":"taps","dest":"rms"}, ...]'
#
# EMPTY BY DEFAULT, and the emitted IR must be byte-identical when it is empty.
_EXTRA_WAVES_JSON = _os.environ.get("DECODE_EXTRA_WAVES", "")
EXTRA_WAVES = _json.loads(_EXTRA_WAVES_JSON) if _EXTRA_WAVES_JSON else []
_EXTRA_KEYS = {"name", "i2", "j2", "iter_lo", "w_off", "x_slot", "x_stride", "dest"}
for _w in EXTRA_WAVES:
    if set(_w) != _EXTRA_KEYS:
        raise SystemExit(
            f"DECODE_EXTRA_WAVES entry {_w.get('name', '?')} has keys "
            f"{sorted(_w)}; expected {sorted(_EXTRA_KEYS)}"
        )
    if _w["x_slot"] < 0 or _w["x_stride"] < 1:
        raise SystemExit(
            f"extra wave {_w['name']}: x_slot must be >= 0 and x_stride >= 1"
        )
    if _w["dest"] not in DEST_NAMES:
        raise SystemExit(
            f"extra wave {_w['name']}: dest {_w['dest']!r} is not one of the "
            f"model's demux destinations {sorted(set(DEST_NAMES))}"
        )
N_EXTRA = len(EXTRA_WAVES)
# Per-wave geometry, in the same units the decode phases use: a wave covers
# I2 row-block iterations of 2*J2 column-blocks over all NCX*NCY cores, its
# column share is that divided by NCX, and the feed walks it NCY blocks at a
# time. `w_off` is where the wave's slab starts in the extra BO, which the
# caller chose -- this only derives the extent so the memref can be sized.
# A wave's contraction width, from its own column-block count -- so the X feed
# and the weight feed cannot disagree about how wide the wave is.
EXTRA_K = [2 * w["j2"] * COL_BLOCK for w in EXTRA_WAVES]
EXTRA_BLOCKS = [w["i2"] * 2 * w["j2"] * NCX * NCY for w in EXTRA_WAVES]
EXTRA_PER_COL = [b // NCX for b in EXTRA_BLOCKS]
EXTRA_NSTEPS = [p // NCY for p in EXTRA_PER_COL]
EXTRA_W_ELEMS = max(
    (w["w_off"] + b * BLOCK_BF16 for w, b in zip(EXTRA_WAVES, EXTRA_BLOCKS)),
    default=0,
)
# An extra wave's X is `n` slots of the X buffer, starting at `x_slot` and
# stepping `x_stride` slots -- one slot when the wave contracts over K, several
# when it contracts over a concatenation of them (the DFlash pre-pass's fc reads
# five hidden-state taps as one 12800-wide row). `n` is derived, so a wave
# cannot claim an X that is not the width its weights expect.
EXTRA_X_NSLOT = []
for _i, _w in enumerate(EXTRA_WAVES):
    if EXTRA_PER_COL[_i] % NCY:
        raise SystemExit(
            f"extra wave {_w['name']}: {EXTRA_BLOCKS[_i]} blocks is "
            f"{EXTRA_PER_COL[_i]} per column, which {NCY} rows do not divide"
        )
    if EXTRA_K[_i] % K:
        raise SystemExit(
            f"extra wave {_w['name']}: K={EXTRA_K[_i]} is not a whole number of "
            f"{K}-wide X slots"
        )
    EXTRA_X_NSLOT.append(EXTRA_K[_i] // K)
# The three scalars an arm hands the proj cores, and the egress round count --
# all in the units the decode phases already use, so an extra wave is a phase
# whose numbers happen to come from a dict:
#
#   ROUNDS_PER_PH[p] = I2P[p]*PAIR_ROWS = NBI_PH[p]/(NCX*NCY)
#
# and `i2` in a wave descriptor is that same row-block-iteration count (the
# packer's, `NCX*NCY` blocks per step). So the wave's egress rounds ARE its i2,
# and the core's own loop trip count -- which emits PAIR_ROWS blocks per step --
# is i2/PAIR_ROWS. The two differ only on a paired model; keeping them separate
# is what stops a PAIR_ROWS=2 build from silently running the wave twice.
EXTRA_RNDS = [w["i2"] for w in EXTRA_WAVES]
# How many @outY rounds the RMS CORE takes from a wave -- which is its egress
# only when the wave is addressed to it. A wave whose demux id is rope or glu
# egresses to that core instead, and a residual1 that still waited for i2 rounds
# would wait forever.
EXTRA_RES1_RNDS = [w["i2"] if w["dest"] == "rms" else 0 for w in EXTRA_WAVES]
EXTRA_J2 = [w["j2"] for w in EXTRA_WAVES]
EXTRA_I2 = []
for _w in EXTRA_WAVES:
    if _w["i2"] % PAIR_ROWS:
        raise SystemExit(
            f"extra wave {_w['name']}: i2={_w['i2']} row-block iterations is not "
            f"a whole number of PAIR_ROWS={PAIR_ROWS} core steps"
        )
    EXTRA_I2.append(_w["i2"] // PAIR_ROWS)
# The packet destination the outA put names -- an index into DEMUX, exactly as
# DEST[ph] is, so air-annotate-packet-ids allocates the same id the decode
# phases targeting that consumer get.
EXTRA_DEST = [DEMUX.index(_w["dest"]) for _w in EXTRA_WAVES]
# The arms an EXISTING consumer must sit out. Every arm-keyed switch in the
# design reads "case 0 -> idle (the LM head produces nothing for me), default ->
# decode", so an extra wave -- arm 2+k -- lands in the DECODE branch of all of
# them and waits for rounds it never produces. Measured as a dispatch timeout on
# the first fc wave to reach the device. Each such switch takes this list instead
# of a bare [0], with the same idle body: a COUNT (zero) and not a program, and
# byte-identical when there are no extra waves.
ARM_IDLE_CASES = [0] + [2 + k for k in range(N_EXTRA)]
# ONLY dest "rms" IS WIRED, and the guard below is the whole reason this is a
# generic hook rather than a DFlash feature. An extra wave gets its arm, its
# weight feed, its proj scalars, its egress and its X (the rms core forwards the
# tap, see _uni_extra) -- and what makes an `rms` dest need nothing else is that
# the rms core's residual1 already takes the wave's rounds (fc's egress is
# exactly OPROJ_RNDS, so that pass needs no count change at all) and @layerOut
# already drains the result. What lands there is `tap + W.tap/rms(tap)`, because
# the accumulate adds into the same buffer the X arrived in; the host subtracts
# the tap it wrote and multiplies rms(tap) back. Exact, and it needs no kernel
# mode that does not already exist.
#
# Every other dest would need its consumer to grow a body for the extra arm --
# a `rope` dest, say, needs k_norm, RoPE and appendK/appendV -- and on the arms
# it does not have one, that core sits in ARM_IDLE_CASES and the egress backs up.
# The DFlash pre-pass wanted a `rope` dest for its context K/V and does not use
# one: the projection comes back on @layerOut like fc's does, and the host --
# which is already writing that KV cache -- applies k_norm and RoPE to the
# [layers, batch, 1024] result in microseconds. Weight bandwidth is what an
# extra wave exists to buy, and the rope arm buys none of it.
EXTRA_PARTIAL = int(_os.environ.get("DECODE_EXTRA_WAVES_PARTIAL", "0"))
_EXTRA_UNWIRED = sorted({_w["dest"] for _w in EXTRA_WAVES} - {"rms"})
if _EXTRA_UNWIRED and not EXTRA_PARTIAL:
    raise SystemExit(
        f"DECODE_EXTRA_WAVES has waves with dest {_EXTRA_UNWIRED}, whose consumer "
        f"has no extra-arm body -- that core sits idle on arm 2+k and the wave's "
        f"egress backs up. Only dest 'rms' is wired "
        f"(docs/DFlashFeasibility.md section 3.15). Set "
        f"DECODE_EXTRA_WAVES_PARTIAL=1 to emit IR for inspection anyway."
    )
UNI_WAVES = UNI_DEC + UNI_LM + N_EXTRA
# Wave-range override (keeps ABI/CDO fixed at UNI_DEC/UNI_LM; only restricts which
# waves the fused launch loop drives). Used to split the fused sequence into a
# decode-part [0,UNI_DEC) and a vocab-part [UNI_DEC,UNI_WAVES) that share ONE CDO,
# to test host-wait quiescence between decode and vocab on one xclbin.
UNI_WAVE_LO = int(_os.environ.get("UNI_WAVE_LO", "0"))
UNI_WAVE_HI = int(_os.environ.get("UNI_WAVE_HI", str(UNI_WAVES)))
# DECODE_NO_LM_WAVES: stop the batched sequence after the decode layers. This is
# what the batched build did unconditionally before the LM head was wired, and it
# is still the right thing when only the decode engine is under test -- the
# batch-equivalence gates read layer outputs, not logits, and skipping seven vocab
# waves is most of the build and most of the run.
#
# It is NOT a fallback for a broken LM head. Left at UNI_WAVES with no vocab arm on
# the rms core, the failure is a deadlock at the FIRST vocab wave and nowhere
# earlier: all UNI_DEC decode layers run, every layer's output lands, every layer's
# KV appends, and then the rms core starts a decode pass into a chip whose every
# other tile has taken its vocab arm and gone idle. It looks like a batching fault
# and it is not one.
NO_LM_WAVES = bool(BATCH > 1 and int(_os.environ.get("DECODE_NO_LM_WAVES", "0")))
if NO_LM_WAVES:
    UNI_WAVE_HI = min(UNI_WAVE_HI, UNI_DEC)

# Weight-buffer grouping: G decode layers per weight BO. A shim BD's byte offset
# is a uint32 (aiex.npu.address_patch $arg_plus -> uint32_t in AIETargetNPU), so
# ONE buffer can only be addressed over a 4 GiB span; qwen3-8b's 36 layers are
# 4.04 GiB of layer weights alone and wrap. G splits the weights over
# ceil(UNI_DEC/G) buffers plus a dedicated lm-head buffer, each addressed from
# its own base -- the same layer-invariance FLM gets by binding one BO per layer,
# except we keep ONE dispatch (our runtime sequence is unrolled, so each wave's
# BDs can name a different arg). G<=0 or G>=UNI_DEC => single buffer, and the
# emitted IR is byte-identical to before this knob existed.
W_GROUP = int(_os.environ.get("DECODE_WGROUP", "0"))
W_SPLIT = 0 < W_GROUP < UNI_DEC
N_WGRP = ((UNI_DEC + W_GROUP - 1) // W_GROUP) if W_SPLIT else 1
# The split keys off the wave induction variable, which only exists in the fused
# single-launch form.
assert not W_SPLIT or UNIFIED, "DECODE_WGROUP needs the unified wave loop"

ROUNDS_PER_PH = [I2P[p] * PAIR_ROWS for p in range(NPH)]  # y0,y1 per v1 -> 2*I2
N_ROUNDS = sum(ROUNDS_PER_PH)  # total egress rounds (phase0 6 + phase1 4 = 10)
# id-demux egress: the main MT MM2S emits each round's assembled packet carrying
# the destination the put names; the switchbox routes the id allocated for dest p
# (reproducer mem_1_1 DMA5: id1->tile_2_3, id4->tile_2_2). Rounds per dest =
# sum of its phases' rounds (here 1:1 phase<->id so [6, 4]).
ROUNDS_PER_DEST = [
    sum(ROUNDS_PER_PH[ph] for ph in range(NPH) if DEST[ph] == p) for p in range(NDEST)
]

# Per-phase weight slab dims (phase0 QKV 96 row-blocks, phase1 o-proj 64). Same K
# -> same NBJ. The weight memtile fan + X memtile are PHASE-AGNOSTIC flat streams
# (reproducer: one continuous w_buffer/x_buffer ring); only the compute cores
# carry phase structure (index_switch). So the runtime concatenates the phase
# weight slabs and the fan/refeed loops just run the summed total step count.
NBI_PH = [I2P[p] * PAIR_ROWS * NCX * NCY for p in range(NPH)]  # [96, 64, 512, 64]
PER_COL_PH = [(NBI_PH[p] // NCX) * NBJ_PH[p] for p in range(NPH)]  # per-phase NBJ
W_FAN_STEPS = sum(pc // NCY for pc in PER_COL_PH)  # per col, all phases
W_TOTAL_BLOCKS = sum(NBI_PH[p] * NBJ_PH[p] for p in range(NPH))  # packed q4k blocks
if W_DUAL_CHAN:
    # The split is spatial (each shim channel owns half the column's cores), so
    # the only requirement is an even core count per column.
    assert NCY % 2 == 0, f"W_DUAL_CHAN needs an even NCY (got {NCY})"
# X 256-blocks the cores consume across all phases: per core per phase = I2*2
# row-blocks, each reading NBJ_PH[p] 256-blocks of that phase's K. The X memtile
# relays this many (matched put/get sizes -> balanced count-free ring, no deadlock).
N_XBLK = sum(I2P[p] * PAIR_ROWS * NBJ_PH[p] for p in range(NPH))

# X re-feed: per phase the cores read the full K once per output row-block; a core
# emits I2*2 row-blocks per phase, so it reads K that many times. With
# every downstream count (XN_REFEED, OPROJ_REFEED, GATEUP_REFEED, DOWN_REFEED,
# _xc_dec, and the ph1/ph3 memtile relays' own refeed counts) follows from this
# one line.
REFEED = [I2P[p] * PAIR_ROWS for p in range(NPH)]  # [6, 4, 32, 4]
REFEED_TOTAL = sum(REFEED)  # all phases
# Two X sources: phases 0..2 read the rmsnorm'd token X (K=2048); the DOWN phase
# reads the GLU output (K=8192) fed back on-chip. Split the re-feed accordingly.
RMS_PHASES = [p for p in range(NPH) if p != DOWN_PHASE]
RMS_REFEED = sum(REFEED[p] for p in RMS_PHASES)  # rms-X whole-2048 re-reads (42)
DOWN_REFEED = REFEED[DOWN_PHASE] if DOWN_PHASE >= 0 else 0  # GLU-X 8192 re-reads (4)
# LOOPCLOSE: ph1 (o-proj) X = attn-o (separate channel), so @xnorm/rms-X covers only
# ph0 + ph2 (ph1 excluded). OPROJ_PHASE=1. XN_REFEED = REFEED[0]+REFEED[2].
OPROJ_PHASE = 1
GATEUP_PHASE = 2
# LOOPCLOSE convergent @xnorm: rms (compute, channel refeed) emits ONLY ph0
# (rmsnorm input); ph1 attn-o, ph2 a_xn, ph3 down are MEMTILE producers (mechanism-2
# per-buffer refeed) converging on @xnorm in phase-time order, read by ONE loop.
XN_REFEED = REFEED[0]
OPROJ_REFEED = REFEED[OPROJ_PHASE]  # ph1 attn-o re-feeds (4)
GATEUP_REFEED = REFEED[GATEUP_PHASE]  # ph2 X re-feeds (32)

# GLU (gate-up id8) -- FAITHFUL: the strip demux delivers gate-up DIRECTLY to the
# GLU compute tile (reproducer packet_flow(8) keep=false: mem_1_1 DMA5 -> tile_5_2
# DMA0; NO relay). The GLU x buffer is 1024 = TWO stripped demux packets (512 each)
# = [up 512 | gate 512]; glu_aie -> silu(gate)*up -> 512. 16 slices -> 8192.
# The gate-up phase (ph2 of the 4-phase proj) is what feeds the GLU herd; its id is
# whatever DEST assigns to that phase. Derive it rather than hardcoding a value --
# the ids are routing labels, not semantics, and mlir-aie #3429 (exact subcube cover)
# removed the constraint that they be one-hot.
GLU_PHASE = 2 if NPH == 4 else -1
GLU_DEST = DEST[GLU_PHASE] if GLU_PHASE >= 0 else -1
# #4 faithful residual stream: o-proj + down (shared id4 -> RMS_DEST) are CONSUMED by
# the rms core (residual1=input+o-proj -> h; residual2=h+down -> layer output), NOT
# drained via the deadlocking memtile relay. The down egresses as the layer output.
# #4 applies only to the full 4-phase proj (QKV, o-proj, gate-up, down) where o-proj
# (ph1) + down (ph3) share id4 -> consumed by the rms residual.
# Structural, not value-based: what #4 needs is the 4-phase proj where o-proj (ph1)
# and down (ph3) SHARE a destination (their common id -> RMS_DEST) while QKV (ph0)
# and gate-up (ph2) each get their own -- i.e. 4 phases over exactly 3 destinations.
# Matching on the literal [1,4,8,4] silently disabled #4 on any valid id relabel.
FULL4 = NPH == 4 and DOWN_PHASE == 3 and DEST[1] == DEST[3] and NDEST == 3
RMS_DEST = DEST[DOWN_PHASE] if FULL4 else -1
HOST_DRAIN = [p for p in range(NDEST) if p != GLU_DEST and p != RMS_DEST]
# Where the VOCAB projection egresses. At batch 1 it shares the o-proj/down id and
# the rms core relays it to the shim, which is what the reference decode superkernel
# does (x broadcast on the rms tile's MM2S0, logits on its MM2S1). Batched, that core
# cannot take a second job: it already holds a BATCH*K residual and spends both MM2S
# on @xnorm and @layerOut. So the batched vocab arm stamps the GATE-UP id instead and
# the logits land on the GLU core, whose vocab arm is otherwise empty.
#
# The GLU core rather than a memtile, and the reason is arithmetic rather than taste.
# The obvious relay is dest 0's memtile -- idle in vocab, already dest 0's endpoint
# for the batched QKV transpose -- but at batch 8 that memtile (col 3) has FIVE MM2S
# in use and reserves the sixth (MM2S 0 carries the q-broadcast's route), so there is
# no channel to drain on and air-to-aie says so: "failed to get MM2S tile for L3
# allocation". The main egress memtile is worse -- all six MM2S spent. The GLU core
# has one MM2S free, its column's shim is completely unused, and it is ONE HOP from
# the demux instead of two.
#
# Nothing on the decode path moves: the id is stamped from the arm-selected `pktv`,
# so decode still stamps its own per-phase ids, and the vocab get on the GLU core is
# the byte-identical descriptor to the decode one (see _voc there).
VOCAB_DEST = GLU_DEST if BATCH > 1 else DEST[OPROJ_PHASE]
# A slice pair is one rotation of the GLU BD ring, and the core's slot sequence is
# statically unrolled and restarts every layer while the ring's rotation carries
# over -- an ODD slice count leaves the two a slot apart and every other layer reads
# a stale slice. Two demux packets per slice is the reference granularity, but when
# that gives an odd count (qwen2.5-7b: INTERMEDIATE/512 = 37) drop to one packet per
# slice: same 512-row interleave stride, half the slice, an even count, and the same
# total. The kernel is generic in the slice length (pseduo_glu<GLU_SLICE>).
GLU_PKTS = 2 if (ROUNDS_PER_DEST[GLU_DEST] // 2) % 2 == 0 else 1
GLU_CHUNK = GLU_PKTS * PAYLOAD // 2  # gate-up interleaves up/gate in chunks this tall
GLU_SLICE = GLU_PKTS * PAYLOAD  # 1024 = [up 512 | gate 512] (GLU_PKTS demux packets/BD)
GLU_HID = GLU_SLICE // 2  # 512 out per 1024 slice
NGLU = ROUNDS_PER_DEST[GLU_DEST] // GLU_PKTS if GLU_DEST >= 0 else 0  # 16 slices
GLU_OUT = NGLU * GLU_HID  # 8192 (INTERMEDIATE) = down_buffer size = down K
GLU_PCOL = 5  # GLU compute tile + down memtile column (reference: tile_5_x + mem_5_1;
# moved 4->5 to free col4 for 4-CU attention, matching the reference layout)
# DOWN phase: the GLU output (8192) is fed back on-chip as the down X (NOT host).
# down_buffer re-broadcasts its 8192 DOWN_REFEED(=4) times to the X memtile, which
# chunks each into 16x512 -> inX for ph3. No gluShim host drain.
# The batched LM head rides the GLU core's gate-up BD, so it drains in the same
# granule: GLU_SLICE per token per get, two of them per ring rotation. Sharing the
# BD is the point -- an index_switch puts BOTH arms' channel ops in the tile's mem
# block, so two arms with DIFFERENT descriptors would put two BDs on one S2MM chain
# and the port would cycle through both. Identical descriptors collapse to one.
VOC_SLICE_RNDS = GLU_SLICE // PAYLOAD  # egress rounds per GLU slice (2 on llama)
VOC_ROTATIONS = VOCAB_RNDS // (2 * VOC_SLICE_RNDS) if BATCH > 1 else 0
if BATCH > 1 and VOC_ROTATIONS * 2 * VOC_SLICE_RNDS != VOCAB_RNDS:
    raise SystemExit(
        f"DECODE_BATCH>1 with the LM head needs VOCAB_RNDS={VOCAB_RNDS} to be a "
        f"multiple of {2 * VOC_SLICE_RNDS} (two GLU slices of "
        f"{VOC_SLICE_RNDS} rounds): the batched logit drain shares the GLU core's "
        f"ping-pong BD ring, and an odd number of slices desyncs it the same way "
        f"an odd NGLU does. Pick a VOCAB_CHUNK_I2 that divides -- VOCAB_RNDS is "
        f"VOCAB_CHUNK_I2*{PAIR_ROWS}."
    )
HOST_ROUNDS = sum(ROUNDS_PER_DEST[p] for p in HOST_DRAIN)  # host-drained egress rounds
# #4: the down egresses as the rms layer output (residual2), drained on its own channel.
LAYER_RNDS = (PAIR_ROWS * I2P[DOWN_PHASE]) if FULL4 else 0

# The probe taps (see DECODE_PROBE): each size is the whole buffer that tile already
# holds, so a tap is one contiguous put of something already resident.
#
# Laid AFTER the region the normal drains use, so a probe build's Y offsets do not
# move any shipping descriptor -- and so `y: nothing written` keeps meaning what it
# meant. Absolute, not per layer: with more than one wave the later layers just
# overwrite, and the question the probe answers is about the FIRST one.
PROBE_LEN, PROBE_OFF = {}, {}
_poff = (HOST_ROUNDS + LAYER_RNDS) * PAYLOAD * BATCH
for _pb, _pn, _plen in (
    (PROBE_Q, "Q", BATCH * DQ_PADDED),
    (PROBE_O, "O", BATCH * DQ),
    (PROBE_D, "D", BATCH * GLU_OUT),
):
    PROBE_LEN[_pn] = _plen if PROBE & _pb else 0
    PROBE_OFF[_pn] = _poff
    _poff += PROBE_LEN[_pn]
PROBE_TOTAL = sum(PROBE_LEN.values())
# Q and O share the mem_tile_5_1 channel, so the shim drains them as ONE task at
# Q's offset -- which is O's offset when Q is off.
PROBE_5_LEN = PROBE_LEN["Q"] + PROBE_LEN["O"]

# DECODE_HIDDEN_TAPS=1: keep every layer's hidden state instead of overwriting it.
#
# The layer chaining ABI writes residual2 (= h + down, the layer's hidden state,
# exactly LAYER_RNDS*PAYLOAD == K elements) back into the X buffer at offset 0,
# and the next layer's rmsX reads offset 0 -- so layer iv+1 destroys layer iv's
# output the moment it lands. Speculative decoding with a DFlash-style drafter
# needs the states at a handful of chosen layers (Qwen3-4B-DFlash-b16 fuses
# target_layer_ids [1, 9, 17, 25, 33]), and there is no way to read them back.
#
# The fix is an offset, not a new drain: give layer iv the read slot iv and the
# write slot iv+1. Same transfers, same BD count, same instruction stream -- the
# bytes already cross the shim. The only cost is the X buffer growing from one
# slot to UNI_DEC+1 (185 KB for qwen3-4b, 68-296 KB across the models).
#
# The write->read dependency is unchanged in kind: layer iv+1 still reads what
# layer iv wrote, just at a different address in the same BO, so the existing
# air.preserve_shim_dma_order program ordering still carries it.
#
# 0 (default) keeps the in-place chain and is a strict no-op: X_SLOTS == 1 and
# both offsets fold back to the literal 0.
HIDDEN_TAPS = int(_os.environ.get("DECODE_HIDDEN_TAPS", "0"))

# ===== Multi-layer fused decode (stitch NLAYERS runtime sub-sequences) =====
# The device (segment/herds) is emitted ONCE; only the launch-scope L3 feeds are
# emitted per layer, with COMPILE-TIME-CONSTANT per-layer DDR offsets. So the
# aie.device (-> xclbin) stays byte-identical to the single-layer build and only
# the runtime instruction sequence grows ("16 sub-sequences stitched one after
# another"). NLAYERS=1 is a strict no-op (all per-layer bases = 0).
NLAYERS = int(_os.environ.get("NLAYERS", "1"))

# ===== DFlash batched decode (DECODE_BATCH tokens per superkernel call) =======
# The superkernel does one token per call. DFlash needs it to do a block of
# them: draft proposes DECODE_BATCH tokens, verify checks all of them in one
# pass. See docs/DFlashFeasibility.md.
#
# 1 (default) is a STRICT NO-OP -- every derived quantity below folds back to
# the literal it replaces, and the emitted IR is byte-identical to HEAD. That
# is gated, not asserted: see the DECODE_HIDDEN_TAPS diff recipe in the doc,
# which covers this flag too.
#
# WHY 8 IS THE DEFAULT CEILING AND NOT 16. The block size was chosen by pricing
# each pass as max(compute, memory) with attention counted -- 8 wins at 1.65x
# where 16 gives 1.06x. It is also the batched matmul's fastest shape by a wide
# margin (71.4 MAC/cycle against batch 16's 55.7), because q4k_mmul_small's 1x4
# blocking at rowA=1 fits where aie::mmul<4,8,8> does not.
# (BATCH itself is read near the top -- ATTN_ROUNDS needs it.)

# Per-core L1 ceilings, MEASURED by batch_l1_budget.py on qwen3-4b (the DFlash
# target) against the 64 KB tile minus DECODE_STACK. These are hard limits, not
# guidance: exceeding one does not produce a slow build, it produces an opaque
# aiecc failure naming no buffer. Fail here instead, naming the buffer.
#
#   proj core        31.3 KB at batch 8, fits to 25. The 16 KB unpacked-weight
#                    scratch is independent of the batch (it holds one WEIGHT
#                    block), so it sets a batch-independent floor.
#   attention CU     51.0 KB at query tile 8, fits to 8 EXACTLY. No headroom:
#                    any future per-CU L1 addition on this path has to be paid
#                    for out of the query tile.
# ---- @xnorm chunk width -----------------------------------------------------
# How much of a token's row travels in ONE @xnorm transfer, in COL_BLOCKs. This
# is a TRANSFER-COUNT knob and nothing else: the bytes, the @inX multicast and
# every descriptor below COL_BLOCK are unchanged, because the X memtile still
# splits whatever it gets into COL_BLOCK-wide broadcasts. What moves is how many
# times the producer and the memtile hand off per layer.
#
# It exists to separate two explanations of why a batched layer costs 3.65x its
# weight-streaming floor with no arithmetic in it (see decode_cost.py and
# "The batched layer is dataflow bound" in docs/DFlashFeasibility.md). At batch
# 1 the core normalizes once and refeed() collapses the re-broadcast into a lock
# init; at batch > 1 the normalized batch cannot be materialized, so
# _rms_batched_norm regenerates a chunk inside the loop and puts it through a
# SINGLE staging buffer -- 152 serialized round trips per layer against none.
# Doubling this halves that count at constant bytes. If the layer tracks the
# count it is handshake-bound; if it does not, it is not.
#
# 2 is what the design has always used and is the default; at 2 the emitted IR
# is byte-identical, which check_batch1_noop.py holds.
XCHUNK_MUL = int(_os.environ.get("XCHUNK_MUL", "2"))
XCHUNK = XCHUNK_MUL * COL_BLOCK
assert K % XCHUNK == 0, f"XCHUNK {XCHUNK} does not divide K {K}"

# ---- proj core input ring depth ---------------------------------------------
# How many physical L1 instances of the batched proj core's xblk/wblk buffers
# exist. 2 (the default) is what `air-label-scf-for-to-ping-pong` produces
# automatically from a single alloc-per-iteration loop -- see PROJ_MM_BATCH's
# _mm below, whose depth-2 path is exactly that pattern, untouched.
#
# Exists to test whether ring depth is why the batched layer runs at 3.65x its
# weight-streaming floor with the arithmetic deleted (PROJ_DELAY, see
# docs/DFlashFeasibility.md "The pipeline has almost no slack"): even a small
# injected per-block delay is 25-45% exposed in wall time, which is the
# signature of thin buffering rather than one buffer with real spare capacity.
# This is the direct test -- deepen the ring, re-run the same PROJ_DELAY sweep,
# and see whether the exposed fraction drops.
#
# AIR hardens the labeling pass's unroll factor at 2
# (AIRDependencyScheduleOpt.cpp: "Unroll factor hardened as 2. TODO: add
# support for an arbitrary factor"), but `isPingPongCandidate` opens with
# `if (forOp->hasAttr("unroll")) return false;` -- an already-labeled loop is
# left alone. So depth > 2 pre-tags the ORIGINAL, unchanged alloc-per-iteration
# loop with `unroll` = N before the standard pipeline runs (see
# `_mm_ring_deeper`), and the SEPARATE downstream transform
# (AIRPingPongTransformationPattern) reads that attribute generically -- this
# borrows its own tested duplication/rotation machinery rather than replacing
# it. Two designs that instead gave N buffers a persistent hoisted lifetime
# (a loop-carried iter_arg phase, then a remui-selected index_switch) were
# tried first and both failed identically in air-opt's `air-dependency` pass
# ("operand #0 does not dominate this use"): a buffer written from inside a
# runtime-selected branch has no single last-writer for the pass to hang a
# token on. Pre-tagging the untouched loop sidesteps the question entirely.
#
# 2 is the default and reproduces the exact prior IR (check_batch1_noop and the
# depth-2 disassembly are both unaffected by this knob's existence).
# RMS_MEMTILE_REFEED: move ph0/ph2's X re-broadcast off the rms CORE and onto
# the X MEMTILE, so the core computes each normed chunk ONCE instead of once per
# refeed sweep.
#
# WHY IT IS NEEDED ONLY AT BATCH > 1. At batch 1 the core holds the whole normed
# X (BATCH*K bf16) and refeed() re-sends that one resident buffer -- no
# recomputation at all. At batch 8 that buffer is 40 KB and the core is already
# at 59424 of 59440 B holding the RAW x (which the residual still needs), so the
# chunked path regenerates: `for r in nrefeed: for c in nchunk: rms_chunk`. That
# is 12*5 chunks for ph0 and 38*5 for ph2 -- 250 a layer for 10 distinct ones,
# and rms_chunk is 30.8 ms of a 134 ms dispatch (measured by deletion,
# RMS_CHUNK_PROBE=2).
#
# The memtile has 512 KB, so it can hold what the core cannot. ph1 and ph3
# already work this way ("mechanism-2": a memtile L2 buffer plus
# refeed(n, _xnorm_put(buf, width))); this gives ph0/ph2 the same treatment.
#
# IT STILL DOES NOT BUILD, BUT NOT FOR THE REASON THIS COMMENT USED TO GIVE.
#
# It used to die in air-to-aie with 'aie.memtile_dma' op has more than 48
# blocks, at =1, at =2, and at =2 with the gather buffer relocated -- and the
# conclusion drawn was that the blocks go to the @inX PUT side and the X memtile
# has no room for a second feed structure. Half of that was right and the wall
# was not the blocks.
#
# The nest in _feed_inX_rf was three deep (nrefeed x nchunk x XCHUNK_MUL)
# because folding XCHUNK_MUL into the descriptor was believed impossible --
# "consecutive windows sit 32 elements apart ... so they interleave rather than
# partition". They partition: the windows are COL_BLOCK apart in ADDRESS, since
# offsets are multiplied by strides. xfeed_bd(BATCH, XCHUNK, 1) is the folded
# descriptor and emits a byte-for-byte identical stream (enumerate both address
# lists; they match in order, not just as sets). With that level gone the
# 48-block error is GONE at =1 and =2.
#
# WHAT IT HITS NOW, and this is the real constraint:
#     'aie.dma_bd' op Allocator exhausted available BD IDs (maximum 24 for
#     channel 0)
# mem_tile_2_1 channel 0 carries 27. Not blocks -- BD IDs, and they are 27
# because that ONE channel has SIX aie.dma_start(MM2S, 0) chains on it, one per
# phase that got its own put structure. A DMA channel has one BD chain
# (check_dma_alloc.py's docstring records a deadlock from exactly two on one
# channel). Every other memtile channel in the design sits at 2-4 BDs, so
# RMS_MEMTILE_COL cannot help: 27 does not fit in 24 anywhere.
#
# So "_feed_inX is ONE loop and the arms differ only in the trip count" IS the
# binding rule, and it is about chains per channel rather than blocks per tile.
#
# THE ROUTE OUT, for whoever picks this up: do not ADD a refeed structure --
# make the gather-then-rebroadcast the ONLY @inX path, for every phase and arm,
# with nrefeed as the trip count (1 where the phase does not refeed). That keeps
# one dma_start on the channel and puts this back inside the rule instead of
# fighting it. It costs X one extra L2 hop, which is ~1% of the traffic the
# regeneration is burning. It is a real rework of the most deadlock-prone part
# of this design, which is why this stops here.
#
# Kept because the ~29 ms is real (rms_chunk is 30.8 ms by deletion and 96% of
# its calls are redundant) and because three measured dead ends are cheaper to
# read than to rediscover. Inert at the default, and check_channel_balance plus
# _chanbal.py both show the emitted IR differs on @xnorm and nothing else.
# Silently inert at batch 1 rather than an error: tools that sweep both batches
# for comparison (check_channel_balance.py builds batch 1 to scale against)
# would otherwise die on the knob rather than on anything real.
# `int(...) and BATCH > 1` collapsed EVERY non-zero value to True, i.e. 1 --
# Python's `and` returns its right operand when the left is truthy. So mode 2
# never ran: every "=2" result recorded above was mode 1 wearing a 2. Keep the
# value.
RMS_MEMTILE_REFEED = int(_os.environ.get("RMS_MEMTILE_REFEED", "0")) if BATCH > 1 else 0
# Modes that convert BOTH rms phases (ph0 and ph2). 2 converts ph2 alone.
# 3 is 1 plus the row-major producer: the rms core ships whole rows and computes
# each value once, and the memtile does the cross-row interleave in its read
# descriptor. See docs/BZeroPlan.md.
_MTR_ALL = RMS_MEMTILE_REFEED in (1, 3)
# Which channel ph0/ph2's X leaves the rms core on. Mode 3 relays through a
# memtile that re-broadcasts onto @xnorm; every other mode puts on @xnorm direct.
#
# The replay count is a lock value baked into a BD and every arm shares one BD
# chain, so the counts cannot simply be each arm's own total (XN_REFEED +
# REFEED[GATEUP_PHASE] on decode, VOCAB_RNDS on the LM head, EXTRA_NREFEED on an
# extra wave). Each arm instead walks a common cycle a whole number of times --
# see _RF_CYCLE, which is where that constraint is discharged and asserted.
#
# WHERE MODE 3'S TWO RELAY SLABS LIVE.
#
# Both on ONE memtile, and it has to be a memtile that is otherwise free,
# because three separate ceilings stack up and none of them is reported at
# build time -- every violation compiles clean and times out on device with
# nothing written. All three are check_memtile_chans.py-visible in
# air_project/npu.air.mlir.
#
#  1. SIX MM2S CHANNELS PER MEMTILE, and col 3 already uses all six (ph3's
#     gluOut relay, the KV slab, two weight pairs). A seventh producer there
#     WRAPS: AIRToAIESchedulingUtils' allocator is
#     `chan = minCh + (num_allocs % (6 - minCh))`, so it emits two
#     `aie.dma_start(MM2S, 1, ...)` chains on one channel and the refeed chain
#     falls through to `aie.end` after two BDs instead of self-looping.
#  2. ONE REFEED RELAY PER PHYSICAL MM2S CHANNEL. The repeat IS the BD's
#     self-loop, so a chain carrying two slabs alternates them -- ph0, ph2, ph0
#     -- while the consumer needs all of ph0 first. Hence two channel names,
#     @xin0 and @xin2, each marked air.dedicated_dma_channel so the allocator
#     does not collapse them back onto one; and two names also mean two packet
#     ids, which the ROUTER requires (two flows sharing an id from two ports of
#     one tile fail with "Unable to find a legal routing").
#  3. THE X MEMTILE (col 2) CANNOT HOST A RELAY. @xnorm and @inX are both packet
#     flows and the MM2S allocator collapses onto any existing packet-flow
#     allocation on the tile, so @inX's put lands on whatever channel the relay
#     took. `air.memtile_dma_channel_min` does not help -- the collapse decision
#     comes before channel selection.
#
# Census: cols 0/1/6/7 have every S2MM taken, col 3 every MM2S, col 5 is ph1's,
# col 3 is ph3's, col 2 is out. Col 4 is the only memtile left -- and it is
# enough for BOTH slabs, because they share one inbound chain (a 2-BD cycle,
# slab0 then slab2, which is the phase order) and need only one MM2S each.
RMSROW_COL = int(_os.environ.get("RMSROW_COL", "4"))
# Mode 3 converts BOTH rms phases. It has to: ph0's chunk-puts and ph2's
# row-puts are different descriptors, so leaving one of them unconverted would
# need two BDs on the core's single @xnorm chain, and they would alternate.
_RMSROW0 = _RMSROW2 = "xnorm"
# Which channel ph1's attnO relay and ph3's gluOut relay drain onto. In mode 3
# all four phases ride ONE channel, @xin, in phase-time order: ph0 and ph2 from
# the relay slab on col RMSROW_COL, ph1 from col 5, ph3 from col 3.
#
# ONE channel because the consumer is ONE flat _feed_inX loop, and two
# _feed_inX calls in one arm collapse into a single chain that cycles both
# channels' buffers -- so it waits on a ph2 buffer three ph0 puts in. One
# consumer loop therefore means one inbound channel, one packet id, and every
# producer on a DISTINCT TILE (two packet flows sharing an id from two ports of
# one tile do not route).
#
# That used to cost four tiles, because ph0 and ph2 needed different re-send
# counts and a count was a property of the BUFFER -- two counts, two buffers,
# two ports, two ids. air.refeed_count is now per FILL, so they share one slab
# and one port and the tile count is THREE: col RMSROW_COL, col 5, col 3.
# Three convergent producers on one channel is exactly what @xnorm does today.
_XN1 = "xin" if RMS_MEMTILE_REFEED == 3 else "xnorm"
_XN3 = _XN1
# Which memtile column holds the gather buffer; see _feed_inX_rf.
RMS_MEMTILE_COL = int(_os.environ.get("RMS_MEMTILE_COL", "3"))
# How many X sweeps the RMS CORE emits per phase, per arm. Without the knob the
# core does the re-broadcasting itself, so this IS the refeed count; with it the
# memtile re-broadcasts and the core emits exactly one sweep (or none, where the
# phase does not run on that arm).
# RMS_MEMTILE_REFEED and the extra waves do not compose; the refusal lives with
# _RF_PH0_EX below, where EXTRA_NREFEED exists and the message can name the
# numbers instead of asserting a bare incompatibility.

PROJ_RING_DEPTH = int(_os.environ.get("PROJ_RING_DEPTH", "2"))
assert PROJ_RING_DEPTH in (2, 3), "only 2 (default) and 3 (the experiment) exist"

# PROJ_WS_NO_SINK: keep the batched proj core's 16 KB unpacked-weight scratch
# OUT of the j loop, with `air.disable_alloc_sink`.
#
# THE DEPTH-2 RING DOES NOT EXIST TODAY, and the comment above PROJ_RING_DEPTH
# saying it is "what air-label-scf-for-to-ping-pong produces automatically" is
# wrong for this loop. Read from the per-pass IR dumps of a qwen3-4b batch-8
# mode-3 build (FUSED_DECODE_DEBUG_IR=1):
#
#   pass_031 air-fuse-alloc-dealloc    SINKS the scratch into the j loop
#                                      (allocs in body: [2048, 2560] ->
#                                       [8192, 2048, 2560], at that pass and
#                                       no other)
#   pass_033 air-label-scf-for-to-     rejects the loop. isUnsafeToDuplicate
#            ping-pong                 wants every body alloc dead on entry;
#                                      the scratch's first access is the
#                                      func.call, and an opaque callee may
#                                      read. 14 loops are labeled in this
#                                      design and this is not one of them.
#   lowered core_0_2                   ONE x buffer, ONE w buffer, one
#                                      mm_acc call: acquire / compute /
#                                      release, no prefetch.
#
# So the proj core cannot fetch the next weight block while it computes the
# current one, which is what makes the arithmetic ADD to the weight stream
# instead of hiding under it -- docs/BZeroPlan.md's item 2, 9.091 = 5.856 +
# 3.235 on the vocab probe path, and its open question ("why a core's compute
# overlaps another core's compute but not its own weight fetch").
#
# The attribute suppresses the sink only. The scratch stays SINGLE and the two
# channel.get-filled buffers get the ping-pong, which is the shape _wscr_alloc's
# docstring already describes and could not previously get: +9 KB of L1, not
# +25 KB. Default 0 so the shipping emit is byte-identical.
#
# It also retires the PROJ_RING_DEPTH=3 datapoint. NEITHER setting produces a
# ring: a depth-3 build's lowered tile_0_2 is one x, one w, one mm_acc call and
# 3 aie.dma_bd, the same shape as the default, because _mm_ring_deeper pre-tags
# `unroll` and isPingPongCandidate returns on its first line. So 8.692 vs 9.091
# is depth 1 against depth 1 and the 0.4 ms is noise, not a 12% recovery.
#
# AND WITH THE HOIST IT HANGS `[hw]`. batch_equiv.py --smoke at L=128, one
# dispatch: default COMPLETED, PROJ_RING_DEPTH=3 COMPLETED, this knob TIMEOUT,
# this knob + depth 3 TIMEOUT, this knob + UNI_WAVE_HI=1 TIMEOUT. Every build
# that RUNS has depth 1 and the only builds with a real ring hang; one decode
# wave reproduces it. The whole-device diff against the working build is 16 proj
# tiles gaining a second x and w slot and 32 lock inits going 1 -> 2 -- no flow,
# producer BD or lock anywhere else moves. So it is dynamic, and the standing
# (untested) suspicion is @inX: a 16-way circuit broadcast where a 2-deep
# consumer ring lets the cores drift a buffer apart. See docs/BZeroPlan.md.
PROJ_WS_NO_SINK = int(_os.environ.get("PROJ_WS_NO_SINK", "0"))

# PROJ_PP_ONLY: with PROJ_WS_NO_SINK=1, ping-pong only ONE of the proj core's
# two inputs -- "x", "w", or "" (default: both, which is what hangs).
#
# The named one is left in the j loop and gets its ring; the OTHER is hoisted
# out with air.disable_alloc_sink, so the transform has nothing to duplicate
# for it. Hoisting is behaviour-preserving by itself -- the get refills one
# buffer per trip either way, which is exactly what the shipping build does to
# both -- so this is a clean A/B on which ring is the one the device rejects.
PROJ_PP_ONLY = _os.environ.get("PROJ_PP_ONLY", "")
assert PROJ_PP_ONLY in ("", "x", "w"), "PROJ_PP_ONLY is x, w, or unset"


def _pp_hoist(ty, which):
    """Allocate `which` OUTSIDE the j loop, pinned there, or None to leave it in.

    Returns None unless PROJ_PP_ONLY names the OTHER input -- i.e. this one is
    the one being taken out of the ring.
    """
    if not PROJ_PP_ONLY or PROJ_PP_ONLY == which:
        return None
    _a = AllocOp(ty, [], [])
    _a.operation.attributes["air.disable_alloc_sink"] = UnitAttr.get()
    return _a


# RMS_BAND_STREAM: the band-streamed residual (docs/DFlashFeasibility.md, "So
# the residual has to leave L1"), built in levels so each is gatable and
# separately verifiable. 0 (default) is byte-identical to HEAD -- see
# check_batch1_noop.py's batch-8 extension.
#
#   1: rms_scale_row_aie's one full-row reduction becomes a band loop of
#      rms_scale_row_partial_aie + one rms_scale_row_finalize_aie. xb is STILL
#      the full resident [BATCH][K] buffer (row_stride=K) -- this validates
#      the two-pass kernel in isolation. Verified on device (spec_accept.py,
#      llama-3.2-1b batch 8): identical acceptance to level 0, no hang.
#   2: level 1, plus the INITIAL rmsX read (ph0's raw x) is banded too --
#      nband ChannelGets landing into xb's band columns instead of one whole
#      get, matched by a Python-unrolled (never scf.for -- see _feed_wcols'
#      comment on why a launch-scope for_ deadlocks the shim) sequence of
#      nband ChannelPuts at launch scope. xb is STILL full-size; this
#      validates the launch<->compute banded-transfer plumbing in isolation
#      from the harder part.
#   3: xb shrinks to one STG_W-wide band, refetched fresh (from _x_in(),
#      in place -- no scratch slot, see the HIDDEN_TAPS refusal below) at
#      each of six phases: ph0 scale pre-pass, ph0 regen, residual1, ph2
#      scale pre-pass, ph2 regen, residual2. residual1/residual2 round-trip
#      their band through DDR via layerOut (write) then rmsX (re-read) --
#      the SAME memref (X) the existing cross-layer chaining already
#      round-trips through, just at band grain and mid-layer instead of
#      between layers. This is the part that actually moves the L1 ceiling.
#      VERIFIED ON DEVICE, qwen3-4b DECODE_BATCH=8 (docs/DFlashFeasibility.md
#      section 5.4 (x)): batch_equiv 3.86e-03 at one layer and 5.56e-03 at
#      36, the same output levels 0 and 2 produce, against a 5e-2 gate. It
#      costs 4.4% -- 166.302 -> 173.624 ms, dispatch_time.py median of 25 --
#      for streaming the whole residual through DDR twice a layer, so it is
#      a viable shape in its own right and not only a batch-16 stepping
#      stone. BATCH_MAX_RMS is 16 with it on.
#
#      This comment used to say level 3 "does NOT get past air-to-aie"
#      ("operand #0 does not dominate this use" on the second @rmsX put).
#      That was true and was fixed -- see 5.4 (iii) RMS_W_ON_X, which takes
#      the norm weights off their own S2MM port so rmsX stops sharing one.
#      It also pointed at a doc heading that is now a section 9 entry
#      RETRACTING the claim. Level 3 at BATCH=1 is still not a supported
#      configuration (5.4 (xiii)): the decode arm's band feed gates on
#      RMS_BAND_STREAM >= 3 alone while the vocab arm also gates on
#      BATCH > 1, and a batch-1 level-3 build hangs with nothing written.
RMS_BAND_STREAM = int(_os.environ.get("RMS_BAND_STREAM", "0"))
assert RMS_BAND_STREAM in (0, 1, 2, 3), "level 3 is the highest level built"

# The X-feed tile-blocking descriptor, IMPORTED rather than restated:
# xfeed_bd.py checks it elementwise against pack_A, which q4k_mm_gate.py proved
# bit-exact on device. A copy here would be a second source of truth for the one
# permutation whose failure mode is "right number of elements, wrong ones".
if BATCH > 1:
    import xfeed_bd as _xfd

    _XFEED_BD = _xfd.xfeed_bd(BATCH, COL_BLOCK, XCHUNK_MUL)
else:
    _XFEED_BD = None

# Egress at batch B: one packet per round, B times longer. GRP_ROWS and
# MAIN_ROWS are HDR + <payload>*B; N_ROUNDS, the BD count and the instruction
# stream are all unchanged, which is what "widen, do not repeat" means.
#
# AND THE ASSEMBLED PACKET STAYS EMITTER-MAJOR. The consumers want token-major,
# so something has to transpose, and the obvious place is the group / main
# gathers -- which is where this was built first, and it deadlocked. The reason
# is the 2-word routing HEADER. It rides once at the front of a packet, not once
# per token, so a token-major landing needs one descriptor for the header and
# another for the grid; and a second get on the same channel endpoint eats the
# memtile's ping-pong ring (visible in the emitted AIE dialect as a BD chain
# alternating header/body on ONE buffer while every other channel alternates
# ping/pong). No single strided BD can do both, because a BD walks its source
# linearly and the header is a 2-element prefix in front of a 64-element grid.
#
# So the transpose moves to the CONSUMERS, where the header is already gone:
# each of them lands a round with a 3-D de-interleave (see OUTY_TOKMAJOR) and
# every gather stays exactly the batch-1 descriptor with a B-times-longer
# payload. One BD per endpoint, headers untouched, rings intact.
GRP_ROWS_B = HDR + LEADS_PER_GRP * PAIR_PAY * BATCH
MAIN_ROWS_B = HDR + PAYLOAD * BATCH
PAYLOAD_B = PAYLOAD * BATCH
if BATCH > 1:
    import kvappend_bd as _kvb


def outy_tokmajor(row_stride, base=0, rounds=1):
    """(offsets, sizes, strides) landing `rounds` egress rounds token-major.

    A round arrives EMITTER-MAJOR -- emitter e's B token blocks back to back,
    for each of the N_PAIRS emitters -- because that is what keeps the packet
    header in one piece (see above). The consumer wants token t's row
    contiguous, so it walks (emitter, token, element):

        dim       extent      dst stride     what it is
        emitter   N_PAIRS     PAIR_PAY       this emitter's slice of a row
        token     BATCH       row_stride     next token's row
        element   PAIR_PAY    1              the slice itself

    Several rounds fold into the FIRST dimension rather than adding a fourth,
    which matters because a COMPUTE tile's BD has only three: round r, emitter e
    lands at r*PAYLOAD + e*PAIR_PAY, and PAYLOAD is N_PAIRS*PAIR_PAY by
    definition, so (r, e) is one index of extent N_PAIRS*rounds and stride
    PAIR_PAY. The GLU core needs that -- its slice is two rounds on llama -- and
    a second get on the same endpoint is not an option: it would eat the
    channel's ping-pong ring, which is the bug this whole layout avoids.

    `row_stride` is the width of the buffer's per-token row: one slice for a
    round buffer, the whole M or K for one that accumulates a token's rounds
    side by side.
    """
    return (
        [0, 0, base],
        [N_PAIRS * rounds, BATCH, PAIR_PAY],
        [PAIR_PAY, row_stride, 1],
    )


BATCH_MAX_PROJ = 25
BATCH_MAX_ATTN_QTILE = 8
if BATCH < 1:
    raise SystemExit(f"DECODE_BATCH must be >= 1, got {BATCH}")
if BATCH > 1 and LM_HEAD:
    # The FUSED lm head is batched and gated (batch_lm_equiv.py). This refuses
    # only the STANDALONE vocab-only build, which is a different launch shape --
    # no decode waves in front of it, `_seg_arm_i` unused on some paths -- and
    # has never been dispatched at batch > 1. Nothing is known to be wrong with
    # it; nothing is known to be right either, and the fused form is what the
    # host driver uses.
    raise SystemExit(
        "DECODE_BATCH>1 with LM_HEAD=1 (the standalone vocab-only build) is "
        "untested. The FUSED lm head IS batched -- build the unified sequence "
        "(LM_HEAD=0, which drives UNI_DEC decode waves then UNI_LM lm-head "
        "waves) and gate it with batch_lm_equiv.py."
    )
if BATCH > BATCH_MAX_PROJ:
    raise SystemExit(
        f"DECODE_BATCH={BATCH} exceeds the proj core's L1 ceiling of "
        f"{BATCH_MAX_PROJ}: xblk+yacc+ypair scale with the batch on top of a "
        f"fixed 16 KB unpacked-weight scratch (wscr). Re-check with "
        f"`batch_l1_budget.py --model {MODEL_NAME} --batch {BATCH} -v "
        f"--scratch-rows 32 --scratch-cols 256`."
    )

# rms / rope / glu do NOT get a whole block at once. qkv_l1 alone wants
# M*BATCH bf16 -- 96 KB for 8 tokens on qwen3-4b against a 54 KB budget -- so
# these phases run the block as ceil(BATCH/ROW_TILE) sub-tiles. ropeq_l1 (fits
# 6) and rms_l1 (fits 10) ride the same tiling. Derived from the widest of the
# three rather than hardcoded, so a model with different M does not silently
# overflow: whichever buffer is tightest sets the tile.
#
# The tile is a LOOP COUNT, not a correctness parameter -- these phases are
# row-independent (per-token rms, per-token rope, per-token glu), so any tile
# that fits gives the same answer.
# The per-herd RTP word is allocated on the tile too, AFTER every buffer, and
# nothing above accounts for it. It costs 4 bytes and it is the difference
# between a design that fits and `'aie.tile' op allocated buffers exceeded
# available memory` naming no buffer: at DECODE_BATCH=16 the rms core lands on
# 65,536 EXACTLY -- stack 6,080 + three 16 KB bands + two 5 KB norm weights + 64
# B of scales -- and __air_herd_rtp_2_2 is placed at 0x10000. Reserve it here so
# the ceiling is checked rather than discovered.
#
# 16, not 64: the allocator packs the RTP word immediately after the last buffer
# with no alignment padding (0xFFC0-0xFFFF then 0x10000-0x10003 in that map), so
# 4 is the true cost and this is a small guard on top. 64 is NOT free -- levels
# 0 and 2 put the rms core at 59,424 B at batch 8, 32 under the un-reserved
# ceiling, so a 64-byte reserve refuses a configuration that is verified on
# hardware.
_L1_RTP_RESERVE = 16
_L1_ROWTILE_BUDGET = 65536 - STACK_SIZE - _L1_RTP_RESERVE


def _tile_for(*widths):
    """Largest tile of rows of `widths` bf16 elements that fits one tile's L1.

    PER PHASE, not one global tile. rms holds K per row and fits 10 at block 8;
    rope holds M (and DQ_PADDED) and fits 4. Taking the min over all of them
    would tile rms at 4 as well -- correct, but twice the calls for no reason.
    """
    return max(1, min(BATCH, _L1_ROWTILE_BUDGET // (2 * max(widths))))


RMS_TILE = _tile_for(K)  # rms in/out/weight
ROPE_TILE = _tile_for(M, DQ_PADDED)  # qkv_l1 is the binding one
GLU_TILE = _tile_for(GLU_SLICE)

# ---- the rms core's batched residency -------------------------------------
# The rms core is the ONE place a row loop is not enough, and it is worth
# spelling out because the shape of the whole batched pass follows from it.
#
# Everything the core does is per row, so a row loop is the obvious answer. But
# the core must hold ALL B rows of two different things at once -- the raw
# batch (the residual stream: x, then h, then the layer output) and the
# normalized batch (what the projection reads) -- and at batch 8 each of those
# is BATCH*K bf16 = 32 KB against a budget of 54 KB. Two do not fit. Neither
# can be dropped: the projection re-reads the normalized batch REFEED[p] times
# per phase, and the raw one is what the post-attention output adds into.
#
# So the normalized batch is never materialized. rms_scale_row_aie keeps one
# float per row and rms_chunk_aie regenerates whichever @xnorm chunk is being
# sent, for all B rows, into a staging buffer the size of ONE chunk. The big
# buffer stays raw and accumulates in place. That is ONE BATCH*K buffer for the
# whole pass, and it is why the batched rms body looks nothing like the batch-1
# one rather than being it with a loop around it.
#
# STG_W is the wider of the two things that staging buffer carries: one @xnorm
# chunk out, or one projection round in (both [BATCH][w], token-major).
STG_W = max(XCHUNK, PAYLOAD)

# How many times the rms core re-emits its resident X row on @xnorm. The X
# memtile wants i2 * (EXTRA_K/XCHUNK) chunks and the core emits one per
# (refeed, chunk) visit over K/XCHUNK chunks, so this is what makes the two
# agree -- the same accounting the decode arm's XN_REFEED does.
EXTRA_NREFEED = []
for _i, _w in enumerate(EXTRA_WAVES):
    _ch = _w["i2"] * (EXTRA_K[_i] // XCHUNK)
    if _ch % (K // XCHUNK):
        raise SystemExit(
            f"extra wave {_w['name']}: {_ch} X chunks is not a whole number of "
            f"{K // XCHUNK}-chunk re-feeds"
        )
    EXTRA_NREFEED.append(_ch // (K // XCHUNK))

# ===== Mode 3's re-feed cycle: one BD chain every arm can traverse ==========
#
# A memtile relay's re-send count is a LOCK VALUE BAKED INTO A BD, and a BD
# chain is hardware: it advances one BD per completed transfer and knows
# nothing about which scf.index_switch arm is running. Everything else in this
# design that varies by arm varies as a TRIP COUNT IN A CORE'S PROGRAM (see
# _cnt), which branches; a BD cannot. So a dispatch has to leave the chain
# where it found it, on EVERY arm -- and if it does not, the failure is a wrong
# lock count on some later dispatch, with nothing wrong at compile time.
#
# Each arm's TOTAL re-sends are fixed by the model and cannot be traded:
#   decode      XN_REFEED (ph0) + REFEED[GATEUP_PHASE] (ph2)
#   LM head     VOCAB_RNDS (ph0; ph2 is a zero-trip loop there, no second
#               projection phase exists on that arm)
#   extra wave  its own EXTRA_NREFEED
# A cycle of sum S is traversed a whole number of times by every arm iff S
# divides all of them, so S is their gcd. The LARGEST such S is what we want:
# one fill is one core sweep, and sweeps are the cost this mode exists to cut.
#
# The cycle is [a, b] with a = XN_REFEED % S, so ph0 ends exactly on a cycle
# boundary and ph2 picks up at the next BD. a == 0 means one BD of S.
#
# qwen3-4b: decode 12+38 = 50, LM head 30 -> S = 10, cycle [2, 8].
#   decode ph0  2,8,2                    = 12  (3 fills)
#   decode ph2  8,2,8,2,8,2,8            = 38  (7 fills)   10 fills, 5 cycles
#   LM head ph0 2,8,2,8,2,8              = 30  (6 fills)    6 fills, 3 cycles
# Both land back on BD 0. Total re-sends per arm are UNCHANGED, so no consumer
# moves: only the number of times the core regenerates the batch does.
_PH0_RELAY = RMS_MEMTILE_REFEED in (1, 3)
_PH2_RELAY = RMS_MEMTILE_REFEED in (1, 2, 3)

# EVERY ARM CONSTRAINS THE CYCLE, INCLUDING ONES THIS BUILD WILL NOT DISPATCH.
# DECODE_NO_LM_WAVES only cuts the runtime sequence at UNI_DEC; the vocab arm's
# IR is still in the design. Dropping VOCAB_RNDS from the gcd there was tried
# and it HANGS the decode-only build on device -- so a narrowed wave range must
# not move the cycle, which is the same rule as "the xclbin is built at the full
# wave table and UNI_WAVE_LO/HI only narrow the stream against it".
# ONE SWEEP PER PHASE, AND THAT IS A CORRECTNESS CONSTRAINT, NOT A PREFERENCE.
#
# The cycle sum S decides how many times the rms core regenerates the batch per
# phase: S = XN_REFEED + REFEED[GATEUP_PHASE] gives one fill per phase, and any
# smaller S gives more. Measured, `dflash_build_diff.py` against the shipping
# build at batch 8:
#
#     one sweep per phase (S = 50)   BIT-IDENTICAL at 1, 4 and 36 layers,
#                                    layer output and the whole KV region
#     three and seven (S = 10)       exact at 1 layer, 10 elements wrong at 2,
#                                    ~55 and KV drifting at 4, 7e-4 at 36
#     five and fifteen (S = 5)       the SAME error as S = 10, not worse
#     one and three  (S = 25)        AS WRONG AS S = 10 (2.6e-4 .. 1.8e-3)
#
# So it is not the number of fills and not the chain -- doubling the fills does
# not move it -- and S = 25 puts ph0 at ONE fill while ph2 has three, which is
# already the full error. PH2 WITH MORE THAN ONE FILL IS NECESSARY AND
# SUFFICIENT.
#
# The undiluted signal is at TWO layers, not 36: ten elements, ALL IN TOKEN 0,
# all 1 ULP. One ULP in the layer output means the PROJECTION differed by less
# than a bf16 ULP -- a different f32 accumulation, not corrupted X -- and token
# 0 is the first row of every sweep, the one the core writes straight after the
# sweep loop's back-edge. By four layers it has reached the other tokens
# through the KV cache, which is all the 36-layer number is.
#
# Ruled out: the relay's BD chain and locks (the lowered aie.memtile_dma is
# exactly as designed -- write lock init 8, fills alternating acquire 8/release
# 2 and acquire 2/release 8, drain one self-looping BD at 1/1); _xc_voc; the
# arm-selected sweep bound (mode 0 ships with the same dynamic bound at 12 and
# 30); this core's lock ordering (mode 0 emits the identical call; acquire;
# release). Root cause not isolated; until it is, S stays at the bit-exact
# value. See docs/BZeroPlan.md item 1.
_RF_S = XN_REFEED + REFEED[GATEUP_PHASE]
# RF_CYCLE_S: force a SMALLER cycle sum than the gcd, i.e. more fills per layer
# for the same re-sends. Diagnostic only -- it must still divide the gcd or an
# arm cannot walk whole cycles, and the asserts below say so. It exists because
# "does the error scale with the number of fills" is the question that separates
# a memtile-chain fault from a core-side staging race, and it is one build.
#
# A value LARGER than the gcd is also allowed, and is diagnostic-only: the LM
# head arm then cannot walk whole cycles, so its sequence is rounded UP to the
# next cycle boundary and it would over-feed. Harmless in a DECODE_NO_LM_WAVES
# build, where that arm is never dispatched -- and it is the only way to build
# the ONE-SWEEP-PER-PHASE shape (S = XN_REFEED + REFEED[GATEUP_PHASE]) with the
# vocab arm's IR still present, which is what isolates "the core sweeps more
# than once" from "the chain has more than one fill".
_RF_S_ENV = int(_os.environ.get("RF_CYCLE_S", "0"))
if _RF_S_ENV:
    _RF_S = _RF_S_ENV
# RF_CYCLE: the cycle SPELLED OUT, e.g. "6,6,38". A cycle needs neither two
# entries nor the [a, S-a] split below -- all that is required is that every arm
# walks it whole. The derived form is the one that puts ph0 on a boundary;
# naming the entries directly is what lets a build isolate ph0's fill count from
# ph2's, which the derived form cannot do (there is no S that gives ph0 more
# than one fill and ph2 exactly one). Diagnostic; every assert below still
# applies, including the per-arm "leaves the chain mid-cycle" ones, so a cycle
# an arm cannot walk is refused rather than built.
_RF_CYCLE_ENV = [
    int(v) for v in _os.environ.get("RF_CYCLE", "").split(",") if v.strip()
]
if _RF_CYCLE_ENV:
    _RF_S = sum(_RF_CYCLE_ENV)
# Whether the LM head arm can walk this cycle a whole number of times. At the
# bit-exact S it cannot (VOCAB_RNDS=30 is not a multiple of 50), so its sequence
# is rounded UP to the next boundary -- which OVER-FEEDS that arm and is only
# safe where it is never dispatched. The guard below refuses to build otherwise.
_RF_VOC_APPROX = bool(VOCAB_RNDS % _RF_S)
if RMS_MEMTILE_REFEED == 3 and _RF_VOC_APPROX and not NO_LM_WAVES:
    raise SystemExit(
        f"RMS_MEMTILE_REFEED=3 needs DECODE_NO_LM_WAVES=1 for this model.\n"
        f"  The relay's re-send count is a lock value baked into a DMA BD and\n"
        f"  both index_switch arms share one BD chain, so every arm has to walk\n"
        f"  one common cycle a whole number of times. The decode arm needs\n"
        f"  {XN_REFEED}+{REFEED[GATEUP_PHASE]}={_RF_S} and the LM head needs\n"
        f"  {VOCAB_RNDS}, which do not share a cycle at one sweep per phase.\n"
        f"  A SMALLER cycle (gcd {_math.gcd(_RF_S, VOCAB_RNDS)}) makes them\n"
        f"  share one, and it is NUMERICALLY WRONG -- bit-exact at one decode\n"
        f"  layer, ten 1-ULP elements in token 0 at two, 7e-4 at 36. Narrowed\n"
        f"  to ph2 with more than one fill; root cause not isolated.\n"
        f"  RF_CYCLE_S= forces it anyway, for whoever picks that up.\n"
        f"  See docs/BZeroPlan.md, 'ONE SWEEP PER PHASE'."
    )
_RF_A = XN_REFEED % _RF_S
_RF_CYCLE = _RF_CYCLE_ENV or ([_RF_S] if _RF_A == 0 else [_RF_A, _RF_S - _RF_A])
assert (
    max(_RF_CYCLE) <= MAX_REFEED
), f"re-feed cycle {_RF_CYCLE} exceeds the {MAX_REFEED} lock ceiling"
# The write lock inits to the LARGEST count on the buffer, and the first fill
# in the cycle acquires the LAST one's (air-to-aie makes a fill wait for the
# previous fill's re-sends, not its own). Those two have to be the same number
# or the lock drifts above its init every cycle and a fill starts running ahead
# of the drain.
assert _RF_CYCLE[-1] == max(
    _RF_CYCLE
), f"re-feed cycle {_RF_CYCLE} must end on its largest count"


def _rf_seq(start, total):
    """Counts drawn cyclically from _RF_CYCLE at `start`, summing to `total`.

    Returns (counts, next_index). The caller asserts next_index == 0 at the end
    of an arm: that IS the alignment condition, and it is the only thing
    standing between this mode and a chain that drifts one BD per dispatch.
    """
    seq, i = [], start
    while sum(seq) < total:
        seq.append(_RF_CYCLE[i % len(_RF_CYCLE)])
        i += 1
    assert (
        sum(seq) == total
    ), f"re-feed cycle {_RF_CYCLE} cannot sum to {total} from BD {start}"
    return seq, i % len(_RF_CYCLE)


# The START index matters as much as the sequence now: it is the slab a phase
# begins on (see _rmsrow_relay), and the relay ping-pongs one slab per cycle
# position so that a fill boundary never leaves a transfer pending.
_RF_PH0_DEC_I0 = 0
_RF_PH0_DEC, _rf_i = _rf_seq(0, XN_REFEED)
_RF_PH2_DEC_I0 = _rf_i
_RF_PH2_DEC, _rf_i = _rf_seq(_rf_i, REFEED[GATEUP_PHASE])
assert _rf_i == 0, "decode arm leaves the re-feed chain mid-cycle"
if _RF_VOC_APPROX:
    # Diagnostic builds only (RF_CYCLE_S not dividing the gcd): walk whole
    # cycles until the LM head's re-sends are covered. It over-feeds, so this
    # is only valid where that arm is never dispatched.
    _rf_n = -(-VOCAB_RNDS // _RF_S) * _RF_S
    _RF_PH0_VOC, _rf_i = _rf_seq(0, _rf_n)
else:
    _RF_PH0_VOC, _rf_i = _rf_seq(0, VOCAB_RNDS)
assert _rf_i == 0, "LM head arm leaves the re-feed chain mid-cycle"
# AN EXTRA WAVE CANNOT WALK THE CYCLE, AND NO WAVE TABLE FIXES THAT.
#
# Only mode 3 replays a fill, so only mode 3 needs a sequence here -- and this
# loop is skipped otherwise, which is what keeps a FOLDED MODE-0 template
# buildable. It did not used to be: the walk ran unconditionally, so a build
# with a wave table and no re-feed knob at all died on the assert inside
# _rf_seq. The emitted-IR gate never saw it because that gate runs with
# DECODE_EXTRA_WAVES unset.
#
# Under mode 3 the refusal is structural. A wave's re-sends are
#
#     EXTRA_NREFEED = i2 * EXTRA_K / K = i2 * j2 / (K / (2*COL_BLOCK))
#
# so a wave that contracts over exactly K with i2 = 1 -- which is every wave
# the DFlash pre-pass emits -- sends its X ONCE and has nothing to re-broadcast.
# One does not divide the decode arm's 50, so the largest cycle all the arms
# share is 1, and a cycle of 1 replays each fill once: mode 0 by another name.
# Reshaping does not help either, because i2 and j2 trade off inside a fixed
# block count -- fc as a single wave is 25 whichever way it is split, and all
# five context-K/V layers merged is 20, for a gcd of 5.
#
# The way out is for an arm with nothing to re-broadcast to BYPASS the relay
# rather than walk its cycle, which needs the extra arm's X to reach the X
# memtile on _XN1 directly. Today it does not: the extra arms sit in
# ARM_IDLE_CASES and run the LM head's relay (_o_voc), while the consumer side
# (_xs_extra) reads "xnorm" -- the channel the relay itself owns.
_RF_PH0_EX = []
# MODE 3 STILL DOES NOT CARRY THE EXTRA WAVES. The build below is complete
# enough to compile and place, and it HANGS on the extra-wave stream -- so it is
# behind a knob rather than available by accident. Measured 2026-09-02.
#
# What works: an extra arm contributes NO fill to the relay (see _o_extra), so
# it leaves the BD chain where it found it, which is the rule the shared chain
# imposes. That is the right shape and it is why this gets as far as it does --
# the failure moved off the verify dispatch entirely.
#
# What does not: on an extra arm the rms core still puts its X on @xnorm,
# because that core has two MM2S and both are taken (@layerOut, @xnorm) so
# there is no second channel to put it on. Under mode 3 @xnorm has TWO
# consumers -- the relay tile, and the X memtile via _xs_extra -- which makes it
# a broadcast. With the relay's fill skipped its S2MM is never armed, the put
# has nowhere to land, and the fc stream times out on its first dispatch
# (dflash_prepass_waves._dispatch("fc"), before any verify dispatch runs).
#
# So the two ways out are unchanged and both are engine work: let the relay
# consume the extra arm's X (needs a BD on the cycle -- impossible while the
# decode arm's ph2 pins it at 38), or free an MM2S on the rms core by moving
# @layerOut off it. See docs/BZeroPlan.md.
#
# THE SECOND ONE IS NOT "GIVE THE EXTRA ARM ITS OWN CHANNEL". That reading is
# wrong for the same reason @tapsX is: a channel only the extra arm names does
# not survive to flow construction (section 3.15), so a free port buys nothing
# by itself. What the port would buy is a SPLIT COUNT -- take ONE of the decode
# arm's twelve ph0 re-sends off the relay and send it straight from the core on
# _XN1, so that EVERY arm puts on _XN1 exactly once and only the @xnorm fill
# counts differ:
#
#     _RF_CYCLE = [11, 38]      ph0 11 (BD 0 -> 1) + 1 direct = 12
#                               ph2 38 (BD 1 -> 0), still ONE fill
#                               extra: 0 fills, leaves the chain at 0
#
# ph2 stays a single fill, so the multi-fill defect is untouched; the extra arm
# still contributes no BD; and _XN1 from the rms core is L1 -> L2, so none of
# route 6's L3-allocation constraints apply. The whole of what it waits on is
# the port. See "Route 3, priced: THE SPLIT COUNT" in docs/BZeroPlan.md.
_RF_EXTRA_OK = int(_os.environ.get("RMS_MEMTILE_REFEED_EXTRA", "0"))
if RMS_MEMTILE_REFEED == 3 and N_EXTRA and not _RF_EXTRA_OK:
    raise SystemExit(
        f"RMS_MEMTILE_REFEED=3 does not carry the {N_EXTRA} extra waves.\n"
        f"  It BUILDS and it HANGS: the extra-wave stream times out on its\n"
        f"  first dispatch. The arms' shared BD chain is not the problem any\n"
        f"  more (an extra arm contributes no fill); the rms core's second\n"
        f"  MM2S is. See the comment above this refusal, and BZeroPlan.\n"
        f"  RMS_MEMTILE_REFEED_EXTRA=1 builds it anyway, for whoever picks\n"
        f"  this up -- the IR and the placement are worth inspecting."
    )
if RMS_MEMTILE_REFEED == 3 and N_EXTRA and max(EXTRA_NREFEED) > 1:
    raise SystemExit(
        f"an extra wave can only ever contribute no fill, which needs it to\n"
        f"  re-send its X exactly ONCE; these need {sorted(set(EXTRA_NREFEED))}.\n"
        f"  A wave that re-sends more has something to re-broadcast, so it\n"
        f"  would need a fill on the relay chain, whose cycle the decode arm's\n"
        f"  ph2 pins at {_RF_CYCLE}."
    )
# Empty by construction: the refusal above means mode 3 never coexists with an
# extra wave, and no other mode reads this. _PH0_SWEEPS' mode-3 branch therefore
# derives the same empty per-wave list the mode-0 branch's [1] * N_EXTRA does.

# One sweep where SOMETHING ELSE re-broadcasts (mode 1's @inX gather), the full
# refeed count where the core still does its own, and one sweep PER FILL under
# mode 3 -- where the relay replays each fill by the cycle count above.
_PH0_SWEEPS = (
    # ONE sweep per extra wave, not len(_RF_PH0_EX): an extra arm bypasses the
    # relay (see _o_extra), so its sweep is not a fill to be replayed -- it is
    # the single X send the wave needs, going straight to the X memtile.
    # ZERO core sweeps for an extra wave under mode 3: its X reaches the relay
    # from the shim on @tapsX (see _o_extra), so the rms core emits nothing on
    # @xnorm for that arm. A sweep here would be an unwanted fill on the shared
    # [12, 38] chain -- the exact thing @tapsX exists to avoid.
    (len(_RF_PH0_VOC), len(_RF_PH0_DEC), [0] * N_EXTRA)
    if RMS_MEMTILE_REFEED == 3
    else (
        (1, 1, [1] * N_EXTRA) if _PH0_RELAY else (VOCAB_RNDS, XN_REFEED, EXTRA_NREFEED)
    )
)
_PH2_SWEEPS = (
    (0, len(_RF_PH2_DEC), [0] * N_EXTRA)
    if RMS_MEMTILE_REFEED == 3
    else (
        (0, 1, [0] * N_EXTRA)
        if _PH2_RELAY
        else (0, REFEED[GATEUP_PHASE], [0] * N_EXTRA)
    )
)

# RMS_BAND_STREAM level 2's DMA transfer granularity for banding rmsX, kept
# SEPARATE from STG_W. AIETargetModel::getBDMaxDims/getDmaBdWrapBits
# (mlir-aie, AIETargetModel.h) give a core (compute) tile's DMA BD an 8-bit
# wrap field per dimension -- max representable extent is well under STG_W's
# 512, and the existing `_rms_batched_norm` comment about "the 3-D chunk-major
# form the MEMTILE producers use would not be legal here" is this exact
# limit, for a different channel. Three device-tested variants (A, B, C; see
# docs/DFlashFeasibility.md) all used STG_W (512) as a BD dimension's extent
# and all failed, two with the same wrong data and one with a hang --
# consistent with a truncated/wrapped 8-bit size field producing a
# deterministic but wrong hardware iteration count. 64 matches PAIR_PAY, an
# extent already proven to work in a 3-D pattern elsewhere (outy_tokmajor).
_RMS_DMA_CHUNK = 64
# Level 3 treats "band" (the resident residual slice), "chunk" (one @xnorm
# regen unit) and "round" (one projection egress) as the SAME STG_W-wide
# unit, so one small buffer and one fetch shape serve all three. True for
# every model in scope today (STG_W == XCHUNK == PAYLOAD == 512, both
# llama-3.2-1b and qwen3-4b) but not implied by any of their definitions --
# assert it rather than assume it, same discipline as the OPROJ_RNDS-vs-
# nband note in _rms_batched.
if RMS_BAND_STREAM >= 3:
    assert STG_W == PAYLOAD, (
        f"RMS_BAND_STREAM>=3 assumes STG_W({STG_W}) == PAYLOAD({PAYLOAD}): "
        "residual1/residual2 write back one band per projection round, and "
        "that only lines up if a band IS a round."
    )
    assert XCHUNK == STG_W, (
        f"RMS_BAND_STREAM>=3 assumes XCHUNK({XCHUNK}) == STG_W({STG_W}): "
        "the ph0/ph2 regen fetch reuses the band-sized buffer for one "
        "@xnorm chunk, and that only lines up if a chunk IS a band."
    )
    assert K % STG_W == 0

# RMS_W_ON_X: the norm weights ride the @rmsX band stream instead of owning
# channels of their own. THE RMS CORE HAS TWO S2MM PORTS AND LEVEL 3 NEEDS
# THREE, and this is what makes it two.
#
# air-to-aie emits a count-free, lock-driven BD ring for a physical S2MM only
# when the whole port collapses to ONE repeat-count group (generateDmaBdProgram:
# `infiniteBDLoopMode = repeat_counts.size() == 1`); anything else becomes a
# SEQUENCE of terminating tasks, and a terminating task fires once per dispatch,
# which is wrong for a per-layer transfer. So every port here must be one ring
# -- and a ring fires each of its slots exactly once per rotation, which means
# every channel on a port must have the SAME per-wave transfer count.
#
# Level 3's counts per layer are rmsX 270, rmsW 1, rmsW2 1, outY 10. `rmsX` has
# to be alone (270 is not a ratio any 16-slot ring expresses), `outY` has to be
# alone (its BD is deliberately unstamped -- the channel pins several packet ids
# and the core writes the routing id into the payload header, so the DMA must
# not filter), and that is both ports. Levels 0-2 do not have this problem:
# their rmsX is ONE whole-row get per layer, so {rmsX, rmsW, rmsW2} is 1:1:1 and
# rings on one port, which is exactly what the shipping build emits.
#
# The way out is to stop the weights needing a port at all. A norm weight is
# K bf16 and a band fetch moves BATCH*STG_W >= K of them into `xb`, so the
# weight fits in ONE band-shaped transfer on @rmsX: the launch side puts it from
# the rms weight buffer with a band-shaped descriptor, the core takes it with
# the SAME `_rms_band_get` every other visit uses, and copies the first K
# elements out to `w`. Identical descriptor => AIR folds it into the same BD as
# every other rmsX get => the ring stays one slot and count-free.
#
# It costs (BATCH*STG_W - K) elements of wasted transfer per weight per layer
# (1,536 at batch 8, 5,632 at batch 16 -- 0.1% of a layer's weight traffic) and
# one K-element on-core copy, and it BUYS the whole level-3 port budget. It also
# retires the rmsW2 lock hazard below: `w`/`w2` are no longer DMA destinations,
# so they have no locks to collide.
RMS_W_ON_X = (RMS_BAND_STREAM >= 3 or _os.environ.get("_RMS_W_ON_X")) and BATCH > 1
# Below level 3 the weight does not need a band-shaped transfer at all: it goes
# straight into `w` with its own 2-D descriptor, and rmsX's port ring is then
# {weight, weight2, row} at 1:1:1 -- the same shape the shipping build already
# rings {rmsX, rmsW, rmsW2} in. That makes _RMS_W_ON_X=1 a BISECTION: it tests
# carrying the norm weights on @rmsX, and nothing else, against a level that is
# verified on device.
# The band-shaped weight read walks BATCH*STG_W elements from the weight's base,
# so the LAST slab it reads from (the vocab arm's final norm, which sits at the
# very end of the rms BO) needs that much room behind it. Exported for
# llms/bench/decode_geometry.py, which sizes the host BO -- both sides read this
# same symbol so they cannot disagree. Zero unless the mode is on, so no
# shipping BO size moves.
RMS_TAIL_SLACK = (BATCH * STG_W - K) if RMS_W_ON_X else 0

# TRACE: hardware event trace for a NAMED SET OF TILES, off unless asked for.
#
#   DECODE_TRACE_SIZE=<bytes>  DECODE_TRACE_TILES=0.2;0.1  ./build_template.sh 8 L
#
# WHY NAMED TILES AND NOT THE WHOLE ARRAY. airrt-to-npu patches every trace flow
# in a column to the SAME shim buffer offset (trace_offset + col*size/ncols, per
# column, not per tile), so a column that traces two tiles has them overwriting
# each other in DDR. This design occupies 27 of 32 tiles; tracing all of them
# would return one tile's events per column and no way to tell which. Naming the
# tiles is also what keeps the instrument off the critical path -- each flow not
# created is a South channel not consumed on a core whose south channels are
# already carrying its data.
#
# WHY IT RIDES THE RMS BUFFER. The trace shim BD is patched against a kernel
# argument, and an argument the design never reads is an argument the launch is
# free to drop. arg2 is read-only for the design and its content all lives at
# offset 0, so a tail on it is the one place a trace region can go without a
# new ABI position, a new host binding, or a live-range question. Same trick,
# same reason, as RMS_TAIL_SLACK above and RMS_SCRATCH below.
TRACE_SIZE = int(_os.environ.get("DECODE_TRACE_SIZE", "0"))
TRACE_TILES = _os.environ.get("DECODE_TRACE_TILES", "")
# In bf16 elements, and generous: the shim BD's length field is not documented
# here in bytes, so reserving 2x the requested trace size means a units error
# reads as a short trace rather than as a corrupted buffer past the end.
TRACE_SLACK = (2 * TRACE_SIZE) // 2 if TRACE_SIZE else 0
# Set by build_module once rms_l3's extent is known; read by run() and by the
# host harness (which imports this module for its geometry).
TRACE_OFFSET_BYTES = 0

# RMS_SCRATCH: one block of hidden states appended to the END of the Y buffer,
# holding `h` (the post-attention residual) between residual1 and residual2 at
# RMS_BAND_STREAM>=3. It is NOT a host output -- see
# _rms_interleave_after_phase for why `h` cannot share X.
#
# In Y rather than in a fifth DDR argument because Y's size is already derived
# from these symbols (llms/bench/decode_geometry.py), so a region appended here
# needs no new host binding and no ABI change. ONE block, not one per layer: it
# is dead the moment residual2 has read it. Zero for every other build, so no
# shipping BO moves.
RMS_SCRATCH = BATCH * K if (RMS_BAND_STREAM >= 3 and BATCH > 1) else 0

# _RMS_RES_ONLY=1|2: keep the banded round-trip for ONE residual stage and drop
# it from the other -- both halves, core and launch, so @rmsX and @layerOut stay
# balanced. The dropped stage still gets its @outY rounds and still calls the
# accumulate kernel, on whatever stale band is in `xb`; its OUTPUT is wrong by
# construction. The question it answers is whether residual2 hangs because it is
# residual2 or because it is SECOND.
_RMS_RES_ONLY = int(_os.environ.get("_RMS_RES_ONLY", "0"))

# _RMS_BANDS_GET / _RMS_BANDS_PUT: which residual stages do the @rmsX band GET
# and which do the @layerOut band PUT-BACK, as digit strings over {1, 2}. The
# two halves are separable on purpose -- residual1's round-trip completes and
# residual2's does not, on identical channel volumes, and the halves are the
# next place to cut. Dropping a half drops the matching launch-side op too, so
# both channels stay balanced; the stage that keeps its get without its put (or
# the reverse) accumulates onto a stale band and its output is garbage.
#
#   _RMS_BANDS_GET=1 _RMS_BANDS_PUT=12   residual2 puts back a band it never got
#   _RMS_BANDS_GET=12 _RMS_BANDS_PUT=1   residual2 gets a band it never returns
#
# _RMS_RES_ONLY=N is the shorthand for setting both to N.
_RMS_BANDS_GET = _os.environ.get(
    "_RMS_BANDS_GET", str(_RMS_RES_ONLY) if _RMS_RES_ONLY else "12"
)
_RMS_BANDS_PUT = _os.environ.get(
    "_RMS_BANDS_PUT", str(_RMS_RES_ONLY) if _RMS_RES_ONLY else "12"
)

# _RMS_STG_COPY: copy each @outY round out of its landing buffer immediately,
# so the round's lock is RELEASED before the accumulate waits for its @rmsX
# band. Without it AIR keeps the landing buffer acquired across the band's
# arrival, because the accumulate kernel reads both.
#
# The bisection says residual2 stalls on its band GET specifically (its
# put-back is fine: _RMS_BANDS_GET=1 _RMS_BANDS_PUT=12 completes, the reverse
# hangs) while residual1's identical get works. What differs is only the
# producer -- the DOWN projection instead of the o-proj -- so the suspect is
# back-pressure: the core holds the outY buffer, the next round's packet has
# nowhere to land, and if it shares a link with the band's route the band never
# arrives. Copying the round out is the one-line version of "do not hold it".
#
# Costs one more BATCH*STG_W band buffer on the rms tile (8 KB at batch 8),
# accounted in _rms_l1_bytes.
_RMS_STG_COPY = bool(_os.environ.get("_RMS_STG_COPY")) and RMS_BAND_STREAM >= 3

# _RMS_LATE_RT: put every level-3 banded feed after the whole weight-feed loop
# again, the placement that shipped before _rms_interleave_after_phase. Kept as
# the A side of the comparison, not as an option -- it DEADLOCKS (see that
# function's docstring: the gate-up weight await needs ph2, ph2 needs
# residual1, residual1 needs a band the host has not armed).
_RMS_LATE_RT = bool(_os.environ.get("_RMS_LATE_RT"))

# _RMS_RT_PER_BAND: emit a residual round-trip as five put/drain PAIRS instead
# of one folded put and one folded get. MEASURED WORSE, kept as the record.
#
# The folded pair is ordered by AIR's whole-op dependency analysis --
#
#   %138 = air.channel.put  @rmsX     (X[0,0,0] [5,8,512])            <- reads X
#   %139 = air.channel.get  @layerOut (X[0,0,0] [5,8,512]) [.. %138]  <- writes X
#
# -- and lowers to `await(put); start(get)`, which the core cannot satisfy: it
# takes band b, accumulates, and PUTS BAND b BACK before asking for band b+1, so
# the folded put does not retire until the drain is armed. Per band that
# dependency is at the granularity the core works at and each put retires when
# its own band is taken.
#
# It trades one deadlock for a worse one. Per band, band b's drain is ordered
# before band b+1's put, so the HOST blocks on band 0's drain before it has
# issued the projection weight feed the drain's data depends on -- and the
# whole round-trip has to be armed before that feed (see
# _rms_interleave_after_phase). Measured: nothing written at all, where folded
# gets through residual1 and ph2.
_RMS_RT_FOLD = not _os.environ.get("_RMS_RT_PER_BAND")

# BOTH norm weights, because the batched body really does hold both. It allocs
# `w` (input norm) and `w2` (post-attention norm) together at the top and frees
# them together at the bottom -- not alloc-at-first-use/free-at-last-use, and
# not `w` reused for the second weight either. Reuse would save K bf16 on a
# tile that is short of exactly that, and it costs the separate lock pair AIR
# gives the second weight, which hangs the batched decode in wave 0. See the
# rmsW2 get in _rms_batched.
#
# Counting one of them was wrong by K bf16 and it was not cosmetic: it put
# qwen3-4b's ceiling at 8 when the tile only holds 7, so DECODE_BATCH=8 sailed
# past this check and died in the mlir-aie allocator on tile_2_2 instead --
# which reports a byte count and not which buffer or which knob.
#
# (POST_RMS is what turns the second norm on. N_NORMS>=4 -- the Gemma sandwich,
# two norms packed per channel -- implies it, and is refused at BATCH>1 anyway.)
_RMS_W_RESIDENT = 2 * K if POST_RMS else K


def _rms_l1_bytes(b):
    """L1 the batched rms body holds at its peak, for a batch of b.

    Below level 3 the residual (`xb`) is the whole K-wide row. At level 3
    it is one STG_W-wide band, refetched fresh instead of held resident --
    that swap is the entire point (see RMS_BAND_STREAM's level-3 comment).

    Level 3 holds a SECOND band buffer (`xr`, the residual round-trip's own),
    and it is a lock-correctness requirement rather than a convenience: see
    the `xr` alloc in _rms_batched. Same width as `xb`, so level 3 is
    2*b*STG_W of band where levels 0-2 are b*K of resident row.
    """
    _xb_w = STG_W if RMS_BAND_STREAM >= 3 else K
    _xr_w = STG_W if RMS_BAND_STREAM >= 3 else 0
    # _RMS_STG_COPY adds a THIRD band buffer, the @outY landing shadow. A
    # diagnostic knob today; if it becomes the shipping shape it is what sets
    # the batch ceiling, since three bands at b=16 is the whole tile.
    _sg2_w = STG_W if _RMS_STG_COPY else 0
    return (
        2 * (b * _xb_w + b * _xr_w + b * _sg2_w + b * STG_W + _RMS_W_RESIDENT) + 4 * b
    )


BATCH_MAX_RMS = max(
    (b for b in range(1, 65) if _rms_l1_bytes(b) <= _L1_ROWTILE_BUDGET), default=1
)
if BATCH > 1 and N_NORMS >= 4:
    # Gemma sandwich norm: h = x + post_attn_norm(o_proj). The norm is applied
    # to the SUBLAYER OUTPUT, so the whole K-wide o-proj row has to be resident
    # before the residual can be formed -- and that is the second BATCH*K
    # buffer the batched rms body exists to avoid. Pre-norm models add the
    # projection output in raw, one round at a time, which is what makes one
    # buffer enough. Refuse rather than silently produce a design that cannot
    # fit; the DFlash target (qwen3-4b) is pre-norm.
    raise SystemExit(
        f"DECODE_BATCH>1 with N_NORMS={N_NORMS} (sandwich norm) is not wired: "
        "normalizing the sublayer output needs the whole projection row "
        "resident, which does not fit beside the residual batch."
    )
if BATCH > BATCH_MAX_RMS:
    raise SystemExit(
        f"DECODE_BATCH={BATCH} exceeds the rms core's L1 ceiling of "
        f"{BATCH_MAX_RMS} for {MODEL_NAME} ({_rms_l1_bytes(BATCH)} B of "
        f"{_L1_ROWTILE_BUDGET} B): one BATCH*{K} residual buffer, a "
        f"BATCH*{STG_W} staging buffer and {_RMS_W_RESIDENT} elements of norm "
        "weight.\n"
        "Tiling the row loop does NOT raise it -- the projection needs all B "
        f"rows of a chunk at once. Moving the BATCH*{STG_W} staging to L2 does "
        f"not raise it far enough either: it leaves the BATCH*{K} residual, "
        f"which caps at {max((b for b in range(1, 65) if 2 * (b * K + _RMS_W_RESIDENT) + 4 * b <= _L1_ROWTILE_BUDGET), default=1)}. "
        "The residual is the term, and the reason it is resident is LIFETIME "
        "and not working set: the core computes on one band at a time, but the "
        "thing added to band c is the o-proj output, which does not exist until "
        "every band has gone through a reduction over all of K. To go past this "
        f"the residual has to live in the DDR X buffer and round-trip a band at "
        f"a time on @rmsX / @layerOut, which caps at "
        f"{max((b for b in range(1, 65) if 2 * (b * STG_W + b * STG_W + _RMS_W_RESIDENT) + 4 * b <= _L1_ROWTILE_BUDGET), default=1)}. "
        "See 'block 16 does not build' in docs/DFlashFeasibility.md, which also "
        "measures why that is probably not worth building."
    )
# The conservative global tile is kept as the default for anything that has to
# pick one, and as the number the doc quotes.
ROW_TILE = min(RMS_TILE, ROPE_TILE, GLU_TILE)
ROW_SUBTILES = (BATCH + ROW_TILE - 1) // ROW_TILE

# Attention gets a query tile too, for the same reason and with a tighter bound
# (the KV block and scores are per-CU fixed, q/o/scores scale). Costs a second
# read of the KV block from L2 per sub-tile; no extra DDR traffic, because the
# block is already resident in L2 when the first sub-tile reads it.
ATTN_QTILE = max(1, min(BATCH, BATCH_MAX_ATTN_QTILE))
ATTN_QSUBTILES = (BATCH + ATTN_QTILE - 1) // ATTN_QTILE
# Per-layer DDR slab sizes (elements). LUT is per-position (shared across layers),
# placed after all NLAYERS rms slabs.
W_LAYER = sum(NCX * PER_COL_PH[p] * BLOCK_BF16 for p in range(NPH))  # weights / layer
RMS_LAYER = N_NORMS * K  # rms weights / layer (2 llama pre-norm / 4 Gemma sandwich)
# Offset of the ones run appended to the rms BO for extra waves (see rms_l3 and
# _uni_extra). Module scope because the HOST has to fill it, and a second copy of
# this arithmetic on that side is exactly how a silently-wrong buffer happens.
RMS_ONES_OFF = (
    UNI_DEC * RMS_LAYER
    + (UNI_DEC if ROPE_W_PER_LAYER else 1) * ROPE_W_LEN * BATCH
    + K
    + RMS_TAIL_SLACK
)
KV_LAYER = ATTN_MAXL * KVSZ_TOK  # KV cache / layer
Y_LAYER = sum(ROUNDS_PER_DEST[p] * PAYLOAD for p in HOST_DRAIN if p != 0)  # Y / layer
# X slots: 1 for the in-place chain, one per layer boundary when tapping. UNI_DEC
# layers have UNI_DEC+1 boundaries (the prompt embedding in, every layer's output
# after). The LM head reads the last one, exactly as it reads slot 0 today.
X_SLOTS = (UNI_DEC + 1) if HIDDEN_TAPS else 1
# Extra waves read (and their destinations write) X slots too, and one of them
# is normally past the layer outputs -- a wave that feeds the next one needs
# somewhere to leave its result. Grow the buffer to whatever they reach rather
# than adding a knob: the slots are already named in the wave list.
EXTRA_OUT_SLOT = []
if N_EXTRA:
    X_SLOTS = max(
        [X_SLOTS]
        + [
            w["x_slot"] + (n - 1) * w["x_stride"] + 1
            for w, n in zip(EXTRA_WAVES, EXTRA_X_NSLOT)
        ]
    )
    # ONE OUTPUT SLOT PER EXTRA WAVE, past every slot the waves read. An extra
    # wave drains the rms core's whole BATCH*K row -- it has to, the core's
    # @layerOut put IS that row and a shorter get would leave the channel
    # unbalanced -- so waves that touch disjoint COLUMN bands still overwrite
    # each other's full row if they share a slot. The 25 fc sub-waves sharing
    # slot 0 leave only the last one's band, over a row of untouched tap.
    #
    # Appended here rather than named in the wave list, so a wave descriptor
    # stays a description of what a wave READS and the engine owns where
    # results land.
    EXTRA_OUT_SLOT = [X_SLOTS + k for k in range(N_EXTRA)]
    X_SLOTS += N_EXTRA


def build_module():
    @module_builder
    def build():
        bf16 = BF16Type.get()
        f32 = F32Type.get()
        i32 = IntegerType.get_signless(32)
        idx_t = IndexType.get()
        l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l2 = IntegerAttr.get(T.i32(), MemorySpace.L2)

        # ---- host operands ----
        # x is chained IN-PLACE (offset 0 every layer), so it is NOT scaled by
        # NLAYERS. The weight / rms / KV DDR buffers hold NLAYERS successive per-layer
        # slabs (offset iv*SLAB), so they scale by NLAYERS. At NLAYERS=1 every size is
        # identical to the single-layer design.
        #
        # HIDDEN_TAPS un-does the in-place part: layer iv reads slot iv and writes
        # slot iv+1, so every layer's hidden state stays readable instead of being
        # overwritten by the next one. See the HIDDEN_TAPS comment above.
        # DECODE_BATCH: B token embeddings in, token-major.
        #
        # RMS_BAND_STREAM>=3 (not yet built) does NOT need a separate scratch
        # slot, on reflection: without HIDDEN_TAPS, _x_in() == _x_out() (the
        # in-place chain), so residual1's intra-layer write of `h` can land
        # at _x_in()'s own address, in place, exactly like the WHOLE-buffer
        # write the unbanded design already does -- just one band at a time
        # instead of all of them. ph2/residual2 then re-read/re-write that
        # SAME address. No new DDR region, no new addressing scheme, and the
        # rms core's already-full packet-id budget is never in question.
        # Refuse the ONE case this doesn't cover instead of silently
        # producing a design that corrupts a HIDDEN_TAPS tap.
        if RMS_BAND_STREAM >= 3 and HIDDEN_TAPS:
            raise SystemExit(
                "RMS_BAND_STREAM>=3 with DECODE_HIDDEN_TAPS is not wired: "
                "level 3 reuses _x_in()'s address as scratch for the "
                "intra-layer h round-trip, which is only safe when "
                "_x_in() == _x_out() (HIDDEN_TAPS off)."
            )
        x_l3 = MemRefType.get([X_SLOTS * BATCH * K], bf16)  # RAW input activation
        # LM_HEAD build carries the vocab weights (VOCAB_W_BLOCKS q4k blocks) instead
        # of the decode phase weights. Separate compile-time size -> decode IR is
        # byte-identical; the device (CDO) is unchanged (only this DDR memref size +
        # the runtime feed differ), so both still share one xclbin.
        _w_blocks = UNI_DEC * W_TOTAL_BLOCKS + UNI_LM * VOCAB_W_BLOCKS
        w_l3 = MemRefType.get(
            [_w_blocks * BLOCK_BF16], bf16
        )  # packed q4k weights (all phases concatenated), NLAYERS slabs
        # W_SPLIT: w_l3 above stays arg1 and holds GROUP 0; the remaining groups and
        # the lm-head slab are APPENDED after the existing args so every current host
        # binding position (x/w/rms/y/kvc) is unchanged.
        _wgrp_len = [
            min(W_GROUP, UNI_DEC - g * W_GROUP) * W_LAYER for g in range(N_WGRP)
        ]
        _wvoc_len = UNI_LM * VOCAB_W_BLOCKS * BLOCK_BF16
        if W_SPLIT:
            w_l3 = MemRefType.get([_wgrp_len[0]], bf16)
        _w_extra = (
            [MemRefType.get([n], bf16) for n in _wgrp_len[1:]]
            + [MemRefType.get([_wvoc_len], bf16)]
            if W_SPLIT
            else []
        )
        # The extra waves' weights, appended LAST for the same reason W_SPLIT's
        # groups are appended at all: every existing host binding position stays
        # where it is. They get their own buffer rather than a tail on w_l3
        # because they are not this model's weights -- the DFlash pre-pass reads
        # the DRAFTER's fc and k/v while running in the target's program -- so
        # concatenating them would put a second model inside the target's
        # requant cache.
        if N_EXTRA:
            _w_extra = _w_extra + [MemRefType.get([EXTRA_W_ELEMS], bf16)]
        # rms weight (K). MULTIBLK appends the rope region AFTER all UNI_DEC rms slabs so
        # the score-path test gets a KNOWN q (q_roped = proj_q) WITHOUT corrupting
        # rms_w[0:K] (which proj_q depends on). Llama: ONE shared per-position LUT
        # (ROPE_W_LEN=64). Per-layer models (ROPE_W_PER_LAYER): UNI_DEC rope_w slabs
        # (dual-theta + per-layer q/k-norm). L=1 ABI unchanged.
        # The rope LUT slab is PER POSITION, and a block of B tokens spans B
        # positions -- so it is the one part of this buffer that scales with the
        # batch. Easy to miss: at batch 1 the host patches a single 64-word
        # cos/sin LUT per token, and nothing in the shape says "per position".
        rms_l3 = MemRefType.get(
            [
                UNI_DEC * RMS_LAYER
                + (
                    (UNI_DEC if ROPE_W_PER_LAYER else 1) * ROPE_W_LEN * BATCH
                    if MULTIBLK
                    else 0
                )
                + K  # dedicated final-norm slot for real lm_head (vocab)
                # RMS_W_ON_X reads each norm weight BATCH*STG_W wide from its own
                # base (one band-shaped transfer), which overruns the LAST of
                # them -- the final-norm slot immediately above -- by
                # RMS_TAIL_SLACK. It has to be in THIS type, not just in the
                # host's BO: the declared memref is what the descriptor is
                # bounds-checked against, and the vocab arm's weight read would
                # otherwise sit past the end of it. Zero unless RMS_W_ON_X, so no
                # shipping buffer moves. llms/bench/decode_geometry.py adds the
                # same term from the same symbol.
                + RMS_TAIL_SLACK
                # An extra wave's X is RAW -- it is a projection input, not a
                # layer's activation -- but the only producer on @xnorm with a
                # route to DDR is the rms core, whose regen multiplies by the
                # norm weight. Feed it a run of ONES and rms_chunk degenerates to
                # the strided gather it already is:
                #   y[t*n+i] = x[t*K + c*n + i] * w[c*n+i] * scales[t]
                # so w == 1 leaves only the per-row scales[t], which the host
                # divides back out (it wrote the tap, so it knows rms(tap)).
                # That is what makes the X forwarding need NO kernel change at
                # all -- see _uni_extra.
                + (K if N_EXTRA else 0)
                # The trace region, past everything the design reads. See TRACE_SIZE.
                + TRACE_SLACK
            ],
            bf16,
        )
        # Byte offset of the trace region within arg2, which is what the trace
        # shim BD is patched with, and which the host reads it back from.
        # Published rather than recomputed: it is derived from the very
        # expression that sizes the buffer, so the two cannot disagree.
        global TRACE_OFFSET_BYTES
        TRACE_OFFSET_BYTES = (rms_l3.shape[0] - TRACE_SLACK) * 2
        # LM_HEAD drains VOCAB_SIZE_PADDED logits into Y (arg3); decode uses Y for the
        # QKV host rounds + rms layer-out. Separate compile-time size (decode unchanged).
        # Every drained PAYLOAD row becomes B rows, token-major (egress_bd.py) --
        # the vocab region included, since the batched LM head produces B tokens'
        # logits per wave. Within a wave they are token-major, so the host reads
        # token t's chunk as Y[base + w*B*VP + t*VP : +VP].
        _y_elems = (
            (HOST_ROUNDS + LAYER_RNDS) * PAYLOAD * BATCH
            + UNI_LM * VOCAB_SIZE_PADDED * BATCH
            + PROBE_TOTAL
        )
        _y_scratch_off = _y_elems
        _y_elems += RMS_SCRATCH
        y_l3 = MemRefType.get(
            [_y_elems], bf16
        )  # host-drain (QKV) rounds + LAYER_RNDS rms layer-out (down) rounds
        # MULTIBLK: DDR KV cache (the reference full-faithful append+readback). Layout
        # [ATTN_MAXL][K: DK_TOT_A | V: DK_TOT_A]; rope appends this token at
        # APPEND_OFF, then the whole cache is streamed back per CU (_d2wip shapes).
        # NLAYERS per-layer caches concatenated (offset iv*KV_LAYER).
        kvc_l3 = MemRefType.get([UNI_DEC * ATTN_MAXL * KVSZ_TOK], bf16)

        # ---- L1 buffers ----
        xblk_l1 = MemRefType.get([COL_BLOCK], bf16, memory_space=l1)  # 256 X chunk
        wblk_l1 = MemRefType.get([BLOCK_BF16], bf16, memory_space=l1)  # 2560 weight
        yacc_l1 = MemRefType.get([ROW_BLOCK], f32, memory_space=l1)  # accumulator
        # b_col_reduce_add cache, one slot (COL_BLOCK/32 bf16) per col-block of
        # the WIDEST projection -- see RCACHE_LEN. 512 B/core for llama-1B.
        rcache_l1 = MemRefType.get([RCACHE_LEN], bf16, memory_space=l1)
        ypair_l1 = MemRefType.get(
            [16 + PAIR_ROWS * ROW_BLOCK], bf16, memory_space=l1  # 80 shared
        )
        # ---- DECODE_BATCH proj-core buffers (emitted only when batched) ----
        # Three of the GEMV's four buffers scale with the batch; the weight
        # block does not, which is the entire point. The fourth buffer is new
        # and is the one that costs: q4k_mm_block unpacks the 4-bit block into
        # bf16 before aie::mmul can see it, and that scratch is 16 KB whatever
        # the batch is. It is also what pays for itself -- the unpack cost then
        # divides by BATCH instead of being redone per token.
        #
        # rcache_l1 has NO batched counterpart, deliberately. The reduce cache
        # exists for the GEMV's +min factorisation (min[r,g] * sum-of-x over
        # group g); the batched path materialises w = q*scale + min elementwise
        # because aie::mmul needs a real B operand, so there is nothing to
        # cache. 608 B of rc traded for the 16 KB scratch.
        if BATCH > 1:
            xblk_mm_l1 = MemRefType.get(
                [BATCH * COL_BLOCK], bf16, memory_space=l1
            )  # TILE-BLOCKED, not [BATCH][COL_BLOCK] -- see pack_A in q4k_mm_gate.py
            yacc_mm_l1 = MemRefType.get([BATCH * ROW_BLOCK], f32, memory_space=l1)
            wscr_mm_l1 = MemRefType.get(
                [ROW_BLOCK * COL_BLOCK], bf16, memory_space=l1  # 8192 bf16 = 16 KB
            )
            # Token-major, pair-interleaved: token t, pair role i at
            # (t*PAIR_ROWS + i)*ROW_BLOCK. That is what proj_qmm_mm_flush_row
            # writes with tok_stride = PAIR_ROWS, and it makes the egress a
            # straight widen of the existing contiguous put rather than a
            # scatter.
            #
            # ROUNDED UP TO 64 BYTES, and that is not cosmetic. mlir-aie aligns
            # a compute-tile buffer to the tile's LOAD/STORE BUS width and packs
            # the rest end to end (AIEAssignBuffers.cpp), and on AIE2p that
            # width is 256 bits -- 32 bytes, not 64. aie::mmul<8,8,8>'s C tile
            # is 64 floats = 256 bytes and Peano moves it in 512-bit chunks, so
            # a 32-byte-aligned accumulator is a MISALIGNED 512-bit access, and
            # AIE2 does not fault on one: it masks the low address bits. The
            # accumulator lands 32 bytes low, every value shifts by 8 floats and
            # the last 8 are never written.
            #
            # 16 + 2*32*8 = 528 bf16 = 1056 bytes = 16.5 x 64. THIS buffer is
            # the only odd-sized one on a proj core and the pair's LEAD tile is
            # the only tile that hosts it, so only lead tiles misplaced what the
            # allocator packed next -- their second accumulator (the _e=1 round)
            # at ...820 instead of ...800. The QKV phase's 6 rounds put K on
            # round 4 (_e=0, aligned, correct) and V on round 5 (_e=1,
            # misaligned, wrong), on the lead half of every emitter block only.
            # That reads exactly like "half of every projection's output rows
            # are computed against the wrong token" and it is not: the X feed,
            # the descriptors and the L2 transpose were all correct throughout.
            # l1_align.py checks the emitted addresses; batch_row_probe.py is
            # the on-device symptom.
            _YPAIR = 16 + PAIR_ROWS * ROW_BLOCK * BATCH
            ypair_mm_l1 = MemRefType.get(
                [_YPAIR + (-_YPAIR % L1_VEC_BF16)], bf16, memory_space=l1
            )
        rms_l1 = MemRefType.get([K], bf16, memory_space=l1)  # rms in/out/weight (2048)
        if BATCH > 1:
            # ---- the rms core's ONE batched buffer, and its two helpers ----
            # rmsb: the residual stream for all B rows, [BATCH][K] row-major.
            # It is raw X on the way in, h after the attention output is added,
            # and the layer output after the MLP output is -- one buffer, three
            # roles, which is what keeps a second BATCH*K allocation off the
            # core (see the STG_W comment above the batch ceilings).
            # rstg: [BATCH][STG_W], the only thing that ever leaves or arrives
            # whole -- one @xnorm chunk out, one projection round in.
            # rscl: one f32 per row, the 1/rms factor the chunk kernel replays.
            rmsb_l1 = MemRefType.get([BATCH * K], bf16, memory_space=l1)
            rstg_l1 = MemRefType.get([BATCH * STG_W], bf16, memory_space=l1)
            rscl_l1 = MemRefType.get([BATCH], T.f32(), memory_space=l1)
        # Gemma 4-norm: two norm weights packed per channel (2K) so the rms tile
        # keeps <=4 packet ids per S2MM port (1 arbiter x 4 msels). lo/hi kernels
        # slice it. Only used when N_NORMS>=4.
        rms_w2k_l1 = MemRefType.get([2 * K], bf16, memory_space=l1)
        # [up|gate] in, silu*up out. At BATCH>1 both hold ONE ROUND for every
        # token -- [BATCH][GLU_SLICE] and [BATCH][GLU_HID] -- because that is
        # how the projection egresses it. The 2-slot ring below doubles both.
        glu_x_l1 = MemRefType.get([BATCH * GLU_SLICE], bf16, memory_space=l1)
        glu_hid_l1 = MemRefType.get([BATCH * GLU_HID], bf16, memory_space=l1)
        # ATTN S1 rope (reference rope_compute): qkv(3072 QKV out)+lut(64) -> q(2048),
        # k(512), v(512) roped. tile_2_3.
        qkv_l1 = MemRefType.get([M], bf16, memory_space=l1)
        ropeq_l1 = MemRefType.get(
            [DQ_PADDED], bf16, memory_space=l1
        )  # rope emits padded Q
        ropekv_l1 = MemRefType.get([DK], bf16, memory_space=l1)
        ropelut_l1 = MemRefType.get([ROPE_W_LEN], bf16, memory_space=l1)
        # ATTN S3a flash-attn (1 CU; attn_iso proven shapes). DH=64, 8 Q heads,
        # 2 KV heads per CU -> DQ=OSZ=512, DK=128, k/v block 16x128, scores 192.
        # q per CU. [BATCH][DQ_PADDED_PER_CU] when batched, and taken in ONE get
        # before the token loop -- see _qk_body. This is a deadlock fix, not a
        # buffering choice: the q memtile's four fan-out puts are a DAISY CHAIN
        # (CU c+1's transfer is gated on CU c's finishing), so a one-token
        # landing buffer makes CU 0's link wait for CU 0 to run the whole block,
        # while CU 1 -- which cannot start without its q -- holds up the KV
        # re-block memtile that CU 0 is waiting on.
        aq_l1 = MemRefType.get([BATCH * DQ_PADDED_PER_CU], bf16, memory_space=l1)
        ak_l1 = MemRefType.get(
            [16 * KVPC_DH], bf16, memory_space=l1
        )  # k block 16xKVPC_DH
        av_l1 = MemRefType.get(
            [16 * KVPC_DH], bf16, memory_space=l1
        )  # v block 16xKVPC_DH
        as_l1 = MemRefType.get([SSZ_BLK], bf16, memory_space=l1)  # shared scores
        # o per CU. [BATCH][DQ_PADDED_PER_CU] when batched: the CU holds every
        # token's output and emits the block once (attn_kv_fin_row).
        ao_l1 = MemRefType.get([BATCH * DQ_PADDED_PER_CU], bf16, memory_space=l1)
        # KV block cache (attn_stream proven): SEPARATE K and V natural block
        # buffers [key16, kvh2, dh64] = 2048 each; memtile reorder -> pack_k/pack_v.
        ak_l2 = MemRefType.get([16 * KVPC_DH], bf16, memory_space=l2)
        av_l2 = MemRefType.get([16 * KVPC_DH], bf16, memory_space=l2)
        # QKV staging memtile (reference mem_1_1 role): assemble the 6 demux rounds
        # into a contiguous 3072, then ONE transfer to rope. Feeding rope's compute-
        # tile S2MM directly from the 6 packet rounds mis-aligned by 1 head (+64).
        qkvmt_l2 = MemRefType.get([M], bf16, memory_space=l2)
        # q broadcast memtile (reference mem_5_1 q_buffer): rope sends q ONCE (2048),
        # the memtile fans out per-CU 512 (reordered). Direct rope->CU q puts cost 1
        # rope MM2S per CU -> only 2 MM2S available, so N>=2 starved k/v (deadlock).
        # Padded Q broadcast. [BATCH][DQ_PADDED] at BATCH>1, and the batch
        # dimension is not an optimisation -- it BREAKS A DEADLOCK. See
        # _qmtb_dec.
        qmt_l2 = MemRefType.get([BATCH * DQ_PADDED], bf16, memory_space=l2)
        # o gather memtile (reference mem_5_1 o_buffer): 4 CUs' o (512 each) gathered
        # into 2048, then ONE egress (-> host now; -> mem_1_1 o-proj X in the loop close).
        # [BATCH][DQ] at BATCH>1: the CUs emit one o per token, token-major, and
        # this is the buffer the chunk-major @xnorm re-broadcast reads -- the
        # same shape contract as the rms core's X and the down buffer's.
        omt_l2 = MemRefType.get([BATCH * DQ], bf16, memory_space=l2)
        # MULTIBLK per-block KV staging memtile (attn_iso ring, PASS L=16..128): one
        # block [K block 2048 | V block 2048] = 4096; a fresh alloc per block gives a
        # count-free ping-pong ring (1 fill : 1 read), unlike a whole-cache buffer
        # multi-read (1 fill : N read = lock deadlock).
        kvblk_l2 = MemRefType.get([2 * 16 * KVPC_DH], bf16, memory_space=l2)
        # buf_ph2 (LOOPCLOSE): ph2 (gate-up) X = a_xn stand-in, re-broadcast from a
        # memtile (mechanism-2 refeed) so it converges on @xnorm AFTER ph1 attn-o.
        bufp2_l2 = MemRefType.get([K], bf16, memory_space=l2)
        # ---- DECODE_BATCH: the QKV drain transposer ----
        # The projection emits (ROUND, TOKEN): round r is a 32-row band of the
        # output for ALL B tokens, because that is what a batched mmul computes
        # in one go. rope wants (TOKEN, ROUND) -- one token's whole M-wide qkv
        # row -- and cannot be given a strided landing for it, because a
        # [B][M] L1 buffer is 96 KB against a 54 KB budget on qwen3-4b.
        #
        # So transpose in L2, where 96 KB is nothing, and leave rope, attention,
        # the KV append and everything downstream operating on ONE token at a
        # time, looped B times. That is also what the cost model already assumes
        # (section 5e: attention does not amortize), so nothing is lost by it.
        #
        # The alternative -- slicing rope by head so it consumes (round, token)
        # directly -- is a kernel change to pseduo_rope plus a rewrite of the
        # attention feed, for a phase that is 3% of the pass.
        if BATCH > 1:
            qkvmt_l2 = MemRefType.get([BATCH * M], bf16, memory_space=l2)

        # ---- L2 buffers ----
        # X memtile = reproducer x_buffer: 512 (2 blocks) so the producer re-feed +
        # broadcast has the same slack as the reference; the proj cores' 256 ring chops it.
        xmt_l2 = MemRefType.get([BATCH * XCHUNK], bf16, memory_space=l2)
        # RMS_MEMTILE_REFEED: the same staging, widened to hold a whole phase's
        # X (every chunk of every token) so the memtile can re-broadcast it
        # instead of the core recomputing it. 40 KB at batch 8 against the
        # memtile's 512 KB.
        xmt_wide_l2 = MemRefType.get([BATCH * K], bf16, memory_space=l2)
        # One fan get. W_DUAL_CHAN halves it: each shim channel feeds its own
        # ring covering half the column's cores (FLM's w_buffer[0:5120] /
        # w_buffer[5120:10240] split).
        wfan_l2 = MemRefType.get(
            [(NCY // (2 if W_DUAL_CHAN else 1)) * BLOCK_BF16], bf16, memory_space=l2
        )
        grp_l2 = MemRefType.get([GRP_ROWS_B], bf16, memory_space=l2)
        main_l2 = MemRefType.get([MAIN_ROWS_B], bf16, memory_space=l2)
        relay_l2 = MemRefType.get(
            [PAYLOAD_B], bf16, memory_space=l2
        )  # demux relay (stripped)
        # GLU out accumulate. Token-major at BATCH>1 ([BATCH][GLU_OUT]), which
        # is what the chunk-major @xnorm re-broadcast below reads with a
        # GLU_OUT token stride -- so the down phase sees the same layout the
        # rms core's X does, and _feed_inX does not need a second shape.
        down_l2 = MemRefType.get([BATCH * GLU_OUT], bf16, memory_space=l2)
        rmsrow_l2 = MemRefType.get([BATCH * K], bf16, memory_space=l2)
        # relay memtile columns for the id-demux dests (free cols, not proj/X/MT).
        # GLU dest (gate-up) goes DIRECT to the GLU tile (no relay).
        RELAY_COLS = [3, 5, 4][:NDEST]
        # QKV_RELAY_COL: which free column the dest-0 (QKV) transposer lands on.
        # It is the LARGEST movable buffer on its memtile (BATCH*M) and column 3
        # is the one that runs out first at batch 16 -- see below. Col 3 also
        # carries the down memtile (DOWN_PCOL, pinned there by a routing
        # constraint) and two of the four attention CUs' KV re-blocks, neither
        # of which moves; this one does.
        #
        #   batch 8, level 3, memtile occupancy of 512 KB   [static]
        #     3,1  319,488 (60%)   down 155,648 + qkv 98,304 + 4 KV blocks
        #     5,1  196,608 (37%)   o-gather + two o-proj relays
        #     4,1   65,536 (12%)   the other two CUs' KV blocks
        #     2,1   32,776  (6%)   the X memtile
        #     0/1/6/7,1  45,064 (8%) each, the weight fan
        #
        # Doubled, only col 3 overflows -- 573,440 of 524,288. Moving the
        # transposer to col 4 leaves col 3 at 376,832 (72%) and col 4 at 262,144
        # (50%), and every other column is under 20%.
        RELAY_COLS[0] = int(_os.environ.get("QKV_RELAY_COL", RELAY_COLS[0]))

        # ---- kernels ----
        zero = FuncOp("proj_qmm_zero", ([yacc_l1, i32], []), visibility="private")
        zero.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        acc256 = FuncOp(
            "proj_qmm_acc256", ([xblk_l1, wblk_l1, yacc_l1], []), visibility="private"
        )
        acc256.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        # Cached-reduction acc + the arm that pins the cache at projection scope.
        acc256_c = FuncOp(
            "proj_qmm_acc256_c",
            ([xblk_l1, wblk_l1, yacc_l1, rcache_l1, i32, i32], []),
            visibility="private",
        )
        acc256_c.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        rc_arm = FuncOp("proj_qmm_rc_arm", ([rcache_l1, i32], []), visibility="private")
        rc_arm.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        flush_row = FuncOp(
            "proj_qmm_flush_row", ([yacc_l1, ypair_l1, i32], []), visibility="private"
        )
        flush_row.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        if BATCH > 1:
            # Batched projection: same three-entry-point split as the GEMV
            # (zero / accumulate / flush), for the same alloc-sinking reason --
            # one call that did all three would keep the accumulator alive
            # across the j loop's channel gets. Behind -DPROJ_MM_BATCH in
            # proj_qmm.cc; gated against the GEMV on device by proj_qmm_gate.py.
            mm_zero = FuncOp(
                "proj_qmm_mm_zero", ([yacc_mm_l1, i32], []), visibility="private"
            )
            mm_zero.attributes["link_with"] = StringAttr.get("proj_qmm.o")
            mm_acc = FuncOp(
                "proj_qmm_mm_acc",
                ([xblk_mm_l1, wblk_l1, yacc_mm_l1, wscr_mm_l1], []),
                visibility="private",
            )
            mm_acc.attributes["link_with"] = StringAttr.get("proj_qmm.o")
            # (y_acc, y_out, pair_role, tok_stride). tok_stride is a VALUE, not
            # a template parameter, so one kernel serves both pair roles and
            # any future non-paired layout.
            mm_flush = FuncOp(
                "proj_qmm_mm_flush_row",
                ([yacc_mm_l1, ypair_mm_l1, i32, i32], []),
                visibility="private",
            )
            mm_flush.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        # input-layernorm producer kernel (reproducer rms_norm_aie_hdr, lock-free).
        rms_norm_aie = FuncOp(
            "rms_norm_aie", ([rms_l1, rms_l1, rms_l1, i32], []), visibility="private"
        )
        rms_norm_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
        if N_NORMS >= 4:
            # Gemma: two norms per 2K weight channel. lo -> w[0:K], hi -> w[K:2K].
            rms_norm_lo_aie = FuncOp(
                "rms_norm_lo_aie",
                ([rms_l1, rms_l1, rms_w2k_l1, i32], []),
                visibility="private",
            )
            rms_norm_lo_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
            rms_norm_hi_aie = FuncOp(
                "rms_norm_hi_aie",
                ([rms_l1, rms_l1, rms_w2k_l1, i32], []),
                visibility="private",
            )
            rms_norm_hi_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
        if BATCH > 1:
            # The batched rms core's three entry points. Row and chunk indices
            # are i32 VALUES, not subviews: an extern C kernel takes a bare
            # pointer, so `memref + i` is the kernel's own arithmetic and the
            # FuncOp signature stays the whole buffer. Passing a
            # memref.subview would mean declaring a strided type that is not
            # the same type as the buffer, for no gain.
            rms_scale_row_aie = FuncOp(
                "rms_scale_row_aie",
                ([rscl_l1, rmsb_l1, i32, i32], []),
                visibility="private",
            )
            rms_scale_row_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
            # (y_chunk, x_batch, w, scales, batch, chunk, chunk_width)
            rms_chunk_aie = FuncOp(
                "rms_chunk_aie",
                ([rstg_l1, rmsb_l1, rms_l1, rscl_l1, i32, i32, i32], []),
                visibility="private",
            )
            rms_chunk_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
            if RMS_MEMTILE_REFEED == 3:
                # (y_row, x_batch, w, scales, row, row_stride, len)
                # RMS_ROW_OUT: one WHOLE ROW out per call instead of a cross-row
                # chunk, so each value is computed exactly once. See
                # docs/BZeroPlan.md and the kernel's own comment for why this
                # cannot just be rms_chunk with c=t (it would offset w as well).
                rms_row_aie = FuncOp(
                    "rms_row_aie",
                    ([rstg_l1, rmsb_l1, rms_l1, rscl_l1, i32, i32, i32], []),
                    visibility="private",
                )
                rms_row_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
            # (acc_batch, round, row, offset_in_row, round_width)
            residual_acc_row_aie = FuncOp(
                "residual_acc_row_aie",
                ([rmsb_l1, rstg_l1, i32, i32, i32], []),
                visibility="private",
            )
            residual_acc_row_aie.attributes["link_with"] = StringAttr.get(
                "rms_residual.o"
            )
            if RMS_BAND_STREAM:
                # Split rms_scale_row_aie's single full-row reduction across a
                # band loop, closed out by a finalize call (see rms_residual.cc).
                # Gated on the flag, not just unused-if-off: an mlir FuncOp decl
                # prints even with no call sites, so declaring these
                # unconditionally would cost check_batch1_noop's batch-8/
                # flag-off byte-identical invariant two lines for nothing.
                # (scales, x, row, row_stride, band_off, band_width, first, arm)
                rms_scale_row_partial_aie = FuncOp(
                    "rms_scale_row_partial_aie",
                    ([rscl_l1, rmsb_l1, i32, i32, i32, i32, i32, i32], []),
                    visibility="private",
                )
                rms_scale_row_partial_aie.attributes["link_with"] = StringAttr.get(
                    "rms_residual.o"
                )
                rms_scale_row_finalize_aie = FuncOp(
                    "rms_scale_row_finalize_aie",
                    ([rscl_l1, i32, i32], []),
                    visibility="private",
                )
                rms_scale_row_finalize_aie.attributes["link_with"] = StringAttr.get(
                    "rms_residual.o"
                )
            if RMS_BAND_STREAM >= 3:
                # Level 3's band == chunk == round (asserted at STG_W's
                # definition), so every fresh-fetch buffer these three take
                # is rstg_l1-shaped -- the same type `stg` already uses, not
                # a new one. func.call needs an exact type match, which is
                # why these are separate symbols from the whole-row kernels
                # above rather than the same one called with a smaller
                # width (see rms_residual.cc's own comment on this).
                # (scales, x_band, row, band_width, first, arm)
                rms_scale_row_partial_banded_aie = FuncOp(
                    "rms_scale_row_partial_banded_aie",
                    ([rscl_l1, rstg_l1, i32, i32, i32, i32], []),
                    visibility="private",
                )
                rms_scale_row_partial_banded_aie.attributes["link_with"] = (
                    StringAttr.get("rms_residual.o")
                )
                # (y_chunk, x_band, w, scales, batch, chunk, chunk_width)
                rms_chunk_banded_aie = FuncOp(
                    "rms_chunk_banded_aie",
                    ([rstg_l1, rstg_l1, rms_l1, rscl_l1, i32, i32, i32], []),
                    visibility="private",
                )
                rms_chunk_banded_aie.attributes["link_with"] = StringAttr.get(
                    "rms_residual.o"
                )
                # (acc_band, round, row, round_width)
                residual_acc_row_banded_aie = FuncOp(
                    "residual_acc_row_banded_aie",
                    ([rstg_l1, rstg_l1, i32, i32], []),
                    visibility="private",
                )
                residual_acc_row_banded_aie.attributes["link_with"] = StringAttr.get(
                    "rms_residual.o"
                )
                # (out_band, acc_band, round, row, band_width, round_width)
                residual_acc_row_banded_out_aie = FuncOp(
                    "residual_acc_row_banded_out_aie",
                    ([rstg_l1, rstg_l1, rstg_l1, i32, i32, i32], []),
                    visibility="private",
                )
                residual_acc_row_banded_out_aie.attributes["link_with"] = (
                    StringAttr.get("rms_residual.o")
                )
            if RMS_W_ON_X:
                # (norm_weight, band_fetch, elems) -- see RMS_W_ON_X.
                band_to_weight_aie = FuncOp(
                    "band_to_weight_aie",
                    ([rms_l1, rstg_l1, i32], []),
                    visibility="private",
                )
                band_to_weight_aie.attributes["link_with"] = StringAttr.get(
                    "rms_residual.o"
                )
            if _RMS_STG_COPY:
                band_copy_aie = FuncOp(
                    "band_copy_aie",
                    ([rstg_l1, rstg_l1, i32], []),
                    visibility="private",
                )
                band_copy_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
        # #4 faithful residual stream (reproducer rms_residual.o): residual_add_aie
        # (y = x_buf + x) for residual1 (input + o-proj-out) and residual2 (h + down-out).
        residual_add_aie = FuncOp(
            "residual_add_aie", ([rms_l1, rms_l1, rms_l1], []), visibility="private"
        )
        residual_add_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
        if ACC_STOP:
            # DECODE_ACC_STOP's batch-1 half: the batched body just drops an add,
            # but this one writes each residual into a fresh buffer, so stopping
            # it needs a copy rather than a skip.
            rms_copy_aie = FuncOp(
                "rms_copy_aie", ([rms_l1, rms_l1], []), visibility="private"
            )
            rms_copy_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
        # GLU: glu_aie(hid, x) = pseduo_glu<1024>: x = [hid 512 | gate 512],
        # hid(512) = silu(gate)*hid. One 1024 slice per call. Prebuilt glu.o.
        glu_aie = FuncOp(
            "glu_aie", ([glu_hid_l1, glu_x_l1, i32], []), visibility="private"
        )
        glu_aie.attributes["link_with"] = StringAttr.get("glu.o")
        if BATCH > 1:
            # (y_batch, x_batch, row, arm) -- one round, one token.
            glu_row_aie = FuncOp(
                "glu_row_aie",
                ([glu_hid_l1, glu_x_l1, i32, i32], []),
                visibility="private",
            )
            glu_row_aie.attributes["link_with"] = StringAttr.get("glu.o")
            # (y_batch, x_batch, src offset, len, arm) -- the LM head arm's
            # relay. See glu_copy_aie in glu.cc for why the logits are copied
            # into the OUTPUT buffer instead of sent straight back out of the
            # input one.
            glu_copy_aie = FuncOp(
                "glu_copy_aie",
                ([glu_hid_l1, glu_x_l1, i32, i32, i32], []),
                visibility="private",
            )
            glu_copy_aie.attributes["link_with"] = StringAttr.get("glu.o")
        # reference rope_compute(q,k,v, qkv, lut): rotate-half RoPE on Q,K (V copied).
        rope_compute = FuncOp(
            "rope_compute",
            ([ropeq_l1, ropekv_l1, ropekv_l1, qkv_l1, ropelut_l1, i32], []),
            visibility="private",
        )
        rope_compute.attributes["link_with"] = StringAttr.get("rope.o")
        # Multi-block (ATTN_L>1) flash-attention: reproducer model A. The block
        # COMPUTE (attn_qk_blk/attn_kv_blk/attn_kv_fin) is proven (attn_iso PASS
        # L=16..128); online-softmax state lives in L1 and persists across the
        # in-core block loop (reset on blk==0 inside the kernels).
        #   qk: m_state(16 bf16 running max) + c_state(8 f32)
        #   kv: y_state(512 f32 accumulator) + l_state(16 f32 denominator)
        m_l1 = MemRefType.get([16], bf16, memory_space=l1)
        c_l1 = MemRefType.get([8], f32, memory_space=l1)
        y_l1 = MemRefType.get([DQ_PADDED_PER_CU], f32, memory_space=l1)
        lden_l1 = MemRefType.get([16], f32, memory_space=l1)
        attn_qk_blk = FuncOp(
            "attn_qk_blk",
            ([aq_l1, ak_l1, m_l1, c_l1, as_l1, i32, i32], []),
            visibility="private",
        )
        _set_attn_link(attn_qk_blk, "attn_qk")
        if BATCH > 1:
            # Same call, plus which row of the resident q block to read. The
            # token index goes AFTER L so s_block stays the last memref and
            # AIR's shared-L1 classifier still tags this call the s producer.
            attn_qk_blk_row = FuncOp(
                "attn_qk_blk_row",
                ([aq_l1, ak_l1, m_l1, c_l1, as_l1, i32, i32, i32], []),
                visibility="private",
            )
            _set_attn_link(attn_qk_blk_row, "attn_qk")
        attn_kv_blk = FuncOp(
            "attn_kv_blk",
            ([as_l1, av_l1, y_l1, lden_l1, i32, i32], []),
            visibility="private",
        )
        _set_attn_link(attn_kv_blk, "attn_kv")
        attn_kv_fin = FuncOp(
            "attn_kv_fin", ([y_l1, lden_l1, ao_l1], []), visibility="private"
        )
        _set_attn_link(attn_kv_fin, "attn_kv")
        if BATCH > 1:
            attn_kv_fin_row = FuncOp(
                "attn_kv_fin_row",
                ([y_l1, lden_l1, ao_l1, i32], []),
                visibility="private",
            )
            _set_attn_link(attn_kv_fin_row, "attn_kv")

        # ---- channels ----
        # Faithful X-feed: host raw X (@xy) + rms weight (@rmsin) -> rms core ->
        # xnorm (re-fed N times on-chip, see refeed()) -> X memtile (512) ->
        # 256-block broadcast to all 16 cores. (reproducer core_2_2 + mem_1_1 x_buffer)
        # #4 (FULL4): rmsX is PACKET so it converges with the id4 demux (o-proj+down)
        # on the rms core's S2MM0 -- the reference's tile_2_2 DMA0 receives @xy(id0)+id4
        # both as packets into one 2-slot ping-pong (input, then o-proj, then down).
        # Debug configs keep the original circuit rmsX.
        # CIRCUIT-SWITCHED AT LEVEL 3. The reason above no longer
        # applies -- RMS_W_ON_X pins @rmsX to S2MM0 and @outY to S2MM1, so the
        # two do not converge on one port and @rmsX has no id to demux against.
        # What it buys is a DEDICATED physical stream channel on the link both
        # flows share (shim 2,0 and memtile 2,1 are both SOUTH of the rms core,
        # so packet_flow(0) and packet_flow(30) are multiplexed on one
        # south->north stream). Packet-switched, a band the core has no credit
        # for parks at the head of that link, and the @outY round behind it
        # never arrives -- which is measured: residual2 blocks on its @outY
        # acquire, and only when residual2 also has an @rmsX get to leave bands
        # queued. _RMS_X_PACKET=1 restores the packet flow, which is the A
        # side of that measurement (see docs/DFlashFeasibility.md 5.4 x).
        _rx_packet = FULL4 and not (RMS_BAND_STREAM >= 3 and BATCH > 1)
        if _os.environ.get("_RMS_X_PACKET"):
            _rx_packet = FULL4  # A/B: put level 3 back on the packet flow
        if _rx_packet:
            _rx = channel_decl("rmsX", size=[1], channel_type="npu_dma_packet")
        else:
            _rx = channel_decl("rmsX", size=[1])
        # tapsX: the extra waves' X, shim -> the RELAY MEMTILE, bypassing the
        # rms core entirely. Only declared where it is used, so every other
        # build's channel set is untouched.
        #
        # It exists because an extra arm's X cannot reach the X memtile any
        # other way under mode 3. It cannot ride @xnorm (that lands on the
        # relay, whose fill an extra arm must not add -- the [12, 38] cycle),
        # it cannot come off the rms core on a second channel (two MM2S, both
        # taken), and it cannot be a shim -> X-memtile channel (that feed
        # collapses to one structure before flows are built, §3.15).
        #
        # What works is a SECOND FILL OF THE SAME SLAB. The relay's drain is
        # count-free and self-loops -- the count lives on the fill -- so col 4
        # can be filled from @xnorm at 12/38 on the decode arm and from @tapsX
        # at 1 here, on a DIFFERENT S2MM chain (chains are per tile AND
        # channel), while both share the one drain BD into @xin. Each chain
        # returns to its own BD 0, which is the rule the shared chain imposes.
        _tx = (
            channel_decl("tapsX", size=[1])
            if (N_EXTRA and RMS_MEMTILE_REFEED == 3)
            else None
        )
        _rw = channel_decl("rmsW", size=[1])
        if POST_RMS:
            # Separate channel for the post_attention_layernorm weight. A single
            # rmsW FIFO re-fed twice does NOT pair in AIR (both gets read the same
            # put -> decode diverges). The rms compute tile has only 2 S2MM
            # (rmsX-convergent + rmsW), so rmsW2 packet-muxes onto the rmsX S2MM;
            # to keep the vocab-active rmsX packet group hole-free, vocab feeds AND
            # consumes a dummy rmsW2 (see _uni_voc / _rms_lm_head).
            _rw2 = channel_decl("rmsW2", size=[1])
            # Gemma sandwich (N_NORMS>=4): the 4 norm weights are packed 2-per-channel
            # (rmsW = [input | post_attn], rmsW2 = [pre_ffn | post_ffn], each 2K) so the
            # rms tile's S2MM0 carries only {rmsX, rmsW, rmsW2, o-proj/down} = 4 packet
            # ids (a compute-tile S2MM port demuxes at most 4). No rmsW3/rmsW4 channels.
        # _RMS_PIN=1 forces the level-3 S2MM pinning at any level. Bisection
        # hatch: it makes the port re-assignment testable against the KNOWN-GOOD
        # level-0 batch-8 design, so a level-3 hang can be attributed to the
        # pinning or cleared of it.
        if RMS_BAND_STREAM >= 3 or _os.environ.get("_RMS_PIN"):
            # LEVEL 3 NEEDS rmsX ALONE ON ITS PHYSICAL S2MM, and nothing below
            # level 3 does. air-to-aie turns a tile's per-channel memcpy list
            # into BD *tasks* -- one per run of equal static trip counts
            # (getRepeatCounts) -- and only emits a count-free, lock-driven
            # infinite BD loop when the whole physical channel collapses to
            # ONE task (generateDmaBdProgram's `infiniteBDLoopMode =
            # repeat_counts.size() == 1`). Level 3's regen fetch sits inside
            # the arm-selected re-feed loop, whose trip count is a RUNTIME
            # value (_cnt's index_switch), so no finite repeat_count can ever
            # be right for it: it has to be count-free. Sharing the port with
            # rmsW/rmsW2 splits it -- their gets land BETWEEN the rmsX gets in
            # program order, giving [rmsW][rmsX x2][rmsW2][rmsX x4] as four
            # tasks -- and the finite chains that result under-deliver (2 BDs
            # x repeat 4 = 10 transfers against 5 + 5*XN_REFEED needed), which
            # is a wave-0 hang with nothing written.
            #
            # This is the same remedy air-to-aie applies on its own to an
            # arm-varying get inside an scf.index_switch
            # (dedicateArmVaryingGetChannels, "must own a dedicated count-free
            # S2MM"); it cannot fire here because _rms_batched is deliberately
            # ONE body for both arms rather than two index_switch regions, so
            # the pin is set by hand. Levels 0-2 keep AIR's own allocation --
            # their rmsX is a single whole-row get with a static count and has
            # no such requirement.
            # With RMS_W_ON_X the rms core's inbound set is just {rmsX, outY},
            # and @outY is already pinned to 1 by its own decl -- so rmsX gets 0
            # and is alone on it, which is the whole requirement. The pin is
            # still explicit rather than left to the round-robin allocator:
            # `simpleDmaChannelAlloc` reuses an existing packet-flow allocation
            # on the tile when one is free, and rmsX sharing outY's port would
            # silently put it back in a multi-op task.
            _rx.operation.attributes["air.tile_dma_channel"] = IntegerAttr.get(i32, 0)
        # FAITHFUL convergent X feed (reproducer x_buffer DMA:3): ONE channel
        # carries BOTH the rmsnorm'd token X (phases 0..2) and the GLU-output X
        # (down phase), as convergent packet sources into ONE X-memtile S2MM, read
        # by ONE count-free feed loop. A single feed loop => one repeat count =>
        # air-to-aie infinite (count-free) BD mode, NO repeat_count (which is a
        # stale-rebroadcast deadlock). Two separate feed loops (the prior bug)
        # lowered to two repeat_count tasks. Packet (npu_dma_packet) so the two
        # producers (rms core L1 + down_buffer L2, time-disjoint, same id) converge.
        _xn = channel_decl("xnorm", size=[1], channel_type="npu_dma_packet")

        # ---- DECODE_BATCH: the @xnorm stream order ----
        # Every producer on @xnorm (rms, the attn-O memtile, the down/GLU
        # memtile) feeds the X memtile, which stages one 2*COL_BLOCK window of
        # ALL B tokens and broadcasts it tile-blocked. So the stream has to
        # arrive CHUNK-major and token-minor --
        #   [chunk 0: t0..tB-1][chunk 1: t0..tB-1]...
        # -- not token-major, or the memtile's contiguous get would land one
        # token's whole row where it wants one chunk of every token.
        #
        # This is the transpose that pairs with xfeed_bd.py's: this one gets the
        # B tokens' chunk side by side in the memtile, that one puts them in
        # aie::mmul A-tile order on the way out.
        _XCHUNK = XCHUNK

        def _xnorm_put(buf, width, ssa=False, chan="xnorm", **kw):
            """Put `width` elements per token on @xnorm, chunk-major.

            `ssa` picks index-SSA operands over static attributes for the
            batch-1 form. Both lower the same, but the attn-O site was written
            with SSA operands and the batch-1 build has to stay byte-identical,
            so the distinction is preserved rather than normalised away.
            """
            if BATCH == 1:
                _w = idx if ssa else (lambda v: v)
                return ChannelPut(
                    chan,
                    buf,
                    offsets=[_w(0)],
                    sizes=[_w(width)],
                    strides=[_w(1)],
                    **kw,
                )
            assert width % _XCHUNK == 0, (
                f"@xnorm producer width {width} is not a multiple of "
                f"{_XCHUNK}; the X memtile stages whole chunks"
            )
            return ChannelPut(
                chan,
                buf,
                offsets=[idx(0), idx(0), idx(0)],
                sizes=[idx(width // _XCHUNK), idx(BATCH), idx(_XCHUNK)],
                strides=[idx(_XCHUNK), idx(width), idx(1)],
                **kw,
            )

        if KV_APPEND:
            # Pin the rms core's two outputs (xnorm o-proj-X feedback -> mem_2_1 on
            # MM2S1; layerOut -> shim on MM2S0) to their known-good split. Adding the
            # append channels otherwise perturbs the global placer into packing BOTH
            # onto rms MM2S0 (dual-fan packet), which flips layerOut circuit->packet
            # and deadlocks. Only the rms core is a compute-tile endpoint of these
            # channels (consumers are memtile/shim), so the pin is local to it.
            _xn.operation.attributes["air.tile_dma_channel"] = IntegerAttr.get(
                T.i32(), 1
            )
        # (FAITHFUL ph2): no toBufP2 / buf_ph2 channel -- the rms core emits ph2 X
        # = rmsnorm(x+oproj) directly on @xnorm. Every re-feed on this channel is
        # written as an n-trip loop around the put (see refeed()); the counts are
        # XN_REFEED for ph0 and REFEED[GATEUP_PHASE] for ph2.
        # air.shared_resident_ring: the two per-v1 GEMV emit passes each re-read
        # this same broadcast X/W stream; mark so air-ping-pong-transform merges
        # the two sibling get-loops onto ONE 2-deep resident ring (reproducer's
        # w_0/w_1) instead of two independent rings (which air-to-aie fuses into a
        # 4-deep interleaved ring -> wrong coverage).
        _inX = Channel("inX", size=[1, 1], broadcast_shape=[NCX, NCY])
        _inX.operation.attributes["air.shared_resident_ring"] = UnitAttr.get()
        # ATTN S1: rope LUT (cos/sin, 64) -> rope core S2MM1. Placeholder source =
        # RMS[0:64] for the dataflow test; the real cos/sin LUT is wired in S4.
        channel_decl("ropeLUT", size=[1])
        if BATCH > 1:
            # QKV drain, transposed to token-major in L2 (see qkvmt_l2). Only
            # exists when batched: at batch 1 the rope core reads @outY dest 0
            # directly, which is the reference-faithful path and the one that
            # does NOT re-introduce the col-2 memtile relay that deadlocked the
            # fused vocab build.
            channel_decl("toRope", size=[1])
        # S3a flash-attn dataflow: rope q -> qk tile (direct); rope k|v -> KV
        # staging memtile (rope's single k/v MM2S) which splits k->qk, v->kv.
        _ropeQ = channel_decl(
            "ropeQ", size=[1]
        )  # rope q (whole 2048) -> q broadcast memtile
        if KV_APPEND:
            # Pin roped Q to rope MM2S0 so the K/V append (pinned to MM2S1
            # below) does not steal it -- matches the reference (Q on 1st MM2S, K/V append
            # on 2nd). Without this the placer puts the packet append on MM2S0
            # (allocated first) and shoves Q to MM2S1, deadlocking the front-end.
            _ropeQ.operation.attributes["air.tile_dma_channel"] = IntegerAttr.get(
                T.i32(), 0
            )
        channel_decl("toAttnQ", size=[N_ATTN_CU])
        # rope k/v -> kv memtiles. PACKET so one rope MM2S can fan to memtiles on
        # MULTIPLE cols (mem_3_1 + mem_4_1) -- the reference routes rope k/v as
        # packets (id14/15) for exactly this multi-destination fan-out. A circuit
        # channel is point-to-point and deadlocks feeding 2 col-memtiles.
        channel_decl(
            "toAttnKV",
            size=[len(ATTN_COL_GROUPS)],
            channel_type="npu_dma_packet",
        )
        channel_decl("toK", size=[N_ATTN_CU])
        channel_decl("toV", size=[N_ATTN_CU])
        if MULTIBLK:
            # the reference full-faithful KV cache (DDR): rope appends this token's roped
            # K/V into the cache (appendK/appendV -> KVC at APPEND_OFF); the whole
            # cache is streamed back per CU (inKV) into a readback memtile that
            # re-blocks into per-block toK/toV.
            if KV_APPEND:
                # the reference-faithful: rope K/V -> shim S2MM -> DDR, mirroring
                # reference pkt14/15. PACKET (not circuit) so the append can leave
                # rope on its 2nd MM2S and fan to distinct cols. TWO channel DECLS
                # so the allocator can place them independently -- one decl's
                # sub-channels are treated as a single logical transfer and may
                # share a shim channel, which is the opposite of what is wanted
                # here. The columns are no longer pinned: the allocator spreads
                # independent packet readbacks over distinct shim tiles, keeping
                # the append off rope's own col2 (whose congestion deadlocks the
                # front-end) without air.shim_col.
                _apK = channel_decl("appendK", size=[1], channel_type="npu_dma_packet")
                _apK.operation.attributes["air.tile_dma_channel"] = IntegerAttr.get(
                    T.i32(), 1
                )
                _apV = channel_decl("appendV", size=[1], channel_type="npu_dma_packet")
                _apV.operation.attributes["air.tile_dma_channel"] = IntegerAttr.get(
                    T.i32(), 1
                )
            if KV_SPLIT:
                # the reference mem_3_1: K and V on SEPARATE shim->memtile flows (one each per
                # col group of 2 CUs), so their memtile S2MM fills are independent.
                channel_decl("inKV_K", size=[len(ATTN_COL_GROUPS)])
                channel_decl("inKV_V", size=[len(ATTN_COL_GROUPS)])
            else:
                channel_decl("inKV", size=[N_ATTN_CU])
        channel_decl("attnO", size=[N_ATTN_CU])
        # One tap channel per SOURCE TILE, not per tap (see DECODE_PROBE).
        if PROBE_5_LEN:
            channel_decl("probe5", size=[1])
        if PROBE_LEN["D"]:
            channel_decl("probe4", size=[1])
        # W: host (per col) -> group memtile -> NCY cores.
        if W_DUAL_CHAN:
            # PER-COLUMN channels rather than one [NCX] bundle, so that both of a
            # column's shim MM2S channels sit on THAT column's shim tile -- the
            # whole point is 2x the per-column weight bandwidth. The columns are
            # not stated here: each channel feeds an L2 buffer the allocator
            # already buckets by column, and AIRToAIE stamps that bucket column
            # on the shim tile it opens.
            for _wc in range(NCX):
                for _nm in (_wname(0, _wc), _wname(1, _wc)):
                    channel_decl(_nm, size=[1])
        else:
            channel_decl("inW", size=[NCX])
        _wL2 = channel_decl("wL2ToL1", size=[NCX, NCY])
        _wL2.operation.attributes["air.shared_resident_ring"] = UnitAttr.get()
        # Output: leads -> group MT -> main MT -> id-demux egress.
        _outA = channel_decl(
            "outA", size=[NCX, PAIRS_PC], channel_type="npu_dma_packet"
        )
        # No keep_pkt_header and no packet_ids. Both are DERIVED.
        #
        # This hop carries a routing decision made here (the dest operand on its
        # put) to a switchbox further downstream, so it MUST preserve the header
        # -- forced by the topology, not a choice, and air-annotate-packet-ids
        # injects it. The ids likewise: the hop is single-destination, so its
        # list is just the demux's set with no meaningful order.
        _toMain = channel_decl("toMain", size=[N_GRP], channel_type="npu_dma_packet")
        # keep_pkt_header derived, as for @outA above.
        # id-demux egress (reproducer mem_1_1 DMA5): the main MT emits each round's
        # assembled 514 packet (carrying the routing header) on ONE MM2S; the
        # switchbox routes the id allocated for dest p (broadcast_shape=[1,NDEST]).
        _outY = Channel("outY", size=[1, 1], broadcast_shape=[1, NDEST])
        _outY.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")
        if BATCH > 1:
            # Pin the COMPUTE-tile ends of the demux to S2MM1, which is where
            # batch 1 puts them anyway.
            #
            # Without this the batched rms core lands outY on S2MM0 -- the port
            # that already carries rmsX, rmsW and rmsW2 -- as a SECOND BD chain,
            # and a DMA channel has one. The emitted AIE dialect shows it: two
            # aie.dma_start(S2MM, 0) on tile_2_2, and the first chain stops
            # cycling, so layer 1 never receives its X. A batched build hangs
            # with no message.
            #
            # The trigger is buffer aliasing, which the batched rms body does on
            # purpose: its staging buffer is both the outY destination and the
            # @xnorm source, so the allocator's packet-flow reuse (it folds a
            # packet flow onto any existing packet allocation in the tile --
            # AIRToAIESchedulingUtils.cpp) picks the wrong port. The pin is an
            # explicit override and is applied even when a channel was already
            # chosen. The memtile ends ignore it by design.
            _outY.operation.attributes["air.tile_dma_channel"] = IntegerAttr.get(
                T.i32(), 1
            )
        # Faithful demux: route by the kernel header, then STRIP it at every
        # dest -> pure payload (PAYLOAD=512) delivered. The main MT PUT stays
        # MAIN_ROWS=514 (header included, for routing); the gets are PAYLOAD.
        #
        # Nothing is declared here either. That this channel demuxes is derived:
        # its destinations partition the stream, and a put naming a `dest`
        # reaches it through the hops above. A dest operand can only mean "this
        # packet is for that leaf", which is exactly the statement that the
        # fanout is over time rather than space.
        # No packet_ids. The ids are ALLOCATED by air-annotate-packet-ids: it
        # reads the demux shape (dests partition the stream) for the count,
        # takes them from the top of the id space so nothing else is
        # renumbered, and rewrites the ordinals the kernel stamps to match.
        channel_decl("toShim", size=[NDEST])
        # #4: layer output (residual2 = h + down) drained to host from the rms core.
        if FULL4:
            _lo = channel_decl("layerOut", size=[1])
            if KV_APPEND:
                # Keep layerOut on rms MM2S0 (circuit) so xnorm (pinned to MM2S1)
                # does not share/flip it to packet. See xnorm pin above.
                _lo.operation.attributes["air.tile_dma_channel"] = IntegerAttr.get(
                    T.i32(), 0
                )
        # GLU path: id-demux delivers gate-up DIRECTLY to the GLU herd (no relay);
        # GLU -> gluOut -> down memtile accumulate (8192). FAITHFUL: that 8192 is
        # fed back on-chip as the DOWN phase X by the down_buffer re-broadcasting it
        # DOWN_REFEED times into the convergent @xnorm channel (counting-lock-N on
        # the down_buffer fill, derived from the DOWN_REFEED loop around the put
        # below), NOT drained to host.
        channel_decl("gluOut", size=[1])
        if RMS_MEMTILE_REFEED == 3:
            # MODE 3 REPOINTS @xnorm INSTEAD OF ADDING A CHANNEL.
            #
            # The rms core has TWO MM2S ports and both are taken (@layerOut,
            # @xnorm), so the whole rows cannot get their own channel out of the
            # core -- a third flow wraps onto MM2S 0 and both chains there lose
            # their self-loop. What is NOT fixed is where @xnorm goes: it is a
            # name. So @xnorm now carries the core's whole rows to the RELAY
            # memtile, and two new channels carry the re-broadcast on to the X
            # memtile.
            #
            #   rms core --@xnorm (2560-element rows, ONE BD)--> col RMSROW_COL
            #   col RMSROW_COL --@xin0 (slab0 x REFEED[0])--> X memtile   ph0
            #   col RMSROW_COL --@xin2 (slab2 x GATEUP)   --> X memtile   ph2
            #   col 5          --@xin0 (attnO x OPROJ)    --> X memtile   ph1
            #   col 3          --@xin2 (gluOut x DOWN)    --> X memtile   ph3
            #
            # ph1 rides on @xin0 behind ph0 and ph3 on @xin2 behind ph2, in
            # phase-time order. That keeps the X memtile at TWO inbound channels
            # -- it has exactly two free S2MM once @xnorm stops arriving there
            # (five of its six are spent, four of them on a 4-way striped fill
            # of one 4098-element buffer that cannot be narrowed without costing
            # ingest bandwidth) -- and each of the two is convergent from two
            # DIFFERENT tiles, which is the arrangement @xnorm already uses.
            #
            # air.dedicated_dma_channel on both: without it AIR's MM2S allocator
            # "collapses promiscuously onto a packet-flow channel" and puts the
            # relay tile's two drains on ONE physical channel, whose BDs then
            # alternate ph0, ph2, ph0, ph2 while the consumer needs all of ph0
            # first. Two distinct channels also mean two distinct packet ids,
            # which the ROUTER requires: two packet flows sharing an id from two
            # ports of one tile fail with "Unable to find a legal routing"
            # (measured -- see docs/BZeroPlan.md).
            _xc = channel_decl("xin", size=[1], channel_type="npu_dma_packet")
            _xc.operation.attributes["air.dedicated_dma_channel"] = UnitAttr.get()
        # Batched LM head: the GLU core's second MM2S, carrying the vocab logits to
        # the host. Its own channel and not @layerOut, because a channel with two
        # producers is a shim fan-in and air-to-aie refuses those ("Fan-in for
        # aie.flow isn't supported"). Only declared when it is used -- at batch 1
        # the rms core forwards the logits and this core stays gate-up-only.
        if BATCH > 1 and GLU_DEST >= 0:
            channel_decl("vocOut", size=[1])

        def idx(v):
            return arith.ConstantOp.create_index(v)

        def _mm_ring_deeper(J2x2, xblk_ty, wblk_ty, mm_acc_fn, a_acc, a_ws, gcx, gcy):
            """PROJ_RING_DEPTH > 2: the SAME alloc/get/compute/dealloc-per-
            iteration shape `air-label-scf-for-to-ping-pong` looks for and
            duplicates to depth 2 -- but pre-tagged with `unroll` = N so its
            own hardcoded factor never gets the chance to fire.

            `isPingPongCandidate` (AIRDependencyScheduleOpt.cpp) opens with
            `if (forOp->hasAttr("unroll")) return false;` -- an already-
            labeled loop is left alone. Setting the attribute here, before
            the standard aircc pipeline runs, makes this loop already-
            labeled, and the SEPARATE downstream transform
            (AIRPingPongTransformationPattern) reads "unroll" generically as
            an integer everywhere it is consumed, not hardcoded to 2. So this
            is not new duplication/rotation logic -- it borrows the pass's
            own, tested machinery for whatever N is asked for.

            Two things were tried FIRST and rejected, both because they gave
            each of N buffers a genuinely persistent lifetime across the
            whole loop (hoisted alloc, one dealloc at the end): a loop-carried
            phase (an iter_arg) and a remui-selected index_switch. Both
            emitted valid IR from this Python layer, and both failed the SAME
            way in air-opt's `air-dependency` pass -- "operand #0 does not
            dominate this use" on the final dealloc, because a buffer written
            from inside a conditional branch, on some iteration nobody can
            resolve statically, has no single last-writer for the pass to
            hang a token on. Tagging the ORIGINAL single-buffer-per-iteration
            loop avoids the question entirely: the shape that already works
            is unchanged, only how many physical instances back it changes.
            """
            for_op = ForOp(idx(0), J2x2, idx(1))
            for_op.operation.attributes["unroll"] = IntegerAttr.get(
                i32, PROJ_RING_DEPTH
            )
            with InsertionPoint(for_op.body):
                a_x = AllocOp(xblk_ty, [], [])
                ChannelGet("inX", a_x, indices=[gcx, gcy])
                a_w = AllocOp(wblk_ty, [], [])
                ChannelGet("wL2ToL1", a_w, indices=[gcx, gcy])
                CallOp(mm_acc_fn, [a_x, a_w, a_acc, a_ws])
                DeallocOp(a_x)
                DeallocOp(a_w)
                yield_([])

        def _probe_put(name, chan, buf):
            """Emit one DECODE_PROBE tap, or nothing when its bit is off.

            A no-op by default, and that is the contract: the batch-1 diff has
            to stay byte-identical with the taps compiled in.

            EMITTED AFTER the shipping transfer it observes, never before, and
            that placement is the whole of what makes the taps usable. In front,
            a tap takes the memtile's first free MM2S and pushes the shipping
            flow up one -- mem_tile_4_1's @xnorm has no route to the X memtile
            from MM2S1, so the build does not place -- and pinning the tap higher
            only trades that for a stall: the q tap sat in front of a fan whose
            first link it was now gating, and reached the shim with 4 of 8 tokens.
            Behind the flow it observes, a tap reads a buffer that is already
            finished with, and is a plain extra reader.
            """
            if not PROBE_LEN[name]:
                return
            _p = ChannelPut(
                chan,
                buf,
                offsets=[idx(0)],
                sizes=[idx(PROBE_LEN[name])],
                strides=[idx(1)],
            )
            # AND pinned high. Emission order is not allocation order here: even
            # behind the re-broadcast it observes, an unpinned tap takes the
            # memtile's MM2S0 and leaves @xnorm on MM2S1, which does not route to
            # the X memtile from either mem_tile_4_1 or mem_tile_5_1.
            _p.operation.attributes["air.memtile_dma_channel_min"] = IntegerAttr.get(
                T.i32(), 2
            )

        def grp_leads(g):
            # the leads (cx,pp) of group g, header-bearer first (k==0). Group g owns
            # NCX//N_GRP consecutive logical columns: llama 2 cols/group (g*2+lc),
            # gemma per-column 1 col/group (g*1 = g).
            _cpg = NCX // N_GRP  # columns per group
            out = []
            for lc in range(_cpg):
                for pp in range(PAIRS_PC):
                    out.append((g * _cpg + lc, pp))
            return out

        # MULTIBLK adds a 5th DDR arg (kv_cache) so the reference's append+readback (and the reference's
        # _gen_sequence) can drive it; the L=1 ABI (4 args) is unchanged when MULTIBLK
        # is off, preserving the bring-up/PASS interface.
        # DYNSEQ appends the context length as a trailing scalar. Kept last so the
        # DDR argument positions -- and every host binding built around them --
        # are unchanged.
        _fn_args = (
            [x_l3, w_l3, rms_l3, y_l3]
            + ([kvc_l3] if MULTIBLK else [])
            + _w_extra
            + ([i32] if DYNSEQ else [])
        )
        # arg index of each weight group's buffer: group 0 is the original arg1;
        # groups 1.. and the lm-head follow the base args. Index into _la is +4.
        _w_base_n = 4 + (1 if MULTIBLK else 0)
        WARG = [1] + [_w_base_n + i for i in range(len(_w_extra))]

        @FuncOp.from_py_func(*_fn_args)
        def q4nx_decode(*_fa):
            X, W, RMS, Y = _fa[0], _fa[1], _fa[2], _fa[3]

            # air.preserve_shim_dma_order: opt out of air-opt-shim-dma-bds'
            # per-channel BD regrouping. The weight channels (inW) are coupled by
            # the X broadcast multicast (all cores advance in lockstep), so the
            # round-major put order is load-bearing and must not be reordered
            # into per-channel (channel-major) BDs.
            def launch_body(*_la):
                X, W, RMS, Y = _la[4], _la[5], _la[6], _la[7]
                # W_SPLIT: every weight buffer, in group order, lm-head last. Which
                # one a weight feed names is decided per-wave inside _feed_wcol; see
                # _wsel below.
                _WBUFS = [_la[4 + a] for a in WARG] if W_SPLIT else [W]
                # The extra waves' buffer is the last appended arg either way.
                _WEXTRA = _la[4 + WARG[-1]] if N_EXTRA else None
                # None => the caller is on the statically-known lm-head buffer; a
                # Value => a runtime group index into _WBUFS[0:N_WGRP].
                _wsel = [None]
                KVC = _la[8] if MULTIBLK else None
                # The dispatch-time context length (DYNSEQ). Last operand before
                # the multi-layer induction variable.
                L_rt = _la[4 + len(_fa) - 1] if DYNSEQ else None

                def _rt_blocks():
                    """ceil(L_blk/16) as an index Value, for the readback's block count.

                    Kept in i32: aie-translate's C++ TXN target emits the integer
                    widths but has no case for index-typed arithmetic, and this
                    expression has to survive all the way into the emitted builder.
                    """
                    _lb0, _ = _split_mask_mode(L_rt)
                    _s = arith.addi(
                        _lb0,
                        arith.ConstantOp(
                            IntegerAttr.get(i32, 15 + BATCH - 1), None
                        ).result,
                    )
                    _q = arith.divui(
                        _s, arith.ConstantOp(IntegerAttr.get(i32, 16), None).result
                    )
                    return arith.index_cast(idx_t, _q)

                # Multi-layer fused decode: the device (segment/herds, below) is
                # emitted ONCE and reused temporally. When NLAYERS>1 the launch is
                # wrapped in an AIR scf.for (see the emit branch after this def) whose
                # induction variable a_iv scales the per-layer DDR offsets; the loop is
                # unrolled LATE (airrt-to-npu), so the AIR op count stays CONSTANT and
                # air-to-aie does not blow up. When NLAYERS==1 there is no scf.for and
                # no a_iv, so the feeds are byte-identical to the single-layer design
                # (a_iv is None -> Python-const 0 offsets). x is chained in-place:
                # layer k's res2 is written back to arg0[0], which layer k+1 reads as
                # its rmsX input.
                a_iv = _la[-1] if len(_la) > 4 + len(_fa) else None

                # Per-layer offset helpers. a_iv is None (single-layer): plain Python
                # ints, byte-identical to the original single-layer feeds. a_iv is a
                # runtime index Value (multi-layer): arith index ops scaled by the
                # scf.for induction variable.
                def _lb(slab):  # per-layer base = a_iv * slab
                    return 0 if a_iv is None else arith.muli(a_iv, idx(slab))

                def _lo(base, extra):  # base + extra, raw-int (lazy-const) form
                    if a_iv is None:
                        return base + extra
                    return arith.addi(base, idx(extra)) if extra else base

                def _loi(base, extra):  # base + extra, idx()-wrapped (eager-const) form
                    if a_iv is None:
                        return idx(base + extra)
                    return arith.addi(base, idx(extra)) if extra else base

                def _slot_off():
                    """(L-1) * REGION_W: this token's slot within a KV region.

                    DYNSEQ makes it a runtime address -- the append has to land on
                    the position being generated, not on the compile-time one.
                    """
                    if not DYNSEQ_APPEND:
                        return (ATTN_L - 1) * REGION_W
                    _lb1, _ = _split_mask_mode(L_rt)
                    _m = arith.subi(
                        _lb1, arith.ConstantOp(IntegerAttr.get(i32, 1), None).result
                    )
                    _p = arith.muli(
                        _m,
                        arith.ConstantOp(IntegerAttr.get(i32, REGION_W), None).result,
                    )
                    return arith.index_cast(idx_t, _p)

                def _loi_slot(base, extra):
                    """_loi with the runtime slot offset folded in."""
                    if not DYNSEQ_APPEND:
                        return _loi(base, extra + (ATTN_L - 1) * REGION_W)
                    _b = base if a_iv is not None else idx(base)
                    _s = arith.addi(_b, _slot_off())
                    return arith.addi(_s, idx(extra)) if extra else _s

                # X slot for this wave (see HIDDEN_TAPS). Without taps both are
                # the literal 0 and the chain is in-place exactly as before.
                # A slot is one whole BLOCK of hidden states, so it strides by
                # BATCH*K -- the block is what a layer consumes and produces.
                _X_SLOT = BATCH * K

                def _x_in():
                    """Slot layer a_iv READS: its input hidden state."""
                    return _lb(_X_SLOT) if HIDDEN_TAPS else 0

                def _x_out():
                    """Slot layer a_iv WRITES: its output, the next layer's input."""
                    return _lo(_lb(_X_SLOT), _X_SLOT) if HIDDEN_TAPS else 0

                # The LM head reads the LAST slot -- what layer UNI_DEC-1 wrote.
                # Its own wave index is >= UNI_DEC, so this is a constant, not a
                # function of a_iv.
                _x_final = (UNI_DEC * _X_SLOT) if HIDDEN_TAPS else 0

                blk = BLOCK_BF16
                wstep = NCY * blk  # 10240 = one fan get

                def _feed_wcol(_cx, _cbase, nsteps, Wb):
                    """One proj column's weight slab for one phase.

                    Single-channel (default): TWO contiguous puts on @inW so the
                    shim-dma coalescer merges+tags them (air.coalesced_shim_feed =
                    the cross-channel phase barrier); a single put would skip
                    coalescing and lose that barrier.

                    W_DUAL_CHAN: the slab is packed [low-row half | high-row half]
                    (see pack_q4k_cascade(dual_chan=True)), so channel 0 takes the
                    first half of the column -- every fan step's cy 0..NCY/2-1
                    blocks -- and channel 1 the second. Each is still one contiguous
                    run, and each is still fed as two puts so BOTH keep their
                    coalesced tag and both appear in the phase group. A phase group
                    is the 2*NCX distinct channels; the next phase's repeat opens a
                    new group (AIRRtToNpuPass phase-barrier grouping).

                    `_cx` is a Python int when W_DUAL_CHAN (per-column channels), and
                    the scf.parallel bundle IV otherwise.
                    """
                    nch = 2 if W_DUAL_CHAN else 1
                    cstep = wstep // nch  # this channel's per-step share
                    span = nsteps * cstep  # elements per channel
                    half = (nsteps // 2) * cstep  # step-aligned inner split
                    for ci in range(nch):
                        base = _cbase if ci == 0 else arith.addi(_cbase, idx(span))
                        ch = _wname(ci, _cx)
                        ix = [idx(0)] if W_DUAL_CHAN else [_cx]
                        ChannelPut(
                            ch,
                            Wb,
                            indices=ix,
                            offsets=[base],
                            sizes=[half],
                            strides=[1],
                        )
                        ChannelPut(
                            ch,
                            Wb,
                            indices=ix,
                            offsets=[arith.addi(base, idx(half))],
                            sizes=[span - half],
                            strides=[1],
                        )

                def _feed_wcols(_wcol0, _colspan, nsteps):
                    """Fan the phase's weight slab over the NCX proj columns.

                    W_DUAL_CHAN uses per-column pinned channels, so the column index
                    has to be a Python constant -> plain unrolled loop. Otherwise the
                    bundle index must be an scf.parallel IV (canonical form; a
                    temporal scf.for over a bundle index is a verifier error), and
                    air-to-aie spatially unrolls it to the same per-column feeds.

                    W_SPLIT wraps the WHOLE fan -- not each column -- in the weight-arg
                    select, because AIRRtToNpuPass's cross-channel phase barrier opens
                    a new phase group at every block boundary. A per-column switch puts
                    each column in its own block, which shrinks the phase group from
                    2*NCX channels to 2 and deadlocks the dispatch.
                    """

                    def _fan(Wb):
                        if W_DUAL_CHAN:
                            for _cxi in range(NCX):
                                _cbase = (
                                    _wcol0 + _cxi * _colspan
                                    if isinstance(_wcol0, int)
                                    else arith.addi(_wcol0, idx(_cxi * _colspan))
                                )
                                if isinstance(_cbase, int):
                                    _cbase = idx(_cbase)
                                _feed_wcol(_cxi, _cbase, nsteps, Wb)
                            return
                        _w0 = idx(_wcol0) if isinstance(_wcol0, int) else _wcol0
                        for _cx in parallel_(NCX):
                            _feed_wcol(
                                _cx,
                                arith.addi(_w0, arith.muli(_cx, idx(_colspan))),
                                nsteps,
                                Wb,
                            )

                    if _wsel[0] == "extra":
                        # Extra waves: their buffer is a compile-time choice, like
                        # the lm-head's -- there is exactly one and the arm has
                        # already selected the slab offset.
                        _fan(_WEXTRA)
                    elif not W_SPLIT:
                        _fan(W)
                    elif _wsel[0] is None:
                        # vocab waves: the lm-head buffer is a compile-time choice.
                        _fan(_WBUFS[N_WGRP])
                    else:
                        # Decode waves: one arm per weight group, differing ONLY in
                        # which arg the puts name (_wcol0 is already group-relative).
                        # The wave loop is unrolled before BD assignment, so the
                        # selector is constant by then and the switch folds to its
                        # single taken arm; the arms exist purely so each wave's BDs
                        # can name a different arg.
                        def _arm(g):
                            _fan(_WBUFS[g])
                            yield_([])

                        index_switch(
                            [],
                            _wsel[0],
                            list(range(N_WGRP - 1)),
                            case_body_builder=lambda op, i, cv: _arm(cv),
                            default_body_builder=lambda op: _arm(N_WGRP - 1),
                        )

                for _layer in range(NLAYERS if a_iv is None else 1):
                    _wbase = _lb(W_LAYER)  # weights slab for this layer
                    _wgi = None  # which weight buffer this decode wave reads
                    if W_SPLIT and a_iv is not None:
                        # Group index and group base, as nested selects over the group
                        # boundaries. iv/G and iv%G would be the obvious spelling, but
                        # airrt-to-npu's post-unroll fold set (AIRRtToNpuPass.cpp,
                        # "Fold constant-condition launch-scope scf.index_switch")
                        # carries cmpi/select/addi/subi/muli and NOT divui/remui -- so
                        # those would leave the switch unfolded, and an index_switch
                        # cannot parent aiex.dma_configure_task_for.
                        _wgi = idx(N_WGRP - 1)
                        _goff = idx((N_WGRP - 1) * W_GROUP)
                        for _g in range(N_WGRP - 2, -1, -1):
                            _gc = arith.cmpi(
                                arith.CmpIPredicate.slt, a_iv, idx((_g + 1) * W_GROUP)
                            )
                            _wgi = arith.select(_gc, idx(_g), _wgi)
                            _goff = arith.select(_gc, idx(_g * W_GROUP), _goff)
                        # Offsets are relative to THIS group's buffer, so the span per
                        # buffer is G*W_LAYER instead of UNI_DEC*W_LAYER.
                        _wbase = arith.muli(arith.subi(a_iv, _goff), idx(W_LAYER))
                    _rbase = _lb(RMS_LAYER)  # rms weights slab for this layer
                    _kbase = _lb(KV_LAYER)  # KV cache slab for this layer
                    _ybase = _lb(Y_LAYER)  # Y (host-drain) region for this layer
                    if UNIFIED:
                        _u1 = arith.ConstantOp(IntegerAttr.get(i32, 1), None).result
                        _u0 = arith.ConstantOp(IntegerAttr.get(i32, 0), None).result
                        if a_iv is None:
                            _uarm = _u1
                        elif not N_EXTRA:
                            _ucmp = arith.cmpi(
                                arith.CmpIPredicate.slt, a_iv, idx(UNI_DEC)
                            )
                            _uarm = arith.select(_ucmp, _u1, _u0)
                        else:
                            # THE ARM CARRIES THE WAVE INDEX, not just the mode:
                            # 1 = decode, 0 = LM head, 2+k = extra wave k.
                            #
                            # It has to. Every consumer of the arm may select a
                            # SCALAR and never a program (see _rms_batched: AIR
                            # counts channel ops naming a buffer across scf.if
                            # arms, so a second arm doubles the lock credit and
                            # the design hangs in wave 0). The extra waves have
                            # different I2/J2/dest from each other, so a plain
                            # "is this an extra wave" bit would leave the proj
                            # core unable to tell WHICH -- and the only way to
                            # tell it without a second RTP is to widen this one.
                            # The arm is already a herd operand on every tile,
                            # so this is free.
                            _uarm = arith.select(
                                arith.cmpi(arith.CmpIPredicate.slt, a_iv, idx(UNI_DEC)),
                                _u1,
                                arith.select(
                                    arith.cmpi(
                                        arith.CmpIPredicate.slt,
                                        a_iv,
                                        idx(UNI_DEC + UNI_LM),
                                    ),
                                    _u0,
                                    arith.addi(
                                        arith.index_cast(
                                            i32,
                                            arith.subi(a_iv, idx(UNI_DEC + UNI_LM)),
                                        ),
                                        arith.ConstantOp(
                                            IntegerAttr.get(i32, 2), None
                                        ).result,
                                    ),
                                ),
                            )
                        _uarm_i = arith.index_cast(idx_t, _uarm)

                        def _rms_x_put_band(base, b, src=None):
                            """Put STG_W-wide band `b` (0-indexed) of the
                            K-wide row at DRAM offset `base` on rmsX -- the
                            launch-side half of the compute tile's
                            _rms_band_get. 3-D / _RMS_DMA_CHUNK-sub-diced
                            for the same reason variant D is (a compute
                            tile's BD wrap field is 8 bits); X's own K
                            row-stride is unchanged, only the SLICE has
                            shrunk.
                            """
                            assert STG_W % _RMS_DMA_CHUNK == 0
                            ChannelPut(
                                "rmsX",
                                X if src is None else src,
                                # On the UNIT-STRIDE dimension: the linear
                                # offset is sum(offsets[i]*strides[i]), so on
                                # dim 0 this would come out multiplied by
                                # _RMS_DMA_CHUNK -- see _rms_w_on_x_put.
                                offsets=[0, base + b * STG_W],
                                sizes=[BATCH, STG_W],
                                strides=[K, 1],
                            )

                        def _rms_x_put_allbands(base, src=None):
                            """Put all _nband STG_W-wide bands of the
                            K-wide row at DRAM offset `base` on rmsX in ONE
                            textual op, band folded into an outer
                            descriptor dimension. _rms_x_put_band (below)
                            emits _nband SEPARATE ops for this same data;
                            this exists because AIRToAIEPass's compute-tile
                            ring-builder cannot fold rmsX and rmsW onto one
                            shared S2MM port when their per-visit launch-
                            side op counts differ this much -- "operand #0
                            does not dominate this use", confirmed to
                            reproduce with vocab removed entirely (so it is
                            rmsX-vs-rmsW parity, not a decode/vocab
                            mismatch) -- see docs/DFlashFeasibility.md.
                            rmsW itself is still ONE op; folding rmsX's
                            _nband down to ONE op per visit is the smallest
                            move toward matching it, tried before reaching
                            for the bigger (and still unproven) fix of
                            splitting rmsW to match rmsX instead.
                            """
                            # FOUR DIMENSIONS IS THE CEILING for a shim put, and
                            # this spends all four: a shim BD carries three
                            # dimensions plus a repeat count, so the band
                            # dimension here becomes the repeat and the rest is
                            # the BD. A fifth dimension is dropped in
                            # AIRRtToNpuPass with no diagnostic -- see
                            # _rms_x_feed_ph, which is what that cost.
                            _nband = K // STG_W
                            ChannelPut(
                                "rmsX",
                                X if src is None else src,
                                # base on the UNIT-STRIDE dimension: the linear
                                # offset is sum(offsets[i]*strides[i]), so on
                                # dim 0 it would come out base*STG_W. Correct
                                # either way while X_SLOTS == 1 makes base 0 --
                                # see _rms_w_on_x_put, where a real base made
                                # this visible.
                                offsets=[0, 0, base],
                                sizes=[_nband, BATCH, STG_W],
                                strides=[STG_W, K, 1],
                            )

                        def _rms_w_on_x_put(base):
                            """One norm weight, band-shaped, on @rmsX.

                            RMS_W_ON_X's launch half. The descriptor MIRRORS
                            _rms_band_get exactly, with the weight's own
                            row stride (STG_W) where a band of X uses K: the
                            core's get lands stream position (c, t, i) at
                            xb[t*STG_W + c*64 + i], so sending
                            RMS[base + t*STG_W + c*64 + i] in that order puts
                            weight element j at xb[j] for every j < K. Anything
                            past K lands in the tail of `xb` and is never read
                            -- band_to_weight_aie copies K.

                            It reads BATCH*STG_W elements from `base`, i.e.
                            RMS_TAIL_SLACK past the end of a K-wide weight; the
                            rms BO carries that slack for the last one (see
                            RMS_TAIL_SLACK / decode_geometry.py).
                            """
                            assert STG_W % _RMS_DMA_CHUNK == 0
                            assert BATCH * STG_W >= K, (
                                f"RMS_W_ON_X needs a band fetch ({BATCH}*{STG_W}) "
                                f"wide enough to carry the whole {K}-wide norm "
                                "weight in one transfer"
                            )
                            # `base` GOES ON THE UNIT-STRIDE DIMENSION. An
                            # air.channel offset list is the usual strided-view
                            # convention -- the linear offset is
                            # sum(offsets[i] * strides[i]), not offsets[0] -- so
                            # putting a whole-buffer element offset on dimension
                            # 0 multiplies it by that dimension's stride. It is
                            # silent: `_rms_w_on_x_put(_final_norm_off)` with
                            # base on dim 0 lowered to `offset = 18874368`,
                            # which is _final_norm_off * _RMS_DMA_CHUNK, read
                            # far outside the weight and hung the dispatch.
                            # The other level-3 helpers are written the same way
                            # and are only correct because their base is always
                            # 0 (X_SLOTS == 1); they carry the same note.
                            if RMS_BAND_STREAM < 3:
                                # Bisection shape -- matches _w_row_get above.
                                ChannelPut(
                                    "rmsX",
                                    RMS,
                                    offsets=[0, base],
                                    sizes=[K // _RMS_DMA_CHUNK, _RMS_DMA_CHUNK],
                                    strides=[_RMS_DMA_CHUNK, 1],
                                )
                                return
                            ChannelPut(
                                "rmsX",
                                RMS,
                                offsets=[0, base],
                                sizes=[BATCH, STG_W],
                                strides=[STG_W, 1],
                            )

                        def _rms_x_feed_ph(base, nrefeed, src=None):
                            """ph0/ph2's launch-side feed at
                            RMS_BAND_STREAM>=3: the prepass sweep plus one
                            sweep per regen visit, each an all-bands put.

                            ONE PUT PER SWEEP, NOT ONE PER PHASE. This used to
                            fold the whole phase into a single op via a stride-0
                            outer dimension of extent (1+nrefeed) -- every visit
                            re-reads the identical `base` row, so a repeat with
                            no address advance is exactly what a regen visit is,
                            and as AIR that is a correct description. It does not
                            SURVIVE lowering. A shim BD holds three dimensions
                            plus a repeat count, `_rms_x_put_allbands` already
                            spends all four (band becomes the repeat), and the
                            fifth is dropped in AIRRtToNpuPass WITHOUT a
                            diagnostic:

                                air   sizes [13, 5, 8, 8, 64] strides [0, 512, 64, 2560, 1]
                                npu   sizes     [5, 8, 8, 64] strides    [512, 64, 2560, 1]
                                      len 4096  repeat_count 4

                            -- 5 bands where the core waits for 65, and the
                            dispatch hangs in wave 0 with nothing written. The
                            band dimension is the one that has to be folded (it
                            is what makes a sweep one op); the refeed dimension
                            has to be unrolled.

                            The fold's original reason is also gone. It was
                            introduced because 1+nrefeed separate puts tripped
                            "operand #0 does not dominate this use" in the
                            compute-tile ring builder, which multiplexed rmsX
                            against rmsW on one S2MM port. Under RMS_W_ON_X
                            there is no rmsW flow and rmsX is alone on its port,
                            so there is no ring to confuse.

                            `nrefeed` is a plain Python int, so the sweep count
                            is static rather than read from a register.
                            """
                            if _os.environ.get("_RMS_NO_FOLD"):
                                # Bisection escape: one put per band per visit,
                                # the shape before the band dimension was folded.
                                _nband = K // STG_W
                                for _ in range(1 + nrefeed):
                                    for b in range(_nband):
                                        _rms_x_put_band(base, b)
                                return
                            # ONE op for the whole phase again -- refeed,
                            # band, token, element -- now that token-major has
                            # freed a dimension. This is the ONLY placement that
                            # works: a shim task is awaited before the host can
                            # issue anything else (air.preserve_shim_dma_order is
                            # global), and a banded sweep completes only once the
                            # core has consumed it, which needs @xnorm to drain
                            # through the projections. One task per sweep leaves
                            # the host blocked mid-phase with the rest of the
                            # layer's feeds unissued.
                            #
                            # The earlier attempt at this fold was 5-D
                            # ([1+nrefeed, nband, STG_W/64, BATCH, 64]) because
                            # the band was chunk-major, and AIRRtToNpuPass
                            # silently dropped the outermost dimension:
                            #     air  [13, 5, 8, 8, 64] / [0, 512, 64, 2560, 1]
                            #     npu      [5, 8, 8, 64] / [512, 64, 2560, 1]
                            # -- 5 bands where the core waits for 65.
                            _nband = K // STG_W
                            if _os.environ.get("_RMS_W3_IN_NORM"):
                                _rms_w_on_x_put(0)
                                return
                            if _os.environ.get("_RMS_FETCH_OFF"):
                                return
                            if _os.environ.get("_RMS_NO_REGEN_FETCH"):
                                nrefeed = 0
                            if _os.environ.get("_RMS_PREPASS_BANDS"):
                                for b in range(int(_os.environ["_RMS_PREPASS_BANDS"])):
                                    if _os.environ.get("_RMS_PREPASS_AS_W"):
                                        # Feed the pre-pass band with the EXACT
                                        # descriptor the norm weights use --
                                        # same buffer, same contiguous shape.
                                        # Data is wrong; if this completes, the
                                        # fault is in the band descriptor, and
                                        # if it hangs the fault is simply "a
                                        # third rmsX transfer".
                                        _rms_w_on_x_put(0)
                                    else:
                                        _rms_x_put_band(base, b)
                                return
                            if _os.environ.get("_RMS_SRC_RMS"):
                                # Bisection: same shape, same count, same
                                # channel -- but sourced from the RMS buffer
                                # instead of X. The DATA is wrong; the question
                                # is whether the transfers complete. The weight
                                # bands (which read RMS) do; every band that
                                # reads X does not.
                                ChannelPut(
                                    "rmsX",
                                    RMS,
                                    offsets=[0, 0, 0, 0],
                                    sizes=[1 + nrefeed, _nband, BATCH, STG_W],
                                    strides=[0, STG_W, STG_W, 1],
                                )
                                return
                            ChannelPut(
                                "rmsX",
                                X if src is None else src,
                                offsets=[0, 0, 0, base],
                                sizes=[1 + nrefeed, _nband, BATCH, STG_W],
                                strides=[0, STG_W, K, 1],
                            )

                        def _rms_layerout_drain_band(dst, base, b):
                            """Get STG_W-wide band `b` of layerOut into DRAM
                            buffer `dst` at offset `base` -- the launch-side
                            half of the compute tile's _rms_band_put_back.
                            """
                            assert STG_W % _RMS_DMA_CHUNK == 0
                            ChannelGet(
                                "layerOut",
                                dst,
                                indices=[idx(0)],
                                # unit-stride dimension, as above.
                                offsets=[0, base + b * STG_W],
                                sizes=[BATCH, STG_W],
                                strides=[K, 1],
                            )

                        def _rms_layerout_drain_allbands(dst, base):
                            """All _nband bands of layerOut into `dst` at
                            `base`, in ONE textual op -- band as an outer
                            descriptor dimension, exactly mirroring
                            _rms_x_put_allbands on the feed side.

                            Needed for the same reason that one is: a CORE
                            tile's DMA holds at most 16 BD blocks
                            (`'aie.mem' op has more than 16 blocks`), and the
                            rms core at level 3 carries rmsX, rmsW, rmsW2,
                            outY, xnorm AND layerOut. Five separate drains per
                            residual phase, twice a layer, does not fit; one
                            repeated BD per phase does.

                            The compute side is unchanged and still alternates
                            get-then-put per round inside its own scf.for --
                            which AIR folds to a repeat-count BD -- so the
                            per-iteration lock handshake is what keeps the two
                            sides in step. That is the same mechanism a
                            repeated transfer into a single-band buffer
                            already relies on; it is NOT an assumption that
                            all five bands are in flight at once (the core has
                            one band buffer, `rstg_l1`, and could not hold
                            them).
                            """
                            _nband = K // STG_W
                            assert STG_W % _RMS_DMA_CHUNK == 0
                            ChannelGet(
                                "layerOut",
                                dst,
                                indices=[idx(0)],
                                # base on the unit-stride dimension -- same
                                # strided-view rule as _rms_x_put_allbands.
                                offsets=[0, 0, base],
                                sizes=[_nband, BATCH, STG_W],
                                strides=[STG_W, K, 1],
                            )

                        def _rms_residual_roundtrip(
                            base, nband_rounds, dst=None, src=None
                        ):
                            """residual1/residual2's launch-side half, at
                            RMS_BAND_STREAM>=3: for each of the nband_rounds
                            (== OPROJ_RNDS or DOWN_RNDS -- asserted equal to
                            K/STG_W at STG_W's definition) rounds, feed the
                            round's CURRENT band (rmsX, unchanged since ph0/
                            ph2 last read it) THEN drain the band the core
                            just updated (layerOut) -- in THAT order,
                            matching the compute side's own get-then-put
                            per round. AIR's dependency pass (which tracks
                            this on the shared memref X, the same mechanism
                            the existing cross-layer chaining already
                            relies on) then sees each round's read
                            happens-before its own write, and -- because
                            every band's feed/drain for THIS phase is
                            emitted before the NEXT phase's feed -- sees
                            the whole round-trip happens-before ph2 (for
                            residual1) or the next layer's ph0 (for
                            residual2, whose drain is exactly the existing
                            cross-layer write, just done band by band
                            instead of in one shot).

                            Called AFTER every other launch-side feed for
                            this layer (weights, KV, attention) has already
                            been emitted: nothing else touches X mid-layer,
                            so placement relative to X-touching ops is all
                            that matters, and placing it early would (with
                            air.preserve_shim_dma_order being GLOBAL, not
                            per-channel) force every later, otherwise-
                            independent feed to queue behind a get that
                            cannot complete until the rms core reaches
                            residual1 -- exactly the ordering trap
                            documented in the PROBE_5_LEN drain comment
                            below. Slower than an interleaved placement
                            would be (nothing here starts until the whole
                            weight feed is queued), not incorrect --
                            recovering that is Stage 7, not this.

                            The rmsX side is folded into ONE op
                            (_rms_x_put_allbands) rather than
                            nband_rounds separate puts: same "operand #0
                            does not dominate this use" reason ph0/ph2
                            were folded for. layerOut's gets stay
                            nband_rounds separate ops -- draining is the
                            core's per-band output, one band at a time,
                            same as before -- so this only reduces how
                            many times rmsX (not layerOut) is put,
                            leaving the read-before-write memref-level
                            ordering the docstring above relies on intact
                            (the whole read still precedes every write in
                            program order).
                            """
                            # `dst`/`dbase`: where the UPDATED bands land. X at
                            # `base`, in place, everywhere but the
                            # _RMS_RES2_TO_Y probe -- see there.
                            _dst, _dbase = (X, base) if dst is None else dst
                            if _os.environ.get("_RMS_NO_FOLD") or not _RMS_RT_FOLD:
                                for b in range(nband_rounds):
                                    _rms_x_put_band(base, b, src)
                                    _rms_layerout_drain_band(_dst, _dbase, b)
                                return
                            _rms_x_put_allbands(base, src)
                            _rms_layerout_drain_allbands(_dst, _dbase)

                        def _rms_interleave_after_phase(p):
                            """Arm the level-3 banded feeds INSIDE the weight
                            feed loop, one phase ahead of the core needing them.

                            A WEIGHT PHASE'S FEED IS AWAITED BEFORE THE NEXT ONE
                            IS ISSUED. `_feed_wcols` emits a coalesced put per
                            column and then an await barrier across all of them,
                            and that await only returns once the proj cores have
                            CONSUMED the phase -- 1.9 M bf16 for gate-up, far
                            more than the memtile holds, so consumption means
                            the phase actually ran. Placing the banded feeds
                            after the whole loop therefore deadlocks:

                              host: await(gate-up weights)   needs ph2 to run
                                    ph2                      needs residual1
                                    residual1                needs @rmsX band
                                    @rmsX band               not armed yet
                              core: residual1's _rms_band_get, forever

                            -- which is exactly what the late placement did
                            (measured: KV cache written, X never written, and
                            the hang is at residual1). Levels 0-2 are immune
                            because their residual never leaves L1: nothing the
                            core needs mid-layer comes from the host at all.

                            So each feed goes one phase EARLIER than the phase
                            whose await depends on it:

                              after p=0 (QKV)     residual1's round-trip, and
                                                  ph2's feed
                              after p=2 (gate-up) residual2's round-trip

                            p=0 rather than p=1 for residual1: the o-proj await
                            (p=1) needs the o-proj to drain 5 rounds of @outY
                            into the rms core, and the rms core takes round r+1
                            only after round r's band round-trip -- so the
                            round-trip has to be armed before the o-proj feed,
                            not after it. Same argument one phase along for
                            residual2 and the down projection.

                            SHIM SLOT PRESSURE IS WHY THIS FITS. @rmsX gets two
                            task slots. At the p=0 boundary they hold the two
                            norm weights (retired) and ph0's sweep, and ph0's
                            sweep is complete by then -- the QKV await that just
                            returned is the proof, since every one of its
                            1+XN_REFEED sweeps feeds that projection. So both
                            slots are free for residual1's put and ph2's feed.
                            At the p=2 boundary residual1's put has retired
                            (residual1 ran during p=1), freeing one for
                            residual2's. @layerOut needs one slot at a time.
                            """
                            _nb = K // STG_W
                            # WHERE `h` LIVES. residual1's output is a whole
                            # intermediate hidden state that ph2 re-reads 39
                            # times and residual2 reads once more, and at level 3
                            # it cannot stay on the tile -- that is the whole
                            # point of the level. Writing it back over X (the
                            # obvious in-place choice, and what levels 0-2 do
                            # with their single whole-row write) puts every
                            # level-3 feed and drain on one memref, and AIR
                            # chains same-memref channel ops in program order.
                            # residual2's drain then lands after its own put,
                            # which the core cannot satisfy: the put does not
                            # retire until the core has taken all five bands,
                            # and the core will not take band b+1 until band b's
                            # drain is armed. (Measured: X carries residual1's
                            # output and the dispatch times out with residual2's
                            # bands undrained.)
                            #
                            # So `h` gets its own region and X is touched twice
                            # per layer, at the two ends, exactly as before:
                            #
                            #   read X   ph0's sweep, residual1's put
                            #   write H  residual1's drain
                            #   read H   ph2's sweep, residual2's put
                            #   write X  residual2's drain   <- the layer output
                            #
                            # Nothing reads X between residual1's put and
                            # residual2's drain, so that drain is ordered only
                            # behind reads that retired phases ago.
                            # AND RESIDUAL2'S TWO HALVES SPLIT ACROSS THE
                            # BOUNDARY. Its drain goes one phase early, with
                            # ph2's feed; only its put waits for the gate-up
                            # boundary. Emitted together at p=2 the lowering
                            # puts `await(put)` between them --
                            #
                            #   start rmsX(res2 put) ; await rmsX ; start layerOut
                            #
                            # -- and that await cannot retire: the core takes
                            # residual2's band 0 only after @outY round 0, which
                            # is the DOWN projection's output, whose weights are
                            # fed at p=3, after the drain. Splitting them puts
                            # the drain ahead of the await, and the put keeps its
                            # own ordering behind ph2's sweep (same channel, same
                            # buffer) which is what it actually depends on.
                            #
                            # Both channels still see their bands in the core's
                            # order -- @rmsX w, w2, ph0, res1, ph2, res2 and
                            # @layerOut res1, res2 -- because the split only
                            # moves the drain earlier past ops on the OTHER
                            # channel.
                            _nb = K // STG_W
                            # WHERE `h` LIVES, and where the layer output goes.
                            # _RMS_H_SWAP inverts them: `h` back in X in place
                            # and the output in the scratch. Not a shipping path
                            # (it breaks chaining), it is the buffer half of the
                            # residual2 bisection -- with it, the round-trip that
                            # hangs is still the LATE one, so the buffer is not
                            # the variable.
                            if _os.environ.get("_RMS_H_SWAP"):
                                _H, _OUT = (X, _x_in()), (Y, _y_scratch_off)
                            else:
                                _H, _OUT = (Y, _y_scratch_off), (X, _x_out())
                            # _RMS_PH2_PHASE / _RMS_RES2_PUT_PHASE /
                            # _RMS_RES2_DRAIN_PHASE: which weight-phase boundary
                            # each of the three late feeds is armed at. The
                            # defaults are the derivation above. @rmsX's ring
                            # order still has to hold, so ph2's feed may not land
                            # later than residual2's put.
                            _p_ph2 = int(
                                _os.environ.get("_RMS_PH2_PHASE", GATEUP_PHASE - 1)
                            )
                            _p_put = int(
                                _os.environ.get("_RMS_RES2_PUT_PHASE", GATEUP_PHASE)
                            )
                            _p_drn = int(
                                _os.environ.get(
                                    "_RMS_RES2_DRAIN_PHASE", GATEUP_PHASE - 1
                                )
                            )
                            assert _p_ph2 <= _p_put, "@rmsX ring order"
                            # Emitted in the order the two rings expect: @rmsX
                            # res1-put, ph2, res2-put and @layerOut res1-drain,
                            # res2-drain. Only entries whose phase matches fire,
                            # so moving one with the knobs above cannot reorder
                            # the others.
                            # _RMS_RT_COMBINED: feed BOTH residuals' bands as
                            # ONE @rmsX task -- ten bands, the five of `h`
                            # repeated via a stride-0 outer dimension, exactly
                            # the shape _rms_x_feed_ph uses for a phase. Both
                            # residuals then read the scratch (residual1's data
                            # is garbage; only completion is being measured).
                            #
                            # The point is the TASK COUNT. residual1's band feed
                            # lands and residual2's does not, on a dedicated
                            # packet flow (shim 2,0 MM2S0 -> tile 2,2 DMA0, one
                            # source one dest) with the core holding lock credit
                            # and the task started. If one task carrying both
                            # works where two do not, the fault is the second
                            # shim task on this channel, not anything about
                            # residual2.
                            if _os.environ.get("_RMS_RT_COMBINED"):
                                if p == 0:
                                    ChannelPut(
                                        "rmsX",
                                        _H[0],
                                        offsets=[0, 0, 0, _H[1]],
                                        sizes=[2, _nb, BATCH, STG_W],
                                        strides=[0, STG_W, K, 1],
                                    )
                                    _rms_layerout_drain_allbands(*_H)
                                elif p == _p_drn:
                                    _rms_layerout_drain_allbands(*_OUT)
                                return
                            if p == 0 and "1" in _RMS_BANDS_GET:
                                _rms_x_put_allbands(_x_in())
                            if p == 0 and "1" in _RMS_BANDS_PUT:
                                _rms_layerout_drain_allbands(*_H)
                            if p == _p_ph2:
                                _rms_x_feed_ph(_H[1], REFEED[GATEUP_PHASE], src=_H[0])
                            if p == _p_put and "2" in _RMS_BANDS_GET:
                                if _os.environ.get("_RMS_RES2_AS_W"):
                                    # Bisection: feed residual2's five bands with
                                    # the NORM WEIGHT's descriptor -- same
                                    # channel, same count, same position, but a
                                    # 2-D contiguous read of the rms BO that is
                                    # known to work (it is how both norm weights
                                    # arrive). Data is wrong by construction. If
                                    # this completes, residual2's descriptor or
                                    # source is the variable; if it hangs, the
                                    # position in the @rmsX ring is.
                                    for _ in range(_nb):
                                        _rms_w_on_x_put(0)
                                else:
                                    _rms_x_put_allbands(_H[1], src=_H[0])
                            if p == _p_drn and "2" in _RMS_BANDS_PUT:
                                _rms_layerout_drain_allbands(*_OUT)

                        # real-lm-head final norm (model.norm.weight): a DEDICATED
                        # slot after the [in|post]*UNI_DEC rms slabs + 64-wide rope
                        # LUT, so the vocab rmsnorm uses the true final norm -- NOT
                        # layer-0's in_LN (mirrors decoding_layer's separate
                        # final_rms_weight). The final norm sits AFTER the rope
                        # region: llama has ONE shared rope LUT (ROPE_W_LEN),
                        # per-layer models have UNI_DEC slabs.
                        #
                        # Hoisted out of _uni_voc (where it used to be computed just
                        # before the rmsW put) because RMS_W_ON_X needs it EARLIER in
                        # that function -- the weights now go on @rmsX ahead of the X
                        # feed. Pure Python arithmetic over module constants, so
                        # hoisting emits no IR and moves no constant.
                        _final_norm_off = (
                            UNI_DEC * RMS_LAYER
                            + (UNI_DEC if ROPE_W_PER_LAYER else 1) * ROPE_W_LEN * BATCH
                        )
                        # The ones run appended to the rms BO (see rms_l3),
                        # from the module constant the host reads.
                        _rms_ones_off = RMS_ONES_OFF
                        assert _rms_ones_off == _final_norm_off + K + RMS_TAIL_SLACK

                        def _uni_voc():
                            _wsel[0] = None  # lm-head buffer, statically known
                            # W_SPLIT gives the lm-head its own buffer, so its slabs
                            # start at 0 instead of after all UNI_DEC layer slabs.
                            # The non-split expression is kept verbatim (constants
                            # materialize in the same order) so the IR stays
                            # byte-identical when the knob is off.
                            if W_SPLIT:
                                _vwb = arith.muli(
                                    arith.subi(a_iv, idx(UNI_DEC)),
                                    idx(VOCAB_W_BLOCKS * BLOCK_BF16),
                                )
                            else:
                                _vwb = arith.addi(
                                    idx(UNI_DEC * W_LAYER),
                                    arith.muli(
                                        arith.subi(a_iv, idx(UNI_DEC)),
                                        idx(VOCAB_W_BLOCKS * BLOCK_BF16),
                                    ),
                                )
                            # Every drained row becomes B rows, here as everywhere
                            # else: a wave's vocab chunk is BATCH*VOCAB_SIZE_PADDED
                            # and lays out token-major inside it (see the drain).
                            _vyb = arith.addi(
                                idx((HOST_ROUNDS + LAYER_RNDS) * PAYLOAD * BATCH),
                                arith.muli(
                                    arith.subi(a_iv, idx(UNI_DEC)),
                                    idx(VOCAB_SIZE_PADDED * BATCH),
                                ),
                            )
                            # ===== LM head (IS_ATTN=0), the reference gen_lm_head_seq analog =====
                            # Same device, RTP arm=0: the proj cores run ONE vocab phase
                            # (VOCAB_I2 row-pairs x NBJ col-blocks, id -> VOCAB_DEST) and
                            # the rms core does the final rmsnorm(x). Feed: x + final rms
                            # weight + vocab weights; drain VOCAB_SIZE_PADDED logits per
                            # token into Y. No attn/rope/glu/KV feeds (those herds are
                            # parked -- RTP-unarmed -- so they need no input; feeding them
                            # would only back-pressure).
                            #
                            # WHO FORWARDS THE LOGITS depends on the batch, and it is the
                            # only thing that does: at batch 1 the rms core relays them on
                            # @layerOut (VOCAB_DEST is the o-proj/down id, which that core
                            # already consumes); batched, they go to dest 0 and its memtile
                            # relays them on @toShim. See VOCAB_DEST.
                            if RMS_BAND_STREAM >= 3 and BATCH > 1:
                                # _rms_batched's compute-tile body is ONE
                                # shared program for both arms: the vocab
                                # arm runs the SAME ph0 scale pre-pass +
                                # regen as decode, just with VOCAB_RNDS
                                # re-feeds instead of XN_REFEED (_cnt()'s
                                # dynamic index_switch picks the count at
                                # runtime) and no residual/ph2 at all
                                # (arm=0 skips every _if_decode body, and
                                # ph2's regen count is 0 on this arm). So
                                # this arm needs ph0's feed and nothing
                                # else from _rms_residual_roundtrip's world.
                                #
                                # The two weights go FIRST and on @rmsX, in the
                                # core's own order (RMS_W_ON_X). On this arm the
                                # first is the real final norm and the second is
                                # the dummy the shared body still consumes.
                                if RMS_W_ON_X:
                                    _rms_w_on_x_put(_final_norm_off)
                                    if POST_RMS:
                                        _rms_w_on_x_put(0)
                                _rms_x_feed_ph(_x_final, VOCAB_RNDS)
                            elif RMS_BAND_STREAM >= 2 and BATCH > 1:
                                # _rms_batched's compute-tile body is ONE
                                # shared program for both arms (see its
                                # docstring): its banded rmsX get runs
                                # identically whichever arm is active, so this
                                # arm's feed has to match it. One static 3-D
                                # put, matching the compute tile's one 3-D get
                                # op-for-op -- see the variant-C/D comment there.
                                # _RMS_DMA_CHUNK, not STG_W: a compute tile's
                                # 8-bit-per-dimension BD wrap field can't hold
                                # STG_W (512) as an extent (variant D).
                                assert K % _RMS_DMA_CHUNK == 0
                                # AFTER the row, not before. The rms core's
                                # rmsX BD ring is emitted in the core's PROGRAM
                                # order, and at level <3 the row get sits at the
                                # top of _rms_batched (with the allocs) while the
                                # weight gets come later -- so the ring is
                                # [row, w, w2] and the launch has to feed it in
                                # that order. Feeding the weights first sends
                                # them into the row's slot: no hang, and every
                                # weight and the row land in each other's
                                # buffers (measured: rms rel 1.56e+00).
                                ChannelPut(
                                    "rmsX",
                                    X,
                                    offsets=[_x_final, 0, 0],
                                    sizes=[K // _RMS_DMA_CHUNK, BATCH, _RMS_DMA_CHUNK],
                                    strides=[_RMS_DMA_CHUNK, K, 1],
                                )
                                if RMS_W_ON_X:
                                    _rms_w_on_x_put(_final_norm_off)
                                    if POST_RMS:
                                        _rms_w_on_x_put(0)
                            else:
                                ChannelPut(
                                    "rmsX",
                                    X,
                                    offsets=[_x_final],
                                    sizes=[BATCH * K],
                                    strides=[1],
                                )
                            # _final_norm_off is hoisted above _uni_voc now -- see
                            # there. RMS_W_ON_X has already fed both norms on
                            # @rmsX at the top of this function, so this whole
                            # weight-put block is skipped.
                            if RMS_W_ON_X:
                                pass
                            elif N_NORMS >= 4:
                                # Gemma: rmsW/rmsW2 are 2K (two norms packed). Put final_norm
                                # in rmsW's HI half (lm_head reads it via rms_norm_hi_aie); the
                                # LO half is the last rope-region K -- a harmless in-bounds
                                # dummy ([_final_norm_off-K .. +2K] is the BO's last 2K). rmsW2
                                # is a 2K dummy. Keeps the shared packet group hole-free.
                                ChannelPut(
                                    "rmsW",
                                    RMS,
                                    offsets=[_final_norm_off - K],
                                    sizes=[2 * K],
                                    strides=[1],
                                )
                                ChannelPut(
                                    "rmsW2",
                                    RMS,
                                    offsets=[0],
                                    sizes=[2 * K],
                                    strides=[1],
                                )
                            else:
                                ChannelPut(
                                    "rmsW",
                                    RMS,
                                    offsets=[_final_norm_off],
                                    sizes=[K],
                                    strides=[1],
                                )
                                if POST_RMS:
                                    # DUMMY post-LN weight: rmsW2 is decode-only but packet-
                                    # muxes onto the same shim MM2S as the vocab-active rmsX
                                    # (rms tile has only 2 S2MM). Feeding + consuming a dummy
                                    # in vocab keeps that packet group hole-free so the vocab
                                    # tail doesn't stall (consumed by _rms_lm_head dummy get).
                                    ChannelPut(
                                        "rmsW2",
                                        RMS,
                                        offsets=[0],
                                        sizes=[K],
                                        strides=[1],
                                    )
                            # vocab weight feed: round-major, NCY-fanned. Python-unrolled
                            # (NOT an AIR for_ -- a launch-scope for_ DEADLOCKS the shim
                            # sequence). inW puts are issue_token=false so the shim reuses BD
                            # IDs -> many puts fit (decode feeds ~464/col fine). With the 9->1
                            # collapse (UNI_LM=1, VOCAB_CHUNK_I2=126) this is 2016 puts/col;
                            # the shim BD-reuse absorbs them (the wave itself is now enabled by
                            # the value-1 xnorm re-broadcast in _rms_lm_head -- see there).
                            # Spatial fan over the NCX vocab-weight columns: bundle index
                            # @inW[cx] is an scf.parallel IV (canonical form). Each column
                            # is one contiguous DDR block fed as two halves so the shim-dma
                            # coalescer merges+tags them (see the decode feed).
                            assert VOCAB_PER_COL % NCY == 0
                            _colspan = VOCAB_PER_COL * blk
                            _feed_wcols(_vwb, _colspan, VOCAB_PER_COL // NCY)
                            # ATTENTION FULLY GATED OFF in vocab (gate-off 2026-07-15b):
                            # the 8 attn cores' bodies index_switch to an empty idle case
                            # in vocab, and every launch-scope attn channel (ropeLUT,
                            # appendK/appendV, inKV, toAttnQ/toK/toV, attnO->xnorm) is
                            # emitted ONLY in the decode branch. So NOTHING attn is fed or
                            # drained in vocab -- no dummy pairing needed, and the 4-slot
                            # count-free KV memtile ring is never touched (was the 3-vocab
                            # hang). _xc_voc already excludes OPROJ_REFEED, so the xnorm
                            # convergence stays balanced with omtb producing no o-proj here.
                            # drain logits (natural order). ONE strided get either
                            # way; only the channel and the number of dims differ.
                            if BATCH > 1:
                                # The GLU core relays 2*VOC_ROTATIONS slices of
                                # BATCH x GLU_SLICE, token-major within a slice.
                                # The get folds them into
                                # [BATCH][VOCAB_SIZE_PADDED], so a token's chunk
                                # of the vocab is contiguous and the host reads
                                # logits[t] as a plain slice.
                                ChannelGet(
                                    "vocOut",
                                    Y,
                                    indices=[idx(0)],
                                    offsets=[_vyb],
                                    sizes=[
                                        VOCAB_RNDS // VOC_SLICE_RNDS,
                                        BATCH,
                                        GLU_SLICE,
                                    ],
                                    strides=[GLU_SLICE, VOCAB_SIZE_PADDED, 1],
                                )
                            else:
                                # rms LM branch forwards VOCAB_RNDS x PAYLOAD via
                                # layerOut.
                                ChannelGet(
                                    "layerOut",
                                    Y,
                                    indices=[idx(0)],
                                    offsets=[_vyb],
                                    sizes=[VOCAB_RNDS, PAYLOAD],
                                    strides=[PAYLOAD, 1],
                                )
                            yield_([])

                        def _uni_extra(k):
                            """Extra wave k's host feed. ONE phase, like vocab.

                            Python-unrolled per wave rather than parameterized at
                            runtime, because `_feed_wcols` takes its step count as
                            a Python int -- it unrolls the puts, and the extra
                            waves do not share a step count (the DFlash pre-pass's
                            fc is 250 steps per column against the context K/V's
                            40). The arms cost instruction stream and not BD ids:
                            `inW` puts are issue_token=false so the shim reuses BD
                            IDs, which is the same property that lets a decode
                            wave issue ~464 puts per column.
                            """
                            _wsel[0] = "extra"
                            # X FIRST, WEIGHTS SECOND, and that order is
                            # load-bearing. air.preserve_shim_dma_order is
                            # global: a shim task is awaited before the host can
                            # issue anything else. Feeding the weights first
                            # blocks the shim on tasks the proj cores cannot
                            # drain, because they are waiting for an X that has
                            # not been issued yet -- measured as a dispatch
                            # timeout, and the vocab arm already feeds in this
                            # order for the same reason.
                            #
                            # X: the tap row on @rmsX, and ONES on @rmsW.
                            #
                            # It cannot come straight from DDR onto @xnorm, and
                            # it cannot have a shim channel of its own. Both were
                            # built and both are refused:
                            #
                            #   launch put on @xnorm
                            #     'air.channel.put' op failed to link to any shim
                            #     dma allocation -- @xnorm's producers are all
                            #     on-array, so it has no shim side to pair a
                            #     launch-scope put with. A channel is L3->L2 or
                            #     L2->L2, not both.
                            #   a dedicated @tapsX shim->X-memtile channel
                            #     'air.channel.put' op failed to get S2MM tile
                            #     for L3 allocation, the flow holding 5 producers
                            #     and ZERO receivers. The X memtile's arm-switched
                            #     feed is collapsed to ONE relay structure before
                            #     flows are built, so a channel named only in a
                            #     non-surviving arm is not in the device by then.
                            #
                            # So it arrives on @xnorm from a producer already
                            # there, and only the rms core has a route to DDR --
                            # the shim's @rmsX at its input, which is these three
                            # puts, in the core's own ring order [row, w, w2].
                            #
                            # And it needs NO KERNEL CHANGE, because rms_chunk is
                            # already the strided gather this wants:
                            #   y[t*n+i] = x[t*K + c*n + i] * w[c*n+i] * scales[t]
                            # Feed w == 1 and only the per-row scales[t] is left,
                            # which the host divides back out -- it wrote the tap,
                            # so it knows rms(tap). The alternative was a third
                            # arm inside rms_scale_row_aie, and a scalar the host
                            # already holds is cheaper than a branch on the tile
                            # whose budget this whole discipline protects.
                            ChannelPut(
                                "rmsX",
                                X,
                                offsets=[EXTRA_WAVES[k]["x_slot"] * BATCH * K],
                                sizes=[BATCH * K],
                                strides=[1],
                            )
                            if _tx is not None:
                                # The SAME rows again, to the relay memtile.
                                # The @rmsX put above stays: the rms core's body
                                # gets it unconditionally and would stall on a
                                # hole, exactly as rmsW2 is consumed and
                                # discarded on the vocab arm. 40 KB against the
                                # wave's own megabytes.
                                ChannelPut(
                                    "tapsX",
                                    X,
                                    offsets=[EXTRA_WAVES[k]["x_slot"] * BATCH * K],
                                    sizes=[BATCH * K],
                                    strides=[1],
                                )
                            ChannelPut(
                                "rmsW",
                                RMS,
                                offsets=[_rms_ones_off],
                                sizes=[K],
                                strides=[1],
                            )
                            if POST_RMS:
                                # Consumed and discarded by the shared body, as
                                # on the vocab arm: rmsW2 packet-muxes onto the
                                # same shim MM2S as rmsX, and a hole in that
                                # group stalls the tail.
                                ChannelPut(
                                    "rmsW2",
                                    RMS,
                                    offsets=[0],
                                    sizes=[K],
                                    strides=[1],
                                )
                            # The layer-output drain, which this wave needs for
                            # the same reason a decode wave does: the rms core's
                            # @layerOut put runs on every arm that is not the LM
                            # head, so arm 2+k puts a BATCH*K row that SOMEONE has
                            # to receive. Without this get the core blocks on it
                            # and the dispatch times out -- measured, and the last
                            # of four stalls between the first extra-wave build
                            # and the first one that runs.
                            #
                            # Same op, same extent, its OWN slot -- see
                            # EXTRA_OUT_SLOT for why sharing one is not enough.
                            #
                            # AND IT GOES LAST, after the weights, for the same
                            # reason the X goes first. It is a GET: under
                            # air.preserve_shim_dma_order the host awaits it
                            # before issuing anything queued behind it, so
                            # placing it above the weight feed makes the host
                            # wait for a result the proj cores cannot compute
                            # until they get the weights it is blocking. That
                            # deadlocks -- it is what the first extra-wave build
                            # that reached the device actually died of, and the
                            # decode arm has always drained layerOut at the very
                            # end of its own feed.
                            _colspan = EXTRA_PER_COL[k] * blk
                            _feed_wcols(
                                idx(EXTRA_WAVES[k]["w_off"]),
                                _colspan,
                                EXTRA_NSTEPS[k],
                            )
                            ChannelGet(
                                "layerOut",
                                X,
                                indices=[idx(0)],
                                offsets=[EXTRA_OUT_SLOT[k] * BATCH * K],
                                sizes=[BATCH * LAYER_RNDS * PAYLOAD],
                                strides=[1],
                            )
                            yield_([])

                        def _uni_dec():
                            _wsel[0] = _wgi  # runtime group index (None if unsplit)
                            # raw X (@xy) + rms weight (@rmsin) to the rms producer core; the
                            # on-chip rms normalizes + re-feeds X (see refeed()). X is
                            # in-place (offset 0 every layer -- the chained hidden state)
                            # unless HIDDEN_TAPS, which reads slot a_iv instead.
                            # BATCH*K: B token embeddings, token-major. The rms
                            # core takes the block in one get and keeps it as
                            # the residual stream for the whole layer.
                            if RMS_BAND_STREAM >= 3:
                                # ph0's feed only -- residual1/ph2/residual2
                                # go at the END of this function (see the
                                # _rms_residual_roundtrip / second
                                # _rms_x_feed_ph calls below the weight-feed
                                # loop), not here: this IS the first thing
                                # the layer needs, so it stays early, but
                                # the rest has real dependencies (o-proj's
                                # output, the down weight) that a
                                # global air.preserve_shim_dma_order would
                                # turn into a deadlock if placed too early
                                # -- see _rms_residual_roundtrip's comment.
                                #
                                # The two norm weights go FIRST, on @rmsX, in
                                # the order the core takes them (RMS_W_ON_X):
                                # one channel means launch order IS core order,
                                # where two channels made them independent
                                # FIFOs. They used to be put after the X feed,
                                # further down this function.
                                #
                                # These two stay EARLY and ph0's feed does not
                                # (see the block at the end of this function).
                                # A weight band is consumed by the core the
                                # moment it arrives and produces nothing
                                # downstream, so nothing queues behind it; a
                                # ph0 sweep produces @xnorm and cannot complete
                                # until the projection cores are fed.
                                if RMS_W_ON_X:
                                    _rms_w_on_x_put(_rbase)
                                    if _os.environ.get("_RMS_W3"):
                                        _rms_w_on_x_put(_rbase)
                                    if POST_RMS:
                                        _rms_w_on_x_put(_lo(_rbase, K))
                                # ph0's feed goes HERE: the top of _uni_dec,
                                # right after the norm weights and before the
                                # rope LUT and the projection weights. That is
                                # where levels 0 and 2 put their single
                                # whole-row rmsX read, and AFTER the weights so
                                # the launch order still matches the order the
                                # core takes them in.
                                #
                                # MEASURED, not chosen. Fed instead from the QKV
                                # weight boundary -- after _feed_wcols(p=0),
                                # just before the KV append, which is where a
                                # dependency argument puts it -- the dispatch
                                # hangs with nothing written at all. Fed from
                                # here it runs through ph0, the QKV projection
                                # and rope, and the KV cache comes back written.
                                # Same descriptor, same count, same core code;
                                # only the launch-side position differs.
                                _rms_x_feed_ph(_x_in(), XN_REFEED)
                                if _os.environ.get("_RMS_RES1_PUT_TOP"):
                                    # residual1's rmsX PUT can come up here too:
                                    # it reads the layer INPUT, which nothing has
                                    # written yet. Its @layerOut drain cannot --
                                    # that one only completes once the core has
                                    # reached residual1, so hoisting it stalls
                                    # every later feed behind it (measured: with
                                    # the whole round-trip at the top, not even
                                    # the KV cache is written).
                                    _rms_x_put_allbands(_x_in())
                            elif RMS_BAND_STREAM >= 2:
                                # Variant D: one static 3-D put/get pair on
                                # both ends (like variant C), but at
                                # _RMS_DMA_CHUNK granularity, not STG_W -- see
                                # the _RMS_DMA_CHUNK comment and the
                                # variant-D note in _rms_batched. Not a loop
                                # either way, so the launch-scope
                                # for_-deadlocks-the-shim rule (_feed_wcols)
                                # doesn't enter into it.
                                assert K % _RMS_DMA_CHUNK == 0
                                # AFTER the row, not before. The rms core's
                                # rmsX BD ring is emitted in the core's PROGRAM
                                # order, and at level <3 the row get sits at the
                                # top of _rms_batched (with the allocs) while the
                                # weight gets come later -- so the ring is
                                # [row, w, w2] and the launch has to feed it in
                                # that order. Feeding the weights first sends
                                # them into the row's slot: no hang, and every
                                # weight and the row land in each other's
                                # buffers (measured: rms rel 1.56e+00).
                                ChannelPut(
                                    "rmsX",
                                    X,
                                    offsets=[_x_in(), 0, 0],
                                    sizes=[K // _RMS_DMA_CHUNK, BATCH, _RMS_DMA_CHUNK],
                                    strides=[_RMS_DMA_CHUNK, K, 1],
                                )
                                if RMS_W_ON_X:
                                    _rms_w_on_x_put(_rbase)
                                    if POST_RMS:
                                        _rms_w_on_x_put(_lo(_rbase, K))
                            else:
                                ChannelPut(
                                    "rmsX",
                                    X,
                                    offsets=[_x_in()],
                                    sizes=[BATCH * K],
                                    strides=[1],
                                )
                            # RMS_W_ON_X feeds both norms on @rmsX at the top of
                            # this function instead -- see the level-3 branch above.
                            if RMS_W_ON_X:
                                pass
                            elif N_NORMS >= 4:
                                # Gemma: pack two norms per 2K channel -- rmsW =
                                # [input | post_attn] (slab 0..2K), rmsW2 = [pre_ffn |
                                # post_ffn] (slab 2K..4K). Keeps the rms tile at <=4 packet
                                # ids per S2MM port; the lo/hi kernels slice each half.
                                ChannelPut(
                                    "rmsW",
                                    RMS,
                                    offsets=[_rbase],
                                    sizes=[2 * K],
                                    strides=[1],
                                )
                                ChannelPut(
                                    "rmsW2",
                                    RMS,
                                    offsets=[_lo(_rbase, 2 * K)],
                                    sizes=[2 * K],
                                    strides=[1],
                                )
                            else:
                                ChannelPut(
                                    "rmsW",
                                    RMS,
                                    offsets=[_rbase],
                                    sizes=[K],
                                    strides=[1],
                                )
                                if POST_RMS:
                                    # post_attention_layernorm weight on its own channel.
                                    ChannelPut(
                                        "rmsW2",
                                        RMS,
                                        offsets=[_lo(_rbase, K)],
                                        sizes=[K],
                                        strides=[1],
                                    )
                            # rope LUT: sits after all UNI_DEC rms slabs in arg2. Llama:
                            # ONE per-position LUT SHARED across layers (single theta) at a
                            # layer-independent offset. ROPE_W_PER_LAYER (gemma/qwen3
                            # qk-norm, qwen2.5 q/k/v bias): rope_w DIFFERS PER LAYER, so
                            # index a per-wave slab (UNI_DEC contiguous rope_w slabs, offset
                            # _lut_off + a_iv*ROPE_W_LEN). UNIFIED sizes arg2 for UNI_DEC decode
                            # waves (module-gen forces NLAYERS=1, which would misplace it).
                            _lut_off = (UNI_DEC * RMS_LAYER) if MULTIBLK else 0
                            # DECODE_BATCH: the LUT is per POSITION and a block
                            # spans B of them, so the slab is B whole
                            # ROPE_W_LEN blocks and rope consumes them in order,
                            # one per token. Easy to miss: nothing in the shape
                            # says "per position", and feeding one LUT for B
                            # tokens gives every token position P's rotation --
                            # plausible output, wrong tokens.
                            _rope_off = (
                                _lo(_lb(BATCH * ROPE_W_LEN), _lut_off)
                                if (ROPE_W_PER_LAYER and MULTIBLK)
                                else _lut_off
                            )
                            # THE PER-TOKEN LOOP, AT EVERY LEVEL. There used to
                            # be a RMS_BAND_STREAM>=3 fold here that collapsed
                            # the BATCH puts below into one op with the
                            # per-token spacing as an outer dimension. It was
                            # added for the same "operand #0 does not dominate
                            # this use" reason rmsX's feeds were folded for --
                            # a compute-tile ring builder tripping over rmsX
                            # and rmsW multiplexed on one S2MM -- and that
                            # reason is gone: RMS_W_ON_X leaves rmsX alone on
                            # its port.
                            #
                            # It was also WRONG, in a way that cost most of a
                            # debugging session. `offsets=[_rope_off, 0]`
                            # against `strides=[ROPE_W_LEN, 1]` does not put
                            # the read at `_rope_off`: a channel offset list is
                            # the strided-view convention, so the linear offset
                            # is sum(offsets[i]*strides[i]) and this read
                            # `_rope_off * ROPE_W_LEN` -- hundreds of thousands
                            # of elements past the end of the rms buffer. The
                            # element COUNT stayed right (3072, so shim_volume.py
                            # and check_channel_balance.py both read balanced),
                            # rope got garbage, and the dispatch hung before the
                            # first KV append with nothing written at all.
                            # The comment below already said a single B-wide
                            # put against B gets is not something this channel
                            # promises. It was right.
                            _tl_range = range(BATCH)
                            # ONE PUT PER TOKEN, matching rope's one get per
                            # token. A single B-wide put paired against B gets
                            # is not something an AIR channel promises: @xnorm
                            # gets away with it because it is a packet channel
                            # whose dest BD reassembles, this one is not.
                            for _tl in _tl_range:
                                _lo_t = _tl * ROPE_W_LEN
                                # _rope_off is a raw int on models with one
                                # shared LUT and an SSA value on per-layer ones;
                                # _lo only bridges that when there is no layer
                                # loop, so pick here rather than there.
                                if not _lo_t:
                                    # NOT arith.addi(x, 0): an addi of zero is
                                    # still an op, and the batch-1 no-op diff
                                    # counts ops. _lo skips it for the same
                                    # reason.
                                    _off_t = _rope_off
                                elif isinstance(_rope_off, int):
                                    _off_t = _rope_off + _lo_t
                                else:
                                    _off_t = arith.addi(_rope_off, idx(_lo_t))
                                ChannelPut(
                                    "ropeLUT",
                                    RMS,
                                    offsets=[_off_t],
                                    sizes=[ROPE_W_LEN],
                                    strides=[1],
                                )

                            # weights: per col, streamed in NCY-block (10240) steps matched
                            # with the memtile weight-fan gets (AIR does not auto-split a big
                            # put into many gets -> size must match or the fan deadlocks).
                            # round-major (fill-step OUTER, column INNER): the cores consume
                            # fill-step i of ALL columns together (X-broadcast lockstep), so
                            # the runtime must issue all columns' fill-i before fill-(i+1).
                            # Phases are concatenated in the host W array; each phase's slab is
                            # fed in its own round-major sweep, so the per-col inW FIFO carries
                            # the cores' total consume order.
                            # the reference full-faithful KV cache (DDR): (1) APPEND this token's roped
                            # K/V into the cache at APPEND_OFF (device S2MM via appendK/appendV);
                            # (2) READ BACK the whole cache per CU (inKV, strided) for the flash
                            # block loop. = the reference _receive + _move.
                            def _emit_append(_kbase=_kbase):
                                # K and V each drain to a shim S2MM; the allocator
                                # picks distinct shim tiles for the two decls.
                                # air-annotate-append-barrier derives the
                                # append->readback ordering from the RAW on the shared DDR
                                # cache: these gets write it, the readback below reads it.
                                if KV_REGION:
                                    # Region-major append (= the reference _receive_kv_cache):
                                    # scatter this token's K (resp V) into the NGRP group
                                    # regions. Channel delivers [g0 K|g1 K|...] (REGION_W
                                    # each, CU-order); the nd write places group gi at its
                                    # region slot (ATTN_L-1)*REGION_W. outer dim=NGRP at
                                    # REGION_STRIDE, inner REGION_W contiguous.
                                    # DECODE_BATCH: B tokens append B CONSECUTIVE
                                    # positions, so this gains a leading token
                                    # dimension at stride REGION_W (kvappend_bd.py).
                                    # The flat base stays on the stride-1
                                    # dimension, where AIR's left-padding already
                                    # puts it at batch 1.
                                    # ONE GET for the whole block, the 3-D
                                    # descriptor kvappend_bd.py derives: token
                                    # t's K lands at position p+t of every
                                    # region.
                                    #
                                    # Per-token gets were tried, for 1:1 pairing
                                    # with rope's per-token puts. They cap rope:
                                    # each is a separate shim task, the fused
                                    # launch paces a preserve_shim_dma_order
                                    # channel at depth 2, and the appends stop
                                    # arriving partway through the block -- 6 of
                                    # 8 tokens reached the cache and rope waited
                                    # forever for the seventh. One task per
                                    # channel drains continuously, and the
                                    # packet flow reassembles B puts into it the
                                    # same way attnO and ropeQ already do.
                                    _bt = [idx(BATCH)] if BATCH > 1 else []
                                    _bs = [idx(REGION_W)] if BATCH > 1 else []
                                    _bo = [idx(0), idx(0)] if BATCH > 1 else []
                                    ChannelGet(
                                        "appendK",
                                        KVC,
                                        indices=[idx(0)],
                                        offsets=_bo + [_loi_slot(_kbase, 0)],
                                        sizes=_bt + [idx(NGRP), idx(REGION_W)],
                                        strides=_bs + [idx(REGION_STRIDE), idx(1)],
                                    )
                                    ChannelGet(
                                        "appendV",
                                        KVC,
                                        indices=[idx(0)],
                                        offsets=_bo + [_loi_slot(_kbase, _vreg_off(0))],
                                        sizes=_bt + [idx(NGRP), idx(REGION_W)],
                                        strides=_bs + [idx(REGION_STRIDE), idx(1)],
                                    )
                                    return

                            def _emit_readback(_kbase=_kbase):
                                # One whole-cache pass PER TOKEN. The block does
                                # not share a KV read: each token runs the
                                # attention CU once, so the shim streams the
                                # cache B times. That is section 5e's "attention
                                # does not amortize" as a descriptor count, and
                                # it is the reason the block size was priced at
                                # 8 rather than 16.
                                # ONE descriptor, B times over the same region:
                                # a leading dimension of extent BATCH and STRIDE
                                # 0. Not a saving -- a correctness fix of the
                                # same kind as the KV append. B separate passes
                                # are B separate shim tasks, the fused launch
                                # paces a preserve_shim_dma_order channel at
                                # depth 2, and the append hit exactly that (6 of
                                # 8 tokens reached the cache and rope waited
                                # forever for the seventh). One task per channel
                                # drains continuously.
                                _emit_readback_one(_kbase)

                            def _emit_readback_one(_kbase=_kbase):
                                # KV readback as ONE 4D strided nd-DMA per CU (was ATTN_ROUNDS
                                # separate per-block puts). The whole per-CU cache
                                # [ATTN_ROUNDS][2(K|V)][16 pos][KVPC_DH] is read in a single shim
                                # BD; the memtile consumer (_reblock_dec) still dequeues it
                                # block-by-block (FIFO stream). Mirrors the reference's few-large-strided
                                # transfers -> cuts inKV shim issues 4*ATTN_ROUNDS*16 -> 4*16 at
                                # L=2048 (the measured 2K bottleneck). Env DECODE_KV_NDDMA=0 falls
                                # back to the rolled per-block ring.
                                if KV_SPLIT and KV_REGION:
                                    # REGION-MAJOR readback (= the reference _move_kv_cache): each
                                    # group's K (resp V) region is CONTIGUOUS in DDR. Split
                                    # each region into NRB contiguous chunks and interleave
                                    # K_gi,V_gi per chunk on the 2 independent inKV_K/inKV_V
                                    # channels. WHY NRB>=dep+1 (default 4): the fused N-wave
                                    # launch paces each preserve_shim_dma_order channel
                                    # PER WAVE at depth 2 (synthesizeDoubleBufferedAwaits);
                                    # with 1 task/channel/wave it FENCES (start;await inline)
                                    # -> serializes K before V -> the qk->score->kv pipeline
                                    # (K's 128-block BD can't drain a depth-2 ring while V
                                    # hasn't started) DEADLOCKS at large L. With >depth
                                    # chunks/channel it BATCHES (2 in flight) so K and V
                                    # stream concurrently -> pipeline flows. Chunks stay
                                    # separate BDs (per-channel folding is off under the
                                    # preserve launch) yet each is CONTIGUOUS (coalescible
                                    # shim burst) -> ~NRB*2*NGRP tasks/layer (e.g. 16) vs the
                                    # interleaved layout's ~1 task/token (~4100 @L2k). The
                                    # memtile (_reblock_dec) dequeues per block (16*REGION_W).
                                    # the reference fires its 4 KV readback memcpy fire-and-free (no
                                    # per-task await); K/V on independent channels stream
                                    # concurrently, backpressured only by the memtile ring
                                    # locks. The readback reaches no broadcast-consuming
                                    # herd, so the compiler keeps it OUT of the
                                    # preserve-launch's depth-2 pacing (whose await-on-drain
                                    # would serialize K before V and deadlock once a BD
                                    # exceeds the ring depth) and lowers it to a
                                    # fire-and-free MM2S feed. With NRB=1 that is exactly
                                    # the reference's 2*NGRP (=4) whole-region contiguous transfers.
                                    _NRB = int(_os.environ.get("DECODE_KV_RB_NRB", "1"))
                                    _nb = RB_ROUNDS
                                    _cbk = (_nb + _NRB - 1) // _NRB  # blocks per chunk
                                    # KV_RB_1D: emit the readback as ONE 1-D descriptor instead of
                                    # the 3-D [cb,16,REGION_W]/[16*REGION_W,REGION_W,1]. Those
                                    # strides are exactly the products of the inner sizes, so the
                                    # region is already perfectly contiguous (max offset
                                    # cb*16*REGION_W-1, no gaps) -- the 3-D form describes a plain
                                    # linear run. It is NOT free, though: the shim DMA then has to
                                    # sequence cb*16 (=2048 @L2k) inner runs of REGION_W*2 (=512) B
                                    # per BD instead of one, and only a contiguous 1-D BD gets the
                                    # wide buffer_length register (same reason the weight feed is
                                    # kept 1-D). FLM issues this identical region as a single
                                    # LINEAR transfer -- see its seq col3/col4 BDs, "A linear
                                    # transfer, no D0", 1,056,768 B. Same bytes, same addresses,
                                    # same order; only the descriptor shape differs.
                                    _KV1D = int(_os.environ.get("KV_RB_1D", "0"))
                                    if DYNSEQ and (_NRB != 1 or _KV1D):
                                        raise SystemExit(
                                            "DECODE_DYNSEQ needs the single whole-region "
                                            "readback (DECODE_KV_RB_NRB=1, KV_RB_1D=0): "
                                            "chunking splits a runtime count across "
                                            "compile-time BDs, and the 1-D form folds it "
                                            "into a length the shim cannot recompute."
                                        )
                                    _ci = 0
                                    while _ci < _nb:
                                        _cb = min(_cbk, _nb - _ci)
                                        _coff = _ci * 16 * REGION_W
                                        # DYNSEQ: the outer block count is the runtime
                                        # ceil(L/16), so the BD moves this token's context
                                        # rather than the padded ATTN_MAXL. Called at each
                                        # use, not hoisted, so the static path's constant
                                        # emission order -- and thus its IR -- is unchanged.
                                        _cbv = (
                                            _rt_blocks
                                            if DYNSEQ_RB
                                            else (lambda: idx(_cb))
                                        )
                                        # Contiguous either way; _KV1D just states it as 1-D.
                                        # Spelled inline (not hoisted) so the default path's
                                        # constant emission order -- and thus the emitted IR --
                                        # is byte-identical to before this flag existed.
                                        # extent BATCH, stride 0 -> re-read.
                                        _rb = [idx(BATCH)] if BATCH > 1 else []
                                        _r0 = [idx(0)] if BATCH > 1 else []
                                        for gi in range(NGRP):
                                            ChannelPut(
                                                "inKV_K",
                                                KVC,
                                                indices=[idx(gi)],
                                                offsets=_r0
                                                + [_loi(_kbase, _kreg_off(gi) + _coff)],
                                                sizes=(
                                                    _rb + [idx(_cb * 16 * REGION_W)]
                                                    if _KV1D
                                                    else _rb
                                                    + [
                                                        _cbv(),
                                                        idx(16),
                                                        idx(REGION_W),
                                                    ]
                                                ),
                                                strides=(
                                                    _r0 + [idx(1)]
                                                    if _KV1D
                                                    else _r0
                                                    + [
                                                        idx(16 * REGION_W),
                                                        idx(REGION_W),
                                                        idx(1),
                                                    ]
                                                ),
                                            )
                                            ChannelPut(
                                                "inKV_V",
                                                KVC,
                                                indices=[idx(gi)],
                                                offsets=_r0
                                                + [_loi(_kbase, _vreg_off(gi) + _coff)],
                                                sizes=(
                                                    _rb + [idx(_cb * 16 * REGION_W)]
                                                    if _KV1D
                                                    else _rb
                                                    + [
                                                        _cbv(),
                                                        idx(16),
                                                        idx(REGION_W),
                                                    ]
                                                ),
                                                strides=(
                                                    _r0 + [idx(1)]
                                                    if _KV1D
                                                    else _r0
                                                    + [
                                                        idx(16 * REGION_W),
                                                        idx(REGION_W),
                                                        idx(1),
                                                    ]
                                                ),
                                            )
                                        _ci += _cb
                                    return

                            # the reference cadence (MULTIBLK): interleave append+readback at the QKV|o
                            # weight boundary -- append after QKV weights (rope has produced
                            # K/V), barrier, readback, THEN o/up/down weights.
                            woff = 0
                            for p in range(NPH):
                                per_col = PER_COL_PH[p]
                                assert per_col % NCY == 0
                                _colspan = per_col * blk
                                # Spatial fan over the NCX proj columns: the bundle index
                                # @inW[cx] must be an scf.parallel IV (canonical form; a
                                # temporal scf.for over a bundle index is a verifier error).
                                # Each column is one contiguous DDR block; _feed_wcol
                                # emits it as TWO halves per shim channel so the
                                # coalescer merges+tags them (air.coalesced_shim_feed =
                                # the cross-channel phase barrier) -- a single put would
                                # skip coalescing and lose that barrier.
                                # air-to-aie spatially unrolls this to the per-column feeds.
                                _wcol0 = _lo(_wbase, woff)  # _wbase + woff (col 0 base)
                                _feed_wcols(_wcol0, _colspan, per_col // NCY)
                                woff += NCX * per_col * blk
                                if MULTIBLK and p == 0:
                                    _emit_append()
                                    _emit_readback()
                                if RMS_BAND_STREAM >= 3 and not _RMS_LATE_RT:
                                    _rms_interleave_after_phase(p)
                            # per-dest host drain: dest p drains ROUNDS_PER_DEST[p] rounds into
                            # this layer's Y region (diagnostic per-layer QKV observation).
                            roff = 0
                            for p in HOST_DRAIN:
                                if p == 0:
                                    # loop close: dest0 (QKV->rope->flash attention) o is
                                    # consumed on-chip as the o-proj X (not drained to host).
                                    pass
                                else:
                                    for rr in range(ROUNDS_PER_DEST[p]):
                                        ChannelGet(
                                            "toShim",
                                            Y,
                                            indices=[idx(p)],
                                            offsets=[
                                                _lo(_ybase, (roff + rr) * PAYLOAD)
                                            ],
                                            sizes=[PAYLOAD],
                                            strides=[1],
                                        )
                                roff += ROUNDS_PER_DEST[p]
                            # #4: drain the rms layer output (residual2 = h + down). the reference
                            # chaining ABI: write res2 (the new hidden states) IN-PLACE into
                            # arg0 (X) at offset 0, so it feeds the NEXT layer from the same BO.
                            # The next layer's rmsX read (above) is program-ordered after this
                            # write (air.preserve_shim_dma_order) -> layer chaining.
                            #
                            # HIDDEN_TAPS moves this to slot a_iv+1 (and the read to
                            # slot a_iv), which keeps the chain -- same BO, same
                            # ordering, still the next layer's input -- while leaving
                            # every earlier layer's output intact to read back.
                            if RMS_BAND_STREAM >= 3:
                                # residual1, ph2 and residual2 all land here,
                                # after every other launch-side feed for
                                # this layer -- see _rms_residual_roundtrip's
                                # comment on why (air.preserve_shim_dma_order
                                # is global, so anything placed earlier
                                # would force later, unrelated feeds to
                                # queue behind a get the rms core cannot
                                # satisfy yet). K // STG_W, not OPROJ_RNDS/
                                # DOWN_RNDS directly: those names live in a
                                # different nested scope, and K // STG_W is
                                # the same value -- asserted equal at
                                # STG_W's own definition.
                                #
                                # ph0's feed lands here too, and that is the
                                # fix for a deadlock rather than tidiness. It
                                # used to be emitted at the TOP of _uni_dec,
                                # where a whole-row level-0 rmsX read is
                                # correct: that read completes the instant the
                                # core takes it, so the projection weight feed
                                # queued behind it goes out immediately. A
                                # BANDED ph0 does not. It completes only after
                                # the core has consumed 1+XN_REFEED sweeps, and
                                # every sweep but the first also PRODUCES
                                # @xnorm chunks, which drain only if the
                                # projection cores are consuming, which needs
                                # the @inW weights -- issued after ph0 in the
                                # global shim order, and therefore never. The
                                # dispatch hangs in wave 0 with nothing written
                                # at all, not even KV.
                                #
                                # So every banded feed goes after the weight
                                # feed, in the core's own order. ph0's is the
                                # exception and sits further up, at the QKV
                                # weight/KV-append boundary -- see there.
                                _nband = K // STG_W
                                if not _RMS_LATE_RT:
                                    # DEFAULT. Every round-trip and ph2's feed
                                    # were armed inside the weight-feed loop,
                                    # one phase ahead of the await that needs
                                    # them -- see _rms_interleave_after_phase
                                    # for why anything left here deadlocks.
                                    pass
                                elif _os.environ.get("_RMS_ALL_TOP"):
                                    # Already emitted at the top of _uni_dec.
                                    pass
                                elif _os.environ.get("_RMS_PH0_ONLY"):
                                    # Bisection probe. Feed ph0 and NOTHING
                                    # else, so the launch side stops where the
                                    # core's residual1 begins. The dispatch is
                                    # then guaranteed to hang -- the core waits
                                    # for a round-trip that is never fed -- but
                                    # WHERE it hangs is the datum: ph0 drives
                                    # the QKV projection and rope, so if the KV
                                    # cache comes back written, everything up
                                    # to and including ph0 works and the fault
                                    # is in the residual round-trip. If nothing
                                    # is written, the fault is in ph0 or
                                    # earlier. Not a shipping path.
                                    pass
                                elif _os.environ.get("_RMS_RES1_AT_P1"):
                                    # residual1 + ph2 were emitted at the o-proj
                                    # weight boundary; only residual2 is left.
                                    _rms_residual_roundtrip(_x_out(), _nband)
                                elif _os.environ.get("_RMS_RES1_PUT_TOP"):
                                    _rms_layerout_drain_allbands(X, _x_in())
                                    _rms_x_feed_ph(_x_in(), REFEED[GATEUP_PHASE])
                                    _rms_residual_roundtrip(_x_out(), _nband)
                                else:
                                    _rms_residual_roundtrip(_x_in(), _nband)
                                    _rms_x_feed_ph(_x_in(), REFEED[GATEUP_PHASE])
                                    _rms_residual_roundtrip(_x_out(), _nband)
                            else:
                                _out_bo = X
                                _out_base = _x_out()
                                # BD-COMPACTION: single full-size drain (matches the rms single
                                # layerOut put) instead of LAYER_RNDS per-round gets.
                                ChannelGet(
                                    "layerOut",
                                    _out_bo,
                                    indices=[idx(0)],
                                    offsets=[_out_base],
                                    sizes=[BATCH * LAYER_RNDS * PAYLOAD],
                                    strides=[1],
                                )
                            # The DECODE_PROBE taps drain LAST, one shim task per
                            # source tile. Not mid-layer, and this cost a build to
                            # learn: preserve_shim_dma_order is global, so a tap
                            # task placed between two weight feeds puts the LATER
                            # feed behind data the layer cannot produce until that
                            # feed has been consumed. Behind the layer output, a
                            # tap is behind everything and gates nothing.
                            if PROBE_5_LEN:
                                ChannelGet(
                                    "probe5",
                                    Y,
                                    indices=[idx(0)],
                                    offsets=[PROBE_OFF["Q"]],
                                    sizes=[PROBE_5_LEN],
                                    strides=[1],
                                )
                            if PROBE_LEN["D"]:
                                ChannelGet(
                                    "probe4",
                                    Y,
                                    indices=[idx(0)],
                                    offsets=[PROBE_OFF["D"]],
                                    sizes=[PROBE_LEN["D"]],
                                    strides=[1],
                                )
                            yield_([])

                        if not N_EXTRA:
                            index_switch(
                                [],
                                _uarm_i,
                                [0],
                                case_body_builder=lambda op, i, cv: _uni_voc(),
                                default_body_builder=lambda op: _uni_dec(),
                            )
                        else:
                            # arm 0 = LM head, 2+k = extra wave k, default = decode.
                            # Decode stays the DEFAULT arm so its (much larger)
                            # body is emitted exactly where it was. Each of these
                            # emits its own scf.yield, as the two-arm form's
                            # bodies already did.
                            _cases = [0] + [2 + k for k in range(N_EXTRA)]

                            def _case(cv):
                                return _uni_voc() if cv == 0 else _uni_extra(cv - 2)

                            index_switch(
                                [],
                                _uarm_i,
                                _cases,
                                case_body_builder=lambda op, i, cv: _case(cv),
                                default_body_builder=lambda op: _uni_dec(),
                            )
                # (No GLU host drain: the GLU output is consumed on-chip by the down
                # phase. The down output egresses via the rms layer output above.)

                _seg_opers = ([a_iv] if a_iv is not None else []) + (
                    [L_rt] if DYNSEQ else []
                )

                @segment(name="seg", operands=_seg_opers)
                def seg(*_sa):
                    _seg_iv = _sa[0] if a_iv is not None else None
                    # The context length reaches the attention herd from here, as a
                    # herd operand: an RTP slot the instruction stream writes per
                    # dispatch, not a constant folded into the core ELF.
                    _seg_L = _sa[-1] if DYNSEQ else None

                    def _seg_blocks():
                        """Blocks the memtile dequeues for a WHOLE token block.

                        BATCH * ceil(L_blk/16), as ONE loop bound rather than a
                        token loop around the block loop. The nesting is what
                        AIR gets to reinterpret -- the fresh per-iteration
                        allocation inside is what makes the count-free ping-pong
                        ring, and an outer loop wrapped around it is a second
                        thing for the ring transform to fold. The body does not
                        depend on which token it is serving, so flattening is
                        free and leaves nothing to fold.
                        """
                        r = _seg_rounds()
                        if BATCH == 1:
                            return r
                        return arith.muli(r, idx(BATCH))

                    def _seg_rounds():
                        """ceil(L_blk/16) for the memtile's block dequeue.

                        The memtile sits between the shim's readback BD and the
                        cores, so its trip count has to be the same ceil(L/16) both
                        of those use -- and at BATCH>1 that is the LAST token's
                        L (see ATTN_L_BLK), uniformly for every token.
                        """
                        if not DYNSEQ_MEM:
                            return idx(ATTN_ROUNDS)
                        # Same decode as the cores'. The memtile's dequeue count,
                        # the shim's push count and the cores' trip count must
                        # agree, so the mode bit has to be stripped here too --
                        # left in, it would ask the memtile for 2^26 blocks.
                        _sb, _ = _split_mask_mode(_seg_L)
                        _s = arith.addi(
                            _sb,
                            arith.ConstantOp(
                                IntegerAttr.get(i32, 15 + BATCH - 1), None
                            ).result,
                        )
                        _q = arith.divui(
                            _s,
                            arith.ConstantOp(IntegerAttr.get(i32, 16), None).result,
                        )
                        return arith.index_cast(idx_t, _q)

                    if _seg_iv is not None:
                        _seg_cmp = arith.cmpi(
                            arith.CmpIPredicate.slt, _seg_iv, idx(UNI_DEC)
                        )
                        if not N_EXTRA:
                            _seg_arm = arith.select(
                                _seg_cmp,
                                arith.ConstantOp(IntegerAttr.get(i32, 1), None).result,
                                arith.ConstantOp(IntegerAttr.get(i32, 0), None).result,
                            )
                        else:
                            # Same encoding as the launch's _uarm: 1 decode, 0 LM
                            # head, 2+k extra wave k. The memtile and the herds
                            # take it from here, so all three scopes agree by
                            # construction rather than by two copies of a rule.
                            _seg_arm = arith.select(
                                _seg_cmp,
                                arith.ConstantOp(IntegerAttr.get(i32, 1), None).result,
                                arith.select(
                                    arith.cmpi(
                                        arith.CmpIPredicate.slt,
                                        _seg_iv,
                                        idx(UNI_DEC + UNI_LM),
                                    ),
                                    arith.ConstantOp(
                                        IntegerAttr.get(i32, 0), None
                                    ).result,
                                    arith.addi(
                                        arith.index_cast(
                                            i32,
                                            arith.subi(_seg_iv, idx(UNI_DEC + UNI_LM)),
                                        ),
                                        arith.ConstantOp(
                                            IntegerAttr.get(i32, 2), None
                                        ).result,
                                    ),
                                ),
                            )
                        _seg_arm_i = arith.index_cast(idx_t, _seg_arm)
                    else:
                        _seg_cmp = None
                        _seg_arm_i = None
                        _seg_arm = arith.ConstantOp(
                            IntegerAttr.get(i32, 0 if LM_HEAD else 1), None
                        ).result

                    # ===== X memtile (the reference mem_1_1 x_buffer): 512 ring, re-fed =====
                    # The cores read X in phase order: phases 0..2 read the rmsnorm'd
                    # token X (K=2048), phase 3 (down) reads the GLU output (K=8192)
                    # fed back on-chip. The SAME inX broadcast carries both, in order.
                    #
                    # (1) rms-X: get the normed X (from the rms core, re-fed RMS_REFEED
                    # times over @xnorm) in 512 chunks -> broadcast
                    # 256-blocks. RMS_REFEED*(2048/512) gets. (reproducer core_2_2 +
                    # mem_1_1 x_buffer 512.)
                    def _rmsrow_relay(out, seq, rb, src="xnorm"):
                        """ph0/ph2 X relay: one fill of a row-major [BATCH][K]
                        slab from @xnorm per entry of `seq`, each re-broadcast
                        that entry's number of times onto `out` -- the same
                        mechanism the o-buf (ph1) and down_buffer (ph3) relays
                        above/below use, one hop further upstream.

                        `seq` is this phase's slice of the re-feed cycle (see
                        _rf_seq). It is a SEQUENCE and not a single count
                        because the count is a lock value baked into a BD, the
                        arms share one BD chain, and the LM head's total
                        re-sends differ from the decode arm's -- so the two
                        arms have to walk the same short cycle a different
                        number of times rather than each name its own count.
                        Every entry folds onto one of the len(_RF_CYCLE) BDs.

                        The rms core now emits BATCH WHOLE ROWS instead of
                        regenerating cross-row chunks once per output row-block,
                        so each value is computed exactly once. At block 1 the
                        core could hold its normalized row and re-feed it itself
                        (one `Release x12` on the @xnorm read lock, no
                        recompute); at block 8 the normalized batch is 40 KB on
                        top of a 40 KB residual and it cannot, which is the
                        whole 4.71 ms.

                        MUST BE CALLED IN PHASE-TIME ORDER, ph0 then ph2. Every
                        fill reads @xnorm from the same tile, so AIR chains them
                        into ONE inbound BD cycle, and that cycle is only
                        correct because it matches the order the core produces
                        the fills in.

                        `out`'s stream is byte-identical to what @xnorm carries
                        today. _xnorm_put's batched form is already the
                        chunk-major read of a row-major slab -- sizes
                        [K/XCHUNK, BATCH, XCHUNK], strides [XCHUNK, K, 1] -- so
                        _feed_inX, @inX and the 16 projection cores see exactly
                        the stream they see now and none of them change.
                        """
                        if RMS_MEMTILE_REFEED != 3:
                            return
                        # ONE SLAB PER CYCLE POSITION, PING-PONGED. With a
                        # single slab, fill N+1's BD is not armed until fill N's
                        # re-sends have all gone -- so the core's row-0 transfer
                        # of the next sweep STALLS, and because a compute tile's
                        # channel.put lowers to `call; acquire(avail);
                        # release(ready)` the core has already written row 1
                        # into the staging buffer before it blocks. The stalled
                        # row-0 transfer then carries row 1's bytes, and every
                        # fill after the first in a phase loses its row 0.
                        # Measured on the LM head, where only that arm has two
                        # fills: token 0's logits come back a BLEND of rows 0
                        # and 1 (corr 0.845 to each, against 0.992/0.681 for the
                        # single-fill build) while rows 1..7 stay bit-identical.
                        #
                        # A SECOND SLAB DOES NOT FIX IT, and the reason is worth
                        # writing down because it looks like it should. Ping-pong
                        # the fills across two slabs and each one's BD is armed
                        # while the other drains, so nothing stalls -- the fill
                        # side lowers perfectly (each fill acquires its OWN count
                        # against its own lock's init; see
                        # memtile_per_fill_refeed_pingpong.mlir). The DRAIN side
                        # does not. Two slabs are two buffers, so the drain
                        # becomes a TWO-BD chain that alternates slab0, slab1,
                        # slab0 -- one transfer per BD, that is what a chain is --
                        # and what is needed is 12 consecutive sends of slab0 and
                        # then 38 of slab1. The locks deadlock after 12 of each:
                        # built, dispatched, hung with nothing written. A
                        # count-free self-looping drain is only expressible over
                        # ONE buffer, which is why this stays at one slab.
                        #
                        # ONE FLAT GET, not a per-token loop. The core ships
                        # BATCH whole rows back to back, so a single contiguous
                        # get of BATCH*K lands exactly the row-major [BATCH][K]
                        # the strided drain below reads -- same bytes, one
                        # dma_start instead of BATCH of them. It also has to be
                        # one op: air-annotate-refeed puts this fill's re-send
                        # count on the GET, and a fill spread over several gets
                        # has no single BD to carry it (the pass says so rather
                        # than guessing).
                        for _cnt in seq:
                            # `src` is @xnorm for the decode and LM-head arms
                            # and @tapsX for an extra wave. A DIFFERENT CHANNEL
                            # is the whole point: DMA chains are per tile AND
                            # channel, so the extra arm's fill sits on its own
                            # S2MM chain and returns to its own BD 0, instead of
                            # advancing the [12, 38] chain the other arms walk.
                            # The drain below is shared -- same buffer, same
                            # descriptor, same channel -- so it stays the ONE
                            # count-free self-looping BD the mode requires.
                            ChannelGet(
                                src,
                                rb,
                                offsets=[idx(0)],
                                sizes=[idx(BATCH * K)],
                                strides=[idx(1)],
                            )
                            refeed(_cnt, lambda: _xnorm_put(rb, K, chan=out))

                    def _feed_inX(src, total_chunks):
                        for _rc in for_(idx(0), idx(total_chunks), idx(1)):
                            xb = AllocOp(xmt_l2, [], [])
                            xb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), XMT_PCOL)
                            )
                            # BATCH*2*COL_BLOCK, contiguous: the producers put
                            # chunk-major (see _xnorm_put), so one get lands the
                            # B tokens' current window side by side.
                            ChannelGet(
                                src,
                                xb,
                                offsets=[0],
                                sizes=[BATCH * XCHUNK],
                                strides=[1],
                            )
                            for _jj in for_(idx(0), idx(XCHUNK_MUL), idx(1)):
                                if BATCH == 1:
                                    joff = arith.muli(_jj, idx(COL_BLOCK))
                                    ChannelPut(
                                        "inX",
                                        xb,
                                        offsets=[joff],
                                        sizes=[COL_BLOCK],
                                        strides=[1],
                                    )
                                else:
                                    # Tile-blocked broadcast: the proj cores'
                                    # q4k_mm_block feeds aie::mmul, which wants
                                    # its A operand in tile order. Descriptor
                                    # from xfeed_bd.py, which checks it against
                                    # pack_A elementwise -- imported rather than
                                    # restated so the two cannot drift.
                                    _sz, _st = _XFEED_BD
                                    _o0 = arith.muli(_jj, idx(COL_BLOCK // _st[0]))
                                    ChannelPut(
                                        "inX",
                                        xb,
                                        offsets=[_o0] + [idx(0)] * (len(_sz) - 1),
                                        sizes=[idx(v) for v in _sz],
                                        strides=[idx(v) for v in _st],
                                    )
                                yield_([])
                            DeallocOp(xb)
                            yield_([])

                    def _feed_inX_rf(src, nchunk, nrefeed, wb=None):
                        """RMS_MEMTILE_REFEED: gather one phase's X, then
                        re-broadcast it -- the memtile doing what the core used
                        to do by recomputing.

                        `wb` lets sibling calls in the SAME arm share one gather
                        buffer. They are sequential -- ph0's re-broadcast is
                        finished before ph2 gathers -- so a second allocation
                        buys nothing and costs its own BD chain on a channel
                        that has none to spare (see the BD-ID note above
                        RMS_MEMTILE_REFEED).

                        The gather is nchunk gets of one [BATCH][XCHUNK] window
                        each, landing back to back, so the wide buffer ends up
                        chunk-major/token-minor -- exactly what _xnorm_put
                        already produces and what the put below indexes. The
                        re-broadcast then walks (refeed, chunk) in that order,
                        because THAT is the order the projection cores consume:
                        row-block r wants col-blocks 0..nchunk-1 in sequence,
                        and its accumulator is live across them. Emitting
                        chunk-major here would hand row-block r the same chunk
                        nrefeed times.
                        """
                        _own = wb is None
                        if _own:
                            wb = AllocOp(xmt_wide_l2, [], [])
                        # NOT on XMT_PCOL. The X memtile is already at its 48-BD-block
                        # ceiling -- adding this gather there fails air-to-aie with
                        # 'aie.memtile_dma' op has more than 48 blocks, both with all
                        # four decode phases converted and with ph2 alone. The other
                        # memtiles carry 18-33 blocks, so the gather goes to one with
                        # headroom and @inX originates from there for these phases.
                        if _own:
                            wb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), RMS_MEMTILE_COL)
                            )
                        _win = BATCH * XCHUNK
                        # ONE flat get for the whole gather. The chunks land back
                        # to back, so nchunk windows of _win ARE one contiguous
                        # nchunk*_win run -- a loop of offset gets says the same
                        # thing in nchunk ops, and each op is a dma_start. The
                        # producer still emits nchunk separate puts; channel
                        # balance is on total elements, not op count.
                        ChannelGet(
                            src,
                            wb,
                            offsets=[idx(0)],
                            sizes=[idx(nchunk * _win)],
                            strides=[idx(1)],
                        )
                        # THE SWEEP CANNOT BE ONE OP, and the reason bounds this
                        # whole approach. Folding it into the xfeed descriptor
                        # needs [nchunk, XCHUNK_MUL] outside _XFEED_BD's three
                        # dimensions -- five in total, and this file already
                        # records that the fifth is dropped in AIRRtToNpuPass
                        # WITHOUT a diagnostic (see _rms_x_feed_ph). The two
                        # cannot merge either: consecutive XCHUNK_MUL windows sit
                        # 32 elements apart while the inner dimension already
                        # walks stride 8, so they interleave rather than
                        # partition.
                        #
                        # So it stays an explicit nest, and the nest costs BD
                        # blocks. Four decode phases split out this way is
                        #   'aie.memtile_dma' op has more than 48 blocks
                        # -- measured. Hence RMS_MEMTILE_REFEED=2, which converts
                        # ph2 alone (190 of the 250 redundant chunks a layer) and
                        # leaves ph0/ph1/ph3 in one flat loop.
                        # ONE put per chunk, not XCHUNK_MUL of them.
                        #
                        # The paragraph above says the two windows a chunk splits
                        # into "interleave rather than partition". They
                        # partition. Consecutive windows are COL_BLOCK apart in
                        # ADDRESS, not 32 elements -- offsets are multiplied by
                        # strides, which is the exact trap chunk_offsets' own
                        # docstring is written to prevent, applied here in
                        # reverse. So the outermost extent going from kcol//s to
                        # XCHUNK//s covers both windows in one descriptor, and
                        # xfeed_bd(BATCH, XCHUNK, 1) IS that descriptor:
                        # [(64,8),(8,512),(8,1)] against [(32,8),(8,512),(8,1)].
                        # Enumerating both address lists gives the same 4096
                        # elements in the same ORDER, so the emitted stream is
                        # byte-for-byte what the two puts produced.
                        #
                        # That drops a level off the nest, and the nest is what
                        # spends the BD blocks this knob dies on.
                        # Mode 3: the producer ships WHOLE ROWS, so the gather
                        # buffer is row-major [BATCH][K] and a token is K apart
                        # rather than XCHUNK. That is the ONLY difference --
                        # xfeed_bd already parameterizes the token pitch, and
                        # enumerating both address lists shows the emitted
                        # (token, column) sequence is identical for every chunk.
                        # The projection cores cannot tell the two apart.
                        _rowmaj = RMS_MEMTILE_REFEED == 3
                        _sz, _st = _xfd.xfeed_bd(
                            BATCH, XCHUNK, (K // XCHUNK) if _rowmaj else 1
                        )
                        # Chunk stride: XCHUNK elements along a row when
                        # row-major, a whole [BATCH][XCHUNK] window when not.
                        _cstride = XCHUNK if _rowmaj else _win
                        # offsets[0] is in units of strides[0], not elements --
                        # the same rule. The chunk base is _win ELEMENTS, so it
                        # is _win // _st[0] here; writing _win flatly would
                        # transfer exactly the right count from 8x the wrong
                        # place.
                        # ONE put covering ALL nchunk chunks, wrapped in refeed().
                        #
                        # This is the whole reason the knob never built. refeed()
                        # requires "an n-trip scf.for around a SINGLE
                        # air.channel.put ... no operand depends on the induction
                        # variable" -- that is the shape air-annotate-refeed
                        # collapses into air.refeed_count. The nest that was here
                        # violated both halves: the body held an inner loop, and
                        # the put's offset depended on that loop's index. So the
                        # pattern never fired, every (refeed, chunk) put stayed a
                        # separate op, and each became its own aie.dma_start --
                        # six of them on one channel, 27 BD IDs against 24.
                        #
                        # Folding the chunk dim into the descriptor makes the body
                        # a bare put again. AIRToAIEPass then does exactly what is
                        # wanted (AIRToAIEPass.cpp ~6618): the fill acquires the
                        # write lock xN and releases the read lock xN, and the
                        # MM2S self-loops count-free, re-sending ONE resident
                        # buffer N times. One dma_start per direction.
                        #
                        # 4 dims is the memtile BD's limit and this uses exactly
                        # 4. The highest stride is _win, NOT 0 -- a stride-0
                        # repeat dim is HW-unsupported on AIE2 memtile/core BDs
                        # (air_channel_producer_refeed.mlir says so), which is
                        # why the repeat is a lock count rather than a dimension.
                        _sz4 = [nchunk] + list(_sz)
                        _st4 = [_cstride] + list(_st)

                        def _put_all():
                            ChannelPut(
                                "inX",
                                wb,
                                offsets=[idx(0)] * len(_sz4),
                                sizes=[idx(v) for v in _sz4],
                                strides=[idx(v) for v in _st4],
                            )

                        refeed(nrefeed, _put_all)
                        if _own:
                            DeallocOp(wb)

                    def _rf_buf():
                        """One gather buffer for every _feed_inX_rf in an arm."""
                        wb = AllocOp(xmt_wide_l2, [], [])
                        wb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(), RMS_MEMTILE_COL
                        )
                        return wb

                    # ONE feed loop reading the convergent @xnorm: rms-X (RMS_REFEED
                    # whole-2048 re-reads, from the rms core) THEN down-X (DOWN_REFEED
                    # whole-8192 re-reads, from the down_buffer) -- both converge on
                    # @xnorm by packet, consumed in phase order (ph0-2 then ph3). ONE
                    # loop => one repeat count => count-free broadcast (no repeat_count).
                    # ONE count-free loop reads the convergent @xnorm. LOOPCLOSE: 4
                    # phase sources converge on @xnorm in TIME order (rms ph0 -> o-buf
                    # ph1 -> buf_ph2 ph2 -> down ph3), each re-broadcast its phase count
                    # by its producer (rms channel-refeed; o/buf_ph2/down memtile-refeed).
                    # Else: rms-X (ph0-2) + down-X (ph3), 2 sources.
                    # DIAGNOSTIC (later43b): _feed_inX is NOT count-free (unlike the
                    # weight fan), so it must be vocab-sized in vocab mode -- else the
                    # col-2 X memtile stalls waiting for decode-many xnorm chunks the
                    # rms never produces. Mode-conditional bound (as the validated
                    # standalone). CDO-identity (single xclbin) needs this made
                    # genuinely count-free later; confirming the deadlock cause first.
                    _xc_dec = (REFEED[0] + OPROJ_REFEED + GATEUP_REFEED) * (
                        K // XCHUNK
                    ) + (DOWN_REFEED * (GLU_OUT // XCHUNK) if DOWN_PHASE >= 0 else 0)
                    _xc_voc = VOCAB_RNDS * (K // XCHUNK)
                    if _seg_arm_i is not None:

                        _NCH = K // XCHUNK

                        def _xs_voc():
                            if RMS_MEMTILE_REFEED == 1:
                                _feed_inX_rf("xnorm", _NCH, VOCAB_RNDS)
                            else:
                                # Mode 3 reads @xin0 here too. Not because this
                                # arm works -- it does not, and the knob refuses
                                # to build unless DECODE_NO_LM_WAVES=1 -- but
                                # because a get left on @xnorm gives that
                                # channel a SECOND consumer (the relay tile is
                                # the first), and air-to-aie builds the flow and
                                # spends an S2MM on the X memtile for it whether
                                # or not the arm is ever selected. That channel
                                # is exactly the one this mode frees.
                                _feed_inX(
                                    _XN1 if RMS_MEMTILE_REFEED == 3 else "xnorm",
                                    _xc_voc,
                                )
                            yield_([])

                        def _xs_dec():
                            # Mode 3 does its re-broadcast one hop upstream (a
                            # memtile relay onto @xnorm, like ph1/ph3), so this
                            # side sees exactly the mode-0 stream and stays the
                            # ONE flat loop the BD budget requires.
                            if RMS_MEMTILE_REFEED in (1, 2):
                                # Phase-time order, matching the producers:
                                # rms ph0 -> o-memtile ph1 -> rms ph2 -> down ph3.
                                # Only the two rms phases gather-and-rebroadcast;
                                # ph1 and ph3 already come from memtiles that
                                # refeed on their own side, so they stay flat.
                                if RMS_MEMTILE_REFEED == 1:
                                    # ph0 and ph2 SHARE the gather buffer: ph0's
                                    # re-broadcast is complete before ph2 gathers,
                                    # and a second allocation costs its own BD
                                    # chain on the channel this knob overflows.
                                    _wb = _rf_buf()
                                    _feed_inX_rf("xnorm", _NCH, REFEED[0], _wb)
                                    _feed_inX("xnorm", OPROJ_REFEED * _NCH)
                                else:
                                    # ph2 only: ph0 and ph1 stay in ONE flat loop,
                                    # which is a BD block this memtile cannot spare.
                                    _feed_inX(
                                        "xnorm", (REFEED[0] + OPROJ_REFEED) * _NCH
                                    )
                                _feed_inX_rf(
                                    "xnorm",
                                    _NCH,
                                    GATEUP_REFEED,
                                    _wb if RMS_MEMTILE_REFEED == 1 else None,
                                )
                                if RMS_MEMTILE_REFEED == 1:
                                    DeallocOp(_wb)
                                if DOWN_PHASE >= 0:
                                    _feed_inX(
                                        "xnorm", DOWN_REFEED * (GLU_OUT // XCHUNK)
                                    )
                            elif RMS_MEMTILE_REFEED == 3:
                                # ONE inbound channel, convergent from three
                                # tiles and consumed in phase-time order: ph0
                                # and ph2 from the relay slab, ph1 from the
                                # attnO memtile, ph3 from the down buffer. Same
                                # loop, same count as mode 0 -- only the tile
                                # the chunks come from changed.
                                _feed_inX(_XN1, _xc_dec)
                            else:
                                _feed_inX("xnorm", _xc_dec)
                            yield_([])

                        if not N_EXTRA:
                            index_switch(
                                [],
                                _seg_arm_i,
                                [0],
                                case_body_builder=lambda op, i, cv: _xs_voc(),
                                default_body_builder=lambda op: _xs_dec(),
                            )
                        else:
                            # An extra wave feeds its own K, I2 times -- the same
                            # count its launch-side @xnorm puts supply. Every arm
                            # calls the SAME _feed_inX on the SAME channel and
                            # differs only in the trip count, which is the rule
                            # this whole mechanism is built around.
                            _xc_extra = [
                                w["i2"] * (EXTRA_K[i] // XCHUNK)
                                for i, w in enumerate(EXTRA_WAVES)
                            ]

                            def _xs_extra(k):
                                # _XN1, NOT the literal "xnorm" -- the comment
                                # above is the rule and this line used to break
                                # it. Under mode 0 _XN1 IS "xnorm" so the two
                                # spellings agree and nothing showed; under
                                # RMS_MEMTILE_REFEED=3 _XN1 is "xin" and this
                                # arm was naming a DIFFERENT CHANNEL from every
                                # other one. A channel named by one arm alone
                                # does not survive to be allocated (§3.15), so
                                # the get vanished and the extra arm was left
                                # with no X source at all -- a verify dispatch
                                # that times out with nothing wrong on paper.
                                _feed_inX(_XN1, _xc_extra[k])
                                yield_([])

                            index_switch(
                                [],
                                _seg_arm_i,
                                [0] + [2 + k for k in range(N_EXTRA)],
                                case_body_builder=lambda op, i, cv: (
                                    _xs_voc() if cv == 0 else _xs_extra(cv - 2)
                                ),
                                default_body_builder=lambda op: _xs_dec(),
                            )
                    else:
                        _feed_inX("xnorm", _xc_voc if LM_HEAD else _xc_dec)

                    # ===== weight fan: per col MT peels NCY blocks/(i,j) -> cores ==
                    # Phase-agnostic flat ring (reproducer single w_buffer): W_FAN_STEPS
                    # = total (i,j) steps across all phases; each get fans NCY cy.
                    #
                    # W_DUAL_CHAN emits ONE SUCH RING PER SHIM CHANNEL, each owning a
                    # disjoint half of the column's cores -- @inW drives cy 0..NCY/2-1,
                    # @inW2 drives the rest. This is FLM's mem_C_1 exactly: shim ch0 on
                    # S2MM4 -> w_buffer[0:5120] -> MM2S0/1 -> rows 2/3, shim ch1 on
                    # S2MM5 -> w_buffer[5120:] -> MM2S2/3 -> rows 4/5, on two
                    # INDEPENDENT lock cycles. A SPATIAL split is what makes the two
                    # channels usable: they share no consumer, so neither is ever
                    # ordered against the other.
                    #
                    # A temporal split (@inW even fan steps / @inW2 odd) was tried and
                    # DEADLOCKS the first decode dispatch: it gives every core a single
                    # MM2S BD chain alternating between both channels' buffers, so the
                    # two shim channels are cross-coupled at every fan step. FLM has no
                    # such edge. Do not reintroduce it.
                    _fan_groups = (
                        [(0, 0, NCY // 2), (1, NCY // 2, NCY)]
                        if W_DUAL_CHAN
                        else [(0, 0, NCY)]
                    )
                    for cx in range(NCX):
                        for _ci, _cy0, _cy1 in _fan_groups:
                            _wch = _wname(_ci, cx)
                            for _ in for_(idx(0), idx(W_FAN_STEPS), idx(1)):
                                wf = AllocOp(wfan_l2, [], [])
                                wf.operation.attributes["air.memtile_col"] = (
                                    IntegerAttr.get(T.i32(), PCOL[cx])
                                )
                                ChannelGet(
                                    _wch,
                                    wf,
                                    indices=[idx(0) if W_DUAL_CHAN else idx(cx)],
                                )
                                for cy in range(_cy0, _cy1):
                                    # 1D fixed-offset read (reproducer w_buffer MM2S
                                    # shape) so AIR detects the 2-buffer rotation ->
                                    # count-free next_bd ring (NOT a repeat_count BD).
                                    ChannelPut(
                                        "wL2ToL1",
                                        wf,
                                        indices=[idx(cx), idx(cy)],
                                        offsets=[(cy - _cy0) * BLOCK_BF16],
                                        sizes=[BLOCK_BF16],
                                        strides=[1],
                                    )
                                DeallocOp(wf)
                                yield_([])

                    # ===== output assembly + id-demux egress (count-free ring) =====
                    # ONE for_ loop over all rounds -> count-free next_bd rings (NOT
                    # Python-unrolled, which overflows the 48-BD memtile limit). Per
                    # round: each group MT gathers its 4 leads' packets (asym, one
                    # header @0); the main MT daisy-chains the 2 groups (514); the
                    # egress (outY, packet) is demuxed by the routing header and
                    # relayed (rb) to the shim drain. NPH=1 -> all rounds id1 -> dest0;
                    # the put+get are interleaved in the same iteration so they
                    # pipeline across tiles (separate put-loop/get-loop would deadlock).
                    # DIAGNOSTIC (later43b): the assembly gather is NOT count-free, so
                    # it must be vocab-sized in vocab mode (as the validated standalone)
                    # -- else the col-0/1/6 assembly memtiles stall on outA rounds the
                    # vocab proj never produces. CDO-identity needs this count-free later.
                    def _egress(_nrc):
                        for _r in for_(idx(0), idx(_nrc), idx(1)):
                            for g in range(N_GRP):
                                grp = AllocOp(grp_l2, [], [])
                                grp.operation.attributes["air.memtile_col"] = (
                                    IntegerAttr.get(T.i32(), GRP_PCOL[g])
                                )
                                for k, (cx, pp) in enumerate(grp_leads(g)):
                                    # THE BATCH-1 DESCRIPTOR, B times longer.
                                    # Emitter k's B token blocks land back to
                                    # back and stay that way: the gather does not
                                    # transpose, the consumers do (see
                                    # outy_tokmajor). One BD, header intact, the
                                    # memtile's ping-pong ring untouched.
                                    _pay = PAIR_PAY * BATCH
                                    off = 0 if k == 0 else HDR + k * _pay
                                    sz = (HDR + _pay) if k == 0 else _pay
                                    ChannelGet(
                                        "outA",
                                        grp,
                                        indices=[idx(cx), idx(pp)],
                                        offsets=[off],
                                        sizes=[sz],
                                        strides=[1],
                                    )
                                ChannelPut(
                                    "toMain",
                                    grp,
                                    indices=[idx(g)],
                                    offsets=[0],
                                    sizes=[GRP_ROWS_B],
                                    strides=[1],
                                )
                                DeallocOp(grp)
                            ml = AllocOp(main_l2, [], [])
                            ml.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), MAIN_PCOL)
                            )
                            for g in range(N_GRP):
                                # Again the batch-1 descriptor, B times longer:
                                # the groups stay laid end to end and the round
                                # stays emitter-major all the way to outY.
                                _gpay = LEADS_PER_GRP * PAIR_PAY * BATCH
                                off = 0 if g == 0 else GRP_ROWS_B + (g - 1) * _gpay
                                sz = GRP_ROWS_B if g == 0 else _gpay
                                ChannelGet(
                                    "toMain",
                                    ml,
                                    indices=[idx(g)],
                                    offsets=[off],
                                    sizes=[sz],
                                    strides=[1],
                                )
                            # id-demux source: emit the assembled 514 packet (kernel id in
                            # the header) on ONE MM2S; the switchbox routes it to the dest
                            # matching its id. One source emit per round.
                            ChannelPut(
                                "outY",
                                ml,
                                indices=[idx(0), idx(0)],
                                offsets=[0],
                                sizes=[MAIN_ROWS_B],
                                strides=[1],
                            )
                            DeallocOp(ml)
                            yield_([])

                    if _seg_arm_i is not None:

                        def _egr_voc():
                            _egress(VOCAB_RNDS)
                            yield_([])

                        def _egr_dec():
                            _egress(N_ROUNDS)
                            yield_([])

                        if not N_EXTRA:
                            index_switch(
                                [],
                                _seg_arm_i,
                                [0],
                                case_body_builder=lambda op, i, cv: _egr_voc(),
                                default_body_builder=lambda op: _egr_dec(),
                            )
                        else:
                            # An extra wave drains its own row-block iterations,
                            # EXTRA_RNDS[k] of them. Every arm calls the SAME
                            # _egress over the SAME outA/toMain/outY ops and
                            # differs only in the trip count -- the rule this
                            # mechanism exists to keep (see _rms_batched's
                            # header: a second arm that names a channel op again
                            # doubles that buffer's lock credit and the wave
                            # never completes).
                            def _egr_extra(k):
                                _egress(EXTRA_RNDS[k])
                                yield_([])

                            index_switch(
                                [],
                                _seg_arm_i,
                                [0] + [2 + k for k in range(N_EXTRA)],
                                case_body_builder=lambda op, i, cv: (
                                    _egr_voc() if cv == 0 else _egr_extra(cv - 2)
                                ),
                                default_body_builder=lambda op: _egr_dec(),
                            )
                    else:
                        _egress(VOCAB_RNDS if LM_HEAD else N_ROUNDS)
                    # id-demux HOST dests (QKV id1, o-proj id4): per-round relay memtile
                    # -> host (strip demux already delivered pure 512). The gate-up dest
                    # (id8) is NOT here -- it goes DIRECTLY to the GLU tile (below).
                    if BATCH > 1:
                        # ---- QKV drain: (round, token) -> (token, round) ----
                        # Gather the ROUNDS_PER_DEST[0] dest-0 rounds into one
                        # [B][M] memtile buffer, landing round r of token t at
                        # t*M + r*PAYLOAD, then emit B whole token rows. rope
                        # then reads one M-wide row per token exactly as it does
                        # at batch 1.
                        #
                        # Guarded by the same arm as the egress: in vocab mode
                        # dest 0 never flows, and a MEMTILE that stalls waiting
                        # for it is precisely the failure that got the original
                        # col-2 QKV relay removed. An idle compute-tile S2MM is
                        # harmless; an idle memtile is not.
                        def _qkv_transpose():
                            qmt = AllocOp(qkvmt_l2, [], [])
                            qmt.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), RELAY_COLS[0])
                            )
                            # ALL the dest-0 rounds in one get, de-interleaved
                            # into a [B][M] buffer: round r of token t lands at
                            # t*M + r*PAYLOAD, so the rope core reads one
                            # contiguous M-wide row per token. One BD, not
                            # ROUNDS_PER_DEST[0] of them.
                            _qo, _qs, _qt = outy_tokmajor(M, rounds=ROUNDS_PER_DEST[0])
                            ChannelGet(
                                "outY",
                                qmt,
                                indices=[idx(0), idx(0)],
                                offsets=[idx(v) for v in _qo],
                                sizes=[idx(v) for v in _qs],
                                strides=[idx(v) for v in _qt],
                            )
                            # ONE flat put of every token's row, not B of
                            # them. The rows are consecutive and M-wide, so B
                            # puts at t*M ARE one contiguous BATCH*M run -- and
                            # a memtile DMA channel holds 24 BD ids SHARED
                            # between its MM2S and S2MM halves. B=16 puts plus
                            # the KV re-block's 8 on the same channel is 24, and
                            # the S2MM side then has nowhere to go: "Allocator
                            # exhausted available BD IDs (maximum 24 available
                            # for channel 1)". Folded it is one BD at any batch.
                            #
                            # The rope core still gets one M-wide row per token;
                            # a single producer BD feeding B consumer firings is
                            # the same shape the shim's banded @rmsX task has
                            # (one 20,480 task, five 4,096 core firings).
                            _pt_all = (
                                [(0, BATCH * M)]
                                if not _os.environ.get("_QKV_ROPE_PERTOKEN")
                                else [(_t * M, M) for _t in range(BATCH)]
                            )
                            for _off, _len in _pt_all:
                                _pt = ChannelPut(
                                    "toRope",
                                    qmt,
                                    offsets=[idx(_off)],
                                    sizes=[idx(_len)],
                                    strides=[idx(1)],
                                )
                                if RELAY_COLS[0] in _ATTN_COLS:
                                    # This memtile is SHARED with that column's
                                    # KV re-block, and an attention column
                                    # reserves MM2S 0: the q-broadcast transits
                                    # this switchbox and a flow on MM2S 0
                                    # deadlocks the route. The KV puts already
                                    # carry this floor; without it here the
                                    # transposer takes MM2S 0 (visible in the
                                    # AIE dump as the memtile gaining MM2S0,
                                    # which batch 1 leaves empty).
                                    #
                                    # Matched on the SET of attention columns,
                                    # not the literal 3: both attention columns
                                    # reserve MM2S 0 for the same reason, and
                                    # QKV_RELAY_COL exists to move this buffer
                                    # between them.
                                    _pt.operation.attributes[
                                        "air.memtile_dma_channel_min"
                                    ] = IntegerAttr.get(T.i32(), 1)
                            DeallocOp(qmt)

                        if _seg_arm_i is not None:

                            def _qt_dec():
                                _qkv_transpose()
                                yield_([])

                            index_switch(
                                [],
                                _seg_arm_i,
                                ARM_IDLE_CASES,
                                case_body_builder=lambda op, i, cv: yield_([]),
                                default_body_builder=lambda op: _qt_dec(),
                            )
                        elif not LM_HEAD:
                            _qkv_transpose()

                    for p in HOST_DRAIN:
                        if p == 0:
                            continue  # QKV (dest0) consumed by the rope herd (below)
                        for _rp in for_(idx(0), idx(ROUNDS_PER_DEST[p]), idx(1)):
                            rb = AllocOp(relay_l2, [], [])
                            rb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), RELAY_COLS[p])
                            )
                            ChannelGet(
                                "outY",
                                rb,
                                indices=[idx(0), idx(p)],
                                offsets=[0],
                                sizes=[PAYLOAD],
                                strides=[1],
                            )
                            ChannelPut(
                                "toShim",
                                rb,
                                indices=[idx(p)],
                                offsets=[0],
                                sizes=[PAYLOAD],
                                strides=[1],
                            )
                            DeallocOp(rb)
                            yield_([])

                    # ===== ATTN S1: rope core (reference tile_2_3) =====
                    # QKV (id1, dest0) -> rope_compute(qkv 3072, lut 64) -> q(2048),
                    # k(512), v(512) roped. S1 drains roped q/k/v -> toShim[0] (the
                    # freed QKV host drain) to verify the QKV->rope dataflow. S3 will
                    # route q->attn and k/v->KV append instead.

                    # the reference-faithful: NO QKV staging memtile. The QKV (id1/dest0) is
                    # assembled directly in the rope COMPUTE core's L1 (see rope_h
                    # below), mirroring layer.mlir mem_2_3 S2MM0 (tile_2_3 gathers the
                    # 3072 qkv_buffer itself). Removing the col-2 memtile relay is the
                    # fix for the fused vocab deadlock: in vocab mode id1 never flows,
                    # and an idle compute-tile S2MM does NOT stall the col-2 memtile
                    # that the vocab X-feed/rms share.

                    # BUG FIX (later43c): rope arm MUST track the mode like proj/rms
                    # (0 in vocab). Hardcoded 1 kept rope in _dec() during vocab -> it
                    # stalled on the 6 id1 QKV gets (never produced in vocab) and never
                    # emitted the appendK/appendV the LM launch waits on -> TIMEOUT.
                    _arm_rope = _seg_arm

                    @herd(name="rope", sizes=[1, 1], operands=[_arm_rope])
                    def rope_h(tx, ty, _sx, _sy, _arm):
                        def _dec_one():
                            """One token. At batch 1 this IS _dec; batched, the
                            caller loops it B times and each iteration consumes
                            one token row from @toRope and one cos/sin LUT from
                            @ropeLUT -- the LUT is per POSITION, and a block of B
                            tokens spans B positions, so the host uploads B of
                            them and this loop consumes them in order."""
                            a_qkv = AllocOp(qkv_l1, [], [])
                            # the reference-faithful (layer.mlir mem_2_3 S2MM0): the rope COMPUTE
                            # core assembles the 6 id1/dest0 demux rounds (512 each)
                            # directly into its own L1 3072 buffer -- NO col-2 memtile
                            # relay. Identical 6x512 offset gets as the old qkvmt (each
                            # get consumes one stripped packet round), just landing in
                            # L1. In vocab mode id1 never flows so this compute-tile
                            # S2MM idles harmlessly.
                            if BATCH > 1:
                                # One whole token row, already transposed in L2.
                                ChannelGet(
                                    "toRope",
                                    a_qkv,
                                    offsets=[idx(0)],
                                    sizes=[idx(M)],
                                    strides=[idx(1)],
                                )
                            else:
                                for _rq in range(ROUNDS_PER_DEST[0]):
                                    ChannelGet(
                                        "outY",
                                        a_qkv,
                                        indices=[idx(0), idx(0)],
                                        offsets=[idx(_rq * PAYLOAD)],
                                        sizes=[idx(PAYLOAD)],
                                        strides=[idx(1)],
                                    )
                            a_lut = AllocOp(ropelut_l1, [], [])
                            ChannelGet("ropeLUT", a_lut, indices=[idx(0)])
                            a_q = AllocOp(ropeq_l1, [], [])
                            a_k = AllocOp(ropekv_l1, [], [])
                            a_v = AllocOp(ropekv_l1, [], [])
                            CallOp(rope_compute, [a_q, a_k, a_v, a_qkv, a_lut, _arm])
                            # S3a: feed flash attention (1 CU = CU0). q[0:512] -> qk
                            # tile directly (MM2S0). k[0:128]+v[0:128] (CU0's 2 KV
                            # heads) -> KV staging memtile on ONE MM2S (rope's 2nd
                            # MM2S, like reference rope k/v packets) which splits them.
                            # q reorder = pack_q (reference mem_5_1 [8,8,8]/[8,64,1]):
                            # natural [qh,dh] -> [dc,qh,de], the kernel's q layout.
                            # q (whole 2048) -> q broadcast memtile (1 rope MM2S);
                            # the memtile fans out per-CU reordered (reference mem_5_1).
                            ChannelPut(
                                "ropeQ",
                                a_q,
                                indices=[idx(0)],
                                offsets=[idx(0)],
                                sizes=[idx(DQ_PADDED)],
                                strides=[idx(1)],
                            )
                            if MULTIBLK:
                                # the reference append: this token's roped K (all heads) and
                                # raw V -> appendK/appendV -> KVC at APPEND_OFF. The
                                # whole cache is then read back for the block loop.
                                if KV_APPEND:
                                    ChannelPut(
                                        "appendK",
                                        a_k,
                                        indices=[idx(0)],
                                        offsets=[idx(0)],
                                        sizes=[idx(DK_TOT_A)],
                                        strides=[idx(1)],
                                    )
                                    ChannelPut(
                                        "appendV",
                                        a_v,
                                        indices=[idx(0)],
                                        offsets=[idx(0)],
                                        sizes=[idx(DK_TOT_A)],
                                        strides=[idx(1)],
                                    )
                            else:
                                # per COLUMN GROUP: that group's CUs' k then v on its
                                # own packet channel (no cross-col FIFO interleave).
                                for gi, (_col, cus) in enumerate(ATTN_COL_GROUPS):
                                    for c in cus:
                                        ChannelPut(
                                            "toAttnKV",
                                            a_k,
                                            indices=[idx(gi)],
                                            offsets=[idx(c * KVPC_DH)],
                                            sizes=[idx(KVPC_DH)],
                                            strides=[idx(1)],
                                        )
                                    for c in cus:
                                        ChannelPut(
                                            "toAttnKV",
                                            a_v,
                                            indices=[idx(gi)],
                                            offsets=[idx(c * KVPC_DH)],
                                            sizes=[idx(KVPC_DH)],
                                            strides=[idx(1)],
                                        )
                            DeallocOp(a_qkv)
                            DeallocOp(a_lut)
                            DeallocOp(a_q)
                            DeallocOp(a_k)
                            DeallocOp(a_v)

                        def _dec():
                            if BATCH == 1:
                                _dec_one()
                            else:
                                # B tokens, sequentially. rope, the Q broadcast,
                                # the KV append and the attention behind them all
                                # stay per-token; only the PROJECTION is batched,
                                # which is where every byte of the weight-traffic
                                # win is.
                                #
                                # An scf.for, not an unroll. The body is five
                                # channel ops (qkv in, LUT in, q out, K out,
                                # V out) and a compute tile has SIXTEEN BDs, so
                                # eight copies do not fit. Nothing in the body
                                # depends on the token index -- each channel
                                # just delivers the next token's data, including
                                # the per-POSITION rope LUT.
                                for _t in for_(idx(0), idx(BATCH), idx(1)):
                                    _dec_one()
                                    yield_([])
                            yield_([])

                        def _voc():
                            # gate-off 2026-07-15b: attn fully idle in vocab -> rope emits
                            # NOTHING (no dummy appendK/appendV; _uni_voc drains neither).
                            yield_([])

                        index_switch(
                            [],
                            arith.index_cast(idx_t, _arm),
                            ARM_IDLE_CASES,
                            case_body_builder=lambda op, i, cv: _voc(),
                            default_body_builder=lambda op: _dec(),
                        )

                    rope_h.attributes["link_with"] = StringAttr.get("rope.o")
                    rope_h.attributes["x_loc"] = IntegerAttr.get(T.i64(), RMS_PCOL)
                    rope_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), 3)

                    # ===== ATTN S3a: 1-CU flash attention (reference tile_3_2/3_3) =====
                    # Proven attn_iso qk/kv herd pair: s_shared (segment-scope L1) is
                    # shared cross-tile (qk writes scores, kv reads). q from rope (direct
                    # to qk), k/v from rope via KV staging memtile (split). L=1 decode =>
                    # 1 block; the 15 pad keys are masked by L inside the kernels. o ->
                    # attnO host drain (S3a verification; S4 routes o -> o-proj X).
                    # q broadcast memtile (reference mem_5_1): get rope q (2048),
                    # fan out per-CU 512 reordered (pack_q [8,8,8]/[8,64,1]).
                    def _qmtb_dec():
                        """Take EVERY token's q before fanning ANY of them out.

                        Not a buffering nicety -- the interleaved form deadlocks,
                        and the cycle is worth writing down because nothing in
                        the element counts shows it:

                          rope must finish all B tokens before the KV readback
                          can start (the append barrier waits for all B appends)
                            -> the attention CUs block on toK until the readback
                            -> the q memtile blocks fanning token 1 to the CUs
                            -> rope blocks putting token 1's q to the memtile
                            -> rope never emits appends 1..B-1.

                        Draining rope first cuts it: the memtile takes all B q
                        rows into its own buffer, rope runs to completion, the
                        appends land, the readback starts, and only then does the
                        fan-out block on the CUs -- by which time nothing is
                        waiting on it. At batch 1 the cycle cannot form and this
                        is the same single get and fan it always was.

                        NO LOOPS HERE, and that is the second half of the fix.
                        Written as `for t: get slice` then `for t: fan slice`,
                        air-ping-pong-transform recognises the two loops as a
                        producer/consumer pair over slices of one buffer and
                        rebuilds them as a TWO-DEEP RING of DQ_PADDED slices --
                        which is the interleaved form again, and it deadlocked
                        after 4 of 8 tokens. One get and one put per CU, with the
                        batch as a BD dimension, gives the transform nothing to
                        rewrite. It is the same shape the o-gather memtile uses
                        for the return trip, which survives untouched.
                        """
                        qmtb = AllocOp(qmt_l2, [], [])
                        qmtb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(),
                            5,  # reference mem_5_1; free for N<=2 (kv on col3).
                            # N=4 needs attn cols 3,4 + GLU->tile_5_2 relayout (TODO).
                        )
                        if BATCH == 1:
                            ChannelGet("ropeQ", qmtb, indices=[idx(0)])
                        else:
                            # ONE get for rope's B puts: the packet flow
                            # reassembles them, exactly as the o-gather's four
                            # channels each reassemble a CU's B outputs.
                            ChannelGet(
                                "ropeQ",
                                qmtb,
                                indices=[idx(0)],
                                offsets=[idx(0)],
                                sizes=[idx(BATCH * DQ_PADDED)],
                                strides=[idx(1)],
                            )
                        _qmtb_fan(qmtb)
                        _probe_put("Q", "probe5", qmtb)
                        DeallocOp(qmtb)

                    def _qmtb_fan(qmtb):
                        # At BATCH>1 the token becomes the OUTER dimension of the
                        # same pack_q reorder -- 4 dims, which is exactly what a
                        # memtile BD has. The CU still gets one q per token.
                        _bt = [idx(BATCH)] if BATCH > 1 else []
                        _bs = [idx(DQ_PADDED)] if BATCH > 1 else []
                        _bo = [idx(0)] if BATCH > 1 else []
                        for c in range(N_ATTN_CU):
                            ChannelPut(
                                "toAttnQ",
                                qmtb,
                                # pack_q (reference mem_5_1): natural [qh, dh] -> the
                                # kernel's [dc, qh, de] mmul layout, dh = dc*8 + de.
                                # CU c reads its Q_HEADS_PER_CU heads starting at head
                                # c*Q_HEADS_PER_CU (stride DH) -> linear base
                                # c*Q_HEADS_PER_CU*DH = c*DQ_PER_CU. dc stride 8, de 1.
                                indices=[idx(c)],
                                # rope emits PADDED Q (each CU's block is
                                # Q_HEADS_PADDED_PER_CU heads incl ATTN_GROUPS_PADDING
                                # zeros); CU c's block starts at c*Q_HEADS_PADDED_PER_CU.
                                # llama pad=0 -> ==Q_HEADS_PER_CU (byte-identical).
                                offsets=_bo
                                + [
                                    idx(0),
                                    idx(c * Q_HEADS_PADDED_PER_CU),
                                    idx(0),
                                ],
                                sizes=_bt
                                + [
                                    idx(DH // 8),
                                    idx(Q_HEADS_PADDED_PER_CU),
                                    idx(8),
                                ],
                                strides=_bs + [idx(8), idx(DH), idx(1)],
                            )

                    # gate-off 2026-07-15b: q-broadcast is decode-only (vocab attn idle).
                    if _seg_arm_i is not None:

                        def _q_voc():
                            yield_([])

                        def _q_dec():
                            _qmtb_dec()
                            yield_([])

                        index_switch(
                            [],
                            _seg_arm_i,
                            ARM_IDLE_CASES,
                            case_body_builder=lambda op, i, cv: _q_voc(),
                            default_body_builder=lambda op: _q_dec(),
                        )
                    else:
                        _qmtb_dec()
                    # ===== N_ATTN_CU flash-attention CUs (reference 4-CU) =====
                    # KV block cache memtile(s): per-CU SEPARATE K/V natural buffers
                    # [key16,kvh2,dh64]; rope's token-0 K/V -> [0:128]; keys 1..15 pad
                    # (masked by L=1). Reorders == attn_stream toK/toV (PROVEN):
                    # nat -> pack_k/pack_v. The memtile gets rope's per-CU k then v
                    # (FIFO: k0..k{N-1}, v0..v{N-1}) and fans out reordered to each CU.
                    # L=1 (single-block) KV staging: rope's this-token k/v via
                    # toAttnKV -> akbs/avbs memtiles -> 1 toK/toV block per CU.
                    # MULTIBLK uses the DDR-cache l2_kv re-block in _make_cu instead
                    # (these memtiles would collide with l2_kv on cols 3/4).
                    if not MULTIBLK:
                        akbs, avbs = [], []
                        for c in range(N_ATTN_CU):
                            col = ATTN_CU_LOC[c][0]
                            akb = AllocOp(ak_l2, [], [])
                            akb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), col)
                            )
                            akbs.append(akb)
                        for c in range(N_ATTN_CU):
                            col = ATTN_CU_LOC[c][0]
                            avb = AllocOp(av_l2, [], [])
                            avb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), col)
                            )
                            avbs.append(avb)
                        # per col group: get its CUs' k then v from toAttnKV[gi]
                        # (matches rope's per-group put order; no cross-col FIFO).
                        for gi, (_col, cus) in enumerate(ATTN_COL_GROUPS):
                            for c in cus:
                                ChannelGet(
                                    "toAttnKV",
                                    akbs[c],
                                    indices=[idx(gi)],
                                    offsets=[idx(0)],
                                    sizes=[idx(KVPC_DH)],
                                    strides=[idx(1)],
                                )
                            for c in cus:
                                ChannelGet(
                                    "toAttnKV",
                                    avbs[c],
                                    indices=[idx(gi)],
                                    offsets=[idx(0)],
                                    sizes=[idx(KVPC_DH)],
                                    strides=[idx(1)],
                                )
                        for c in range(N_ATTN_CU):
                            _pk = ChannelPut(
                                "toK",
                                akbs[c],
                                indices=[idx(c)],
                                offsets=[idx(0), idx(0), idx(0)],
                                sizes=[idx(KVPC_DH // 8), idx(16), idx(8)],
                                strides=[idx(8), idx(KVPC_DH), idx(1)],
                            )
                            _pv = ChannelPut(
                                "toV",
                                avbs[c],
                                indices=[idx(c)],
                                offsets=[idx(0), idx(0), idx(0), idx(0)],
                                sizes=[idx(2), idx(KVPC_DH // 8), idx(8), idx(8)],
                                strides=[
                                    idx(8 * KVPC_DH),
                                    idx(8),
                                    idx(KVPC_DH),
                                    idx(1),
                                ],
                            )
                            # col-3 KV: reserve memtile MM2S 0 (the q-broadcast
                            # transits this memtile's switchbox; KV on MM2S 0
                            # deadlocks the route). col 4 already has GLU/down on
                            # MM2S 0, so its KV naturally lands on 1-4. Gate on
                            # LOOPCLOSE to keep GREEN's layout/PASS unchanged.
                            if ATTN_CU_LOC[c][0] == 3:
                                _pk.operation.attributes[
                                    "air.memtile_dma_channel_min"
                                ] = IntegerAttr.get(T.i32(), 1)
                                _pv.operation.attributes[
                                    "air.memtile_dma_channel_min"
                                ] = IntegerAttr.get(T.i32(), 1)
                        for c in range(N_ATTN_CU):
                            DeallocOp(akbs[c])
                            DeallocOp(avbs[c])

                    def _make_cu(c):
                        col, qk_row, kv_row = ATTN_CU_LOC[c]
                        a_sh = AllocOp(as_l1, [], [])

                        if MULTIBLK:
                            # ===== reproducer model A: online-softmax block loop
                            # over ATTN_ROUNDS=(L+15)/16 KV blocks. Per-CU state
                            # m/c (qk) and y/l (kv) persists across blocks (reset
                            # on blk==0 in-kernel); attn_kv_fin normalizes after
                            # the last block. Lh = RTP_L herd operand (kernel masks
                            # the last partial block). Compute proven in attn_iso.
                            L_c = (
                                _seg_L
                                if DYNSEQ_RTP
                                else arith.ConstantOp(
                                    IntegerAttr.get(i32, ATTN_L), None
                                ).result
                            )

                            # per-block KV staging ring (attn_iso PASS): fresh kvb
                            # per block -> count-free ping-pong ring (1 fill : 1
                            # read). Each block: get this block's [K|V] from the
                            # readback (inKV) then re-block to toK/toV. Strides
                            # mirror attn_iso exactly.
                            # gate-off 2026-07-15b: KV re-block (inKV get + toK/toV put)
                            # is DECODE-ONLY. In vocab the attn cores are idle (empty
                            # index_switch case) so they neither need toK/toV nor consume
                            # the 4-slot count-free KV memtile ring (mem_tile_3_1/4_1) --
                            # that ring drain was the 16dec+3voc-then-hang bug.
                            def _reblock_dec():
                                # B tokens, B whole passes over the cache. This
                                # is where the batch does NOT amortize, and it
                                # is the measured, accepted cost (section 5e):
                                # every token re-reads the same KV. Hoisting it
                                # means feeding the CU a QUERY TILE and looping
                                # the blocks once, which is worth 1.45x on
                                # attention and needs attn_qk/attn_kv to carry
                                # per-query m/c/y/l -- a kernel change, not a
                                # wiring one.
                                _reblock_one()

                            def _reblock_one():
                                if KV_SPLIT:
                                    # the reference mem_3_1: per col GROUP, separate K/V buffers each
                                    # with its own count-free ring (independent S2MM fill from
                                    # inKV_K / inKV_V). Emit ONCE per group (on the lead CU);
                                    # the lead produces toK/toV for every CU in the group. This
                                    # removes the shared-buffer backward edge (qk-K no longer
                                    # lock-chained to kv-V drain).
                                    _gi = ATTN_CU_GROUP[c]
                                    _gcol, _cus = ATTN_COL_GROUPS[_gi]
                                    if c != _cus[0]:
                                        return
                                    _gw = len(_cus) * KVPC_DH
                                    for _blk in for_(idx(0), _seg_blocks(), idx(1)):
                                        _kbuf = AllocOp(kvblk_l2, [], [])
                                        _kbuf.operation.attributes[
                                            "air.memtile_col"
                                        ] = IntegerAttr.get(T.i32(), col)
                                        _vbuf = AllocOp(kvblk_l2, [], [])
                                        _vbuf.operation.attributes[
                                            "air.memtile_col"
                                        ] = IntegerAttr.get(T.i32(), col)
                                        ChannelGet("inKV_K", _kbuf, indices=[idx(_gi)])
                                        ChannelGet("inKV_V", _vbuf, indices=[idx(_gi)])
                                        for _lc, _cc in enumerate(_cus):
                                            _pk = ChannelPut(
                                                "toK",
                                                _kbuf,
                                                indices=[idx(_cc)],
                                                offsets=[
                                                    idx(0),
                                                    idx(0),
                                                    idx(_lc * KVPC_DH),
                                                ],
                                                sizes=[
                                                    idx(KVPC_DH // 8),
                                                    idx(16),
                                                    idx(8),
                                                ],
                                                strides=[idx(8), idx(_gw), idx(1)],
                                            )
                                            _pv = ChannelPut(
                                                "toV",
                                                _vbuf,
                                                indices=[idx(_cc)],
                                                offsets=[
                                                    idx(0),
                                                    idx(0),
                                                    idx(0),
                                                    idx(_lc * KVPC_DH),
                                                ],
                                                sizes=[
                                                    idx(2),
                                                    idx(KVPC_DH // 8),
                                                    idx(8),
                                                    idx(8),
                                                ],
                                                strides=[
                                                    idx(_gw * 8),
                                                    idx(8),
                                                    idx(_gw),
                                                    idx(1),
                                                ],
                                            )
                                            if col == 3:
                                                _pk.operation.attributes[
                                                    "air.memtile_dma_channel_min"
                                                ] = IntegerAttr.get(T.i32(), 1)
                                                _pv.operation.attributes[
                                                    "air.memtile_dma_channel_min"
                                                ] = IntegerAttr.get(T.i32(), 1)
                                        DeallocOp(_kbuf)
                                        DeallocOp(_vbuf)
                                        yield_([])
                                    return
                                # ROLLED (was Python for blk in range(ATTN_ROUNDS)): AIR for_
                                # -> count-free 2-buffer ring on the memtile (mirror the
                                # weight-fan) so large ATTN_L stays under the 16-BD limit.
                                # Fresh kvb per iter (no_split, memtile_col) = the share-ring
                                # pattern AIR lowers to next_bd rotation, not a repeat_count BD.
                                for _blk in for_(idx(0), _seg_blocks(), idx(1)):
                                    kvb = AllocOp(kvblk_l2, [], [])
                                    kvb.operation.attributes["air.memtile_col"] = (
                                        IntegerAttr.get(T.i32(), col)
                                    )
                                    ChannelGet("inKV", kvb, indices=[idx(c)])
                                    _pk = ChannelPut(
                                        "toK",
                                        kvb,
                                        indices=[idx(c)],
                                        offsets=[idx(0), idx(0), idx(0)],
                                        sizes=[idx(KVPC_DH // 8), idx(16), idx(8)],
                                        strides=[idx(8), idx(KVPC_DH), idx(1)],
                                    )
                                    _pv = ChannelPut(
                                        "toV",
                                        kvb,
                                        indices=[idx(c)],
                                        offsets=[idx(2), idx(0), idx(0), idx(0)],
                                        sizes=[
                                            idx(2),
                                            idx(KVPC_DH // 8),
                                            idx(8),
                                            idx(8),
                                        ],
                                        strides=[
                                            idx(8 * KVPC_DH),
                                            idx(8),
                                            idx(KVPC_DH),
                                            idx(1),
                                        ],
                                    )
                                    # MULTIBLK KV re-block: same col-3 switchbox collision
                                    # as the on-chip path (KV on memtile MM2S 0 deadlocks
                                    # in LOOPCLOSE - the o-proj feedback + q-broadcast
                                    # transit col-3's switchbox). Reserve MM2S 0 by steering
                                    # col-3 KV onto channels 1-4. (col 4 already lands on
                                    # 1-4 via GLU/down; gate on LOOPCLOSE to keep GREEN.)
                                    if col == 3:
                                        _pk.operation.attributes[
                                            "air.memtile_dma_channel_min"
                                        ] = IntegerAttr.get(T.i32(), 1)
                                        _pv.operation.attributes[
                                            "air.memtile_dma_channel_min"
                                        ] = IntegerAttr.get(T.i32(), 1)
                                    DeallocOp(kvb)
                                    yield_([])

                            _gated = _seg_arm_i is not None
                            if _gated:

                                def _rb_voc():
                                    yield_([])

                                def _rb_dec():
                                    _reblock_dec()
                                    yield_([])

                                index_switch(
                                    [],
                                    _seg_arm_i,
                                    ARM_IDLE_CASES,
                                    case_body_builder=lambda op, i, cv: _rb_voc(),
                                    default_body_builder=lambda op: _rb_dec(),
                                )
                            else:
                                _reblock_dec()

                            def _core_rounds(Lh):
                                """ceil(L_blk/16) as a core-side loop bound.

                                Lh is the RTP-L herd block-arg, so this is opaque to
                                folding and survives to core codegen as a real runtime
                                trip count -- the same count the shim's readback BD
                                pushes, which is what keeps the core off a channel get
                                that never arrives.

                                UNIFORM across the batch, and deliberately: the
                                per-token L varies but the push count cannot,
                                so every token loops the LAST token's block
                                count and the kernels drop the blocks past
                                their own L. The get still runs on those
                                blocks -- the stream has to drain either way.
                                """
                                if not DYNSEQ_RTP:
                                    return idx(ATTN_ROUNDS)
                                # base + BATCH - 1 is the longest token's L in
                                # BOTH modes, so only the mode bit comes off.
                                _base, _ = _split_mask_mode(Lh)
                                _s = arith.addi(
                                    _base,
                                    arith.ConstantOp(
                                        IntegerAttr.get(i32, 15 + BATCH - 1), None
                                    ).result,
                                )
                                _q = arith.divui(
                                    _s,
                                    arith.ConstantOp(
                                        IntegerAttr.get(i32, 16), None
                                    ).result,
                                )
                                return arith.index_cast(idx_t, _q)

                            def _tok_L(Lh, t_iv):
                                """This token's context length: Lh + t*step.

                                A block of B tokens occupies B CONSECUTIVE cache
                                positions, so token t attends to t more keys
                                than token 0. Still ONE RTP per dispatch -- the
                                per-token part is the loop's own induction
                                variable, not a second runtime value.

                                Under MASK_MODE_RTP the step is 0 or 1 decoded
                                from the same scalar: step 1 is that causal
                                staircase (DFlash verify), step 0 gives every
                                token the same L (DFlash draft, where the block
                                attends to itself bidirectionally). See
                                batch_attn_mask.py for why a per-query mask is
                                exactly a per-query L.
                                """
                                _base, _step = _split_mask_mode(Lh)
                                _t = arith.index_cast(i32, t_iv)
                                if isinstance(_step, int):
                                    if _step == 0:
                                        # Baked bidirectional: every token of
                                        # the block sees base + (B-1) -- the
                                        # whole block, DFlash's draft pass.
                                        return arith.addi(
                                            _base,
                                            arith.ConstantOp(
                                                IntegerAttr.get(i32, BATCH - 1), None
                                            ).result,
                                        )
                                    return arith.addi(_base, _t)
                                # base + (B-1) - step*(B-1-t):
                                #   step 1 -> base + t        the causal staircase
                                #   step 0 -> base + (B-1)    the whole block
                                _bm1 = arith.ConstantOp(
                                    IntegerAttr.get(i32, BATCH - 1), None
                                ).result
                                return arith.subi(
                                    arith.addi(_base, _bm1),
                                    arith.muli(_step, arith.subi(_bm1, _t)),
                                )

                            def _tok_loop(one, sh, Lh, _c):
                                """Run `one` for each token in the block.

                                An scf.for, NOT an unroll, and that is a BD
                                budget decision: a compute tile has 16 BDs and
                                one pass already uses two channels, so eight
                                copies would not fit. The loop body is
                                identical per token -- only Lh moves, and it
                                moves by the induction variable.
                                """
                                if BATCH == 1:
                                    one(sh, Lh, Lh, _c)
                                    return
                                for _t in for_(idx(0), idx(BATCH), idx(1)):
                                    one(sh, _tok_L(Lh, _t), Lh, _c)
                                    yield_([])

                            def _qk_body(sh, Lh, _c):
                                if BATCH == 1:
                                    _tok_loop(_qk_body_one, sh, Lh, _c)
                                    return
                                # ONE get of the whole block's q, before the
                                # token loop -- the same shape the q memtile uses
                                # one hop up, and for the same reason.
                                #
                                # The memtile fans q to the four CUs as a DAISY
                                # CHAIN: CU c+1's transfer only starts when CU c's
                                # has finished. A per-token get makes CU 0's link
                                # 8 tokens long and gates it on CU 0 running the
                                # whole block -- but CU 0 cannot, because the KV
                                # re-block memtile fans each block to BOTH CUs of
                                # the column in lockstep and CU 1 is still waiting
                                # for the q it will not get until CU 0 finishes.
                                # Taking all B rows at once breaks the chain: the
                                # transfer completes on arrival, with no core in
                                # the loop.
                                a_q = AllocOp(aq_l1, [], [])
                                ChannelGet("toAttnQ", a_q, indices=[idx(_c)])
                                for _t in for_(idx(0), idx(BATCH), idx(1)):
                                    _qk_body_one(sh, _tok_L(Lh, _t), Lh, _c, a_q, _t)
                                    yield_([])
                                DeallocOp(a_q)

                            def _kv_body(sh, Lh, _c):
                                if BATCH == 1:
                                    _kv_body_one(sh, Lh, Lh, _c)
                                    return
                                # The o buffer OUTLIVES the token loop: every
                                # token writes its row and the whole block
                                # leaves in one transfer afterwards. See
                                # attn_kv_fin_row -- a transfer per token
                                # deadlocks against the o-gather's daisy chain.
                                a_o = AllocOp(ao_l1, [], [])
                                for _t in for_(idx(0), idx(BATCH), idx(1)):
                                    _kv_body_one(sh, _tok_L(Lh, _t), Lh, _c, a_o, _t)
                                    yield_([])
                                if _os.environ.get("_ATTN_O_PERTOKEN"):
                                    # A/B: the per-token form, which is one BD
                                    # block per token and does not fit at 16.
                                    for _to in range(BATCH):
                                        _attn_o_put(a_o, _c, _to)
                                else:
                                    _attn_o_put_block(a_o, _c)
                                DeallocOp(a_o)

                            def _qk_body_one(sh, Lh, Lblk, _c, a_q=None, t_iv=None):
                                _own_q = a_q is None
                                if _own_q:
                                    a_q = AllocOp(aq_l1, [], [])
                                    ChannelGet("toAttnQ", a_q, indices=[idx(_c)])
                                a_m = AllocOp(m_l1, [], [])
                                a_cc = AllocOp(c_l1, [], [])
                                # RUNTIME-L block count = ceil(Lh/16) from the RTP-L herd
                                # block-arg (opaque region arg -> not const-folded -> stays a
                                # runtime scf.for bound; the AIE core loops per the RTP-L the
                                # shim writes, exactly like the reference's in-core rounds=(L+15)/16).
                                # unrollSCFFors only unrolls all-constant loops, so this
                                # survives to core codegen as a real runtime loop.
                                _nblk_qk = _core_rounds(Lblk)
                                for _blk in for_(idx(0), _nblk_qk, idx(1)):
                                    # REQUIRED single-buffer: ping-pong would unroll-by-2 +
                                    # 1-remainder over a 3-buffer toK ring whose remainder reads
                                    # the wrong buffer vs the DMA rotation -> misaligned KV ->
                                    # garbage chat. Single-buffer is aligned.
                                    a_k = AllocOp(ak_l1, [], [])
                                    ChannelGet("toK", a_k, indices=[idx(_c)])
                                    blk_c = arith.index_cast(i32, _blk)
                                    if _own_q:
                                        CallOp(
                                            attn_qk_blk,
                                            [a_q, a_k, a_m, a_cc, sh, blk_c, Lh],
                                        )
                                    else:
                                        CallOp(
                                            attn_qk_blk_row,
                                            [
                                                a_q,
                                                a_k,
                                                a_m,
                                                a_cc,
                                                sh,
                                                blk_c,
                                                Lh,
                                                arith.index_cast(i32, t_iv),
                                            ],
                                        )
                                    DeallocOp(a_k)
                                    yield_([])
                                if _own_q:
                                    DeallocOp(a_q)
                                DeallocOp(a_m)
                                DeallocOp(a_cc)

                            def _kv_body_one(sh, Lh, Lblk, _c, a_o=None, t_iv=None):
                                a_y = AllocOp(y_l1, [], [])
                                a_l = AllocOp(lden_l1, [], [])
                                _own_o = a_o is None
                                if _own_o:
                                    a_o = AllocOp(ao_l1, [], [])
                                # RUNTIME-L block count = ceil(Lh/16) (see _qk_body). Core
                                # loops per RTP-L; matched by the shim readback push count.
                                _nblk_kv = _core_rounds(Lblk)
                                for _blk in for_(idx(0), _nblk_kv, idx(1)):
                                    # REQUIRED single-buffer (see _qk_body): keeps toV/toK
                                    # consumption aligned with the DMA rotation (no unroll-by-2
                                    # remainder desync -> no misaligned KV).
                                    a_v = AllocOp(av_l1, [], [])
                                    ChannelGet("toV", a_v, indices=[idx(_c)])
                                    blk_c = arith.index_cast(i32, _blk)
                                    CallOp(
                                        attn_kv_blk,
                                        [sh, a_v, a_y, a_l, blk_c, Lh],
                                    )
                                    DeallocOp(a_v)
                                    yield_([])
                                if _own_o:
                                    CallOp(attn_kv_fin, [a_y, a_l, a_o])
                                    _attn_o_put(a_o, _c, None)
                                    DeallocOp(a_o)
                                else:
                                    CallOp(
                                        attn_kv_fin_row,
                                        [a_y, a_l, a_o, arith.index_cast(i32, t_iv)],
                                    )
                                DeallocOp(a_y)
                                DeallocOp(a_l)

                            def _attn_o_put_block(a_o, _c):
                                """Every token's o, in ONE flat transfer.

                                ONE BD BLOCK, INDEPENDENT OF BATCH, and that is
                                what makes batch 16 build. The per-token form
                                below spends a BD block per token -- a compute
                                tile holds 16, the attention core already needs
                                several for @toAttnQ/@toK/@toV, and BATCH=16
                                puts it over ('aie.mem' op has more than 16
                                blocks, on air.herd @attn_blk). A single flat op
                                is one block at any batch.

                                It is flat because the UN-INTERLEAVE MOVED TO THE
                                MEMTILE. The transpose from the kernel's
                                [q_head, dc, de] to natural (q_head, dh) needs
                                three descriptor dimensions, which is all a
                                compute tile has, so the token could not be a
                                fourth and had to ride the offset -- one op per
                                token. A MEMTILE BD has four, so the same
                                permutation plus the token fits there in one op;
                                see _o_gather. Nothing about the data or the
                                order changes, only which end of the channel
                                describes it.
                                """
                                ChannelPut(
                                    "attnO",
                                    a_o,
                                    indices=[idx(_c)],
                                    offsets=[idx(0)],
                                    sizes=[idx(BATCH * DQ_PADDED_PER_CU)],
                                    strides=[idx(1)],
                                )

                            def _attn_o_put(a_o, _c, t):
                                # o un-interleave: kernel [q_head, dc, de] -> natural
                                # (q_head, dh). sizes=[Q_HEADS_PER_CU, DH//8, 8].
                                # Three dimensions, which is all a compute tile
                                # has -- so the token cannot be a fourth, and it
                                # rides the stride-1 offset instead. One put per
                                # token, but issued AFTER the token loop: what
                                # deadlocks the o-gather is interleaving the
                                # CUs, not the number of transfers.
                                #
                                # BATCH == 1 ONLY now -- see _attn_o_put_block.
                                ChannelPut(
                                    "attnO",
                                    a_o,
                                    indices=[idx(_c)],
                                    # Built in place, not hoisted: the constants
                                    # have to be CREATED in this order or the
                                    # batch-1 diff sees different SSA names for
                                    # the same design.
                                    offsets=[
                                        idx(0),
                                        idx(0),
                                        idx(0 if t is None else t * DQ_PADDED_PER_CU),
                                    ],
                                    sizes=[
                                        idx(Q_HEADS_PER_CU),
                                        idx(DH // 8),
                                        idx(8),
                                    ],
                                    strides=[idx(8), idx(Q_HEADS_PER_CU * 8), idx(1)],
                                )

                            # Segment-level per-CU setup done (a_sh scores buffer, L_c,
                            # the memtile KV reblock, and the qk/kv body closures). The
                            # herd is NOT emitted here -- all 8 attn cores are fused into
                            # ONE [2,4] block herd after the loop (see below).
                            return (a_sh, col, qk_row, L_c, _qk_body, _kv_body)

                    _cus = [_make_cu(c) for c in range(N_ATTN_CU)]
                    # Fuse ALL 8 attn cores into ONE [2,4] block over the contiguous
                    # cols 3,4. tx=0 -> col3 (cu0 rows2,3; cu1 rows4,5), tx=1 -> col4
                    # (cu2 rows2,3; cu3 rows4,5). Column by tx==0, pair by ty<2, role
                    # (qk=even/kv=odd) by exact ty==const -- every selector is a direct
                    # tile-IV guard so it folds per-tile at clone. Each CU's score buffer
                    # is shared only across its 2 vertically-adjacent cores (qk writes via
                    # attn_qk_blk, kv reads via attn_kv_blk; Gate-1 strict-subset infers
                    # the cross-core RAW from the opaque calls). Per-core link files derive
                    # from the kernel func each core calls (attn_qk.ll on qk rows,
                    # attn_kv.ll on kv rows) -- no herd link_with. gate-off index_switch on
                    # the per-wave arm keeps vocab idle. This is the attn floor: cols 3,4
                    # are one contiguous block, so 8 -> 1.
                    _sh = [t[0] for t in _cus]
                    _Lc = _cus[0][3]
                    _qkb = _cus[0][4]
                    _kvb = _cus[0][5]

                    def _attn_leaf(ty_arg, cu, sh, Lh, qk_ty):
                        _isqk = arith.cmpi(arith.CmpIPredicate.eq, ty_arg, idx(qk_ty))
                        _if = IfOp(_isqk, [], has_else=True)
                        with InsertionPoint(_if.thenRegion.blocks[0]):
                            _qkb(sh, Lh, cu)
                            yield_([])
                        with InsertionPoint(_if.elseRegion.blocks[0]):
                            _kvb(sh, Lh, cu)
                            yield_([])

                    def _attn_pairsel(ty_arg, shs, Lh, cu_lo, cu_hi):
                        _lo = arith.cmpi(arith.CmpIPredicate.slt, ty_arg, idx(2))
                        _ifp = IfOp(_lo, [], has_else=True)
                        with InsertionPoint(_ifp.thenRegion.blocks[0]):
                            _attn_leaf(ty_arg, cu_lo, shs[cu_lo], Lh, 0)
                            yield_([])
                        with InsertionPoint(_ifp.elseRegion.blocks[0]):
                            _attn_leaf(ty_arg, cu_hi, shs[cu_hi], Lh, 2)
                            yield_([])

                    def _attn_dec(tx_arg, ty_arg, shs, Lh):
                        _isc0 = arith.cmpi(arith.CmpIPredicate.eq, tx_arg, idx(0))
                        _ifc = IfOp(_isc0, [], has_else=True)
                        with InsertionPoint(_ifc.thenRegion.blocks[0]):
                            _attn_pairsel(ty_arg, shs, Lh, 0, 1)  # col3: cu0, cu1
                            yield_([])
                        with InsertionPoint(_ifc.elseRegion.blocks[0]):
                            _attn_pairsel(ty_arg, shs, Lh, 2, 3)  # col4: cu2, cu3
                            yield_([])

                    if _seg_arm_i is not None:

                        @herd(
                            name="attn_blk",
                            sizes=[2, 4],
                            operands=[
                                _sh[0].result,
                                _sh[1].result,
                                _sh[2].result,
                                _sh[3].result,
                                _Lc,
                                _seg_arm,
                            ],
                        )
                        def attn_blk(_tx, _ty, _sx, _sy, s0, s1, s2, s3, Lh, _arm):
                            shs = [s0, s1, s2, s3]

                            def _voc():
                                yield_([])

                            def _dec():
                                _attn_dec(_tx, _ty, shs, Lh)
                                yield_([])

                            index_switch(
                                [],
                                arith.index_cast(idx_t, _arm),
                                ARM_IDLE_CASES,
                                case_body_builder=lambda op, i, cv: _voc(),
                                default_body_builder=lambda op: _dec(),
                            )

                    else:

                        @herd(
                            name="attn_blk",
                            sizes=[2, 4],
                            operands=[
                                _sh[0].result,
                                _sh[1].result,
                                _sh[2].result,
                                _sh[3].result,
                                _Lc,
                            ],
                        )
                        def attn_blk(_tx, _ty, _sx, _sy, s0, s1, s2, s3, Lh):
                            _attn_dec(_tx, _ty, [s0, s1, s2, s3], Lh)

                    attn_blk.attributes["x_loc"] = IntegerAttr.get(
                        T.i64(), ATTN_CU_LOC[0][0]
                    )
                    attn_blk.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

                    # o gather memtile (reference mem_5_1 o_buffer): gather the 4
                    # CUs' o (512 each, already natural [qh,dh] from the egress
                    # reorder) into 2048, then ONE egress -> host (oGathered). This
                    # is the reference o_buffer; the loop-close step routes it to
                    # mem_1_1 (id2) = o-proj X instead of host.
                    def _omtb_dec():
                        omtb = AllocOp(omt_l2, [], [])
                        omtb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(), 5
                        )

                        # loop close: gathered o (2048) is ph1 o-proj X, re-broadcast
                        # OPROJ_REFEED times into the convergent @xnorm, AFTER ph0 (rms)
                        # and BEFORE ph2. Reference mem_5_1 o_buffer -> mem_1_1 x_buffer.
                        # Token-major: the CUs run their token loop in order, so
                        # this is the order the four attnO channels deliver in.
                        # The token dimension is a loop and the CU fan is not:
                        # the four CUs are four DIFFERENT channels (distinct BDs
                        # either way), the tokens are the same four.
                        def _o_gather(_toff):
                            for c in range(N_ATTN_CU):
                                ChannelGet(
                                    "attnO",
                                    omtb,
                                    indices=[idx(c)],
                                    offsets=[
                                        (
                                            idx(c * DQ_PER_CU)
                                            if _toff is None
                                            else arith.addi(_toff, idx(c * DQ_PER_CU))
                                        )
                                    ],
                                    sizes=[idx(DQ_PER_CU)],
                                    strides=[idx(1)],
                                )

                        def _o_gather_block():
                            """All four CUs' o for the whole batch, four gets.

                            The CU sends its block FLAT now (one BD block on the
                            compute tile at any batch -- see _attn_o_put_block),
                            so this end does the un-interleave. A memtile BD has
                            FOUR dimensions where a compute tile has three, which
                            is exactly the one this needs.

                            The stream arrives in `a_o` order, which decomposes
                            token-major then dc, q_head, de:

                                src  j = t*DQ_PER_CU + dc*(QH*8) + qh*8 + de
                                dst  p = t*DQ + c*DQ_PER_CU + qh*DH + dc*8 + de

                            so iterating [t, dc, qh, de] walks the stream in
                            order and lands it transposed -- destination strides
                            [DQ, 8, DH, 1]. Same permutation the compute tile's
                            3-D descriptor used to express, plus the token.
                            """
                            assert DH % 8 == 0
                            for c in range(N_ATTN_CU):
                                ChannelGet(
                                    "attnO",
                                    omtb,
                                    indices=[idx(c)],
                                    # On the UNIT-STRIDE dimension: an
                                    # air.channel offset is the strided-view
                                    # convention, so a whole-buffer offset on
                                    # dim 0 comes out multiplied by DQ.
                                    offsets=[
                                        idx(0),
                                        idx(0),
                                        idx(0),
                                        idx(c * DQ_PER_CU),
                                    ],
                                    sizes=[
                                        idx(BATCH),
                                        idx(DH // 8),
                                        idx(Q_HEADS_PER_CU),
                                        idx(8),
                                    ],
                                    strides=[idx(DQ), idx(8), idx(DH), idx(1)],
                                )

                        if BATCH == 1:
                            _o_gather(None)
                        elif _os.environ.get("_ATTN_O_PERTOKEN"):
                            for _t in for_(idx(0), idx(BATCH), idx(1)):
                                _o_gather(arith.muli(_t, idx(DQ)))
                                yield_([])
                        else:
                            _o_gather_block()
                        # BOTH rms phases relay from the SAME set of slabs (one
                        # per re-feed cycle position -- see _rmsrow_relay), and
                        # they live
                        # HERE rather than at segment scope: a value defined
                        # outside this index_switch arm and used inside it makes
                        # the SWITCH carry an async token, which the dependency
                        # canonicalizer has no vertex for. Keeping the slab and
                        # both of its fill/drain pairs in one region also gives
                        # air-annotate-refeed what it needs -- each drain loop
                        # with its own fill in the same block.
                        #
                        # PHASE-TIME ORDER on the convergent @xin, which is what
                        # the single consumer loop reads: ph0 (this slab), ph1
                        # (the attnO buffer), ph2 (this slab again), then ph3
                        # (the down buffer, at segment level below). ph2's fill
                        # cannot run before the core has produced ph2's X, which
                        # needs o-proj, which needs ph1's X -- delivered by the
                        # refeed just above it. So program order and data order
                        # agree and there is no cycle.
                        _rb = None
                        if RMS_MEMTILE_REFEED == 3:
                            _rb = AllocOp(rmsrow_l2, [], [])
                            _rb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), RMSROW_COL)
                            )
                            _rmsrow_relay(_XN1, _RF_PH0_DEC, _rb)
                        refeed(
                            OPROJ_REFEED,
                            lambda: _xnorm_put(
                                omtb,
                                N_ATTN_CU * DQ_PER_CU,
                                ssa=True,
                                chan=_XN1,
                                indices=[idx(0)],
                            ),
                        )
                        if _rb is not None:
                            _rmsrow_relay(_XN3, _RF_PH2_DEC, _rb)
                            DeallocOp(_rb)
                        _probe_put("O", "probe5", omtb)
                        DeallocOp(omtb)

                    # gate-off 2026-07-15b: o-gather (attnO get + xnorm o-proj put) is
                    # DECODE-ONLY. In vocab attn produces no attnO, and _xc_voc already
                    # excludes OPROJ_REFEED, so the xnorm convergence stays balanced.
                    if _seg_arm_i is not None:

                        def _tapsx_fill(n):
                            """The extra waves' X fill, present on EVERY arm.

                            `n` is a PYTHON int -- 0 on the arms that have no
                            extra wave to fetch, 1 on the ones that do -- so the
                            loop is statically zero-trip rather than dynamically
                            so. Both of the other spellings were built and both
                            are refused:

                              only in the extra arm
                                'air.channel.put' op failed to get S2MM tile for
                                L3 allocation. A channel named by ONE arm does
                                not survive to flow construction, and an L3
                                channel then dies instead of being dropped
                                quietly. This is §3.15's failure, reproduced.
                              outside the switch, with an arm-selected count
                                'scf.for' op failed to fully unroll. A memtile's
                                BD program is static; a segment-scope loop over
                                a runtime bound has nothing to unroll into.

                            A zero-trip loop with CONSTANT bounds is the design's
                            own idiom for this -- it keeps the op, which is what
                            the flow and the lock credit are counted from, and
                            emits no transfer.
                            """
                            if RMS_MEMTILE_REFEED != 3 or not N_EXTRA:
                                return
                            _eb = AllocOp(rmsrow_l2, [], [])
                            _eb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), RMSROW_COL)
                            )
                            for _ in for_(idx(0), idx(n), idx(1)):
                                _rmsrow_relay(_XN1, [1], _eb, src="tapsX")
                                yield_([])
                            DeallocOp(_eb)

                        def _o_voc():
                            # The LM head's ph0 relay. No o-gather here (vocab
                            # produces no attnO) and no ph2 (no second
                            # projection phase on this arm), so this is the
                            # whole arm: VOCAB_RNDS re-sends walked out of the
                            # SAME cycle the decode arm walks, a whole number of
                            # times, so both arms leave the BD chain on BD 0.
                            #
                            # Its slab is a second alloc, which air-to-aie's
                            # unifyOnChipArmBuffers folds onto one shared buffer
                            # with the decode arm's -- correct here, because
                            # every count now lives on a fill (air.refeed_count
                            # per get) rather than on the alloc, and the two
                            # arms' fills fold onto the same len(_RF_CYCLE) BDs.
                            if RMS_MEMTILE_REFEED == 3:
                                _vb = AllocOp(rmsrow_l2, [], [])
                                _vb.operation.attributes["air.memtile_col"] = (
                                    IntegerAttr.get(T.i32(), RMSROW_COL)
                                )
                                _rmsrow_relay(_XN1, _RF_PH0_VOC, _vb)
                                DeallocOp(_vb)
                            _tapsx_fill(0)
                            yield_([])

                        def _o_dec():
                            _omtb_dec()
                            _tapsx_fill(0)
                            yield_([])

                        def _o_extra():
                            _tapsx_fill(1)
                            yield_([])

                        index_switch(
                            [],
                            _seg_arm_i,
                            ARM_IDLE_CASES,
                            case_body_builder=lambda op, i, cv: (
                                _o_voc() if cv == 0 else _o_extra()
                            ),
                            default_body_builder=lambda op: _o_dec(),
                        )
                    else:
                        _omtb_dec()

                    # ===== GLU compute tile (reproducer tile_5_2): silu(gate)*up =====
                    # gate-up (id8) demux dest -> relay memtile (strip demux already
                    # delivered pure 512) -> toGlu -> GLU herd: glu_aie512 on each 512
                    # round [up 256 | gate 256] -> 256. 32 slices. -> down memtile.
                    if GLU_DEST >= 0:

                        # BUG FIX (later43c): glu arm must track the mode (0 in vocab)
                        # like proj/rms; hardcoded 1 kept glu in _dec() during vocab,
                        # stalling on gate-up (id8) never produced in vocab mode.
                        _arm_glu = _seg_arm

                        @herd(name="glu", sizes=[1, 1], operands=[_arm_glu])
                        def glu_h(tx, ty, _sx, _sy, _arm):
                            # The batched LM head's logits leave the chip HERE
                            # (see VOCAB_DEST), so at batch>1 this core has two
                            # arms and they have to share one ring. Three
                            # constraints pin the shape, each found by building
                            # the alternative:
                            #
                            #  - Two arms with their own allocs put SEVEN 16 KB
                            #    buffers on a 64 KB tile: air-to-aie does not
                            #    merge equal-typed allocs across an
                            #    index_switch. So the ring is allocated ONCE,
                            #    above the arms, and both use it.
                            #  - One arm with an RTP-selected trip count is
                            #    worse: @outY's demux ids are derived from its
                            #    gets' static VOLUME, and a dynamic loop bound
                            #    makes that unknowable -- "selects a
                            #    destination, but its routing domain has no
                            #    demux", on all 16 proj emitters. So each arm
                            #    keeps its own STATIC count.
                            #  - The two arms' @outY gets are then the same
                            #    descriptor on the same buffer, which is what
                            #    keeps them one BD on that S2MM.
                            #
                            # scf.if and not index_switch: air-dependency's
                            # graph builder has no IndexSwitchOp async case
                            # (Util/Dependency.cpp) and these regions hold
                            # channel ops on hoisted (async-allocated) memrefs.
                            # _emit uses scf.if for the same reason.
                            _voc_arm = BATCH > 1 and GLU_DEST >= 0

                            def _get_slice(gx):
                                # get 1024 = TWO stripped demux packets DIRECTLY from
                                # the id-demux dest (reproducer mem_1_1 DMA5 ->
                                # tile_5_2 DMA0); no relay.
                                if BATCH == 1:
                                    ChannelGet(
                                        "outY",
                                        gx,
                                        indices=[idx(0), idx(GLU_DEST)],
                                        offsets=[idx(0)],
                                        sizes=[idx(GLU_SLICE)],
                                        strides=[idx(1)],
                                    )
                                    return
                                # Same de-interleave as the rms core. A GLU
                                # slice is [up | gate] and that is TWO egress
                                # rounds on llama, folded into the emitter
                                # dimension. The vocab arm reads the identical
                                # descriptor -- a slice of the vocab is just
                                # [BATCH][GLU_SLICE] of logits.
                                assert GLU_SLICE % PAYLOAD == 0
                                _go, _gs, _gt = outy_tokmajor(
                                    GLU_SLICE, rounds=GLU_SLICE // PAYLOAD
                                )
                                ChannelGet(
                                    "outY",
                                    gx,
                                    indices=[idx(0), idx(GLU_DEST)],
                                    offsets=[idx(v) for v in _go],
                                    sizes=[idx(v) for v in _gs],
                                    strides=[idx(v) for v in _gt],
                                )

                            def _glu_slice(gx, mk_gh):
                                # gh is allocated HERE, after the get, and not
                                # hoisted beside gx. Both halves of that matter:
                                # emitting the alloc before the get moves the
                                # batch-1 IR (the no-op gate says so), and
                                # hoisting it out of the loop moves the emitted
                                # `release gh-ready` from once per slice to
                                # twice at the end of a rotation, which changes
                                # the decode arm's pipelining for a buffer the
                                # vocab arm never touches. Only gx has to be
                                # shared, so only gx is hoisted.
                                _get_slice(gx)
                                gh = mk_gh()
                                if BATCH == 1:
                                    CallOp(glu_aie, [gh, gx, _arm])
                                else:
                                    # A round is [BATCH][GLU_SLICE]: the GLU is
                                    # per token, so this is a row loop and
                                    # nothing else. Unrolled -- the trip count
                                    # is the batch, and the ring depends on the
                                    # body being one get / one put.
                                    for _t in range(BATCH):
                                        CallOp(
                                            glu_row_aie,
                                            [
                                                gh,
                                                gx,
                                                arith.ConstantOp(
                                                    IntegerAttr.get(i32, _t), None
                                                ).result,
                                                _arm,
                                            ],
                                        )
                                ChannelPut(
                                    "gluOut",
                                    gh,
                                    offsets=[idx(0)],
                                    sizes=[idx(BATCH * GLU_HID)],
                                    strides=[idx(1)],
                                )
                                # Returned, not deallocated here: the decode arm
                                # frees gx before gh and the batch-1 no-op gate
                                # compares that order.
                                return gh

                            def _voc_slice(gx, mk_gh):
                                """One vocab slice: in on @outY, out on @vocOut.

                                Through gh and a COPY, rather than sending gx
                                straight back out, and the reason is the lock
                                credit rather than the wire. AIR sizes a
                                buffer's credit from the ratio of channel ops
                                that WRITE it to ops that READ it
                                (getLockValuePair), counted across both arms
                                because it cannot know they are exclusive.
                                Sending gx out makes gx both-directional and the
                                ratio comes out at 2 -- the MM2S then waits for
                                two core releases per fire and gets one, and the
                                relay stalls halfway through the wave. Copying
                                keeps gx write-only and gh read-only, which is
                                the case AIR gives credit 1 without counting
                                anything, and is what the decode arm already
                                looks like.

                                Two halves because gh is half a slice. They are
                                contiguous and consecutive, so the stream is the
                                same one a single put would make and the host's
                                logits[t] stays a plain stride (see the @vocOut
                                get in _uni_voc).
                                """
                                _get_slice(gx)
                                gh = mk_gh()
                                _half = BATCH * GLU_HID
                                for _h in range(2):
                                    CallOp(
                                        glu_copy_aie,
                                        [
                                            gh,
                                            gx,
                                            arith.ConstantOp(
                                                IntegerAttr.get(i32, _h * _half), None
                                            ).result,
                                            arith.ConstantOp(
                                                IntegerAttr.get(i32, _half), None
                                            ).result,
                                            _arm,
                                        ],
                                    )
                                    ChannelPut(
                                        "vocOut",
                                        gh,
                                        offsets=[idx(0)],
                                        sizes=[idx(_half)],
                                        strides=[idx(1)],
                                    )
                                return gh

                            # An odd slice count desyncs the ring: the core's
                            # slot sequence restarts every layer while the
                            # ring's rotation carries over, so every other layer
                            # reads a stale slice. GLU_PKTS keeps the count
                            # even, and VOC_ROTATIONS refuses a vocab chunk that
                            # would not.
                            assert NGLU % 2 == 0, (
                                f"odd NGLU={NGLU} desyncs the GLU BD ring "
                                f"(GLU_PKTS={GLU_PKTS})"
                            )

                            def _dec():
                                # FAITHFUL 2-slot ring (reproducer core_5_2: TWO glu_aie
                                # calls per loop iter, ping x_0/hid_0 + pong x_1/hid_1).
                                # Two distinct allocs per iter give air-to-aie a 2-deep
                                # S2MM/MM2S ring (lock init 2), matching tile_5_2 -- a
                                # rolled 1-call loop collapses to 1-slot (no overlap).
                                def _slice():
                                    gx = AllocOp(glu_x_l1, [], [])
                                    gh = _glu_slice(
                                        gx, lambda: AllocOp(glu_hid_l1, [], [])
                                    )
                                    DeallocOp(gx)
                                    DeallocOp(gh)

                                for _s in for_(idx(0), idx(NGLU // 2), idx(1)):
                                    _slice()  # ping
                                    _slice()  # pong
                                    yield_([])

                            if not _voc_arm:

                                def _dec_case():
                                    _dec()
                                    yield_([])

                                index_switch(
                                    [],
                                    arith.index_cast(idx_t, _arm),
                                    ARM_IDLE_CASES,
                                    case_body_builder=lambda op, i, cv: yield_([]),
                                    default_body_builder=lambda op: _dec_case(),
                                )
                            else:
                                # The @outY landing ring, hoisted -- two
                                # slots because the ring is two deep, and shared
                                # because it is the one buffer both arms fill.
                                # gh is not here: see _glu_slice.
                                # gh is hoisted too, and only here: the vocab
                                # arm needs a gh of its own and four 8 KB slots
                                # beside two 16 KB ones do not fit. Sharing the
                                # ring costs nothing in credit -- gh is
                                # read-only to the DMA in both arms, which is
                                # the case AIR gives 1 without counting.
                                _gx = [AllocOp(glu_x_l1, [], []) for _ in range(2)]
                                _gh = [AllocOp(glu_hid_l1, [], []) for _ in range(2)]

                                def _glu_dec_body():
                                    for _s in for_(idx(0), idx(NGLU // 2), idx(1)):
                                        _glu_slice(_gx[0], lambda: _gh[0])
                                        _glu_slice(_gx[1], lambda: _gh[1])
                                        yield_([])

                                def _glu_voc_body():
                                    for _s in for_(idx(0), idx(VOC_ROTATIONS), idx(1)):
                                        _voc_slice(_gx[0], lambda: _gh[0])
                                        _voc_slice(_gx[1], lambda: _gh[1])
                                        yield_([])

                                # THREE arms, from TWO scf.ifs, because an
                                # scf.index_switch cannot carry an async token
                                # here -- the four hoisted allocs precede it, so
                                # the dependency pass tries to give it one and
                                # fails with "unknown op type producing async
                                # token". (Measured; the rope and attn herds get
                                # away with index_switch because theirs is the
                                # first op in the body.) An scf.if is handled,
                                # so the extra arm becomes an ENCLOSING one and
                                # the existing two-way select is left exactly as
                                # it was -- one copy of every channel op, still
                                # inside a single air.arm_select pair.
                                _outer = None
                                if N_EXTRA:
                                    _outer = IfOp(
                                        arith.cmpi(
                                            arith.CmpIPredicate.slt,
                                            arith.index_cast(idx_t, _arm),
                                            idx(2),
                                        ),
                                        [],
                                        has_else=False,
                                    )
                                    _outer.operation.attributes["air.arm_select"] = (
                                        UnitAttr.get()
                                    )
                                _ctx = (
                                    InsertionPoint(_outer.thenRegion.blocks[0])
                                    if _outer is not None
                                    else contextlib.nullcontext()
                                )
                                with _ctx:
                                    _ifa = IfOp(
                                        arith.cmpi(
                                            arith.CmpIPredicate.ne,
                                            _arm,
                                            arith.ConstantOp(
                                                IntegerAttr.get(i32, 0), None
                                            ).result,
                                        ),
                                        [],
                                        has_else=True,
                                    )
                                    # then = decode, else = LM head. See
                                    # _if_decode on the rms core for what the
                                    # label is for.
                                    _ifa.operation.attributes["air.arm_select"] = (
                                        UnitAttr.get()
                                    )
                                    with InsertionPoint(_ifa.thenRegion.blocks[0]):
                                        _glu_dec_body()
                                        yield_([])
                                    with InsertionPoint(_ifa.elseRegion.blocks[0]):
                                        _glu_voc_body()
                                        yield_([])
                                    if _outer is not None:
                                        yield_([])
                                for _b in _gx + _gh:
                                    DeallocOp(_b)

                        glu_h.attributes["link_with"] = StringAttr.get("glu.o")
                        glu_h.attributes["x_loc"] = IntegerAttr.get(T.i64(), GLU_PCOL)
                        glu_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), 3)

                        # GLU output -> down memtile accumulate (8192). FAITHFUL: feed
                        # it back on-chip as the DOWN phase X, re-broadcast DOWN_REFEED
                        # times into the convergent @xnorm. The DOWN_REFEED loop
                        # around the put (see refeed()) makes air-to-aie emit a
                        # counting-lock-N on the fill (S2MM) side so the count-free MM2S
                        # re-reads the resident 8192 N times (reproducer down_buffer
                        # lock_5_1 init=4: one GLU fill -> 4 re-sends = the 4 down output
                        # row-blocks each re-reading all 8192). The X memtile chunks each
                        # 8192 into 16x512 -> inX for ph3.
                        db = AllocOp(down_l2, [], [])
                        db.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(), DOWN_PCOL
                        )
                        for _s in for_(idx(0), idx(NGLU), idx(1)):
                            soff = arith.muli(_s, idx(GLU_HID))
                            if BATCH == 1:
                                ChannelGet(
                                    "gluOut",
                                    db,
                                    offsets=[soff],
                                    sizes=[idx(GLU_HID)],
                                    strides=[idx(1)],
                                )
                            else:
                                # (round, token) in, (token, round) out: slice s
                                # of token t belongs at t*GLU_OUT + s*GLU_HID.
                                # The transpose is free here because the memtile
                                # is the one place a strided landing costs
                                # nothing -- the same trick as the QKV drain.
                                ChannelGet(
                                    "gluOut",
                                    db,
                                    offsets=[idx(0), soff],
                                    sizes=[idx(BATCH), idx(GLU_HID)],
                                    strides=[idx(GLU_OUT), idx(1)],
                                )
                            yield_([])
                        # re-broadcast the resident 8192 into the convergent X feed.
                        refeed(
                            DOWN_REFEED,
                            lambda: _xnorm_put(db, GLU_OUT, chan=_XN3),
                        )
                        _probe_put("D", "probe4", db)

                        DeallocOp(db)

                    # (FAITHFUL ph2): the gate-up (ph2) X is now emitted by the rms core
                    # itself (rmsnorm(x+oproj) on @xnorm with per-put refeed=32), NOT a
                    # separate buf_ph2 memtile -- see rms_h step2. This frees the memtile
                    # and gates ph2 on o-proj (reproducer core_2_2 step2).

                    # ===== 16 cascade-pair proj cores =====
                    # NB: a herd body is an isolated region -- every SSA value it uses
                    # must be created INSIDE the body (or be an operand). So all index/
                    # i32 constants are built inside _core, not captured from segment.
                    def _psw(ph, vals, ty):
                        if len(vals) == 1:
                            return vals[0]
                        return index_switch(
                            [ty],
                            ph,
                            list(range(len(vals) - 1)),
                            case_body_builder=lambda op, i, cv: yield_([vals[i]]),
                            default_body_builder=lambda op: yield_([vals[-1]]),
                        )

                    def _wscr_alloc():
                        """The batched core's unpacked-weight scratch: ONE per core.

                        q4k_mm_block materializes a whole 32x256 weight block as
                        bf16 so aie::mmul has a real B operand, and that is 16 KB
                        -- a quarter of the tile. It is scratch: dead between j
                        steps and dead between the pair's two emit passes. But an
                        AllocOp inside _mm() gives each emit pass its own, and the
                        first batch-8 build failed on exactly that: 66116 B of
                        buffers against 65536, with two 16 KB scratches in the
                        map. Allocated once here, at the top of the core body,
                        where nothing can ping-pong it.

                        None at batch 1 -- the GEMV has no scratch at all, and an
                        unused alloc is still an op in the no-op diff.

                        "where nothing can ping-pong it" is the intent and it is
                        NOT what happens: air-fuse-alloc-dealloc sinks this alloc
                        back into the j loop, and its presence there disqualifies
                        the loop's OTHER allocs -- x and w -- from the ping-pong
                        they want. PROJ_WS_NO_SINK=1 makes the hoist stick; see
                        the knob's comment for the pass-by-pass reading.
                        """
                        if BATCH <= 1:
                            return None
                        _a = AllocOp(wscr_mm_l1, [], [])
                        if PROJ_WS_NO_SINK:
                            _a.operation.attributes["air.disable_alloc_sink"] = (
                                UnitAttr.get()
                            )
                        return _a

                    def _core_blk_np(base_cx):
                        # FLM-gemma NON-PAIRED proj egress (PAIR_ROWS==1): the same
                        # [2,4] block herd, but each of the 8 tiles emits its OWN
                        # single 32-row block via its own outA put -- no lead/partner
                        # neighbor-L1 share. The y buffer is tile-local (alloc inside
                        # the body); outA index = [logical col (base_cx+tx), row ty].
                        # Mirrors gemma_npu_bin's independent-CT + memtile-4-gather;
                        # D=2560's 5 blocks/CT is native here (odd blocks OK, no pair).
                        def body(tx, ty, _sx, _sy, _arm):
                            gcx = arith.addi(idx(base_cx), tx)
                            gcy = ty
                            i2c = [idx(v) for v in I2P]
                            j2c = [idx(v) for v in J2P]
                            # Name the DESTINATION, not a packet id. DEST[ph]
                            # is the same index the receiving gets sit at
                            # (`indices=[0, p]`); the put carries it as `dest`
                            # and air-annotate-packet-ids allocates that
                            # destination's id and stores the routing header. The
                            # wire number lives in exactly one place instead of
                            # being written here and on the channel and hoped to
                            # agree.
                            pktc = [idx(d) for d in DEST]
                            c2 = idx(2)
                            # Row 0 of the packet payload; the pair's partner
                            # writes row 1. See proj_qmm_flush_row.
                            c0i = arith.ConstantOp(IntegerAttr.get(i32, 0), None).result
                            _ws = _wscr_alloc()

                            def _wscr():
                                return _ws

                            def _gemv(J2v):
                                J2x2 = arith.muli(J2v, c2)
                                a_acc = AllocOp(yacc_l1, [], [])
                                CallOp(zero, [a_acc, _arm])
                                for _j in for_(idx(0), J2x2, idx(1)):
                                    a_x = AllocOp(xblk_l1, [], [])
                                    ChannelGet("inX", a_x, indices=[gcx, gcy])
                                    a_w = AllocOp(wblk_l1, [], [])
                                    ChannelGet("wL2ToL1", a_w, indices=[gcx, gcy])
                                    CallOp(acc256, [a_x, a_w, a_acc])
                                    DeallocOp(a_x)
                                    DeallocOp(a_w)
                                    yield_([])
                                return a_acc

                            def _mm(J2v):
                                # See _core_blk's _mm for why the scratch is
                                # hoisted and why there is no reduce cache. The
                                # only difference here is that there is no pair
                                # to share with, so tok_stride is 1 and the
                                # token blocks are simply contiguous.
                                J2x2 = arith.muli(J2v, c2)
                                a_acc = AllocOp(yacc_mm_l1, [], [])
                                a_ws = AllocOp(wscr_mm_l1, [], [])
                                if PROJ_WS_NO_SINK:
                                    # Same hoist, same reason as _wscr_alloc's.
                                    a_ws.operation.attributes[
                                        "air.disable_alloc_sink"
                                    ] = UnitAttr.get()
                                CallOp(mm_zero, [a_acc, _arm])
                                if PROJ_RING_DEPTH <= 2:
                                    # PROJ_PP_ONLY: hoist the input that is NOT
                                    # being ringed, so the transform duplicates
                                    # only the other one. See the knob.
                                    _hx = _pp_hoist(xblk_mm_l1, "x")
                                    _hw = _pp_hoist(wblk_l1, "w")
                                    for _j in for_(idx(0), J2x2, idx(1)):
                                        a_x = _hx or AllocOp(xblk_mm_l1, [], [])
                                        ChannelGet("inX", a_x, indices=[gcx, gcy])
                                        a_w = _hw or AllocOp(wblk_l1, [], [])
                                        ChannelGet("wL2ToL1", a_w, indices=[gcx, gcy])
                                        CallOp(mm_acc, [a_x, a_w, a_acc, a_ws])
                                        if _hx is None:
                                            DeallocOp(a_x)
                                        if _hw is None:
                                            DeallocOp(a_w)
                                        yield_([])
                                    if _hx is not None:
                                        DeallocOp(_hx)
                                    if _hw is not None:
                                        DeallocOp(_hw)
                                else:
                                    _mm_ring_deeper(
                                        J2x2,
                                        xblk_mm_l1,
                                        wblk_l1,
                                        mm_acc,
                                        a_acc,
                                        a_ws,
                                        gcx,
                                        gcy,
                                    )
                                return a_acc

                            _proj = _mm if BATCH > 1 else _gemv
                            # PAIR_ROWS is 1 on this path, so tok_stride is 1
                            # and token t lands at 16 + t*ROW_BLOCK. Guarded for
                            # the same reason as the paired core's: an unused
                            # constant is still an op in the batch-1 diff.
                            _tokstr = (
                                arith.ConstantOp(
                                    IntegerAttr.get(i32, PAIR_ROWS), None
                                ).result
                                if BATCH > 1
                                else None
                            )

                            def _emit(a_acc, destv):
                                yb = AllocOp(
                                    ypair_mm_l1 if BATCH > 1 else ypair_l1, [], []
                                )
                                if BATCH > 1:
                                    CallOp(mm_flush, [a_acc, yb, c0i, _tokstr])
                                else:
                                    CallOp(flush_row, [a_acc, yb, c0i])
                                # dest = which egress consumer this round feeds.
                                # The compiler allocates that destination's packet
                                # id and emits the header store at offsets[0]; the
                                # kernel no longer touches routing.
                                ChannelPut(
                                    "outA",
                                    yb,
                                    indices=[gcx, ty],
                                    offsets=[idx(14)],
                                    sizes=[idx(HDR + PAIR_PAY * BATCH)],
                                    strides=[idx(1)],
                                    dest=destv,
                                )
                                DeallocOp(yb)
                                DeallocOp(a_acc)

                            _arm_i = arith.index_cast(idx_t, _arm)
                            _id4 = idx(VOCAB_DEST)  # vocab-arm packet dest

                            def _sel(voc_val, dec_thunk, ty_, ex=None):
                                # arm 0 -> vocab, arm 2+k -> extra wave k, else
                                # decode. Adding arms is safe here and only here
                                # because every arm yields a SCALAR into the same
                                # loop nest over the same channel ops; an arm
                                # that named a channel op would double its lock
                                # credit and blow the tile's BD budget both.
                                if not N_EXTRA:
                                    return index_switch(
                                        [ty_],
                                        _arm_i,
                                        [0],
                                        case_body_builder=lambda op, i, cv: yield_(
                                            [voc_val]
                                        ),
                                        default_body_builder=lambda op: yield_(
                                            [dec_thunk()]
                                        ),
                                    )
                                assert ex is not None and len(ex) == N_EXTRA
                                if len(set(ex)) == 1:
                                    # INVERT when every extra wave agrees, which
                                    # is the common case: their core scalars are
                                    # the wave SHAPE, and the shapes that differ
                                    # between waves (w_off, x_slot) are all
                                    # launch-side. One case per wave costs core
                                    # .data, and 25 fc sub-waves overflowed it by
                                    # 80 bytes across the four switches below.
                                    # Two cases and a default is the same
                                    # function of the arm.
                                    return index_switch(
                                        [ty_],
                                        _arm_i,
                                        [0, 1],
                                        case_body_builder=lambda op, i, cv: yield_(
                                            [voc_val if cv == 0 else dec_thunk()]
                                        ),
                                        default_body_builder=lambda op: yield_(
                                            [idx(ex[0])]
                                        ),
                                    )
                                return index_switch(
                                    [ty_],
                                    _arm_i,
                                    [0] + [2 + k for k in range(N_EXTRA)],
                                    case_body_builder=lambda op, i, cv: yield_(
                                        [voc_val if cv == 0 else idx(ex[cv - 2])]
                                    ),
                                    default_body_builder=lambda op: yield_(
                                        [dec_thunk()]
                                    ),
                                )

                            # An extra wave is ONE phase, like a vocab chunk.
                            nph_v = _sel(idx(1), lambda: idx(NPH), idx_t, [1] * N_EXTRA)
                            for ph in for_(idx(0), nph_v, idx(1)):
                                I2v = _sel(
                                    idx(VOCAB_I2),
                                    lambda: _psw(ph, i2c, idx_t),
                                    idx_t,
                                    EXTRA_I2,
                                )
                                J2v = _sel(
                                    idx(VOCAB_J2),
                                    lambda: _psw(ph, j2c, idx_t),
                                    idx_t,
                                    EXTRA_J2,
                                )
                                pktv = _sel(
                                    _id4,
                                    lambda: _psw(ph, pktc, idx_t),
                                    idx_t,
                                    EXTRA_DEST,
                                )
                                for _v1 in for_(idx(0), I2v, idx(1)):
                                    for _e in range(PAIR_ROWS):  # 1 (non-paired)
                                        _emit(_proj(J2v), pktv)
                                    yield_([])  # v1
                                yield_([])  # ph

                        return body

                    def _core_blk(base_cx):
                        if PAIR_ROWS == 1:
                            return _core_blk_np(base_cx)

                        def body(
                            tx,
                            ty,
                            _sx,
                            _sy,
                            c0a0,
                            c0a1,
                            c0b0,
                            c0b1,
                            c1a0,
                            c1a1,
                            c1b0,
                            c1b1,
                            _arm,
                        ):
                            # [2,4] block herd over TWO contiguous proj columns.
                            # tx in {0,1} = the block's two columns (logical col =
                            # base_cx + tx); ty in 0..3 = the four rows (row = 2 + ty).
                            # 8 shared L1 buffers = 2 columns x 2 pairs x (y0,y1). Each
                            # pair's (y0,y1) is shared only across its 2 adjacent cores.
                            col0_pairA = [c0a0, c0a1]
                            col0_pairB = [c0b0, c0b1]
                            col1_pairA = [c1a0, c1a1]
                            col1_pairB = [c1b0, c1b1]
                            # gcx = logical column index (base_cx + tx). tx is a spatial
                            # herd IV, so this bundle index spatially unrolls per column.
                            gcx = arith.addi(idx(base_cx), tx)
                            gcy = ty
                            i2c = [idx(v) for v in I2P]
                            j2c = [idx(v) for v in J2P]
                            # Stamp the DESTINATION ORDINAL, not a packet id.
                            # DEST[ph] is the same index the receiving gets sit at
                            # (`indices=[0, p]`); air-annotate-packet-ids
                            # allocates the ids and rewrites these constants to
                            # match, so the wire number lives in exactly one
                            # place instead of being written here and on the
                            # channel and hoped to agree.
                            pktc = [idx(d) for d in DEST]
                            c0i = arith.ConstantOp(IntegerAttr.get(i32, 0), None).result
                            c1i = arith.ConstantOp(IntegerAttr.get(i32, 1), None).result
                            # fill=0 for every row-block after the phase's first
                            _c0i = arith.ConstantOp(
                                IntegerAttr.get(i32, 0), None
                            ).result

                            c2 = idx(2)
                            _ws = _wscr_alloc()

                            def _wscr():
                                return _ws

                            # ONE GEMV pass: 2*J2 j-steps, single inX/wL2ToL1 get site
                            # -> AIR 2-deep x_0/x_1 + w_0/w_1 rings (the reproducer's
                            # resident 2-buffer alternation). Separate a_x0/a_x1 gets
                            # would explode the core-mem BD count (>16).
                            def _gemv(J2v, a_rc=None, fill=None):
                                J2x2 = arith.muli(J2v, c2)
                                a_acc = AllocOp(yacc_l1, [], [])
                                CallOp(zero, [a_acc, _arm])
                                for _j in for_(idx(0), J2x2, idx(1)):
                                    a_x = AllocOp(xblk_l1, [], [])
                                    ChannelGet("inX", a_x, indices=[gcx, gcy])
                                    a_w = AllocOp(wblk_l1, [], [])
                                    ChannelGet("wL2ToL1", a_w, indices=[gcx, gcy])
                                    if a_rc is None:
                                        CallOp(acc256, [a_x, a_w, a_acc])
                                    else:
                                        # slot = this col-block; fill only on the
                                        # projection's first row-block.
                                        _ji = arith.index_cast(i32, _j)
                                        CallOp(
                                            acc256_c,
                                            [a_x, a_w, a_acc, a_rc, _ji, fill],
                                        )
                                    DeallocOp(a_x)
                                    DeallocOp(a_w)
                                    yield_([])
                                return a_acc

                            def _mm(J2v, a_rc=None, fill=None):
                                """_gemv's batched twin: BATCH tokens per pass.

                                Same j-step structure, same single get site for
                                the same BD-count reason. Three differences, all
                                of them the point of the swap:

                                  - the X chunk is BATCH*COL_BLOCK and arrives
                                    TILE-BLOCKED (aie::mmul's A order), not as
                                    a plain [BATCH][COL_BLOCK] row-major buffer;
                                  - the unpacked-weight scratch is allocated
                                    ONCE outside the j loop, not per step. It is
                                    scratch, dead across steps, and 16 KB is far
                                    too much to ping-pong;
                                  - a_rc / fill are accepted and IGNORED. The
                                    reduce cache is machinery for the GEMV's
                                    +min factorisation, which this path does not
                                    use; taking the arguments keeps one call
                                    shape at every site instead of branching the
                                    callers too.
                                """
                                del a_rc, fill
                                J2x2 = arith.muli(J2v, c2)
                                a_acc = AllocOp(yacc_mm_l1, [], [])
                                a_ws = _wscr()
                                CallOp(mm_zero, [a_acc, _arm])
                                if PROJ_RING_DEPTH <= 2:
                                    # PROJ_PP_ONLY: hoist the input that is NOT
                                    # being ringed, so the transform duplicates
                                    # only the other one. See the knob.
                                    _hx = _pp_hoist(xblk_mm_l1, "x")
                                    _hw = _pp_hoist(wblk_l1, "w")
                                    for _j in for_(idx(0), J2x2, idx(1)):
                                        a_x = _hx or AllocOp(xblk_mm_l1, [], [])
                                        ChannelGet("inX", a_x, indices=[gcx, gcy])
                                        a_w = _hw or AllocOp(wblk_l1, [], [])
                                        ChannelGet("wL2ToL1", a_w, indices=[gcx, gcy])
                                        CallOp(mm_acc, [a_x, a_w, a_acc, a_ws])
                                        if _hx is None:
                                            DeallocOp(a_x)
                                        if _hw is None:
                                            DeallocOp(a_w)
                                        yield_([])
                                    if _hx is not None:
                                        DeallocOp(_hx)
                                    if _hw is not None:
                                        DeallocOp(_hw)
                                else:
                                    _mm_ring_deeper(
                                        J2x2,
                                        xblk_mm_l1,
                                        wblk_l1,
                                        mm_acc,
                                        a_acc,
                                        a_ws,
                                        gcx,
                                        gcy,
                                    )
                                return a_acc

                            _proj = _mm if BATCH > 1 else _gemv

                            # tok_stride for the batched flush: token t, pair
                            # role i land at (t*PAIR_ROWS + i)*ROW_BLOCK, which
                            # is token-major with the pair interleaved -- so the
                            # egress stays ONE contiguous put and only its size
                            # changes. Unused at BATCH 1.
                            # Guarded, not unconditional: an unused constant
                            # is still an op, and on llama-3.2-1b (PAIR_ROWS 2)
                            # it survived CSE and broke the batch-1 no-op diff.
                            # qwen3-4b hid it -- PAIR_ROWS is 1 there, so it
                            # folded into the existing c1i and the IR matched by
                            # luck. Both models have to be checked, always.
                            _tokstr = (
                                arith.ConstantOp(
                                    IntegerAttr.get(i32, PAIR_ROWS), None
                                ).result
                                if BATCH > 1
                                else None
                            )

                            def _flush(acc, buf, role):
                                if BATCH > 1:
                                    CallOp(mm_flush, [acc, buf, role, _tokstr])
                                else:
                                    CallOp(flush_row, [acc, buf, role])

                            def _emit(a_acc, yb, pktv):
                                # Nested exact-IV select: column by tx==0, pair by ty<2,
                                # role by ty==const (even row = lead). Every guard is a
                                # DIRECT tile-IV comparison (IV==const / IV<const) so it
                                # folds per-tile at the air-to-aie clone -- reachableUnderIvs
                                # folds only those (NOT tx&&ty, tx*2+.., ty%2, ty/2) --
                                # keeping each pair's shared-L1 + owner-tile analysis exact.
                                # scf.if (not index_switch): air-dependency's graph builder
                                # has no IndexSwitchOp async case (Util/Dependency.cpp).
                                def _role(bufs, lead_row, pp_c):
                                    _is_lead = arith.cmpi(
                                        arith.CmpIPredicate.eq, ty, idx(lead_row)
                                    )
                                    _if = IfOp(_is_lead, [], has_else=True)
                                    with InsertionPoint(_if.thenRegion.blocks[0]):
                                        _flush(a_acc, bufs[yb], c0i)
                                        # WIDEN, do not repeat: one packet per
                                        # round, BATCH times longer. N_ROUNDS,
                                        # the BD count and the instruction
                                        # stream all stay put.
                                        ChannelPut(
                                            "outA",
                                            bufs[yb],
                                            indices=[gcx, idx(pp_c)],
                                            offsets=[idx(14)],
                                            sizes=[idx(HDR + PAIR_PAY * BATCH)],
                                            strides=[idx(1)],
                                            dest=pktv,
                                        )
                                        yield_([])
                                    with InsertionPoint(_if.elseRegion.blocks[0]):
                                        _flush(a_acc, bufs[yb], c1i)
                                        yield_([])

                                def _pairs(pA, pB):
                                    _lo = arith.cmpi(
                                        arith.CmpIPredicate.slt, ty, idx(2)
                                    )
                                    _ifp = IfOp(_lo, [], has_else=True)
                                    with InsertionPoint(_ifp.thenRegion.blocks[0]):
                                        _role(pA, 0, 0)
                                        yield_([])
                                    with InsertionPoint(_ifp.elseRegion.blocks[0]):
                                        _role(pB, 2, 1)
                                        yield_([])

                                _isc0 = arith.cmpi(arith.CmpIPredicate.eq, tx, idx(0))
                                _ifc = IfOp(_isc0, [], has_else=True)
                                with InsertionPoint(_ifc.thenRegion.blocks[0]):
                                    _pairs(col0_pairA, col0_pairB)
                                    yield_([])
                                with InsertionPoint(_ifc.elseRegion.blocks[0]):
                                    _pairs(col1_pairA, col1_pairB)
                                    yield_([])
                                DeallocOp(a_acc)

                            # SHARED-DMA (the reference proj_main style): ONE _gemv/_emit structure
                            # (one set of inX/wL2ToL1/outA ring BDs) with RTP-SELECTED
                            # phase COUNT + per-phase params -- so decode + LM modes do
                            # NOT double the tile BDs (index_switch over the dataflow
                            # would -> >16). _arm==1 -> NPH decode phases; _arm==0 -> 1
                            # vocab phase (I2=VOCAB_I2, J2=VOCAB_J2, pkt=id4=RMS_DEST).
                            _arm_i = arith.index_cast(idx_t, _arm)
                            _id4 = idx(VOCAB_DEST)  # vocab-arm packet dest

                            def _sel(voc_val, dec_thunk, ty, ex=None):
                                # See _core_blk_np's _sel: arm 0 vocab, 2+k the
                                # k-th extra wave, default decode -- scalars only.
                                if not N_EXTRA:
                                    return index_switch(
                                        [ty],
                                        _arm_i,
                                        [0],
                                        case_body_builder=lambda op, i, cv: yield_(
                                            [voc_val]
                                        ),
                                        default_body_builder=lambda op: yield_(
                                            [dec_thunk()]
                                        ),
                                    )
                                assert ex is not None and len(ex) == N_EXTRA
                                return index_switch(
                                    [ty],
                                    _arm_i,
                                    [0] + [2 + k for k in range(N_EXTRA)],
                                    case_body_builder=lambda op, i, cv: yield_(
                                        [voc_val if cv == 0 else idx(ex[cv - 2])]
                                    ),
                                    default_body_builder=lambda op: yield_(
                                        [dec_thunk()]
                                    ),
                                )

                            nph_v = _sel(idx(1), lambda: idx(NPH), idx_t, [1] * N_EXTRA)
                            for ph in for_(idx(0), nph_v, idx(1)):
                                I2v = _sel(
                                    idx(VOCAB_I2),
                                    lambda: _psw(ph, i2c, idx_t),
                                    idx_t,
                                    EXTRA_I2,
                                )
                                J2v = _sel(
                                    idx(VOCAB_J2),
                                    lambda: _psw(ph, j2c, idx_t),
                                    idx_t,
                                    EXTRA_J2,
                                )
                                pktv = _sel(
                                    _id4,
                                    lambda: _psw(ph, pktc, idx_t),
                                    idx_t,
                                    EXTRA_DEST,
                                )
                                # b_col_reduce_add cache: one per core, scoped to
                                # the PROJECTION (x changes per phase, so it is
                                # refilled on each phase's first row-block). The
                                # rc_arm call is what holds the alloc at this
                                # scope -- without a use outside the v1/j loops
                                # AIR would sink it into the col-block loop and
                                # the cache would reset every row-block.
                                a_rc = None
                                if PROJ_RC_CACHE:
                                    a_rc = AllocOp(rcache_l1, [], [])
                                    CallOp(rc_arm, [a_rc, _arm])
                                for _v1 in for_(idx(0), I2v, idx(1)):
                                    # PAIR_ROWS GEMV emits per v1 into the PAIR_ROWS y
                                    # buffers: paired (llama) -> y_0 then y_1 (2 blocks/
                                    # round, lead+partner); non-paired (gemma) -> y_0 only
                                    # (1 block/round per tile, handles odd blocks/tile).
                                    _fill0 = None
                                    if PROJ_RC_CACHE:
                                        # row-block index = _v1*PAIR_ROWS + _e, so
                                        # only (_v1==0, _e==0) is the phase's first.
                                        _fill0 = (
                                            c1i
                                            if PROJ_RC_FILL_ALL
                                            else arith.extui(
                                                i32,
                                                arith.cmpi(
                                                    arith.CmpIPredicate.eq, _v1, idx(0)
                                                ),
                                            )
                                        )
                                    for _e in range(PAIR_ROWS):
                                        _f = (
                                            None
                                            if not PROJ_RC_CACHE
                                            else (
                                                c1i
                                                if PROJ_RC_FILL_ALL
                                                else (_fill0 if _e == 0 else _c0i)
                                            )
                                        )
                                        _emit(_proj(J2v, a_rc, _f), _e, pktv)
                                    yield_([])  # v1
                                if PROJ_RC_CACHE:
                                    DeallocOp(a_rc)
                                yield_([])  # ph

                        return body

                    _arm_proj = _seg_arm
                    # Fuse all 16 proj cores into TWO [2,4] block herds: west = logical
                    # cols 0,1 (phys 0,1), east = logical cols 2,3 (phys 6,7). Each block
                    # is a contiguous 2-col x 4-row rectangle. The two blocks cannot merge
                    # into one herd -- cols 2-5 (rms/rope/attn/glu) split them. Per block:
                    # 8 shared L1 buffers (2 cols x 2 pairs x y0/y1); each pair's buffers
                    # are shared only across its 2 vertically-adjacent cores; air-to-aie
                    # infers each from the per-pair cross-core RAW and owns it on the lead
                    # tile (outA-put DMA owner). Column (tx), pair (ty<2), and role (ty==0/2)
                    # all fold per-tile from the tile IVs at clone.
                    for blk in range(NCX // 2):
                        base_cx = blk * 2  # logical col of tx=0
                        if PAIR_ROWS == 1:
                            # non-paired: each tile allocs its own y buffer locally.
                            _ops = [_arm_proj]
                        else:
                            _ypair_t = ypair_mm_l1 if BATCH > 1 else ypair_l1
                            bufs = [AllocOp(_ypair_t, [], []) for _ in range(8)]
                            _ops = [b.result for b in bufs] + [_arm_proj]
                        blk_h = herd(
                            name=f"proj_blk{blk}",
                            sizes=[2, 4],
                            operands=_ops,
                        )(_core_blk(base_cx))
                        blk_h.attributes["link_with"] = StringAttr.get("proj_qmm.o")
                        blk_h.attributes["x_loc"] = IntegerAttr.get(
                            T.i64(), PCOL[base_cx]
                        )
                        blk_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

                    # ===== rms producer core (reproducer core_2_2, tile_2_2) =====
                    # input-layernorm: raw X + rms weight -> normed X -> xnorm (re-fed
                    # on-chip REFEED times, see refeed() -> the X memtile).

                    OPROJ_RNDS = PAIR_ROWS * I2P[1]  # 4 o-proj egress rounds
                    DOWN_RNDS = PAIR_ROWS * I2P[DOWN_PHASE]  # 4 down egress rounds
                    if RMS_BAND_STREAM >= 3:
                        # residual1/residual2 do one band per round (STG_W
                        # == PAYLOAD, asserted at STG_W's definition), so
                        # this only lines up if a round's worth of egress
                        # covers exactly one band -- true because each
                        # projection's output covers the whole K-wide
                        # residual in RNDS rounds of PAYLOAD each, same as
                        # K // STG_W bands of STG_W each. The launch side
                        # (_uni_dec) re-derives this same count as
                        # K // STG_W directly, since OPROJ_RNDS/DOWN_RNDS
                        # aren't in its scope -- assert here that the two
                        # ways of counting agree.
                        assert OPROJ_RNDS == K // STG_W
                        assert DOWN_RNDS == K // STG_W

                    # per-token RTP ARM (the reference-faithful re-dispatch): scalar herd operand ->
                    # AIR emits __air_herd_rtp + __air_herd_lock acquired per token; the
                    # runtime re-arms it each dispatch so the core does 1 token/dispatch.
                    _arm_rms = _seg_arm

                    def _rms_body(tx, ty, _sx, _sy, _arm):
                        # DIAGNOSTIC (later43e): make rms SINGLE-mode in the LM_HEAD build
                        # (standalone form). The dual-mode index_switch over DATAFLOW puts
                        # BOTH branches' channel ops in the rms mem block -> doubled BDs on
                        # the tile's 2 S2MM + 2 MM2S -> suspected over-subscription that
                        # breaks the vocab compute. If single-mode rms (attention still
                        # un-gated) makes vocab WORK, the fix is to rewrite rms dual-mode
                        # the proj way (scalar _sel over one BD set), preserving CDO.
                        _SINGLE_RMS = True  # fixed config

                        def _rms_lm_head():
                            # mode 0 (LM head): final rmsnorm(x) -> feed proj X
                            # (refeed VOCAB_RNDS via xnorm), then forward the vocab
                            # projection (id4/RMS_DEST) to shim as logits (1 channel =
                            # layerOut) via a 2-deep ring (mirrors rms_residual.cc:211).
                            # Gemma (N_NORMS>=4): rmsW/rmsW2 are 2K (two norms packed);
                            # vocab feeds final_norm in rmsW's HI half (_uni_voc) -> use
                            # rms_norm_hi_aie below. rmsW2 is a 2K dummy. No rmsW3/rmsW4.
                            _rms_w_ty = rms_w2k_l1 if N_NORMS >= 4 else rms_l1
                            _rms_final = (
                                rms_norm_hi_aie if N_NORMS >= 4 else rms_norm_aie
                            )
                            a_xl = AllocOp(rms_l1, [], [])
                            ChannelGet("rmsX", a_xl, indices=[idx(0)])
                            a_wl = AllocOp(_rms_w_ty, [], [])
                            ChannelGet("rmsW", a_wl, indices=[idx(0)])
                            if POST_RMS:
                                # consume the vocab dummy rmsW2 (see _uni_voc) so the
                                # shared rmsX/rmsW2 packet group has no vocab-mode hole.
                                a_w2l = AllocOp(_rms_w_ty, [], [])
                                ChannelGet("rmsW2", a_w2l, indices=[idx(0)])
                                DeallocOp(a_w2l)
                            a_xnl = AllocOp(rms_l1, [], [])
                            # x re-broadcast. Baking the WHOLE count N=VOCAB_RNDS into one
                            # re-broadcast puts it in a single producer-side credit lock, and
                            # the AIE-ML lock is 7-bit (max +63, xaie_locks_aieml.c 0x7F), so
                            # N>63 (I2>=32) made AcquireGreaterEqual(N) unsatisfiable ->
                            # DEADLOCK. That is why the count is split: the re-broadcast below
                            # carries only XN_PER_BLK (=K/PAYLOAD=4) and the outer loop supplies
                            # the rest, so the credit lock is a constant 4 -- independent of
                            # VOCAB_I2, and the same 1/2/4 lock values the reference uses.
                            # (Constraint 4 in the VOCAB_I2 derivation above therefore no longer
                            # binds through this path; it has not been re-tested at larger I2.)
                            # The sends are INTERLEAVED with the outY->layerOut relay -- this ONE
                            # rms core both produces x and relays logits, and the reference does
                            # the same on its rms tile CT02=tile(2,2) (x broadcast on MM2S DMA0
                            # -> mem_tile_1_1, logits on MM2S DMA1 -> shim_noc_tile_3_0; AIR
                            # mirrors that port split, layerOut on MM2S0 / xnorm on MM2S1). The
                            # interleave keeps the producer from serializing ahead of the drain
                            # and backpressure-deadlocking: XN_PER_BLK x-sends per
                            # drained K-block. Total x-sends = VOCAB_RNDS, drain blocks =
                            # VOCAB_RNDS*PAYLOAD/K. rms recompute per round is negligible vs
                            # the vocab GEMV (matches the reference re-running rms per row-block).
                            a_v = AllocOp(rms_l1, [], [])
                            _voc_blks_2k = VOCAB_RNDS * PAYLOAD // K
                            _xn_per_blk = K // PAYLOAD  # = VOCAB_RNDS // _voc_blks_2k
                            # A floor-truncated round count emits too few xnorm
                            # broadcasts and drains short -> the vocab wave
                            # deadlocks on device. Catch it at build time.
                            assert _voc_blks_2k * _xn_per_blk == VOCAB_RNDS, (
                                f"VOCAB_CHUNK_I2={VOCAB_I2} gives VOCAB_RNDS="
                                f"{VOCAB_RNDS}, not a multiple of K/PAYLOAD="
                                f"{_xn_per_blk}"
                            )
                            # FLM-FAITHFUL final norm (rms_residual.cc, the
                            # IS_ATTN[0]==0 branch): the reference calls rms_norm
                            # ONCE for the whole LM head, before its vocab loop, and
                            # the loop body then only re-AUTHORIZES DMA re-sends of
                            # that one buffer (_lock_release(y_cons_lock,
                            # y_repeats_per_round)) plus a memcpy relay of the
                            # logits. It does NOT re-normalize per round -- the
                            # earlier comment here claimed it did, and that is what
                            # put the call inside the loop.
                            # a_xl (raw x), a_wl (the norm weight) and _arm are all
                            # loop-invariant, so every one of those calls recomputed
                            # the identical bytes: VOCAB_RNDS per wave, 252 per token
                            # at ~1.5k cycles each, sitting directly in front of each
                            # x-send that the 16 proj cores wait on.
                            CallOp(_rms_final, [a_xnl, a_xl, a_wl, _arm])
                            for _rv in for_(idx(0), idx(_voc_blks_2k), idx(1)):
                                # a_xnl is now loop-invariant, so this IS a
                                # re-broadcast: air-annotate-refeed collapses it to
                                # one put whose credit is _xn_per_blk -- 4 (1B), 6
                                # (3B), 5 (gemma), all far inside the 7-bit (max +63)
                                # AIE-ML lock. Total productions stay
                                # _voc_blks_2k*_xn_per_blk == VOCAB_RNDS, so the X
                                # memtile's get count is untouched.
                                #
                                # No backpressure hazard, for a stronger reason than
                                # the send/drain interleave (which is unchanged, but
                                # is NOT what makes this safe -- gemma has
                                # _voc_blks_2k=1, i.e. one group and one drain both
                                # before and after the hoist, so there is no
                                # interleave there to rely on). The rms core's three
                                # @xnorm re-broadcasts SHARE one buf-free lock, and
                                # the DECODE path's counts are larger in every model:
                                # 1B 4 vs 6/32, 3B 6 vs 10/32, gemma 5 vs 8/40. The
                                # lock init is that max (32/32/40) and is unchanged by
                                # this hoist, so the LM-head AcquireGreaterEqual has
                                # 6-8x credit slack and cannot block ahead of the
                                # drain. More generally N x Acquire(1) and 1 x
                                # Acquire(N) have identical liveness on a counting
                                # semaphore when the credit supply is unchanged and
                                # nothing else contends -- both hold here, so the
                                # collapse can delay but never deadlock.
                                #
                                # Verified rather than argued: the pre/post lowering
                                # diff shows zero aie.buffer / dma_bd / dma_start /
                                # flow changes and a bit-identical 189-entry lock
                                # table, and of 27 core ELFs exactly one (core_2_2,
                                # this core) differs -- in all three models. On
                                # device: llama-1B emits a bit-identical 64-token
                                # greedy id sequence vs a pre-hoist control, 3B passes
                                # top-5 verify, gemma passes its Paris gate.
                                refeed(
                                    _xn_per_blk,
                                    lambda: _xnorm_put(a_xnl, K),
                                )
                                ChannelGet(
                                    "outY",
                                    a_v,
                                    indices=[idx(0), idx(RMS_DEST)],
                                    offsets=[idx(0)],
                                    sizes=[idx(K)],
                                    strides=[idx(1)],
                                )
                                ChannelPut(
                                    "layerOut",
                                    a_v,
                                    offsets=[idx(0)],
                                    sizes=[idx(K)],
                                    strides=[idx(1)],
                                )
                                yield_([])
                            DeallocOp(a_xl)
                            DeallocOp(a_wl)
                            DeallocOp(a_xnl)
                            DeallocOp(a_v)

                        # Batched: the same DUAL-mode index_switch, over the
                        # batched bodies. The vocab arm is NOT the batch-1 one
                        # -- it cannot be. Its @xnorm put would be the
                        # memtile-shaped chunk-major descriptor whose
                        # 512-element wrap does not fit a compute tile's 8-bit
                        # wrap field (batch_wire.py's AIE2p limits), and it
                        # relays the logits through this core, which at batch 8
                        # neither fits the tile nor its DMA ports. Both are
                        # answered by _rms_lm_head_batched: the same chunked
                        # @xnorm put the decode arm makes, and no logit path at
                        # all (the memtile drains them -- see the dest-0 vocab
                        # relay). Every flow it uses is one the decode arm
                        # already has, on the same buffer, which is what the
                        # port rule needs.
                        if BATCH > 1:
                            _rms_batched(_arm)
                            return
                        # FUSED: rms is always DUAL-mode (index_switch on arm) so the
                        # device (mem_2_2 BDs) is IDENTICAL in the decode and lm_head
                        # builds -> one shared CDO. arm=1 -> decode residual; arm=0 ->
                        # vocab final-norm + logit forward.
                        if _SINGLE_RMS and LM_HEAD:
                            _rms_lm_head()
                            return

                        def _rms_lm_case():
                            _rms_lm_head()
                            yield_([])  # index_switch case terminator

                        def _rms_decode():
                            _rms_decode_body(_arm)
                            yield_([])  # index_switch default terminator

                        _arm_i = arith.index_cast(idx_t, _arm)
                        index_switch(
                            [],
                            _arm_i,
                            [0],
                            case_body_builder=lambda op, i, cv: _rms_lm_case(),
                            default_body_builder=lambda op: _rms_decode(),
                        )

                    def _rms_band_get(dst):
                        """Level 3: refill `dst` (rstg_l1-shaped) with one
                        fresh STG_W-wide band of _x_in()/_x_final -- WHICH
                        band is entirely the launch side's (_uni_dec /
                        _uni_voc) choice; this side only knows it is
                        draining one band's worth, landing it tightly
                        packed. `dst` is 3-D for the same reason the
                        level-2 rmsX fetch is (a compute tile's BD wrap
                        field is 8 bits): [chunk-subdivision, batch,
                        element], with `dst`'s own STG_W row-stride in the
                        batch dimension -- NOT the K stride a resident row
                        would use, since `dst` is only ever one band wide.
                        """
                        assert STG_W % _RMS_DMA_CHUNK == 0
                        # TOKEN-MAJOR, not chunk-major, and the reason is on the
                        # LAUNCH side. A band is [BATCH][STG_W] in `dst`; walking
                        # it token-major makes each token's STG_W elements
                        # CONTIGUOUS in the stream, which lets the shim describe
                        # a whole band in two dimensions ([BATCH, STG_W]) instead
                        # of three. That is what buys ph0's entire feed --
                        # refeed, band, token, element -- a legal FOUR-dimension
                        # shim descriptor, and a shim BD holds three dimensions
                        # plus a repeat. Chunk-major needs three dims per band
                        # and pushes the phase to five, which is dropped without
                        # a diagnostic (see _rms_x_feed_ph).
                        # The _RMS_DMA_CHUNK sub-dicing stays: it is here because
                        # a COMPUTE tile's BD wrap field is 8 bits and STG_W does
                        # not fit in it. The shim's is 10, so the shim side does
                        # not need it.
                        ChannelGet(
                            "rmsX",
                            dst,
                            offsets=[0, 0, 0],
                            sizes=[BATCH, STG_W // _RMS_DMA_CHUNK, _RMS_DMA_CHUNK],
                            strides=[STG_W, _RMS_DMA_CHUNK, 1],
                        )

                    def _rms_band_put_back(src):
                        """Level 3: write `src`'s one resident band back to
                        _x_in()'s DDR address, in place -- the layerOut
                        half of residual1/residual2's round-trip (the
                        launch side's matching rmsX put later re-reads
                        exactly this). A band is not contiguous across
                        rows in DRAM (each row's STG_W slice is followed by
                        the rest of that K-wide row), so this needs the
                        same 3-D shape as _rms_band_get, mirrored: `src`
                        gets the STG_W-strided (tightly packed) side.
                        """
                        assert STG_W % _RMS_DMA_CHUNK == 0
                        # Token-major, mirroring _rms_band_get -- see there.
                        ChannelPut(
                            "layerOut",
                            src,
                            offsets=[0, 0, 0],
                            sizes=[BATCH, STG_W // _RMS_DMA_CHUNK, _RMS_DMA_CHUNK],
                            strides=[STG_W, _RMS_DMA_CHUNK, 1],
                        )

                    def _rms_batched_norm(
                        xb, stg, scl, wbuf, nrefeed, _arm, ch="xnorm"
                    ):
                        """Re-broadcast rmsnorm(xb) nrefeed times, by chunk.

                        The scale pass is per row and runs once; the chunk loop
                        is what the X memtile actually gets, one
                        [BATCH][2*COL_BLOCK] window per get, which is the layout
                        xfeed_bd.py re-blocks for the mmul.

                        Shared by both arms, and that is the point rather than a
                        tidiness: the decode residual and the LM head differ
                        only in the WEIGHT and the COUNT. Every buffer, every
                        descriptor and therefore every BD on this tile's @xnorm
                        port is the same op in both, which is what lets the two
                        arms fold onto one dma_start chain.
                        """
                        _nchunk = K // XCHUNK

                        def _i32(v):
                            return arith.ConstantOp(
                                IntegerAttr.get(i32, v), None
                            ).result

                        if RMS_BAND_STREAM >= 3:
                            # xb is now ONE STG_W-wide band (rstg_l1-shaped),
                            # refetched fresh -- band-outer/token-inner, not
                            # token-outer/band-inner like levels 1/2, because
                            # each fetch lands ALL BATCH rows of one band at
                            # once and there is no resident copy to revisit:
                            # fetching band-outer means every band is
                            # fetched exactly once, total, for the whole
                            # scale pass (levels 1/2 could afford either
                            # order since xb was fully resident either way).
                            # `first` still resets scales[t] per TOKEN at
                            # band 0 -- correct regardless of loop order,
                            # since each token's accumulator is independent.
                            #
                            # AN scf.for, NOT A PYTHON-UNROLLED ONE, AND THAT
                            # IS THE WHOLE REASON BATCH 16 BUILDS. A core
                            # tile's `aie.mem` holds at most 16 BD blocks, and
                            # a get inside an scf.for costs ONE (the loop
                            # becomes a repeat count on a single BD) where
                            # _nband unrolled gets cost _nband. Unrolled, the
                            # rms core needed 2*_nband + 12 blocks -- 22 for
                            # qwen3-4b's 5 bands, over the limit, which is
                            # exactly the `'aie.mem' op has more than 16
                            # blocks` that level 3 used to die on. Rolled, the
                            # pre-pass costs 1 block per phase instead of
                            # _nband and the core lands at 14, INDEPENDENT of
                            # K. Nothing about the transfer changes: every
                            # visit is the same op (see _rms_band_get -- the
                            # offset is always 0, the launch side picks the
                            # band), so the unrolled form was _nband textually
                            # identical copies of one op.
                            #
                            # `first` was the only thing the unrolled form
                            # varied, and it does not need unrolling either:
                            # it is (band == 0), which the loop index already
                            # carries. Same extui(cmpi) shape PROJ_RC_CACHE's
                            # own first-visit flag uses.
                            _nband = K // STG_W
                            if _os.environ.get("_RMS_W3_IN_NORM"):
                                # Bisection: one extra band get at the top of the
                                # norm body -- same descriptor as the weight gets,
                                # straight-line, consumed immediately. The ONLY
                                # difference from _RMS_W3 (which works) is that
                                # this sits AFTER the scf.index_switch that _cnt
                                # emits for the arm-selected refeed count.
                                _rms_band_get(xb)
                            if _os.environ.get("_RMS_UNROLL_PREPASS"):
                                # Bisection: the pre-pass Python-unrolled, so its
                                # band gets are STRAIGHT-LINE instead of inside an
                                # scf.for. Every rmsX get on this tile shares one
                                # descriptor and one buffer, so getUniqueBDPattern
                                # folds them to a single BD either way -- unrolling
                                # costs no BD blocks now that rmsX is alone on its
                                # port (it did when it shared with rmsW/rmsW2,
                                # which is why the loop was introduced).
                                # _RMS_PREPASS_BANDS=N: fetch on the first N
                                # bands only (paired with the launch feeding N).
                                # Sweeps how many band-from-X transfers the
                                # pipeline survives.
                                _nb = int(_os.environ.get("_RMS_PREPASS_BANDS", _nband))
                                for _bi in range(_nband):
                                    if _bi < _nb:
                                        _rms_band_get(xb)
                                    for t in range(BATCH):
                                        CallOp(
                                            rms_scale_row_partial_banded_aie,
                                            [
                                                scl,
                                                xb,
                                                _i32(t),
                                                _i32(STG_W),
                                                _i32(1 if _bi == 0 else 0),
                                                _arm,
                                            ],
                                        )
                                for t in range(BATCH):
                                    CallOp(
                                        rms_scale_row_finalize_aie,
                                        [scl, _i32(t), _arm],
                                    )
                                _n = nrefeed if isinstance(nrefeed, int) else None
                                for _r in for_(
                                    idx(0),
                                    idx(_n) if _n is not None else nrefeed,
                                    idx(1),
                                ):
                                    for _c in for_(idx(0), idx(_nchunk), idx(1)):
                                        CallOp(
                                            rms_chunk_banded_aie,
                                            [
                                                stg,
                                                xb,
                                                wbuf,
                                                scl,
                                                _i32(BATCH),
                                                arith.index_cast(i32, _c),
                                                _i32(XCHUNK),
                                            ],
                                        )
                                        ChannelPut(
                                            "xnorm",
                                            stg,
                                            offsets=[idx(0)],
                                            sizes=[idx(BATCH * XCHUNK)],
                                            strides=[idx(1)],
                                        )
                                        yield_([])
                                    yield_([])
                                return
                            for _b in for_(idx(0), idx(_nband), idx(1)):
                                # _RMS_FETCH_OFF: bisection, drops BOTH the
                                # pre-pass and the regen refetch and feeds no ph0
                                # sweep at all, leaving @rmsX carrying only the
                                # two weight bands and the residual round-trips.
                                # Output is wrong by construction; the question
                                # is whether it COMPLETES.
                                if not _os.environ.get("_RMS_FETCH_OFF"):
                                    _rms_band_get(xb)
                                _first = arith.extui(
                                    i32,
                                    arith.cmpi(arith.CmpIPredicate.eq, _b, idx(0)),
                                )
                                for t in range(BATCH):
                                    CallOp(
                                        rms_scale_row_partial_banded_aie,
                                        [
                                            scl,
                                            xb,
                                            _i32(t),
                                            _i32(STG_W),
                                            _first,
                                            _arm,
                                        ],
                                    )
                                yield_([])
                            for t in range(BATCH):
                                CallOp(rms_scale_row_finalize_aie, [scl, _i32(t), _arm])
                        elif RMS_BAND_STREAM:
                            # Transitional: xb is still the full resident row
                            # (row_stride=K), only the REDUCTION is banded, to
                            # prove the two-pass kernel matches rms_scale_row_aie
                            # before xb itself shrinks (see the RMS_BAND_STREAM
                            # comment above PROJ_RING_DEPTH).
                            assert K % STG_W == 0
                            _nband = K // STG_W
                            for t in range(BATCH):
                                for _b in range(_nband):
                                    CallOp(
                                        rms_scale_row_partial_aie,
                                        [
                                            scl,
                                            xb,
                                            _i32(t),
                                            _i32(K),
                                            _i32(_b * STG_W),
                                            _i32(STG_W),
                                            _i32(1 if _b == 0 else 0),
                                            _arm,
                                        ],
                                    )
                                CallOp(rms_scale_row_finalize_aie, [scl, _i32(t), _arm])
                        else:
                            for t in range(BATCH):
                                CallOp(rms_scale_row_aie, [scl, xb, _i32(t), _arm])
                        if RMS_MEMTILE_REFEED == 3:
                            # ONE WHOLE ROW per visit, BATCH visits, and that is
                            # ONE SWEEP -- no chunk loop, so each value is
                            # computed once per sweep rather than once per
                            # output row-block. `nrefeed` is the number of
                            # SWEEPS this arm makes (not re-sends): one per
                            # relay fill, because the relay replays a fill by a
                            # count baked into a BD and the arms share that BD
                            # chain. See the _RF_CYCLE derivation -- qwen3-4b
                            # lands at 10 sweeps a layer on the decode arm and 6
                            # on the LM head, against 250 chunk regenerations.
                            #
                            # A sweep is a pure recompute from RESIDENT data
                            # (xb is the [BATCH][K] residual, scl the finished
                            # scales), so a second sweep costs no transfer.
                            # The cross-row interleave
                            # the projection cores need is done by the memtile's
                            # READ descriptor instead (_feed_inX_rf, row-major
                            # form), which costs nothing: it is the same
                            # descriptor with the token stride at K rather than
                            # XCHUNK, verified to emit the identical (token,
                            # column) sequence.
                            #
                            # An scf.for, NOT a Python-unrolled loop: a compute
                            # tile's aie.mem holds at most 16 BD blocks and a put
                            # inside an scf.for costs ONE (the loop becomes a
                            # repeat count on a single BD), where BATCH unrolled
                            # puts cost BATCH. Same reason RMS_BAND_STREAM 3
                            # rolls its band loop.
                            #
                            # stg is [BATCH*STG_W] = 4096 and K is 2560, so the
                            # row lands in the buffer that is already there; no
                            # new allocation and no change to this core's L1.
                            _ns = nrefeed if isinstance(nrefeed, int) else None
                            for _s in for_(
                                idx(0),
                                idx(_ns) if _ns is not None else nrefeed,
                                idx(1),
                            ):
                                for _t in for_(idx(0), idx(BATCH), idx(1)):
                                    CallOp(
                                        rms_row_aie,
                                        [
                                            stg,
                                            xb,
                                            wbuf,
                                            scl,
                                            arith.index_cast(i32, _t),
                                            _i32(K),
                                            _i32(K),
                                        ],
                                    )
                                    # 1-D contiguous, like the chunk put below
                                    # and for the same reason: a compute tile's
                                    # wrap field is 8 bits.
                                    ChannelPut(
                                        ch,
                                        stg,
                                        offsets=[idx(0)],
                                        sizes=[idx(K)],
                                        strides=[idx(1)],
                                    )
                                    yield_([])
                                yield_([])
                            return
                        _n = nrefeed if isinstance(nrefeed, int) else None
                        for _r in for_(
                            idx(0), idx(_n) if _n is not None else nrefeed, idx(1)
                        ):
                            for _c in for_(idx(0), idx(_nchunk), idx(1)):
                                if RMS_BAND_STREAM >= 3:
                                    # A fresh fetch per (refeed, chunk) visit
                                    # -- xb holds no data across visits, so
                                    # the raw x this chunk needs has to come
                                    # back from DDR every time. Confirmed
                                    # (docs/DFlashFeasibility.md, "checked,
                                    # not assumed") that this per-visit count
                                    # cannot be reduced: L1 usage is set by
                                    # xb's declared size, not refill
                                    # frequency, and there is no spare BD
                                    # dimension to fold the refeed count
                                    # into. `xb`'s own offset here is always
                                    # 0 (see _rms_band_get) -- WHICH band
                                    # this is is the launch side's choice,
                                    # not this side's; `_c` only selects `w`'s
                                    # slice (unchanged, still fully resident).
                                    # _RMS_NO_REGEN_FETCH: bisection. Drop the
                                    # per-visit refetch and let the regen read
                                    # whatever band is already in xb. The OUTPUT
                                    # is wrong by construction; the question is
                                    # whether the dispatch COMPLETES, which
                                    # splits ph0 into (weights + scale pre-pass)
                                    # and (regen refetch). Paired with
                                    # _rms_x_feed_ph feeding nrefeed=0 so the
                                    # channel still balances.
                                    if not _os.environ.get(
                                        "_RMS_NO_REGEN_FETCH"
                                    ) and not _os.environ.get("_RMS_FETCH_OFF"):
                                        _rms_band_get(xb)
                                    CallOp(
                                        rms_chunk_banded_aie,
                                        [
                                            stg,
                                            xb,
                                            wbuf,
                                            scl,
                                            _i32(BATCH),
                                            arith.index_cast(i32, _c),
                                            _i32(XCHUNK),
                                        ],
                                    )
                                else:
                                    CallOp(
                                        rms_chunk_aie,
                                        [
                                            stg,
                                            xb,
                                            wbuf,
                                            scl,
                                            _i32(BATCH),
                                            arith.index_cast(i32, _c),
                                            _i32(XCHUNK),
                                        ],
                                    )
                                # 1-D and contiguous on purpose: a compute
                                # tile's wrap field is 8 bits, so the 3-D
                                # chunk-major form the MEMTILE producers use
                                # would not be legal here.
                                ChannelPut(
                                    "xnorm",
                                    stg,
                                    offsets=[idx(0)],
                                    sizes=[idx(BATCH * XCHUNK)],
                                    strides=[idx(1)],
                                )
                                yield_([])
                            yield_([])

                    def _rms_batched(_arm):
                        """Both arms, for B tokens, on ONE BATCH*K buffer.

                        ONE BODY, not one per arm, and that is forced. The two
                        arms have to share these buffers -- two sets is
                        2x(32+8+4+4) KB on a 64 KB tile -- and AIR derives a
                        buffer's lock credit from how many channel ops NAME it
                        (getLockValuePair, AIRToAIESchedulingUtils.cpp), counting
                        across scf.if arms because it cannot know they are
                        exclusive. Two arms therefore doubled the credit on
                        @layerOut's MM2S and @outY's S2MM -- the DMA waits for
                        two releases per wave and the core issues one -- and the
                        decode hung in wave 0 with the KV written and the layer
                        output never landing. That is the same signature the four
                        earlier attempts at this had, and this is what it was.
                        One body, one op per role, credit 1: no arm can perturb
                        the other's locks because there is only one of each op.

                        What the arm selects, then, is a COUNT and never a
                        program:

                          @xnorm re-feeds   XN_REFEED / REFEED[GATEUP] for
                                            decode, VOCAB_RNDS / 0 for the LM
                                            head. Dynamic loop bounds are fine
                                            here -- @xnorm is not the demux, so
                                            nothing needs its volume statically.
                          the residual      the two _accumulate passes and the
                                            layer-out are decode-only, under an
                                            scf.if. Their @outY gets keep STATIC
                                            bounds, which the demux id analysis
                                            does need.

                        The batch layout, unchanged:

                          rmsb   raw X -> h -> layer output. Never copied,
                                 never duplicated; the projections' outputs are
                                 added into it a round at a time.
                          rstg   the only thing that moves whole -- one @xnorm
                                 chunk out, or one projection round in.
                          rscl   one f32 per row, so the normalized batch can
                                 be REGENERATED chunk by chunk instead of
                                 stored.

                        The cost is that a chunk is recomputed once per
                        re-broadcast round rather than once per token. That is
                        a multiply pass over BATCH*K per round on a core whose
                        alternative is not fitting at all; whether it lands on
                        the critical path is a measurement nobody has taken.
                        """
                        assert RMS_DEST >= 0, (
                            "DECODE_BATCH>1 needs the FULL4 residual path "
                            "(RMS_DEST >= 0); the debug configs feed no o-proj"
                        )
                        assert K % XCHUNK == 0

                        def _i32(v):
                            return arith.ConstantOp(
                                IntegerAttr.get(i32, v), None
                            ).result

                        def _cnt(voc, dec, ex=None):
                            """An arm-selected loop bound.

                            `ex` is the per-extra-wave value, one per wave; the
                            arm is 2+k for extra wave k. Only ever a COUNT --
                            every arm runs the same ops over the same buffers.
                            """
                            if not N_EXTRA:
                                return index_switch(
                                    [idx_t],
                                    arith.index_cast(idx_t, _arm),
                                    [0],
                                    case_body_builder=lambda op, i, cv: yield_(
                                        [idx(voc)]
                                    ),
                                    default_body_builder=lambda op: yield_([idx(dec)]),
                                )
                            assert ex is not None and len(ex) == N_EXTRA
                            if len(set(ex)) == 1:
                                # One case per wave costs core .data -- see the
                                # same inversion in the proj core's _sel.
                                return index_switch(
                                    [idx_t],
                                    arith.index_cast(idx_t, _arm),
                                    [0, 1],
                                    case_body_builder=lambda op, i, cv: yield_(
                                        [idx(voc if cv == 0 else dec)]
                                    ),
                                    default_body_builder=lambda op: yield_(
                                        [idx(ex[0])]
                                    ),
                                )
                            return index_switch(
                                [idx_t],
                                arith.index_cast(idx_t, _arm),
                                [0] + [2 + k for k in range(N_EXTRA)],
                                case_body_builder=lambda op, i, cv: yield_(
                                    [idx(voc if cv == 0 else ex[cv - 2])]
                                ),
                                default_body_builder=lambda op: yield_([idx(dec)]),
                            )

                        def _if_decode(body, decode_only=False):
                            """Run `body` on the decode arm (and extra waves).

                            air.arm_select marks this as an ARM select rather
                            than ordinary control flow, so a static analysis
                            counts the `then` region and not both. Without it
                            check_channel_balance.py reads the LM head's traffic
                            as extra decode traffic and calls @outY unbalanced
                            -- the same thing it already does for the
                            index_switch arms, which it knows about by op name.

                            `decode_only` narrows it from "not the vocab arm" to
                            "the decode arm", which is how an extra wave skips a
                            residual pass it has no rounds for. A PREDICATE, not
                            a dynamic bound: the @outY gets inside keep their
                            STATIC trip counts, which the demux id analysis
                            needs (see _accumulate). Making the count dynamic
                            instead would take that away.
                            """
                            _pred = (
                                arith.cmpi(arith.CmpIPredicate.eq, _arm, _i32(1))
                                if decode_only
                                else arith.cmpi(arith.CmpIPredicate.ne, _arm, _i32(0))
                            )
                            _if = IfOp(_pred, [], has_else=False)
                            _if.operation.attributes["air.arm_select"] = UnitAttr.get()
                            with InsertionPoint(_if.thenRegion.blocks[0]):
                                body()
                                yield_([])

                        xb = AllocOp(
                            rstg_l1 if RMS_BAND_STREAM >= 3 else rmsb_l1, [], []
                        )
                        # THE @layerOut DRAIN GETS ITS OWN BAND BUFFER, and
                        # that is a LOCK requirement, not tidiness. A tile-DMA
                        # BD's acquire/release COUNT comes from
                        # air::getLockValuePair(targetModel, buffer) --
                        # AIRToAIEPass.cpp, the two-argument overload -- which
                        # is ceil(reads/writes) over the memcpy ops naming that
                        # buffer, while the CORE side is hardwired to 1. Share
                        # one band buffer for both roles and `xb` is the dst of
                        # every rmsX get (6) and the src of both layerOut puts
                        # (2), so the layerOut MM2S BD is emitted as
                        # acquire>=3 / release 3 against a core that releases
                        # 1 per band: the DMA never fires, the core blocks on
                        # its second band, and the dispatch times out with
                        # nothing written at all. (Measured, qwen3-4b batch 8:
                        # lock_2_2_128 init 0, DMA AcquireGreaterEqual 3, core
                        # Release 1.) Splitting the roles makes BOTH 1:1 --
                        # `xb` is written by 6 gets and read by no memcpy at
                        # all (the ratio path is skipped outright for a
                        # write-only buffer), `xr` is read by 2 puts and
                        # written by none -- so every count on this tile is 1,
                        # matching the core. residual_acc_row_banded_out_aie
                        # is what keeps them apart: out = acc + x rather than
                        # acc += x. Keeping `xb` the ONLY rmsX destination
                        # also keeps that channel's count-free BD ring uniform
                        # -- see the kernel's comment.
                        xr = AllocOp(rstg_l1, [], []) if RMS_BAND_STREAM >= 3 else None
                        # _RMS_STG_COPY's landing-buffer shadow -- see there.
                        sg2 = AllocOp(rstg_l1, [], []) if _RMS_STG_COPY else None
                        if RMS_BAND_STREAM >= 3:
                            # No upfront fetch: xb is refilled fresh inside
                            # _rms_batched_norm's scale pre-pass (the first
                            # thing _emit_norm below does), again at every
                            # regen visit, and again at every residual1/
                            # residual2 round -- see RMS_BAND_STREAM's
                            # level-3 comment and _rms_band_get.
                            pass
                        elif RMS_BAND_STREAM >= 2:
                            # NOT WORKING YET -- three variants tried, see "Level
                            # 2: two failures, not yet a fix" in
                            # docs/DFlashFeasibility.md before changing this again.
                            #
                            # xb is still full-size here -- this is ONLY meant to
                            # test that the launch side's matching sequence
                            # (_uni_dec) lands data at the right offsets, not
                            # shrinking anything yet.
                            #
                            # Variant A (a scf.for of 1 get): AIR folds a
                            # static-trip-count scf.for into ONE 3-D BD (confirmed
                            # in air_project/placed.air.mlir: one
                            # `air.channel.get ... [4, 8, 512] [512, 2048, 1]`)
                            # against the launch side's four separate, Python-
                            # unrolled `air.channel.put`s. NO hang, WRONG DATA
                            # (0/8 accepted instead of 8/8, plausible-looking
                            # garbage) -- 1 op against 4 on a packet channel did
                            # not raise an error anywhere.
                            #
                            # Variant B (Python-unrolled to 4 gets, matching the
                            # launch side's op count 1:1): HUNG
                            # (ERT_CMD_STATE_TIMEOUT); recovered cleanly after
                            # restoring the shipping templates, so this is the
                            # design, not a wedged rig.
                            #
                            # Read AIRToAIESchedulingUtils.cpp before trying B: the
                            # `_rms_batched` docstring's lock-credit hazard
                            # (op-instance counting, `getLockValuePair`'s
                            # no-air::ChannelOp overload) applies to MEMTILE
                            # buffers. `xb` is an L1 (compute-tile) buffer, which
                            # goes through the OTHER overload (takes an
                            # `air::ChannelOp`) -- that one counts DISTINCT
                            # BUFFER INSTANCES touched, not textual op count, and
                            # `xb` is one AllocOp reused by every band either way.
                            # So B's credit is provably identical to A's (1, not
                            # 4) -- going 1-op-to-4-op did NOT change xb's lock
                            # value, and B's hang is not that mechanism. No
                            # confirmed cause found for B; a real but unconfirmed
                            # lead is that BD chains are grouped per PHYSICAL DMA
                            # channel, shared with rmsW/rmsW2/the o-proj-down
                            # relay on the same S2MM0 port -- growing rmsX from 1
                            # BD to 4 changes its position/length inside that
                            # SHARED chain relative to the other three channels'
                            # BDs, which A (still 1 BD) does not.
                            #
                            # Variant C: neither side loops. The consumer
                            # already gets folded to ONE static 3-D op by AIR
                            # regardless (that's A); this makes the PRODUCER a
                            # single static 3-D op too, matching op count
                            # 1-vs-1 instead of 1-vs-4 -- not a scf.for (so
                            # the shim-deadlock rule doesn't apply), just one
                            # ChannelGet call with a 3-D shape, the same
                            # pattern _xnorm_put already uses with no loop at
                            # all. RESULT: WRONG DATA, BYTE-IDENTICAL to
                            # variant A's wrong logits (same first-mismatch
                            # token values at every position) -- two producer
                            # op counts (1 and 4) against the same consumer
                            # shape gave the IDENTICAL wrong answer, which
                            # rules OUT op-count/synchronization as the cause.
                            #
                            # A follow-up investigation traced the actual
                            # dimension-order semantics through both the L1
                            # and L3 lowering paths (AIRToAIESchedulingUtils
                            # .cpp's getWrapsAndStrides, AIRLoweringPass.cpp,
                            # AIRRtToNpuPass.cpp, and mlir-aie's own dimension-
                            # reversal code in AIERT.cpp/AIEDMATasksToNPU.cpp)
                            # and found dim 0 = outermost consistently on BOTH
                            # ends, matching this code's own assumption and
                            # ruling that theory out too.
                            #
                            # THE ACTUAL CAUSE (variant D fixes it, below): a
                            # core (compute) tile's DMA BD has an 8-BIT wrap
                            # field PER DIMENSION (AIETargetModel.h,
                            # getDmaBdWrapBits: 8 for core tiles, 10 for mem/
                            # shim) once sizes/strides are given explicitly --
                            # this is the SAME limit `_rms_batched_norm`'s own
                            # comment already names for a different channel
                            # ("a compute tile's wrap field is 8 bits, so the
                            # 3-D chunk-major form the MEMTILE producers use
                            # would not be legal here"). Every variant above
                            # used STG_W (512) as a BD dimension's extent --
                            # 512 does not fit in 8 bits. A truncated/wrapped
                            # size field is silently accepted (no verifier
                            # error) and produces a deterministic but wrong
                            # hardware iteration count -- consistent with A
                            # and C's identical wrong data, and plausibly with
                            # B's hang too (B's 2-D per-op shape [BATCH,
                            # STG_W] still has 512 as a wrap-encoded extent).
                            assert K % _RMS_DMA_CHUNK == 0
                            ChannelGet(
                                "rmsX",
                                xb,
                                offsets=[0, 0, 0],
                                sizes=[K // _RMS_DMA_CHUNK, BATCH, _RMS_DMA_CHUNK],
                                strides=[_RMS_DMA_CHUNK, K, 1],
                            )
                        else:
                            ChannelGet("rmsX", xb, indices=[idx(0)])
                        stg = AllocOp(rstg_l1, [], [])
                        scl = AllocOp(rscl_l1, [], [])
                        w = AllocOp(rms_l1, [], [])
                        w2 = AllocOp(rms_l1, [], []) if POST_RMS else None

                        def _emit_norm(wbuf, nrefeed, ch="xnorm"):
                            _rms_batched_norm(xb, stg, scl, wbuf, nrefeed, _arm, ch)

                        def _accumulate(nrnds, stage, guard=None):
                            """Add a projection's output into the residual.

                            The projection egresses (round, token): round r is
                            a PAYLOAD-wide band of every token's row. So a
                            round lands whole in rstg and goes into rmsb at a
                            fixed offset inside each row -- no transpose, and
                            no K-wide landing buffer.

                            DECODE_ACC_STOP drops the ADD from `stage` on, and
                            keeps the GET: the residual stops advancing while
                            every channel stays balanced, so layerOut carries an
                            intermediate without moving a single transfer.
                            """
                            _add = not ACC_STOP or stage < ACC_STOP
                            _rof, _rsz, _rst = outy_tokmajor(PAYLOAD)

                            def _outy_get():
                                # De-interleave here, not at the memtile: the
                                # round arrives emitter-major so its packet
                                # header stays in one piece upstream.
                                ChannelGet(
                                    "outY",
                                    stg,
                                    indices=[idx(0), idx(RMS_DEST)],
                                    offsets=[idx(v) for v in _rof],
                                    sizes=[idx(v) for v in _rsz],
                                    strides=[idx(v) for v in _rst],
                                )

                            # _RMS_BAND_FIRST: ask for the @rmsX band BEFORE the
                            # @outY round instead of after. The two arrive on
                            # different S2MM ports of the same tile but share the
                            # switchbox path into it, so taking outY first and
                            # then blocking on the band can leave the band's
                            # packet queued behind an outY packet the core cannot
                            # accept -- head-of-line, and the core waits forever.
                            _band_first = RMS_BAND_STREAM >= 3 and _os.environ.get(
                                "_RMS_BAND_FIRST"
                            )
                            for _r in for_(idx(0), idx(nrnds), idx(1)):
                                # GUARD: an extra wave egresses its OWN i2 rounds, which is not
                                # OPROJ_RNDS. The bound cannot simply become that count -- an
                                # arm-selected trip count here fails air-to-aie with 'selects a
                                # destination, but its routing domain has no demux', because the
                                # demux id analysis reads the @outY gets' static extent. So the
                                # loop stays statically OPROJ_RNDS and the BODY is skipped past
                                # the wave's round count: one get op site, one static count, and
                                # only the number of executions varies.
                                _g = (
                                    IfOp(
                                        arith.cmpi(arith.CmpIPredicate.ult, _r, guard),
                                        [],
                                        has_else=False,
                                    )
                                    if guard is not None
                                    else None
                                )
                                if _g is not None:
                                    _g.operation.attributes["air.arm_select"] = (
                                        UnitAttr.get()
                                    )
                                _gctx = (
                                    InsertionPoint(_g.thenRegion.blocks[0])
                                    if _g is not None
                                    else contextlib.nullcontext()
                                )
                                with _gctx:
                                    _do_get = str(stage) in _RMS_BANDS_GET
                                    _do_put = str(stage) in _RMS_BANDS_PUT
                                    # _RMS_RES2_PUT_FIRST: a PROGRESS WITNESS, not a
                                    # fix. In residual2 only, put a band back as the
                                    # FIRST op of the loop body -- ahead of the @outY
                                    # get as well as the @rmsX get. Its contents are
                                    # stale garbage, but the drain is host-visible,
                                    # so reaching this loop at all lands one band in
                                    # X (BATCH runs of STG_W, one per token row) and
                                    # X staying empty means the core never got here.
                                    # Everything else the core does after ph2 is
                                    # invisible from the host.
                                    # ..._PUT_FIRST puts it ahead of the @outY get
                                    # too; ..._PUT_MID puts it BETWEEN the @outY get
                                    # and the @rmsX get. One band in X under MID and
                                    # none under neither says @outY arrived and the
                                    # BAND did not; none under MID says @outY is
                                    # what blocks. Nothing else separates the two.
                                    _pm = _os.environ.get("_RMS_RES2_PUT_MID")
                                    _pf = _os.environ.get("_RMS_RES2_PUT_FIRST")
                                    _wit = (_pf or _pm) and _do_put and stage == 2
                                    if _pf and _wit:
                                        _rms_band_put_back(xr)
                                    if _band_first and _do_get:
                                        _rms_band_get(xb)
                                    _outy_get()
                                    if _pm and _wit:
                                        _rms_band_put_back(xr)
                                    if _RMS_STG_COPY:
                                        # `stg`'s ONLY reader in this round, so AIR
                                        # releases the @outY landing buffer here
                                        # rather than after the accumulate.
                                        CallOp(
                                            band_copy_aie,
                                            [sg2, stg, _i32(BATCH * STG_W)],
                                        )
                                    if RMS_BAND_STREAM >= 3:
                                        # Length 0, not a dropped call: see the
                                        # comment below -- unchanged reasoning,
                                        # just no _off term to compute (the
                                        # band fetch already selects the right
                                        # data; see residual_acc_row_banded_aie).
                                        _n = _i32(PAYLOAD if _add else 0)
                                        # Round r IS band r (nrnds == nband,
                                        # asserted via STG_W == PAYLOAD): xb
                                        # holds no data between rounds, so the
                                        # band this round updates has to come
                                        # back from _x_in() every round, and the
                                        # updated band has to go back out before
                                        # ph2's scale pre-pass (residual1) or as
                                        # the final output (residual2) can
                                        # correctly re-read it. No scratch slot
                                        # -- see the HIDDEN_TAPS refusal above.
                                        if _do_get and not _band_first:
                                            _rms_band_get(xb)
                                        for t in range(BATCH):
                                            CallOp(
                                                residual_acc_row_banded_out_aie,
                                                [
                                                    xr,
                                                    xb,
                                                    sg2 if _RMS_STG_COPY else stg,
                                                    _i32(t),
                                                    _i32(STG_W),
                                                    _n,
                                                ],
                                            )
                                        if _do_put and not _wit:
                                            _rms_band_put_back(xr)
                                    else:
                                        _off = arith.muli(
                                            arith.index_cast(i32, _r), _i32(PAYLOAD)
                                        )
                                        # Length 0, not a dropped call: the kernel's loop
                                        # runs zero times, so nothing is added -- but the
                                        # get still has a reader, and without one AIR is
                                        # free to move it and the channel stops balancing.
                                        # (It did, and the dispatch hung.)
                                        _n = _i32(PAYLOAD if _add else 0)
                                        for t in range(BATCH):
                                            CallOp(
                                                residual_acc_row_aie,
                                                [xb, stg, _i32(t), _off, _n],
                                            )
                                    if _g is not None:
                                        yield_([])
                                yield_([])

                        def _layer_out():
                            if RMS_BAND_STREAM >= 3:
                                # Already written, band by band, by
                                # residual2's own _rms_band_put_back calls
                                # inside _accumulate -- there is no whole-
                                # row xb left to put here.
                                return
                            ChannelPut(
                                "layerOut",
                                xb,
                                offsets=[idx(0)],
                                sizes=[idx(BATCH * K)],
                                strides=[idx(1)],
                            )

                        # ph0: input layernorm -> the QKV X feed. On the LM
                        # head arm this is the FINAL norm (model.norm.weight,
                        # fed in rmsW's slot -- see _uni_voc) and the count is
                        # one re-feed per vocab egress round, which is what
                        # _xc_voc sizes the X memtile's get loop for. N_NORMS>=4
                        # (the Gemma sandwich, which would need rms_chunk_hi_aie
                        # here) is already refused at BATCH>1 upstream.
                        if RMS_W_ON_X and RMS_BAND_STREAM < 3:
                            # Bisection shape (_RMS_W_ON_X at level <3): the
                            # weight goes straight into its own buffer, no band
                            # and no copy. Same channel, same ordering rule.
                            def _w_row_get(dst):
                                ChannelGet(
                                    "rmsX",
                                    dst,
                                    offsets=[0, 0],
                                    sizes=[K // _RMS_DMA_CHUNK, _RMS_DMA_CHUNK],
                                    strides=[_RMS_DMA_CHUNK, 1],
                                )

                            _w_row_get(w)
                            if POST_RMS:
                                _w_row_get(w2)
                            _emit_norm(w, _cnt(*_PH0_SWEEPS), _RMSROW0)
                        elif RMS_W_ON_X:
                            # BOTH weights first, back to back, before anything
                            # else touches the band buffer -- and in the SAME
                            # order the launch side puts them, because they now
                            # share one channel where they used to have two of
                            # their own (RMS_W_ON_X). `xb` is overwritten by the
                            # very next fetch, so each has to be lifted out
                            # before the following get, which is what makes
                            # "get, copy, get, copy" the only legal shape here.
                            #
                            # rmsW2 moves UP from its old position after ph0.
                            # The reason it sat there -- rmsW2 packet-muxed onto
                            # the same S2MM as the o-proj id, and a packet whose
                            # BD is not armed yet blocks the port behind it --
                            # no longer applies: there is no rmsW2 flow.
                            _rms_band_get(xb)
                            CallOp(band_to_weight_aie, [w, xb, _i32(K)])
                            if _os.environ.get("_RMS_W3"):
                                # Bisection: a THIRD straight-line, weight-shaped
                                # get at the very top, consumed immediately, with
                                # every other band fetch off (_RMS_FETCH_OFF).
                                # Separates "three transfers on @rmsX is the
                                # limit" from "the third transfer is fine but the
                                # pre-pass position is not".
                                _rms_band_get(xb)
                                CallOp(band_to_weight_aie, [w, xb, _i32(K)])
                            if POST_RMS:
                                _rms_band_get(xb)
                                CallOp(band_to_weight_aie, [w2, xb, _i32(K)])
                            _emit_norm(w, _cnt(*_PH0_SWEEPS), _RMSROW0)
                        else:
                            ChannelGet("rmsW", w, indices=[idx(0)])
                            _emit_norm(w, _cnt(*_PH0_SWEEPS), _RMSROW0)
                            if POST_RMS:
                                # Swap in the post-attention weight BEFORE the
                                # first o-proj get, not after: rmsW2 packet-muxes
                                # onto the same S2MM as the o-proj id, and a
                                # packet whose BD is not armed yet blocks the
                                # port behind it.
                                # Its OWN buffer, not `w` reused. Reusing it
                                # saves 4 KB on a tile that has them, and it
                                # costs the separate lock pair AIR gives the
                                # second weight -- which serialises the ph0 norm
                                # against the rmsW2 fill and hangs the batched
                                # decode in wave 0.
                                # On the LM head arm this weight is a DUMMY and
                                # the get exists only to consume it: rmsW2 is
                                # decode-only but packet-muxes onto the same shim
                                # MM2S as the vocab-active rmsX, so a hole in that
                                # group stalls the tail (see _uni_voc).
                                ChannelGet("rmsW2", w2, indices=[idx(0)])
                        # residual1: h = x + o-proj, in place.
                        _if_decode(
                            lambda: _accumulate(
                                OPROJ_RNDS,
                                1,
                                guard=(
                                    _cnt(OPROJ_RNDS, OPROJ_RNDS, EXTRA_RES1_RNDS)
                                    if N_EXTRA
                                    else None
                                ),
                            ),
                        )
                        # ph2: pre-MLP layernorm of h -> the gate-up X feed.
                        # Zero re-feeds on the LM head arm: there is no second
                        # projection phase there, and a zero-trip scf.for emits
                        # no @xnorm chunk while keeping the put op -- and the op
                        # is what the lock credit counts.
                        _emit_norm(
                            w2 if POST_RMS else w,
                            _cnt(*_PH2_SWEEPS),
                            _RMSROW2,
                        )
                        # residual2: layer output = h + down, in place.
                        _if_decode(
                            lambda: _accumulate(DOWN_RNDS, 2), decode_only=bool(N_EXTRA)
                        )
                        _if_decode(_layer_out)
                        for _b in (
                            ([w2] if POST_RMS else [])
                            + ([xr] if xr is not None else [])
                            + ([sg2] if sg2 is not None else [])
                            + [w, scl, stg, xb]
                        ):
                            DeallocOp(_b)

                    def _rms_decode_body(_arm):
                        if N_NORMS >= 4:
                            # ===== Gemma sandwich (4 norms) =====================
                            # input / post_attn / pre_ffn / post_ffn. The two "post"
                            # norms are applied to the SUBLAYER OUTPUT (o-proj, down)
                            # BEFORE the residual add (HF Gemma3; validated numpy ref):
                            #   h    = x + post_attn_norm(o_proj)
                            #   res2 = h + post_ffn_norm(down)
                            # The 4 norm weights are packed 2-per-channel to keep the rms
                            # tile's S2MM0 at <=4 packet ids (1 arbiter x 4 msels -> a port
                            # demuxes at most 4; 6 silently deadlocks). g_wa=rmsW=[input|
                            # post_attn], g_wb=rmsW2=[pre_ffn|post_ffn] (each 2K), sliced by
                            # the lo/hi kernels. o-proj & down share g_sub, their norm-out
                            # g_subn; residual2 reuses g_x (dead after residual1).
                            g_x = AllocOp(rms_l1, [], [])
                            ChannelGet("rmsX", g_x, indices=[idx(0)])
                            g_wa = AllocOp(rms_w2k_l1, [], [])
                            ChannelGet("rmsW", g_wa, indices=[idx(0)])
                            g_wb = AllocOp(rms_w2k_l1, [], [])
                            ChannelGet("rmsW2", g_wb, indices=[idx(0)])
                            g_xn = AllocOp(rms_l1, [], [])
                            g_sub = AllocOp(rms_l1, [], [])  # o-proj, then down
                            g_subn = AllocOp(
                                rms_l1, [], []
                            )  # post-norm out (attn, then ffn)
                            g_h = AllocOp(rms_l1, [], [])
                            # step1: input_layernorm (g_wa lo) -> QKV X feed (ph0).
                            # Normalize once, then re-broadcast the resident result
                            # XN_REFEED times.
                            CallOp(rms_norm_lo_aie, [g_xn, g_x, g_wa, _arm])
                            refeed(
                                XN_REFEED,
                                lambda: _xnorm_put(g_xn, K),
                            )
                            # step2 (residual1): h = x + post_attention_norm(o_proj) [g_wa hi]
                            ChannelGet(
                                "outY",
                                g_sub,
                                indices=[idx(0), idx(RMS_DEST)],
                                offsets=[idx(0)],
                                sizes=[idx(OPROJ_RNDS * PAYLOAD)],
                                strides=[idx(1)],
                            )
                            CallOp(rms_norm_hi_aie, [g_subn, g_sub, g_wa, _arm])
                            CallOp(residual_add_aie, [g_h, g_x, g_subn])
                            DeallocOp(g_wa)
                            # step3: pre_feedforward_norm(h) [g_wb lo] -> GLU X feed (ph2).
                            # baked per-put refeed = GLU proj rounds that re-read X.
                            CallOp(rms_norm_lo_aie, [g_xn, g_h, g_wb, _arm])
                            refeed(
                                REFEED[GATEUP_PHASE],
                                lambda: _xnorm_put(g_xn, K),
                            )
                            DeallocOp(g_xn)
                            # step4 (residual2): res2 = h + post_feedforward_norm(down) [g_wb hi].
                            # reuse g_sub/g_subn, and g_x (dead) for result.
                            ChannelGet(
                                "outY",
                                g_sub,
                                indices=[idx(0), idx(RMS_DEST)],
                                offsets=[idx(0)],
                                sizes=[idx(DOWN_RNDS * PAYLOAD)],
                                strides=[idx(1)],
                            )
                            CallOp(rms_norm_hi_aie, [g_subn, g_sub, g_wb, _arm])
                            CallOp(residual_add_aie, [g_x, g_h, g_subn])
                            DeallocOp(g_h)
                            DeallocOp(g_sub)
                            DeallocOp(g_subn)
                            DeallocOp(g_wb)
                            ChannelPut(
                                "layerOut",
                                g_x,
                                offsets=[idx(0)],
                                sizes=[idx(DOWN_RNDS * PAYLOAD)],
                                strides=[idx(1)],
                            )
                            DeallocOp(g_x)
                            return
                        a_x = AllocOp(rms_l1, [], [])
                        ChannelGet("rmsX", a_x, indices=[idx(0)])
                        a_w = AllocOp(rms_l1, [], [])
                        ChannelGet("rmsW", a_w, indices=[idx(0)])
                        a_w2 = None
                        if POST_RMS:
                            # post_attention_layernorm weight (own channel).
                            a_w2 = AllocOp(rms_l1, [], [])
                            ChannelGet("rmsW2", a_w2, indices=[idx(0)])
                        # step1: input layernorm -> X feed (re-fed RMS_REFEED via xnorm)
                        a_xn = AllocOp(rms_l1, [], [])
                        CallOp(rms_norm_aie, [a_xn, a_x, a_w, _arm])
                        refeed(
                            XN_REFEED,
                            lambda: _xnorm_put(a_xn, K),
                        )
                        # a_w and a_xn are kept for the ph2 (gate-up) emission (step2).
                        if RMS_DEST < 0:
                            # debug configs: original single-step rms (no residual).
                            DeallocOp(a_x)
                        else:
                            # step2 (#4 residual1): h = input + o-proj output. The
                            # o-proj output (id4 -> RMS_DEST) is CONSUMED here (faithful
                            # rms cadence) not via the deadlocking memtile relay.
                            a_op = AllocOp(rms_l1, [], [])
                            # BD-COMPACTION TEST: single full-size get (the id-4 packet
                            # flow reassembles the OPROJ_RNDS 512-packets into one 2048
                            # dest BD) instead of OPROJ_RNDS per-round gets. Verifies
                            # air-to-aie emits 1 BD (vs re-expanding).
                            ChannelGet(
                                "outY",
                                a_op,
                                indices=[idx(0), idx(RMS_DEST)],
                                offsets=[idx(0)],
                                sizes=[idx(OPROJ_RNDS * PAYLOAD)],
                                strides=[idx(1)],
                            )
                            a_h = AllocOp(rms_l1, [], [])
                            # DECODE_ACC_STOP=1: h = x. The get above still runs,
                            # so the channel is balanced and the only thing that
                            # changes is what layerOut ends up carrying.
                            if ACC_STOP == 1:
                                CallOp(rms_copy_aie, [a_h, a_x])
                            else:
                                CallOp(residual_add_aie, [a_h, a_x, a_op])
                            DeallocOp(a_x)
                            DeallocOp(a_op)
                            # FAITHFUL ph2 (reproducer core_2_2 step2): gate-up X
                            # = rmsnorm(residual1) = rmsnorm(x + o-proj), emitted
                            # AFTER o-proj (gates phase order) on the SAME @xnorm
                            # channel as ph0, REUSING a_xn (single y buffer), with
                            # a REFEED[ph2]-trip (32) re-broadcast loop. This is
                            # the per-step single-channel re-feed (ph0 x6, ph2 x32)
                            # -- replaces the invented buf_ph2 memtile stand-in.
                            CallOp(
                                rms_norm_aie,
                                [a_xn, a_h, a_w2 if POST_RMS else a_w, _arm],
                            )
                            refeed(
                                REFEED[GATEUP_PHASE],
                                lambda: _xnorm_put(a_xn, K),
                            )
                            DeallocOp(a_xn)
                            DeallocOp(a_w)
                            if POST_RMS:
                                DeallocOp(a_w2)
                            # step3 (#4 residual2): res2 = h + down -> layer out.
                            a_dn = AllocOp(rms_l1, [], [])
                            # BD-COMPACTION TEST: single full-size get (packet reassembly).
                            ChannelGet(
                                "outY",
                                a_dn,
                                indices=[idx(0), idx(RMS_DEST)],
                                offsets=[idx(0)],
                                sizes=[idx(DOWN_RNDS * PAYLOAD)],
                                strides=[idx(1)],
                            )
                            a_r2 = AllocOp(rms_l1, [], [])
                            if ACC_STOP:
                                CallOp(rms_copy_aie, [a_r2, a_h])
                            else:
                                CallOp(residual_add_aie, [a_r2, a_h, a_dn])
                            DeallocOp(a_h)
                            DeallocOp(a_dn)
                            # BD-COMPACTION: single full-size layerOut put.
                            ChannelPut(
                                "layerOut",
                                a_r2,
                                offsets=[idx(0)],
                                sizes=[idx(DOWN_RNDS * PAYLOAD)],
                                strides=[idx(1)],
                            )
                            DeallocOp(a_r2)

                    rms_h = herd(name="rms", sizes=[1, 1], operands=[_arm_rms])(
                        _rms_body
                    )
                    rms_h.attributes["link_with"] = StringAttr.get("rms_residual.o")
                    rms_h.attributes["x_loc"] = IntegerAttr.get(T.i64(), RMS_PCOL)
                    rms_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

            # Emit the launch: single-layer (NLAYERS==1) with no scf.for and no IV
            # (byte-identical to the original single-layer design), or NLAYERS
            # dispatches wrapped in an AIR scf.for whose induction variable is threaded
            # in as the last launch operand so the per-layer DDR offsets are
            # loop-carried. The device inside the launch is identical either way --
            # only the runtime sequence (insts) grows.
            def _emit_wave_loop(lo, hi):
                if lo >= hi:
                    return
                for _iv in for_(idx(lo), idx(hi), idx(1)):
                    launch(
                        sizes=[1, 1],
                        operands=list(_fa) + [_iv],
                        attributes={"air.preserve_shim_dma_order": UnitAttr.get()},
                    )(launch_body)
                    yield_([])

            # ONE rolled loop, split or not. Peeling it per weight group was tried and
            # abandoned: each peeled launch carries its own segment, which multiplies
            # the device-level resources (segment symbols, locks, and packet ids --
            # the 5-bit dma_bd Packet ID field caps that at ~4 launches). The split
            # instead lives entirely in _feed_wcol's per-group index_switch.
            _emit_wave_loop(UNI_WAVE_LO, UNI_WAVE_HI)

    return build()


def run():
    import pyxrt as xrt

    module = build_module()

    # Emit-only hook: dump the built AIR MLIR and stop before the (expensive) NPU
    # compile. Used to byte-diff the IR across no-op refactors (e.g. the incremental
    # model-config parametrization) without an aircc/NPU build. Inert unless set.
    if _os.environ.get("FUSED_DECODE_EMIT_ONLY"):
        print(str(module))
        # The trace region's byte offset into arg2, for a harness that reads the
        # ABI off this same emit. Printed rather than recomputed there: the
        # offset comes off rms_l3's own extent. Only under a traced build, so
        # the byte-diff the gate does is untouched.
        if TRACE_SIZE:
            print(f"// TRACE_OFFSET_BYTES {TRACE_OFFSET_BYTES}")
        return 0

    # use_lock_race_condition_fix_v2: emit the reference-style daisy-chained locks for the
    # shared-L2 fan-in (group/main asymmetric gather) -- matches the reproducer's
    # serialized 4-writer chain (mem_0_1 lock_0_1->_159->...->_162). Without it AIR
    # emits a counting lock whose writer/reader counts mismatch -> deadlock.
    backend = XRTBackend(
        omit_while_true_loop=False,
        output_format="xclbin",
        kernel_name="MLIR_AIE",
        stack_size=STACK_SIZE,
        use_lock_race_condition_fix_v2=True,
        coalesce_shim_dma=bool(COALESCE),
        # DYNSEQ: the runtime sequence now holds a scalar, so the stream is built
        # per dispatch from the emitted header instead of read from insts.bin.
        emit_txn_cpp=bool(DYNSEQ),
        debug_ir=bool(_os.environ.get("FUSED_DECODE_DEBUG_IR")),
        trace_size=TRACE_SIZE,
        trace_offset=TRACE_OFFSET_BYTES,
        trace_tiles=TRACE_TILES,
        trace_ddr_id=2,  # arg2, the rms buffer -- see TRACE_SIZE
        # EVERY HERD IN THIS DESIGN IS PINNED with x_loc, at physical columns
        # 0 through 7. aircc's default anchor is 0 and it would be right on its
        # own -- but tracing shifts the anchor right by one to reserve a column,
        # which would put every col-0 pin outside the segment, and
        # air-place-herds answers that by dropping the pin and placing
        # automatically. Stating the anchor keeps the traced build and the
        # untraced build the same design, which is the only reason to trace it.
        col_offset=0,
    )
    print(
        f"[q4nx_decode] proj: M={M} K={K} {NCX}x{NCY}=16 cores, "
        f"8 cascade pairs, NPH={NPH} dests={DEMUX}"
    )
    art = backend.compile(module, output_binary_name="decode", insts="decode.insts.bin")
    print(f"[q4nx_decode] emitted {art.output_binary} + {art.insts}")
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    exit(run())
