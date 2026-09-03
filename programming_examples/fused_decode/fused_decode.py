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
    DmaMemcpyNd,
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
from air.dialects.scf import for_, yield_, index_switch, ParallelOp, ReduceOp, IfOp
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
    # Qwen2.5-3B-Instruct: same family as qwen2.5-7b, one third the width, and
    # the only entry with 2 kv heads -- so the only one whose attention herd is
    # 2 compute units in a single column rather than 4 across two.
    #   I2P = [M, K, 2*INTER, K]/512 = [2560, 2048, 22528, 2048]/512 = [5, 4, 44, 4]
    #   J2P = [K, DQ, K, INTER]/512  = [2048, 2048, 2048, 11264]/512 = [4, 4, 4, 22]
    # PAIR_ROWS=1 as for qwen2.5-7b: paired egress needs every phase output
    # divisible by 1024 and M=2560 is not.
    # LM head: VOCAB_SIZE_PADDED_FULL = ceil(151936/2048)*2048 = 153600 -> 4800
    # rowblocks; 16*VOCAB_I2 must divide it, so UNI_LM*VOCAB_I2 = 300, and
    # K/PAYLOAD = 4 must divide VOCAB_I2, leaving {4,12,20} under 2*VOCAB_I2<=63.
    # Of those only 12 runs: VOCAB_I2=20 (UNI_LM=15) satisfies every divisibility
    # rule above and still DEADLOCKS the vocab wave on device, the same way the
    # 1B default does on llama-3.2-3b. Driver MUST set VOCAB_CHUNK_I2=12.
    "qwen2.5-3b": dict(
        K=2048,
        M=2560,  # DQ+DK+DV = 2048+256+256
        DH_A=128,
        KV_PER_CU=1,  # 2 kv / 2 CU
        N_ATTN_CU=2,
        NPH=4,
        I2P=[5, 4, 44, 4],
        J2P=[4, 4, 4, 22],
        DEST=["rope", "rms", "glu", "rms"],
        GQA_SEG=8,  # ATTN_IMPL_1x8x1
        PAIR_ROWS=1,
        N_NORMS=2,
        HAS_QKV_BIAS=True,
        VOCAB_SIZE=151936,
        UNI_DEC=36,  # 36 decoder layers
        UNI_LM=25,  # VOCAB_CHUNK_I2=12
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
    # ===== LFM2-1.2B: a HYBRID conv-attention decoder, whole model ==========
    #
    # 16 layers in one wave loop, with the mixer chosen per wave by the arm
    # (0=lm-head, 1=ShortConv, 2=attention). Only 6 of the 16 layers are
    # attention; the other 10 are Lfm2ShortConv, a gated causal depthwise
    # convolution. Both mixers are placed on two vertically adjacent tiles of
    # the one design and the kernels branch on the arm internally.
    #
    # THE PHASE SCHEDULE IS UNIFORM. Both layer types run the same NPH=5 /
    # I2P=[3,3,2,16,2], so the proj, rms and glu cores need no per-arm
    # divergence at all and the weight/rms/Y slabs stay a plain `iv * SLAB`.
    # An attention layer only needs 3072 of ph0+ph1's 6144 rows, so its second
    # mixer phase is PADDING: +1,966,080 bf16 per attention layer, 23.6 MB per
    # token, +3.6% of layer weight traffic. That is the whole price of the
    # uniform schedule, and it buys back a per-arm phase count, per-arm trip
    # counts and an irregular weight-offset lookup -- each of which would have
    # to be threaded through every DMA in the launch.
    #
    # ATTN_LAYERS is the IRREGULAR schedule (gaps 2,2,1,1,1,0). Drive off the
    # list, never a modulo.
    "lfm2-1.2b": dict(
        K=2048,
        M=3072,  # PER MIXER PHASE: conv in_proj 6144 = 2x, attn QKV 3072 + pad
        DH_A=64,
        KV_PER_CU=2,
        N_ATTN_CU=4,
        NPH=5,
        I2P=[3, 3, 2, 16, 2],
        J2P=[4, 4, 4, 4, 16],
        DEST=["mix", "mix", "rms", "glu", "rms"],
        GQA_SEG=4,
        PAIR_ROWS=2,
        N_NORMS=2,
        HAS_QK_NORM=True,  # the attention layers carry per-head QK-norm
        VOCAB_SIZE=65536,
        UNI_DEC=16,  # ALL 16 layers
        UNI_LM=4,
        ATTN_LAYERS=(2, 5, 8, 10, 12, 14),
    ),
}
MODEL_NAME = _os.environ.get("DECODE_MODEL", "llama-3.2-1b")
MODEL = _MODELS[MODEL_NAME]

# ph0 egress consumer. "rope" = attention (RoPE -> KV append -> block attention);
# "conv" = LFM2 Lfm2ShortConv (gate -> causal depthwise k=3 -> gate), which needs
# NO KV cache, NO attention CUs and NO append channels -- its mixer output feeds
# out_proj directly. Anything else is unimplemented and must fail loudly rather
# than silently emit rope_compute over the wrong width.
CONV_MIXER = MODEL["DEST"][0].split("#")[0] in ("conv", "mix")
# HYBRID: both mixers in one design, chosen per wave by the arm. The conv mixer
# exists (CONV_MIXER) AND so does the whole attention subsystem (ATTN_SUBSYS) --
# rope, the flash-attention CUs, the KV cache and the append/readback channels.
# ATTN_LAYERS is the wave-index schedule; it is irregular, so it is a LIST.
ATTN_LAYERS = tuple(MODEL.get("ATTN_LAYERS", ()))
HYBRID_MIXER = bool(ATTN_LAYERS)
# Spell the rope LUT feed as an air.dma_memcpy_nd naming @ropeLUT and let
# air-dma-to-channel derive the shim put, instead of writing a put/get pair.
# Only the dedicated @rope herd carries RMS as an operand; a HYBRID build runs
# _rope_body on the conv mixer's stage tile, which does not, so that build keeps
# the hand-written pair.
ROPELUT_DMA = True
# Where the derived put belongs. On a non-hybrid the hand-written one sits ahead
# of the rms group; on a HYBRID the whole mixer feed block is deferred into the
# phase loop, to the last mixer phase, and the LUT lands immediately before
# @convW there. Naming @rmsX on a hybrid would move it ~40 slots earlier.
_ROPELUT_KW, _ROPELUT_ANCHOR = (
    ("hoist_after", "inKV_V") if HYBRID_MIXER else ("hoist_before", "rmsX")
)
# Same treatment for the rms WEIGHT feed. The rms herd has to carry RMS (and the
# wave index, for the decode arm's per-layer slab offset) for the DMA to name
# both endpoints. @rmsX and @ropeLUT are anchored to @rmsW, so once @rmsW is
# derived too the anchor chain is inW0c0 <- rmsW <- {rmsX, ropeLUT} -- which is
# why @rmsW anchors to a channel that stays hand-written rather than to @rmsX
# (that would close a cycle, and a cyclic chain has no correct hoist order).
# The anchor is @rmsW2 on a POST_RMS model, not @inW0c0. Every one of the four
# `if not RMSW_DMA:` suppression sites is immediately followed by a @rmsW2
# endpoint, so the hand-written order is `rmsX rmsW rmsW2 inW0c0` and the
# derived put belongs ahead of @rmsW2 -- @inW0c0 is simply the wrong neighbour,
# and it is the FOURTH occurrence of a channel that repeats once per wave.
# Off POST_RMS there is no @rmsW2 and @inW0c0 is right.
#
# This flag was off for several sessions because @inW0c0 hung qwen3_8b_q4nx,
# gemma3_4b_q4nx and qwen25_7b_q4nx while passing the other six -- deriving
# @rmsW collapsed the three shim tasks at slots 2-4 into one and pushed two
# @rmsW2 tasks ~70 slots later, the same shape as the @ropeLUT 66->28 move that
# hung lfm2. No predicate distinguished the two groups: POST_RMS, N_NORMS,
# ROPE_W_PER_LAYER, UNI_DEC and NGLU were each checked against the split and
# each refuted. There was no discriminator because the anchor, not the model,
# was wrong.
#
# With @rmsW2 the derived form is order-identical to the hand-written one on all
# ten models that share this builder -- same channel order at air-dma-to-channel
# and the same shim task order, task for task, at airrt-to-npu (95 to 287 tasks
# depending on the model), each measured at the LBUILD its own verify lit uses.
RMSW_DMA = True

# @rmsW2 is @rmsW's twin: same L3 weight BO, same shim MM2S, emitted one slot
# later. It is ported on the same terms. Its anchor is not the same, though --
# see _RMSW2_ANCHOR, which has to name whichever channel followed the
# hand-written put in THIS model's arm.
RMSW2_DMA = True
# The mixer -> CU broadcast exists exactly when a hybrid has a mixer. A get with
# no put is "channel op not in pairs" at emit time, so the decl, the put and the
# four gets all key off this one predicate.
#
# NOT straight to the convergent @xnorm X ring: that would be a FOURTH same-id
# producer on it, and four do not route (three is the budget -- see the @attnO
# note at the put site). The o-gather memtile is the ph1 producer for both arms
# and its only input is @attnO, so the mixer feeds the CUs and they choose.
MIX_TO_CU = HYBRID_MIXER
# Does this build have an attention subsystem at all? Previously this was
# exactly `not CONV_MIXER`; a hybrid has both, so the two concepts split here.
ATTN_SUBSYS = HYBRID_MIXER or not CONV_MIXER
assert not HYBRID_MIXER or CONV_MIXER, "a hybrid needs the conv mixer too"
# The single decode arm a NON-hybrid build has. A hybrid has both and picks per
# wave; everyone else is one kind, decided at build time.
DEC_ARM_KIND = 1 if CONV_MIXER else 2


def _arm_only(arm_i, kinds, body, in_dec=False):
    """Emit `body` only on the arm kinds listed; idle otherwise.

    Arms mirror the reference's IS_ATTN RTP: 0 = lm-head wave, 1 = conv layer,
    2 = attention layer.

    The switch is always ONE case (idle) plus a default (run), never a list of
    idle arms -- so the 3-valued arm is first reduced to a 0/1 predicate for
    this gate. That shape is not cosmetic: an scf.index_switch with more than
    one CASE region breaks air-to-aie's L2 receiver allocation, which then
    reports the failure on the far side of the flow as "'air.channel.put' op
    failed to get S2MM tile for L3 allocation" on a shim put that is itself
    fine. Every gate in the shipped designs is single-case; measured directly
    -- same design, two idle cases fails, one idle case builds.

    `in_dec` says the caller is ALREADY inside the decode arm (e.g. within
    _uni_dec, or a herd body's decode region), so arm 0 cannot reach here. It
    only matters for a non-hybrid build: with one decode arm known at build
    time the kind test is static, but the vocab gate is not, so outside the
    decode arm the original decode-vs-vocab switch still has to be emitted --
    dropping it is what silently ran the attention memtiles during the LM-head
    waves.
    """
    if not HYBRID_MIXER:
        if DEC_ARM_KIND not in kinds:
            return
        if in_dec:
            body()
            return

        def _v(op, i, cv):
            yield_([])

        def _d(op):
            body()
            yield_([])

        index_switch([], arm_i, [0], case_body_builder=_v, default_body_builder=_d)
        return

    _pred = None
    for _k in sorted(kinds):
        _c = arith.cmpi(
            arith.CmpIPredicate.eq, arm_i, arith.ConstantOp.create_index(_k)
        )
        _pred = _c if _pred is None else arith.ori(_pred, _c)
    _sel = arith.select(
        _pred,
        arith.ConstantOp.create_index(1),
        arith.ConstantOp.create_index(0),
    )

    def _idle(op, i, cv):
        yield_([])

    def _run(op):
        body()
        yield_([])

    index_switch([], _sel, [0], case_body_builder=_idle, default_body_builder=_run)


# ShortConv channel count. in_proj emits [B | C | X], so CONV_DIM = in_proj/3.
# in_proj is split over several mixer phases ("lfm2-1.2b" uses two), in which
# case M is the PER-PHASE width and the full in_proj is their sum.
# A mixer may occupy several phases. Naming them "<mixer>" and "<mixer>#<n>"
# puts each on its OWN demux destination; naming them all "<mixer>" shares one.
# Both are supported and both were measured -- neither on its own fixes the
# full-width deadlock (the two adjacent tiles below are what does).
MIXER_NAME = MODEL["DEST"][0]


def _is_mixer_dest(name):
    return name == MIXER_NAME or name.startswith(MIXER_NAME + "#")


N_MIX_PH = sum(1 for d in MODEL["DEST"] if _is_mixer_dest(d))
CONV_IN = MODEL["M"] * N_MIX_PH if CONV_MIXER else 0  # full in_proj width
CONV_DIM = CONV_IN // 3 if CONV_MIXER else 0
# ShortConv cores. The reference splits the mixer over FOUR cores at D/4 each,
# and it has to: one core holding the whole layer needs
#   [B|C|X] 3*D + [w0|w1|w2|BX(t-2)|BX(t-1)] 5*D + new state 2*D + out D = 11*D
# elements of L1, which at D=2048 is 44 KB before AIR ping-pongs the
# channel-attached buffers -- the linker scripts show the mixer tile left with
# 4 KB for stack and locals, and the design deadlocks. At D/4 per core the same
# footprint is 11 KB.
#
# The mixer runs on TWO VERTICALLY ADJACENT TILES, mirroring the reference.
#
#   stage tile  gets ph0's egress in CONV_WAVES landing-buffer-sized waves and
#               writes each into the assembled [B|C|X]; also holds the taps
#   conv  tile  owns the assembled [B|C|X] and the carried state, computes, and
#               emits the mixer output plus the new state
#
# The two big buffers -- the assembled 3*D input and the 3*D taps -- are
# SEGMENT-SCOPE L1 shared between the two herd tiles, so they cross the tile
# boundary as ordinary neighbour-memory loads and stores with no DMA at all.
# That is what the reference does (`conv_bcx_buffer` lives on its conv tile and
# is written by its rope core; `rope_buffer` lives on its rope tile and is read
# by its conv core), and it is the difference between the designs: every build
# that DMA'd the whole 3*D into one tile deadlocked at full width. Our
# attention path already uses the same mechanism for its shared scores buffer.
#
# The landing buffer is the ATTENTION ph0 width, so a conv layer's 3*D arrives
# in CONV_WAVES of it -- the granularity the reference uses, and the one the
# eventual single-binary IS_ATTN build needs.
CONV_WAVES = int(_os.environ.get("CONV_WAVES", "2")) if CONV_MIXER else 1
CONV_LAND = CONV_IN // CONV_WAVES if CONV_MIXER else 1  # D + DK + DV at waves=2
# Where the mixer's two adjacent tiles go. A CONV-ONLY build has no attention,
# so column 3 is free. A HYBRID does not: ATTN_CU_LOC puts flash-attention CU0
# on column 3 rows 2-3, exactly on top of the mixer. Put the hybrid's mixer in
# the rms column instead, with the STAGE tile on the row the rope core already
# occupied -- so the attention path keeps the tile and the routing it was
# validated with, and the conv tile is simply added above it.
# The hybrid sits at column 5: its mixer has to reach the four flash-attention
# CUs over @mixToCU, and from column 2 the pathfinder cannot route that on top of
# the existing loop-close traffic ("packet flow source ... could not be routed to
# destination (1, 1) DMA2"). Column 5 rows 4-5 are free -- only the GLU tile
# (5,3) is occupied -- and are adjacent to both CU columns and the o memtile.
MIX_PCOL = 5 if HYBRID_MIXER else 3
MIX_PROW = 4 if HYBRID_MIXER else 2
# "rope" = attention, "conv" = ShortConv only, "mix" = HYBRID (both, chosen per
# wave by the arm). Anything else is unimplemented and must fail loudly rather
# than silently emit one mixer's kernel over the other's width.
if MODEL["DEST"][0].split("#")[0] not in ("rope", "conv", "mix"):
    raise NotImplementedError(
        f"DECODE_MODEL={MODEL_NAME} wants ph0 mixer '{MODEL['DEST'][0]}'; this "
        "engine implements only 'rope', 'conv' and 'mix'."
    )
assert (
    MODEL["DEST"][0].split("#")[0] == "mix"
) == HYBRID_MIXER, (
    "a 'mix' ph0 dest needs ATTN_LAYERS, and ATTN_LAYERS needs a 'mix' dest"
)

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
# Attention's per-layer rope-weight payload: cos/sin LUT, plus q/k-norm or
# q/k/v-bias where the model has them.
ROPE_LUT_LEN = (
    (DH + MODEL["M"]) if HAS_QKV_BIAS else (3 * DH) if HAS_QK_NORM else ROPE_DIM
)  # 64 / 768 / 96 / 4736
# The conv mixer's per-layer payload: tap-major [3][CONV_DIM] depthwise taps.
CONV_W_LEN = CONV_IN if CONV_MIXER else 0  # 6144
# The per-layer STRIDE in the rms BO's rope region. A hybrid layer may be either
# kind, so the slab has to hold the larger payload and each kind reads its own
# prefix -- 16 x 6144 x 2B = 196 KB of DDR, which is nothing next to the 671 MB
# of weights. Non-hybrid builds keep the exact previous expression.
ROPE_W_LEN = (
    max(CONV_W_LEN, ROPE_LUT_LEN)
    if HYBRID_MIXER
    else CONV_W_LEN if CONV_MIXER else ROPE_LUT_LEN
)  # 64 / 768 / 96 / 4736 / 6144(conv, hybrid)
# Does rope_w DIFFER PER LAYER? Llama's is a single per-position cos/sin LUT shared
# by every layer, so one slab suffices; qk-norm (gemma/qwen3) and q/k/v-bias
# (qwen2.5) both append per-layer weights, so the RMS BO needs UNI_DEC slabs and
# the feed has to index the current wave's. Getting this wrong DEADLOCKS: the host
# writes UNI_DEC slabs and puts final_norm after them, so a device that sized the
# region for one slab reads final_norm from inside the rope region.
ROPE_W_PER_LAYER = HAS_QK_NORM or HAS_QKV_BIAS or CONV_MIXER
NUM_KV_HEADS = MODEL["N_ATTN_CU"] * MODEL["KV_PER_CU"]  # 8 / 4
NUM_Q_HEADS = (MODEL["M"] - 2 * NUM_KV_HEADS * DH) // DH  # 32 / 8
Q_HEADS_PER_CU = NUM_Q_HEADS // MODEL["N_ATTN_CU"]  # 8 / 2
DQ = (
    CONV_DIM if CONV_MIXER else NUM_Q_HEADS * DH
)  # q width 2048 / 2048 (conv: mixer out)
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
DQ_PADDED = (
    CONV_DIM if CONV_MIXER else MODEL["N_ATTN_CU"] * DQ_PADDED_PER_CU
)  # 2048 / 4096
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
DOWN_PCOL = 3 if MODEL["PAIR_ROWS"] == 1 else 4

# Phases (reproducer I2=[3,2,16,2], J2=[4,4,4,16], pkt=[1,4,8,4]). Phase 0 = QKV
# (id1), phase 1 = o-proj (id4), phase 2 = gate-up MLP (id8), phase 3 = DOWN (id4).
# Phases 0-2 contract over K=MODEL_DIM=2048 reading the rmsnorm'd token X; gate-up's
# 16384 output is consumed on-chip by the GLU tile (silu(gate)*up -> 8192), and that
# 8192 is fed back ON-CHIP as the DOWN phase X (K=INTERMEDIATE=8192) -> layer output
# 2048. Per-phase K differs: ph0-2 K=2048 (NBJ=8), ph3 K=8192 (NBJ=32).
import os as _os

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
ATTN_ROUNDS = (ATTN_L + 15) // 16
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
RB_ROUNDS = int(_os.environ.get("DECODE_RB_ROUNDS", str((ATTN_L + 15) // 16)))
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

# And the KV append. Both halves are visible in one place only if the KV cache
# reaches the rope herd, so KVC is threaded through the segment the way RMS and
# X are. Guarded on DYNSEQ_APPEND: with a runtime context length the append
# offset depends on the L RTP, which would have to be threaded in as well.
# ON for hybrids too, as of the KVC threading below. It was off for several
# sessions, and the reason recorded for that was wrong twice over.
#
# The stated reason was that `hoist_before="inKV_K"` lands 40 shim slots off on
# a hybrid. Forcing the flag True showed there was no misplaced BD because there
# was no DMA at all -- `@appendK: 1 put(s), 0 get(s), 0 dma(s)`. On a hybrid the
# rope body runs inside the CONV herd, called positionally as
# `_rope_body(_arm, _lands[0])` with no kvc, so the `kvc is not None` guard was
# false while the get suppression tested only APPEND_DMA. Fifth asymmetric port,
# third caught by the emit-time pairing assertion.
#
# Threading KVC and the layer index into conv_h fixes that (see _conv_kv_opers).
# What is left is a 4-line shim task move: @convStIn from slot 64 to 26 and the
# second wave's @ropeLUT from 66 to 30. Three experiments place it:
#
#   anchor removed entirely  -> @ropeLUT STILL moves, and the appends fall to
#                               the end of the queue. Anchor is innocent.
#   operands threaded, DMA   -> shim task order IDENTICAL, 114/114. Threading
#     suppressed                is innocent.
#   both                     -> the 4-line move.
#
# So it is the hoist of the external half out of conv_h, and it first appears
# inside air-to-aie: the channel op order is identical through pass 048 and
# diverges at 049. Not an anchoring problem, which is where two sessions of
# effort went.
#
# The move is BENIGN, measured rather than assumed: lfm2-1.2b -- the only model
# with HYBRID_MIXER -- verifies topk 2/0 through its own run_npu2_verify.lit,
# which is 32 greedy tokens x 2 prompts and would not survive a wrong KV append
# offset. The historical "device timed out at decode pos 8" was recorded against
# a build that could not emit the DMA, so it was never this design.
#
# For the record: `inKV_K` DOES repeat -- two launch-scope endpoints on lfm2,
# one per ph0 wave -- which is the property that made `inW0c0` the wrong anchor
# for @rmsW. But the hand-written `get @appendK` sits at launch-scope index 51
# and the first `put @inKV_K` at 53, so first-occurrence resolution is right.
APPEND_DMA = not DYNSEQ_APPEND
# The hybrid's conv state, the same way. @convStIn reads [BX(t-2)|BX(t-1)] out
# of the KV BO and @convStOut writes the shifted state back over the SAME slot,
# so both name KVC -- which already reaches conv_h for the ported append. Only
# a hybrid has them.
CONVST_DMA = HYBRID_MIXER
# And the depthwise taps. Same slab as @ropeLUT, read from the other end -- the
# LUT takes the cos/sin prefix, @convW the taps -- so the same RMS operand and
# the same rebuilt offset serve both.
CONVW_DMA = HYBRID_MIXER
# The GLU hidden slices. Two adjacent slots per loop iteration on one channel,
# which needs the loop fold to recognise a tiled contiguous run -- otherwise the
# derived consumer is NGLU descriptors where the hand-written one is a single
# fill, and the refeed MM2S's counting lock is off by that factor.
GLUOUT_DMA = True
# And rope's Q broadcast, the one L1 -> L2 feed. Its consumer is a SEGMENT-scope
# get on an L2 buffer, which air-dma-to-channel handles natively -- the only
# obstacle was that the L2 buffer is allocated after the rope herd, so it does
# not dominate it. Allocating it up front is the whole change; no new op.
ROPEQ_DMA = True
TOATTNQ_DMA = True
ATTNO_DMA = True
# And the KV re-block feed. Its consumers sit inside the attn herd's inner
# scf.for, and its producer is a guarded segment-scope loop, so the derived
# put has to end up back in that loop -- air-dma-to-channel does that itself
# (it merges the rebuilt guard and fuses the loop), which is why this needs no
# anchor. The staging buffers move to segment scope so they dominate the herd;
# air-fuse-alloc-dealloc sinks them back into the loop afterwards.
TOKV_DMA = True
# And the proj egress gather: the 8 paired core emitters landing their row
# blocks in a group memtile. Three things the DMA spelling needs that a
# hand-written put did not:
#   - `dest`, the runtime packet-demux destination. This is the ORIGINATOR of
#     @outY's routing header; without it the demux has no header source.
#   - static `channel_indices`. The sub-channel is [logical col, pair] and the
#     column looks like a herd IV -- but inside the tx guard below it is a
#     COMPILE-TIME CONSTANT, so no runtime selector is needed and the derived
#     memtile get is the same constant-indexed op the design used to hand-write.
#   - a `hoist_before` anchor on @toMain, so the derived gets land back inside
#     each phase arm's round loop next to the put that forwards the assembled
#     buffer, rather than beside the herd.
# The group buffers move to segment scope so they dominate the herds, as
# @gluOut's did. The header asymmetry -- the first lead sends HDR+PAIR_PAY and
# the rest send PAIR_PAY -- costs nothing: a dma_memcpy_nd has independent
# source and destination extents, so it is one op either way.
#
# Paired emitters only. The non-paired path (PAIR_ROWS == 1, gemma3-4b and
# friends) is a different herd body with one group per COLUMN rather than two
# per block, so its emitter and its group-to-herd map both differ; porting it is
# its own change, and threading a buffer into a herd whose body does not take
# one is a TypeError at build time, not a compile error.
OUTA_DMA = True and MODEL["PAIR_ROWS"] != 1
# LAYEROUT_DMA: the rms core names @layerOut on an air.dma_memcpy_nd whose far
# operand is X, the L3 buffer it already holds for @rmsX. Only the DECODE arm:
# the vocab arm drains into Y, which is not a rms-herd operand.
#
# The anchor is @appendV, which has EXACTLY ONE launch-scope endpoint, in the
# slot immediately before this drain. That uniqueness is the whole reason this
# one is portable and @inKV is not -- an anchor names a channel and resolves to
# its LAST endpoint, so a channel that repeats cannot name a slot in the middle
# of its own run.
LAYEROUT_DMA = int(_os.environ.get("LAYEROUT_DMA", "1")) != 0
# INX_DMA: the cores name @inX on an air.dma_memcpy_nd whose far operand is the
# X memtile buffer, and air-dma-to-channel derives the memtile puts -- window,
# count and position all. The `_jj` sub-block loop in the feed goes away
# entirely: it existed only to step a window the compiler can read off the two
# buffer sizes (512 memtile / 256 core = two pieces, ascending, one lock event
# each). The buffer has to be a herd operand for that, which is why it is
# allocated once at segment scope instead of per feed round.
# Scoped off the non-paired proj body for the same reason OUTA_DMA is: that
# herd body does not take the extra operand, and threading one in is a
# TypeError at build time rather than a compile error.
INX_DMA = int(_os.environ.get("INX_DMA", "1")) != 0 and MODEL["PAIR_ROWS"] != 1
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
# the reference-faithful on-device KV append: the rope core writes this token's roped-K/raw-V
# into the DDR cache (appendK/appendV S2MM -> KVC at slot L-1 = the reference _receive_kv_cache),
# then the whole cache is read back for the block-loop attention (the reference _move_kv_cache).
# The append->readback RAW on the shared cache is ordered in the runtime sequence by
# air-annotate-append-barrier, which derives it from the shared L3 memref (= the
# reference's dma_wait). Only for MULTIBLK (L>1); L=1 uses the trivial on-chip-KV path.
# ---- port predicates -------------------------------------------------------
# One per ported feed, tested by BOTH the producer suppression and the consumer
# emission. Four separate device failures in this file came from a producer
# keyed on its flag alone while the consumer also required a buffer or a config
# term: the consumer correctly fell back to the hand-written channel op and the
# producer had already been deleted, leaving it unpaired (or, worse, compiling
# and hanging). Anything a consumer tests belongs here.
#
# The per-buffer half (`rms is not None`, `qmt is not None`, ...) cannot live
# here -- those are herd operands -- so each site ANDs it in. The static half
# must not be duplicated.
# The per-layer clause is gone: _rope_off_h rebuilds the slab offset from the
# wave index now, so ROPE_W_PER_LAYER is no longer a reason to keep the
# hand-written pair. _rope_body asserts if that index is ever missing, which
# turns a silently unpaired channel into a build error.
ROPELUT_DMA_OK = ROPELUT_DMA

# Which channel the derived @rmsW2 put has to land ahead of in a DECODE arm:
# whichever one followed the hand-written put there.
#
# @ropeLUT is emitted between them in the source, but it is DERIVED on every
# model now and anchored `hoist_after="rmsW"`, so it lands between @rmsW and
# @rmsW2 rather than after the pair. @inW0c0 -- the first weight put -- is the
# next fixed thing along, and it is the answer for the hybrid too, which has no
# @ropeLUT in this arm at all.
#
# Naming @ropeLUT instead was measured while the LUT was still hand-written on
# five models, and it is wrong now that it is not: it moves the whole rms group,
# because @rmsW chains onto @rmsW2 and @rmsX onto @rmsW.
_RMSW2_ANCHOR = "inW0c0"


KV_APPEND = MULTIBLK
# the reference layer-chaining ABI: the layer output (res2 = new hidden states) is written
# IN-PLACE into arg0 (the hidden_states BO), so layer N's output == layer N+1's input
# in the same buffer -- matching the reference's decoding_layer (output S2MM back to x_arg_id,
# no separate output arg). Frees arg3 (== the reference's rope_rms slot).
# Reference 4-CU layout: attn cols 3,4 (CU0,1 col3 / CU2,3 col4), adjacent to q/o on
# mem_5_1 (col5). kv on mem_3_1/mem_4_1. (col4 freed by GLU->col5 relayout.)
# A 2-CU model (2 kv heads) fills one column instead of two. Which column is not
# free: the placement decides the routing, and the routing decides whether the
# decode wedges. Swept on NPU2 with qwen2.5-3b -- cols 0/1/2/6/7 do not build
# (occupied), col 3 reaches COMPLETED about 1 run in 10, cols 4 and 5 are both
# 20/20 over 29 dispatches. 4 is taken: it is inside the footprint the 4-CU
# models already reserve for attention, where col 5 also carries the GLU tile.
# Overridable so the sweep can be repeated.
ATTN_PCOL = int(_os.environ.get("ATTN_PCOL", "4"))
ATTN_CU_LOC = (
    [(ATTN_PCOL, 2, 3), (ATTN_PCOL, 4, 5)]
    if N_ATTN_CU == 2
    else [(3, 2, 3), (3, 4, 5), (4, 2, 3), (4, 4, 5)][:N_ATTN_CU]
)
# Attn herd geometry: dim0 = columns spanned, dim1 = 2 rows per CU (qk, kv).
# 4 CUs span cols {3,4} -> [2, 4]; 2 CUs sit in one column -> [1, 4].
# Spreading 2 CUs one-per-column instead ([(3,2,3),(4,2,3)]) builds but times out
# at pos0 on every run: the k/v fan and the o-proj gather both assume a column's
# CUs are a contiguous pair, and one CU per column is not that.
ATTN_COLS = len({_l[0] for _l in ATTN_CU_LOC})
CU_PER_COL = N_ATTN_CU // ATTN_COLS
ATTN_HERD_SIZES = [ATTN_COLS, 2 * CU_PER_COL]
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
# Decode waves in the unified sequence. Overridable so a build can be shortened
# to N layers for a numerics BISECT -- the weight buffer is
# [UNI_DEC layer slabs | UNI_LM vocab waves], so a short build is a prefix of the
# full one and reuses its packed cache. Unset leaves every model's IR unchanged.
UNI_DEC = int(_os.environ.get("DECODE_UNI_DEC", MODEL["UNI_DEC"]))
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
UNI_WAVES = UNI_DEC + UNI_LM
# ATTN_LAYERS indexes DECODE waves, and the unified sequence continues past them
# into UNI_LM lm-head waves. A SHORT bisect build (DECODE_UNI_DEC below the
# model's depth) therefore leaves schedule entries pointing at wave indices that
# are no longer decode waves -- and _arm_of_wave would promote those to the
# attention arm, so that vocab chunk computes a layer instead of logits and its
# slice of Y is never written. Measured on device before this filter:
# DECODE_UNI_DEC=2 wrote 32768/65536 logit words, losing exactly the vocab waves
# at iv=2 and iv=5 (the two entries of (2,5,8,10,12,14) that fall in
# [UNI_DEC, UNI_WAVES)); DECODE_UNI_DEC=1 lost iv=2 alone -- the long-unexplained
# "fb_C writes 49152/65536". Every full-depth build has max(ATTN_LAYERS) <
# UNI_DEC, so this is a no-op for them and their IR is unchanged.
#
# HYBRID_MIXER deliberately keeps its ORIGINAL value: a short all-ShortConv
# reduction of a hybrid model still wants the hybrid machinery compiled in, it
# just has no attention wave to run it on.
ATTN_WAVES = tuple(_k for _k in ATTN_LAYERS if _k < UNI_DEC)
# Wave-range override (keeps ABI/CDO fixed at UNI_DEC/UNI_LM; only restricts which
# waves the fused launch loop drives). Used to split the fused sequence into a
# decode-part [0,UNI_DEC) and a vocab-part [UNI_DEC,UNI_WAVES) that share ONE CDO,
# to test host-wait quiescence between decode and vocab on one xclbin.
UNI_WAVE_LO = int(_os.environ.get("UNI_WAVE_LO", "0"))
UNI_WAVE_HI = int(_os.environ.get("UNI_WAVE_HI", str(UNI_WAVES)))

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
# How many of dest0's rounds carry the attention layer's QKV, and how many are
# the PADDING a hybrid's uniform phase schedule produces. An attention layer
# needs M = DQ+DK+DV of the mixer dest's total width; a conv layer needs all of
# it (in_proj is exactly N_MIX_PH x M). Non-hybrid builds have no padding, so
# MIX_RNDS_PAD is 0 and every expression below reduces to the previous one.
MIX_RNDS_QKV = M // PAYLOAD
MIX_RNDS_PAD = (ROUNDS_PER_DEST[0] - MIX_RNDS_QKV) if HYBRID_MIXER else 0

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
# emits I2*2 row-blocks per phase, so it reads K that many times.
REFEED = [I2P[p] * PAIR_ROWS for p in range(NPH)]  # [6, 4, 32, 4]
REFEED_TOTAL = sum(REFEED)  # all phases
# Two X sources: phases 0..2 read the rmsnorm'd token X (K=2048); the DOWN phase
# reads the GLU output (K=8192) fed back on-chip. Split the re-feed accordingly.
RMS_PHASES = [p for p in range(NPH) if p != DOWN_PHASE]
RMS_REFEED = sum(REFEED[p] for p in RMS_PHASES)  # rms-X whole-2048 re-reads (42)
DOWN_REFEED = REFEED[DOWN_PHASE] if DOWN_PHASE >= 0 else 0  # GLU-X 8192 re-reads (4)
# LOOPCLOSE: ph1 (o-proj) X = attn-o (separate channel), so @xnorm/rms-X covers only
# ph0 + ph2 (ph1 excluded). OPROJ_PHASE=1. XN_REFEED = REFEED[0]+REFEED[2].
#
# The phase indices are DERIVED from DEST_NAMES rather than hardcoded, so a model
# may split one projection across more than one phase and still reach the right
# o-proj / gate-up / down phases. "lfm2-1.2b" is the worked example (in_proj as
# two mixer phases); every other shipped model has one phase per projection and
# emits byte-identical IR either way.
MIXER_PHASES = [p for p in range(NPH) if _is_mixer_dest(DEST_NAMES[p])]
MIXER_DESTS = sorted({DEST[p] for p in MIXER_PHASES})  # one demux id per wave
OPROJ_PHASE = next(
    p
    for p in range(NPH)
    if p != DOWN_PHASE and p not in MIXER_PHASES and DEST_NAMES[p] == "rms"
)
GATEUP_PHASE = next((p for p in range(NPH) if DEST_NAMES[p] == "glu"), OPROJ_PHASE + 1)
# LOOPCLOSE convergent @xnorm: rms (compute, channel refeed) emits ONLY ph0
# (rmsnorm input); ph1 attn-o, ph2 a_xn, ph3 down are MEMTILE producers (mechanism-2
# per-buffer refeed) converging on @xnorm in phase-time order, read by ONE loop.
XN_REFEED = sum(REFEED[p] for p in MIXER_PHASES)

# Which proj phase carries the KV append + cache readback. It has to be the
# LAST mixer phase (see the use site): the append feeds off rope's K/V, and rope
# runs once the ph0 landings are in, so issuing the append earlier makes the
# shim block on K/V it has not caused to be produced yet. For every model with a
# single mixer phase that is phase 0, exactly as it was before the hybrid.
KV_PHASE = MIXER_PHASES[-1] if MIXER_PHASES else 0
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
GLU_PHASE = next((p for p in range(NPH) if DEST_NAMES[p] == "glu"), -1)
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
FULL4 = (
    GLU_PHASE >= 0
    and DEST[OPROJ_PHASE] == DEST[DOWN_PHASE]
    and DOWN_PHASE not in MIXER_PHASES
    and DEST[GLU_PHASE] != DEST[OPROJ_PHASE]
    and DEST[GLU_PHASE] not in MIXER_DESTS
    and DEST[OPROJ_PHASE] not in MIXER_DESTS
)
RMS_DEST = DEST[DOWN_PHASE] if FULL4 else -1
HOST_DRAIN = [p for p in range(NDEST) if p != GLU_DEST and p != RMS_DEST]
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
HOST_ROUNDS = sum(ROUNDS_PER_DEST[p] for p in HOST_DRAIN)  # host-drained egress rounds
# #4: the down egresses as the rms layer output (residual2), drained on its own channel.
LAYER_RNDS = (PAIR_ROWS * I2P[DOWN_PHASE]) if FULL4 else 0

# ===== Multi-layer fused decode (stitch NLAYERS runtime sub-sequences) =====
# The device (segment/herds) is emitted ONCE; only the launch-scope L3 feeds are
# emitted per layer, with COMPILE-TIME-CONSTANT per-layer DDR offsets. So the
# aie.device (-> xclbin) stays byte-identical to the single-layer build and only
# the runtime instruction sequence grows ("16 sub-sequences stitched one after
# another"). NLAYERS=1 is a strict no-op (all per-layer bases = 0).
NLAYERS = int(_os.environ.get("NLAYERS", "1"))
# Per-layer DDR slab sizes (elements). LUT is per-position (shared across layers),
# placed after all NLAYERS rms slabs.
W_LAYER = sum(NCX * PER_COL_PH[p] * BLOCK_BF16 for p in range(NPH))  # weights / layer
RMS_LAYER = N_NORMS * K  # rms weights / layer (2 llama pre-norm / 4 Gemma sandwich)
KV_LAYER = ATTN_MAXL * KVSZ_TOK  # KV cache / layer
# ShortConv carried state [BX[t-2] | BX[t-1]] per layer, also in arg4.
CONV_ST_LAYER = 2 * CONV_DIM if CONV_MIXER else 0
# A HYBRID puts the state region AFTER the whole KV region so BOTH stay a plain
# `iv * slab`. Each region then carries UNI_DEC slabs of which only its own layer
# kind's are ever touched -- ~42 MB of arg4 that is allocated and never read at
# ATTN_MAXL=2048. That is DDR footprint, not bandwidth, and it is the price of
# keeping every offset affine in the wave index.
CONV_ST_BASE = UNI_DEC * KV_LAYER if HYBRID_MIXER else 0
Y_LAYER = sum(
    ROUNDS_PER_DEST[p] * PAYLOAD for p in HOST_DRAIN if p not in MIXER_DESTS
)  # Y / layer


def _assert_channels_paired(module):
    """Every air.channel must have at least one producer and one consumer.

    A ported feed is spelled as an air.dma_memcpy_nd naming the channel, and
    air-dma-to-channel derives the other half -- so the hand-written op on that
    side has to be deleted, or the feed would be doubled. Deleting it under a
    weaker condition than the one the DMA is emitted under leaves the surviving
    half unpaired. Ten models import this builder with different configs, and
    that mistake has been made four times (@toAttnQ, @attnO, @appendK/@appendV,
    @ropeLUT), each time found only on the model it broke.

    aircc does catch it -- `'air.channel.get' op found channel op not in pairs`
    -- but tens of minutes later, pointing at a line of generated IR, naming
    neither the feed nor the flag. Catch it here, where the name is still known.
    """
    import re as _re
    from collections import Counter as _Counter

    text = str(module)
    puts = _Counter(_re.findall(r"air\.channel\.put[^@]*@([A-Za-z0-9_]+)", text))
    gets = _Counter(_re.findall(r"air\.channel\.get[^@]*@([A-Za-z0-9_]+)", text))
    # A DMA naming a channel supplies whichever half the pass will derive, so it
    # counts for both.
    dmas = _Counter(_re.findall(r"channel = @([A-Za-z0-9_]+)", text))
    bad = []
    for ch in sorted(set(puts) | set(gets) | set(dmas)):
        has_src = puts[ch] or dmas[ch]
        has_dst = gets[ch] or dmas[ch]
        if not (has_src and has_dst):
            bad.append(
                f"@{ch}: {puts[ch]} put(s), {gets[ch]} get(s), {dmas[ch]} dma(s)"
            )
    if bad:
        raise AssertionError(
            f"DECODE_MODEL={MODEL_NAME}: channel op not in pairs, at emit time:\n  "
            + "\n  ".join(bad)
            + "\nA port's producer suppression must test exactly what its "
            "consumer tests -- see the port predicates near KV_APPEND."
        )


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
        x_l3 = MemRefType.get([K], bf16)  # RAW input activation (in-place chain)
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
        # rms weight (K). MULTIBLK appends the rope region AFTER all UNI_DEC rms slabs so
        # the score-path test gets a KNOWN q (q_roped = proj_q) WITHOUT corrupting
        # rms_w[0:K] (which proj_q depends on). Llama: ONE shared per-position LUT
        # (ROPE_W_LEN=64). Per-layer models (ROPE_W_PER_LAYER): UNI_DEC rope_w slabs
        # (dual-theta + per-layer q/k-norm). L=1 ABI unchanged.
        rms_l3 = MemRefType.get(
            [
                UNI_DEC * RMS_LAYER
                + ((UNI_DEC if ROPE_W_PER_LAYER else 1) * ROPE_W_LEN if MULTIBLK else 0)
                + K  # dedicated final-norm slot for real lm_head (vocab)
            ],
            bf16,
        )
        # LM_HEAD drains VOCAB_SIZE_PADDED logits into Y (arg3); decode uses Y for the
        # QKV host rounds + rms layer-out. Separate compile-time size (decode unchanged).
        _y_elems = (HOST_ROUNDS + LAYER_RNDS) * PAYLOAD + UNI_LM * VOCAB_SIZE_PADDED
        y_l3 = MemRefType.get(
            [_y_elems], bf16
        )  # host-drain (QKV) rounds + LAYER_RNDS rms layer-out (down) rounds
        # MULTIBLK: DDR KV cache (the reference full-faithful append+readback). Layout
        # [ATTN_MAXL][K: DK_TOT_A | V: DK_TOT_A]; rope appends this token at
        # APPEND_OFF, then the whole cache is streamed back per CU (_d2wip shapes).
        # NLAYERS per-layer caches concatenated (offset iv*KV_LAYER).
        kvc_l3 = MemRefType.get(
            (
                [CONV_ST_BASE + UNI_DEC * CONV_ST_LAYER]  # conv state, after any KV
                if CONV_MIXER
                else [UNI_DEC * KV_LAYER]
            ),
            bf16,
        )

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
        rms_l1 = MemRefType.get([K], bf16, memory_space=l1)  # rms in/out/weight (2048)
        # Gemma 4-norm: two norm weights packed per channel (2K) so the rms tile
        # keeps <=4 packet ids per S2MM port (1 arbiter x 4 msels). lo/hi kernels
        # slice it. Only used when N_NORMS>=4.
        rms_w2k_l1 = MemRefType.get([2 * K], bf16, memory_space=l1)
        glu_x_l1 = MemRefType.get([GLU_SLICE], bf16, memory_space=l1)  # 1024 [up|gate]
        glu_hid_l1 = MemRefType.get([GLU_HID], bf16, memory_space=l1)  # 512 silu*up
        # ATTN S1 rope (reference rope_compute): qkv(3072 QKV out)+lut(64) -> q(2048),
        # k(512), v(512) roped. tile_2_3.
        qkv_l1 = MemRefType.get([M if ATTN_SUBSYS else CONV_IN], bf16, memory_space=l1)
        ropeq_l1 = MemRefType.get(
            [DQ_PADDED], bf16, memory_space=l1
        )  # rope emits padded Q
        ropekv_l1 = MemRefType.get([DK], bf16, memory_space=l1)
        ropelut_l1 = MemRefType.get([ROPE_LUT_LEN], bf16, memory_space=l1)
        # ShortConv L1: wbx = [w0|w1|w2|BX[t-2]|BX[t-1]] (taps + carried state,
        # contiguous, as the reference kernel expects); convo = mixer out / new BX.
        # SEGMENT-SCOPE, shared across the two mixer tiles with no DMA between
        # them: [B | C | X | w0 | w1 | w2]. ONE allocation, so AIR places it once
        # and both cores reach it as neighbour memory.
        convmix_l1 = MemRefType.get(
            [CONV_IN + CONV_W_LEN if CONV_MIXER else 1], bf16, memory_space=l1
        )
        # Per-tile. Landing buffer on the stage tile; output and the two state
        # halves on the conv tile.
        convland_l1 = MemRefType.get(
            [CONV_LAND if CONV_MIXER else 1], bf16, memory_space=l1
        )
        convo_l1 = MemRefType.get(
            [CONV_DIM if CONV_MIXER else 1], bf16, memory_space=l1
        )
        convst_l1 = MemRefType.get(
            [2 * CONV_DIM if CONV_MIXER else 1], bf16, memory_space=l1
        )
        # ATTN S3a flash-attn (1 CU; attn_iso proven shapes). DH=64, 8 Q heads,
        # 2 KV heads per CU -> DQ=OSZ=512, DK=128, k/v block 16x128, scores 192.
        aq_l1 = MemRefType.get([DQ_PADDED_PER_CU], bf16, memory_space=l1)  # q per CU
        ak_l1 = MemRefType.get(
            [16 * KVPC_DH], bf16, memory_space=l1
        )  # k block 16xKVPC_DH
        av_l1 = MemRefType.get(
            [16 * KVPC_DH], bf16, memory_space=l1
        )  # v block 16xKVPC_DH
        as_l1 = MemRefType.get([SSZ_BLK], bf16, memory_space=l1)  # shared scores
        ao_l1 = MemRefType.get([DQ_PADDED_PER_CU], bf16, memory_space=l1)  # o per CU
        # HYBRID: this CU's slice of the ShortConv mixer output, landing in the kv
        # core so it can take the place of the attention result on a conv layer.
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
        qmt_l2 = MemRefType.get(
            [DQ_PADDED], bf16, memory_space=l2
        )  # padded Q broadcast
        # o gather memtile (reference mem_5_1 o_buffer): 4 CUs' o (512 each) gathered
        # into 2048, then ONE egress (-> host now; -> mem_1_1 o-proj X in the loop close).
        omt_l2 = MemRefType.get([DQ], bf16, memory_space=l2)
        # MULTIBLK per-block KV staging memtile (attn_iso ring, PASS L=16..128): one
        # block [K block 2048 | V block 2048] = 4096; a fresh alloc per block gives a
        # count-free ping-pong ring (1 fill : 1 read), unlike a whole-cache buffer
        # multi-read (1 fill : N read = lock deadlock).
        kvblk_l2 = MemRefType.get([2 * 16 * KVPC_DH], bf16, memory_space=l2)
        # buf_ph2 (LOOPCLOSE): ph2 (gate-up) X = a_xn stand-in, re-broadcast from a
        # memtile (mechanism-2 refeed) so it converges on @xnorm AFTER ph1 attn-o.
        bufp2_l2 = MemRefType.get([K], bf16, memory_space=l2)

        # ---- L2 buffers ----
        # X memtile = reproducer x_buffer: 512 (2 blocks) so the producer re-feed +
        # broadcast has the same slack as the reference; the proj cores' 256 ring chops it.
        xmt_l2 = MemRefType.get([2 * COL_BLOCK], bf16, memory_space=l2)
        # One fan get. W_DUAL_CHAN halves it: each shim channel feeds its own
        # ring covering half the column's cores (FLM's w_buffer[0:5120] /
        # w_buffer[5120:10240] split).
        wfan_l2 = MemRefType.get(
            [(NCY // (2 if W_DUAL_CHAN else 1)) * BLOCK_BF16], bf16, memory_space=l2
        )
        grp_l2 = MemRefType.get([GRP_ROWS], bf16, memory_space=l2)
        main_l2 = MemRefType.get([MAIN_ROWS], bf16, memory_space=l2)
        relay_l2 = MemRefType.get(
            [PAYLOAD], bf16, memory_space=l2
        )  # demux relay (stripped)
        down_l2 = MemRefType.get([GLU_OUT], bf16, memory_space=l2)  # GLU out accumulate
        # relay memtile columns for the id-demux dests (free cols, not proj/X/MT).
        # GLU dest (gate-up) goes DIRECT to the GLU tile (no relay).
        # Only HOST-DRAINED, non-mixer dests are relayed; under FULL4 there are
        # none, so the tail entries are padding for a wider demux.
        RELAY_COLS = ([3, 5, 4] + [3] * NDEST)[:NDEST]

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
        # #4 faithful residual stream (reproducer rms_residual.o): residual_add_aie
        # (y = x_buf + x) for residual1 (input + o-proj-out) and residual2 (h + down-out).
        residual_add_aie = FuncOp(
            "residual_add_aie", ([rms_l1, rms_l1, rms_l1], []), visibility="private"
        )
        residual_add_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
        # GLU: glu_aie(hid, x) = pseduo_glu<1024>: x = [hid 512 | gate 512],
        # hid(512) = silu(gate)*hid. One 1024 slice per call. Prebuilt glu.o.
        glu_aie = FuncOp(
            "glu_aie", ([glu_hid_l1, glu_x_l1, i32], []), visibility="private"
        )
        glu_aie.attributes["link_with"] = StringAttr.get("glu.o")
        # reference rope_compute(q,k,v, qkv, lut): rotate-half RoPE on Q,K (V copied).
        rope_compute = FuncOp(
            "rope_compute",
            ([ropeq_l1, ropekv_l1, ropekv_l1, qkv_l1, ropelut_l1, i32], []),
            visibility="private",
        )
        rope_compute.attributes["link_with"] = StringAttr.get("rope.o")
        if HYBRID_MIXER:
            # The hybrid's rope shares its input buffer with the ShortConv
            # staging call, and rope_compute rewrites that buffer in place (see
            # rope.cc). rope_compute_hyb is rope_compute plus the reference's
            # IS_ATTN branch. Declared only for a hybrid so every shipped
            # model's IR stays byte-identical.
            rope_compute_hyb = FuncOp(
                "rope_compute_hyb",
                ([ropeq_l1, ropekv_l1, ropekv_l1, qkv_l1, ropelut_l1, i32], []),
                visibility="private",
            )
            rope_compute_hyb.attributes["link_with"] = StringAttr.get("rope.o")
        if CONV_MIXER:
            shortconv_compute = FuncOp(
                "shortconv_compute",
                # OPERAND ORDER IS LOAD-BEARING: AIR reads the LAST memref of an
                # external call as the written one, and that is what decides
                # whether @convmix is placed once and shared or cloned onto both
                # tiles (air::herdBufferHasCrossCoreDependence). See shortconv.cc.
                ([convmix_l1, convst_l1, convo_l1, convst_l1, i32], []),
                visibility="private",
            )
            shortconv_compute.attributes["link_with"] = StringAttr.get("shortconv.o")
            shortconv_stage = FuncOp(
                "shortconv_stage",
                ([convland_l1, convmix_l1, i32, i32], []),
                visibility="private",
            )
            shortconv_stage2 = FuncOp(
                "shortconv_stage2",
                ([convland_l1, convland_l1, convmix_l1, i32], []),
                visibility="private",
            )
            shortconv_stage2.attributes["link_with"] = StringAttr.get("shortconv.o")
            shortconv_stage.attributes["link_with"] = StringAttr.get("shortconv.o")
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
        if MIX_TO_CU:
            # Hybrid only: overwrite this CU's o with its slice of the ShortConv
            # mixer output when the wave's arm says ShortConv. The pick has to
            # happen in a CORE, not at the o-gather memtile: a memtile is segment
            # scope and cannot see the arm, so it cannot select between two
            # sources per wave.
            conv_o_pass = FuncOp(
                "conv_o_pass",
                ([convo_l1, ao_l1, i32, i32], []),
                visibility="private",
            )
            _set_attn_link(conv_o_pass, "attn_kv")

        # ---- channels ----
        # Faithful X-feed: host raw X (@xy) + rms weight (@rmsin) -> rms core ->
        # xnorm (re-fed N times on-chip, see refeed()) -> X memtile (512) ->
        # 256-block broadcast to all 16 cores. (reproducer core_2_2 + mem_1_1 x_buffer)
        # #4 (FULL4): rmsX is PACKET so it converges with the id4 demux (o-proj+down)
        # on the rms core's S2MM0 -- the reference's tile_2_2 DMA0 receives @xy(id0)+id4
        # both as packets into one 2-slot ping-pong (input, then o-proj, then down).
        # Debug configs keep the original circuit rmsX.
        if FULL4:
            channel_decl("rmsX", size=[1], channel_type="npu_dma_packet")
        else:
            channel_decl("rmsX", size=[1])
        channel_decl("rmsW", size=[1])
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
        # FAITHFUL convergent X feed (reproducer x_buffer DMA:3): ONE channel
        # carries BOTH the rmsnorm'd token X (phases 0..2) and the GLU-output X
        # (down phase), as convergent packet sources into ONE X-memtile S2MM, read
        # by ONE count-free feed loop. A single feed loop => one repeat count =>
        # air-to-aie infinite (count-free) BD mode, NO repeat_count (which is a
        # stale-rebroadcast deadlock). Two separate feed loops (the prior bug)
        # lowered to two repeat_count tasks. Packet (npu_dma_packet) so the two
        # producers (rms core L1 + down_buffer L2, time-disjoint, same id) converge.
        # The rms core's two outputs (this xnorm o-proj-X feedback -> mem_2_1, and
        # layerOut -> shim) used to be pinned to a known-good split, because
        # adding the append channels packed BOTH onto rms MM2S0 and deadlocked.
        # AIRToAIE reaches the same split by itself now: that packing is a ring
        # its own diagnoseBDChain calls out of step, and rather than only
        # refusing it, spreadCollapsedPacketChannels peels one flow onto the
        # channel that was sitting idle.
        channel_decl("xnorm", size=[1], channel_type="npu_dma_packet")
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
        # S3a flash-attn dataflow: rope q -> qk tile (direct); rope k|v -> KV
        # staging memtile (rope's single k/v MM2S) which splits k->qk, v->kv.
        channel_decl("ropeQ", size=[1])  # rope q (whole 2048) -> q broadcast memtile
        # Q used to be pinned to rope MM2S0 here, to keep the packet K/V append
        # off the channel carrying this circuit flow. The compiler derives that
        # now: a DMA channel's port is either statically connected or packet-
        # switched, never both, so AIRToAIE separates the two by itself
        # (TileDMAAllocator::spreadCollapsedPacketChannels) and reproduces this
        # exact placement.
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
                channel_decl("appendK", size=[1], channel_type="npu_dma_packet")
                # Only appendK names a channel. It is what holds the pair on
                # rope's second MM2S, clear of the circuit ropeQ; appendV joins
                # it there on its own.
                channel_decl("appendV", size=[1], channel_type="npu_dma_packet")
            if KV_SPLIT:
                # the reference mem_3_1: K and V on SEPARATE shim->memtile flows (one each per
                # col group of 2 CUs), so their memtile S2MM fills are independent.
                channel_decl("inKV_K", size=[len(ATTN_COL_GROUPS)])
                channel_decl("inKV_V", size=[len(ATTN_COL_GROUPS)])
            else:
                channel_decl("inKV", size=[N_ATTN_CU])
        channel_decl("attnO", size=[N_ATTN_CU])
        if MIX_TO_CU:
            # ShortConv mixer -> all four CU kv cores, one BROADCAST put of the
            # whole CONV_DIM (each CU keeps its own slice in C). A broadcast is
            # ONE acquire, so it sidesteps rule 5 -- four puts of four DIFFERENT
            # slices out of one L1 buffer would fold to four Acquire(1) against
            # init=1 and block on the second. The extra 3x on the wire is 12 KB a
            # token against ~42 MB of weights.
            # broadcast_shape is what makes this ONE flow with four
            # destinations. A plain size=[1] channel read by four tiles keeps
            # only ONE of them in air-to-aie's channel specialization -- the
            # other three are left with a complete DMA program and no aie.flow
            # behind them, so they stall on their acquire forever (measured: one
            # aie.flow, tile_5_5 -> tile_3_3, and nothing to the other three).
            Channel("mixToCU", size=[1, 1], broadcast_shape=[1, N_ATTN_CU])
        if CONV_MIXER:
            # convO is the attnO analogue: ShortConv mixer -> o memtile -> o-proj X.
            # A HYBRID has no convO. The o-gather memtile is segment scope, so it
            # cannot choose its source per wave, and @attnO cannot simply take a
            # second producer either: a channel with two producer TILES loses one
            # of them in air-to-aie's channel specialization, and the dropped tile
            # is left with a complete DMA program and no aie.flow behind it, so it
            # stalls on its acquire forever (measured -- that is what hung the
            # first hybrid, identically on both arms). So the mixer output goes to
            # the CUs instead, over @mixToCU, and the four CU kv cores stay the
            # SOLE producers of @attnO. Which of the two results a kv core puts is
            # chosen inside the core from its RTP arm -- exactly the reference's
            # IS_ATTN, and the only layer-type branch left in the whole design.
            # convStIn/Out carry the per-layer [BX[t-2] | BX[t-1]] state to and from
            # arg4 (the KV slot, which a conv layer does not otherwise use).
            # One channel each; all four are plain shim/memtile <-> tile feeds.
            # The two big buffers move by neighbour memory and have no channel.
            if not HYBRID_MIXER:
                channel_decl("convO", size=[1])
            channel_decl("convW", size=[1])
            channel_decl("convStIn", size=[1])
            # Packet so it can converge with the KV append on the shim's arg4
            # S2MM -- but CIRCUIT whenever the mixer core ALSO feeds @xnorm
            # directly, because then both of its outputs would be packet flows on
            # one tile and simpleDmaChannelAlloc multiplexes any two of those onto
            # a single MM2S. That is fine at equal rates; these differ. The X is a
            # refeed of OPROJ_REFEED and the state write-back fires once, so the
            # chained BD ring interleaves them 1:1, the core's
            # Acquire(OPROJ_REFEED) is never satisfied, and the design deadlocks
            # with nothing written.
            # A HYBRID cannot take the circuit escape: this has to converge with
            # the KV append on the shim's arg4 S2MM, and the pass converts it to
            # a packet flow regardless. `air.tile_dma_channel` does not help
            # either -- the channel is fixed before the pin is consulted.
            # Unblocking the hybrid's ShortConv arm needed a compiler fix: a
            # rate-mismatch hazard in spreadCollapsedPacketChannels, keyed on
            # air.refeed_count.
            _stout_pkt = int(
                _os.environ.get("CONV_STOUT_PKT", "0" if HYBRID_MIXER else "1")
            )
            channel_decl(
                "convStOut",
                size=[1],
                **({"channel_type": "npu_dma_packet"} if _stout_pkt else {}),
            )
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
            # layerOut and the xnorm above are the rms core's only two outputs,
            # and AIRToAIE gives them a channel each: collapsed onto one they
            # form a ring its diagnoseBDChain oracle calls out of step, and
            # spreadCollapsedPacketChannels peels one onto the idle channel.
            channel_decl("layerOut", size=[1])
        # GLU path: id-demux delivers gate-up DIRECTLY to the GLU herd (no relay);
        # GLU -> gluOut -> down memtile accumulate (8192). FAITHFUL: that 8192 is
        # fed back on-chip as the DOWN phase X by the down_buffer re-broadcasting it
        # DOWN_REFEED times into the convergent @xnorm channel (counting-lock-N on
        # the down_buffer fill, derived from the DOWN_REFEED loop around the put
        # below), NOT drained to host.
        channel_decl("gluOut", size=[1])

        def idx(v):
            return arith.ConstantOp.create_index(v)

        def _arm_of_wave(iv, arm):
            """Promote a decode arm to ATTN (2) on the attention waves.

            The arm encoding mirrors the reference's IS_ATTN RTP: 0 = lm-head
            wave, 1 = conv layer, 2 = attention layer. Non-hybrid builds never
            reach the promotion, so their arm stays the original 1/0 select and
            their IR is unchanged.

            The LFM2 schedule is irregular, so this is a chain of equality
            selects over ATTN_WAVES rather than any arithmetic predicate. The
            wave loop is fully unrolled by the time it reaches the shim, so
            every one of these folds to a constant.

            ATTN_WAVES, not ATTN_LAYERS: an entry at or past UNI_DEC names an
            lm-head wave, and promoting one to the attention arm silently drops
            that vocab chunk's logits. See its definition.
            """
            if not HYBRID_MIXER or iv is None:
                return arm
            _a2 = arith.ConstantOp(IntegerAttr.get(i32, 2), None).result
            for _k in ATTN_WAVES:
                arm = arith.select(
                    arith.cmpi(arith.CmpIPredicate.eq, iv, idx(_k)), _a2, arm
                )
            return arm

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
                # None => the caller is on the statically-known lm-head buffer; a
                # Value => a runtime group index into _WBUFS[0:N_WGRP].
                _wsel = [None]
                KVC = _la[8] if MULTIBLK else None
                # The dispatch-time context length (DYNSEQ). Last operand before
                # the multi-layer induction variable.
                L_rt = _la[4 + len(_fa) - 1] if DYNSEQ else None

                def _rt_blocks():
                    """ceil(L/16) as an index Value, for the readback's block count.

                    Kept in i32: aie-translate's C++ TXN target emits the integer
                    widths but has no case for index-typed arithmetic, and this
                    expression has to survive all the way into the emitted builder.
                    """
                    _s = arith.addi(
                        L_rt, arith.ConstantOp(IntegerAttr.get(i32, 15), None).result
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
                    _m = arith.subi(
                        L_rt, arith.ConstantOp(IntegerAttr.get(i32, 1), None).result
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

                    if not W_SPLIT:
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
                        else:
                            _ucmp = arith.cmpi(
                                arith.CmpIPredicate.slt, a_iv, idx(UNI_DEC)
                            )
                            _uarm = _arm_of_wave(a_iv, arith.select(_ucmp, _u1, _u0))
                        _uarm_i = arith.index_cast(idx_t, _uarm)

                        def _mix_gate(
                            attn_body,
                            conv_body,
                            attn_always=False,
                            conv_always=False,
                            conv_skip=False,
                        ):
                            """Per-layer-type feeds, inside the decode arm.

                            A hybrid feeds different things for the two layer
                            types: the rope LUT + the KV append/readback for an
                            attention layer, the depthwise taps + the carried
                            ShortConv state for a conv layer. Everything else --
                            weights, both norms, X, the egress drain -- is
                            identical, which is what the uniform phase schedule
                            buys, so only these blocks are switched.

                            Arm 0 cannot reach here (this is the decode arm), so
                            the switch is case-2 / default. A non-hybrid build
                            has one layer type and emits it directly, leaving
                            its IR untouched.
                            """
                            # *_always: that side's consumer runs on every decode
                            # wave -- a memtile, or a core that has to run every
                            # wave to drain one -- so the feed cannot be armed.
                            for _always, _body, _kind in (
                                (attn_always, attn_body, 2),
                                (conv_always, conv_body, 1),
                            ):
                                if conv_skip and _kind == 1:
                                    continue
                                if HYBRID_MIXER and _always:
                                    _body()
                                else:
                                    _arm_only(_uarm_i, {_kind}, _body, in_dec=True)

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
                            _vyb = arith.addi(
                                idx((HOST_ROUNDS + LAYER_RNDS) * PAYLOAD),
                                arith.muli(
                                    arith.subi(a_iv, idx(UNI_DEC)),
                                    idx(VOCAB_SIZE_PADDED),
                                ),
                            )
                            # ===== LM head (IS_ATTN=0), the reference gen_lm_head_seq analog =====
                            # Same device, RTP arm=0: the proj cores run ONE vocab phase
                            # (VOCAB_I2 row-pairs x NBJ col-blocks, id4 -> RMS_DEST) and the
                            # rms core does final rmsnorm(x) then forwards VOCAB_RNDS logit
                            # rounds out via layerOut. Feed: x + final rms weight + vocab
                            # weights; drain VOCAB_SIZE_PADDED logits into Y. No attn/rope/
                            # glu/KV feeds (those herds are parked -- RTP-unarmed -- so they
                            # need no input; feeding them would only back-pressure).
                            # real-lm-head final norm (model.norm.weight): a DEDICATED slot
                            # after the [in|post]*UNI_DEC rms slabs + 64-wide rope LUT, so the
                            # vocab rmsnorm uses the true final norm -- NOT layer-0's in_LN
                            # (mirrors decoding_layer's separate final_rms_weight).
                            # final norm sits AFTER the rope region: llama has ONE shared
                            # rope LUT (ROPE_W_LEN), per-layer models have UNI_DEC slabs.
                            _final_norm_off = (
                                UNI_DEC * RMS_LAYER
                                + (UNI_DEC if ROPE_W_PER_LAYER else 1) * ROPE_W_LEN
                            )
                            if N_NORMS >= 4:
                                # Gemma: rmsW/rmsW2 are 2K (two norms packed). Put final_norm
                                # in rmsW's HI half (lm_head reads it via rms_norm_hi_aie); the
                                # LO half is the last rope-region K -- a harmless in-bounds
                                # dummy ([_final_norm_off-K .. +2K] is the BO's last 2K). rmsW2
                                # is a 2K dummy. Keeps the shared packet group hole-free.
                                if not RMSW_DMA:
                                    ChannelPut(
                                        "rmsW",
                                        RMS,
                                        offsets=[_final_norm_off - K],
                                        sizes=[2 * K],
                                        strides=[1],
                                    )
                                if not RMSW2_DMA:
                                    ChannelPut(
                                        "rmsW2",
                                        RMS,
                                        offsets=[0],
                                        sizes=[2 * K],
                                        strides=[1],
                                    )
                            else:
                                if not RMSW_DMA:
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
                                    if not RMSW2_DMA:
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
                            # drain logits (natural order): rms LM branch
                            # forwards VOCAB_RNDS x PAYLOAD via layerOut; ONE 2D-strided get.
                            ChannelGet(
                                "layerOut",
                                Y,
                                indices=[idx(0)],
                                offsets=[_vyb],
                                sizes=[VOCAB_RNDS, PAYLOAD],
                                strides=[PAYLOAD, 1],
                            )
                            yield_([])

                        def _uni_dec():
                            _wsel[0] = _wgi  # runtime group index (None if unsplit)
                            # raw X (@xy) + rms weight (@rmsin) to the rms producer core; the
                            # on-chip rms normalizes + re-feeds X (see refeed()). X is
                            # in-place (offset 0 every layer -- the chained hidden state).
                            if N_NORMS >= 4:
                                # Gemma: pack two norms per 2K channel -- rmsW =
                                # [input | post_attn] (slab 0..2K), rmsW2 = [pre_ffn |
                                # post_ffn] (slab 2K..4K). Keeps the rms tile at <=4 packet
                                # ids per S2MM port; the lo/hi kernels slice each half.
                                if not RMSW_DMA:
                                    ChannelPut(
                                        "rmsW",
                                        RMS,
                                        offsets=[_rbase],
                                        sizes=[2 * K],
                                        strides=[1],
                                    )
                                if not RMSW2_DMA:
                                    ChannelPut(
                                        "rmsW2",
                                        RMS,
                                        offsets=[_lo(_rbase, 2 * K)],
                                        sizes=[2 * K],
                                        strides=[1],
                                    )
                            else:
                                if not RMSW_DMA:
                                    ChannelPut(
                                        "rmsW",
                                        RMS,
                                        offsets=[_rbase],
                                        sizes=[K],
                                        strides=[1],
                                    )
                                if POST_RMS:
                                    # post_attention_layernorm weight on its own channel.
                                    if not RMSW2_DMA:
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
                            _rope_off = (
                                _lo(_lb(ROPE_W_LEN), _lut_off)
                                if (ROPE_W_PER_LAYER and MULTIBLK)
                                else _lut_off
                            )

                            # Both layer types read their mixer weights from the
                            # SAME per-layer rope_w slab, each taking its own prefix:
                            # tap-major [w0|w1|w2] for conv, cos/sin + qk-norm for
                            # attention. The slab is sized for the larger.
                            # attn_always: rope runs on every decode wave in a
                            # hybrid (its q-broadcast memtile is wave-invariant),
                            # so its LUT has to arrive on every decode wave too.
                            # On a ShortConv wave it reads the taps as a cos/sin
                            # table and the result is discarded.

                            # A HYBRID defers the whole mixer feed block into the
                            # phase loop, to the LAST mixer phase -- same reason
                            # KV_PHASE exists (see its use site). @convStOut is a
                            # shim GET on the ShortConv core's output, and that
                            # core cannot run until every mixer phase has landed;
                            # issuing it here, before the phase loop that feeds
                            # those phases' weights, deadlocks exactly as the KV
                            # append did. Every other model emits it right here,
                            # unchanged.
                            def _emit_mixer_feeds():
                                _mix_gate(
                                    # ROPELUT_DMA: the rope core spells this feed
                                    # as an air.dma_memcpy_nd naming @ropeLUT, so
                                    # air-dma-to-channel derives this put and
                                    # writing it here as well would double it.
                                    (
                                        (lambda: None)
                                        if (ROPELUT_DMA_OK and RMS is not None)
                                        else lambda: ChannelPut(
                                            "ropeLUT",
                                            RMS,
                                            offsets=[_rope_off],
                                            sizes=[ROPE_LUT_LEN],
                                            strides=[1],
                                        )
                                    ),
                                    (
                                        (lambda: None)
                                        if (CONVW_DMA and RMS is not None)
                                        else lambda: ChannelPut(
                                            "convW",
                                            RMS,
                                            offsets=[_rope_off],
                                            sizes=[CONV_W_LEN],
                                            strides=[1],
                                        )
                                    ),
                                    # Rope AND the mixer both run on every decode
                                    # wave, so both LUTs have to arrive on every
                                    # decode wave. They read the same per-layer
                                    # rope_w slab from either end, so this is two
                                    # puts of one DDR region, not extra traffic.
                                    attn_always=HYBRID_MIXER,
                                    conv_always=HYBRID_MIXER,
                                )
                                if CONV_MIXER:
                                    # Conv state lives in arg4 alongside the KV cache
                                    # (a conv layer has no KV, an attention layer no
                                    # state): [BX[t-2] | BX[t-1]] per layer. Read it
                                    # out, and write the kernel's shifted state back
                                    # over the SAME slot -- the RAW on this DDR region
                                    # gives air-annotate-append-barrier the read->write
                                    # order, exactly as it does for the KV cache.
                                    _cst = _lo(_lb(CONV_ST_LAYER), CONV_ST_BASE)

                                    def _conv_state():
                                        if CONVST_DMA:
                                            return
                                        ChannelPut(
                                            "convStIn",
                                            KVC,
                                            offsets=[_cst],
                                            sizes=[CONV_ST_LAYER],
                                            strides=[1],
                                        )
                                        ChannelGet(
                                            "convStOut",
                                            KVC,
                                            offsets=[_cst],
                                            sizes=[CONV_ST_LAYER],
                                            strides=[1],
                                        )

                                    # The mixer core runs unarmed, so its state
                                    # read-back and write-back are issued on every
                                    # decode wave. An attention layer reads and
                                    # rewrites its own (unused) state slot with a
                                    # value its kernel computed from garbage and
                                    # nothing downstream ever reads.
                                    _mix_gate(
                                        lambda: None,
                                        _conv_state,
                                        conv_always=HYBRID_MIXER,
                                    )

                            if not HYBRID_MIXER:
                                _emit_mixer_feeds()

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
                                    if APPEND_DMA:
                                        # Derived from the DMAs in the rope herd.
                                        # Converting a get to a DMA means deleting
                                        # the matching hand-written put -- and
                                        # here the pair is the other way round, so
                                        # it is these gets that go.
                                        return
                                    _apkG = ChannelGet(
                                        "appendK",
                                        KVC,
                                        indices=[idx(0)],
                                        offsets=[_loi_slot(_kbase, 0)],
                                        sizes=[idx(NGRP), idx(REGION_W)],
                                        strides=[idx(REGION_STRIDE), idx(1)],
                                    )
                                    _apvG = ChannelGet(
                                        "appendV",
                                        KVC,
                                        indices=[idx(0)],
                                        offsets=[_loi_slot(_kbase, _vreg_off(0))],
                                        sizes=[idx(NGRP), idx(REGION_W)],
                                        strides=[idx(REGION_STRIDE), idx(1)],
                                    )
                                    return

                            def _emit_readback(_kbase=_kbase):
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
                                        for gi in range(NGRP):
                                            ChannelPut(
                                                "inKV_K",
                                                KVC,
                                                indices=[idx(gi)],
                                                offsets=[
                                                    _loi(_kbase, _kreg_off(gi) + _coff)
                                                ],
                                                sizes=(
                                                    [idx(_cb * 16 * REGION_W)]
                                                    if _KV1D
                                                    else [
                                                        _cbv(),
                                                        idx(16),
                                                        idx(REGION_W),
                                                    ]
                                                ),
                                                strides=(
                                                    [idx(1)]
                                                    if _KV1D
                                                    else [
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
                                                offsets=[
                                                    _loi(_kbase, _vreg_off(gi) + _coff)
                                                ],
                                                sizes=(
                                                    [idx(_cb * 16 * REGION_W)]
                                                    if _KV1D
                                                    else [
                                                        _cbv(),
                                                        idx(16),
                                                        idx(REGION_W),
                                                    ]
                                                ),
                                                strides=(
                                                    [idx(1)]
                                                    if _KV1D
                                                    else [
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
                                # LAST mixer phase, not the first. The shim is a
                                # sequential instruction stream: the append blocks
                                # on rope's K/V, and rope cannot run until the
                                # stage tile has landed every mixer phase, whose
                                # weight feeds come LATER in that same stream. With
                                # one mixer phase (every attention-only model) the
                                # two coincide and it cannot be hit; with two it
                                # deadlocks the whole design. Measured: the hybrid
                                # machinery passes with one mixer phase and hangs
                                # with two, everything else held fixed.
                                if MULTIBLK and p == KV_PHASE and ATTN_SUBSYS:

                                    def _kv_traffic():
                                        _emit_append()
                                        _emit_readback()

                                    # Attention waves only. The KV memtile behind
                                    # this cannot be armed (segment scope), but it
                                    # does not need to be: with no traffic issued
                                    # it simply waits, and so do the CUs. Nothing
                                    # else is waiting on either of them -- which is
                                    # how the reference behaves, and is why nothing
                                    # here has to be made wave-invariant.
                                    # In a hybrid the CUs run their block loop on
                                    # a ShortConv wave too, so the cache read-back
                                    # that loop consumes is issued on every decode
                                    # wave. The append is part of the same ordered
                                    # block; a conv layer appends rope's garbage
                                    # K/V into its own (never-read) cache slot.
                                    _mix_gate(
                                        _kv_traffic,
                                        lambda: None,
                                        attn_always=HYBRID_MIXER,
                                    )
                                if HYBRID_MIXER and p == KV_PHASE:
                                    # Deferred to here from before the phase loop
                                    # (see _emit_mixer_feeds), and deliberately
                                    # AFTER the KV traffic above.
                                    #
                                    # @convStOut is a shim GET on the ShortConv
                                    # core, and that core blocks first on its four
                                    # @mixToCU puts -- which the CUs only take once
                                    # they are out of their block loop, which needs
                                    # the cache readback. Issue the readback first
                                    # and the chain runs; issue it second and the
                                    # shim waits on a core that is waiting on the
                                    # shim.
                                    _emit_mixer_feeds()
                            # per-dest host drain: dest p drains ROUNDS_PER_DEST[p] rounds into
                            # this layer's Y region (diagnostic per-layer QKV observation).
                            roff = 0
                            for p in HOST_DRAIN:
                                if p in MIXER_DESTS:
                                    # loop close: the mixer dests (QKV->rope->flash
                                    # attention, or in_proj->ShortConv) are consumed
                                    # on-chip as the o-proj X, not drained to host.
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
                            _out_bo = X
                            _out_base = 0
                            # BD-COMPACTION: single full-size drain (matches the rms single
                            # layerOut put) instead of LAYER_RNDS per-round gets.
                            # LAYEROUT_DMA: derived from the rms core's DMA, which
                            # names X as the far end -- writing it here as well
                            # would double the drain.
                            if not LAYEROUT_DMA:
                                ChannelGet(
                                    "layerOut",
                                    _out_bo,
                                    indices=[idx(0)],
                                    offsets=[_out_base],
                                    sizes=[LAYER_RNDS * PAYLOAD],
                                    strides=[1],
                                )
                            yield_([])

                        index_switch(
                            [],
                            _uarm_i,
                            [0],
                            case_body_builder=lambda op, i, cv: _uni_voc(),
                            default_body_builder=lambda op: _uni_dec(),
                        )
                # (No GLU host drain: the GLU output is consumed on-chip by the down
                # phase. The down output egresses via the rms layer output above.)

                # HYBRID: the layer-type arm must reach the segment as an i32
                # OPERAND computed out here, not be recomputed inside from the
                # wave index. cloneL2AndL3MemcpysToDeviceOp pins INDEX-typed
                # segment arguments to constant 0, so an arm derived inside the
                # segment folds to the wave-0 arm and every other branch is
                # erased -- in cores as much as in memtiles, which is why the
                # all-attention and all-ShortConv hybrids compiled to literally
                # identical flow sets, both of them the ShortConv one. An i32
                # operand is left alone (this is how DYNSEQ's L already reaches
                # the attention herd) and survives as a real per-dispatch RTP.
                _seg_arm_rt = (
                    _arm_of_wave(
                        a_iv,
                        arith.select(
                            arith.cmpi(arith.CmpIPredicate.slt, a_iv, idx(UNI_DEC)),
                            arith.ConstantOp(IntegerAttr.get(i32, 1), None).result,
                            arith.ConstantOp(IntegerAttr.get(i32, 0), None).result,
                        ),
                    )
                    if (HYBRID_MIXER and a_iv is not None)
                    else None
                )
                # RMS reaches segment scope because the rope LUT feed is spelled
                # as an air.dma_memcpy_nd naming @ropeLUT, and a DMA has to name
                # BOTH endpoints in one place -- air-dma-to-channel derives the
                # shim put from it. The explicit put/get form does not need this,
                # which is why no other L3 buffer is a segment operand.
                _seg_opers = (
                    ([a_iv] if a_iv is not None else [])
                    + ([_seg_arm_rt] if _seg_arm_rt is not None else [])
                    + [RMS, X]
                    + (
                        [KVC]
                        if ((APPEND_DMA or CONVST_DMA) and KVC is not None)
                        else []
                    )
                    + ([L_rt] if DYNSEQ else [])
                )
                # Index of RMS above; keeps _sa[-1] meaning L_rt for DYNSEQ.
                _seg_rms_idx = (1 if a_iv is not None else 0) + (
                    1 if _seg_arm_rt is not None else 0
                )

                @segment(name="seg", operands=_seg_opers)
                def seg(*_sa):
                    _seg_iv = _sa[0] if a_iv is not None else None
                    _seg_RMS = _sa[_seg_rms_idx]
                    # X follows RMS, for the @rmsX feed spelled as a DMA.
                    _seg_X = _sa[_seg_rms_idx + 1]
                    # KVC follows X, for the @appendK/@appendV feeds spelled as
                    # DMAs. Appended after X so _sa[-1] still means L_rt.
                    _seg_KVC = (
                        _sa[_seg_rms_idx + 2]
                        if ((APPEND_DMA or CONVST_DMA) and KVC is not None)
                        else None
                    )
                    # The context length reaches the attention herd from here, as a
                    # herd operand: an RTP slot the instruction stream writes per
                    # dispatch, not a constant folded into the core ELF.
                    _seg_L = _sa[-1] if DYNSEQ else None

                    def _seg_rounds():
                        """ceil(L/16) for the memtile's block dequeue.

                        The memtile sits between the shim's readback BD and the
                        cores, so its trip count has to be the same ceil(L/16) both
                        of those use.
                        """
                        if not DYNSEQ_MEM:
                            return idx(ATTN_ROUNDS)
                        _s = arith.addi(
                            _seg_L,
                            arith.ConstantOp(IntegerAttr.get(i32, 15), None).result,
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
                        _seg_arm = _arm_of_wave(
                            _seg_iv,
                            arith.select(
                                _seg_cmp,
                                arith.ConstantOp(IntegerAttr.get(i32, 1), None).result,
                                arith.ConstantOp(IntegerAttr.get(i32, 0), None).result,
                            ),
                        )
                        _seg_arm_i = arith.index_cast(idx_t, _seg_arm)
                    else:
                        _seg_cmp = None
                        _seg_arm_i = None
                        _seg_arm = arith.ConstantOp(
                            IntegerAttr.get(i32, 0 if LM_HEAD else 1), None
                        ).result

                    # TWO arms, and the distinction is load-bearing.
                    #
                    # _seg_arm is derived from the segment's wave index, which
                    # cloneL2AndL3MemcpysToDeviceOp pins to constant 0 -- so every
                    # gate written on it folds to the wave-0 arm. That is exactly
                    # what the shipped models want from it (it is their
                    # decode-vs-vocab gate, and it folds to "decode"), and it is
                    # all a memtile can express anyway, since a memtile's DMA
                    # program is static.
                    #
                    # _core_arm is the hybrid's per-layer IS_ATTN. It arrives as an
                    # i32 segment OPERAND computed at launch scope, the same route
                    # DYNSEQ's L already takes, so it is a real per-dispatch RTP
                    # rather than something the folder can see through. Use it for
                    # herd operands ONLY. Deriving the layer type inside the
                    # segment instead compiled the all-attention and the
                    # all-ShortConv hybrid to byte-identical flow sets -- both of
                    # them the ShortConv one, with the CUs' @attnO puts erased.
                    _core_arm = _sa[1] if _seg_arm_rt is not None else _seg_arm
                    _core_arm_i = (
                        arith.index_cast(idx_t, _core_arm)
                        if _seg_arm_rt is not None
                        else _seg_arm_i
                    )

                    # ===== X memtile (the reference mem_1_1 x_buffer): 512 ring, re-fed =====
                    # The cores read X in phase order: phases 0..2 read the rmsnorm'd
                    # token X (K=2048), phase 3 (down) reads the GLU output (K=8192)
                    # fed back on-chip. The SAME inX broadcast carries both, in order.
                    #
                    # (1) rms-X: get the normed X (from the rms core, re-fed RMS_REFEED
                    # times over @xnorm) in 512 chunks -> broadcast
                    # 256-blocks. RMS_REFEED*(2048/512) gets. (reproducer core_2_2 +
                    # mem_1_1 x_buffer 512.)
                    # Allocated UP FRONT under INX_DMA so it dominates every proj
                    # herd: the cores name both endpoints in one op, so the X
                    # memtile buffer has to be a herd operand. Same move @outA's
                    # group buffers and @gluOut's down buffer make.
                    _xb_pre = None
                    if INX_DMA:
                        _xb_pre = AllocOp(xmt_l2, [], [])
                        _xb_pre.operation.attributes["air.memtile_col"] = (
                            IntegerAttr.get(T.i32(), XMT_PCOL)
                        )

                    def _feed_inX(src, total_chunks):
                        for _rc in for_(idx(0), idx(total_chunks), idx(1)):
                            if INX_DMA:
                                xb = _xb_pre
                            else:
                                xb = AllocOp(xmt_l2, [], [])
                                xb.operation.attributes["air.memtile_col"] = (
                                    IntegerAttr.get(T.i32(), XMT_PCOL)
                                )
                            ChannelGet(
                                src, xb, offsets=[0], sizes=[2 * COL_BLOCK], strides=[1]
                            )
                            if not INX_DMA:
                                # The sub-block loop exists only to step the window,
                                # which is why it disappears when the cores spell the
                                # transfer: the compiler reads the two pieces off the
                                # buffer sizes and emits one put per piece, in order,
                                # right here after the fill.
                                for _jj in for_(idx(0), idx(2), idx(1)):
                                    joff = arith.muli(_jj, idx(COL_BLOCK))
                                    ChannelPut(
                                        "inX",
                                        xb,
                                        offsets=[joff],
                                        sizes=[COL_BLOCK],
                                        strides=[1],
                                    )
                                    yield_([])
                                DeallocOp(xb)
                            yield_([])

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
                    # XN_REFEED, not REFEED[0]: a mixer split over several
                    # phases re-feeds the rms X once per phase, and this count
                    # must match what the cores consume or the X memtile
                    # starves them.
                    _xc_dec = (XN_REFEED + OPROJ_REFEED + GATEUP_REFEED) * (
                        K // (2 * COL_BLOCK)
                    ) + (
                        DOWN_REFEED * (GLU_OUT // (2 * COL_BLOCK))
                        if DOWN_PHASE >= 0
                        else 0
                    )
                    _xc_voc = VOCAB_RNDS * (K // (2 * COL_BLOCK))
                    if _seg_arm_i is not None:

                        def _xs_voc():
                            _feed_inX("xnorm", _xc_voc)
                            yield_([])

                        def _xs_dec():
                            _feed_inX("xnorm", _xc_dec)
                            yield_([])

                        index_switch(
                            [],
                            _seg_arm_i,
                            [0],
                            case_body_builder=lambda op, i, cv: _xs_voc(),
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
                    # Allocated UP FRONT so they dominate the proj herds: with
                    # OUTA_DMA the emitters name both endpoints in one op, so the
                    # group buffer has to be an operand of the herd. Same move
                    # @gluOut's down buffer makes.
                    _grp_pre = []
                    if OUTA_DMA:
                        for g in range(N_GRP):
                            _gb = AllocOp(grp_l2, [], [])
                            _gb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), GRP_PCOL[g])
                            )
                            _grp_pre.append(_gb)

                    def _egress(_nrc):
                        for _r in for_(idx(0), idx(_nrc), idx(1)):
                            for g in range(N_GRP):
                                if OUTA_DMA:
                                    grp = _grp_pre[g]
                                else:
                                    grp = AllocOp(grp_l2, [], [])
                                    grp.operation.attributes["air.memtile_col"] = (
                                        IntegerAttr.get(T.i32(), GRP_PCOL[g])
                                    )
                                for k, (cx, pp) in enumerate(grp_leads(g)):
                                    if OUTA_DMA:
                                        # derived from the emitters' DMAs, anchored
                                        # back into this loop by hoist_before=@toMain
                                        continue
                                    off = 0 if k == 0 else HDR + k * PAIR_PAY
                                    sz = (HDR + PAIR_PAY) if k == 0 else PAIR_PAY
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
                                    sizes=[GRP_ROWS],
                                    strides=[1],
                                )
                                if not OUTA_DMA:
                                    DeallocOp(grp)
                            ml = AllocOp(main_l2, [], [])
                            ml.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), MAIN_PCOL)
                            )
                            for g in range(N_GRP):
                                off = (
                                    0
                                    if g == 0
                                    else (GRP_ROWS + (g - 1) * LEADS_PER_GRP * PAIR_PAY)
                                )
                                sz = GRP_ROWS if g == 0 else LEADS_PER_GRP * PAIR_PAY
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
                                sizes=[MAIN_ROWS],
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

                        index_switch(
                            [],
                            _seg_arm_i,
                            [0],
                            case_body_builder=lambda op, i, cv: _egr_voc(),
                            default_body_builder=lambda op: _egr_dec(),
                        )
                    else:
                        _egress(VOCAB_RNDS if LM_HEAD else N_ROUNDS)
                    # id-demux HOST dests (QKV id1, o-proj id4): per-round relay memtile
                    # -> host (strip demux already delivered pure 512). The gate-up dest
                    # (id8) is NOT here -- it goes DIRECTLY to the GLU tile (below).
                    for p in HOST_DRAIN:
                        if p in MIXER_DESTS:
                            continue  # consumed by the mixer herd (below)
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

                    # ===== ATTN S1: the rope mixer (reference tile_2_3) =====
                    # QKV (dest0) -> rope_compute(qkv M, lut) -> q/k/v roped, then
                    # q -> the attention CUs and k/v -> the KV append.
                    #
                    # This is a FUNCTION, not a herd body, because a HYBRID build
                    # runs it on the conv mixer's STAGE tile -- the reference does
                    # the same, branching on IS_ATTN[0] inside one rope.cc.
                    #
                    # the reference-faithful: NO QKV staging memtile. The QKV
                    # (dest0) is assembled directly in this core's L1, mirroring
                    # layer.mlir mem_2_3 S2MM0. Removing the col-2 memtile relay is
                    # the fix for the fused vocab deadlock: in vocab mode dest0
                    # never flows, and an idle compute-tile S2MM does NOT stall the
                    # col-2 memtile that the vocab X-feed/rms share.
                    def _lut_slab_off(kiv):
                        """The rope_w slab offset, rebuilt inside a herd.

                        _rope_off is a launch-scope expression and a herd is
                        IsolatedFromAbove, so it has to be recomputed from the
                        wave index. Both readers of that slab want it: @ropeLUT
                        takes the cos/sin + qk-norm prefix and @convW the
                        depthwise taps, out of the SAME per-layer region.
                        """
                        _o = (UNI_DEC * RMS_LAYER) if MULTIBLK else 0
                        if not (ROPE_W_PER_LAYER and MULTIBLK):
                            return _o
                        assert kiv is not None, (
                            "the rope_w slab is per-layer but the wave index did "
                            "not reach this herd: the launch-scope put is "
                            "suppressed and the get would be unpaired"
                        )
                        _b = arith.muli(kiv, idx(ROPE_W_LEN))
                        return arith.addi(_b, idx(_o)) if _o else _b

                    def _rope_body(
                        _arm, a_qkv=None, rms=None, kvc=None, kiv=None, qmt=None
                    ):
                        # a_qkv given => the caller owns the buffer and has
                        # already filled it (the hybrid hands rope the ph0
                        # landing). None => rope allocates and fills its own.
                        _own_qkv = a_qkv is None
                        if _own_qkv:
                            a_qkv = AllocOp(qkv_l1, [], [])
                        # the reference-faithful (layer.mlir mem_2_3 S2MM0): the rope COMPUTE
                        # core assembles the 6 id1/dest0 demux rounds (512 each)
                        # directly into its own L1 3072 buffer -- NO col-2 memtile
                        # relay. Identical 6x512 offset gets as the old qkvmt (each
                        # get consumes one stripped packet round), just landing in
                        # L1. In vocab mode id1 never flows so this compute-tile
                        # S2MM idles harmlessly.
                        for _rq in range(MIX_RNDS_QKV if _own_qkv else 0):
                            _rn = PAYLOAD
                            ChannelGet(
                                "outY",
                                a_qkv,
                                indices=[idx(0), idx(0)],
                                offsets=[idx(_rq * PAYLOAD)],
                                sizes=[idx(_rn)],
                                strides=[idx(1)],
                            )
                        a_lut = AllocOp(ropelut_l1, [], [])
                        _rope_off_h = _lut_slab_off(kiv)
                        if ROPELUT_DMA_OK and rms is not None:
                            # Spelled as a DMA naming @ropeLUT rather than as a
                            # get with a matching put at launch scope: the pass
                            # hoists the shim put out for us. The declaration is
                            # untouched, so channel_type and the placement pins
                            # on it survive.
                            DmaMemcpyNd(
                                a_lut,
                                rms,
                                src_offsets=[_rope_off_h],
                                src_sizes=[ROPE_LUT_LEN],
                                src_strides=[1],
                                channel="ropeLUT",
                                channel_indices=[0],
                                # Keep this feed's shim BD where the hand-written
                                # put had it: straight after @rmsW, ahead of the
                                # weight stream. Without it the derived put lands
                                # at the herd's position, slot 6 -> 18, and the
                                # rope core deadlocks waiting on its LUT.
                                **{_ROPELUT_KW: _ROPELUT_ANCHOR},
                            )
                        else:
                            ChannelGet("ropeLUT", a_lut, indices=[idx(0)])
                        a_q = AllocOp(ropeq_l1, [], [])
                        a_k = AllocOp(ropekv_l1, [], [])
                        a_v = AllocOp(ropekv_l1, [], [])
                        CallOp(
                            rope_compute_hyb if HYBRID_MIXER else rope_compute,
                            [a_q, a_k, a_v, a_qkv, a_lut, _arm],
                        )
                        # S3a: feed flash attention (1 CU = CU0). q[0:512] -> qk
                        # tile directly (MM2S0). k[0:128]+v[0:128] (CU0's 2 KV
                        # heads) -> KV staging memtile on ONE MM2S (rope's 2nd
                        # MM2S, like reference rope k/v packets) which splits them.
                        # q reorder = pack_q (reference mem_5_1 [8,8,8]/[8,64,1]):
                        # natural [qh,dh] -> [dc,qh,de], the kernel's q layout.
                        # q (whole 2048) -> q broadcast memtile (1 rope MM2S);
                        # the memtile fans out per-CU reordered (reference mem_5_1).
                        if ROPEQ_DMA and qmt is not None:
                            # rope's Q broadcast, spelled as a DMA naming @ropeQ. Unlike
                            # the other feeds this one never touches the shim: it is
                            # L1 -> L2, so the derived half lands at SEGMENT scope. The
                            # anchor still matters, because the get has to precede the
                            # per-CU fan-out that reads the same buffer.
                            DmaMemcpyNd(
                                qmt,
                                a_q,
                                dst_offsets=[0],
                                dst_sizes=[DQ_PADDED],
                                dst_strides=[1],
                                src_offsets=[0],
                                src_sizes=[DQ_PADDED],
                                src_strides=[1],
                                channel="ropeQ",
                                channel_indices=[0],
                                hoist_unguarded=HYBRID_MIXER,
                            )
                        else:
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
                            if KV_APPEND and APPEND_DMA and kvc is not None:
                                # Spelled as DMAs naming @appendK / @appendV.
                                # The pass derives the shim S2MM half, and the
                                # SECOND run of air-annotate-append-barrier --
                                # after air-dma-to-channel, once both L3
                                # endpoints share the launch block -- puts the
                                # append->readback barrier back on it.
                                #
                                # The dst pattern is the launch-scope scatter the
                                # hand-written get had: group gi lands at its
                                # region slot. _kbase and the slot offset are
                                # launch-scope, so they are rebuilt here from the
                                # wave index (a herd is IsolatedFromAbove).
                                _kb_h = (
                                    arith.muli(kiv, idx(KV_LAYER))
                                    if kiv is not None
                                    else 0
                                )
                                _slot_h = (ATTN_L - 1) * REGION_W

                                def _apoff(extra):
                                    _e = extra + _slot_h
                                    if kiv is None:
                                        return _e
                                    return arith.addi(_kb_h, idx(_e)) if _e else _kb_h

                                DmaMemcpyNd(
                                    kvc,
                                    a_k,
                                    dst_offsets=[_apoff(0)],
                                    dst_sizes=[NGRP, REGION_W],
                                    dst_strides=[REGION_STRIDE, 1],
                                    src_offsets=[0],
                                    src_sizes=[DK_TOT_A],
                                    src_strides=[1],
                                    channel="appendK",
                                    channel_indices=[0],
                                    # Keep the shim BD where the hand-written get
                                    # had it: ahead of the readback that must
                                    # observe it.
                                    hoist_before="inKV_K",
                                )
                                DmaMemcpyNd(
                                    kvc,
                                    a_v,
                                    dst_offsets=[_apoff(_vreg_off(0))],
                                    dst_sizes=[NGRP, REGION_W],
                                    dst_strides=[REGION_STRIDE, 1],
                                    src_offsets=[0],
                                    src_sizes=[DK_TOT_A],
                                    src_strides=[1],
                                    channel="appendV",
                                    channel_indices=[0],
                                    hoist_after="appendK",
                                )
                            elif KV_APPEND:
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
                        if _own_qkv:
                            DeallocOp(a_qkv)
                        DeallocOp(a_lut)
                        DeallocOp(a_q)
                        DeallocOp(a_k)
                        DeallocOp(a_v)

                    # The q broadcast memtile buffer, hoisted ahead of the rope
                    # herd so it DOMINATES it and can be passed in as an operand.
                    # That is what lets rope's Q feed be spelled as a DMA naming
                    # @ropeQ: a DMA names both endpoints in one place, and this
                    # is the only place both are visible. Allocated at segment
                    # scope rather than inside the decode arm for the same
                    # reason -- the rope herd is a sibling of that arm.
                    _omtb_pre = None
                    _qmtb_pre = None
                    if ATTN_SUBSYS and ATTNO_DMA:
                        _omtb_pre = AllocOp(omt_l2, [], [])
                        _omtb_pre.operation.attributes["air.memtile_col"] = (
                            IntegerAttr.get(T.i32(), 5)
                        )
                    if ATTN_SUBSYS and ROPEQ_DMA:
                        _qmtb_pre = AllocOp(qmt_l2, [], [])
                        # Same pin the hand-written qmtb carries (#1969): the
                        # derived column is template-length dependent, and at
                        # ATTN_MAXL=128 qwen3-4b lands on mem_2_1 and times out.
                        # The ported path returns before that alloc is reached,
                        # so the attribute has to be repeated here or the port
                        # silently drops the fix.
                        _qmtb_pre.operation.attributes["air.memtile_col"] = (
                            IntegerAttr.get(T.i32(), 5)
                        )
                        _qmtb_pre.operation.attributes["air.no_split"] = UnitAttr.get()
                    # One (K, V) staging pair per attn column group. Per-iteration
                    # in the channel form; at segment scope here because a herd
                    # operand must dominate the herd, and the producer loop is a
                    # sibling of the arm the herd lives in.
                    _kvstage_pre = []
                    if ATTN_SUBSYS and KV_SPLIT and TOKV_DMA:
                        for _gcol, _gcus in ATTN_COL_GROUPS:
                            _pair = []
                            for _ in range(2):
                                _b = AllocOp(kvblk_l2, [], [])
                                _b.operation.attributes["air.memtile_col"] = (
                                    IntegerAttr.get(T.i32(), _gcol)
                                )
                                _pair.append(_b)
                            _kvstage_pre.append(_pair)
                    if ATTN_SUBSYS and not HYBRID_MIXER:
                        # BUG FIX (later43c): the rope arm MUST track the mode like
                        # proj/rms (0 in vocab). Hardcoded 1 kept rope in _dec()
                        # during vocab -> it stalled on the dest0 QKV gets (never
                        # produced in vocab) and never emitted the appendK/appendV
                        # the LM launch waits on -> TIMEOUT.
                        _arm_rope = _seg_arm

                        # The wave index rides along on its OWN terms, not on the
                        # KV cache's. @appendK wants both; @ropeLUT wants only the
                        # index, because a per-layer rope_w slab is offset by it.
                        # Coupling them kept the LUT hand-written on every qk-norm
                        # model for no reason but the operand list's shape.
                        _has_kvc = _seg_KVC is not None
                        _has_riv = _seg_iv is not None
                        _rope_opers = (
                            [_arm_rope, _seg_RMS]
                            + ([_seg_KVC] if _has_kvc else [])
                            + ([_seg_iv] if _has_riv else [])
                            + ([_qmtb_pre] if _qmtb_pre is not None else [])
                        )
                        _n_kv = (1 if _has_kvc else 0) + (1 if _has_riv else 0)

                        @herd(name="rope", sizes=[1, 1], operands=_rope_opers)
                        def rope_h(tx, ty, _sx, _sy, _arm, _rms, *_rest):
                            _kvc = _rest[0] if _has_kvc else None
                            _kiv = _rest[1 if _has_kvc else 0] if _has_riv else None
                            _qmt = _rest[_n_kv] if len(_rest) > _n_kv else None

                            def _dec():
                                _rope_body(_arm, rms=_rms, kvc=_kvc, kiv=_kiv, qmt=_qmt)
                                yield_([])

                            def _voc():
                                # gate-off 2026-07-15b: attn fully idle in vocab -> rope emits
                                # NOTHING (no dummy appendK/appendV; _uni_voc drains neither).
                                yield_([])

                            index_switch(
                                [],
                                arith.index_cast(idx_t, _arm),
                                [0],
                                case_body_builder=lambda op, i, cv: _voc(),
                                default_body_builder=lambda op: _dec(),
                            )

                        rope_h.attributes["link_with"] = StringAttr.get("rope.o")
                        rope_h.attributes["x_loc"] = IntegerAttr.get(T.i64(), RMS_PCOL)
                        rope_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), 3)

                    if CONV_MIXER:
                        # ===== CONV S1: ShortConv mixer, on TWO ADJACENT TILES =====
                        # in_proj emits [B | C | X] (3*CONV_DIM, X LAST). The mixer
                        # computes
                        #   BX = B*X ; conv = w0*BX[t-2] + w1*BX[t-1] + w2*BX ; y = C*conv
                        # and hands y (CONV_DIM) to the o-proj X, exactly where
                        # attention hands its gathered o.
                        #
                        # Tile ty=0 (STAGE) receives ph0's egress in CONV_WAVES waves
                        # of CONV_LAND and writes each into the assembled [B|C|X]; it
                        # also receives the taps. Tile ty=1 (CONV) owns the assembled
                        # input's storage and the carried state, computes, and emits.
                        #
                        # `a_bcx` and `a_w` are SEGMENT-SCOPE L1 handed to both tiles as
                        # herd operands, so they cross the tile boundary as neighbour
                        # memory rather than DMA -- the reference's dataflow, and the
                        # thing every deadlocking full-width build was missing. The
                        # attention path shares its scores buffer the same way.
                        a_mix = AllocOp(convmix_l1, [], [])
                        # _core_arm, not _seg_arm: this herd's arm IS the
                        # per-layer IS_ATTN and has to survive to runtime.
                        _arm_conv = _core_arm

                        # The hybrid runs rope inside THIS herd, so the ported
                        # append's operands have to reach it here. On the
                        # attention-only path they reach the rope herd instead
                        # (see _rope_opers) and this list is empty, which keeps
                        # conv_h's signature unchanged for every model that is
                        # not a hybrid.
                        # Same split as the rope herd's: @appendK wants the KV
                        # cache AND the wave index, @ropeLUT and @convW only the
                        # index. Coupling them made forcing APPEND_DMA off drop the
                        # index and trip _lut_slab_off's assertion -- caught by the
                        # attribute-parity gate, which is the arm that exercises it.
                        _has_ckvc = (APPEND_DMA or CONVST_DMA) and _seg_KVC is not None
                        _has_civ = _seg_iv is not None
                        _conv_append_opers = ([_seg_KVC] if _has_ckvc else []) + (
                            [_seg_iv] if _has_civ else []
                        )
                        _n_capp = len(_conv_append_opers)
                        _n_cqmt = 1 if _qmtb_pre is not None else 0
                        # RMS rides along too, for the ported @ropeLUT: a hybrid
                        # runs _rope_body on THIS herd's stage tile rather than on
                        # a dedicated rope herd, so the weight BO the LUT is read
                        # from has to be visible here. That absence, not anything
                        # about the LUT itself, is why the hybrid kept the
                        # hand-written pair.
                        _conv_rms_opers = (
                            [_seg_RMS]
                            if ((ROPELUT_DMA or CONVW_DMA) and _seg_RMS is not None)
                            else []
                        )
                        _conv_kv_opers = (
                            _conv_append_opers
                            + ([_qmtb_pre] if _qmtb_pre is not None else [])
                            + _conv_rms_opers
                        )

                        @herd(
                            name="conv",
                            # Two vertically adjacent tiles: ty=0 stages the ph0
                            # landings into the shared [B|C|X], ty=1 owns that
                            # buffer and computes. They hand it over as NEIGHBOUR
                            # MEMORY, not DMA -- see shortconv.cc.
                            sizes=[1, 2],
                            # KVC and the layer index ride along only for the
                            # ported append: the DMA names both endpoints in one
                            # place, so the L3 cache has to be visible inside the
                            # herd. A herd is IsolatedFromAbove, which is why the
                            # layer index comes in too -- the append offset is
                            # rebuilt from it here rather than carried in.
                            operands=[a_mix.result, _arm_conv] + _conv_kv_opers,
                        )
                        def conv_h(tx, ty, _sx, _sy, mix, _arm, *_ckv):
                            def _stage_ingest():
                                """Land ph0's whole egress, WHATEVER the layer type.

                                These gets stay OUT of the arm switch on purpose.
                                air-annotate-packet-ids proves @outY is a demux by
                                checking that its per-destination get volumes sum
                                to the put volume; it cannot know that two
                                index_switch arms are mutually exclusive, so one
                                copy of the gets per arm doubles dest0's apparent
                                volume, the partition check fails, and every proj
                                core's `dest(...)` put is rejected with "its
                                routing domain has no demux". Ingest once, branch
                                after -- which is also what the reference does:
                                its ph0 S2MM lands in qkv_buffer for both layer
                                types and only the core code differs.

                                Both waves reuse ONE landing buffer each, so only
                                one is in flight (reference rope.cc).
                                """
                                _lands = [
                                    AllocOp(convland_l1, [], [])
                                    for _ in range(CONV_WAVES)
                                ]
                                _seq = [
                                    _wd
                                    for _wd in MIXER_DESTS
                                    for _ in range(CONV_WAVES // len(MIXER_DESTS))
                                ]

                                def _land(_wv):
                                    ChannelGet(
                                        "outY",
                                        _lands[_wv],
                                        indices=[idx(0), idx(_seq[_wv])],
                                        offsets=[idx(0)],
                                        sizes=[idx(CONV_LAND)],
                                        strides=[idx(1)],
                                    )

                                return _lands, _land

                            def _convw_get():
                                # Taps land in the tail of the shared buffer.
                                if CONVW_DMA and _crms is not None:
                                    DmaMemcpyNd(
                                        mix,
                                        _crms,
                                        dst_offsets=[CONV_IN],
                                        dst_sizes=[CONV_W_LEN],
                                        dst_strides=[1],
                                        src_offsets=[_lut_slab_off(_ckiv)],
                                        src_sizes=[CONV_W_LEN],
                                        src_strides=[1],
                                        channel="convW",
                                        channel_indices=[0],
                                        # The group is a CHAIN rooted at the last
                                        # hand-written endpoint before it:
                                        #   inKV_V <- ropeLUT <- convW
                                        #          <- convStIn <- convStOut
                                        # Two transfers sharing one hoist_after
                                        # anchor come out reversed, so each links
                                        # to the one it follows rather than all of
                                        # them to @inKV_V.
                                        hoist_after="ropeLUT",
                                    )
                                else:
                                    ChannelGet(
                                        "convW",
                                        mix,
                                        indices=[idx(0)],
                                        offsets=[idx(CONV_IN)],
                                        sizes=[idx(CONV_W_LEN)],
                                        strides=[idx(1)],
                                    )

                            def _stage(_lands):
                                # Taking the taps on THIS core also makes the stage
                                # tile a visible writer of @convmix, which is half of
                                # what tells AIR the buffer is a cross-core hand-off
                                # (the other half is shortconv_stage's write below).
                                #
                                # The stage tile wants THREE S2MM inputs -- the rope
                                # LUT, the taps, and the ph0 landings -- and a compute
                                # tile has TWO, so @ropeLUT and @convW are COLLAPSED
                                # onto one S2MM BD chain here. That chain fires in
                                # CHAIN ORDER rather than by packet id, so it relies
                                # on the shim delivering LUT-then-taps in strict
                                # alternation, which it does.
                                _convw_get()
                                # ONE staging call for ALL waves. AIR wraps each
                                # external call in its own acquire/release on the
                                # shared buffer, so a call PER WAVE makes the stage
                                # core signal CONV_WAVES times against a conv core
                                # that waits once -- measured: 1 wave passes, 2
                                # waves hang. The reference has the same split and
                                # releases its cross-tile lock once, after both
                                # copies.
                                if CONV_WAVES == 1:
                                    CallOp(
                                        shortconv_stage,
                                        [
                                            _lands[0],
                                            mix,
                                            arith.ConstantOp(
                                                IntegerAttr.get(i32, 0), None
                                            ).result,
                                            _arm,
                                        ],
                                    )
                                else:
                                    CallOp(
                                        shortconv_stage2,
                                        [_lands[0], _lands[1], mix, _arm],
                                    )

                            # The conv state slot, rebuilt inside the herd. _cst is
                            # a launch-scope value and a herd is IsolatedFromAbove,
                            # so the wave index comes in as an operand and the
                            # offset is recomputed from it -- the same friction
                            # @appendK's _apoff and @rmsW's _rbase_h have.
                            _cst_kvc = _ckv[0] if _has_ckvc else None
                            _cst_iv = _ckv[1 if _has_ckvc else 0] if _has_civ else None
                            _ckiv = _cst_iv
                            _crms = (
                                _ckv[_n_capp + _n_cqmt]
                                if len(_ckv) > _n_capp + _n_cqmt
                                else None
                            )

                            def _cst_h():
                                if _cst_iv is None:
                                    return CONV_ST_BASE
                                b = arith.muli(_cst_iv, idx(CONV_ST_LAYER))
                                return (
                                    arith.addi(b, idx(CONV_ST_BASE))
                                    if CONV_ST_BASE
                                    else b
                                )

                            def _mix():
                                a_st = AllocOp(convst_l1, [], [])
                                if CONVST_DMA and _cst_kvc is not None:
                                    DmaMemcpyNd(
                                        a_st,
                                        _cst_kvc,
                                        src_offsets=[_cst_h()],
                                        src_sizes=[2 * CONV_DIM],
                                        src_strides=[1],
                                        channel="convStIn",
                                        channel_indices=[0],
                                        # @convW, not @inW0c0: the hand-written pair
                                        # sits in a LATER wave than the decode arm's
                                        # first weight put, and first-occurrence
                                        # resolution would land it ~40 slots early.
                                        hoist_after="convW",
                                    )
                                else:
                                    ChannelGet(
                                        "convStIn",
                                        a_st,
                                        indices=[idx(0)],
                                        offsets=[idx(0)],
                                        sizes=[idx(2 * CONV_DIM)],
                                        strides=[idx(1)],
                                    )
                                a_y = AllocOp(convo_l1, [], [])
                                a_bx = AllocOp(convst_l1, [], [])
                                CallOp(shortconv_compute, [mix, a_st, a_y, a_bx, _arm])

                                # y -> o-proj X (the attnO slot); the shifted state ->
                                # arg4 as the next token's [BX(t-2)|BX(t-1)].
                                #
                                # A hybrid has no @convO: it writes the CUs' own
                                # four CU kv cores over @mixToCU, one slice each.
                                # The CU decides there whether its @attnO put
                                # carries this or the attention result, which
                                # keeps @attnO down to ONE producer tile per
                                # sub-channel -- the constraint that the first
                                # hybrid violated and hung on.
                                def _put_stout():
                                    if CONVST_DMA and _cst_kvc is not None:
                                        DmaMemcpyNd(
                                            _cst_kvc,
                                            a_bx,
                                            dst_offsets=[_cst_h()],
                                            dst_sizes=[2 * CONV_DIM],
                                            dst_strides=[1],
                                            src_offsets=[0],
                                            src_sizes=[2 * CONV_DIM],
                                            src_strides=[1],
                                            channel="convStOut",
                                            channel_indices=[0],
                                            # Chained off its own partner, not off
                                            # @convW: two transfers sharing one
                                            # anchor come out REVERSED, because each
                                            # is inserted directly after it.
                                            hoist_after="convStIn",
                                        )
                                    else:
                                        ChannelPut(
                                            "convStOut",
                                            a_bx,
                                            indices=[idx(0)],
                                            offsets=[idx(0)],
                                            sizes=[idx(2 * CONV_DIM)],
                                            strides=[idx(1)],
                                        )

                                if MIX_TO_CU:
                                    # NOT straight to @xnorm: that would be a
                                    # FOURTH same-id producer on the convergent X
                                    # ring and four do not route (rule 7). The
                                    # o-gather memtile is the ph1 producer for
                                    # both arms; its only input is @attnO, so the
                                    # mixer feeds the CUs and they choose. This
                                    # put is UNGATED -- the mixer core runs every
                                    # wave anyway (shortconv_compute branches on
                                    # _arm internally), so gating it would only
                                    # add an arm the CU's get would have to match.
                                    ChannelPut(
                                        "mixToCU",
                                        a_y,
                                        indices=[idx(0), idx(0)],
                                        offsets=[idx(0)],
                                        sizes=[idx(CONV_DIM)],
                                        strides=[idx(1)],
                                    )
                                    _put_stout()
                                else:
                                    ChannelPut(
                                        "convO",
                                        a_y,
                                        indices=[idx(0)],
                                        offsets=[idx(0)],
                                        sizes=[idx(CONV_DIM)],
                                        strides=[idx(1)],
                                    )
                                    _put_stout()
                                DeallocOp(a_st)
                                DeallocOp(a_y)
                                DeallocOp(a_bx)

                            def _by_tile(_top, _bot):
                                _is_stage = arith.cmpi(
                                    arith.CmpIPredicate.eq, ty, idx(0)
                                )
                                _if = IfOp(_is_stage, [], has_else=True)
                                with InsertionPoint(_if.thenRegion.blocks[0]):
                                    _top()
                                with InsertionPoint(_if.elseRegion.blocks[0]):
                                    _bot()
                                yield_([])

                            def _stage_tile():
                                # Ingest FIRST and unconditionally (see
                                # _stage_ingest), then branch on the layer type.
                                # arm 2 runs the attention mixer on this very
                                # tile, taking the first landing buffer as its
                                # QKV -- CONV_LAND == M, and the reference's
                                # qkv_buffer is likewise the landing buffer. The
                                # trailing landing buffers are the uniform
                                # schedule's padding and are simply not read.
                                _lands, _land = _stage_ingest()
                                # Land ph0, run rope, and only THEN land the rest.
                                #
                                # Landing every mixer phase before rope deadlocks a
                                # hybrid, and the failure is in the SHIM's
                                # instruction order, not on chip: the KV
                                # append/readback is issued at phase 0, so the shim
                                # blocks there waiting for rope's K/V -- while rope
                                # is waiting for phase 1's egress, whose weight feed
                                # the shim has not reached yet. One mixer phase
                                # (every attention-only model) cannot hit it, and a
                                # ShortConv-only model has no append to block on.
                                # This is also the reference's order: its ph0 S2MM
                                # fills qkv_buffer and rope runs on that.
                                _land(0)
                                _ai = arith.index_cast(idx_t, _arm)
                                # Rope is UNARMED: its consumer, the q-broadcast
                                # memtile, is segment scope and so runs on every
                                # decode wave. On a
                                # ShortConv wave rope reads the conv in_proj as if
                                # it were QKV and reads the conv taps as if they
                                # were its cos/sin table; the q/k/v that come out
                                # are meaningless and every consumer of them
                                # discards them. It is a 2048-element kernel on a
                                # tile whose wave is dominated by the ShortConv
                                # stage, so the cost is noise.
                                # UNARMED, as the comment above describes: the
                                # CUs downstream run on every decode wave, so
                                # q/k/v have to arrive on every decode wave too.
                                # rope_compute_hyb's own IS_ATTN branch is where
                                # the layer type is decided.
                                # kvc/kiv are the ported append's operands, empty
                                # unless APPEND_DMA is on for this build. Passed
                                # by keyword so the positional a_qkv stays
                                # _lands[0] -- the hybrid's rope reads the ph0
                                # landing as its QKV.
                                _rope_body(
                                    _arm,
                                    _lands[0],
                                    rms=(
                                        _ckv[_n_capp + _n_cqmt]
                                        if len(_ckv) > _n_capp + _n_cqmt
                                        else None
                                    ),
                                    kvc=_cst_kvc,
                                    kiv=_cst_iv,
                                    qmt=(_ckv[_n_capp] if _n_cqmt else None),
                                )
                                for _w in range(1, CONV_WAVES):
                                    _land(_w)
                                # Same rule one level up: the mixer core runs
                                # unarmed and waits on the shared [B|C|X] this
                                # call fills. Gate the stage and the mixer waits
                                # forever on an attention wave -- the cold
                                # deadlock every conv-present build hit.
                                _stage(_lands)
                                for _l in _lands:
                                    DeallocOp(_l)
                                yield_([])

                            def _conv_tile():
                                # UNARMED, as the @mixToCU put's own comment
                                # assumes: the four CUs take that broadcast on
                                # every decode wave, so the core that produces it
                                # has to run on every decode wave.
                                # shortconv_compute and conv_o_pass branch on
                                # _arm internally, so an attention wave's pass is
                                # computed and discarded. The conv core is idle on
                                # an attention layer and idle is free: it holds
                                # the shared [B|C|X|taps] buffer either way, so
                                # hosting both mixers costs no extra L1 and no
                                # extra tile.
                                _mix()
                                yield_([])  # scf.if region terminator

                            def _dec():
                                _by_tile(_stage_tile, _conv_tile)

                            def _voc():
                                yield_([])

                            index_switch(
                                [],
                                arith.index_cast(idx_t, _arm),
                                [0],
                                case_body_builder=lambda op, i, cv: _voc(),
                                default_body_builder=lambda op: _dec(),
                            )

                        if not HYBRID_MIXER:
                            conv_h.attributes["link_with"] = StringAttr.get(
                                "shortconv.o"
                            )
                        conv_h.attributes["x_loc"] = IntegerAttr.get(T.i64(), MIX_PCOL)
                        conv_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), MIX_PROW)

                    # A CONV-ONLY build has no attention at all: no KV cache, no
                    # flash-attention CUs, no append channels and no block loop -- its
                    # mixer output goes straight to the o-proj X. A HYBRID build has
                    # the whole subsystem and gates it on the arm, layer by layer.
                    if ATTN_SUBSYS:
                        # ===== ATTN S3a: 1-CU flash attention (reference tile_3_2/3_3) =====
                        # Proven attn_iso qk/kv herd pair: s_shared (segment-scope L1) is
                        # shared cross-tile (qk writes scores, kv reads). q from rope (direct
                        # to qk), k/v from rope via KV staging memtile (split). L=1 decode =>
                        # 1 block; the 15 pad keys are masked by L inside the kernels. o ->
                        # attnO host drain (S3a verification; S4 routes o -> o-proj X).
                        # q broadcast memtile (reference mem_5_1): get rope q (2048),
                        # fan out per-CU 512 reordered (pack_q [8,8,8]/[8,64,1]).
                        def _qmtb_dec():
                            if _qmtb_pre is not None:
                                # Allocated ahead of the rope herd and filled by
                                # the @ropeQ DMA inside it, so the get here is
                                # derived and the hand-written one is gone. The
                                # buffer outlives the arm, so it is not
                                # deallocated here either.
                                _qmtb_fan(_qmtb_pre, dealloc=False)
                                return
                            qmtb = AllocOp(qmt_l2, [], [])
                            # Pinned: the derived column is template-length
                            # dependent (qwen3-4b at ATTN_MAXL=128 lands on
                            # mem_2_1, and that build times out on NPU2).
                            qmtb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), 5)
                            )
                            qmtb.operation.attributes["air.no_split"] = UnitAttr.get()
                            ChannelGet("ropeQ", qmtb, indices=[idx(0)])
                            _qmtb_fan(qmtb)

                        def _qmtb_fan(qmtb, dealloc=True):
                            # Suppress the hand-written fan only when the DMA
                            # form is actually in effect. _qmtb_pre is None on a
                            # HYBRID build (the staging buffer is allocated only
                            # for `ATTN_SUBSYS and not HYBRID_MIXER`), and there
                            # the consumer falls back to a ChannelGet -- keying
                            # this on the flag alone left those gets unpaired.
                            for c in range(
                                0
                                if (TOATTNQ_DMA and _qmtb_pre is not None)
                                else N_ATTN_CU
                            ):
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
                                    offsets=[
                                        idx(0),
                                        idx(c * Q_HEADS_PADDED_PER_CU),
                                        idx(0),
                                    ],
                                    sizes=[
                                        idx(DH // 8),
                                        idx(Q_HEADS_PADDED_PER_CU),
                                        idx(8),
                                    ],
                                    strides=[idx(8), idx(DH), idx(1)],
                                )
                            if dealloc:
                                DeallocOp(qmtb)

                        # gate-off 2026-07-15b: q-broadcast is decode-only (vocab attn idle).
                        if _seg_arm_i is not None:

                            def _q_voc():
                                yield_([])

                            def _q_dec():
                                _qmtb_dec()
                                yield_([])

                            if HYBRID_MIXER:
                                # Segment scope => wave-invariant (the arm folds
                                # to wave 0 and the other branches are erased).
                                # So the q broadcast runs on EVERY decode wave,
                                # which in turn forces rope to run on every wave
                                # to feed it. Rope is a 2048-element kernel and
                                # an on-chip broadcast; on a ShortConv wave it
                                # reads the conv in_proj as if it were QKV and
                                # the result is dropped by the CUs.
                                _qmtb_dec()
                            else:
                                _arm_only(_seg_arm_i, {2}, _qmtb_dec)
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
                                akb = AllocOp(ak_l2, [], [])
                                akb.operation.attributes["air.no_split"] = (
                                    UnitAttr.get()
                                )
                                akbs.append(akb)
                            for c in range(N_ATTN_CU):
                                avb = AllocOp(av_l2, [], [])
                                avb.operation.attributes["air.no_split"] = (
                                    UnitAttr.get()
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
                                ChannelPut(
                                    "toK",
                                    akbs[c],
                                    indices=[idx(c)],
                                    offsets=[idx(0), idx(0), idx(0)],
                                    sizes=[idx(KVPC_DH // 8), idx(16), idx(8)],
                                    strides=[idx(8), idx(KVPC_DH), idx(1)],
                                )
                                ChannelPut(
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
                                        _pre = (
                                            _kvstage_pre[_gi]
                                            if TOKV_DMA and _kvstage_pre
                                            else None
                                        )
                                        for _blk in for_(idx(0), _seg_rounds(), idx(1)):
                                            if _pre is not None:
                                                _kbuf, _vbuf = _pre
                                            else:
                                                _kbuf = AllocOp(kvblk_l2, [], [])
                                                _kbuf.operation.attributes[
                                                    "air.memtile_col"
                                                ] = IntegerAttr.get(T.i32(), col)
                                                _vbuf = AllocOp(kvblk_l2, [], [])
                                                _vbuf.operation.attributes[
                                                    "air.memtile_col"
                                                ] = IntegerAttr.get(T.i32(), col)
                                            ChannelGet(
                                                "inKV_K", _kbuf, indices=[idx(_gi)]
                                            )
                                            ChannelGet(
                                                "inKV_V", _vbuf, indices=[idx(_gi)]
                                            )
                                            if _pre is None:
                                                for _lc, _cc in enumerate(_cus):
                                                    ChannelPut(
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
                                                        strides=[
                                                            idx(8),
                                                            idx(_gw),
                                                            idx(1),
                                                        ],
                                                    )
                                                    ChannelPut(
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
                                            if _pre is None:
                                                DeallocOp(_kbuf)
                                                DeallocOp(_vbuf)
                                            yield_([])
                                        return
                                    # ROLLED (was Python for blk in range(ATTN_ROUNDS)): AIR for_
                                    # -> count-free 2-buffer ring on the memtile (mirror the
                                    # weight-fan) so large ATTN_L stays under the 16-BD limit.
                                    # Fresh kvb per iter (no_split, memtile_col) = the share-ring
                                    # pattern AIR lowers to next_bd rotation, not a repeat_count BD.
                                    for _blk in for_(idx(0), _seg_rounds(), idx(1)):
                                        kvb = AllocOp(kvblk_l2, [], [])
                                        kvb.operation.attributes["air.memtile_col"] = (
                                            IntegerAttr.get(T.i32(), col)
                                        )
                                        ChannelGet("inKV", kvb, indices=[idx(c)])
                                        ChannelPut(
                                            "toK",
                                            kvb,
                                            indices=[idx(c)],
                                            offsets=[idx(0), idx(0), idx(0)],
                                            sizes=[idx(KVPC_DH // 8), idx(16), idx(8)],
                                            strides=[idx(8), idx(KVPC_DH), idx(1)],
                                        )
                                        ChannelPut(
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
                                        DeallocOp(kvb)
                                        yield_([])

                                _gated = _seg_arm_i is not None
                                if _gated:

                                    def _rb_voc():
                                        yield_([])

                                    def _rb_dec():
                                        _reblock_dec()
                                        yield_([])

                                    if HYBRID_MIXER:
                                        # Segment scope => wave-invariant, so
                                        # the KV readback runs on EVERY decode
                                        # wave. That is the hybrid's one real
                                        # cost (+40 MB/token at ctx 2048, ~6%).
                                        # It only works because its producer
                                        # (the shim inKV feed) and its consumer
                                        # (the CUs) are ungated too -- an
                                        # ungated memtile between a gated
                                        # producer and a gated consumer is the
                                        # configuration that deadlocked even
                                        # with zero attention waves in the
                                        # build.
                                        _reblock_dec()
                                    else:
                                        _arm_only(_seg_arm_i, {2}, _reblock_dec)
                                else:
                                    _reblock_dec()

                                def _core_rounds(Lh):
                                    """ceil(Lh/16) as a core-side loop bound.

                                    Lh is the RTP-L herd block-arg, so this is opaque to
                                    folding and survives to core codegen as a real runtime
                                    trip count -- the same count the shim's readback BD
                                    pushes, which is what keeps the core off a channel get
                                    that never arrives.
                                    """
                                    if not DYNSEQ_RTP:
                                        return idx(ATTN_ROUNDS)
                                    _s = arith.addi(
                                        Lh,
                                        arith.ConstantOp(
                                            IntegerAttr.get(i32, 15), None
                                        ).result,
                                    )
                                    _q = arith.divui(
                                        _s,
                                        arith.ConstantOp(
                                            IntegerAttr.get(i32, 16), None
                                        ).result,
                                    )
                                    return arith.index_cast(idx_t, _q)

                                # KV staging geometry for CU `_c`: its group's
                                # (K, V) pair, and where this CU's slice sits.
                                def _kv_src(_c, kvs):
                                    if not (TOKV_DMA and kvs):
                                        return None
                                    _g = ATTN_CU_GROUP[_c]
                                    _cus_g = ATTN_COL_GROUPS[_g][1]
                                    return (
                                        kvs[_g],
                                        _cus_g.index(_c),
                                        len(_cus_g) * KVPC_DH,
                                    )

                                def _qk_body(sh, Lh, _c, _arm=None, qmt=None, kvs=None):
                                    a_q = AllocOp(aq_l1, [], [])
                                    if TOATTNQ_DMA and qmt is not None:
                                        DmaMemcpyNd(
                                            a_q,
                                            qmt,
                                            src_offsets=[
                                                0,
                                                _c * Q_HEADS_PADDED_PER_CU,
                                                0,
                                            ],
                                            src_sizes=[
                                                DH // 8,
                                                Q_HEADS_PADDED_PER_CU,
                                                8,
                                            ],
                                            src_strides=[8, DH, 1],
                                            channel="toAttnQ",
                                            channel_indices=[_c],
                                            hoist_after="ropeQ",
                                        )
                                    else:
                                        ChannelGet("toAttnQ", a_q, indices=[idx(_c)])
                                    a_m = AllocOp(m_l1, [], [])
                                    a_cc = AllocOp(c_l1, [], [])
                                    # RUNTIME-L block count = ceil(Lh/16) from the RTP-L herd
                                    # block-arg (opaque region arg -> not const-folded -> stays a
                                    # runtime scf.for bound; the AIE core loops per the RTP-L the
                                    # shim writes, exactly like the reference's in-core rounds=(L+15)/16).
                                    # unrollSCFFors only unrolls all-constant loops, so this
                                    # survives to core codegen as a real runtime loop.
                                    _nblk_qk = _core_rounds(Lh)
                                    for _blk in for_(idx(0), _nblk_qk, idx(1)):
                                        # REQUIRED single-buffer: ping-pong would unroll-by-2 +
                                        # 1-remainder over a 3-buffer toK ring whose remainder reads
                                        # the wrong buffer vs the DMA rotation -> misaligned KV ->
                                        # garbage chat. Single-buffer is aligned.
                                        a_k = AllocOp(ak_l1, [], [])
                                        _src = _kv_src(_c, kvs)
                                        if _src is not None:
                                            _pair, _lc, _gw = _src
                                            DmaMemcpyNd(
                                                a_k,
                                                _pair[0],
                                                src_offsets=[0, 0, _lc * KVPC_DH],
                                                src_sizes=[KVPC_DH // 8, 16, 8],
                                                src_strides=[8, _gw, 1],
                                                channel="toK",
                                                channel_indices=[_c],
                                            )
                                        else:
                                            ChannelGet("toK", a_k, indices=[idx(_c)])
                                        blk_c = arith.index_cast(i32, _blk)
                                        CallOp(
                                            attn_qk_blk,
                                            [a_q, a_k, a_m, a_cc, sh, blk_c, Lh],
                                        )
                                        DeallocOp(a_k)
                                        yield_([])
                                    DeallocOp(a_q)
                                    DeallocOp(a_m)
                                    DeallocOp(a_cc)

                                def _kv_body(sh, Lh, _c, _arm=None, omt=None, kvs=None):
                                    a_y = AllocOp(y_l1, [], [])
                                    a_l = AllocOp(lden_l1, [], [])
                                    a_o = AllocOp(ao_l1, [], [])
                                    # RUNTIME-L block count = ceil(Lh/16) from the RTP-L herd
                                    # block-arg (opaque region arg -> not const-folded -> stays a
                                    # runtime scf.for bound; the AIE core loops per the RTP-L the
                                    # shim writes, exactly like the reference's in-core rounds=(L+15)/16).
                                    # unrollSCFFors only unrolls all-constant loops, so this
                                    # survives to core codegen as a real runtime loop.
                                    _nblk_qk = _core_rounds(Lh)
                                    for _blk in for_(idx(0), _nblk_qk, idx(1)):
                                        # REQUIRED single-buffer: ping-pong would unroll-by-2 +
                                        # 1-remainder over a 3-buffer toK ring whose remainder reads
                                        # the wrong buffer vs the DMA rotation -> misaligned KV ->
                                        # garbage chat. Single-buffer is aligned.
                                        a_k = AllocOp(ak_l1, [], [])
                                        _src = _kv_src(_c, kvs)
                                        if _src is not None:
                                            _pair, _lc, _gw = _src
                                            DmaMemcpyNd(
                                                a_k,
                                                _pair[0],
                                                src_offsets=[0, 0, _lc * KVPC_DH],
                                                src_sizes=[KVPC_DH // 8, 16, 8],
                                                src_strides=[8, _gw, 1],
                                                channel="toK",
                                                channel_indices=[_c],
                                            )
                                        else:
                                            ChannelGet("toK", a_k, indices=[idx(_c)])
                                        blk_c = arith.index_cast(i32, _blk)
                                        CallOp(
                                            attn_qk_blk,
                                            [a_q, a_k, a_m, a_cc, sh, blk_c, Lh],
                                        )
                                        DeallocOp(a_k)
                                        yield_([])
                                    DeallocOp(a_q)
                                    DeallocOp(a_m)
                                    DeallocOp(a_cc)

                                def _kv_body(sh, Lh, _c, _arm=None, omt=None, kvs=None):
                                    a_y = AllocOp(y_l1, [], [])
                                    a_l = AllocOp(lden_l1, [], [])
                                    a_o = AllocOp(ao_l1, [], [])
                                    # RUNTIME-L block count = ceil(Lh/16) (see _qk_body). Core
                                    # loops per RTP-L; matched by the shim readback push count.
                                    _nblk_kv = _core_rounds(Lh)
                                    for _blk in for_(idx(0), _nblk_kv, idx(1)):
                                        # REQUIRED single-buffer (see _qk_body): keeps toV/toK
                                        # consumption aligned with the DMA rotation (no unroll-by-2
                                        # remainder desync -> no misaligned KV).
                                        a_v = AllocOp(av_l1, [], [])
                                        _src = _kv_src(_c, kvs)
                                        if _src is not None:
                                            _pair, _lc, _gw = _src
                                            DmaMemcpyNd(
                                                a_v,
                                                _pair[1],
                                                src_offsets=[0, 0, 0, _lc * KVPC_DH],
                                                src_sizes=[2, KVPC_DH // 8, 8, 8],
                                                src_strides=[_gw * 8, 8, _gw, 1],
                                                channel="toV",
                                                channel_indices=[_c],
                                            )
                                        else:
                                            ChannelGet("toV", a_v, indices=[idx(_c)])
                                        blk_c = arith.index_cast(i32, _blk)
                                        CallOp(
                                            attn_kv_blk,
                                            [sh, a_v, a_y, a_l, blk_c, Lh],
                                        )
                                        DeallocOp(a_v)
                                        yield_([])
                                    CallOp(attn_kv_fin, [a_y, a_l, a_o])
                                    if MIX_TO_CU:
                                        # Take the mixer's broadcast and, on a
                                        # ShortConv wave, overwrite o with this
                                        # CU's slice. The get is UNGATED to match
                                        # the mixer's ungated put; the pick is the
                                        # kernel's `arm != 1` early-out, which is
                                        # the reference's IS_ATTN and the only
                                        # layer-type branch left in a core.
                                        a_mix = AllocOp(convo_l1, [], [])
                                        ChannelGet(
                                            "mixToCU",
                                            a_mix,
                                            # Broadcast POSITION, not a bundle
                                            # index: every CU receives the whole
                                            # CONV_DIM and keeps its own slice in
                                            # C. Indexing by _c is what makes
                                            # air-to-aie keep all four
                                            # destinations on the one flow.
                                            indices=[idx(0), idx(_c)],
                                            offsets=[idx(0)],
                                            sizes=[idx(CONV_DIM)],
                                            strides=[idx(1)],
                                        )
                                        CallOp(
                                            conv_o_pass,
                                            [
                                                a_mix,
                                                a_o,
                                                arith.ConstantOp(
                                                    IntegerAttr.get(i32, _c), None
                                                ).result,
                                                _arm,
                                            ],
                                        )
                                        DeallocOp(a_mix)

                                    def _put_o():
                                        if ATTNO_DMA and omt is not None:
                                            # All four name @attnO as anchor: the
                                            # first finds no endpoint yet and builds
                                            # the arm, the rest resolve to it and
                                            # land in that same arm.
                                            DmaMemcpyNd(
                                                omt,
                                                a_o,
                                                dst_offsets=[_c * DQ_PER_CU],
                                                dst_sizes=[DQ_PER_CU],
                                                dst_strides=[1],
                                                src_offsets=[0, 0, 0],
                                                src_sizes=[
                                                    Q_HEADS_PER_CU,
                                                    DH // 8,
                                                    8,
                                                ],
                                                src_strides=[
                                                    8,
                                                    Q_HEADS_PER_CU * 8,
                                                    1,
                                                ],
                                                channel="attnO",
                                                channel_indices=[_c],
                                                hoist_after="attnO",
                                            )
                                            return
                                        ChannelPut(
                                            "attnO",
                                            a_o,
                                            indices=[idx(_c)],
                                            # o un-interleave: kernel [q_head, dc, de] ->
                                            # natural (q_head, dh).
                                            # sizes=[Q_HEADS_PER_CU, DH//8, 8].
                                            offsets=[idx(0), idx(0), idx(0)],
                                            sizes=[
                                                idx(Q_HEADS_PER_CU),
                                                idx(DH // 8),
                                                idx(8),
                                            ],
                                            strides=[
                                                idx(8),
                                                idx(Q_HEADS_PER_CU * 8),
                                                idx(1),
                                            ],
                                        )

                                    # Unconditional on both decode arms: the CU is
                                    # the sole @attnO producer, and running the
                                    # block loop is also how it drains toAttnQ /
                                    # toK / toV, which the (necessarily
                                    # wave-invariant) KV memtile pushes every wave.
                                    # The wasted attention compute on a ShortConv
                                    # wave is largely free -- the CU is paced by
                                    # the KV readback DMA, which we pay anyway.
                                    _put_o()
                                    DeallocOp(a_o)
                                    DeallocOp(a_y)
                                    DeallocOp(a_l)

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

                        def _attn_leaf(
                            ty_arg,
                            cu,
                            sh,
                            Lh,
                            qk_ty,
                            _arm=None,
                            qmt=None,
                            omt=None,
                            kvs=None,
                        ):
                            _isqk = arith.cmpi(
                                arith.CmpIPredicate.eq, ty_arg, idx(qk_ty)
                            )
                            _if = IfOp(_isqk, [], has_else=True)
                            with InsertionPoint(_if.thenRegion.blocks[0]):
                                _qkb(sh, Lh, cu, _arm, qmt, kvs)
                                yield_([])
                            with InsertionPoint(_if.elseRegion.blocks[0]):
                                _kvb(sh, Lh, cu, _arm, omt, kvs)
                                yield_([])

                        def _attn_pairsel(
                            ty_arg,
                            shs,
                            Lh,
                            cu_lo,
                            cu_hi,
                            _arm=None,
                            qmt=None,
                            omt=None,
                            kvs=None,
                        ):
                            _lo = arith.cmpi(arith.CmpIPredicate.slt, ty_arg, idx(2))
                            _ifp = IfOp(_lo, [], has_else=True)
                            with InsertionPoint(_ifp.thenRegion.blocks[0]):
                                _attn_leaf(
                                    ty_arg,
                                    cu_lo,
                                    shs[cu_lo],
                                    Lh,
                                    0,
                                    _arm,
                                    qmt,
                                    omt,
                                    kvs,
                                )
                                yield_([])
                            with InsertionPoint(_ifp.elseRegion.blocks[0]):
                                _attn_leaf(
                                    ty_arg,
                                    cu_hi,
                                    shs[cu_hi],
                                    Lh,
                                    2,
                                    _arm,
                                    qmt,
                                    omt,
                                    kvs,
                                )
                                yield_([])

                        def _attn_col(
                            ty_arg, shs, Lh, ci, _arm=None, qmt=None, omt=None, kvs=None
                        ):
                            """The CU_PER_COL compute units of attn column `ci`,
                            selected by the herd's row index."""
                            _lo = ci * CU_PER_COL
                            if CU_PER_COL == 1:
                                _attn_leaf(
                                    ty_arg, _lo, shs[_lo], Lh, 0, _arm, qmt, omt, kvs
                                )
                            else:
                                _attn_pairsel(
                                    ty_arg, shs, Lh, _lo, _lo + 1, _arm, qmt, omt, kvs
                                )

                        def _attn_dec(
                            tx_arg,
                            ty_arg,
                            shs,
                            Lh,
                            _arm=None,
                            qmt=None,
                            omt=None,
                            kvs=None,
                        ):
                            if ATTN_COLS == 1:
                                _attn_col(ty_arg, shs, Lh, 0, _arm, qmt, omt, kvs)
                                return
                            _isc0 = arith.cmpi(arith.CmpIPredicate.eq, tx_arg, idx(0))
                            _ifc = IfOp(_isc0, [], has_else=True)
                            with InsertionPoint(_ifc.thenRegion.blocks[0]):
                                _attn_col(
                                    ty_arg, shs, Lh, 0, _arm, qmt, omt, kvs
                                )  # first attn col
                                yield_([])
                            with InsertionPoint(_ifc.elseRegion.blocks[0]):
                                _attn_col(
                                    ty_arg, shs, Lh, 1, _arm, qmt, omt, kvs
                                )  # second attn col
                                yield_([])

                        _has_qmt = bool(TOATTNQ_DMA and _qmtb_pre)
                        _has_omt = bool(ATTNO_DMA and _omtb_pre)
                        # The staging pairs go in group order, so a CU picks its
                        # own out with ATTN_CU_GROUP.
                        _kv_flat = [b for pair in _kvstage_pre for b in pair]
                        _attn_extra = (
                            ([_qmtb_pre] if _has_qmt else [])
                            + ([_omtb_pre] if _has_omt else [])
                            + _kv_flat
                        )

                        if _seg_arm_i is not None:

                            @herd(
                                name="attn_blk",
                                sizes=ATTN_HERD_SIZES,
                                operands=[t.result for t in _sh]
                                # _seg_arm, not _core_arm, for THIS herd only.
                                #
                                # @attnO's consumer sits in a segment-scope arm
                                # switching on a value derived from the layer
                                # INDEX. Hoisting the ported put rebuilds the
                                # herd-side guard out there, and a guard on
                                # _core_arm rebuilds as an index_cast of the
                                # segment's i32 RTP block argument, which does
                                # not survive the segment becoming an aie.device.
                                # Feeding the herd the segment-derived arm makes
                                # the rebuild a clone of a legal segment-scope
                                # chain, and no anchor is needed at all.
                                #
                                # The note on _core_arm above says deriving the
                                # layer type inside the segment once compiled
                                # both hybrids to identical flow sets with the
                                # CUs' @attnO puts erased. That does NOT
                                # reproduce here: the arm reaches air-to-aie as a
                                # live select chain rather than a folded
                                # constant, all four puts survive as DMAs, and
                                # lfm2_1_2b_q4nx -- mixed attention and ShortConv
                                # layers -- verifies topk 2/0 through its lit.
                                # No-op off a hybrid, where _core_arm IS
                                # _seg_arm already.
                                + [_Lc, _seg_arm] + _attn_extra,
                            )
                            def attn_blk(_tx, _ty, _sx, _sy, *_a):
                                shs = list(_a[:N_ATTN_CU])
                                Lh, _arm = _a[N_ATTN_CU], _a[N_ATTN_CU + 1]
                                _ei = N_ATTN_CU + 2
                                _qmt = _a[_ei] if _has_qmt else None
                                _ei += 1 if _has_qmt else 0
                                _omt = _a[_ei] if _has_omt else None
                                _ei += 1 if _has_omt else 0
                                _kvs = [
                                    list(_a[_ei + 2 * _g : _ei + 2 * _g + 2])
                                    for _g in range(len(_kvstage_pre))
                                ]

                                def _voc():
                                    yield_([])

                                def _dec():
                                    _attn_dec(_tx, _ty, shs, Lh)
                                    yield_([])

                                # A hybrid WITH A MIXER runs the CUs on BOTH decode
                                # arms. They
                                # are the sole @attnO producers and the o-gather
                                # memtile behind them is segment scope, so it
                                # cannot skip its gather on a ShortConv wave --
                                # gate the CUs off and it blocks, and @xnorm loses
                                # a chunk. Running them is also how @toAttnQ/@toK/
                                # @toV drain, which the (wave-invariant) KV
                                # readback pushes every wave. The wasted attention
                                # compute is largely free: the CU is paced by that
                                # readback DMA either way.
                                #
                                # Without a mixer there is nothing to produce on a
                                # ShortConv wave, so gate to attention only: a CU
                                # with nothing to do blocks on its input lock, and
                                # the KV memtile behind it blocks too --
                                # harmlessly, because the shim issues no KV
                                # traffic then and nothing else waits on either.
                                _arm_only(
                                    arith.index_cast(idx_t, _arm),
                                    {1, 2} if MIX_TO_CU else {2},
                                    lambda: _attn_dec(
                                        _tx, _ty, shs, Lh, _arm, _qmt, _omt, _kvs
                                    ),
                                )

                        else:

                            @herd(
                                name="attn_blk",
                                sizes=ATTN_HERD_SIZES,
                                operands=[t.result for t in _sh] + [_Lc],
                            )
                            def attn_blk(_tx, _ty, _sx, _sy, *_a):
                                _attn_dec(_tx, _ty, list(_a[:N_ATTN_CU]), _a[N_ATTN_CU])

                        attn_blk.attributes["x_loc"] = IntegerAttr.get(
                            T.i64(), ATTN_CU_LOC[0][0]
                        )
                        attn_blk.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

                    # o gather memtile (reference mem_5_1 o_buffer): gather the 4
                    # CUs' o (512 each, already natural [qh,dh] from the egress
                    # reorder) into 2048, then ONE egress -> host (oGathered). This
                    # is the reference o_buffer; the loop-close step routes it to
                    # mem_1_1 (id2) = o-proj X instead of host.
                    def _omtb_dec(_conv=CONV_MIXER, omtb=None):
                        if _omtb_pre is not None:
                            omtb = _omtb_pre
                        if omtb is None:
                            omtb = AllocOp(omt_l2, [], [])
                        # Stays pinned: gemma3-4b fails to place without it
                        # ('aie.masterset' op targets same destination DMA: 0).
                        omtb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(), 5
                        )

                        # loop close: gathered o (2048) is ph1 o-proj X, re-broadcast
                        # OPROJ_REFEED times into the convergent @xnorm, AFTER ph0 (rms)
                        # and BEFORE ph2. Reference mem_5_1 o_buffer -> mem_1_1 x_buffer.
                        # Only the SOURCE get is switched; the OPROJ_REFEED onto
                        # @xnorm below stays outside, emitted once. Duplicating it
                        # per arm would double @xnorm's static put volume against an
                        # X memtile that still gets it once -- the same volume-
                        # accounting trap that breaks the @outY demux proof, and
                        # here it would unbalance the xnorm convergence instead.
                        def _src_conv():
                            # The mixer emits its whole CONV_DIM output from one
                            # core, into the same o buffer attention's 4 CUs
                            # gather into.
                            ChannelGet(
                                "convO",
                                omtb,
                                indices=[idx(0)],
                                offsets=[idx(0)],
                                sizes=[idx(CONV_DIM)],
                                strides=[idx(1)],
                            )

                        def _src_attn():
                            # Same coupling as the @toAttnQ fan above: keyed on
                            # the buffer, not the flag, so a HYBRID build keeps its
                            # hand-written puts.
                            for c in range(
                                0
                                if (ATTNO_DMA and _omtb_pre is not None)
                                else N_ATTN_CU
                            ):
                                ChannelGet(
                                    "attnO",
                                    omtb,
                                    indices=[idx(c)],
                                    offsets=[idx(c * DQ_PER_CU)],
                                    sizes=[idx(DQ_PER_CU)],
                                    strides=[idx(1)],
                                )

                        if HYBRID_MIXER:
                            # WAVE-INVARIANT, and it must be: this is segment
                            # scope, so an arm gate here would fold to the wave-0
                            # arm and the other branch would be ERASED. It
                            # costs nothing --
                            # the four CU kv cores put @attnO on every decode
                            # wave whichever mixer ran, so the gather is the
                            # attention build's, unchanged.
                            _src_attn()
                        else:
                            _arm_only(_seg_arm_i, {2}, _src_attn, in_dec=True)
                            _arm_only(_seg_arm_i, {1}, _src_conv, in_dec=True)
                        refeed(
                            OPROJ_REFEED,
                            lambda: ChannelPut(
                                "xnorm",
                                omtb,
                                indices=[idx(0)],
                                offsets=[idx(0)],
                                sizes=[idx(N_ATTN_CU * DQ_PER_CU)],
                                strides=[idx(1)],
                            ),
                        )
                        if _omtb_pre is None:
                            # The hoisted buffer outlives this arm.
                            DeallocOp(omtb)

                    _skip_omtb = False
                    # gate-off 2026-07-15b: o-gather (attnO get + xnorm o-proj put) is
                    # DECODE-ONLY. In vocab attn produces no attnO, and _xc_voc already
                    # excludes OPROJ_REFEED, so the xnorm convergence stays balanced.
                    if _skip_omtb:
                        pass
                    elif _seg_arm_i is not None:

                        def _o_voc():
                            yield_([])

                        def _o_dec():
                            _omtb_dec()
                            yield_([])

                        index_switch(
                            [],
                            _seg_arm_i,
                            [0],
                            case_body_builder=lambda op, i, cv: _o_voc(),
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

                        # Allocated UP FRONT so it dominates the herd: @gluOut is
                        # spelled as a DMA on the core side, which names both
                        # endpoints in one place.
                        db = AllocOp(down_l2, [], [])
                        db.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(), DOWN_PCOL
                        )

                        @herd(
                            name="glu",
                            sizes=[1, 1],
                            operands=[_arm_glu] + ([db] if GLUOUT_DMA else []),
                        )
                        def glu_h(tx, ty, _sx, _sy, _arm, *_gd):
                            _gdb = _gd[0] if _gd else None

                            def _dec():
                                # FAITHFUL 2-slot ring (reproducer core_5_2: TWO glu_aie
                                # calls per loop iter, ping x_0/hid_0 + pong x_1/hid_1).
                                # Two distinct allocs per iter give air-to-aie a 2-deep
                                # S2MM/MM2S ring (lock init 2), matching tile_5_2 -- a
                                # rolled 1-call loop collapses to 1-slot (no overlap).
                                def _slice(_sl=None):
                                    gx = AllocOp(glu_x_l1, [], [])
                                    # get 1024 = TWO stripped demux packets DIRECTLY from
                                    # the id-demux dest (reproducer mem_1_1 DMA5 ->
                                    # tile_5_2 DMA0); no relay.
                                    ChannelGet(
                                        "outY",
                                        gx,
                                        indices=[idx(0), idx(GLU_DEST)],
                                        offsets=[idx(0)],
                                        sizes=[idx(GLU_SLICE)],
                                        strides=[idx(1)],
                                    )
                                    gh = AllocOp(glu_hid_l1, [], [])
                                    CallOp(glu_aie, [gh, gx, _arm])
                                    if GLUOUT_DMA and _gdb is not None:
                                        # Slot 2*s + ping/pong. The two slots per
                                        # iteration are adjacent, so the pair tiles
                                        # a contiguous run and the loop fold
                                        # recovers the single whole-buffer fill the
                                        # hand-written get had -- which is what the
                                        # refeed MM2S's counting lock is derived
                                        # from.
                                        DmaMemcpyNd(
                                            _gdb,
                                            gh,
                                            dst_offsets=[_sl],
                                            dst_sizes=[GLU_HID],
                                            dst_strides=[1],
                                            src_offsets=[0],
                                            src_sizes=[GLU_HID],
                                            src_strides=[1],
                                            channel="gluOut",
                                            channel_indices=[0],
                                        )
                                    else:
                                        ChannelPut(
                                            "gluOut",
                                            gh,
                                            offsets=[idx(0)],
                                            sizes=[idx(GLU_HID)],
                                            strides=[idx(1)],
                                        )
                                    DeallocOp(gx)
                                    DeallocOp(gh)

                                # An odd slice count desyncs that ring: the core's
                                # slot sequence restarts every layer while the ring's
                                # rotation carries over, so every other layer reads a
                                # stale slice. GLU_PKTS keeps the count even.
                                assert NGLU % 2 == 0, (
                                    f"odd NGLU={NGLU} desyncs the GLU BD ring "
                                    f"(GLU_PKTS={GLU_PKTS})"
                                )
                                for _s in for_(idx(0), idx(NGLU // 2), idx(1)):
                                    _base = arith.muli(_s, idx(2 * GLU_HID))
                                    _slice(_base)  # ping
                                    _slice(arith.addi(_base, idx(GLU_HID)))  # pong
                                    yield_([])

                                yield_([])

                            def _voc():
                                yield_([])

                            index_switch(
                                [],
                                arith.index_cast(idx_t, _arm),
                                [0],
                                case_body_builder=lambda op, i, cv: _voc(),
                                default_body_builder=lambda op: _dec(),
                            )

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
                        for _s in for_(idx(0), idx(NGLU), idx(1)):
                            soff = arith.muli(_s, idx(GLU_HID))
                            if not GLUOUT_DMA:
                                ChannelGet(
                                    "gluOut",
                                    db,
                                    offsets=[soff],
                                    sizes=[idx(GLU_HID)],
                                    strides=[idx(1)],
                                )
                            yield_([])
                        # re-broadcast the resident 8192 into the convergent X feed.
                        refeed(
                            DOWN_REFEED,
                            lambda: ChannelPut(
                                "xnorm",
                                db,
                                offsets=[0],
                                sizes=[GLU_OUT],
                                strides=[1],
                            ),
                        )
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

                            def _emit(a_acc, destv):
                                yb = AllocOp(ypair_l1, [], [])
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
                                    sizes=[idx(HDR + PAIR_PAY)],
                                    strides=[idx(1)],
                                    dest=destv,
                                )
                                DeallocOp(yb)
                                DeallocOp(a_acc)

                            _arm_i = arith.index_cast(idx_t, _arm)
                            _id4 = idx(DEST[OPROJ_PHASE])

                            def _sel(voc_val, dec_thunk, ty_):
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

                            nph_v = _sel(idx(1), lambda: idx(NPH), idx_t)
                            for ph in for_(idx(0), nph_v, idx(1)):
                                _ephv = ph
                                I2v = _sel(
                                    idx(VOCAB_I2),
                                    lambda: _psw(_ephv, i2c, idx_t),
                                    idx_t,
                                )
                                J2v = _sel(
                                    idx(VOCAB_J2),
                                    lambda: _psw(_ephv, j2c, idx_t),
                                    idx_t,
                                )
                                pktv = _sel(
                                    _id4, lambda: _psw(_ephv, pktc, idx_t), idx_t
                                )
                                for _v1 in for_(idx(0), I2v, idx(1)):
                                    for _e in range(PAIR_ROWS):  # 1 (non-paired)
                                        _emit(_gemv(J2v), pktv)
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
                            *_og,
                        ):
                            # Trailing operands, in the order _ops appends them.
                            _ogi = 0
                            _ogrp = None
                            if OUTA_DMA:
                                _ogrp = _og[_ogi]
                                _ogi += 1
                            _oxb = None
                            if INX_DMA:
                                _oxb = _og[_ogi]
                                _ogi += 1
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
                                    if INX_DMA:
                                        # Both endpoints in one op. The core's
                                        # window is its whole 256 block; the
                                        # memtile side is derived -- the 512
                                        # buffer holds two of it, so two puts,
                                        # ascending, placed after the @xnorm fill
                                        # that writes it.
                                        DmaMemcpyNd(
                                            a_x,
                                            _oxb,
                                            channel="inX",
                                            dynamic_channel_indices=[gcx, gcy],
                                        )
                                    else:
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

                            def _emit(a_acc, yb, pktv):
                                # Nested exact-IV select: column by tx==0, pair by ty<2,
                                # role by ty==const (even row = lead). Every guard is a
                                # DIRECT tile-IV comparison (IV==const / IV<const) so it
                                # folds per-tile at the air-to-aie clone -- reachableUnderIvs
                                # folds only those (NOT tx&&ty, tx*2+.., ty%2, ty/2) --
                                # keeping each pair's shared-L1 + owner-tile analysis exact.
                                # scf.if (not index_switch): air-dependency's graph builder
                                # has no IndexSwitchOp async case (Util/Dependency.cpp).
                                def _outa_dma(a_acc, buf, pp_c, pktv):
                                    # Where in the group buffer this emitter lands
                                    # is k = tx*PAIRS_PC + pp_c, and both the offset
                                    # and the LENGTH depend on it -- only k == 0
                                    # carries the two header words. A BD length
                                    # cannot be a runtime value, so specialise on
                                    # tx. The guard is a direct tile-IV comparison
                                    # and folds per-tile at the air-to-aie clone,
                                    # leaving one constant-extent descriptor per
                                    # core -- and it makes the logical column a
                                    # constant, so the sub-channel is static too.
                                    for _txc in range(NCX // N_GRP):
                                        _k = _txc * PAIRS_PC + pp_c
                                        _off = 0 if _k == 0 else HDR + _k * PAIR_PAY
                                        _sz = (HDR + PAIR_PAY) if _k == 0 else PAIR_PAY
                                        _ift = IfOp(
                                            arith.cmpi(
                                                arith.CmpIPredicate.eq, tx, idx(_txc)
                                            ),
                                            [],
                                            has_else=False,
                                        )
                                        with InsertionPoint(_ift.thenRegion.blocks[0]):
                                            # The flush comes INSIDE the guard,
                                            # with the transfer, not before it.
                                            # The lock placer brackets each
                                            # buffer-touching op on its own, so a
                                            # guard standing between the write and
                                            # the send splits one critical section
                                            # into two: the buffer is released to
                                            # the consumer once with no data, and
                                            # the consumer lock is signalled twice
                                            # per production. Siblings in one
                                            # region are one section, which is what
                                            # the hand-written put had.
                                            CallOp(flush_row, [a_acc, buf, c0i])
                                            DmaMemcpyNd(
                                                _ogrp,
                                                buf,
                                                dst_offsets=[_off],
                                                dst_sizes=[_sz],
                                                dst_strides=[1],
                                                src_offsets=[14],
                                                src_sizes=[HDR + PAIR_PAY],
                                                src_strides=[1],
                                                channel="outA",
                                                channel_indices=[
                                                    base_cx + _txc,
                                                    pp_c,
                                                ],
                                                dest=pktv,
                                                hoist_before="toMain",
                                            )
                                            yield_([])

                                def _role(bufs, lead_row, pp_c):
                                    _is_lead = arith.cmpi(
                                        arith.CmpIPredicate.eq, ty, idx(lead_row)
                                    )
                                    _if = IfOp(_is_lead, [], has_else=True)
                                    with InsertionPoint(_if.thenRegion.blocks[0]):
                                        if OUTA_DMA and _ogrp is not None:
                                            _outa_dma(a_acc, bufs[yb], pp_c, pktv)
                                        else:
                                            CallOp(flush_row, [a_acc, bufs[yb], c0i])
                                            ChannelPut(
                                                "outA",
                                                bufs[yb],
                                                indices=[gcx, idx(pp_c)],
                                                offsets=[idx(14)],
                                                sizes=[idx(HDR + PAIR_PAY)],
                                                strides=[idx(1)],
                                                dest=pktv,
                                            )
                                        yield_([])
                                    with InsertionPoint(_if.elseRegion.blocks[0]):
                                        CallOp(flush_row, [a_acc, bufs[yb], c1i])
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
                            _id4 = idx(DEST[OPROJ_PHASE])

                            def _sel(voc_val, dec_thunk, ty):
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

                            nph_v = _sel(idx(1), lambda: idx(NPH), idx_t)
                            for ph in for_(idx(0), nph_v, idx(1)):
                                _ephv = ph
                                I2v = _sel(
                                    idx(VOCAB_I2),
                                    lambda: _psw(_ephv, i2c, idx_t),
                                    idx_t,
                                )
                                J2v = _sel(
                                    idx(VOCAB_J2),
                                    lambda: _psw(_ephv, j2c, idx_t),
                                    idx_t,
                                )
                                pktv = _sel(
                                    _id4, lambda: _psw(_ephv, pktc, idx_t), idx_t
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
                                        _emit(_gemv(J2v, a_rc, _f), _e, pktv)
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
                            bufs = [AllocOp(ypair_l1, [], []) for _ in range(8)]
                            _ops = [b.result for b in bufs] + [_arm_proj]
                        # A block spans NCX//N_GRP contiguous logical columns and a
                        # group gathers exactly those, so the block-to-group map is
                        # the column division -- one buffer per herd, never two.
                        if OUTA_DMA:
                            _ops = _ops + [_grp_pre[base_cx // (NCX // N_GRP)]]
                        # The X memtile buffer, for the same reason: with INX_DMA
                        # the cores name it as the far end of their own transfer.
                        # One buffer for every block -- it IS one buffer.
                        if INX_DMA:
                            _ops = _ops + [_xb_pre]
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

                    OPROJ_RNDS = PAIR_ROWS * I2P[OPROJ_PHASE]  # 4 o-proj egress rounds
                    DOWN_RNDS = PAIR_ROWS * I2P[DOWN_PHASE]  # 4 down egress rounds

                    # per-token RTP ARM (the reference-faithful re-dispatch): scalar herd operand ->
                    # AIR emits __air_herd_rtp + __air_herd_lock acquired per token; the
                    # runtime re-arms it each dispatch so the core does 1 token/dispatch.
                    _arm_rms = _seg_arm

                    def _xn_refeed(buf, arm):
                        """Re-broadcast the normed X once per mixer phase."""

                        def _put():
                            ChannelPut(
                                "xnorm", buf, offsets=[0], sizes=[K], strides=[1]
                            )

                        refeed(XN_REFEED, _put)

                    def _rbase_h(_iv):
                        """a_iv * RMS_LAYER, recomputed inside the herd.

                        _rbase is a LAUNCH-scope value and a herd is
                        IsolatedFromAbove, so the @rmsW DMA cannot reference it;
                        the wave index comes in as a herd operand instead and the
                        offset is rebuilt here. Same friction @ropeLUT's
                        _rope_off had.
                        """
                        return 0 if _iv is None else arith.muli(_iv, idx(RMS_LAYER))

                    def _rbase_h2(_iv, extra):
                        """_rbase_h(_iv) + extra, in whichever form _iv is."""
                        b = _rbase_h(_iv)
                        if _iv is None:
                            return b + extra
                        return arith.addi(b, idx(extra)) if extra else b

                    def _rms_body(tx, ty, _sx, _sy, _arm, _x, _rms=None, _iv=None):
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
                            # @rmsX spelled as a DMA: air-dma-to-channel derives the shim
                            # put from it, so the hand-written launch-scope put is gone.
                            # hoist_before pins the derived put to the slot that put had --
                            # opening the arm, immediately ahead of @rmsW -- so the shim BD
                            # order this design depends on is unchanged.
                            DmaMemcpyNd(
                                a_xl,
                                _x,
                                src_offsets=[0],
                                src_sizes=[K],
                                src_strides=[1],
                                channel="rmsX",
                                channel_indices=[0],
                                hoist_before="rmsW",
                            )
                            a_wl = AllocOp(_rms_w_ty, [], [])
                            # @rmsW spelled as a DMA, same as @rmsX above: the pass
                            # derives the shim put, so the hand-written launch-scope
                            # put is gone. Anchored to @inW0c0 -- a channel that stays
                            # hand-written -- because @rmsX and @ropeLUT are themselves
                            # anchored to @rmsW, and anchoring @rmsW back onto @rmsX
                            # would make the chain cyclic. On POST_RMS that neighbour
                            # is @rmsW2, which follows every hand-written @rmsW get;
                            # @inW0c0 repeats once per wave and resolved to the wrong
                            # occurrence.
                            if RMSW_DMA and _rms is not None:
                                _fn_off = (
                                    UNI_DEC * RMS_LAYER
                                    + (UNI_DEC if ROPE_W_PER_LAYER else 1) * ROPE_W_LEN
                                )
                                DmaMemcpyNd(
                                    a_wl,
                                    _rms,
                                    src_offsets=[
                                        _fn_off - K if N_NORMS >= 4 else _fn_off
                                    ],
                                    src_sizes=[2 * K if N_NORMS >= 4 else K],
                                    src_strides=[1],
                                    channel="rmsW",
                                    channel_indices=[0],
                                    hoist_before="rmsW2" if POST_RMS else "inW0c0",
                                )
                            else:
                                ChannelGet("rmsW", a_wl, indices=[idx(0)])
                            if POST_RMS:
                                # consume the vocab dummy rmsW2 (see _uni_voc) so the
                                # shared rmsX/rmsW2 packet group has no vocab-mode hole.
                                a_w2l = AllocOp(_rms_w_ty, [], [])
                                if RMSW2_DMA and _rms is not None:
                                    DmaMemcpyNd(
                                        a_w2l,
                                        _rms,
                                        src_offsets=[0],
                                        src_sizes=[2 * K if N_NORMS >= 4 else K],
                                        src_strides=[1],
                                        channel="rmsW2",
                                        channel_indices=[0],
                                        # Vocab arm: always @inW0c0. There is no
                                        # @ropeLUT endpoint in this arm, and an
                                        # anchor that misses its arm resolves onto
                                        # the decode arm's copy, emitting the vocab
                                        # feed over there.
                                        hoist_before="inW0c0",
                                    )
                                else:
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
                                    lambda: ChannelPut(
                                        "xnorm",
                                        a_xnl,
                                        offsets=[0],
                                        sizes=[K],
                                        strides=[1],
                                    ),
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
                            _rms_decode_body(_arm, _x, _rms, _iv)
                            yield_([])  # index_switch default terminator

                        _arm_i = arith.index_cast(idx_t, _arm)
                        index_switch(
                            [],
                            _arm_i,
                            [0],
                            case_body_builder=lambda op, i, cv: _rms_lm_case(),
                            default_body_builder=lambda op: _rms_decode(),
                        )

                    def _rms_decode_body(_arm, _x, _rms=None, _iv=None):
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
                            # @rmsX spelled as a DMA: air-dma-to-channel derives the shim
                            # put from it, so the hand-written launch-scope put is gone.
                            # hoist_before pins the derived put to the slot that put had --
                            # opening the arm, immediately ahead of @rmsW -- so the shim BD
                            # order this design depends on is unchanged.
                            DmaMemcpyNd(
                                g_x,
                                _x,
                                src_offsets=[0],
                                src_sizes=[K],
                                src_strides=[1],
                                channel="rmsX",
                                channel_indices=[0],
                                hoist_before="rmsW",
                            )
                            g_wa = AllocOp(rms_w2k_l1, [], [])
                            # @rmsW spelled as a DMA, same as @rmsX above: the pass
                            # derives the shim put, so the hand-written launch-scope
                            # put is gone. Anchored to @inW0c0 -- a channel that stays
                            # hand-written -- because @rmsX and @ropeLUT are themselves
                            # anchored to @rmsW, and anchoring @rmsW back onto @rmsX
                            # would make the chain cyclic. On POST_RMS that neighbour
                            # is @rmsW2, which follows every hand-written @rmsW get;
                            # @inW0c0 repeats once per wave and resolved to the wrong
                            # occurrence.
                            if RMSW_DMA and _rms is not None:
                                DmaMemcpyNd(
                                    g_wa,
                                    _rms,
                                    src_offsets=[_rbase_h(_iv)],
                                    src_sizes=[2 * K],
                                    src_strides=[1],
                                    channel="rmsW",
                                    channel_indices=[0],
                                    hoist_before="rmsW2" if POST_RMS else "inW0c0",
                                )
                            else:
                                ChannelGet("rmsW", g_wa, indices=[idx(0)])
                            g_wb = AllocOp(rms_w2k_l1, [], [])
                            if RMSW2_DMA and _rms is not None:
                                DmaMemcpyNd(
                                    g_wb,
                                    _rms,
                                    src_offsets=[_rbase_h2(_iv, 2 * K)],
                                    src_sizes=[2 * K],
                                    src_strides=[1],
                                    channel="rmsW2",
                                    channel_indices=[0],
                                    hoist_before=_RMSW2_ANCHOR,
                                )
                            else:
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
                            _xn_refeed(g_xn, _arm)
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
                                lambda: ChannelPut(
                                    "xnorm", g_xn, offsets=[0], sizes=[K], strides=[1]
                                ),
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
                        # @rmsX spelled as a DMA: air-dma-to-channel derives the shim
                        # put from it, so the hand-written launch-scope put is gone.
                        # hoist_before pins the derived put to the slot that put had --
                        # opening the arm, immediately ahead of @rmsW -- so the shim BD
                        # order this design depends on is unchanged.
                        DmaMemcpyNd(
                            a_x,
                            _x,
                            src_offsets=[0],
                            src_sizes=[K],
                            src_strides=[1],
                            channel="rmsX",
                            channel_indices=[0],
                            hoist_before="rmsW",
                        )
                        a_w = AllocOp(rms_l1, [], [])
                        # @rmsW spelled as a DMA, same as @rmsX above: the pass
                        # derives the shim put, so the hand-written launch-scope
                        # put is gone. Anchored to @inW0c0 -- a channel that stays
                        # hand-written -- because @rmsX and @ropeLUT are themselves
                        # anchored to @rmsW, and anchoring @rmsW back onto @rmsX
                        # would make the chain cyclic. On POST_RMS that neighbour is
                        # @rmsW2, which follows every hand-written @rmsW get; @inW0c0
                        # repeats once per wave and resolved to the wrong occurrence.
                        if RMSW_DMA and _rms is not None:
                            DmaMemcpyNd(
                                a_w,
                                _rms,
                                src_offsets=[_rbase_h(_iv)],
                                src_sizes=[K],
                                src_strides=[1],
                                channel="rmsW",
                                channel_indices=[0],
                                hoist_before="rmsW2" if POST_RMS else "inW0c0",
                            )
                        else:
                            ChannelGet("rmsW", a_w, indices=[idx(0)])
                        a_w2 = None
                        if POST_RMS:
                            # post_attention_layernorm weight (own channel).
                            a_w2 = AllocOp(rms_l1, [], [])
                            if RMSW2_DMA and _rms is not None:
                                DmaMemcpyNd(
                                    a_w2,
                                    _rms,
                                    src_offsets=[_rbase_h2(_iv, K)],
                                    src_sizes=[K],
                                    src_strides=[1],
                                    channel="rmsW2",
                                    channel_indices=[0],
                                    hoist_before=_RMSW2_ANCHOR,
                                )
                            else:
                                ChannelGet("rmsW2", a_w2, indices=[idx(0)])
                        # step1: input layernorm -> X feed (re-fed RMS_REFEED via xnorm)
                        a_xn = AllocOp(rms_l1, [], [])
                        CallOp(rms_norm_aie, [a_xn, a_x, a_w, _arm])
                        _xn_refeed(a_xn, _arm)
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
                                lambda: ChannelPut(
                                    "xnorm",
                                    a_xn,
                                    offsets=[0],
                                    sizes=[K],
                                    strides=[1],
                                ),
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
                            CallOp(residual_add_aie, [a_r2, a_h, a_dn])
                            DeallocOp(a_h)
                            DeallocOp(a_dn)
                            # BD-COMPACTION: single full-size layerOut drain.
                            if LAYEROUT_DMA:
                                DmaMemcpyNd(
                                    _x,
                                    a_r2,
                                    dst_offsets=[0],
                                    dst_sizes=[DOWN_RNDS * PAYLOAD],
                                    dst_strides=[1],
                                    channel="layerOut",
                                    channel_indices=[0],
                                    hoist_after="appendV",
                                )
                            else:
                                ChannelPut(
                                    "layerOut",
                                    a_r2,
                                    offsets=[idx(0)],
                                    sizes=[idx(DOWN_RNDS * PAYLOAD)],
                                    strides=[idx(1)],
                                )
                            DeallocOp(a_r2)

                    # RMS and the wave index reach the rms herd because the @rmsW
                    # feed is spelled as an air.dma_memcpy_nd, and a DMA has to name
                    # both endpoints in one place. _seg_iv is absent in a
                    # single-layer build, where the slab offset is the constant 0.
                    _rms_opers = [_arm_rms, _seg_X] + (
                        [_seg_RMS] + ([_seg_iv] if _seg_iv is not None else [])
                        if RMSW_DMA or RMSW2_DMA
                        else []
                    )
                    rms_h = herd(name="rms", sizes=[1, 1], operands=_rms_opers)(
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
    # Print a module-level constant and stop, so a Makefile can hand the kernel
    # compile the builder's own value (see glu.cc's GLU_SLICE_EXPECTED). Inert
    # unless set, and deliberately before the import below: this must stay
    # runnable without XRT.
    _const = _os.environ.get("FUSED_DECODE_PRINT_CONST")
    if _const:
        print(globals()[_const])
        return 0

    module = build_module()
    _assert_channels_paired(module)

    # Emit-only hook: dump the built AIR MLIR and stop before the (expensive) NPU
    # compile. Used to byte-diff the IR across no-op refactors (e.g. the incremental
    # model-config parametrization) without an aircc/NPU build. Inert unless set.
    if _os.environ.get("FUSED_DECODE_EMIT_ONLY"):
        print(str(module))
        return 0

    # Imported here, not at the top of main: emitting the IR is supposed to stop
    # before anything that needs the runtime, and a CI job that only checks the
    # IR has no pyxrt.
    import pyxrt as xrt

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
