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
#        (aiecc.py --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host
#         --xclbin-kernel-name=MLIR_AIE --peano=$PEANO --no-xchesscc --no-xbridge)
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
#   16 hdr region + 2*32 payload) on the LEAD tile. The lead writes the packet id
#   into the header (proj_qmm_flush_hdr -> id@14, its row@16) and emits a 2-row
#   packet (offset 14, size 66); the partner writes its row (proj_qmm_flush_row
#   i=1 -> @48, no header) cross-tile into the SAME lead buffer. Output flows are
#   PACKET channels carrying the kernel-written id; the group memtile does the
#   asymmetric one-header gather (258 = hdr + 4*64), the main memtile a 2-slot
#   daisy chain (514), and ONE egress demuxes by id (id1 = QKV).
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
from air.dialects.scf import for_, yield_, index_switch, ParallelOp, ReduceOp, IfOp
from air.backend.xrt import XRTBackend


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
    ref_gemv_matrix,
)

# ============================ faithful config ===============================
# Reproducer: Llama-3.2-1B decode layer, single token. QKV proj = q(2048)+k(512)+
# v(512) = 3072 rows out of K=2048 model dim. 16 proj cores at the reference columns
# 0,1,6,7, rows 2..5.
M = 3072  # QKV proj output rows
K = 2048  # model dim (proj contraction)
NCX = 4  # proj columns
NCY = 4  # proj rows (2..5)
PCOL = [0, 1, 6, 7]  # physical proj columns
NBI = M // ROW_BLOCK  # 96 output row-blocks
NBJ = K // COL_BLOCK  # 8 col-blocks (256-wide each)

# Cascade pairs: per col cx, two pairs pp; lead cy=2*pp (rows 2/4), partner
# cy=2*pp+1 (rows 3/5). 2 GROUPS of 4 leads (group g = cols {2g,2g+1}); group
# memtiles at phys cols 0 and 6; main memtile at phys col 1.
PAIRS_PC = NCY // 2  # 2 cascade pairs per column
N_PAIRS = NCX * PAIRS_PC  # 8 emitter pairs
N_GRP = 2  # group memtiles
LEADS_PER_GRP = N_PAIRS // N_GRP  # 4
GRP_PCOL = [0, 6]  # phys cols of the 2 group memtiles
MAIN_PCOL = 1  # phys col of the main memtile (mem_tile_1_1)
# Faithful X-feed (reproducer core_2_2 + air.refeed_count): the rms producer core
# (tile_2_2, col2) normalizes raw X once and re-feeds it via an output-lock release
# of N (= REFEED) into a 512 x_buffer (col2 memtile) that broadcasts 256-blocks to
# the 16 proj cores. (the reference puts x_buffer on col1/mem_1_1, but col1 congestion makes
# AIR's weight-fan MM2S a repeat_count BD; the golden uses col2, kept here.)
RMS_PCOL = 2  # rms producer core + X memtile column
XMT_PCOL = RMS_PCOL

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
N_ATTN_CU = 4  # fixed the reference dimension (4 CUs = 32 q heads)
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
# ALWAYS single-buffered (air.disable_ping_pong, set unconditionally below): ping-pong would
# unroll-by-2 + 1-remainder over a 3-buffer toK/toV ring whose remainder reads the wrong buffer vs
# the DMA rotation -> misaligned KV -> coherent first token then garbage chat.
# DECODE_RB_ROUNDS overrides the shim KV-readback nd-DMA outer block count (default ATTN_ROUNDS).
# Used to (a) locate the readback-count word in insts.bin by diffing two builds, and (b) let the
# host patch it to ceil(L/16) per token so the shim pushes exactly what the runtime core consumes.
RB_ROUNDS = int(_os.environ.get("DECODE_RB_ROUNDS", str((ATTN_L + 15) // 16)))
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
KV_PER_CU = 2
DH_A = 64
KVPC_DH = KV_PER_CU * DH_A  # 128
DK_TOT_A = N_ATTN_CU * KVPC_DH  # all-CU K (or V) width
KVSZ_TOK = 2 * DK_TOT_A  # per-token K ++ V (all heads)
ATTN_MAXL = ATTN_ROUNDS * 16  # padded context (compile-time block count)
APPEND_OFF = (ATTN_L - 1) * KVSZ_TOK  # this token's slot in the cache
# the reference-faithful on-device KV append: the rope core writes this token's roped-K/raw-V
# into the DDR cache (appendK/appendV S2MM -> KVC at slot L-1 = the reference _receive_kv_cache),
# then the whole cache is read back for the block-loop attention (the reference _move_kv_cache).
# The append->readback RAW on the shared cache is ordered in the runtime sequence via
# air.await_appends (= the reference's dma_wait; AIRRtToNpu moves the append awaits before the
# tagged readback). Only for MULTIBLK (L>1); L=1 uses the trivial on-chip-KV path.
KV_APPEND = MULTIBLK
# the reference layer-chaining ABI: the layer output (res2 = new hidden states) is written
# IN-PLACE into arg0 (the hidden_states BO), so layer N's output == layer N+1's input
# in the same buffer -- matching the reference's decoding_layer (output S2MM back to x_arg_id,
# no separate output arg). Frees arg3 (== the reference's rope_rms slot).
# Reference 4-CU layout: attn cols 3,4 (CU0,1 col3 / CU2,3 col4), adjacent to q/o on
# mem_5_1 (col5). kv on mem_3_1/mem_4_1. (col4 freed by GLU->col5 relayout.)
ATTN_CU_LOC = [(3, 2, 3), (3, 4, 5), (4, 2, 3), (4, 4, 5)][:N_ATTN_CU]
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


NPH = 4
I2P = [3, 2, 16, 2]  # row-pair iters per phase
J2P = [4, 4, 4, 16]  # col-block pairs (2*J2 = NBJ = K/COL_BLOCK)
KIDP = [1, 4, 8, 4]  # one-hot mask-safe packet ids (down reuses o-proj id4)
DISTINCT_IDS = list(dict.fromkeys(KIDP))  # ordered-unique
NDEST = len(DISTINCT_IDS)
DEST = [DISTINCT_IDS.index(k) for k in KIDP]
DOWN_PHASE = NPH - 1
NBJ_PH = [2 * J2P[p] for p in range(NPH)]  # per-phase col-blocks: [8,8,8,32]
KPH = [NBJ_PH[p] * COL_BLOCK for p in range(NPH)]  # per-phase K: [2048,2048,2048,8192]

# Output wire layout (reproducer y_0_2_0 memref<80>, group 258, main 514).
HDR = 2  # wire header words (kernel writes id@elem14)
PAIR_ROWS = 2  # rows per emitted packet (lead + partner)
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
VOCAB_SIZE = 128256  # llama-3.2-1b (models/llama3.2-1b.h)
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
# keep it small so the feed op-count + shim BDs + refeed lock (<=~32) all fit. 14 ->
# RNDS 28, 448 rowblocks/chunk, 4032/448 = 9 dispatches, proven (cos 0.9979 argmax MATCH).
VOCAB_I2 = int(_os.environ.get("VOCAB_CHUNK_I2", "14"))
VOCAB_ROWBLKS = VOCAB_I2 * (NCX * NCY) * PAIR_ROWS  # rowblocks per chunk/dispatch
VOCAB_SIZE_PADDED = VOCAB_ROWBLKS * ROW_BLOCK  # logits per chunk (device drain size)
assert VOCAB_FULL_ROWBLKS % VOCAB_ROWBLKS == 0, "chunk must divide the full vocab"
N_VOCAB_CHUNKS = VOCAB_FULL_ROWBLKS // VOCAB_ROWBLKS  # host dispatches (9)
VOCAB_J2 = J2P[0]  # 4 (K=MODEL_DIM=2048 -> NBJ=8 col-blocks)
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
UNI_DEC = 16  # fixed for Llama-3.2-1B: decode waves in the unified sequence
UNI_LM = 9  # fixed for Llama-3.2-1B: lm-head waves in the unified sequence
UNI_WAVES = UNI_DEC + UNI_LM
# Wave-range override (keeps ABI/CDO fixed at UNI_DEC/UNI_LM; only restricts which
# waves the fused launch loop drives). Used to split the fused sequence into a
# decode-part [0,UNI_DEC) and a vocab-part [UNI_DEC,UNI_WAVES) that share ONE CDO,
# to test host-wait quiescence between decode and vocab on one xclbin.
UNI_WAVE_LO = int(_os.environ.get("UNI_WAVE_LO", "0"))
UNI_WAVE_HI = int(_os.environ.get("UNI_WAVE_HI", str(UNI_WAVES)))

ROUNDS_PER_PH = [I2P[p] * PAIR_ROWS for p in range(NPH)]  # y0,y1 per v1 -> 2*I2
N_ROUNDS = sum(ROUNDS_PER_PH)  # total egress rounds (phase0 6 + phase1 4 = 10)
# id-demux egress: the main MT MM2S emits each round's assembled packet carrying
# the kernel-written id; the switchbox routes id DISTINCT_IDS[p] -> dest p
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
GLU_ID = 8
GLU_DEST = DISTINCT_IDS.index(GLU_ID) if GLU_ID in DISTINCT_IDS else -1
GLU_PHASE = KIDP.index(GLU_ID) if GLU_ID in KIDP else -1
# #4 faithful residual stream: o-proj + down (shared id4 -> RMS_DEST) are CONSUMED by
# the rms core (residual1=input+o-proj -> h; residual2=h+down -> layer output), NOT
# drained via the deadlocking memtile relay. The down egresses as the layer output.
# #4 applies only to the full 4-phase proj (QKV, o-proj, gate-up, down) where o-proj
# (ph1) + down (ph3) share id4 -> consumed by the rms residual.
FULL4 = NPH == 4 and DOWN_PHASE == 3 and KIDP[:4] == [1, 4, 8, 4]
RMS_DEST = DEST[DOWN_PHASE] if FULL4 else -1
HOST_DRAIN = [p for p in range(NDEST) if p != GLU_DEST and p != RMS_DEST]
GLU_CHUNK = PAYLOAD  # 512 (gate-up packs up/gate interleaved in 512-row chunks)
GLU_SLICE = 2 * PAYLOAD  # 1024 = [up 512 | gate 512] (TWO demux packets/BD)
GLU_HID = PAYLOAD  # 512 out per 1024 slice
NGLU = ROUNDS_PER_DEST[GLU_DEST] // 2 if GLU_DEST >= 0 else 0  # 16 slices (2 rnd/slice)
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
RMS_LAYER = K + (K if POST_RMS else 0)  # rms weights / layer
KV_LAYER = ATTN_MAXL * KVSZ_TOK  # KV cache / layer
Y_LAYER = sum(ROUNDS_PER_DEST[p] * PAYLOAD for p in HOST_DRAIN if p != 0)  # Y / layer


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
        # rms weight (K). MULTIBLK appends a 64-elem identity rope LUT (cos=1,sin=0)
        # AFTER all NLAYERS rms slabs so the score-path test gets a KNOWN q (q_roped =
        # proj_q) WITHOUT corrupting rms_w[0:K] (which proj_q depends on). The LUT is
        # per-position (shared across layers). L=1 ABI unchanged (size K [+K][+64]).
        rms_l3 = MemRefType.get(
            [
                UNI_DEC * (K + (K if POST_RMS else 0))
                + (64 if MULTIBLK else 0)
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
        kvc_l3 = MemRefType.get([UNI_DEC * ATTN_MAXL * KVSZ_TOK], bf16)

        # ---- L1 buffers ----
        xblk_l1 = MemRefType.get([COL_BLOCK], bf16, memory_space=l1)  # 256 X chunk
        wblk_l1 = MemRefType.get([BLOCK_BF16], bf16, memory_space=l1)  # 2560 weight
        yacc_l1 = MemRefType.get([ROW_BLOCK], f32, memory_space=l1)  # accumulator
        ypair_l1 = MemRefType.get(
            [16 + PAIR_ROWS * ROW_BLOCK], bf16, memory_space=l1  # 80 shared
        )
        rms_l1 = MemRefType.get([K], bf16, memory_space=l1)  # rms in/out/weight (2048)
        glu_x_l1 = MemRefType.get([GLU_SLICE], bf16, memory_space=l1)  # 1024 [up|gate]
        glu_hid_l1 = MemRefType.get([GLU_HID], bf16, memory_space=l1)  # 512 silu*up
        # ATTN S1 rope (reference rope_compute): qkv(3072 QKV out)+lut(64) -> q(2048),
        # k(512), v(512) roped. tile_2_3.
        qkv_l1 = MemRefType.get([3072], bf16, memory_space=l1)
        ropeq_l1 = MemRefType.get([2048], bf16, memory_space=l1)
        ropekv_l1 = MemRefType.get([512], bf16, memory_space=l1)
        ropelut_l1 = MemRefType.get([64], bf16, memory_space=l1)
        # ATTN S3a flash-attn (1 CU; attn_iso proven shapes). DH=64, 8 Q heads,
        # 2 KV heads per CU -> DQ=OSZ=512, DK=128, k/v block 16x128, scores 192.
        aq_l1 = MemRefType.get([512], bf16, memory_space=l1)  # q per CU
        ak_l1 = MemRefType.get([2048], bf16, memory_space=l1)  # k block 16x128 flat
        av_l1 = MemRefType.get([2048], bf16, memory_space=l1)  # v block 16x128 flat
        as_l1 = MemRefType.get([192], bf16, memory_space=l1)  # shared scores
        ao_l1 = MemRefType.get([512], bf16, memory_space=l1)  # o per CU
        # KV block cache (attn_stream proven): SEPARATE K and V natural block
        # buffers [key16, kvh2, dh64] = 2048 each; memtile reorder -> pack_k/pack_v.
        ak_l2 = MemRefType.get([2048], bf16, memory_space=l2)
        av_l2 = MemRefType.get([2048], bf16, memory_space=l2)
        # QKV staging memtile (reference mem_1_1 role): assemble the 6 demux rounds
        # into a contiguous 3072, then ONE transfer to rope. Feeding rope's compute-
        # tile S2MM directly from the 6 packet rounds mis-aligned by 1 head (+64).
        qkvmt_l2 = MemRefType.get([3072], bf16, memory_space=l2)
        # q broadcast memtile (reference mem_5_1 q_buffer): rope sends q ONCE (2048),
        # the memtile fans out per-CU 512 (reordered). Direct rope->CU q puts cost 1
        # rope MM2S per CU -> only 2 MM2S available, so N>=2 starved k/v (deadlock).
        qmt_l2 = MemRefType.get([2048], bf16, memory_space=l2)
        # o gather memtile (reference mem_5_1 o_buffer): 4 CUs' o (512 each) gathered
        # into 2048, then ONE egress (-> host now; -> mem_1_1 o-proj X in the loop close).
        omt_l2 = MemRefType.get([2048], bf16, memory_space=l2)
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
        wfan_l2 = MemRefType.get([NCY * BLOCK_BF16], bf16, memory_space=l2)
        grp_l2 = MemRefType.get([GRP_ROWS], bf16, memory_space=l2)
        main_l2 = MemRefType.get([MAIN_ROWS], bf16, memory_space=l2)
        relay_l2 = MemRefType.get(
            [PAYLOAD], bf16, memory_space=l2
        )  # demux relay (stripped)
        down_l2 = MemRefType.get([GLU_OUT], bf16, memory_space=l2)  # GLU out accumulate
        # relay memtile columns for the id-demux dests (free cols, not proj/X/MT).
        # GLU dest (gate-up) goes DIRECT to the GLU tile (no relay).
        RELAY_COLS = [3, 5, 4][:NDEST]

        # ---- kernels ----
        zero = FuncOp("proj_qmm_zero", ([yacc_l1, i32], []), visibility="private")
        zero.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        acc256 = FuncOp(
            "proj_qmm_acc256", ([xblk_l1, wblk_l1, yacc_l1], []), visibility="private"
        )
        acc256.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        flush_hdr = FuncOp(
            "proj_qmm_flush_hdr", ([yacc_l1, ypair_l1, i32], []), visibility="private"
        )
        flush_hdr.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        flush_row = FuncOp(
            "proj_qmm_flush_row", ([yacc_l1, ypair_l1, i32], []), visibility="private"
        )
        flush_row.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        # input-layernorm producer kernel (reproducer rms_norm_aie_hdr, lock-free).
        rms_norm_aie = FuncOp(
            "rms_norm_aie", ([rms_l1, rms_l1, rms_l1, i32], []), visibility="private"
        )
        rms_norm_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
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
        # Multi-block (ATTN_L>1) flash-attention: reproducer model A. The block
        # COMPUTE (attn_qk_blk/attn_kv_blk/attn_kv_fin) is proven (attn_iso PASS
        # L=16..128); online-softmax state lives in L1 and persists across the
        # in-core block loop (reset on blk==0 inside the kernels).
        #   qk: m_state(16 bf16 running max) + c_state(8 f32)
        #   kv: y_state(512 f32 accumulator) + l_state(16 f32 denominator)
        m_l1 = MemRefType.get([16], bf16, memory_space=l1)
        c_l1 = MemRefType.get([8], f32, memory_space=l1)
        y_l1 = MemRefType.get([512], f32, memory_space=l1)
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

        # ---- channels ----
        # Faithful X-feed: host raw X (@xy) + rms weight (@rmsin) -> rms core ->
        # xnorm (re-fed N times on-chip via air.refeed_count) -> X memtile (512) ->
        # 256-block broadcast to all 16 cores. (reproducer core_2_2 + mem_1_1 x_buffer)
        # #4 (FULL4): rmsX is PACKET so it converges with the id4 demux (o-proj+down)
        # on the rms core's S2MM0 -- the reference's tile_2_2 DMA0 receives @xy(id0)+id4
        # both as packets into one 2-slot ping-pong (input, then o-proj, then down).
        # Debug configs keep the original circuit + dedicated-channel rmsX.
        if FULL4:
            _rx = channel_decl("rmsX", size=[1], channel_type="npu_dma_packet")
            _rx.operation.attributes["air.shim_col"] = IntegerAttr.get(
                T.i32(), RMS_PCOL
            )
        else:
            _rx = channel_decl("rmsX", size=[1])
            _rx.operation.attributes["air.shim_col"] = IntegerAttr.get(
                T.i32(), RMS_PCOL
            )
            _rx.operation.attributes["air.dedicated_dma_channel"] = UnitAttr.get()
        _rw = channel_decl("rmsW", size=[1])
        if POST_RMS:
            # Separate channel for the post_attention_layernorm weight. A single
            # rmsW FIFO re-fed twice does NOT pair in AIR (both gets read the same
            # put -> decode diverges). The rms compute tile has only 2 S2MM
            # (rmsX-convergent + rmsW), so rmsW2 packet-muxes onto the rmsX S2MM;
            # to keep the vocab-active rmsX packet group hole-free, vocab feeds AND
            # consumes a dummy rmsW2 (see _uni_voc / _rms_lm_head).
            _rw2 = channel_decl("rmsW2", size=[1])
        _rw.operation.attributes["air.shim_col"] = IntegerAttr.get(T.i32(), RMS_PCOL)
        _rw.operation.attributes["air.dedicated_dma_channel"] = UnitAttr.get()
        # FAITHFUL convergent X feed (reproducer x_buffer DMA:3): ONE channel
        # carries BOTH the rmsnorm'd token X (phases 0..2) and the GLU-output X
        # (down phase), as convergent packet sources into ONE X-memtile S2MM, read
        # by ONE count-free feed loop. A single feed loop => one repeat count =>
        # air-to-aie infinite (count-free) BD mode, NO repeat_count (which is a
        # stale-rebroadcast deadlock). Two separate feed loops (the prior bug)
        # lowered to two repeat_count tasks. Packet (npu_dma_packet) so the two
        # producers (rms core L1 + down_buffer L2, time-disjoint, same id) converge.
        _xn = channel_decl("xnorm", size=[1], channel_type="npu_dma_packet")
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
        # rms-X producer-side re-feed (mechanism 1): the rms core releases its
        # output lock RMS_REFEED times. (The down_buffer producer re-broadcasts via
        # its own counting-lock from air.refeed_count on its alloc -- mechanism 2 --
        # and must NOT be multiplied by this channel count; air-to-aie skips the
        # channel re-feed for memtile producers.)
        _xn.operation.attributes["air.refeed_count"] = IntegerAttr.get(
            T.i32(), XN_REFEED  # LOOPCLOSE: rms emits ONLY ph0 (6); ph1/ph2/ph3 memtile
        )
        # (FAITHFUL ph2): no toBufP2 / buf_ph2 channel -- the rms core now emits ph2 X
        # = rmsnorm(x+oproj) directly on @xnorm with a PER-PUT air.refeed_count (32),
        # the per-step single-channel re-feed (reproducer core_2_2 step2; AIRToAIE
        # per-emission refeed override). ph0 uses the channel-level count (6).
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
                # the reference-faithful: rope K/V -> shim col3 (K) / col4 (V) S2MM -> DDR,
                # mirroring reference pkt14/15. PACKET (not circuit) so the append
                # can leave rope on its 2nd MM2S and fan to distinct cols. TWO
                # channels (one per col) because air.shim_col pins a single col
                # per channel DECL (compiler reads it from the decl). Pinning
                # cols 3/4 (attention CU cols; readback reuses them on MM2S) keeps
                # the append OFF rope's own col2 (whose congestion deadlocks the
                # front-end).
                _apK = channel_decl("appendK", size=[1], channel_type="npu_dma_packet")
                _apK.operation.attributes["air.shim_col"] = IntegerAttr.get(
                    T.i32(), ATTN_COL_GROUPS[0][0]
                )
                _apK.operation.attributes["air.tile_dma_channel"] = IntegerAttr.get(
                    T.i32(), 1
                )
                _apV = channel_decl("appendV", size=[1], channel_type="npu_dma_packet")
                _apV.operation.attributes["air.shim_col"] = IntegerAttr.get(
                    T.i32(), ATTN_COL_GROUPS[1][0]
                )
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
        # W: host (per col) -> group memtile -> NCY cores.
        channel_decl("inW", size=[NCX])
        _wL2 = channel_decl("wL2ToL1", size=[NCX, NCY])
        _wL2.operation.attributes["air.shared_resident_ring"] = UnitAttr.get()
        # Output: leads -> group MT -> main MT -> id-demux egress.
        _pin = ArrayAttr.get([IntegerAttr.get(i32, k) for k in DISTINCT_IDS])
        _outA = channel_decl(
            "outA", size=[NCX, PAIRS_PC], channel_type="npu_dma_packet"
        )
        _outA.operation.attributes["keep_pkt_header"] = UnitAttr.get()
        _outA.operation.attributes["packet_ids"] = _pin
        _toMain = channel_decl("toMain", size=[N_GRP], channel_type="npu_dma_packet")
        _toMain.operation.attributes["keep_pkt_header"] = UnitAttr.get()
        _toMain.operation.attributes["packet_ids"] = _pin
        # id-demux egress (reproducer mem_1_1 DMA5): the main MT emits each round's
        # assembled 514 packet (carrying the kernel-written id) on ONE MM2S; the
        # switchbox routes id DISTINCT_IDS[p] -> dest p (broadcast_shape=[1,NDEST]).
        # keep_pkt_header keeps each dest's header so the host can strip it.
        _outY = Channel("outY", size=[1, 1], broadcast_shape=[1, NDEST])
        _outY.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")
        # strip_pkt_header (faithful demux): route by the kernel header, then STRIP
        # it at every dest -> pure payload (PAYLOAD=512) delivered. Compiler sets
        # src_writes (no stamp) + keep=false on all split flows. The main MT PUT
        # stays MAIN_ROWS=514 (with header for routing); gets are PAYLOAD.
        _outY.operation.attributes["air.strip_pkt_header"] = UnitAttr.get()
        _outY.operation.attributes["packet_ids"] = _pin
        # per-dest host drain: dest p drains its phases' rounds.
        channel_decl("toShim", size=[NDEST])
        # #4: layer output (residual2 = h + down) drained to host from the rms core.
        if FULL4:
            _lo = channel_decl("layerOut", size=[1])
            _lo.operation.attributes["air.shim_col"] = IntegerAttr.get(T.i32(), 5)
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
        # the down_buffer fill -- mechanism 2, set via air.refeed_count on the
        # down_buffer alloc below), NOT drained to host.
        channel_decl("gluOut", size=[1])

        def idx(v):
            return arith.ConstantOp.create_index(v)

        def grp_leads(g):
            # the 4 lead (cx,pp) of group g, header-bearer first (k==0).
            out = []
            for lc in range(NCX // N_GRP):  # 2 cols / group
                for pp in range(PAIRS_PC):  # 2 pairs / col
                    out.append((2 * g + lc, pp))
            return out

        # MULTIBLK adds a 5th DDR arg (kv_cache) so the reference's append+readback (and the reference's
        # _gen_sequence) can drive it; the L=1 ABI (4 args) is unchanged when MULTIBLK
        # is off, preserving the bring-up/PASS interface.
        _fn_args = [x_l3, w_l3, rms_l3, y_l3] + ([kvc_l3] if MULTIBLK else [])

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
                KVC = _la[8] if MULTIBLK else None
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

                blk = BLOCK_BF16
                wstep = NCY * blk  # 10240 = one fan get
                for _layer in range(NLAYERS if a_iv is None else 1):
                    _wbase = _lb(W_LAYER)  # weights slab for this layer
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
                            _uarm = arith.select(_ucmp, _u1, _u0)
                        _uarm_i = arith.index_cast(idx_t, _uarm)

                        def _uni_voc():
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
                            ChannelPut("rmsX", X, offsets=[0], sizes=[K], strides=[1])
                            # real-lm-head final norm (model.norm.weight): a DEDICATED slot
                            # after the [in|post]*UNI_DEC rms slabs + 64-wide rope LUT, so the
                            # vocab rmsnorm uses the true final norm -- NOT layer-0's in_LN
                            # (mirrors decoding_layer's separate final_rms_weight).
                            ChannelPut(
                                "rmsW",
                                RMS,
                                offsets=[UNI_DEC * RMS_LAYER + 64],
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
                                    "rmsW2", RMS, offsets=[0], sizes=[K], strides=[1]
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
                            _half = _colspan // 2
                            for _cx in parallel_(NCX):
                                _cbase = arith.addi(
                                    _vwb, arith.muli(_cx, idx(_colspan))
                                )
                                ChannelPut(
                                    "inW",
                                    W,
                                    indices=[_cx],
                                    offsets=[_cbase],
                                    sizes=[_half],
                                    strides=[1],
                                )
                                ChannelPut(
                                    "inW",
                                    W,
                                    indices=[_cx],
                                    offsets=[arith.addi(_cbase, idx(_half))],
                                    sizes=[_colspan - _half],
                                    strides=[1],
                                )
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
                            # raw X (@xy) + rms weight (@rmsin) to the rms producer core; the
                            # on-chip rms normalizes + re-feeds X via air.refeed_count. X is
                            # in-place (offset 0 every layer -- the chained hidden state).
                            ChannelPut("rmsX", X, offsets=[0], sizes=[K], strides=[1])
                            ChannelPut(
                                "rmsW", RMS, offsets=[_rbase], sizes=[K], strides=[1]
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
                            # rope LUT: per-position, SHARED across layers; sits after all
                            # rms slabs in arg2 (layer-independent offset). UNIFIED sizes
                            # arg2 for UNI_DEC decode waves, so the LUT sits after UNI_DEC
                            # slabs (module-gen forces NLAYERS=1, which would misplace it).
                            _lut_off = (UNI_DEC * RMS_LAYER) if MULTIBLK else 0
                            ChannelPut(
                                "ropeLUT",
                                RMS,
                                offsets=[_lut_off],
                                sizes=[64],
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
                                # K -> shim col3 S2MM, V -> shim col4 S2MM (attention CU cols;
                                # cols pinned on the appendK/appendV channel DECLS via
                                # air.shim_col). Tag each append S2MM with air.append_barrier so
                                # AIRRtToNpu moves its completion await before the
                                # air.await_appends readback below (append->readback RAW on the
                                # shared DDR cache).
                                if KV_REGION:
                                    # Region-major append (= the reference _receive_kv_cache):
                                    # scatter this token's K (resp V) into the NGRP group
                                    # regions. Channel delivers [g0 K|g1 K|...] (REGION_W
                                    # each, CU-order); the nd write places group gi at its
                                    # region slot (ATTN_L-1)*REGION_W. outer dim=NGRP at
                                    # REGION_STRIDE, inner REGION_W contiguous.
                                    _apkG = ChannelGet(
                                        "appendK",
                                        KVC,
                                        indices=[idx(0)],
                                        offsets=[_loi(_kbase, (ATTN_L - 1) * REGION_W)],
                                        sizes=[idx(NGRP), idx(REGION_W)],
                                        strides=[idx(REGION_STRIDE), idx(1)],
                                    )
                                    _apkG.operation.attributes["air.append_barrier"] = (
                                        UnitAttr.get()
                                    )
                                    _apvG = ChannelGet(
                                        "appendV",
                                        KVC,
                                        indices=[idx(0)],
                                        offsets=[
                                            _loi(
                                                _kbase,
                                                _vreg_off(0) + (ATTN_L - 1) * REGION_W,
                                            )
                                        ],
                                        sizes=[idx(NGRP), idx(REGION_W)],
                                        strides=[idx(REGION_STRIDE), idx(1)],
                                    )
                                    _apvG.operation.attributes["air.append_barrier"] = (
                                        UnitAttr.get()
                                    )
                                    return

                            def _emit_readback(barrier=False, _kbase=_kbase):
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
                                    # locks. We reproduce that by tagging each readback put
                                    # air.shim_feed_no_pace: the AIR compiler then keeps it
                                    # OUT of the preserve-launch's depth-2 pacing (whose
                                    # await-on-drain would serialize K before V and deadlock
                                    # once a BD exceeds the ring depth) and lowers it to a
                                    # fire-and-free MM2S feed. With NRB=1 that is exactly
                                    # the reference's 2*NGRP (=4) whole-region contiguous transfers.
                                    _NRB = int(_os.environ.get("DECODE_KV_RB_NRB", "1"))
                                    _nb = RB_ROUNDS
                                    _cbk = (_nb + _NRB - 1) // _NRB  # blocks per chunk
                                    _first = True
                                    _ci = 0
                                    while _ci < _nb:
                                        _cb = min(_cbk, _nb - _ci)
                                        _coff = _ci * 16 * REGION_W
                                        for gi in range(NGRP):
                                            _pk = ChannelPut(
                                                "inKV_K",
                                                KVC,
                                                indices=[idx(gi)],
                                                offsets=[
                                                    _loi(_kbase, _kreg_off(gi) + _coff)
                                                ],
                                                sizes=[
                                                    idx(_cb),
                                                    idx(16),
                                                    idx(REGION_W),
                                                ],
                                                strides=[
                                                    idx(16 * REGION_W),
                                                    idx(REGION_W),
                                                    idx(1),
                                                ],
                                            )
                                            _pv = ChannelPut(
                                                "inKV_V",
                                                KVC,
                                                indices=[idx(gi)],
                                                offsets=[
                                                    _loi(_kbase, _vreg_off(gi) + _coff)
                                                ],
                                                sizes=[
                                                    idx(_cb),
                                                    idx(16),
                                                    idx(REGION_W),
                                                ],
                                                strides=[
                                                    idx(16 * REGION_W),
                                                    idx(REGION_W),
                                                    idx(1),
                                                ],
                                            )
                                            _pk.operation.attributes[
                                                "air.shim_feed_no_pace"
                                            ] = UnitAttr.get()
                                            _pv.operation.attributes[
                                                "air.shim_feed_no_pace"
                                            ] = UnitAttr.get()
                                            if barrier and _first:
                                                _pk.operation.attributes[
                                                    "air.await_appends"
                                                ] = UnitAttr.get()
                                                _first = False
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
                                _half = _colspan // 2
                                # Spatial fan over the NCX proj columns: the bundle index
                                # @inW[cx] must be an scf.parallel IV (canonical form; a
                                # temporal scf.for over a bundle index is a verifier error).
                                # Each column is one contiguous DDR block fed as TWO halves
                                # so the shim-dma coalescer merges+tags them
                                # (air.coalesced_shim_feed = the cross-channel phase barrier);
                                # a single put would skip coalescing and lose that barrier.
                                # air-to-aie spatially unrolls this to the per-column feeds.
                                _wcol0 = _lo(_wbase, woff)  # _wbase + woff (col 0 base)
                                for _cx in parallel_(NCX):
                                    _cbase = arith.addi(
                                        _wcol0, arith.muli(_cx, idx(_colspan))
                                    )
                                    ChannelPut(
                                        "inW",
                                        W,
                                        indices=[_cx],
                                        offsets=[_cbase],
                                        sizes=[_half],
                                        strides=[1],
                                    )
                                    ChannelPut(
                                        "inW",
                                        W,
                                        indices=[_cx],
                                        offsets=[arith.addi(_cbase, idx(_half))],
                                        sizes=[_colspan - _half],
                                        strides=[1],
                                    )
                                woff += NCX * per_col * blk
                                if MULTIBLK and p == 0:
                                    _emit_append()
                                    _emit_readback(barrier=True)
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
                            _out_bo = X
                            _out_base = 0
                            # BD-COMPACTION: single full-size drain (matches the rms single
                            # layerOut put) instead of LAYER_RNDS per-round gets.
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

                @segment(name="seg", operands=([a_iv] if a_iv is not None else []))
                def seg(*_sa):
                    _seg_iv = _sa[0] if _sa else None
                    if _seg_iv is not None:
                        _seg_cmp = arith.cmpi(
                            arith.CmpIPredicate.slt, _seg_iv, idx(UNI_DEC)
                        )
                        _seg_arm = arith.select(
                            _seg_cmp,
                            arith.ConstantOp(IntegerAttr.get(i32, 1), None).result,
                            arith.ConstantOp(IntegerAttr.get(i32, 0), None).result,
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
                    # times via @xnorm air.refeed_count) in 512 chunks -> broadcast
                    # 256-blocks. RMS_REFEED*(2048/512) gets. (reproducer core_2_2 +
                    # mem_1_1 x_buffer 512.)
                    def _feed_inX(src, total_chunks):
                        for _rc in for_(idx(0), idx(total_chunks), idx(1)):
                            xb = AllocOp(xmt_l2, [], [])
                            xb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), XMT_PCOL)
                            )
                            xb.operation.attributes["air.no_split"] = UnitAttr.get()
                            ChannelGet(
                                src, xb, offsets=[0], sizes=[2 * COL_BLOCK], strides=[1]
                            )
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
                    _xc_dec = (REFEED[0] + OPROJ_REFEED + GATEUP_REFEED) * (
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
                    for cx in range(NCX):
                        for _ in for_(idx(0), idx(W_FAN_STEPS), idx(1)):
                            wf = AllocOp(wfan_l2, [], [])
                            wf.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), PCOL[cx])
                            )
                            wf.operation.attributes["air.no_split"] = UnitAttr.get()
                            ChannelGet("inW", wf, indices=[idx(cx)])
                            for cy in range(NCY):
                                # 1D fixed-offset read (reproducer w_buffer MM2S
                                # shape) so AIR detects the 2-buffer rotation ->
                                # count-free next_bd ring (NOT a repeat_count BD).
                                ChannelPut(
                                    "wL2ToL1",
                                    wf,
                                    indices=[idx(cx), idx(cy)],
                                    offsets=[cy * BLOCK_BF16],
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
                    # egress (outY, packet) is demuxed by the kernel-written id and
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
                                grp.operation.attributes["air.no_split"] = (
                                    UnitAttr.get()
                                )
                                for k, (cx, pp) in enumerate(grp_leads(g)):
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
                                DeallocOp(grp)
                            ml = AllocOp(main_l2, [], [])
                            ml.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), MAIN_PCOL)
                            )
                            ml.operation.attributes["air.no_split"] = UnitAttr.get()
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
                        if p == 0:
                            continue  # QKV (dest0) consumed by the rope herd (below)
                        for _rp in for_(idx(0), idx(ROUNDS_PER_DEST[p]), idx(1)):
                            rb = AllocOp(relay_l2, [], [])
                            rb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), RELAY_COLS[p])
                            )
                            rb.operation.attributes["air.no_split"] = UnitAttr.get()
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
                        def _dec():
                            a_qkv = AllocOp(qkv_l1, [], [])
                            # the reference-faithful (layer.mlir mem_2_3 S2MM0): the rope COMPUTE
                            # core assembles the 6 id1/dest0 demux rounds (512 each)
                            # directly into its own L1 3072 buffer -- NO col-2 memtile
                            # relay. Identical 6x512 offset gets as the old qkvmt (each
                            # get consumes one stripped packet round), just landing in
                            # L1. In vocab mode id1 never flows so this compute-tile
                            # S2MM idles harmlessly.
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
                                sizes=[idx(2048)],
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
                                            offsets=[idx(c * 128)],
                                            sizes=[idx(128)],
                                            strides=[idx(1)],
                                        )
                                    for c in cus:
                                        ChannelPut(
                                            "toAttnKV",
                                            a_v,
                                            indices=[idx(gi)],
                                            offsets=[idx(c * 128)],
                                            sizes=[idx(128)],
                                            strides=[idx(1)],
                                        )
                            DeallocOp(a_qkv)
                            DeallocOp(a_lut)
                            DeallocOp(a_q)
                            DeallocOp(a_k)
                            DeallocOp(a_v)

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

                    # ===== ATTN S3a: 1-CU flash attention (reference tile_3_2/3_3) =====
                    # Proven attn_iso qk/kv herd pair: s_shared (segment-scope L1) is
                    # shared cross-tile (qk writes scores, kv reads). q from rope (direct
                    # to qk), k/v from rope via KV staging memtile (split). L=1 decode =>
                    # 1 block; the 15 pad keys are masked by L inside the kernels. o ->
                    # attnO host drain (S3a verification; S4 routes o -> o-proj X).
                    # q broadcast memtile (reference mem_5_1): get rope q (2048),
                    # fan out per-CU 512 reordered (pack_q [8,8,8]/[8,64,1]).
                    def _qmtb_dec():
                        qmtb = AllocOp(qmt_l2, [], [])
                        qmtb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(),
                            5,  # reference mem_5_1; free for N<=2 (kv on col3).
                            # N=4 needs attn cols 3,4 + GLU->tile_5_2 relayout (TODO).
                        )
                        qmtb.operation.attributes["air.no_split"] = UnitAttr.get()
                        ChannelGet("ropeQ", qmtb, indices=[idx(0)])
                        for c in range(N_ATTN_CU):
                            ChannelPut(
                                "toAttnQ",
                                qmtb,
                                # CU c reads q heads 8c..8c+7 = element base c*512.
                                # The linear base is sum(offset[i]*stride[i]); put the
                                # CU shift on the head dim (stride DH=64) as c*8 so the
                                # base = c*8*64 = c*512. (offset[0]=c*512 was WRONG:
                                # c*512*stride0(8) = c*4096, OOB for the 2048 buffer ->
                                # CU1-3 read garbage Q dh-order -> uniform attention on
                                # real per-dh K; invisible to constant-over-dh oracles.)
                                indices=[idx(c)],
                                offsets=[idx(0), idx(c * 8), idx(0)],
                                sizes=[idx(8), idx(8), idx(8)],
                                strides=[idx(8), idx(64), idx(1)],
                            )
                        DeallocOp(qmtb)

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
                            [0],
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
                            akb.operation.attributes["air.no_split"] = UnitAttr.get()
                            akbs.append(akb)
                        for c in range(N_ATTN_CU):
                            col = ATTN_CU_LOC[c][0]
                            avb = AllocOp(av_l2, [], [])
                            avb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), col)
                            )
                            avb.operation.attributes["air.no_split"] = UnitAttr.get()
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
                                    sizes=[idx(128)],
                                    strides=[idx(1)],
                                )
                            for c in cus:
                                ChannelGet(
                                    "toAttnKV",
                                    avbs[c],
                                    indices=[idx(gi)],
                                    offsets=[idx(0)],
                                    sizes=[idx(128)],
                                    strides=[idx(1)],
                                )
                        for c in range(N_ATTN_CU):
                            _pk = ChannelPut(
                                "toK",
                                akbs[c],
                                indices=[idx(c)],
                                offsets=[idx(0), idx(0), idx(0)],
                                sizes=[idx(16), idx(16), idx(8)],
                                strides=[idx(8), idx(128), idx(1)],
                            )
                            _pv = ChannelPut(
                                "toV",
                                avbs[c],
                                indices=[idx(c)],
                                offsets=[idx(0), idx(0), idx(0), idx(0)],
                                sizes=[idx(2), idx(16), idx(8), idx(8)],
                                strides=[idx(1024), idx(8), idx(128), idx(1)],
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
                            L_c = arith.ConstantOp(
                                IntegerAttr.get(i32, ATTN_L), None
                            ).result

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
                                    for _blk in for_(idx(0), idx(ATTN_ROUNDS), idx(1)):
                                        _kbuf = AllocOp(kvblk_l2, [], [])
                                        _kbuf.operation.attributes[
                                            "air.memtile_col"
                                        ] = IntegerAttr.get(T.i32(), col)
                                        _kbuf.operation.attributes["air.no_split"] = (
                                            UnitAttr.get()
                                        )
                                        _vbuf = AllocOp(kvblk_l2, [], [])
                                        _vbuf.operation.attributes[
                                            "air.memtile_col"
                                        ] = IntegerAttr.get(T.i32(), col)
                                        _vbuf.operation.attributes["air.no_split"] = (
                                            UnitAttr.get()
                                        )
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
                                                sizes=[idx(16), idx(16), idx(8)],
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
                                                sizes=[idx(2), idx(16), idx(8), idx(8)],
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
                                for _blk in for_(idx(0), idx(ATTN_ROUNDS), idx(1)):
                                    kvb = AllocOp(kvblk_l2, [], [])
                                    kvb.operation.attributes["air.memtile_col"] = (
                                        IntegerAttr.get(T.i32(), col)
                                    )
                                    kvb.operation.attributes["air.no_split"] = (
                                        UnitAttr.get()
                                    )
                                    ChannelGet("inKV", kvb, indices=[idx(c)])
                                    _pk = ChannelPut(
                                        "toK",
                                        kvb,
                                        indices=[idx(c)],
                                        offsets=[idx(0), idx(0), idx(0)],
                                        sizes=[idx(16), idx(16), idx(8)],
                                        strides=[idx(8), idx(128), idx(1)],
                                    )
                                    _pv = ChannelPut(
                                        "toV",
                                        kvb,
                                        indices=[idx(c)],
                                        offsets=[idx(2), idx(0), idx(0), idx(0)],
                                        sizes=[idx(2), idx(16), idx(8), idx(8)],
                                        strides=[idx(1024), idx(8), idx(128), idx(1)],
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
                                    [0],
                                    case_body_builder=lambda op, i, cv: _rb_voc(),
                                    default_body_builder=lambda op: _rb_dec(),
                                )
                            else:
                                _reblock_dec()

                            def _qk_body(sh, Lh, _c):
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
                                _nblk_qk = idx(
                                    ATTN_ROUNDS
                                )  # compile-time 128-block loop
                                for _blk in for_(idx(0), _nblk_qk, idx(1)):
                                    # REQUIRED single-buffer: ping-pong would unroll-by-2 +
                                    # 1-remainder over a 3-buffer toK ring whose remainder reads
                                    # the wrong buffer vs the DMA rotation -> misaligned KV ->
                                    # garbage chat. Single-buffer is aligned.
                                    _blk.owner.owner.attributes[
                                        "air.disable_ping_pong"
                                    ] = UnitAttr.get()
                                    a_k = AllocOp(ak_l1, [], [])
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

                            def _kv_body(sh, Lh, _c):
                                a_y = AllocOp(y_l1, [], [])
                                a_l = AllocOp(lden_l1, [], [])
                                a_o = AllocOp(ao_l1, [], [])
                                # RUNTIME-L block count = ceil(Lh/16) (see _qk_body). Core
                                # loops per RTP-L; matched by the shim readback push count.
                                _nblk_kv = idx(
                                    ATTN_ROUNDS
                                )  # compile-time 128-block loop
                                for _blk in for_(idx(0), _nblk_kv, idx(1)):
                                    # REQUIRED single-buffer (see _qk_body): keeps toV/toK
                                    # consumption aligned with the DMA rotation (no unroll-by-2
                                    # remainder desync -> no misaligned KV).
                                    _blk.owner.owner.attributes[
                                        "air.disable_ping_pong"
                                    ] = UnitAttr.get()
                                    a_v = AllocOp(av_l1, [], [])
                                    ChannelGet("toV", a_v, indices=[idx(_c)])
                                    blk_c = arith.index_cast(i32, _blk)
                                    CallOp(
                                        attn_kv_blk,
                                        [sh, a_v, a_y, a_l, blk_c, Lh],
                                    )
                                    DeallocOp(a_v)
                                    yield_([])
                                CallOp(attn_kv_fin, [a_y, a_l, a_o])
                                ChannelPut(
                                    "attnO",
                                    a_o,
                                    indices=[idx(_c)],
                                    offsets=[idx(0), idx(0), idx(0)],
                                    sizes=[idx(8), idx(8), idx(8)],
                                    strides=[idx(8), idx(64), idx(1)],
                                )
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
                                [0],
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
                        omtb.operation.attributes["air.no_split"] = UnitAttr.get()
                        # loop close: gathered o (2048) is ph1 o-proj X. As a MEMTILE
                        # producer it re-broadcasts OPROJ_REFEED times (mechanism-2)
                        # into the convergent @xnorm, AFTER ph0 (rms) and BEFORE ph2
                        # (buf_ph2). Reference mem_5_1 o_buffer -> mem_1_1 x_buffer.
                        omtb.operation.attributes["air.refeed_count"] = IntegerAttr.get(
                            T.i32(), OPROJ_REFEED
                        )
                        for c in range(N_ATTN_CU):
                            ChannelGet(
                                "attnO",
                                omtb,
                                indices=[idx(c)],
                                offsets=[idx(c * 512)],
                                sizes=[idx(512)],
                                strides=[idx(1)],
                            )
                        ChannelPut(
                            "xnorm",
                            omtb,
                            indices=[idx(0)],
                            offsets=[idx(0)],
                            sizes=[idx(N_ATTN_CU * 512)],
                            strides=[idx(1)],
                        )
                        DeallocOp(omtb)

                    # gate-off 2026-07-15b: o-gather (attnO get + xnorm o-proj put) is
                    # DECODE-ONLY. In vocab attn produces no attnO, and _xc_voc already
                    # excludes OPROJ_REFEED, so the xnorm convergence stays balanced.
                    if _seg_arm_i is not None:

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

                        @herd(name="glu", sizes=[1, 1], operands=[_arm_glu])
                        def glu_h(tx, ty, _sx, _sy, _arm):
                            def _dec():
                                # FAITHFUL 2-slot ring (reproducer core_5_2: TWO glu_aie
                                # calls per loop iter, ping x_0/hid_0 + pong x_1/hid_1).
                                # Two distinct allocs per iter give air-to-aie a 2-deep
                                # S2MM/MM2S ring (lock init 2), matching tile_5_2 -- a
                                # rolled 1-call loop collapses to 1-slot (no overlap).
                                def _slice():
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
                                    ChannelPut(
                                        "gluOut",
                                        gh,
                                        offsets=[idx(0)],
                                        sizes=[idx(GLU_HID)],
                                        strides=[idx(1)],
                                    )
                                    DeallocOp(gx)
                                    DeallocOp(gh)

                                for _s in for_(idx(0), idx(NGLU // 2), idx(1)):
                                    _slice()  # ping
                                    _slice()  # pong
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
                        # times into the convergent @xnorm. air.refeed_count on the
                        # down_buffer ALLOC (mechanism 2) makes air-to-aie emit a
                        # counting-lock-N on the fill (S2MM) side so the count-free MM2S
                        # re-reads the resident 8192 N times (reproducer down_buffer
                        # lock_5_1 init=4: one GLU fill -> 4 re-sends = the 4 down output
                        # row-blocks each re-reading all 8192). The X memtile chunks each
                        # 8192 into 16x512 -> inX for ph3.
                        db = AllocOp(down_l2, [], [])
                        db.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            # LOOPCLOSE: down on col4 (distinct from o on col5) so the
                            # @xnorm convergence doesn't merge o+down onto one MM2S ring.
                            T.i32(),
                            4,
                        )
                        db.operation.attributes["air.no_split"] = UnitAttr.get()
                        db.operation.attributes["air.refeed_count"] = IntegerAttr.get(
                            T.i32(), DOWN_REFEED
                        )
                        for _s in for_(idx(0), idx(NGLU), idx(1)):
                            soff = arith.muli(_s, idx(GLU_HID))
                            ChannelGet(
                                "gluOut",
                                db,
                                offsets=[soff],
                                sizes=[idx(GLU_HID)],
                                strides=[idx(1)],
                            )
                            yield_([])
                        # re-broadcast the resident 8192 into the convergent X feed
                        # (counting-lock-N from the alloc refeed_count above).
                        ChannelPut(
                            "xnorm",
                            db,
                            offsets=[0],
                            sizes=[GLU_OUT],
                            strides=[1],
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

                    def _core_blk(base_cx):
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
                            pktc = [
                                arith.ConstantOp(IntegerAttr.get(i32, k), None).result
                                for k in KIDP
                            ]
                            c1i = arith.ConstantOp(IntegerAttr.get(i32, 1), None).result

                            c2 = idx(2)

                            # ONE GEMV pass: 2*J2 j-steps, single inX/wL2ToL1 get site
                            # -> AIR 2-deep x_0/x_1 + w_0/w_1 rings (the reproducer's
                            # resident 2-buffer alternation). Separate a_x0/a_x1 gets
                            # would explode the core-mem BD count (>16).
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
                                        CallOp(flush_hdr, [a_acc, bufs[yb], pktv])
                                        ChannelPut(
                                            "outA",
                                            bufs[yb],
                                            indices=[gcx, idx(pp_c)],
                                            offsets=[idx(14)],
                                            sizes=[idx(HDR + PAIR_PAY)],
                                            strides=[idx(1)],
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
                            _id4 = arith.ConstantOp(
                                IntegerAttr.get(i32, KIDP[OPROJ_PHASE]), None
                            ).result

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
                                I2v = _sel(
                                    idx(VOCAB_I2), lambda: _psw(ph, i2c, idx_t), idx_t
                                )
                                J2v = _sel(
                                    idx(VOCAB_J2), lambda: _psw(ph, j2c, idx_t), idx_t
                                )
                                pktv = _sel(_id4, lambda: _psw(ph, pktc, i32), i32)
                                for _v1 in for_(idx(0), I2v, idx(1)):
                                    # two unrolled GEMV loops -> y_0 then y_1
                                    _emit(_gemv(J2v), 0, pktv)
                                    _emit(_gemv(J2v), 1, pktv)
                                    yield_([])  # v1
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
                        bufs = [AllocOp(ypair_l1, [], []) for _ in range(8)]
                        blk_h = herd(
                            name=f"proj_blk{blk}",
                            sizes=[2, 4],
                            operands=[b.result for b in bufs] + [_arm_proj],
                        )(_core_blk(base_cx))
                        blk_h.attributes["link_with"] = StringAttr.get("proj_qmm.o")
                        blk_h.attributes["x_loc"] = IntegerAttr.get(
                            T.i64(), PCOL[base_cx]
                        )
                        blk_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

                    # ===== rms producer core (reproducer core_2_2, tile_2_2) =====
                    # input-layernorm: raw X + rms weight -> normed X -> xnorm (re-fed
                    # on-chip REFEED times via air.refeed_count -> the X memtile).

                    OPROJ_RNDS = PAIR_ROWS * I2P[1]  # 4 o-proj egress rounds
                    DOWN_RNDS = PAIR_ROWS * I2P[DOWN_PHASE]  # 4 down egress rounds

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
                            a_xl = AllocOp(rms_l1, [], [])
                            ChannelGet("rmsX", a_xl, indices=[idx(0)])
                            a_wl = AllocOp(rms_l1, [], [])
                            ChannelGet("rmsW", a_wl, indices=[idx(0)])
                            if POST_RMS:
                                # consume the vocab dummy rmsW2 (see _uni_voc) so the
                                # shared rmsX/rmsW2 packet group has no vocab-mode hole.
                                a_w2l = AllocOp(rms_l1, [], [])
                                ChannelGet("rmsW2", a_w2l, indices=[idx(0)])
                                DeallocOp(a_w2l)
                            a_xnl = AllocOp(rms_l1, [], [])
                            # the reference-FAITHFUL x re-broadcast (was air.refeed_count=VOCAB_RNDS).
                            # the reference re-supplies x every vocab round through value-1 ping-pong
                            # rings and re-runs rms each persistent-loop iteration; the count
                            # lives in the loop + runtime, NEVER in a lock (all the reference lock
                            # values are 1/2/4). Our old air.refeed_count baked N=VOCAB_RNDS
                            # into ONE producer-side credit lock; the AIE-ML lock is 7-bit
                            # (max +63, xaie_locks_aieml.c 0x7F), so N>63 (I2>=32) made
                            # AcquireGreaterEqual(N) unsatisfiable -> DEADLOCK, forcing the
                            # 9-wave split (2*I2<=63). Here we mirror the reference: re-normalize + put
                            # xnorm PER ROUND (value-1; a_xnl is rewritten each round so AIR
                            # does not canonicalize the puts back into refeed_count). The
                            # sends are INTERLEAVED with the outY->layerOut relay (this ONE
                            # rms core both produces x and relays logits, unlike the reference's split
                            # tiles) so the producer never serializes ahead of the drain and
                            # backpressure-deadlocks: XN_PER_BLK (=K/PAYLOAD) x-sends per
                            # drained K-block. Total x-sends = VOCAB_RNDS, drain blocks =
                            # VOCAB_RNDS*PAYLOAD/K. rms recompute per round is negligible vs
                            # the vocab GEMV (matches the reference re-running rms per row-block).
                            # air.disable_ping_pong keeps single-buffer rings (a_xnl / a_v
                            # reused; fill==drain, trivially aligned; lm_head not perf-crit).
                            a_v = AllocOp(rms_l1, [], [])
                            _voc_blks_2k = VOCAB_RNDS * PAYLOAD // K
                            _xn_per_blk = K // PAYLOAD  # = VOCAB_RNDS // _voc_blks_2k
                            for _rv in for_(idx(0), idx(_voc_blks_2k), idx(1)):
                                _rv.owner.owner.attributes["air.disable_ping_pong"] = (
                                    UnitAttr.get()
                                )
                                for _xr in for_(idx(0), idx(_xn_per_blk), idx(1)):
                                    _xr.owner.owner.attributes[
                                        "air.disable_ping_pong"
                                    ] = UnitAttr.get()
                                    CallOp(rms_norm_aie, [a_xnl, a_xl, a_wl, _arm])
                                    _pl = ChannelPut(
                                        "xnorm",
                                        a_xnl,
                                        offsets=[0],
                                        sizes=[K],
                                        strides=[1],
                                    )
                                    # PER-PUT override of the channel-level
                                    # air.refeed_count (XN_REFEED=6, meant for the
                                    # decode ph0 path): each vocab round emits x
                                    # ONCE (value-1, n=1 hits the n>1 guard in
                                    # AIRToAIE -> no credit multiply). VOCAB_RNDS
                                    # distinct puts give VOCAB_RNDS broadcasts,
                                    # matching the X memtile's VOCAB_RNDS*(K/(2*
                                    # COL_BLOCK)) gets. Without this the puts inherit
                                    # refeed_count=6 -> over-broadcast.
                                    _pl.operation.attributes["air.refeed_count"] = (
                                        IntegerAttr.get(T.i32(), 1)
                                    )
                                    yield_([])
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

                    def _rms_decode_body(_arm):
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
                        ChannelPut("xnorm", a_xn, offsets=[0], sizes=[K], strides=[1])
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
                            # PER-PUT air.refeed_count = REFEED[ph2] (32). This is
                            # the per-step single-channel re-feed (ph0 x6 via the
                            # channel count, ph2 x32 via this per-put count) --
                            # replaces the invented buf_ph2 memtile stand-in.
                            CallOp(
                                rms_norm_aie,
                                [a_xn, a_h, a_w2 if POST_RMS else a_w, _arm],
                            )
                            _p2 = ChannelPut(
                                "xnorm",
                                a_xn,
                                offsets=[0],
                                sizes=[K],
                                strides=[1],
                            )
                            _p2.operation.attributes["air.refeed_count"] = (
                                IntegerAttr.get(T.i32(), REFEED[GATEUP_PHASE])
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
            for _iv in for_(idx(UNI_WAVE_LO), idx(UNI_WAVE_HI), idx(1)):
                launch(
                    sizes=[1, 1],
                    operands=list(_fa) + [_iv],
                    attributes={"air.preserve_shim_dma_order": UnitAttr.get()},
                )(launch_body)
                yield_([])

    return build()


def run():
    import pyxrt as xrt

    module = build_module()

    # use_lock_race_condition_fix_v2: emit the reference-style daisy-chained locks for the
    # shared-L2 fan-in (group/main asymmetric gather) -- matches the reproducer's
    # serialized 4-writer chain (mem_0_1 lock_0_1->_159->...->_162). Without it AIR
    # emits a counting lock whose writer/reader counts mismatch -> deadlock.
    backend = XRTBackend(
        omit_while_true_loop=False,
        output_format="xclbin",
        kernel_name="MLIR_AIE",
        stack_size=10240,
        use_lock_race_condition_fix_v2=True,
        coalesce_shim_dma=True,
    )
    print(
        f"[q4nx_decode] proj: M={M} K={K} {NCX}x{NCY}=16 cores, "
        f"8 cascade pairs, NPH={NPH} ids={DISTINCT_IDS}"
    )
    art = backend.compile(module, output_binary_name="decode", insts="decode.insts.bin")
    print(f"[q4nx_decode] emitted {art.output_binary} + {art.insts}")
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    exit(run())
