# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen2.5-3B decode layer — AIR reimplementation (SKELETON, stage 1: proj grid).
#
# WHY A SEPARATE FILE: fused_decode.py (the Llama-1B design) is explicitly NOT
# parametrized — it is hardcoded to the the Llama proj CASCADE topology (16 cores as
# 8 lead+partner pairs at cols {0,1,6,7}, group/main memtiles). The reference's Qwen2.5 decode
# layer uses a DIFFERENT device topology (verified by IR diff of the reference
# LLAMA_3_2_1B layer IR vs the reference QWEN2_3B layer IR):
#   proj = INDEPENDENT 4x4 SPMD grid, cols {2,3,4,5} x rows {2,3,4,5}, NO cascade;
#          each core writes only its own y buffer, packet-routed by phase-id to its
#          OWN per-column memtile (mem_tile_C_1, DMA 0-3 = rows 2-5).
#   rms=tile(1,2) rope=tile(1,3) glu=tile(6,2) attn=col7 (2 CUs).
# Keeping this in a separate builder leaves the Llama path byte-identical.
#
# SHIM/MEMTILE HOPPING (mirrors reference Qwen flows; cores NEVER touch shim directly):
#   X BROADCAST : host -> X memtile (col 1, reference mem_tile_1_1) -> inX broadcast to all 16.
#   WEIGHTS     : host -> per-column W memtile (col 2+cx) -> wL2ToL1 fan to that col's 4 cores.
#   OUTPUT      : core -> per-column gather memtile (col 2+cx) -> outY -> shim drain.
# This spreads shim DMA per-column (like the reference: shim_C_0 DMA0,1 -> mem_C_1) instead of
# collapsing 16 per-core routes onto one centroid shim tile. Reuses the two-hop staging
# pattern from fused_decode.py (_feed_inX / weight-fan / egress).
#
# proj SPMD contract (the reference proj kernel + array layout):
#   MVM_CORES=16; each core computes M_total/16 output rows per phase, in m=32-row
#   blocks (outer i loop), contracting K in k=256 col-blocks (inner j loop). Each
#   32-row block is emitted as ONE packet: y buffer = 16 hdr + 32 payload = 48.
#   proj_qmm_flush_row writes the payload at y+16; the routing id at y+14 is stored
#   by the compiler, from the `dest` the air.channel.put names.
#   Phases (IS_ATTN==1): ph1 QKV M=2560 K=2048 ->5 rnd ->ROPE; ph2 oproj M=2048
#   K=2048 ->4 rnd ->RMS; ph3 gateup M=22528 K=2048 ->44 rnd ->GLU; ph4 down
#   M=2048 K=11264 ->4 rnd ->RMS. round-major core-interleave: round r core c emits r*16+c.
#
# STAGE 1 (this file): QKV-only proj grid, host-fed X/W (memtile-staged), per-col gather.
# Goal = lower clean through aircc + IR-diff the proj-tile region vs the reference QWEN2_3B layer IR.
import argparse
import os as _os

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
from air.dialects.scf import for_, yield_, index_switch, IfOp, ForOp
from air.backend.xrt import XRTBackend


def refeed(n, emit):
    """Re-send ONE resident buffer n times: an n-trip scf.for around a single
    air.channel.put. The body holds nothing but the put and no operand depends
    on the induction variable, so this is a re-broadcast, not n productions --
    air-annotate-refeed recognizes the shape, collapses the loop, and derives
    the count for the lock init. n <= 1 emits the bare put."""
    if n <= 1:
        emit()
        return
    c0 = arith.ConstantOp(IndexType.get(), 0).result
    cn = arith.ConstantOp(IndexType.get(), n).result
    c1 = arith.ConstantOp(IndexType.get(), 1).result
    for _rf in for_(c0, cn, c1):
        emit()
        yield_([])


# ============================ Qwen2.5-3B config =============================
MODEL_DIM = 2048
K = MODEL_DIM
ROW_BLOCK = 32  # Q4NX_ROW_BLOCK_SIZE (m)
COL_BLOCK = 256  # Q4NX_COL_BLOCK_SIZE (k)
GROUP = 32  # Q4NX_GROUP_SIZE
BLOCK_BF16 = 2560  # one packed q4k block (bf16 units)

MVM_CORES = 16
NCX = 4  # proj grid columns (physical cols 2,3,4,5)
NCY = 4  # proj grid rows (physical rows 2,3,4,5)
PROJ_COL0 = 2  # physical column of grid tx=0
PROJ_ROW0 = 2  # physical row of grid ty=0
# X broadcast memtile column (reference mem_tile_1_1 = the hub column). NOTE Llama puts its X
# memtile on a column SEPARATE from its hub (col2 vs col1); QWEN_XMT_COL overrides for
# the loopclose bisect.
XMT_COL = int(_os.environ.get("QWEN_XMT_COL", "1"))

# W_DUAL_CHAN=1: feed each proj column's weights on BOTH of its shim MM2S channels
# instead of ch0 only, following FLM (its mem_C_1 takes shim ch0 on S2MM4 ->
# w_buffer[0:5120] -> MM2S0/1 -> rows 2/3, and shim ch1 on S2MM5 -> w_buffer[5120:]
# -> MM2S2/3 -> rows 4/5, on two INDEPENDENT lock cycles). The split is SPATIAL, by
# cascade pair: each channel owns a disjoint half of the column's cores, so the two
# are never ordered against each other. It also changes the DDR cascade order, so the
# packer must run with the same flag (qwen25_3b_requant reads fd.W_DUAL_CHAN).
# Qwen already has its X memtile on the hub column (XMT_COL=1), which is the
# placement the llama engine had to move to before its dual feed would route.
W_DUAL_CHAN = int(_os.environ.get("W_DUAL_CHAN", "1"))


def _wname(ci, cx):
    """Weight channel: the single @inW bundle when the flag is off, else the
    per-column shim-col-pinned channel for shim channel ci of proj column cx."""
    return f"inW{ci}c{cx}" if W_DUAL_CHAN else "inW"


# X memtile get granularity (one packet per BD on the @xnorm packet channel).
XCH = 2 * COL_BLOCK
# attn-o (ph1 X) refeed memtile column. Default 6 (next to attn col 7), but that makes
# the oref -> @xnorm -> X memtile(col 1) hop cross the BUSY proj grid (cols 2-5) -- the
# same cross-column class as the glu bug. QWEN_OREF_COL overrides for the bisect.
OREF_COL = int(_os.environ.get("QWEN_OREF_COL", "6"))
# QWEN_OREF_NOPUT=1 (bisect): oref still GATHERS attnO (nothing dangles) but does NOT
# re-broadcast it as ph1's X; the rms core supplies ph1 X instead (numerically wrong,
# topology-only). Isolates the oref -> @xnorm -> X memtile path from proj ph1 itself.
OREF_NOPUT = int(_os.environ.get("QWEN_OREF_NOPUT", "0"))
# QWEN_OREF_VIA_RMS=1: the attn-o gather memtile does NOT feed @xnorm directly. It hands
# the gathered o to the rms core over @orms, and the rms core re-emits it as ph1's X on
# @xnorm (re-broadcast OPROJ_REFEED times). That makes the rms tile the ONLY
# physical producer into the hub X ring -- exactly the shape that PASSES at NPH=2 when the
# oref put is removed (QWEN_OREF_NOPUT=1) -- while keeping the real attn data. Isolates
# "a second physical X producer starves ph0" from "the attn data itself starves ph0".
# NOTE: QWEN_PH1_XCHAN is NOT a valid test of the same question -- it turns the hub's X
# broadcast MM2S into a rigid ph0/ph1 round-robin, which cannot serve 5 ph0 sends then 4
# ph1 sends and so deadlocks for its own reason (device+IR confirmed 2026-08-05v).
# QWEN_ORMS_HOST=1 (bisect, requires QWEN_OREF_VIA_RMS=1): feed @orms from the HOST instead
# of from the attn-o gather memtile, and drain attnO to the host as usual. Splits "the
# rms-side @orms get + ph1 @xnorm re-emit" from "the attn -> oref memtile -> rms transport".
ORMS_HOST = int(_os.environ.get("QWEN_ORMS_HOST", "0"))
# QWEN_OREF_NOCHAIN=1: tag the attn-o gather buffer air.no_chain_lock so the two CU gets do
# NOT daisy-chain (CU1's write waiting on CU0's). The two attn CUs are cross-coupled through
# the shared-L1 score buffer, so serializing their outputs can deadlock the pair.
OREF_NOCHAIN = int(_os.environ.get("QWEN_OREF_NOCHAIN", "0"))
# QWEN_OREF_DRAIN=1 (bisect, use with QWEN_ORMS_HOST=1): keep the oref memtile GATHER of the
# two attn CU outputs, but drain the gathered buffer to the host instead of putting @orms
# (which is host-fed in that mode). Exercises the attn -> oref-memtile leg on its own,
# without the oref -> rms hop. Coherent: nothing dangles, and the drain keeps the gather
# live so AIR cannot DCE it (the QWEN_OREF_NOPUT=2 trap).
OREF_DRAIN = int(_os.environ.get("QWEN_OREF_DRAIN", "0"))
# QWEN_OREF_HOSTSRC=1 (bisect, use with QWEN_OREF_VIA_RMS=1): the oref memtile is fed from
# the HOST instead of from attnO (attnO drains to host as usual), but still puts @orms to
# the rms core over the identical on-chip path. Splits "memtile -> rms transport" from
# "@orms depends on attn finishing". Host-fed-at-the-CHANNEL (QWEN_ORMS_HOST) already passes,
# so this pins down whether the memtile hop or the attn dependency is the blocker.
OREF_HOSTSRC = int(_os.environ.get("QWEN_OREF_HOSTSRC", "0"))
# QWEN_PH1_XCHAN=1 (diagnostic): give ph1 its own X channel @xnorm2 instead of sharing
# the convergent @xnorm with the rms ph0 producer.
PH1_XCHAN = int(_os.environ.get("QWEN_PH1_XCHAN", "0"))
ROPE_KV_FIRST = int(_os.environ.get("QWEN_ROPE_KV_FIRST", "0"))
# q-broadcast memtile column (default 6). At 6 it shares mem_6_1 with the oref gather, so
# ONE memtile both FEEDS attn (toAttnQ) and CONSUMES attn output (attnO). QWEN_QMT_COL=7
# co-locates the q-broadcast with the attn column and leaves mem_6_1 to oref alone.
QMT_COL = int(_os.environ.get("QWEN_QMT_COL", "6"))
EGR_RNDS = int(_os.environ.get("QWEN_EGR_RNDS", "0"))  # 0 = all DEST_TOTAL rounds
# QWEN_OREF_2HOP=1: split the attn-o gather from the refeed across TWO memtile buffers so
# the REFEED buffer has exactly ONE fill BD. AIR scales EVERY fill BD of a refeed buffer by N
# (AIRToAIEPass.cpp:6492); with the 2-CU gather that creates an intermediate xN lock link that
# The reference Qwen design does NOT have (the reference: xN only on the LAST fill, intermediate link = 1).
OREF_2HOP = int(_os.environ.get("QWEN_OREF_2HOP", "0"))
OREF2_COL = int(_os.environ.get("QWEN_OREF2_COL", "0"))
# QWEN_ROPE_ECHO=1 (witness/experiment): rope drains the qkv it RECEIVED straight to the
# host instead of emitting ropeQ. attn then never gets q, so the ph0->rope->attn->oref->ph1-X
# feedback loop is BROKEN while the oref memtile still exists and is armed. If the echo shows
# up in Y, the ph0 egress is healthy with oref present and the blocker is the loop dynamics;
# if Y stays zero, an armed-but-unfilled oref alone starves the ph0 egress.
ROPE_ECHO = int(_os.environ.get("QWEN_ROPE_ECHO", "0"))
# QWEN_XN_SPLIT=1: emit the ph0 X at the consumer granularity (1 put per get) instead
# of one whole-K put re-fed XN_REFEED times. See the rms body for the rationale.
XN_SPLIT = int(_os.environ.get("QWEN_XN_SPLIT", "0"))

DQ = 2048  # NUM_ATTN_HEADS(16) * DH(128)
DK = 256  # NUM_KV_HEADS(2) * DH(128)
DV = 256
# 11008 padded to a multiple of GLU_HID(512) *whose glu trip count NGLU//2 IS EVEN*.
# 11264 (NGLU=22 -> 11 iterations, ODD) is the natural pad and it CORRUPTS the MLP:
# AIR unrolls the 2-slice glu loop by 2, and an odd trip count leaves a 2-slice
# REMAINDER EPILOGUE, so the core lowers to 6 glu_aie calls (4 in the loop + 2 peeled)
# with a non-uniform lock/release tail and a 4+2 = 6-deep ring. The down X then loses
# 2 of every 6 chunks and backfills them with stale ones -- device-measured
# [h0 h1 h2 h3 . . h0 h1 h8 h9 . . h12..h15 . . h12..h15], full-layer cos 0.805.
# 12288 (NGLU=24 -> 12 iterations, EVEN) unrolls cleanly: 4 calls, uniform 4-deep ring,
# locks 4/4, full-layer cos 0.998789. The extra pad rows are quantized zeros, so
# silu(0)*0 = 0 and down's pad columns contribute nothing -- numerics are unchanged.
# This is exactly why AIR-Llama never showed the bug: its NGLU//2 is 16, already even.
INTERMEDIATE = int(_os.environ.get("QWEN_INTERMEDIATE", "12288"))

HDR = 16  # y buffer header region (pkt_id at +14)
YBUF = HDR + ROW_BLOCK  # 48
PKT_OFF = 14  # dma egress start offset (2 id words + 32 payload)
PKT_PAY = 2 + ROW_BLOCK  # 34 (matches reference dma_bd %y,14,34)

# Egress consumer names. The routing id is allocated by air-annotate-packet-ids;
# the design only says which consumer a phase feeds, and the ordinal is that
# name's position in first-appearance order (= the @outY broadcast index the
# receiving gets use).
D_ROPE = "rope"
D_RMS = "rms"
D_GLU = "glu"
PHASES = [
    ("qkv", DQ + DK + DV, MODEL_DIM, D_ROPE),
    ("oproj", MODEL_DIM, DQ, D_RMS),
    ("gateup", 2 * INTERMEDIATE, MODEL_DIM, D_GLU),
    ("down", MODEL_DIM, INTERMEDIATE, D_RMS),
]
RB = [m // MVM_CORES // ROW_BLOCK for (_, m, _, _) in PHASES]  # [5,4,44,4]
NJ = [kk // COL_BLOCK for (_, _, kk, _) in PHASES]  # [8,8,8,44]

# STAGE 2: all 4 proj phases (QKV, o-proj, gate-up, down). vocab (IS_ATTN=0) later.
# QWEN_NPH = phase-isolation bisection (use ONLY with QWEN_ATTN=0): build just the
# first N proj phases. NPH=1 = QKV-only (pkt1 -> rope), the earliest phase (the hang
# is early). Since the truncated phases' packet ids (pkt4 rms / pkt8 glu) are no longer
# produced, the rms/glu herds + their host feeds/drains + DEST lookups are gated off
# (HAS_RMS/HAS_GLU below) so the build stays COHERENT (no consumer waits on a packet id
# that is never emitted = would be a FALSE deadlock). If phase0+feeds COMPLETE alone ->
# the hang is a multi-phase/phase-transition interaction; if phase0 alone HANGS -> it's
# the core X/weight-feed + phase-0 egress->rope path.
import os as _os

NPH = int(_os.environ.get("QWEN_NPH", "4"))
_RB = RB[:NPH]
_NJ = NJ[:NPH]
_DST = [p[3] for p in PHASES][:NPH]
_MTOT = [p[1] for p in PHASES][:NPH]
# distinct packet ids (ordered) for the id-demux egress: 1->rope, 4->rms, 8->glu.
DEMUX = list(dict.fromkeys(_DST))  # ordinal order: ["rope", "rms", "glu"]
NDEST = len(DEMUX)
# Phase -> DESTINATION ORDINAL. This is what the kernel stamps now; the packet
# id on the wire is allocated by air-annotate-packet-ids and substituted into
# these constants. The ordinal is the same index the receiving gets sit at
# (`indices=[0, d]`), so there is only one numbering to keep straight.
DEST = [DEMUX.index(d) for d in _DST]
# egress rounds per dest (id): rope=QKV(5); rms=oproj(4)+down(4)=8; glu=gateup(44).
DEST_RNDS = [sum(_RB[p] for p in range(NPH) if _DST[p] == d) for d in DEMUX]
DEST_TOTAL = sum(DEST_RNDS)  # 57 rounds total
PAYLOAD = MVM_CORES * ROW_BLOCK  # 512 = one round (16 cores x 32 rows)
# Step B relay drain columns (free cols; rope/rms/glu compute cores replace these in C/D).
RELAY_COLS = [0, 6, 7][:NDEST]

# ---- surround compute (Step C) ----
DH = 128  # head_dim (rope cos/sin LUT length)
ROPE_COL, ROPE_ROW = 1, 3  # reference rope tile(1,3); consumes dest0 (pkt1 QKV)
# The reference places glu at tile(6,2), but the outY pkt8 demux route hub(col1)->glu@col6
# runs EASTWARD across the busy proj grid (cols 2-5) and DEADLOCKS (device-confirmed
# 2026-08-05f: glu@col6 = global stall; glu@col1 = COMPLETES). Llama avoids this because
# its proj cols are {0,1,6,7}, leaving cols 2-5 free for the hub->glu route; Qwen's
# contiguous proj 2-5 does not. So glu lives on the HUB column (col1, free row 4), making
# the pkt8 demux hub-local like rope(1,3)/rms(1,2). QWEN_GLU_COL/ROW override for diag.
GLU_COL, GLU_ROW = 1, 4
GLU_COL = int(_os.environ.get("QWEN_GLU_COL", str(GLU_COL)))
GLU_ROW = int(_os.environ.get("QWEN_GLU_ROW", str(GLU_ROW)))
GLU_SLICE = 2 * PAYLOAD  # 1024 = [up 512 | gate 512] (two demux rounds per slice)
GLU_HID = PAYLOAD  # 512 silu(gate)*up out per slice
# Phase-isolation (QWEN_NPH<4): a surround consumer exists only if its packet id is
# still produced by one of the kept phases. HAS_ROPE is always true (phase 0 = QKV).
HAS_ROPE = D_ROPE in DEMUX
HAS_RMS = D_RMS in DEMUX  # oproj (ph1) and/or down (ph3) kept
HAS_GLU = D_GLU in DEMUX  # gate-up (ph2) kept
NGLU = (
    DEST_RNDS[DEMUX.index(D_GLU)] // 2 if HAS_GLU else 0
)  # 44/2 = 22 slices -> 11264 = INTERMEDIATE
DEST_ROPE = DEMUX.index(D_ROPE) if HAS_ROPE else None  # 0
DEST_RMS = DEMUX.index(D_RMS) if HAS_RMS else None  # 1
DEST_GLU = DEMUX.index(D_GLU) if HAS_GLU else None  # 2

# ---- attention (Step 2b): 1x8x1, 2 CUs on col 7, DH=128 ----
# reference Qwen attn topology (from the reference QWEN2_3B layer IR): col 7; attn_qk tile_7_2 &
# tile_7_4, attn_kv tile_7_3 & tile_7_5 (2 CUs); mem_tile_7_1 stages K/V. 1x8x1 =
# 1 kv head + 8 q heads per CU (NUM_KV_HEADS=2 -> 2 CUs). We use the DECOMPOSED
# attn_qk_blk / attn_kv_blk / attn_kv_fin wrappers (lock-free; AIR drives the block
# loop + ping-pong), which under the Qwen header select the SAME 1x8x1 peano-fixed
# _attn_qk / attn_fv / calculate_l / scale_div templates as the monolithic kernels.
# This reuses the proven Llama AIR attn dataflow at DH=128 / 2-CU instead of trying
# to replicate the monolithic kernel's in-kernel lock protocol.
import os as _os

N_ATTN_CU = 2
KV_PER_CU = 1
Q_HEADS_PER_CU = 8  # 16 attn heads / 2 CUs
KVPC_DH = KV_PER_CU * DH  # 128 (k/v width per CU)
ATTN_COL = 7
ATTN_CU_LOC = [(7, 2, 3), (7, 4, 5)]  # (col, qk_row, kv_row) per CU
# score slot padded to a multiple of 64 for the v64 loads: logical
# Q_HEADS_PADDED_PER_CU(8)*16 scores + 8 c floats (=16 bf16 units) = 144 -> 192.
SSZ_BLK = ((Q_HEADS_PER_CU * 16 + 16 + 63) // 64) * 64  # 192
QCU = Q_HEADS_PER_CU * DH  # 1024 q (and o) per CU
KVBLK = 16 * KVPC_DH  # 2048 one 16-key K (or V) block per CU
# compile-time block count (context length). ATTN_L env for the lowering gate; the
# kernel masks the last partial block, so one ATTN_MAXL build serves any L (Phase 4
# patches runtime L). Default small for a fast lower-check.
ATTN_L = int(_os.environ.get("ATTN_L", "32"))
ATTN_ROUNDS = (ATTN_L + 15) // 16
# DECODE_DYNSEQ=1: take the context length as a runtime scalar instead of baking it
# in. It becomes a launch operand driving the KV readback's block count, the append's
# slot address, the memtile dequeue and the attention herd's RTP-L, so the shim pushes
# exactly what the cores consume at whatever L the host dispatches. The instruction
# stream is then assembled per token from the emitted TXN builder rather than read
# from a frozen insts.bin. Off by default; the template pair stays the shipping path.
DYNSEQ = int(_os.environ.get("DECODE_DYNSEQ", "0"))


def _for_no_pingpong(start, stop, step):
    """`for_` that opts the loop out of ping-pong (air.disable_ping_pong).

    The memtile KV dequeue below is a ping-pong candidate here (llama's is
    declined for live state). Labelling unrolls the body by 2 and leaves a
    remainder; with a runtime trip count the remainder's bound arithmetic is
    hoisted out of the loop, and air-to-aie then rejects the memtile DMA region
    for using a value defined outside it.
    """
    params = [start, stop, step]
    for i, p in enumerate(params):
        if isinstance(p, int):
            params[i] = arith.ConstantOp.create_index(p)
    for_op = ForOp(*params, [])
    for_op.operation.attributes["air.disable_ping_pong"] = UnitAttr.get()
    with InsertionPoint(for_op.body):
        yield for_op.induction_variable


# Attn kernel linkage. Same mechanism as the Llama driver (fused_decode.py): the
# attn_qk/attn_kv kernels are linked as LLVM IR (.ll) rather than objects, so they
# can be llvm-linked and INLINED into the core (kernels built alwaysinline via
# -DDECODE_INLINE_ATTN). This is upstream mlir-aie's func-level inline-kernel API:
# the kernel func.func declaration carries link_with = "<name>.ll" together with
# link_with_mode = "merge", which aiecc's aie-assign-core-link-files pass routes
# into the core's link_merge_files -> llvm-link merges the alwaysinline body into
# the core module before opt/llc (no surviving func.call, no object link).
# air-to-aie copies the decl's discardable attrs onto the lowered AIE func.func,
# so setting link_with_mode here is all that is needed. The attn HERDS carry no
# link_with at all -- the linkage comes from the kernel func each core calls.
_ATTN_EXT = ".ll"  # fixed config: inline-attn merge-mode (.ll) is the only decode path
_ATTN_MERGE = _ATTN_EXT == ".ll"  # emit link_with_mode="merge" for the inline path


def _set_attn_link(op, base):
    """Attach the kernel link_with (+ link_with_mode="merge" for the .ll inline path)."""
    op.attributes["link_with"] = StringAttr.get(base + _ATTN_EXT)
    if _ATTN_MERGE:
        op.attributes["link_with_mode"] = StringAttr.get("merge")


# ---- reference-mirroring KV-cache + attn config (ported from fused_decode.py, the proven
# Llama AIR design, re-parameterized for Qwen 1x8x1 / DH=128 / 2 CUs on col 7). ----
# The whole rope->q-broadcast(mem_6_1)->attn->attn-o(mem_6_1)->xnorm loopclose and the
# DDR KV cache append+readback mirror the reference QWEN2_3B layer IR exactly (verified by IR
# diff). See fused_decode.py lines ~239-303 for the Llama originals of these constants.
DH_A = DH  # 128 (attn head_dim)
KVPC_DH = KV_PER_CU * DH_A  # 1*128 = 128 (K or V width per CU)
# CUs grouped by column: Qwen has BOTH CUs on col 7 -> ONE group (reference mem_7_1 serves
# all 4 attn tiles on col 7). Llama had 4 CUs on 2 cols (2 groups).
ATTN_COL_GROUPS = []  # [(col, [cu_idx,...]), ...] in CU order
for _c, _loc in enumerate(ATTN_CU_LOC):
    if ATTN_COL_GROUPS and ATTN_COL_GROUPS[-1][0] == _loc[0]:
        ATTN_COL_GROUPS[-1][1].append(_c)
    else:
        ATTN_COL_GROUPS.append((_loc[0], [_c]))
ATTN_CU_GROUP = {c: gi for gi, (_, cus) in enumerate(ATTN_COL_GROUPS) for c in cus}
NGRP = len(ATTN_COL_GROUPS)  # 1
REGION_W = len(ATTN_COL_GROUPS[0][1]) * KVPC_DH  # 2*128 = 256 (per-token per-group K/V)
DK_TOT_A = N_ATTN_CU * KVPC_DH  # 2*128 = 256 (all-CU K or V width)
KVSZ_TOK = 2 * DK_TOT_A  # 512 (per-token K++V)
ATTN_MAXL = ATTN_ROUNDS * 16  # padded context (compile-time block count)
APPEND_OFF = (ATTN_L - 1) * KVSZ_TOK  # this token's slot in the cache
REGION_STRIDE = ATTN_MAXL * REGION_W  # per-group region span (one K or V region)
KV_SPLIT = True  # decoupled K/V memtile rings (separate inKV_K/inKV_V S2MM)
KV_REGION = True  # region-major DDR KV quadrants [K_grp0 | V_grp0]
KV_APPEND = True  # rope writes this token's roped-K/raw-V into the DDR cache

# ---- X-loopclose (Step 2c): reuse the proven Llama convergent-xnorm mechanism ----
# The proj X for each phase comes on-chip (not host toX): ph0 QKV X = rmsnorm(input),
# ph1 oproj X = attn-o, ph2 gateup X = rmsnorm(input+oproj), ph3 down X = glu-down.
# All four converge (in phase-time order) on ONE npu_dma_packet channel @xnorm, read
# by ONE count-free feed loop that broadcasts 256-blocks to the 16 proj cores (inX).
# Each phase source re-broadcasts REFEED[p] = RB[p] times (rounds/phase): the rms core
# from the rms core, the attn-o + glu-down from their gather memtiles. Every
# re-feed is an n-trip loop around the put. Mirrors fused_decode.py, re-parameterized.
# QWEN_ATTN=0 = device-deadlock bisection: drop the whole attn subsystem (KV
# staging + attn herd + o-gather), rope drains q to host, forces LOOPCLOSE off.
# Isolates whether a device hang is in attn vs the base proj/surround.
ATTN = int(_os.environ.get("QWEN_ATTN", "1"))
# LOOPCLOSE defaults off when attn is cut (ph1's X is the attn output), but can be forced
# on for the ATTN=0 + NPH=1 bisect: that config has NO attn-sourced X (ph0's X is the rms
# xnorm), so it isolates the on-chip X-feed loopclose mechanism from the attn subsystem.
LOOPCLOSE = int(_os.environ.get("QWEN_LOOPCLOSE", "1" if ATTN else "0"))

# DEFAULT ON whenever the on-chip attn-o -> o-proj X path exists: making the rms tile the
# sole physical producer into the hub X ring is the correct topology, and a second physical
# producer starves ph0 (device-confirmed: ph0 never completes, KV=0).
OREF_VIA_RMS = int(_os.environ.get("QWEN_OREF_VIA_RMS", "1" if ATTN else "0"))

# DEFAULT ON when the loopclose/attn X path exists, OFF for the host-toX base.
# With the on-chip X path the host MUST NOT await the whole weight feed before the data
# that unblocks proj ph1 is issued: the weight feed only fully drains once ph1 runs, ph1
# needs its X from the rms tile, and that X depends on attn -- a host-side circular wait
# (device-confirmed 2026-08-05af; visible only in the aie.runtime_sequence task order, not
# in any device-side view). Per-phase inW tasks break the cycle. The attn-less base uses
# host toX, has no such dependency, and is REGRESSED by phase-major (TIMEOUT), so this
# must stay conditional rather than global.
W_PHASE_MAJOR = int(
    _os.environ.get("QWEN_W_PHASE_MAJOR", "1" if (ATTN and LOOPCLOSE) else "0")
)

# X-buffer section layout. The @ropeLUT / @rmsIn / @rmsW / @orms host feeds all used to
# read X[0:], i.e. the cos/sin+bias slab, the token activation and the RMSNorm weight
# were ALIASED onto the same DDR bytes. That is harmless for a topology gate (any bytes
# will do) but makes real numerics impossible. Give each its own section; sizes and BD
# structure are unchanged (only the source address differs), so the shim task shape the
# topology gates depend on is preserved.
# X_SPLIT=0 restores the old aliased layout for A/B.
X_SPLIT = int(_os.environ.get("QWEN_X_SPLIT", "1"))
# Unroll the NGLU-chunk gluOut gather into the down-X memtile buffer (see the
# comment at the gather). QWEN_DOWN_GATHER_UNROLL=0 restores the rolled form.
# NOTE: unrolling DEADLOCKS the device (2026-08-06) -- 22 constant-offset gets
# blow the memtile BD budget. Default OFF; kept only for A/B.
DOWN_GATHER_UNROLL = int(_os.environ.get("QWEN_DOWN_GATHER_UNROLL", "0"))
# Emit the down X as NGLU separate GLU_HID puts instead of one whole-INTERMEDIATE
# put, so the emission granularity matches the X memtile's 512-element gets 1:1
# (the same rule XN_SPLIT applies to the rms ph0 X feed: @xnorm is a PACKET
# channel, one put = one packet, and one get = one BD expecting one packet).
# NOTE: splitting the put BLOWS THE BD BUDGET (aiecc: "Allocator exhausted
# available BD IDs (maximum 24 available for channel 0)") -- NGLU puts x
# DOWN_REFEED re-sends is far past 24, so the single whole-buffer put is
# structurally required here. Default OFF; kept only for A/B.
DOWN_PUT_SPLIT = int(_os.environ.get("QWEN_DOWN_PUT_SPLIT", "0"))
# Gather the gluOut chunks two-at-a-time, matching the glu core's 2-slice
# ping/pong cadence, instead of one per rolled iteration. With the 1-per-iter
# form the device drops every THIRD glu pair (measured: in pair units the down X
# came out [P0,P1,--,P0,P4,--,P6,P7,--,P6,P7]), which is a ring-depth mismatch
# against the depth-2 producer.
# NOTE: pair-cadence gather DEADLOCKS the device (2026-08-06). Default OFF;
# kept only for A/B. The rolled 1-get-per-iteration form is the only one of the
# three that runs -- it is also the one that drops every third pair, so the
# down-X corruption is still OPEN.
DOWN_GATHER_PAIRS = int(_os.environ.get("QWEN_DOWN_GATHER_PAIRS", "0"))
# Give the glu core the reference's depth-2 ping/pong ring (see _glu_body). QWEN_GLU_RING2=0
# restores the per-slice-alloc form that lowers to a 6-deep ring.
GLU_RING2 = int(_os.environ.get("QWEN_GLU_RING2", "1"))
XO_ROPE = 0  # [ROPE_W]  cos|sin LUT + q|k|v bias
XO_RMSIN = 4096 if X_SPLIT else 0  # [K]  token activation (layer input)
XO_RMSW = 8192 if X_SPLIT else 0  # [K]  input_layernorm weight
XO_ORMS = 12288 if X_SPLIT else 0  # [K]  o-handover (QWEN_ORMS_HOST diag only)
XO_RMSW2 = 16384 if X_SPLIT else 0  # [K]  post_attention_layernorm (ph2 rmsnorm)
# Multi-layer X layout. The ACTIVATION slot is layer-invariant (it is the in-place
# residual chain: layer k's @layerOut is layer k+1's @rmsIn), while the rope slab
# (cos/sin LUT + q|k|v bias) and BOTH rms weights are PER LAYER. With NLAYERS==1 the
# per-layer stride is 0 and every offset is exactly the single-layer constant, so the
# verified build is untouched.
XLAYER = 20480 if _os.environ.get("QWEN_NLAYERS", "1") != "1" else 0
# Feed post_attention_layernorm for the ph2 rmsnorm. Without it the ph2 norm reuses the
# ph0 buffer (input_layernorm), i.e. the device computes a DIFFERENT function than HF --
# device-confirmed by A/B against the golden (w_in 0.9988 vs w_post 0.8961 at
# INTERMEDIATE=12288). Carried as a SECOND put/get on the EXISTING @rmsW channel (one
# extra S2MM BD) rather than a new channel: the rms tile's S2MM budget is the documented
# pressure point (rmsIn pkt2 + rmsW pkt3 are already a rigid positional chain).
PH2_RMSW = int(_os.environ.get("QWEN_PH2_RMSW", "1"))

# ===== Multi-layer fused decode (stitch NLAYERS runtime sub-sequences) =====
# Mirrors Llama fused_decode.py: the DEVICE (segment/herds) is emitted ONCE and reused
# temporally; only the launch-scope L3 feeds get per-layer DDR offsets, scaled by an AIR
# scf.for induction variable that airrt-to-npu unrolls LATE. So the AIR op count stays
# CONSTANT (a Python unroll blows up air-to-aie) and the aie.device/xclbin is unchanged --
# only the runtime instruction sequence grows. NLAYERS=1 is a strict no-op: the loop is
# 0..1 and every per-layer base folds to 0.
NLAYERS = int(_os.environ.get("QWEN_NLAYERS", "1"))

# Under LOOPCLOSE the rms core is ALSO the ph0 X producer (@xnorm), so it must exist even
# when phase-isolation cuts every pkt4 phase (its outY read groups are then empty).
# QWEN_FORCE_RMS=1 builds the rms tile even without LOOPCLOSE (X still host-fed via toX,
# rms just does rmsIn -> layerOut). Splits "adding the rms tile" from "the xnorm X-feed"
# as the cause of the LOOPCLOSE hang.
FORCE_RMS = int(_os.environ.get("QWEN_FORCE_RMS", "0"))
BUILD_RMS = HAS_RMS or LOOPCLOSE or FORCE_RMS
# QWEN_ATTN_NOCOMPUTE=1 = attn-isolation bisection: keep the ENTIRE attn dataflow
# + lock handshakes (q feed, KV feed, cross-core score handoff, attnO, o-gather)
# but SKIP the 3 attn kernel calls (attn_qk_blk/attn_kv_blk/attn_kv_fin). attnO then
# carries uninitialized data (fine -- garbage flows, only the DATAFLOW is tested). If
# the full design RUNS with this on, the deadlock is INSIDE the attn kernel; if it
# still HANGS, the deadlock is in the attn dataflow/handshake (score buffer, feeds).
ATTN_NOCOMPUTE = int(_os.environ.get("QWEN_ATTN_NOCOMPUTE", "0"))
# finer isolation: skip only the qk or only the kv compute (dataflow/handshake
# stays). Lets us tell WHICH attn kernel hangs (qk vs kv/fin) on device.
SKIP_QK = int(_os.environ.get("QWEN_SKIP_QK", "0")) or ATTN_NOCOMPUTE
SKIP_KV = int(_os.environ.get("QWEN_SKIP_KV", "0")) or ATTN_NOCOMPUTE
SKIP_KVFIN = int(_os.environ.get("QWEN_SKIP_KVFIN", "0")) or SKIP_KV
# QWEN_SURROUND_NOCOMPUTE=1 (use ONLY with QWEN_ATTN=0): keep the ENTIRE base dataflow
# (proj MVM + egress + X/weight feeds + rope/rms/glu channel gets+puts) but SKIP the
# rope/rms/glu compute CallOps. COHERENT (not a header-skip): under ATTN=0/LOOPCLOSE=0
# the surround outputs (qDrain/layerOut/gluDrain) are circuit drains to host, NOT packet
# flows, and proj (whose put carries the dest the header is stored from) still runs. If the base then
# COMPLETES, the deadlock is inside a surround kernel (rope/rms/glu miscompile); if it
# still HANGS, the deadlock is in the proj/egress/feed dataflow.
SURROUND_NOCOMPUTE = int(_os.environ.get("QWEN_SURROUND_NOCOMPUTE", "0"))
REFEED = list(_RB)  # [5,4,44,4] rounds/phase == re-broadcast counts
OPROJ_PHASE, GATEUP_PHASE, DOWN_PHASE = 1, 2, 3
XN_REFEED = REFEED[0]  # ph0 QKV: rms channel-level refeed (5)
# Phase-isolation (QWEN_NPH<4): a phase's refeed count is only meaningful (and only
# used under LOOPCLOSE, which is off when ATTN=0) if that phase is built. Guard so
# import doesn't IndexError when the phase list is truncated.
OPROJ_REFEED = REFEED[OPROJ_PHASE] if OPROJ_PHASE < NPH else 0  # ph1 attn-o memtile
GATEUP_REFEED = REFEED[GATEUP_PHASE] if GATEUP_PHASE < NPH else 0  # ph2 rms per-put
DOWN_REFEED = REFEED[DOWN_PHASE] if DOWN_PHASE < NPH else 0  # ph3 glu-down memtile
# The attn-o gather memtile (oref) consumes attnO only when the o-proj phase (ph1) is
# built; otherwise attnO has no on-chip consumer and must be drained to the host.
# OREF_NOPUT: 0 = normal; 1 = oref removed entirely (attnO -> host, rms supplies ph1 X);
# 2 = oref still GATHERS attnO but does not put @xnorm (rms supplies ph1 X). 1 vs 2 splits
# "the attnO gather blocks ph0" from "the oref xnorm put blocks ph0".
DRAIN_ATTNO = bool(ATTN) and (
    OREF_NOPUT == 1
    or OREF_HOSTSRC
    or (ORMS_HOST and not OREF_DRAIN)
    or not (LOOPCLOSE and OPROJ_PHASE < NPH)
)
RMS_PCOL = 1  # rms core + X memtile column (reference Qwen rms tile_1_2)
# o-proj rounds / down rounds consumed by the rms residual stream.
OPROJ_RNDS = OPROJ_REFEED  # 4 (oproj output rounds -> residual1)
DOWN_RNDS = DOWN_REFEED  # 4 (down output rounds -> residual2)

# ---- QKV bias (Step 3): Qwen2.5 q/k/v_proj.bias (no o bias) ----
# The rope core adds the bias to qkv BEFORE RoPE (rope.cc add_q_k_v_bias under
# HAS_QKV_BIAS). The bias slab [q(DQ)|k(DK)|v(DV)] = 2560 is appended after the DH
# cos/sin LUT in the rope weight buffer, so rope_buffer = DH + DQ+DK+DV = 2688
# (matches the reference's rope_buffer<2688>). The bias is fed via the ropeLUT channel.
QKV_BIAS = True
ROPE_W = DH + (DQ + DK + DV if QKV_BIAS else 0)  # 2688 (128 LUT + 2560 bias)

# per-core counts (all phases)
X_READS = sum(_RB[p] * _NJ[p] for p in range(NPH))  # 40+32+352+176 = 600 inX gets/core
W_READS = X_READS  # 600 wL2ToL1 gets/core
Y_PUTS = sum(_RB)  # 5+4+44+4 = 57 outA puts/core
# memtile-hop counts
assert X_READS % 2 == 0
X_CHUNKS = X_READS // 2  # X memtile 512-chunk gets (each -> 2 inX broadcasts)
PER_COL_W = NCY * W_READS  # weight blocks per column = 4*600 = 2400
W_FAN_STEPS = W_READS  # per-col fan steps (each fans NCY blocks) = 600
# Per-layer DDR slab sizes (elements), used only when NLAYERS>1.
W_LAYER = NCX * PER_COL_W * BLOCK_BF16  # weights per layer


def build_module():
    @module_builder
    def build():
        bf16 = BF16Type.get()
        f32 = F32Type.get()
        i32 = IntegerType.get_signless(32)
        idx_t = IndexType.get()
        l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l2 = IntegerAttr.get(T.i32(), MemorySpace.L2)

        def idx(v):
            return arith.ConstantOp.create_index(v)

        # ---- host operands (L3) ----
        x_l3 = MemRefType.get([max(X_CHUNKS * 2 * COL_BLOCK, NLAYERS * XLAYER)], bf16)
        w_l3 = MemRefType.get([NLAYERS * W_LAYER], bf16)
        # KV blocks (separate operand so the router places its feed on a far shim
        # column, mirroring the reference's KV on shim col 7; reusing the col-1 X operand
        # oversubscribed the col-1 shim MM2S).
        kv_l3 = MemRefType.get([NLAYERS * N_ATTN_CU * 2 * ATTN_ROUNDS * KVBLK], bf16)
        # 57 rounds x 512, but at least K + DQ: the DRAIN_ATTNO diagnostic path puts
        # the attention output at Y[K:K+DQ], and under phase isolation DEST_TOTAL
        # shrinks below that.
        y_l3 = MemRefType.get([max(DEST_TOTAL * PAYLOAD, K + DQ)], bf16)

        # ---- L1 buffers ----
        xblk_l1 = MemRefType.get([COL_BLOCK], bf16, memory_space=l1)
        wblk_l1 = MemRefType.get([BLOCK_BF16], bf16, memory_space=l1)
        yacc_l1 = MemRefType.get([ROW_BLOCK], f32, memory_space=l1)
        ybuf_l1 = MemRefType.get([YBUF], bf16, memory_space=l1)

        # ---- L2 memtile staging buffers ----
        xmt_l2 = MemRefType.get([2 * COL_BLOCK], bf16, memory_space=l2)  # 512
        # one fan get; W_DUAL_CHAN halves it (each shim channel feeds its own ring
        # covering half the column's cores -- FLM's w_buffer[0:5120]/[5120:] split).
        wfan_l2 = MemRefType.get(
            [(NCY // (2 if W_DUAL_CHAN else 1)) * BLOCK_BF16], bf16, memory_space=l2
        )
        # per-col gather (reference y_buffer<130> = 2 hdr + 4x32; keeps core0's pkt header).
        col_l2 = MemRefType.get([2 + NCY * ROW_BLOCK], bf16, memory_space=l2)  # 130
        # hub gather (reference y_buffer<514> = 2 hdr + 4x128; keeps col0's pkt header).
        hub_l2 = MemRefType.get(
            [2 + NCX * (NCY * ROW_BLOCK)], bf16, memory_space=l2
        )  # 514
        relay_l2 = MemRefType.get([PAYLOAD], bf16, memory_space=l2)  # 512 demux relay
        # ---- surround compute L1 buffers ----
        qkv_l1 = MemRefType.get([DQ + DK + DV], bf16, memory_space=l1)  # 2560 QKV
        ropeq_l1 = MemRefType.get([DQ], bf16, memory_space=l1)  # 2048 roped q
        ropekv_l1 = MemRefType.get([DK], bf16, memory_space=l1)  # 256 roped k / v
        ropew_l1 = MemRefType.get(
            [ROPE_W], bf16, memory_space=l1
        )  # 2688 = 128 cos/sin LUT + 2560 QKV bias
        glu_x_l1 = MemRefType.get([GLU_SLICE], bf16, memory_space=l1)  # 1024 [up|gate]
        glu_hid_l1 = MemRefType.get([GLU_HID], bf16, memory_space=l1)  # 512 silu*up
        rms_l1 = MemRefType.get([K], bf16, memory_space=l1)  # 2048 rms in/resid/out
        # ---- attention L1 buffers (DH=128, 8 q/CU, 1 kv/CU) ----
        aq_l1 = MemRefType.get([QCU], bf16, memory_space=l1)  # 1024 q per CU
        ak_l1 = MemRefType.get([KVBLK], bf16, memory_space=l1)  # 2048 k block
        av_l1 = MemRefType.get([KVBLK], bf16, memory_space=l1)  # 2048 v block
        as_l1 = MemRefType.get([SSZ_BLK], bf16, memory_space=l1)  # 192 shared scores
        ao_l1 = MemRefType.get([QCU], bf16, memory_space=l1)  # 1024 o per CU
        m_l1 = MemRefType.get([16], bf16, memory_space=l1)  # running max (8 used)
        c_l1 = MemRefType.get([8], f32, memory_space=l1)  # softmax correction
        y_l1 = MemRefType.get([QCU], f32, memory_space=l1)  # 1024 S.V accumulator
        lden_l1 = MemRefType.get([16], f32, memory_space=l1)  # softmax denominator
        # q-broadcast memtile (reference mem_tile_6_1 q_buffer<2048>): rope q(2048) fanned
        # per-CU (1024 each) with the head-reshape stride to the qk cores.
        qmt_l2 = MemRefType.get([DQ], bf16, memory_space=l2)  # 2048 q broadcast
        # KV block staging memtile (reference mem_tile_7_1): combined K (resp V) block
        # [16 keys, REGION_W] for the whole col-group, reshaped per CU to toK/toV.
        kvcomb_l2 = MemRefType.get([16 * REGION_W], bf16, memory_space=l2)  # 4096
        # X-loopclose refeed memtiles (reference mem_tile_6_1 role): attn-o (ph1 oproj X =
        # MODEL_DIM) and glu-down (ph3 down X = INTERMEDIATE) gathered then re-broadcast
        # into the convergent @xnorm.
        oref_l2 = MemRefType.get([K], bf16, memory_space=l2)  # 2048 attn-o
        downref_l2 = MemRefType.get([INTERMEDIATE], bf16, memory_space=l2)  # 11264

        # ---- kernels (reuse proj_qmm.o primitives) ----
        zero = FuncOp("proj_qmm_zero", ([yacc_l1, i32], []), visibility="private")
        zero.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        acc256 = FuncOp(
            "proj_qmm_acc256", ([xblk_l1, wblk_l1, yacc_l1], []), visibility="private"
        )
        acc256.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        flush_row = FuncOp(
            "proj_qmm_flush_row", ([yacc_l1, ybuf_l1, i32], []), visibility="private"
        )
        flush_row.attributes["link_with"] = StringAttr.get("proj_qmm.o")
        # rope_compute(q,k,v, qkv, lut, arm): rotate-half RoPE on Q/K (V copied).
        rope_compute = FuncOp(
            "rope_compute",
            ([ropeq_l1, ropekv_l1, ropekv_l1, qkv_l1, ropew_l1, i32], []),
            visibility="private",
        )
        rope_compute.attributes["link_with"] = StringAttr.get("rope.o")
        # glu_aie(hid, x, arm): x = [up 512 | gate 512] -> hid = silu(gate)*up (512).
        glu_aie = FuncOp(
            "glu_aie", ([glu_hid_l1, glu_x_l1, i32], []), visibility="private"
        )
        glu_aie.attributes["link_with"] = StringAttr.get("glu.o")
        # residual_add_aie(y, x_buf, x): y = x_buf + x (residual stream).
        residual_add_aie = FuncOp(
            "residual_add_aie", ([rms_l1, rms_l1, rms_l1], []), visibility="private"
        )
        residual_add_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
        # rms_copy_aie(dst, src): dst = src (MODEL_DIM bf16). Used for the ph1
        # o-proj X handover -- o-proj's input IS the gathered attention output,
        # so the rms tile only re-emits it, it does not combine it with anything.
        rms_copy_aie = FuncOp(
            "rms_copy_aie", ([rms_l1, rms_l1], []), visibility="private"
        )
        rms_copy_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
        # rms_norm_aie(y, x, w, arm): y = rmsnorm(x) * w. Produces the proj X feed
        # (ph0 = rmsnorm(input); ph2 = rmsnorm(input+oproj)) for the X-loopclose.
        rms_norm_aie = FuncOp(
            "rms_norm_aie", ([rms_l1, rms_l1, rms_l1, i32], []), visibility="private"
        )
        rms_norm_aie.attributes["link_with"] = StringAttr.get("rms_residual.o")
        # attn (decomposed, lock-free; AIR drives the block loop). s_block is the
        # LAST memref of attn_qk_blk (producer tag) and a non-last memref of
        # attn_kv_blk (consumer tag) -> AIR shared-L1 cross-core RAW between the
        # qk (writer) and kv (reader) tiles of each CU. blk,L passed by value.
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

        # ---- channels (two-hop, mirroring the reference) ----
        channel_decl("toX", size=[1])  # host -> X memtile (col 1)
        _inX = Channel(
            "inX", size=[1, 1], broadcast_shape=[NCX, NCY]
        )  # Xmt -> 16 cores
        _inX.operation.attributes["air.shared_resident_ring"] = UnitAttr.get()
        if W_DUAL_CHAN:
            # Per-column channels rather than one [NCX] bundle, so each column's
            # two shim MM2S channels land on that column's shim tile. The column
            # itself is not stated: the W memtile these feed is already bucketed
            # by column, and AIRToAIE stamps that bucket column on the shim tile.
            for _wc in range(NCX):
                for _ci in range(2):
                    channel_decl(_wname(_ci, _wc), size=[1])
        else:
            channel_decl("inW", size=[NCX])  # host -> per-col W memtile
        channel_decl("wL2ToL1", size=[NCX, NCY])  # W memtile -> col cores
        # proj core egress is a PACKET flow (the reference: packet_flow tile_C_r -> mem_C_1,
        # keep_pkt_header). A routing header at y+14 rides the packet so the hub can demux
        # QKV/oproj+down/gateup -> rope/rms/glu. The design names the DESTINATION on the put
        # (dest=...); air-annotate-packet-ids allocates the id and stores the header. No
        # numeric id appears here -- the reference's 1/4/8 were the pinned spelling this
        # replaced.
        _outA = channel_decl("outA", size=[NCX, NCY], channel_type="npu_dma_packet")
        # No keep_pkt_header and no packet_ids. Both are DERIVED.
        #
        # This hop carries a routing decision made here (the dest operand on its
        # put) to a switchbox further downstream, so it MUST preserve the header
        # -- forced by the topology, not a choice, and air-annotate-packet-ids
        # injects it. The ids likewise: the hop is single-destination, so its
        # list is just the demux's set with no meaningful order.
        # col memtile -> mem_1_1 hub (packet, keep header so the hub can demux by id).
        _toHub = channel_decl("toHub", size=[NCX], channel_type="npu_dma_packet")
        # keep_pkt_header derived, as for @outA above.
        # hub id-demux egress: emit the assembled 514 packet on ONE MM2S; switchbox routes
        # id 1->dest0(rope), 4->dest1(rms), 8->dest2(glu); strip header -> pure 512 payload.
        _outY = Channel("outY", size=[1, 1], broadcast_shape=[1, NDEST])
        _outY.operation.attributes["channel_type"] = StringAttr.get("npu_dma_packet")
        # Nothing declared here. That this channel demuxes is DERIVED: its
        # destinations partition the stream, and a put naming a `dest` reaches
        # it through the hops above. A dest operand can only mean "this packet
        # is for that leaf", which is exactly the statement that the fanout is
        # over time rather than over space.
        #
        # No packet_ids either. air-annotate-packet-ids allocates them from the
        # top of the id space (so nothing else is renumbered) and rewrites the
        # ordinals the kernel stamps to match.
        # Keep the hub demux on S2MM0 at every consumer tile; the shim-sourced feeds
        # (ropeLUT/rmsIn/rmsW) are pinned to S2MM1 below. Without an explicit pin,
        # unpinned packet flows REUSE whatever packet channel the tile already has
        # (AIRToAIESchedulingUtils.cpp:1391), which merges two independent producers
        # (hub + shim) into ONE ordered BD chain = head-of-line deadlock.
        _lut = channel_decl("ropeLUT", size=[1])  # host cos/sin LUT -> rope core
        # Pin the LUT feed to the rope tile's S2MM1 (the outY QKV demux keeps S2MM0).
        # Under LOOPCLOSE the rms core adds two shim packet feeds (rmsIn/rmsW) which
        # oversubscribe shim col 1, so AIR converts ropeLUT from a circuit flow into a
        # PACKET flow -- and packet flows on one tile reuse a single physical channel
        # (AIRToAIESchedulingUtils.cpp:1391) unless pinned. That merged ropeLUT and the
        # outY QKV payload into ONE strictly-ordered 6-BD chain on rope S2MM0 fed by TWO
        # INDEPENDENT producers (shim + hub mem_1_1) -> head-of-line deadlock (device-
        # confirmed 2026-08-05j: the ONLY diff between the hanging LOOPCLOSE build and
        # the working host-toX build). The pin restores the two-separate-channel layout.
        # ...and feed it from a FREE shim column: the packetization above is triggered by
        # shim col 1 being oversubscribed (ropeLUT + rmsIn + rmsW + layerOut + qDrain).
        # Off col 1 the LUT feed stays circuit-switched, which cannot share the rope
        # tile's packet S2MM at all.
        _lut.operation.attributes["air.shim_col"] = IntegerAttr.get(i32, 0)
        channel_decl(
            "gluDrain", size=[1]
        )  # glu out (down-X) -> shim (Step C; loopclose D)
        _rmsin = channel_decl(
            "rmsIn", size=[1]
        )  # host raw input activation -> rms core
        channel_decl("layerOut", size=[1])  # rms residual2 (layer output) -> shim
        # ---- attention channels (reference-mirroring: rope q -> mem_6_1 q-broadcast tile ->
        # per-CU qk; K/V via KV staging mem_7_1). Ports fused_decode.py exactly. ----
        if ATTN:
            channel_decl("ropeQ", size=[1])  # rope q(2048) -> mem_6_1 q-broadcast tile
            channel_decl(
                "toAttnQ", size=[N_ATTN_CU]
            )  # mem_6_1 -> per-CU qk core (reshaped)
            # reference mem_7_1 KV staging: combined K (resp V) block [16 keys, REGION_W] per
            # col-group on SEPARATE flows (KV_SPLIT: independent K/V rings), reshaped per
            # CU to toK (qk S2MM1) / toV (kv S2MM1). Ports fused_decode.py inKV_K/inKV_V.
            channel_decl("inKV_K", size=[NGRP])  # cache K [16,REGION_W] -> mem_7_1
            channel_decl("inKV_V", size=[NGRP])  # cache V [16,REGION_W] -> mem_7_1
            if KV_APPEND:
                # reference-faithful on-chip KV-cache append (mirror fused_decode.py:762-775):
                # rope's roped-K / raw-V leave on ONE dedicated rope MM2S as PACKET flows
                # pinned to the attn shim col (7) -> DDR cache. Qwen has ONE col group
                # (NGRP=1) so both K and V transit col 7 (Llama split cols 3/4).
                #
                # Which rope MM2S they take is no longer stated. rope also emits ropeQ
                # as a CIRCUIT flow (-> the col-6 q-broadcast memtile), and a physical
                # output port cannot carry both a static circuit route and packet
                # routes -- the packet BDs would be steered by the circuit connection
                # instead of their packet dests. AIRToAIE separates them itself now
                # (TileDMAAllocator::spreadCollapsedPacketChannels), reaching the same
                # placement this used to name.
                _apK = channel_decl("appendK", size=[1], channel_type="npu_dma_packet")
                _apK.operation.attributes["air.shim_col"] = IntegerAttr.get(
                    i32, ATTN_COL
                )
                _apV = channel_decl("appendV", size=[1], channel_type="npu_dma_packet")
                _apV.operation.attributes["air.shim_col"] = IntegerAttr.get(
                    i32, ATTN_COL
                )
            channel_decl(
                "toK", size=[N_ATTN_CU]
            )  # staging -> qk core (k block, reshaped)
            channel_decl(
                "toV", size=[N_ATTN_CU]
            )  # staging -> kv core (v block, reshaped)
            _ao = channel_decl(
                "attnO", size=[N_ATTN_CU]
            )  # kv o -> drain (o-proj X Step D2)
            if DRAIN_ATTNO:
                # Bisection path only (no on-chip oref consumer): send the o drain out a
                # shim column away from col 7, which already carries appendK/appendV +
                # inKV_K/inKV_V. Sharing col 7 makes two flows target the same shim
                # dest -> "aie.masterset op targets same destination South: 2" at route
                # time (this is why the earlier attn-alone bisect would not build).
                _ao.operation.attributes["air.shim_col"] = IntegerAttr.get(i32, 6)
        else:
            channel_decl("qDrain", size=[1])  # bisection: rope q -> host
        if ATTN and ROPE_ECHO:
            channel_decl(
                "qDrain", size=[1]
            )  # echo witness: rope's received qkv -> host
            # loop-break: attn's q now comes from the HOST, not from rope.
            channel_decl("hostQ", size=[1])
        # ---- X-loopclose channels (Step 2c; reuse Llama convergent-xnorm) ----
        # ONE convergent packet channel carries all 4 phase X sources (rms ph0/ph2,
        # attn-o ph1, glu-down ph3) into the X memtile, read by ONE feed loop. Every
        # re-feed on it is written as an n-trip loop around the put (see refeed()).
        # Pin the rms core's xnorm output to tile MM2S1 (layerOut keeps MM2S0) so the
        # placer does not flip layerOut circuit->packet.
        _xn = channel_decl("xnorm", size=[1], channel_type="npu_dma_packet")
        if OREF_2HOP:
            channel_decl("oref2", size=[1], channel_type="npu_dma_packet")
        if OREF_HOSTSRC:
            _oi = channel_decl("orefIn", size=[1])
            _oi.operation.attributes["air.shim_col"] = IntegerAttr.get(i32, 0)
        if OREF_DRAIN:
            _od = channel_decl("orefDrain", size=[1])
            _od.operation.attributes["air.shim_col"] = IntegerAttr.get(i32, 6)
        if OREF_VIA_RMS:
            # attn-o gather memtile -> rms core. PIN to the rms tile's S2MM0.
            # Unpinned, the packet-flow channel reuse collapses it onto S2MM1, which already
            # carries rmsIn (pkt2) and rmsW (pkt3) as a rigid 3-BD POSITIONAL chain over
            # three DIFFERENT buffers with different lock pairs. A memtile-sourced @orms then
            # arrives on a different switchbox port than the shim-sourced rmsIn/rmsW while
            # both rules target the SAME amsel, so the arbiter interleaves the two streams at
            # packet granularity and the positional chain desynchronizes -> deadlock.
            # (device+routed-IR confirmed 2026-08-05x: passing shim-fed @orms has ONE slave
            # port on tile_1_2 masterset DMA:1; every failing memtile-fed variant has TWO,
            # independent of oref column and of the attn dependency.)
            # S2MM0 carries only the o-proj outY get, and the @orms get is emitted BEFORE it
            # in _rms_body, so the chain order [orms, outY] matches the dataflow order.
            _orms = channel_decl("orms", size=[1], channel_type="npu_dma_packet")
        if PH1_XCHAN:
            channel_decl("xnorm2", size=[1], channel_type="npu_dma_packet")
        # rmsW lands on the rms core's S2MM1 alongside rmsIn, which is pinned
        # above; it does not need to say so itself.
        channel_decl("rmsW", size=[1])  # host rms weight -> rms core
        channel_decl("gluOut", size=[1])  # glu-down -> down-X refeed memtile (ph3)

        # ============================ proj grid core ============================
        # The 4 phases run in ONE AIR for_ loop (NOT Python-unrolled) so the core reuses
        # a SINGLE set of DMA BDs (inX/wL2ToL1 S2MM + outA MM2S). Python-unrolling the
        # phases makes 4x the BDs -> overflows the 16-BD aie.mem limit. Per-phase rounds
        # (RB), col-blocks (NJ), and pkt id are index_switch-selected on the phase IV
        # (mirrors fused_decode.py _psw / the reference proj.cc phase loop).
        def _core(tx, ty, _sx, _sy):
            gcx = tx
            gcy = ty
            one = arith.ConstantOp(IntegerAttr.get(i32, 1), None).result
            c2 = idx(2)
            i2c = [idx(v) for v in _RB]
            # HALF the col-block count per phase; _gemv doubles it (2*half) so the
            # inner reduction loop has a PROVABLY-EVEN trip count. AIR's ping-pong
            # pass unrolls the innermost candidate loop by 2; an opaque
            # index_switch bound (e.g. [8,8,8,44]) forces a remainder/epilogue copy
            # -> a THIRD X/W buffer (depth-3). A provably-even 2*half bound has no
            # epilogue -> clean depth-2 x_0/x_1 + w_0/w_1 rings (matches the reference/Llama).
            j2h = [idx(v // 2) for v in _NJ]  # [4,4,4,22]
            pktc = [idx(d) for d in DEST]

            def _psw(ph, vals, ty_):
                if len(vals) == 1:
                    return vals[0]
                return index_switch(
                    [ty_],
                    ph,
                    list(range(len(vals) - 1)),
                    case_body_builder=lambda op, i, cv: yield_([vals[i]]),
                    default_body_builder=lambda op: yield_([vals[-1]]),
                )

            # Y is depth-1 (single alloc/put per round). The reference/Llama use a depth-2 Y ring,
            # but AIR can't reproduce it here: the ping-pong pass only reaches the inner
            # _j reduction (not the round loop); output-unrolling the round loop to get 2
            # static put sites spawns extra _j rings (X/W depth blows up, worse with the
            # QKV ODD round count RB=5 needing a tail _gemv -> X/W depth-6); and a
            # runtime parity-selected buffer fails air-to-aie lock allocation. Device
            # testing confirmed proj X/W depth (2 vs 3) is NOT the deadlock, so Y depth-1
            # is left as-is (a serialization difference, not a correctness/deadlock one).
            for ph in for_(idx(0), idx(NPH), idx(1)):
                I2v = _psw(ph, i2c, idx_t)
                J2v = _psw(ph, j2h, idx_t)
                pktv = _psw(ph, pktc, idx_t)
                for _i in for_(idx(0), I2v, idx(1)):
                    # ONE reduction pass per round (single inX/wL2ToL1 get site) with a
                    # PROVABLY-EVEN 2*half trip count so the ping-pong unroll has no
                    # remainder/epilogue copy -> clean depth-2 x_0/x_1 + w_0/w_1 rings.
                    J2x2 = arith.muli(J2v, c2)
                    a_acc = AllocOp(yacc_l1, [], [])
                    CallOp(zero, [a_acc, one])
                    for _j in for_(idx(0), J2x2, idx(1)):
                        a_x = AllocOp(xblk_l1, [], [])
                        ChannelGet("inX", a_x, indices=[gcx, gcy])
                        a_w = AllocOp(wblk_l1, [], [])
                        ChannelGet("wL2ToL1", a_w, indices=[gcx, gcy])
                        CallOp(acc256, [a_x, a_w, a_acc])
                        DeallocOp(a_x)
                        DeallocOp(a_w)
                        yield_([])
                    a_y = AllocOp(ybuf_l1, [], [])
                    # Row 0 of the payload; this core produces one row-block, so
                    # there is no partner row. No routing header is written here
                    # -- the compiler stores it from the put's `dest`.
                    _c0i = arith.ConstantOp(IntegerAttr.get(i32, 0), None).result
                    CallOp(flush_row, [a_acc, a_y, _c0i])
                    # dest = which egress consumer this round feeds. The compiler
                    # allocates that destination's packet id and emits the header
                    # store at offsets[0]; the kernel no longer touches routing.
                    ChannelPut(
                        "outA",
                        a_y,
                        indices=[gcx, gcy],
                        offsets=[idx(PKT_OFF)],
                        sizes=[idx(PKT_PAY)],
                        strides=[idx(1)],
                        dest=pktv,
                    )
                    DeallocOp(a_acc)
                    DeallocOp(a_y)
                    yield_([])
                yield_([])

        # ============================ top function ==============================
        # DYNSEQ appends the context length as a trailing scalar, so the DDR
        # argument positions -- and the host binding built around them -- are unchanged.
        _fn_args = [x_l3, w_l3, kv_l3, y_l3] + ([i32] if DYNSEQ else [])

        @FuncOp.from_py_func(*_fn_args)
        def qwen_decode(*_fa):
            def launch_body(*_la):
                X, W, KV, Y = _la[4], _la[5], _la[6], _la[7]
                # The dispatch-time context length (DYNSEQ), last operand before the
                # multi-layer induction variable.
                L_rt = _la[4 + len(_fa) - 1] if DYNSEQ else None
                # Multi-layer: the last launch operand is the scf.for induction variable
                # (see the emit branch at the end of build()). a_iv is None for the
                # single-layer build, so every per-layer offset below folds to the plain
                # Python constant it had before and the IR is byte-identical.
                a_iv = _la[4 + len(_fa)] if len(_la) > 4 + len(_fa) else None

                def _ceil16(v):
                    """ceil(v/16) as an index Value, computed in i32.

                    aie-translate's C++ TXN target emits the integer widths but has no
                    case for index-typed arithmetic, and this has to survive into the
                    emitted builder.
                    """
                    _s = arith.addi(
                        v, arith.ConstantOp(IntegerAttr.get(i32, 15), None).result
                    )
                    _q = arith.divui(
                        _s, arith.ConstantOp(IntegerAttr.get(i32, 16), None).result
                    )
                    return arith.index_cast(idx_t, _q)

                KV_LAYER = N_ATTN_CU * 2 * ATTN_ROUNDS * KVBLK

                def _po(slab, off):
                    """DDR offset `off` inside this layer's `slab`-sized region."""
                    if a_iv is None or not slab:
                        return idx(off)
                    base = arith.muli(a_iv, idx(slab))
                    return arith.addi(base, idx(off)) if off else base

                def _xo(off):  # per-layer X slab (rope LUT + bias, rms weights)
                    return _po(XLAYER, off)

                def _wo(off):  # per-layer weight slab
                    return _po(W_LAYER, off)

                def _kvo(off):  # per-layer KV cache slab
                    return _po(KV_LAYER, off)

                # --- host X feed -> X memtile (launch scope) ---
                # LOOPCLOSE: X is produced on-chip (rmsnorm/attn-o/glu-down converge on
                # @xnorm), so there is NO host toX feed. Stub path host-feeds toX.
                if not LOOPCLOSE:
                    for _c in for_(idx(0), idx(X_CHUNKS), idx(1)):
                        off = arith.muli(_c, idx(2 * COL_BLOCK))
                        ChannelPut(
                            "toX",
                            X,
                            offsets=[off],
                            sizes=[idx(2 * COL_BLOCK)],
                            strides=[idx(1)],
                        )
                        yield_([])

                # --- host cos/sin LUT + rms activation feeds, BEFORE the weight feed ---
                # ORDERING IS LOAD-BEARING (device-confirmed 2026-08-05l). The launch
                # carries air.preserve_shim_dma_order, so the runtime sequence issues shim
                # tasks in program order, and it AWAITS the four big inW tasks before
                # issuing anything that comes later. Under LOOPCLOSE the proj X is produced
                # ON CHIP by the rms core, so if rmsIn/rmsW are emitted after the weight
                # feed the host blocks in dma_await_task(inW) -> the weights cannot drain
                # because the proj cores are stalled on inX -> inX never arrives because the
                # X memtile is waiting on @xnorm -> the rms core is still waiting for
                # rmsIn/rmsW the host has not issued = CIRCULAR WAIT. The non-loopclose
                # build survives only because its @toX feed sits here, ahead of the awaits.
                ChannelPut(
                    "ropeLUT",
                    X,
                    offsets=[_xo(XO_ROPE)],
                    sizes=[idx(ROPE_W)],
                    strides=[idx(1)],
                )
                # Phase-isolation: rms exists if it consumes pkt4 OR (LOOPCLOSE) it is
                # the ph0 X producer.
                if BUILD_RMS:
                    ChannelPut(
                        "rmsIn",
                        X,
                        offsets=[idx(XO_RMSIN)],
                        sizes=[idx(K)],
                        strides=[idx(1)],
                    )
                    # host rms weight -> rms core (LOOPCLOSE: rmsnorm needs the weight)
                    if LOOPCLOSE:
                        ChannelPut(
                            "rmsW",
                            X,
                            offsets=[_xo(XO_RMSW)],
                            sizes=[idx(K)],
                            strides=[idx(1)],
                        )
                        if PH2_RMSW:
                            ChannelPut(
                                "rmsW",
                                X,
                                offsets=[_xo(XO_RMSW2)],
                                sizes=[idx(K)],
                                strides=[idx(1)],
                            )
                    if OREF_VIA_RMS and ORMS_HOST and OPROJ_PHASE < NPH:
                        ChannelPut(
                            "orms",
                            X,
                            offsets=[idx(XO_ORMS)],
                            sizes=[idx(K)],
                            strides=[idx(1)],
                        )

                # --- host weight feed -> per-col W memtile, ROUND-MAJOR (phase-outer,
                # col-inner) so the NCX proj columns advance in lockstep with the X
                # broadcast multicast. A channel-major whole-column feed lets
                # air-opt-shim-dma-bds regroup into per-channel BDs, breaking the
                # cross-column phase barrier -> deadlock (this is why Llama's
                # fused_decode.py feeds phase-major + sets air.preserve_shim_dma_order
                # on the launch). Each column's phase block is fed as TWO contiguous
                # halves so the shim-dma coalescer merges+tags them
                # (air.coalesced_shim_feed = the cross-channel phase barrier).
                colspan = PER_COL_W * BLOCK_BF16
                STEP_BF16 = NCY * BLOCK_BF16  # one fan step = NCY rows' blocks
                # PHASE-MAJOR DDR LAYOUT [phase][col][data] (was [col][phase][data]).
                # The ISSUE order was already phase-major, but a column-major LAYOUT makes a
                # column's consecutive phase blocks CONTIGUOUS in DDR, so the shim coalescer
                # merges every phase into ONE awaited task per column. The host then blocks
                # awaiting weights for phases whose X does not exist yet (phase 1's X is the
                # attn output, gated on the KV readback issued after that await) = circular
                # wait. Phase-major separates a column's phase blocks by the other columns'
                # data, so each PHASE coalesces on its own -- the two halves within a phase
                # stay adjacent, preserving the air.coalesced_shim_feed cross-channel
                # barrier -- and the host awaits only the phase it needs next. NOTE the
                # weight feed is lockstep-coupled to the X broadcast, so the compiler paces
                # it; a feed with NO shared broadcast consumer is exempted instead. Dropping
                # that barrier here deadlocked phase 0 -- device-confirmed 2026-08-05o.
                # Total size unchanged (a permutation).
                # DEFAULT = column-major [col][phase]: the whole column coalesces into ONE
                # shim task, which the ATTN=0 base REQUIRES (device-confirmed: phase-major
                # regressed it to a timeout). Phase-major is kept behind QWEN_W_PHASE_MAJOR
                # for the attn work, where the all-phase await is a circular wait.
                _pbase = [0]
                for p in range(NPH):
                    _pbase.append(_pbase[-1] + NCX * _RB[p] * _NJ[p] * STEP_BF16)
                _woff = [0]
                for p in range(NPH):
                    _woff.append(_woff[-1] + _RB[p] * _NJ[p] * STEP_BF16)

                def _feed_w_phase(p):
                    span_p = _RB[p] * _NJ[p] * STEP_BF16
                    nch = 2 if W_DUAL_CHAN else 1
                    # per-channel share of this (col, phase) slab. W_DUAL_CHAN: the
                    # packer laid the slab out [low-row half | high-row half], so
                    # channel ci reads one contiguous run at cbase + ci*span_c.
                    span_c = span_p // nch
                    hp = span_c // 2
                    for cx in range(NCX):
                        cbase = (
                            _pbase[p] + cx * span_p
                            if W_PHASE_MAJOR
                            else cx * colspan + _woff[p]
                        )
                        for ci in range(nch):
                            ch = _wname(ci, cx)
                            # index built per put (not hoisted) so the flag-off IR stays
                            # byte-identical: the original emitted one constant per put.
                            wix = lambda: [idx(0) if W_DUAL_CHAN else idx(cx)]
                            base = cbase + ci * span_c
                            # two contiguous puts per channel so the shim coalescer
                            # merges+tags them (air.coalesced_shim_feed); a single put
                            # would skip coalescing and lose the phase barrier.
                            ChannelPut(
                                ch,
                                W,
                                indices=wix(),
                                offsets=[_wo(base)],
                                sizes=[idx(hp)],
                                strides=[idx(1)],
                            )
                            ChannelPut(
                                ch,
                                W,
                                indices=wix(),
                                offsets=[_wo(base + hp)],
                                sizes=[idx(span_c - hp)],
                                strides=[idx(1)],
                            )

                # SPLIT THE WEIGHT FEED AROUND THE KV READBACK (mirror Llama's runtime
                # sequence). The 4 phase groups are CONTIGUOUS in DDR per column, so the
                # shim coalescer merges them into ONE ~409600-element awaited task/column.
                # With attn on, the host would then block in dma_await_task(inW) until ALL
                # phases' weights drain -- but phase 1's X is the attn output, attn needs
                # the KV readback, and inKV is issued AFTER that await = circular wait
                # (same class as the rms-feed ordering above). Llama avoids it by emitting
                # one inW group per phase with the KV readback started between phase 0 and
                # phase 1. Do the same: phase 0 here, phases 1.. after the KV block.
                # ATTN=0 keeps the original single contiguous feed (verified-good base).
                _W_SPLIT = bool(ATTN) and NPH > 1
                for p in range(1 if _W_SPLIT else NPH):
                    _feed_w_phase(p)

                # --- host drains -> Y ---
                # rms layer output (residual2 = input + o-proj + down): K=2048.
                if BUILD_RMS:
                    # Multi-layer: chain the residual stream IN PLACE -- layer k writes its
                    # output back over the X activation slot, which layer k+1 reads as its
                    # @rmsIn. Same mechanism as Llama (x at a layer-invariant offset). The
                    # single-layer build keeps draining to Y[0:K] so the verified gate and
                    # every existing checker are unchanged.
                    ChannelGet(
                        "layerOut",
                        X if NLAYERS > 1 else Y,
                        indices=[idx(0)],
                        offsets=[idx(XO_RMSIN if NLAYERS > 1 else 0)],
                        sizes=[idx(K)],
                        strides=[idx(1)],
                    )
                # --- reference-faithful on-chip KV cache (mirror fused_decode.py
                # _emit_append / _emit_readback); KV is the persistent DDR cache
                # [K region (ATTN_MAXL*REGION_W) | V region (ATTN_MAXL*REGION_W)]. ---
                # (1) APPEND this token's roped-K / raw-V into the cache at this token's
                #     slot ((ATTN_L-1)*REGION_W within each region); the ChannelGet
                #     receives rope's appendK/appendV packet flow into DDR.
                # (2) READ BACK the whole cache region-major on inKV_K/inKV_V (one 3D nd-DMA
                #     per region: [ATTN_ROUNDS blocks][16 keys][REGION_W]);
                #     no shared broadcast consumer -> fire-and-free feed.
                # air-annotate-append-barrier derives the append->readback RAW ordering
                # from the shared L3 cache memref.
                if ATTN:
                    # DYNSEQ: this token's slot is a runtime address, so the append
                    # lands on the position being generated rather than a fixed one.
                    if DYNSEQ:
                        _rwv = arith.index_cast(
                            idx_t,
                            arith.muli(
                                arith.subi(
                                    L_rt,
                                    arith.ConstantOp(
                                        IntegerAttr.get(i32, 1), None
                                    ).result,
                                ),
                                arith.ConstantOp(
                                    IntegerAttr.get(i32, REGION_W), None
                                ).result,
                            ),
                        )

                        def _kvo_slot(base):
                            return arith.addi(_kvo(base), _rwv)

                    _rw = (ATTN_L - 1) * REGION_W  # this token's slot within a region
                    _apk = ChannelGet(
                        "appendK",
                        KV,
                        indices=[idx(0)],
                        offsets=[_kvo_slot(0) if DYNSEQ else _kvo(_rw)],
                        sizes=[idx(NGRP), idx(REGION_W)],
                        strides=[idx(REGION_STRIDE), idx(1)],
                    )
                    _apv = ChannelGet(
                        "appendV",
                        KV,
                        indices=[idx(0)],
                        offsets=[
                            (
                                _kvo_slot(NGRP * REGION_STRIDE)
                                if DYNSEQ
                                else _kvo(NGRP * REGION_STRIDE + _rw)
                            )
                        ],
                        sizes=[idx(NGRP), idx(REGION_W)],
                        strides=[idx(REGION_STRIDE), idx(1)],
                    )
                    for gi in range(NGRP):
                        ChannelPut(
                            "inKV_K",
                            KV,
                            indices=[idx(gi)],
                            offsets=[_kvo(gi * REGION_STRIDE)],
                            sizes=[
                                _ceil16(L_rt) if DYNSEQ else idx(ATTN_ROUNDS),
                                idx(16),
                                idx(REGION_W),
                            ],
                            strides=[idx(16 * REGION_W), idx(REGION_W), idx(1)],
                        )
                        ChannelPut(
                            "inKV_V",
                            KV,
                            indices=[idx(gi)],
                            offsets=[_kvo((NGRP + gi) * REGION_STRIDE)],
                            sizes=[
                                _ceil16(L_rt) if DYNSEQ else idx(ATTN_ROUNDS),
                                idx(16),
                                idx(REGION_W),
                            ],
                            strides=[idx(16 * REGION_W), idx(REGION_W), idx(1)],
                        )
                if ATTN and ROPE_ECHO:
                    ChannelPut(
                        "hostQ", X, offsets=[idx(0)], sizes=[idx(DQ)], strides=[idx(1)]
                    )
                    # NOTE: this OVERLAPS layerOut, which also owns Y[0:K], so Y[0:K]
                    # is ambiguous in the ATTN=0 / ROPE_ECHO bisect paths. Moving the
                    # drain to Y[K:] to deconflict was tried and REGRESSED the ATTN=0
                    # base to TIMEOUT (device-confirmed 2026-08-05ai, zero weights) --
                    # the offset change perturbs the shim task structure. Left as-is;
                    # do not "fix" this offset without re-running the ATTN=0 gate.
                    ChannelGet(
                        "qDrain",
                        Y,
                        offsets=[idx(0)],
                        sizes=[idx(DQ + DK + DV)],
                        strides=[idx(1)],
                    )
                if OREF_HOSTSRC:
                    ChannelPut(
                        "orefIn", X, offsets=[idx(0)], sizes=[idx(K)], strides=[idx(1)]
                    )
                if OREF_DRAIN:
                    ChannelGet(
                        "orefDrain",
                        Y,
                        offsets=[idx(0)],
                        sizes=[idx(K)],
                        strides=[idx(1)],
                    )
                if not ATTN:
                    # bisection: drain rope q (2 CUs) to host so rope still completes.
                    for _cu in range(N_ATTN_CU):
                        # NOTE: overlaps layerOut, which also owns Y[0:K], so Y[0:K] is
                        # ambiguous in the ATTN=0 bisect path. Moving this drain to Y[K:]
                        # to deconflict REGRESSES the ATTN=0 base to TIMEOUT even with
                        # zero weights -- re-confirmed on a FREE device 2026-08-05aj, so
                        # it is a real effect of the offset (it perturbs the shim task
                        # structure), not the earlier contention. Do not "fix" this
                        # without re-running the ATTN=0 gate.
                        ChannelGet(
                            "qDrain",
                            Y,
                            offsets=[idx(_cu * QCU)],
                            sizes=[idx(QCU)],
                            strides=[idx(1)],
                        )
                # Remaining weight phases, issued AFTER the KV readback start (see the
                # _W_SPLIT note above) so the host never awaits phase-N weights that only
                # drain once attn has run.
                if _W_SPLIT:
                    for p in range(1, NPH):
                        _feed_w_phase(p)

                # LOOPCLOSE: attn-o and glu-down are consumed on-chip as proj X (ph1/ph3)
                # via @xnorm, NOT drained to host. Stub path drains them to Y.
                if DRAIN_ATTNO:
                    # --- attn o drain (no on-chip oref consumer) -> Y[K:] ---
                    # Placed AFTER layerOut's Y[0:K] rather than on top of it: this
                    # drain only exists on diagnostic paths (the default build has an
                    # on-chip oref consumer and never emits it), so unlike the qDrain
                    # offsets it costs the verified base nothing, and it makes the
                    # attention output readable on its own bytes.
                    for _cu in range(N_ATTN_CU):
                        ChannelGet(
                            "attnO",
                            Y,
                            indices=[idx(_cu)],
                            offsets=[idx(K + _cu * QCU)],
                            sizes=[idx(QCU)],
                            strides=[idx(1)],
                        )
                if not LOOPCLOSE and HAS_GLU:
                    # glu (dest2/pkt8): NGLU x 512 silu(gate)*up via gluDrain (stub).
                    _yb = K
                    for _sg in for_(idx(0), idx(NGLU), idx(1)):
                        ChannelGet(
                            "gluDrain",
                            Y,
                            offsets=[
                                arith.addi(idx(_yb), arith.muli(_sg, idx(GLU_HID)))
                            ],
                            sizes=[idx(GLU_HID)],
                            strides=[idx(1)],
                        )
                        yield_([])

                @segment(name="seg", operands=([L_rt] if DYNSEQ else []))
                def seg_body(*_sa):
                    # The context length reaches the attention herd from here as a herd
                    # operand: an RTP slot the instruction stream writes per dispatch,
                    # not a constant folded into the core ELF.
                    _seg_L = _sa[0] if DYNSEQ else None

                    def _seg_ceil16():
                        if not DYNSEQ:
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

                    def _core_ceil16(Lh):
                        if not DYNSEQ:
                            return idx(ATTN_ROUNDS)
                        _s = arith.addi(
                            Lh, arith.ConstantOp(IntegerAttr.get(i32, 15), None).result
                        )
                        _q = arith.divui(
                            _s, arith.ConstantOp(IntegerAttr.get(i32, 16), None).result
                        )
                        return arith.index_cast(idx_t, _q)

                    # --- X memtile hop (col 1): get 512 -> broadcast 2x256 to 16 cores ---
                    # LOOPCLOSE: the X source is the convergent @xnorm (rms/attn-o/glu),
                    # NOT host toX. ONE count-free feed loop reads all 4 phases in order.
                    _xsrc = "xnorm" if LOOPCLOSE else "toX"

                    def _x_feed(src, nchunks):
                        for _rc in for_(idx(0), idx(nchunks), idx(1)):
                            xb = AllocOp(xmt_l2, [], [])
                            xb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), XMT_COL)
                            )
                            ChannelGet(
                                src,
                                xb,
                                offsets=[idx(0)],
                                sizes=[idx(2 * COL_BLOCK)],
                                strides=[idx(1)],
                            )
                            for _jj in for_(idx(0), idx(2), idx(1)):
                                joff = arith.muli(_jj, idx(COL_BLOCK))
                                ChannelPut(
                                    "inX",
                                    xb,
                                    offsets=[joff],
                                    sizes=[idx(COL_BLOCK)],
                                    strides=[idx(1)],
                                )
                                yield_([])
                            DeallocOp(xb)
                            yield_([])

                    if PH1_XCHAN and LOOPCLOSE:
                        # DIAGNOSTIC: read ph1's X from its OWN channel (@xnorm2, fed by the
                        # oref memtile) instead of the convergent @xnorm, one feed loop per
                        # phase. Tests whether sharing @xnorm between the rms (ph0) producer
                        # and the attn-dependent oref (ph1) producer is what starves ph0.
                        for _p in range(NPH):
                            _x_feed(
                                "xnorm2" if _p == OPROJ_PHASE else _xsrc,
                                _RB[_p] * _NJ[_p] // 2,
                            )
                    else:
                        _x_feed(_xsrc, X_CHUNKS)

                    # --- weight fan: per-col memtile peels NCY blocks/step -> cores ---
                    # W_DUAL_CHAN emits ONE ring PER SHIM CHANNEL, each owning a
                    # disjoint half of the column's cores (FLM's mem_C_1: ch0 -> rows
                    # 2/3, ch1 -> rows 4/5, two independent lock cycles). A SPATIAL
                    # split is what makes both channels usable -- they share no
                    # consumer, so neither is ever ordered against the other. Do NOT
                    # split temporally (alternating fan steps): that gives every core
                    # one MM2S chain alternating between both channels' buffers and
                    # deadlocks (proven on the llama engine).
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
                                    IntegerAttr.get(T.i32(), PROJ_COL0 + cx)
                                )
                                ChannelGet(
                                    _wch,
                                    wf,
                                    indices=[idx(0) if W_DUAL_CHAN else idx(cx)],
                                )
                                for cy in range(_cy0, _cy1):
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

                    # --- egress gather (mirror the reference/Llama _egress): BOTH hops INTERLEAVED
                    # in ONE round loop so they pipeline across tiles. SEPARATE put-loop
                    # (col->toHub) then get-loop (toHub->outY) DEADLOCKS: the col memtiles
                    # fill while the hub isn't draining yet -> backpressure hang (this was
                    # the base device deadlock). Per round r: gather each col's 4 core
                    # packets -> toHub[cx], THEN gather the 4 toHub -> outY (id-demux).
                    # col y_buffer<130> = 2 hdr (core0's kept pkt header) + 4x32 (cores 1-3
                    # stripped) at 0/34/66/98; hub y_buffer<514> = 2 hdr + 4x128.
                    _CW = NCY * ROW_BLOCK  # 128 per-col payload
                    # QWEN_EGR_RNDS caps the egress loop (controlled experiment): run only
                    # ph0's rounds so the later, attn-gated rounds are never armed. If ph0's
                    # egress then delivers (KV witness fires), the ph0 path is healthy and the
                    # blockage comes from the LATER rounds of the same count-free loop.
                    _egr_n = EGR_RNDS if EGR_RNDS else DEST_TOTAL
                    for _r in for_(idx(0), idx(_egr_n), idx(1)):
                        for cx in range(NCX):
                            cbuf = AllocOp(col_l2, [], [])
                            cbuf.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), PROJ_COL0 + cx)
                            )
                            ChannelGet(
                                "outA",
                                cbuf,
                                indices=[idx(cx), idx(0)],
                                offsets=[idx(0)],
                                sizes=[idx(2 + ROW_BLOCK)],
                                strides=[idx(1)],
                            )
                            for cy in range(1, NCY):
                                ChannelGet(
                                    "outA",
                                    cbuf,
                                    indices=[idx(cx), idx(cy)],
                                    offsets=[idx(2 + cy * ROW_BLOCK)],
                                    sizes=[idx(ROW_BLOCK)],
                                    strides=[idx(1)],
                                )
                            ChannelPut(
                                "toHub",
                                cbuf,
                                indices=[idx(cx)],
                                offsets=[idx(0)],
                                sizes=[idx(2 + NCY * ROW_BLOCK)],
                                strides=[idx(1)],
                            )
                            DeallocOp(cbuf)
                        hbuf = AllocOp(hub_l2, [], [])
                        hbuf.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(), XMT_COL
                        )
                        ChannelGet(
                            "toHub",
                            hbuf,
                            indices=[idx(0)],
                            offsets=[idx(0)],
                            sizes=[idx(2 + _CW)],
                            strides=[idx(1)],
                        )
                        for cx in range(1, NCX):
                            ChannelGet(
                                "toHub",
                                hbuf,
                                indices=[idx(cx)],
                                offsets=[idx(2 + cx * _CW)],
                                sizes=[idx(_CW)],
                                strides=[idx(1)],
                            )
                        ChannelPut(
                            "outY",
                            hbuf,
                            indices=[idx(0), idx(0)],
                            offsets=[idx(0)],
                            sizes=[idx(2 + NCX * _CW)],
                            strides=[idx(1)],
                        )
                        DeallocOp(hbuf)
                        yield_([])

                    # ===== rms core (reference tile_1_2): pkt4 (oproj+down) residual stream =====
                    # h = input + o-proj; res2 = h + down -> layer output. Consumes the 8
                    # pkt4 demux packets in arrival order (4 o-proj, then 4 down). X-production
                    # loopclose (rms rmsnorm -> proj X, o-proj X = attn-o) deferred to Phase 2.
                    # Phase-aware: the pkt4 stream carries the o-proj group (ph1) and the
                    # down group (ph3) INDEPENDENTLY. Under phase-isolation (QWEN_NPH<4)
                    # either may be absent, so size/emit each read group from its own phase
                    # round count (OPROJ_RNDS / DOWN_RNDS, already 0 when the phase is cut)
                    # instead of splitting DEST_RNDS in half.
                    _RMS_OP = OPROJ_RNDS  # 4 o-proj rounds (0 if ph1 cut)
                    _RMS_DN = DOWN_RNDS  # 4 down rounds (0 if ph3 cut)

                    def _rms_body(tx, ty, _sx, _sy):
                        _one = arith.ConstantOp(IntegerAttr.get(i32, 1), None).result
                        a_in = AllocOp(rms_l1, [], [])
                        ChannelGet("rmsIn", a_in, indices=[idx(0)])
                        a_w = None
                        a_xn = None
                        if LOOPCLOSE:
                            a_w = AllocOp(rms_l1, [], [])
                            ChannelGet("rmsW", a_w, indices=[idx(0)])
                            a_w2 = a_w
                            if PH2_RMSW:
                                # post_attention_layernorm, second emission on @rmsW
                                a_w2 = AllocOp(rms_l1, [], [])
                                ChannelGet("rmsW", a_w2, indices=[idx(0)])
                            # ph0 QKV X = rmsnorm(input); channel-level refeed XN_REFEED.
                            # a_xn is KEPT (single y buffer) and REUSED for the ph2 put
                            # below -- mirrors Llama _rms_decode_body. Two separate xnorm
                            # buffers make the scheduler form a 2-deep ring and HOIST both
                            # prod-acquires (c_gateup + c_xn) to the core loop top, but the
                            # ring's prod lock inits to only the max refeed -> the second
                            # acquire blocks forever (startup deadlock, rms never emits X).
                            a_xn = AllocOp(rms_l1, [], [])
                            CallOp(rms_norm_aie, [a_xn, a_in, a_w, _one])
                            if XN_SPLIT:
                                # EMISSION-GRANULARITY match: @xnorm is a PACKET channel, so
                                # one put = one packet and one get = one BD expecting one
                                # packet. The X memtile gets XCH=2*COL_BLOCK (512) per BD, but
                                # the refeed form emits the whole K=2048 buffer as ONE packet
                                # (x XN_REFEED) -> 5 packets vs 20 BDs (1:4 mismatch). The
                                # working host-toX feed is 1:1 (20 puts of 512). Emit the same
                                # 1:1 shape here, in ROUND-MAJOR order (r, then chunk) so the
                                # X memtile sees c0,c1,c2,c3 per round. These are XN_REFEED
                                # x (K/XCH) distinct puts, not a re-broadcast loop.
                                for _xr in range(XN_REFEED):
                                    for _xc in range(K // XCH):
                                        ChannelPut(
                                            "xnorm",
                                            a_xn,
                                            offsets=[idx(_xc * XCH)],
                                            sizes=[idx(XCH)],
                                            strides=[idx(1)],
                                        )
                            else:
                                refeed(
                                    XN_REFEED,
                                    lambda: ChannelPut(
                                        "xnorm",
                                        a_xn,
                                        offsets=[idx(0)],
                                        sizes=[idx(K)],
                                        strides=[idx(1)],
                                    ),
                                )
                            if OREF_NOPUT and OPROJ_PHASE < NPH:
                                # bisect stand-in for the muted oref as ph1's X source.
                                # MUST be emitted HERE, before the o-proj outY get below:
                                # the get blocks on proj ph1's output, and proj ph1 cannot
                                # run until this put supplies its X -> emitting it after the
                                # get is a rms<->proj cycle (device-confirmed 2026-08-05x:
                                # the core lock trace acquired the outY data lock before the
                                # ph1 prod-acquire). Also makes the stand-in work at NPH>=3,
                                # where the gate-up branch below is taken instead.
                                refeed(
                                    OPROJ_REFEED,
                                    lambda: ChannelPut(
                                        "xnorm",
                                        a_xn,
                                        offsets=[idx(0)],
                                        sizes=[idx(K)],
                                        strides=[idx(1)],
                                    ),
                                )
                            if OREF_VIA_RMS and OPROJ_PHASE < NPH:
                                # ph1 X = the attn-o handed over by the oref memtile. Placed
                                # HERE (right after the ph0 put, before the o-proj outY get)
                                # so the phase order is ph0 -> ph1 -> ph2.
                                # DEDICATED landing buffer -- do NOT reuse a_xn. a_xn is the
                                # buffer MM2S1 is still re-sending XN_REFEED times for ph0,
                                # and the @orms S2MM acquires a DIFFERENT lock pair than the
                                # one MM2S1 releases, so AIR does not interlock the incoming
                                # write against the in-flight refeed sends from the same
                                # buffer -- the DMA overwrites X mid-refeed (device+IR
                                # confirmed 2026-08-05x: S2MM1 BD3 Acq lock_1_2_1 vs MM2S1
                                # Rel lock_1_2_7, no shared lock).
                                # Land it in a DEDICATED buffer, then write a_xn with a core
                                # op and emit from a_xn -- the same shape the ph2 gate-up put
                                # uses. Two reasons it must be this shape:
                                #  - putting @xnorm straight from a_o makes a SECOND xnorm
                                #    producer buffer -> AIR forms a 2-deep ring and hoists
                                #    both prod-acquires to the core loop top -> startup
                                #    deadlock (device 2026-08-05x: ph0 never completes, KV=0).
                                #  - getting @orms straight into a_xn gives the incoming S2MM
                                #    a different lock pair than the one MM2S1 releases, so the
                                #    in-flight ph0 refeed sends from a_xn are NOT interlocked
                                #    against the overwrite.
                                # The core write is what forces the a_xn prod-acquire, which
                                # blocks until all XN_REFEED ph0 sends have drained.
                                # NUMERICS: a_xn = a_o. o-proj consumes the attention output
                                # verbatim (the residual add against the layer input happens
                                # AFTER o-proj, in the _RMS_OP block below), so this handover
                                # is a pure copy.
                                a_o = AllocOp(rms_l1, [], [])
                                ChannelGet(
                                    "orms",
                                    a_o,
                                    indices=[idx(0)],
                                    offsets=[idx(0)],
                                    sizes=[idx(K)],
                                    strides=[idx(1)],
                                )
                                if not SURROUND_NOCOMPUTE:
                                    CallOp(rms_copy_aie, [a_xn, a_o])
                                refeed(
                                    OPROJ_REFEED,
                                    lambda: ChannelPut(
                                        "xnorm",
                                        a_xn,
                                        offsets=[idx(0)],
                                        sizes=[idx(K)],
                                        strides=[idx(1)],
                                    ),
                                )
                                DeallocOp(a_o)
                        a_op = None
                        a_h = a_in  # h = input when the o-proj group is cut
                        if _RMS_OP:
                            a_op = AllocOp(rms_l1, [], [])
                            # o-proj: single full-size get (the id-4 packet flow reassembles
                            # the _RMS_OP 512-packets into ONE dest BD) -- mirror Llama
                            # fused_decode.py:2662. Per-round 512 gets (the old form) cycle 4
                            # S2MM BDs on dest_rms and DEADLOCK once the down read-group is
                            # added (device-confirmed 2026-08-05g: NPH=4 per-round = TIMEOUT).
                            ChannelGet(
                                "outY",
                                a_op,
                                indices=[idx(0), idx(DEST_RMS)],
                                offsets=[idx(0)],
                                sizes=[idx(_RMS_OP * PAYLOAD)],
                                strides=[idx(1)],
                            )
                            a_h = AllocOp(rms_l1, [], [])
                            if not SURROUND_NOCOMPUTE:
                                CallOp(residual_add_aie, [a_h, a_in, a_op])
                        if LOOPCLOSE and GATEUP_PHASE < NPH:
                            # ph2 gate-up X = rmsnorm(input+oproj); per-put refeed GATEUP.
                            # REUSE a_xn (the ph0 buffer) so the two puts share ONE single-
                            # buffer ring -> the prod-acquire for each phase stays INTERLEAVED
                            # after its rms_norm (Llama's working cadence), not batched at the
                            # loop top.
                            CallOp(rms_norm_aie, [a_xn, a_h, a_w2, _one])
                            refeed(
                                GATEUP_REFEED,
                                lambda: ChannelPut(
                                    "xnorm",
                                    a_xn,
                                    offsets=[idx(0)],
                                    sizes=[idx(K)],
                                    strides=[idx(1)],
                                ),
                            )
                            DeallocOp(a_xn)
                            DeallocOp(a_w)
                            if PH2_RMSW:
                                DeallocOp(a_w2)
                        elif LOOPCLOSE:
                            # gate-up phase cut: the ph0 xnorm buffers are still live.
                            # (the OREF_NOPUT ph1 stand-in is emitted above, before the
                            # o-proj outY get, to avoid the rms<->proj cycle.)
                            DeallocOp(a_xn)
                            DeallocOp(a_w)
                            if PH2_RMSW:
                                DeallocOp(a_w2)
                        a_dn = None
                        a_r2 = a_h  # layer out = h when the down group is cut
                        if _RMS_DN:
                            a_dn = AllocOp(rms_l1, [], [])
                            # down: single full-size get (packet reassembly) -- mirror Llama
                            # fused_decode.py:2703.
                            ChannelGet(
                                "outY",
                                a_dn,
                                indices=[idx(0), idx(DEST_RMS)],
                                offsets=[idx(0)],
                                sizes=[idx(_RMS_DN * PAYLOAD)],
                                strides=[idx(1)],
                            )
                            a_r2 = AllocOp(rms_l1, [], [])
                            if not SURROUND_NOCOMPUTE:
                                CallOp(residual_add_aie, [a_r2, a_h, a_dn])
                        ChannelPut(
                            "layerOut",
                            a_r2,
                            offsets=[idx(0)],
                            sizes=[idx(K)],
                            strides=[idx(1)],
                        )
                        # Dealloc each distinct buffer once (a_h aliases a_in when the
                        # o-proj group is cut; a_r2 aliases a_h when the down group is).
                        _seen = []
                        for _b in (a_in, a_op, a_h, a_dn, a_r2):
                            if _b is not None and not any(_b is _s for _s in _seen):
                                _seen.append(_b)
                                DeallocOp(_b)

                    # Phase-isolation: instantiate rms only if pkt4 (oproj/down) is
                    # produced by a kept phase; else it would block on outY gets that
                    # never arrive = a FALSE deadlock. (_rms_body is defined but unused.)
                    if BUILD_RMS:
                        rms_h = herd(name="rms", sizes=[1, 1], operands=[])(_rms_body)
                        rms_h.attributes["link_with"] = StringAttr.get("rms_residual.o")
                        rms_h.attributes["x_loc"] = IntegerAttr.get(
                            T.i64(), 1
                        )  # tile(1,2)
                        rms_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

                    proj_h = herd(name="proj", sizes=[NCX, NCY], operands=[])(_core)
                    proj_h.attributes["link_with"] = StringAttr.get("proj_qmm.o")
                    # proj_qmm q4k dequant is stack-heavy (4x vector<float,128> +
                    # 4x bf16 + accums + sfix lambda per iter); 10K overflows into
                    # adjacent L1 buffers -> downstream egress lock waits forever
                    # (device hang). Bump ONLY proj to 24K (Qwen3.5 proj same fix).
                    proj_h.attributes["stack_size"] = IntegerAttr.get(T.i32(), 24576)
                    proj_h.attributes["x_loc"] = IntegerAttr.get(T.i64(), PROJ_COL0)
                    proj_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), PROJ_ROW0)

                    # ===== rope core (reference tile_1_3): QKV(dest0/pkt1) -> roped q/k/v =====
                    def _rope_body(tx, ty, _sx, _sy):
                        a_qkv = AllocOp(qkv_l1, [], [])
                        for _rq in range(DEST_RNDS[DEST_ROPE]):  # 5 QKV rounds
                            ChannelGet(
                                "outY",
                                a_qkv,
                                indices=[idx(0), idx(DEST_ROPE)],
                                offsets=[idx(_rq * PAYLOAD)],
                                sizes=[idx(PAYLOAD)],
                                strides=[idx(1)],
                            )
                        a_lut = AllocOp(ropew_l1, [], [])
                        ChannelGet("ropeLUT", a_lut, indices=[idx(0)])
                        a_q = AllocOp(ropeq_l1, [], [])
                        a_k = AllocOp(ropekv_l1, [], [])
                        a_v = AllocOp(ropekv_l1, [], [])
                        _one = arith.ConstantOp(IntegerAttr.get(i32, 1), None).result
                        if not SURROUND_NOCOMPUTE:
                            CallOp(rope_compute, [a_q, a_k, a_v, a_qkv, a_lut, _one])
                        # The reference: rope emits the FULL roped q (DQ=2048) on ONE MM2S -> mem_6_1
                        # q-broadcast tile, which fans+reshapes per-CU to the qk cores. This
                        # frees rope's 2nd MM2S for the k/v cache-append (added next). The
                        # ATTN=0 bisection still drains q to host per-CU (qDrain).
                        if ATTN:
                            # QWEN_ROPE_KV_FIRST=1 (witness): emit the KV appends BEFORE the
                            # ropeQ put. The KV cache readback is a progress witness, so this
                            # tells apart "rope never received its qkv" (KV still 0) from
                            # "rope received it but blocked on the ropeQ put" (KV becomes
                            # nonzero once the appends are no longer behind that put).
                            if ROPE_ECHO:
                                ChannelPut(
                                    "qDrain",
                                    a_qkv,
                                    offsets=[idx(0)],
                                    sizes=[idx(DQ + DK + DV)],
                                    strides=[idx(1)],
                                )
                            if not ROPE_KV_FIRST and not ROPE_ECHO:
                                ChannelPut(
                                    "ropeQ",
                                    a_q,
                                    indices=[idx(0)],
                                    offsets=[idx(0)],
                                    sizes=[idx(DQ)],
                                    strides=[idx(1)],
                                )
                            if KV_APPEND:
                                # reference-faithful append (mirror fused_decode.py:1596-1611):
                                # this token's roped-K (all heads) + raw-V -> appendK/appendV
                                # packet flows -> DDR cache. a_k/a_v are the full-width
                                # DK_TOT_A(256) all-CU K/V (were previously dropped).
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
                            if ROPE_KV_FIRST and not ROPE_ECHO:
                                ChannelPut(
                                    "ropeQ",
                                    a_q,
                                    indices=[idx(0)],
                                    offsets=[idx(0)],
                                    sizes=[idx(DQ)],
                                    strides=[idx(1)],
                                )
                        else:
                            for _cu in range(N_ATTN_CU):
                                ChannelPut(
                                    "qDrain",
                                    a_q,
                                    indices=[idx(0)],
                                    offsets=[idx(_cu * QCU)],
                                    sizes=[idx(QCU)],
                                    strides=[idx(1)],
                                )
                        DeallocOp(a_qkv)
                        DeallocOp(a_lut)
                        DeallocOp(a_q)
                        DeallocOp(a_k)
                        DeallocOp(a_v)

                    rope_h = herd(name="rope", sizes=[1, 1], operands=[])(_rope_body)
                    rope_h.attributes["link_with"] = StringAttr.get("rope.o")
                    rope_h.attributes["x_loc"] = IntegerAttr.get(T.i64(), ROPE_COL)
                    rope_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), ROPE_ROW)

                    # ===== glu core (reference tile_6_2): gate-up(dest2/pkt8) -> silu(gate)*up =====
                    # FAITHFUL 2-slot ping/pong ring (mirror Llama fused_decode.py:2216-2244):
                    # TWO glu slices per loop iter (ping gx_0/gh_0 + pong gx_1/gh_1) so air-to-aie
                    # forms a depth-2 S2MM/MM2S ring (lock init 2). A ROLLED 1-slice loop collapses
                    # to a 1-slot ring: the glu get is 1024 = TWO demux packets, so depth-1 cannot
                    # overlap the next 1024 fill with the current compute -> the glu core stalls on
                    # its gets and the pkt8 egress backpressures (device-confirmed 2026-08-05e:
                    # rolled loop = TIMEOUT even with glu west of the proj grid).
                    def _glu_body(tx, ty, _sx, _sy):
                        _one = arith.ConstantOp(IntegerAttr.get(i32, 1), None).result

                        def _slice(gx, gh):
                            ChannelGet(
                                "outY",
                                gx,
                                indices=[idx(0), idx(DEST_GLU)],
                                offsets=[idx(0)],
                                sizes=[idx(GLU_SLICE)],
                                strides=[idx(1)],
                            )
                            if not SURROUND_NOCOMPUTE:
                                CallOp(glu_aie, [gh, gx, _one])
                            # LOOPCLOSE: glu-down feeds the down-X refeed memtile (gluOut);
                            # stub path drains to host (gluDrain).
                            ChannelPut(
                                "gluOut" if LOOPCLOSE else "gluDrain",
                                gh,
                                offsets=[idx(0)],
                                sizes=[idx(GLU_HID)],
                                strides=[idx(1)],
                            )

                        if GLU_RING2:
                            # reference-EXACT depth-2 ring: allocate ONE ping pair and ONE pong
                            # pair OUTSIDE the loop and reuse them, mirroring reference tile_6_2
                            # (x_0/x_1 1024 S2MM, hid_0/hid_1 512 MM2S, 2 BDs per channel).
                            # Allocating fresh buffers per slice INSIDE the loop instead
                            # lowers to a SIX-deep ring on both channels (measured in
                            # air_project/npu.air.mlir tile_1_4: buf113/116/118/120/111/109
                            # MM2S + buf114/115/117/119/112/110 S2MM), and the down X then
                            # loses every third glu PAIR -- a period of exactly 6 chunks,
                            # matching that depth.
                            _g = [
                                (AllocOp(glu_x_l1, [], []), AllocOp(glu_hid_l1, [], []))
                                for _ in range(2)
                            ]
                            for _s in for_(idx(0), idx(NGLU // 2), idx(1)):
                                _slice(*_g[0])  # ping
                                _slice(*_g[1])  # pong
                                yield_([])
                            for _gx, _gh in _g:
                                DeallocOp(_gx)
                                DeallocOp(_gh)
                        else:
                            for _s in for_(idx(0), idx(NGLU // 2), idx(1)):
                                for _ in range(2):
                                    _gx = AllocOp(glu_x_l1, [], [])
                                    _gh = AllocOp(glu_hid_l1, [], [])
                                    _slice(_gx, _gh)
                                    DeallocOp(_gx)
                                    DeallocOp(_gh)
                                yield_([])

                    # Phase-isolation: instantiate glu only if pkt8 (gate-up) is
                    # produced; else it blocks on outY gets that never arrive.
                    if HAS_GLU:
                        glu_h = herd(name="glu", sizes=[1, 1], operands=[])(_glu_body)
                        glu_h.attributes["link_with"] = StringAttr.get("glu.o")
                        glu_h.attributes["x_loc"] = IntegerAttr.get(T.i64(), GLU_COL)
                        glu_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), GLU_ROW)

                    # QWEN_ATTN=0 bisection: skip the whole attn subsystem (it is the
                    # tail of seg_body) to isolate a device hang to attn vs base.
                    if not ATTN:
                        return

                    # ===== q-broadcast memtile (reference mem_tile_6_1 q_buffer<2048>) =====
                    # Get rope's full roped q (DQ=2048) and fan per-CU (1024 each) with
                    # the head-reshape stride to the qk cores. Mirrors fused_decode.py
                    # _qmtb_dec + the reference's q_buffer_1_6 MM2S reshape [<16,8>,<8,128>,<8,1>].
                    # CU c reads q heads 8c..8c+7 = element base c*1024 (= c*8 * stride 128).
                    qmtb = AllocOp(qmt_l2, [], [])
                    qmtb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                        T.i32(), QMT_COL
                    )
                    ChannelGet(
                        "hostQ" if ROPE_ECHO else "ropeQ", qmtb, indices=[idx(0)]
                    )
                    for cu in range(N_ATTN_CU):
                        ChannelPut(
                            "toAttnQ",
                            qmtb,
                            indices=[idx(cu)],
                            offsets=[idx(0), idx(cu * 8), idx(0)],
                            sizes=[idx(16), idx(8), idx(8)],
                            strides=[idx(8), idx(DH), idx(1)],
                        )
                    DeallocOp(qmtb)

                    # ===== KV staging (reference mem_tile_7_1): reshape combined K/V -> per-CU =====
                    # Ports fused_decode.py _reblock_dec (KV_SPLIT): per block, get the
                    # group's combined K (resp V) [16 keys, REGION_W] on the independent
                    # inKV_K/inKV_V rings, then fan+reshape to each CU's toK/toV with the
                    # mmul-tiled head layout (reference K [<16,8>,<16,256>,<8,1>], V
                    # [<2,2048>,<16,8>,<8,256>,<8,1>]). _gw = per-token group width = REGION_W.
                    _gw = (
                        REGION_W  # 256 (= len(cus)*KVPC_DH for the single col-7 group)
                    )
                    for gi, (_gcol, _cus) in enumerate(ATTN_COL_GROUPS):
                        _kvfor = _for_no_pingpong if DYNSEQ else for_
                        for _b in _kvfor(idx(0), _seg_ceil16(), idx(1)):
                            kb = AllocOp(kvcomb_l2, [], [])
                            kb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), ATTN_COL)
                            )
                            vb = AllocOp(kvcomb_l2, [], [])
                            vb.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), ATTN_COL)
                            )
                            ChannelGet("inKV_K", kb, indices=[idx(gi)])
                            ChannelGet("inKV_V", vb, indices=[idx(gi)])
                            for _lc, _cc in enumerate(_cus):
                                _pk = ChannelPut(
                                    "toK",
                                    kb,
                                    indices=[idx(_cc)],
                                    offsets=[idx(0), idx(0), idx(_lc * KVPC_DH)],
                                    sizes=[idx(16), idx(16), idx(8)],
                                    strides=[idx(8), idx(_gw), idx(1)],
                                )
                                _pv = ChannelPut(
                                    "toV",
                                    vb,
                                    indices=[idx(_cc)],
                                    offsets=[
                                        idx(0),
                                        idx(0),
                                        idx(0),
                                        idx(_lc * KVPC_DH),
                                    ],
                                    sizes=[idx(2), idx(16), idx(8), idx(8)],
                                    strides=[idx(_gw * 8), idx(8), idx(_gw), idx(1)],
                                )
                                # Reserve mem_7_1 MM2S 0 for the q-broadcast (mem_6_1->qk)
                                # + attn-o feedback (kv->mem_6_1) that transit col-7's
                                # switchbox: KV on MM2S 0 collides with that transit and
                                # deadlocks once attn actively couples qk<->kv. Mirror the
                                # proven Llama _reblock_dec fix (col-3 there) -> steer toK/
                                # toV onto memtile MM2S channels 1+.
                                _pk.operation.attributes[
                                    "air.memtile_dma_channel_min"
                                ] = IntegerAttr.get(T.i32(), 1)
                                _pv.operation.attributes[
                                    "air.memtile_dma_channel_min"
                                ] = IntegerAttr.get(T.i32(), 1)
                            DeallocOp(kb)
                            DeallocOp(vb)
                            yield_([])

                    # per-CU shared score buffers (L1, seg scope) -> herd operands so the
                    # qk (writer) + kv (reader) tiles of each CU share L1 (AIR shared-L1).
                    s_bufs = [AllocOp(as_l1, [], []) for _ in range(N_ATTN_CU)]

                    def _qk_body(sh, cu, Lh):
                        a_q = AllocOp(aq_l1, [], [])
                        ChannelGet("toAttnQ", a_q, indices=[idx(cu)])
                        a_m = AllocOp(m_l1, [], [])
                        a_cc = AllocOp(c_l1, [], [])
                        for _blk in for_(idx(0), _core_ceil16(Lh), idx(1)):
                            a_k = AllocOp(ak_l1, [], [])
                            ChannelGet("toK", a_k, indices=[idx(cu)])
                            blk_c = arith.index_cast(i32, _blk)
                            if not SKIP_QK:
                                CallOp(
                                    attn_qk_blk, [a_q, a_k, a_m, a_cc, sh, blk_c, Lh]
                                )
                            DeallocOp(a_k)
                            yield_([])
                        DeallocOp(a_q)
                        DeallocOp(a_m)
                        DeallocOp(a_cc)

                    def _kv_body(sh, cu, Lh):
                        a_y = AllocOp(y_l1, [], [])
                        a_l = AllocOp(lden_l1, [], [])
                        a_o = AllocOp(ao_l1, [], [])
                        for _blk in for_(idx(0), _core_ceil16(Lh), idx(1)):
                            a_v = AllocOp(av_l1, [], [])
                            ChannelGet("toV", a_v, indices=[idx(cu)])
                            blk_c = arith.index_cast(i32, _blk)
                            if not SKIP_KV:
                                CallOp(attn_kv_blk, [sh, a_v, a_y, a_l, blk_c, Lh])
                            DeallocOp(a_v)
                            yield_([])
                        if not SKIP_KVFIN:
                            CallOp(attn_kv_fin, [a_y, a_l, a_o])
                        ChannelPut(
                            "attnO",
                            a_o,
                            indices=[idx(cu)],
                            offsets=[idx(0)],
                            sizes=[idx(QCU)],
                            strides=[idx(1)],
                        )
                        DeallocOp(a_o)
                        DeallocOp(a_y)
                        DeallocOp(a_l)

                    def _leaf(ty, cu, sh, Lh, qk_ty):
                        _isqk = arith.cmpi(arith.CmpIPredicate.eq, ty, idx(qk_ty))
                        _if = IfOp(_isqk, [], has_else=True)
                        with InsertionPoint(_if.thenRegion.blocks[0]):
                            _qk_body(sh, cu, Lh)
                            yield_([])
                        with InsertionPoint(_if.elseRegion.blocks[0]):
                            _kv_body(sh, cu, Lh)
                            yield_([])

                    def _dispatch(ty, shs, Lh):
                        _lo = arith.cmpi(arith.CmpIPredicate.slt, ty, idx(2))
                        _ifp = IfOp(_lo, [], has_else=True)
                        with InsertionPoint(_ifp.thenRegion.blocks[0]):
                            _leaf(ty, 0, shs[0], Lh, 0)  # ty0=qk, ty1=kv (CU0)
                            yield_([])
                        with InsertionPoint(_ifp.elseRegion.blocks[0]):
                            _leaf(ty, 1, shs[1], Lh, 2)  # ty2=qk, ty3=kv (CU1)
                            yield_([])

                    # RTP-L: segment scope so this is a herd OPERAND (an RTP slot the
                    # instruction stream writes, hence patchable per token), not a
                    # herd-body constant that folds into the core ELF.
                    _Lc = (
                        _seg_L
                        if DYNSEQ
                        else arith.ConstantOp(IntegerAttr.get(i32, ATTN_L), None).result
                    )

                    @herd(
                        name="attn",
                        sizes=[1, 2 * N_ATTN_CU],
                        operands=[s_bufs[0].result, s_bufs[1].result, _Lc],
                    )
                    def attn_h(_tx, _ty, _sx, _sy, s0, s1, Lh):
                        _dispatch(_ty, [s0, s1], Lh)

                    attn_h.attributes["x_loc"] = IntegerAttr.get(T.i64(), ATTN_COL)
                    attn_h.attributes["y_loc"] = IntegerAttr.get(T.i64(), 2)

                    # X-loopclose refeed memtiles: emitted LAST, AFTER the q-broadcast
                    # and attn blocks. Emission order in seg_body is the order the
                    # memtile DMA programs are built, and mem_6_1 hosts BOTH the
                    # q-broadcast that FEEDS attn and the oref gather that CONSUMES
                    # attn's output. Emitting oref first armed the consumer ahead of
                    # the producer it depends on.
                    # ===== X-loopclose refeed memtiles (reference mem_tile_6_1 role) =====
                    # attn-o (ph1 oproj X) + glu-down (ph3 down X) gathered into col-6
                    # memtiles and re-broadcast into the convergent @xnorm via
                    # an OPROJ_REFEED-trip loop around the put. Convergence order:
                    # rms ph0 -> o ph1 -> rms ph2 -> glu ph3.
                    # Phase-isolation: each refeed memtile serves ONE phase's X, so emit it
                    # only when that phase is built (else it gathers a producer output that
                    # is never consumed / never produced = incoherent).
                    if (
                        LOOPCLOSE
                        and OPROJ_PHASE < NPH
                        and OREF_NOPUT != 1
                        and (not ORMS_HOST or OREF_DRAIN)
                    ):
                        # ph1: gather the 2 CUs' o (1024 each) into 2048, re-broadcast
                        # OPROJ_REFEED times into @xnorm as the o-proj X.
                        omtb = AllocOp(oref_l2, [], [])
                        omtb.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(), OREF_COL
                        )
                        # OREF_VIA_RMS: no refeed at the memtile -- it hands the gathered o
                        # over once and the rms core does the OPROJ_REFEED re-broadcast.
                        _oref_refeed_here = not (OREF_2HOP or OREF_VIA_RMS)
                        if OREF_NOCHAIN:
                            omtb.operation.attributes["air.no_chain_lock"] = (
                                UnitAttr.get()
                            )
                        if OREF_HOSTSRC:
                            ChannelGet(
                                "orefIn",
                                omtb,
                                offsets=[idx(0)],
                                sizes=[idx(K)],
                                strides=[idx(1)],
                            )
                        else:
                            for cu in range(N_ATTN_CU):
                                ChannelGet(
                                    "attnO",
                                    omtb,
                                    indices=[idx(cu)],
                                    offsets=[idx(cu * QCU)],
                                    sizes=[idx(QCU)],
                                    strides=[idx(1)],
                                )
                        if OREF_DRAIN:
                            ChannelPut(
                                "orefDrain",
                                omtb,
                                offsets=[idx(0)],
                                sizes=[idx(K)],
                                strides=[idx(1)],
                            )
                        elif not OREF_NOPUT:
                            refeed(
                                OPROJ_REFEED if _oref_refeed_here else 1,
                                lambda: ChannelPut(
                                    (
                                        "oref2"
                                        if OREF_2HOP
                                        else (
                                            "orms"
                                            if OREF_VIA_RMS
                                            else ("xnorm2" if PH1_XCHAN else "xnorm")
                                        )
                                    ),
                                    omtb,
                                    offsets=[idx(0)],
                                    sizes=[idx(K)],
                                    strides=[idx(1)],
                                ),
                            )
                        DeallocOp(omtb)
                        if OREF_2HOP and not OREF_NOPUT:
                            # refeed buffer: ONE fill BD (single get) so AIR's per-BD xN
                            # scaling lands only on it -> the reference's protocol shape.
                            omt2 = AllocOp(oref_l2, [], [])
                            omt2.operation.attributes["air.memtile_col"] = (
                                IntegerAttr.get(T.i32(), OREF2_COL)
                            )
                            ChannelGet(
                                "oref2",
                                omt2,
                                offsets=[idx(0)],
                                sizes=[idx(K)],
                                strides=[idx(1)],
                            )
                            refeed(
                                1 if OREF_VIA_RMS else OPROJ_REFEED,
                                lambda: ChannelPut(
                                    (
                                        "orms"
                                        if OREF_VIA_RMS
                                        else ("xnorm2" if PH1_XCHAN else "xnorm")
                                    ),
                                    omt2,
                                    offsets=[idx(0)],
                                    sizes=[idx(K)],
                                    strides=[idx(1)],
                                ),
                            )
                            DeallocOp(omt2)
                    if LOOPCLOSE and DOWN_PHASE < NPH:
                        # ph3: gather NGLU glu slices into INTERMEDIATE, re-broadcast
                        # DOWN_REFEED times into @xnorm as the down X.
                        db = AllocOp(downref_l2, [], [])
                        # glu-down on col0 (DISTINCT from attn-o on col6) so the @xnorm
                        # convergence does NOT merge o (ph1) + down (ph3) onto ONE mem_6_1
                        # MM2S ring. Sharing the ring forces the DMA to alternate o,down,o,
                        # down..., but o must be re-fed OPROJ_REFEED times for proj ph1
                        # BEFORE down (ph3) is ever produced (down needs ph1->ph2->glu) ->
                        # the ring blocks on the not-yet-ready down after 1 o send ->
                        # deadlock. reference keeps both on mem_6_1 but on SEPARATE MM2S channels
                        # (o=MM2S5, down=MM2S0); AIR can't force that per-channel, so mirror
                        # Llama and split onto distinct memtile columns.
                        db.operation.attributes["air.memtile_col"] = IntegerAttr.get(
                            T.i32(), 0
                        )
                        # UNROLLED gather (constant offsets), not a rolled scf.for with
                        # an IV-derived offset. Rolled, the NGLU=22 gets lower to a BD ring
                        # whose wrap does not divide 22, so chunks are duplicated and
                        # dropped: the down X came out as
                        #   [h0 h1 h2 h3 . . h0 h1 h8 h9 . . h12..h15 . . h12..h15]
                        # (device-measured 2026-08-06 by fitting the down contribution
                        # per hid chunk, recon cos 0.998). Llama's equivalent gather is 16
                        # chunks -- a power of two -- which is why it never showed there.
                        if DOWN_GATHER_UNROLL:
                            for _s in range(NGLU):
                                ChannelGet(
                                    "gluOut",
                                    db,
                                    offsets=[idx(_s * GLU_HID)],
                                    sizes=[idx(GLU_HID)],
                                    strides=[idx(1)],
                                )
                        elif DOWN_GATHER_PAIRS:
                            for _s in for_(idx(0), idx(NGLU // 2), idx(1)):
                                base = arith.muli(_s, idx(2 * GLU_HID))
                                for _h in range(2):
                                    ChannelGet(
                                        "gluOut",
                                        db,
                                        offsets=[arith.addi(base, idx(_h * GLU_HID))],
                                        sizes=[idx(GLU_HID)],
                                        strides=[idx(1)],
                                    )
                                yield_([])
                        else:
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
                        if DOWN_PUT_SPLIT:
                            for _s in range(NGLU):
                                ChannelPut(
                                    "xnorm",
                                    db,
                                    offsets=[idx(_s * GLU_HID)],
                                    sizes=[idx(GLU_HID)],
                                    strides=[idx(1)],
                                )
                        else:
                            refeed(
                                DOWN_REFEED,
                                lambda: ChannelPut(
                                    "xnorm",
                                    db,
                                    offsets=[idx(0)],
                                    sizes=[idx(INTERMEDIATE)],
                                    strides=[idx(1)],
                                ),
                            )
                        DeallocOp(db)

            # air.preserve_shim_dma_order: opt out of air-opt-shim-dma-bds'
            # per-channel BD regrouping. The weight channels (inW) are coupled by
            # the X broadcast multicast (all proj cores advance in lockstep), so the
            # round-major host put order (phase-outer, col-inner, above) is
            # load-bearing and must not be reordered into channel-major BDs (mirrors
            # Llama fused_decode.py).
            # Single-layer: emit the launch directly (no scf.for, no IV operand) so the
            # IR stays byte-identical to the verified build. Multi-layer: wrap it in an
            # AIR scf.for and thread the induction variable in as the last operand, so the
            # per-layer DDR offsets are loop-carried. airrt-to-npu unrolls this LATE, so
            # the AIR op count -- and the aie.device/xclbin -- stay constant; only the
            # runtime instruction sequence grows by NLAYERS sub-sequences.
            if NLAYERS > 1:
                for _iv in for_(idx(0), idx(NLAYERS), idx(1)):
                    launch(
                        sizes=[1, 1],
                        operands=list(_fa) + [_iv],
                        attributes={"air.preserve_shim_dma_order": UnitAttr.get()},
                    )(launch_body)
                    yield_([])
            else:
                launch(
                    sizes=[1, 1],
                    operands=list(_fa),
                    attributes={"air.preserve_shim_dma_order": UnitAttr.get()},
                )(launch_body)

    return build()


def run():
    module = build_module()

    # Emit-only hook (mirrors fused_decode.py): dump the built AIR MLIR and stop
    # before the expensive NPU compile, so a no-op refactor can be byte-diffed.
    if _os.environ.get("FUSED_DECODE_EMIT_ONLY"):
        print(str(module))
        return 0
    backend = XRTBackend(
        omit_while_true_loop=False,
        output_format="xclbin",
        kernel_name="MLIR_AIE",
        stack_size=10240,  # proj_qmm deepest frame ~7KB (<10K); stack-overflow ruled
        # out for this proj (measured; 16K global did not clear the hang).
        use_lock_race_condition_fix_v2=True,
        coalesce_shim_dma=True,
        debug_ir=bool(int(_os.environ.get("QWEN_DEBUG_IR", "0"))),
        # DYNSEQ: the runtime sequence holds a scalar, so the stream is built per
        # dispatch from the emitted header instead of read from insts.bin.
        emit_txn_cpp=bool(DYNSEQ),
    )
    print(
        f"[qwen_decode] proj grid {NCX}x{NCY}=16 cores cols "
        f"{PROJ_COL0}-{PROJ_COL0+NCX-1}, QKV rounds/core={_RB} nj={_NJ}"
    )
    art = backend.compile(
        module, output_binary_name="decode_qwen", insts="decode_qwen.insts.bin"
    )
    # Stamp the flag next to the artifact. The Makefile's own build stamp
    # (llms/qwen25_3b_q4/.decode_build_flags) only guards the BUILD; this one
    # guards the RUN, so that driving the builder directly cannot leave
    # qwen_prefill_to_decode feeding a dual-layout weight stream to a
    # single-channel xclbin (or vice versa) -- silent garbage rather than an
    # error. The runner checks this file; keep it one `KEY=VALUE` line.
    with open("decode_qwen.flags", "w") as _f:
        _f.write(f"W_DUAL_CHAN={W_DUAL_CHAN}\n")
    print(
        f"[qwen_decode] emitted {art.output_binary} + {art.insts} (W_DUAL_CHAN={W_DUAL_CHAN})"
    )
    return 0


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    exit(run())
