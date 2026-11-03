# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Geometry and buffer ABI of the Gemma4-E4B eight-token decoder.

The projection geometry is fused_decode's "gemma4-e4b-b8" entry, imported in
a saved and restored environment as dflash/qwen3_4b/target_config.py does.
Gemma4 adds two projection phases per layer for the per-layer embedding (its
gate, 2560 -> 256 padded to 512, and its projection, 256 padded to 512 ->
2560), so the per-layer weight and round counts are recomputed here.

The graph is built at the full-attention width, but each layer streams only
its class's weights (class_geometry). A sliding layer's q/k/v rows arrive
compact and the rope core spreads them into the full-width head slots; its
o projection reads only the first half of each head slot (v sits at the
front), which the attention gather puts first (half-major columns). A
KV-shared layer has no k/v rows. The projection cores keep the build's
rounds and x blocks, and take no weights for the ones past their class's.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[2]
FUSED = EXAMPLES / "fused_decode"
COMMON = HERE.parent / "common"

ABI_VERSION = 1
NUM_LAYERS = 42
# Layers that own a KV cache; the rest read layer 22's (sliding) or 23's (full)
KV_OWNERS = 24
FULL_PERIOD = 6
SLIDING_WINDOW = 512
TAP_LAYERS = (5, 11, 17, 23, 35)
DH = 512
KV_HEADS = 2
Q_HEADS = 8
# The per-layer embedding's width, padded to a whole projection row block per
# core (16 cores x 32 rows) and to two column blocks.
PLI_D = 256
PLE_PAD = 512
# The gate's output rows, padded to a whole GLU slice (two rounds), so the GLU
# core takes it like every other slice: every DMA channel then repeats one
# transfer shape in every wave, which one circular BD chain can express.
PLE_GATE_N = 1024
# Its two matrices are packed as two Q4 terms each (weights_b8.two_term_q4),
# and its input is fed twice.
PLE_TERMS = 2
# A rope table row: cos | sin | q_norm | k_norm | v_norm width (+ padding)
ROPE_W_LEN = 3 * DH + 16
# The phases of a layer and the core each one's output goes to.
PHASES = ("qkv", "o", "gate_up", "down", "ple_gate", "ple_proj")
DEST_NAMES = ("rope", "rms", "glu", "rms", "glu", "rms")


@dataclass(frozen=True)
class Config:
    capacity: int = 4096
    layers: int = NUM_LAYERS
    batch: int = 8
    stack_bytes: int = 6080

    def __post_init__(self):
        if self.batch != 8:
            raise ValueError("the Gemma4 decoder is built for eight tokens")
        if not 1 <= self.layers <= NUM_LAYERS:
            raise ValueError("layer count exceeds the model")
        if self.capacity < 16 or self.capacity % 16:
            raise ValueError("KV capacity must be a positive multiple of 16")

    @property
    def build_length(self):
        """One dynamic template covers every position in this capacity."""
        return self.capacity - self.batch + 1


def kv_slabs(layers):
    """KV slabs: one per owning layer, plus scratch for the shared layers'
    appends, which nothing reads."""
    return min(layers, KV_OWNERS) + 1


def kv_read_slab(layer):
    if layer < KV_OWNERS:
        return layer
    full = layer % FULL_PERIOD == FULL_PERIOD - 1
    return KV_OWNERS - 1 if full else KV_OWNERS - 2


def kv_append_slab(layer, layers):
    return layer if layer < KV_OWNERS else kv_slabs(layers) - 1


# A decode wave's arm: 1, plus these bits for its layer's class (arm 0 is an
# LM-head wave).
SLIDING_BIT = 2
SHARED_BIT = 4
ARMS = (1, 1 | SLIDING_BIT, 1 | SHARED_BIT, 1 | SLIDING_BIT | SHARED_BIT)


def is_sliding(layer):
    return layer % FULL_PERIOD != FULL_PERIOD - 1


def arm_of_layer(layer):
    return 1 | SLIDING_BIT * is_sliding(layer) | SHARED_BIT * (layer >= KV_OWNERS)


def class_geometry(arm, i2p, j2p, round_rows):
    """(I2, J2) per phase of an arm's real weights: the qkv rows of its q
    heads and, unless KV-shared, its k/v heads; a sliding layer's o
    projection over the first half of each head slot."""
    dh = DH // 2 if arm & SLIDING_BIT else DH
    rows = Q_HEADS * dh + (0 if arm & SHARED_BIT else 2 * KV_HEADS * dh)
    assert rows % round_rows == 0, rows
    i2, j2 = list(i2p), list(j2p)
    i2[0] = rows // round_rows
    j2[1] = j2p[1] * dh // DH
    return i2, j2


@contextmanager
def builder_environment(config, length):
    saved = dict(os.environ)
    try:
        for name in list(os.environ):
            if name.startswith(("DECODE_", "RMS_", "PROJ_", "Q4K_", "UNI_")):
                del os.environ[name]
        os.environ.update(
            DECODE_MODEL="gemma4-e4b-b8",
            VOCAB_CHUNK_I2="20",
            DECODE_GOLDEN="1",
            DECODE_GOLDEN_L=str(length),
            DECODE_UNI_DEC=str(config.layers),
            DECODE_STACK=str(config.stack_bytes),
            DECODE_HIDDEN_TAPS="1",
            W_DUAL_CHAN="1",
            NLAYERS="1",
            LM_HEAD="0",
            DECODE_DYNSEQ="1",
        )
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def load_geometry(config, length=None):
    length = config.build_length if length is None else length
    if str(FUSED) not in sys.path:
        sys.path.insert(0, str(FUSED))
    with builder_environment(config, length):
        spec = importlib.util.spec_from_file_location(
            "_gemma4_b8_geometry", FUSED / "fused_decode.py"
        )
        fd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fd)
    assert fd.ATTN_MAXL == config.capacity
    assert (fd.K, fd.M, fd.DQ_PADDED, fd.DK, fd.PAIR_ROWS) == (
        2560,
        6144,
        4096,
        1024,
        1,
    )
    return fd


def phases(fd):
    """Per-phase block counts, the four of fd's model entry plus the two
    per-layer-embedding ones."""
    i2p = list(fd.I2P) + [PLE_GATE_N // (fd.ROW_BLOCK * fd.NCX * fd.NCY), fd.I2P[1]]
    j2p = list(fd.J2P) + [
        PLE_TERMS * fd.J2P[0],
        PLE_TERMS * PLE_PAD // (2 * fd.COL_BLOCK),
    ]
    assert i2p == [12, 5, 40, 5, 2, 5] and j2p == [5, 8, 5, 20, 10, 2], (i2p, j2p)
    nbi = [i * fd.NCX * fd.NCY for i in i2p]
    nbj = [2 * j for j in j2p]
    per_col = [(nbi[p] // fd.NCX) * nbj[p] for p in range(len(i2p))]
    assert all(pc % fd.NCY == 0 for pc in per_col), per_col
    dest = [fd.DEMUX.index(name) for name in DEST_NAMES]
    rounds = list(i2p)
    round_rows = fd.ROW_BLOCK * fd.NCX * fd.NCY
    geom = {a: class_geometry(a, i2p, j2p, round_rows) for a in ARMS}
    class_per_col = {
        a: [i2[p] * fd.NCY * 2 * j2[p] for p in range(len(i2p))]
        for a, (i2, j2) in geom.items()
    }
    return SimpleNamespace(
        CLASS_GEOM=geom,
        CLASS_PER_COL=class_per_col,
        CLASS_SLAB={
            a: sum(fd.NCX * pc * fd.BLOCK_BF16 for pc in cpc)
            for a, cpc in class_per_col.items()
        },
        NPH=len(i2p),
        I2P=i2p,
        J2P=j2p,
        NBI_PH=nbi,
        NBJ_PH=nbj,
        PER_COL_PH=per_col,
        DEST=dest,
        W_LAYER=sum(fd.NCX * pc * fd.BLOCK_BF16 for pc in per_col),
        W_TOTAL_BLOCKS=sum(nbi[p] * nbj[p] for p in range(len(i2p))),
        W_FAN_STEPS=sum(pc // fd.NCY for pc in per_col),
        N_XBLK=sum(i2p[p] * nbj[p] for p in range(len(i2p))),
        REFEED=rounds,
        ROUNDS_PER_PH=rounds,
        ROUNDS_PER_DEST=[
            sum(r for r, d in zip(rounds, dest) if d == k) for k in range(fd.NDEST)
        ],
        N_ROUNDS=sum(rounds),
    )


def weight_offsets(config, fd):
    """Each layer's slab offset in the weight buffer, then the LM head's."""
    slab = phases(fd).CLASS_SLAB
    offs = [0]
    for layer in range(config.layers):
        offs.append(offs[-1] + slab[arm_of_layer(layer)])
    return offs


def abi(config, fd):
    b, k = config.batch, fd.K
    rms_layer = 5 * k + 16
    rope_offset = config.layers * rms_layer
    final_offset = rope_offset + config.layers * b * ROPE_W_LEN
    logits = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD * b
    q_scratch = logits + fd.UNI_LM * b * fd.VOCAB_SIZE_PADDED
    ph = phases(fd)
    return {
        "version": ABI_VERSION,
        "model": "Gemma4-E4B (Nemotron-3.5-Bolt-E4B)",
        "config": asdict(config),
        "quantization": "Q4NX packed q4k cascade, dual channel",
        "geometry": {
            "hidden": k,
            "vocab": 262144,
            "vocab_chunks": fd.UNI_LM,
            "vocab_chunk_width": fd.VOCAB_SIZE_PADDED,
            "rope_width": ROPE_W_LEN,
            "rms_layer": rms_layer,
            "w_offsets": weight_offsets(config, fd),
            "kv_layer": fd.KV_LAYER,
            "kv_slabs": kv_slabs(config.layers),
            "phases": list(PHASES),
        },
        "elements": {
            "x": (config.layers + 1) * b * k,
            "w": weight_offsets(config, fd)[-1]
            + fd.UNI_LM * fd.VOCAB_W_BLOCKS * fd.BLOCK_BF16,
            "rms": final_offset + k,
            "y": q_scratch + b * Q_HEADS * DH,
            "kv": kv_slabs(config.layers) * fd.KV_LAYER,
            "pli": config.layers * b * PLE_PAD,
        },
        "offsets": {
            "rope": rope_offset,
            "final_norm": final_offset,
            "logits": logits,
            "q_scratch": q_scratch,
        },
        "rms_layout": "[input | pre_ffn | post_attn | post_ffn | post_ple | unused x16]",
        "kv_layout": "[slab, K rows then V rows, position, KV heads x 512]",
        "logits_layout": "[vocabulary chunk, token, within-chunk vocabulary]",
        "hidden_layout": "[layer boundary, token, hidden]; boundary 0 is the input",
    }


def load_layout(config, length=None):
    fd = load_geometry(config, length)
    b = config.batch
    ph = phases(fd)

    def outy_tokmajor(row_stride, base=0, rounds=1):
        return (
            [0, 0, 0, base],
            [rounds, b, fd.N_PAIRS, fd.PAIR_PAY],
            [fd.PAYLOAD, row_stride, fd.PAIR_PAY, 1],
        )

    ns = vars(fd).copy()
    ns.update(vars(ph))
    ns.update(
        BATCH=b,
        CAPACITY=config.capacity,
        LAYERS=config.layers,
        KV_SLABS=kv_slabs(config.layers),
        W_OFFS=weight_offsets(config, fd),
        SLIDING_BIT=SLIDING_BIT,
        SHARED_BIT=SHARED_BIT,
        arm_of_layer=arm_of_layer,
        KV_OWNERS=KV_OWNERS,
        FULL_PERIOD=FULL_PERIOD,
        SLIDING_WINDOW_G=SLIDING_WINDOW,
        DH_G=DH,
        KV_HEADS_G=KV_HEADS,
        Q_HEADS_G=Q_HEADS,
        PLE_PAD=PLE_PAD,
        # re-reads of the gate's input (the residual, chunked) and of the
        # projection's (GELU(gate) * pli, one 512 chunk per row block and term)
        PLE_GATE_N=PLE_GATE_N,
        PLE_X_REPLAYS=PLE_TERMS * (PLE_GATE_N // (fd.ROW_BLOCK * fd.NCX * fd.NCY)),
        PLE_G_REFEEDS=PLE_TERMS * fd.I2P[1],
        PLE_TERMS_G=PLE_TERMS,
        ROPE_W_LEN=ROPE_W_LEN,
        RMS_LAYER=5 * fd.K + 16,
        ABI=abi(config, fd),
        XMT_PCOL=4,
        # The down projection's input shares the X memory tile, so its re-reads
        # never cross the array: a copy across it and an egress packet flow can
        # meet at one arbiter with no route keeping them apart.
        DOWN_PCOL=4,
        MAIN_PCOL=2,
        GRP_ROWS_B=fd.HDR + fd.LEADS_PER_GRP * fd.PAIR_PAY,
        MAIN_ROWS_B=fd.HDR + fd.PAYLOAD,
        PAYLOAD_B=fd.PAYLOAD * b,
        L1_VEC_BF16=32,
        STG_W=max(2 * fd.COL_BLOCK, fd.PAYLOAD),
        XCHUNK=2 * fd.COL_BLOCK,
        XCHUNK_MUL=2,
        VOCAB_DEST=fd.GLU_DEST,
        VOC_SLICE_RNDS=fd.GLU_SLICE // fd.PAYLOAD,
        VOC_ROTATIONS=fd.VOCAB_RNDS // (2 * (fd.GLU_SLICE // fd.PAYLOAD)),
        _XFEED_BD=([fd.COL_BLOCK // 8, b, 8], [8, 2 * fd.COL_BLOCK, 1]),
        outy_tokmajor=outy_tokmajor,
    )
    assert ns["VOCAB_RNDS"] % (2 * ns["VOC_SLICE_RNDS"]) == 0
    from air.ir import (
        ArrayAttr,
        InsertionPoint,
        StringAttr,
        UnitAttr,
    )
    from air.dialects.air import (
        Channel,
        ChannelGet,
        ChannelPut,
        herd,
        launch,
        module_builder,
        segment,
        channel,
    )
    from air.dialects.func import FuncOp, CallOp
    from air.dialects.memref import AllocOp, DeallocOp
    from air.dialects.scf import IfOp

    ns.update(
        ArrayAttr=ArrayAttr,
        InsertionPoint=InsertionPoint,
        StringAttr=StringAttr,
        UnitAttr=UnitAttr,
        Channel=Channel,
        ChannelGet=ChannelGet,
        ChannelPut=ChannelPut,
        herd=herd,
        launch=launch,
        module_builder=module_builder,
        segment=segment,
        channel_decl=channel,
        FuncOp=FuncOp,
        CallOp=CallOp,
        AllocOp=AllocOp,
        DeallocOp=DeallocOp,
        IfOp=IfOp,
    )
    return SimpleNamespace(**ns)
