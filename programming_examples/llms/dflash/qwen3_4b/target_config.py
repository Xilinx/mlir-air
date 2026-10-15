# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Explicit configuration and buffer ABI for the standalone NPU target.

Importing the upstream builder is confined to a save/restore environment adapter.
The runtime consumes the saved manifest rather than ambient DECODE_* settings.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[2]
REPO = HERE.parents[3]
FUSED = EXAMPLES / "fused_decode"
ABI_VERSION = 1


@dataclass(frozen=True)
class TargetConfig:
    role: str = "target"
    capacity: int = 4096
    batch: int = 8
    layers: int = 36
    taps: bool = True
    causal: bool = True
    stack_bytes: int = 6080
    l2_pingpong: bool = False
    chain_pingpong: bool = True
    activation_memtile_col: int = 4
    gather_mode: str = "packet"
    token_packets: bool = True
    dynamic_context: bool = True
    projection_weight_pingpong: bool = True
    attention_query_tile: int = 4
    attention_layout: str = "fused-shards"
    normalization_replay: int = 1
    normalization_cache: bool = True
    attention_pair_mode: int = 3
    vector_attention_norm: bool = True
    attention_optimization: str = "pv"
    q4k_unpack_fma: bool = False
    q4k_unpack_unroll: int = 0
    attention_native_key: bool = True
    attention_early_key_release: bool = True
    attention_pv_reuse: bool = True
    projection_schedule: str = "groups"

    def __post_init__(self):
        # Mapping choices are recorded in each artifact, but are not tuning
        # options: this example ships one B8 floorplan and arithmetic path.
        from dataclasses import fields

        variable = {"role", "capacity", "layers", "causal"}
        for field in fields(self):
            if (
                field.name not in variable
                and getattr(self, field.name) != field.default
            ):
                raise ValueError(f"Unsupported B8 mapping option: {field.name}")
        if self.role not in ("target", "drafter"):
            raise ValueError("role must be target or drafter")
        if self.causal != (self.role == "target"):
            raise ValueError(
                "Target attention is causal; drafter attention is bidirectional"
            )
        if not 1 <= self.layers <= (36 if self.role == "target" else 5):
            raise ValueError("Layer count exceeds the selected checkpoint")
        if self.capacity < 16 or self.capacity % 16:
            raise ValueError("KV capacity must be a positive multiple of 16")

    @property
    def build_lengths(self):
        last = self.capacity - self.batch + 1
        return (last, last - 1)


@contextmanager
def builder_environment(config, length):
    saved = dict(os.environ)
    try:
        for name in list(os.environ):
            if name.startswith(("DECODE_", "RMS_", "PROJ_", "Q4K_", "UNI_")):
                del os.environ[name]
        os.environ.update(
            DECODE_MODEL="qwen3-4b",
            VOCAB_CHUNK_I2="30",
            DECODE_GOLDEN="1",
            DECODE_GOLDEN_L=str(length),
            DECODE_UNI_DEC=str(config.layers),
            DECODE_STACK=str(config.stack_bytes),
            DECODE_HIDDEN_TAPS=str(int(config.taps)),
            W_DUAL_CHAN="1",
            NLAYERS="1",
            LM_HEAD="0",
            DECODE_DYNSEQ=str(int(config.dynamic_context)),
        )
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def load_geometry(config, length=None):
    length = config.build_lengths[0] if length is None else length
    if length not in config.build_lengths:
        raise ValueError("build length must be one of the explicit calibration pair")
    if str(FUSED) not in sys.path:
        sys.path.insert(0, str(FUSED))
    with builder_environment(config, length):
        spec = importlib.util.spec_from_file_location(
            "_dflash_upstream_geometry", FUSED / "fused_decode.py"
        )
        fd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fd)
    assert fd.ATTN_MAXL == config.capacity
    assert (fd.K, fd.M, fd.DK_TOT_A, fd.DH, fd.N_ATTN_CU, fd.PAIR_ROWS) == (
        2560,
        6144,
        1024,
        128,
        4,
        1,
    )
    return fd


def abi(config, fd):
    b = config.batch
    rope_offset = config.layers * fd.RMS_LAYER
    final_offset = rope_offset + config.layers * b * fd.ROPE_W_LEN
    decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD * b
    return {
        "version": ABI_VERSION,
        "model": (
            "z-lab/Qwen3-4B-DFlash-b16" if config.role == "drafter" else "Qwen/Qwen3-4B"
        ),
        "config": asdict(config),
        "quantization": "Q4NX packed q4k cascade, dual channel",
        "attention_context": (
            "live blocks" if config.dynamic_context else "compiled capacity"
        ),
        "attention_output": (
            "cascade join then memory-tile gather"
            if config.attention_layout == "fused-shards"
            else "memory-tile gather"
        ),
        "activation": "BF16",
        "projection_compute": "BFP16 mmul<8,8,8>" if b >= 8 else "GEMV",
        "q4k_unpack_arithmetic": (
            (
                "single BF16 rounding after FP32 multiply-add"
                if config.q4k_unpack_fma
                else "BF16 product rounded before offset addition"
            )
            if b >= 8
            else "fused GEMV"
        ),
        "rounding": (
            "nearest-even explicitly set at kernel entry"
            if b >= 8
            else "current-main Qwen default"
        ),
        "geometry": {
            "hidden": fd.K,
            "vocab": fd.VOCAB_SIZE,
            "vocab_chunks": fd.UNI_LM,
            "vocab_chunk_width": fd.VOCAB_SIZE_PADDED,
            "kv_groups": fd.NGRP,
            "kv_region_width": fd.REGION_W,
            "rope_width": fd.ROPE_W_LEN,
        },
        "elements": {
            "x": (config.layers + 1 if config.taps else 1) * b * fd.K,
            "w": config.layers * fd.W_LAYER
            + fd.UNI_LM * fd.VOCAB_W_BLOCKS * fd.BLOCK_BF16,
            "r": final_offset + fd.K,
            "y": decode_y + fd.UNI_LM * b * fd.VOCAB_SIZE_PADDED,
            "kv": config.layers * config.capacity * fd.KVSZ_TOK,
        },
        "offsets": {
            "rope": rope_offset,
            "final_norm": final_offset,
            "logits": decode_y,
        },
        "kv_layout": "[layer, K-groups then V-groups, position, region-width]",
        "logits_layout": "[vocabulary chunk, token, within-chunk vocabulary]",
        "hidden_layout": "[layer boundary, token, hidden]; boundary 0 is the input",
    }


def load_layout(config, length=None):
    fd = load_geometry(config, length)
    b = config.batch

    def outy_tokmajor(row_stride, base=0, rounds=1):
        if config.token_packets:
            return (
                [0, 0, 0, base],
                [rounds, b, fd.N_PAIRS, fd.PAIR_PAY],
                [fd.PAYLOAD, row_stride, fd.PAIR_PAY, 1],
            )
        return (
            [0, 0, base],
            [fd.N_PAIRS * rounds, b, fd.PAIR_PAY],
            [fd.PAIR_PAY, row_stride, 1],
        )

    extra = {
        "BATCH": b,
        "CAUSAL": config.causal,
        "ARM_IDLE_CASES": [0],
        "MULTIBLK": True,
        "DYNSEQ_RB": True,
        "DYNSEQ_MEM": True,
        "DYNSEQ_TRIP": True,
        "PROJECTION_WEIGHT_PINGPONG": config.projection_weight_pingpong,
        "ATTENTION_QUERY_TILE": config.attention_query_tile,
        "PARALLEL_ATTENTION": config.attention_layout == "fused-shards",
        "EARLY_KEY_RELEASE": config.attention_early_key_release,
        "KV_QUERY_GROUPS": (
            b // 8
            if config.attention_layout == "fused-shards"
            else b // config.attention_query_tile
        ),
        "NORMALIZATION_REPLAY": config.normalization_replay,
        "NORMALIZATION_CACHE": config.normalization_cache,
        "XMT_PCOL": config.activation_memtile_col,
        "MAIN_PCOL": 2,  # Selected B8 gather owner, independent of B1 placement.
        "CIRCUIT_GATHER": config.gather_mode == "circuit",
        "CIRCUIT_MAIN": config.gather_mode != "packet",
        "TOKEN_PACKETS": config.token_packets,
        "PACKET_BATCH": 1 if config.token_packets else b,
        "GRP_ROWS_B": fd.HDR
        + fd.LEADS_PER_GRP * fd.PAIR_PAY * (1 if config.token_packets else b),
        "MAIN_ROWS_B": fd.HDR + fd.PAYLOAD * (1 if config.token_packets else b),
        "PAYLOAD_B": fd.PAYLOAD * b,
        "L1_VEC_BF16": 32,
        "STG_W": max(2 * fd.COL_BLOCK, fd.PAYLOAD),
        "XCHUNK": 2 * fd.COL_BLOCK,
        "XCHUNK_MUL": 2,
        "VOCAB_DEST": fd.GLU_DEST,
        "VOC_SLICE_RNDS": fd.GLU_SLICE // fd.PAYLOAD,
        "VOC_ROTATIONS": fd.VOCAB_RNDS // (2 * (fd.GLU_SLICE // fd.PAYLOAD)),
        "_ATTN_COLS": {col for col, _, _ in fd.ATTN_CU_LOC},
        "_XFEED_BD": ([fd.COL_BLOCK // 8, b, 8], [8, 2 * fd.COL_BLOCK, 1]),
        "outy_tokmajor": outy_tokmajor,
        "contextlib": __import__("contextlib"),
    }
    from fused_attention_graph import ParallelAttentionGraph

    ns = vars(fd).copy()
    ns.update(extra)
    ns["ParallelAttentionGraph"] = ParallelAttentionGraph
    from air.ir import ArrayAttr, StringAttr, UnitAttr
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

    ns.update(
        ArrayAttr=ArrayAttr,
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
        HIDDEN_TAPS=config.taps,
        X_SLOTS=config.layers + 1 if config.taps else 1,
    )

    def set_attention_link(op, base):
        op.attributes["link_with"] = StringAttr.get(base + ".ll")
        op.attributes["link_with_mode"] = StringAttr.get("merge")

    ns["_set_attn_link"] = set_attention_link
    return SimpleNamespace(**ns)


def build_module(config, length):
    import batch_graph

    layout = load_layout(config, length)
    return batch_graph.build_module(layout), abi(config, layout)
