# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The models on the shared fused prefill (dense.py, lfm2.py), with the shape
the device is built for. Norm eps and rope come from the weights' config.json
at load, which must agree with these."""

from typing import NamedTuple

from . import device as D
from .build import attn_points


class Desc(NamedTuple):
    repo: str  # HF repo of the weights
    d: int
    dh: int
    heads: int
    kv_heads: int
    inter: int
    layers: int
    vocab: int
    qk_norm: bool = False
    qkv_bias: bool = False
    tied: bool = True  # LM head is the bf16 embedding, else the Q4NX lm_head
    gemma: bool = False  # norms around both sublayers, GELU
    window: int = 0  # sliding window of the layers that are not global
    global_every: int = 0  # layer L is global when (L + 1) % global_every == 0
    max_len: int = 2048  # KV records per head; longer prompts take the per-op prefill
    lkp: int = 32
    family: str = "dense"  # "lfm2": lfm2.py
    tol: float = 0.99  # verify's logit cosine gate against the fp32 reference


def layer_op(desc, L):
    """The attention op of layer L."""
    if not desc.window:
        return "a"
    return "f" if (L + 1) % desc.global_every == 0 else "s"


class Spec(NamedTuple):
    CFG: D.Config
    ATTN_POINTS: dict
    shapes: tuple

    def gemm_shapes(self):
        return list(self.shapes)


def spec(desc):
    bfp = desc.dh <= 128
    a = D.Attn(
        col=0,
        dh=desc.dh,
        lkp=desc.lkp,
        dvt=min(desc.dh, 128),
        kv_heads=desc.kv_heads // 2,
        # attn_bfp16 reads an even number of blocks, so one past the last
        max_blocks=desc.max_len // desc.lkp + bfp,
        window_rtp=bool(desc.window),
        kern="bfp16" if bfp else "npu2",
    )
    groups = ("a0", "a1")
    cfg = D.Config(
        attn={"a0": a, "a1": a._replace(col=7)},
        act="gelu" if desc.gemma else "silu",
        ops={op: groups for op in (("f", "s") if desc.window else ("a",))},
        windows={"s": desc.window} if desc.window else None,
    )
    n = lambda v: -(-v // D.NR) * D.NR  # noqa: E731
    dq, dk = desc.heads * desc.dh, desc.kv_heads * desc.dh
    shapes = {
        (desc.d, n(dq + 2 * dk), 0, 1),  # q | k | v
        (dq, n(desc.d), 0, 1),  # o
        (desc.d, n(desc.inter), 1, 1),  # gate, activation in the drain
        (desc.d, n(desc.inter), 0, 1),  # up
        (desc.inter, n(desc.d), 0, 1),  # down
    }
    if desc.family == "lfm2":
        shapes.add((desc.d, n(3 * desc.d), 0, 1))  # ShortConv in_proj
    # LM head, at most MAX_N columns per op
    shapes |= {
        (desc.d, n(n1 - n0), 0, 0 if desc.tied else 1)
        for n0, n1 in D.n_split(desc.vocab)
    }
    return Spec(cfg, attn_points(desc.heads, even=bfp), tuple(sorted(shapes)))


MODELS = {
    "llama32_1b_q4nx": Desc(
        "FastFlowLM/Llama-3.2-1B-NPU2", 2048, 64, 32, 8, 8192, 16, 128256
    ),
    "llama32_3b_q4nx": Desc(
        "FastFlowLM/Llama-3.2-3B-NPU2", 3072, 128, 24, 8, 8192, 28, 128256
    ),
    "llama31_8b_q4nx": Desc(
        "FastFlowLM/Llama-3.1-8B-NPU2", 4096, 128, 32, 8, 14336, 32, 128256, tied=False
    ),
    "qwen3_4b_q4nx": Desc(
        "FastFlowLM/Qwen3-4B-NPU2", 2560, 128, 32, 8, 9728, 36, 151936, qk_norm=True
    ),
    "qwen3_8b_q4nx": Desc(
        "FastFlowLM/Qwen3-8B-NPU2",
        4096,
        128,
        32,
        8,
        12288,
        36,
        151936,
        qk_norm=True,
        tied=False,
    ),
    "gemma3_4b_q4nx": Desc(
        "FastFlowLM/Gemma3-4B-NPU2",
        2560,
        256,
        8,
        4,
        10240,
        34,
        262208,
        qk_norm=True,
        tied=False,
        gemma=True,
        window=1024,
        global_every=6,
        tol=0.98,
    ),
    "lfm2_1_2b_q4nx": Desc(
        "LiquidAI/LFM2-1.2B",
        2048,
        64,
        32,
        8,
        8192,
        16,
        65536,
        qk_norm=True,
        family="lfm2",
    ),
    "phi4_mini_q4nx": Desc(
        "FastFlowLM/Phi4-mini-Instruct-NPU2", 3072, 128, 24, 8, 8192, 32, 200064
    ),
}
