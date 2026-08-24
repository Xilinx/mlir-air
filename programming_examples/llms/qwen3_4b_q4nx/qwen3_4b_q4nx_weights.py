# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader + dequant for the Qwen3-4B Q4NX example -- the DFlash
# target model (docs/DFlashFeasibility.md).
#
# Qwen3-4B is the SAME topology as Qwen3-8B: llama-shaped q4nx, DH=128, standard
# 2-norm pre-norm with plain (un-folded) weights, SiLU GLU, single-theta RoPE,
# plus Qwen3's per-head qk-norm riding in the rope_w slab. So rather than copy
# qwen3_8b_q4nx_weights.py and edit seven numbers, this RE-PARAMETERIZES it: the
# codec, the header parsing, the accessors and the reference forward are all
# dimension-independent given _PROJ, and are inherited.
#
# TWO deltas from 8B, both real:
#
#   1. Dimensions. D=2560 (not 4096), INTER=9728 (not 12288). Head counts,
#      head_dim, layer count, vocab and rope theta are identical. Taken from
#      llms/qwen3_4b/qwen3_4b_weights.py's LlamaConfig, which is the repo's
#      existing bf16 Qwen3-4B, not from a datasheet.
#   2. THE LM HEAD IS TIED. Qwen3-4B sets tie_word_embeddings=true where 8B does
#      not, so the head is the bf16 embedding matrix and the bundle carries no
#      separate lm_head.weight. Asking for one -- which is what inheriting 8B's
#      embed_norm_lmhead would do -- raises a KeyError on a tensor that is not
#      there. This is the whole reason the class is subclassed at all.
#
# NOT Gemma: no (1+w) norm fold, no embedding scale, no dual-theta or sliding
# window.
import sys
from pathlib import Path

import numpy as np

_Q8B = str(Path(__file__).resolve().parents[1] / "qwen3_8b_q4nx")
if _Q8B not in sys.path:
    sys.path.insert(0, _Q8B)

from qwen3_8b_q4nx_weights import (  # noqa: E402
    Q4nxModel as _Q4nxModel8B,
    _bf,
    generate_rope_lut,
    resolve_q4nx_model,
    rope_w_layer,
)

# Qwen3-4B dims. Identical to 8B except D and INTER.
D = 2560  # hidden_size          (8B: 4096)
DH = 128  # head_dim
N_Q_HEADS = 32
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4 (GQA)
DQ = N_Q_HEADS * DH  # 4096 -- NOT equal to D, so o_proj is 4096 -> 2560
DK = N_KV_HEADS * DH  # 1024
DV = DK
INTER = 9728  # mlp intermediate     (8B: 12288)
NUM_LAYERS = 36
VOCAB = 151936
RMS_EPS = 1e-6
ROPE_THETA = 1000000.0

# The decoupled q dim is the one thing that trips people up here: 8B has
# DQ == D == 4096 so its o_proj is square, and 4B's is not.
assert DQ != D, "Qwen3-4B has a decoupled q dim; o_proj contracts 4096 -> 2560"


class Q4nxModel(_Q4nxModel8B):
    """Qwen3-8B's Q4NX reader at Qwen3-4B's dimensions, with a TIED head.

    Everything inherited (header parse, `bf16`, `dequant`, `layer_weights`,
    `layer_rms`, `layer_qk_norm`) is written against `_PROJ` and the generic
    block constants, so overriding `_PROJ` re-targets it."""

    _PROJ = {
        "q": ("self_attn.q_proj", DQ, D),
        "k": ("self_attn.k_proj", DK, D),
        "v": ("self_attn.v_proj", DV, D),
        "o": ("self_attn.o_proj", D, DQ),
        "up": ("mlp.up_proj", INTER, D),
        "gate": ("mlp.gate_proj", INTER, D),
        "down": ("mlp.down_proj", D, INTER),
    }

    def embed_norm_lmhead(self):
        """(embed_in [VOCAB,D] bf16, final_norm [D] f32, lm_head [VOCAB,D] f32).

        TIED: the head IS the embedding matrix. 8B's override reads a separate
        Q4NX `lm_head.weight`; this bundle does not carry one."""
        embed_in = self.bf16("model.embed_tokens.weight")
        norm = self.bf16("model.norm.weight").astype(np.float32)
        return embed_in, norm, embed_in


def qwen3_4b_q4nx_config(n_layers=NUM_LAYERS):
    """The qwen3_4b LlamaConfig, which is already Qwen3-4B -- only the tie flag
    is restated, to make it explicit at this call site rather than a default."""
    _QWEN3_4B = str(Path(__file__).resolve().parents[1] / "qwen3_4b")
    if _QWEN3_4B not in sys.path:
        sys.path.insert(0, _QWEN3_4B)
    from qwen3_4b_weights import LlamaConfig

    return LlamaConfig(
        n_layers=n_layers,
        emb_dim=D,
        n_heads=N_Q_HEADS,
        head_dim=DH,
        n_kv_heads=N_KV_HEADS,
        hidden_dim=INTER,
        vocab_size=VOCAB,
        rope_base=ROPE_THETA,
        tie_word_embeddings=True,  # 4B ties; 8B does not
    )


__all__ = [
    "Q4nxModel",
    "qwen3_4b_q4nx_config",
    "resolve_q4nx_model",
    "generate_rope_lut",
    "rope_w_layer",
    "_bf",
    "D",
    "DH",
    "DQ",
    "DK",
    "DV",
    "INTER",
    "NUM_LAYERS",
    "VOCAB",
    "RMS_EPS",
    "ROPE_THETA",
    "N_Q_HEADS",
    "N_KV_HEADS",
    "Q_PER_KV",
]
