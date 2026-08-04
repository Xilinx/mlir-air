# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader for the Llama-3.2-3B Q4NX example.
#
# Reuses the shape-agnostic Q4NX safetensors reader/dequantizer (`Q4nxModel`)
# from the 1B example and assembles the dequantized bf16 matrices into the
# `llama32_3b` LlamaWeights container. From there the bf16 llama32_3b driver
# (prefill + decode + LM head) runs unchanged — Q4NX only changes the weight
# source. 3B ties word embeddings (tie_word_embeddings=true), so lm_head IS the
# full-precision embed matrix.
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_LLMS = _HERE.parent
_LLAMA1B_Q4NX = _LLMS / "llama32_1b_q4nx"
_LLAMA3B = _LLMS / "llama32_3b"
for _p in (str(_LLMS), str(_LLAMA1B_Q4NX), str(_LLAMA3B), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llama32_1b_q4nx_weights import Q4nxModel  # noqa: E402  (shape-agnostic)
from llama32_3b_weights import (  # noqa: E402
    LlamaConfig,
    LayerWeights,
    LlamaWeights,
    generate_rope_lut,
)


def load_q4nx_weights(model_source, config=None):
    """Load Llama-3.2-3B weights from a `model.q4nx` bundle into a LlamaWeights.

    `model_source` may be an HF repo id (contains '/'), a directory containing
    model.q4nx, or a direct model.q4nx file path. Q4NX projections are dequant'd
    to bf16 on the host; norms/embeddings are read as bf16.
    """
    if config is None:
        config = LlamaConfig()

    qm = Q4nxModel(model_source)
    embed, norm, lm_head = qm.embed_norm_lmhead()  # (tied: lm_head is embed)
    embed = np.asarray(embed, bfloat16)
    norm = np.asarray(norm, bfloat16)
    lm_head = np.asarray(lm_head, bfloat16)

    layers = []
    for k in range(config.n_layers):
        w = qm.layer_weights(k)  # {q,k,v,o,up,gate,down}, each [K, out] bf16
        rms_in, rms_post = qm.layer_rms(k)
        layers.append(
            LayerWeights(
                attn_norm=np.asarray(rms_in, bfloat16),
                wq=np.asarray(w["q"], bfloat16),
                wk=np.asarray(w["k"], bfloat16),
                wv=np.asarray(w["v"], bfloat16),
                wo=np.asarray(w["o"], bfloat16),
                ffn_norm=np.asarray(rms_post, bfloat16),
                w_gate=np.asarray(w["gate"], bfloat16),
                w_up=np.asarray(w["up"], bfloat16),
                w_down=np.asarray(w["down"], bfloat16),
            )
        )

    return LlamaWeights(
        embed_table=embed,
        layers=layers,
        final_norm=norm,
        lm_head=lm_head,
    )
