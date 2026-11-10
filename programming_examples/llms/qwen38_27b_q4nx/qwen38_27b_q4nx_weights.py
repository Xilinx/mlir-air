# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""FastFlowLM's Qwen3.8-27B `model.q4nx` (FastFlowLM/Qwen3.8-27B-NPU2), text
weights only.

The block formats are Qwen3.5's (q35_weights): every projection is Q4K in
32 x 256 blocks; lm_head and the embedding are Q8 in the same blocking (the
two are separate tables, not tied). Norm weights already hold the (1 + w) of
the model's RMSNorm, and ssm_a holds -exp(A_log).

Layer l is a full-attention layer when l % 4 == 3, else a Gated DeltaNet
layer. Per layer:

  GDN        qkv_proj [q 2048 | k 2048 | v 6144], z (stored as
             self_attn.gate_proj) 6144, ssm_out 5120 x 6144, alpha / beta
             [48, 5120] bf16 (also as Q8, 48 rows padded to 64), conv1d
             [4, 10240], ssm_norm [128]
  attention  q_proj [q 6144 | gate 6144], k / v 1024, o 5120 x 6144,
             q_norm / k_norm [256]
  MLP        gate / up [17408, 5120], down [5120, 17408]

The multi-token prediction (MTP) head is model.mtp.*: fc [5120, 10240] over
[norm(embedding) | norm(hidden)], one full-attention layer with its MLP, and
a final norm before the shared lm_head.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np

# llms/ is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]

from air_examples.llms.qwen35_0_8b_q4nx import q35_weights  # noqa: E402

REPO = "FastFlowLM/Qwen3.8-27B-NPU2"
REVISION = "071def594ad89b0dadd104a117722d6090a5fa26"  # main
SHA256 = "2f0c797f525d5bbacb9f0cde5ff2ea06a3b498a6fc55f5cccc8c97d7eaf00fa1"

N_LAYERS, D, VOCAB, EPS = 64, 5120, 248320, 1e-6
# Gated DeltaNet
NK, NVH, DK, DV, CONV = 16, 48, 128, 128, 4
QKV = 2 * NK * DK + NVH * DV
# full attention
NQ, NKV, DH, ROPE_DIM, THETA = 24, 4, 256, 64, 1e7
# MLP
I = 17408


def is_attention(layer):
    return layer % 4 == 3


def bundle_path(source=None):
    """`source` (else $Q4NX_MODEL_SOURCE, else $QWEN38_FLM_DIR) when it is
    model.q4nx or a directory holding it, else the Hub download of REPO at
    REVISION. A downloaded file is checked against SHA256."""
    source = (
        source
        or os.environ.get("Q4NX_MODEL_SOURCE")
        or os.environ.get("QWEN38_FLM_DIR")
    )
    if source and os.path.isdir(source):
        source = os.path.join(source, "model.q4nx")
    if source and os.path.isfile(source):
        return source
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "model.q4nx", revision=REVISION)
    got = q35_weights.sha256(path)
    if got != SHA256:
        raise ValueError(f"{path}: sha256 {got}, expected {SHA256}")
    return path


class Qwen38Q4nx(q35_weights.Q4nx):
    """The bundle by layer, as float32 numpy (dequantized) or raw blocks."""

    def __init__(self, path=None):
        import json

        self.path = Path(bundle_path(path))
        with open(self.path, "rb") as f:
            n = int(np.frombuffer(f.read(8), "<u8")[0])
            self.hdr = json.loads(f.read(n))
        self.off = 8 + n
        self.mm = np.memmap(self.path, np.uint8, "r")

    def tokenizer(self):
        """The bundle's tokenizer, from next to it or from REPO."""
        from tokenizers import Tokenizer

        path = self.path.parent / "tokenizer.json"
        if not path.is_file():
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(REPO, "tokenizer.json", revision=REVISION)
        return Tokenizer.from_file(str(path))

    @staticmethod
    def name(layer, part):
        """Layer `layer`'s tensor `part`; layer "mtp" is the MTP head's."""
        if layer == "mtp":
            return f"model.mtp.layers.0.{part}"
        return f"model.layers.{layer}.{part}"

    def mat(self, layer, part, rows, K):
        """A whole projection, dequantized: float32 [rows, K]."""
        return self.dequant(self.name(layer, part), rows, K)

    def vec(self, layer, part):
        return self.bf16(self.name(layer, part))

    def gdn(self, layer):
        """(qkv, z, out, alpha [48, D], beta [48, D], conv [10240, 4], norm,
        a, dt_bias) of a Gated DeltaNet layer."""
        la = "linear_attn."
        return (
            self.mat(layer, la + "qkv_proj.weight", QKV, D),
            self.mat(layer, "self_attn.gate_proj.weight", NVH * DV, D),
            self.mat(layer, la + "ssm_out_proj.weight", D, NVH * DV),
            self.vec(layer, la + "ssm_alpha_proj.bf16.weight"),
            self.vec(layer, la + "ssm_beta_proj.bf16.weight"),
            self.vec(layer, la + "ssm_conv1d.weight").T,
            self.vec(layer, la + "ssm_norm.weight"),
            self.vec(layer, la + "ssm_a"),
            self.vec(layer, la + "ssm_dt.bias"),
        )

    def attn(self, layer):
        """(q [6144, D], gate [6144, D], k, v [1024, D], o [D, 6144], q_norm,
        k_norm) of a full-attention layer (or of the MTP head, layer "mtp")."""
        sa = "self_attn."
        qg = self.mat(layer, sa + "q_proj.weight", 2 * NQ * DH, D)
        return (
            qg[: NQ * DH],
            qg[NQ * DH :],
            self.mat(layer, sa + "k_proj.weight", NKV * DH, D),
            self.mat(layer, sa + "v_proj.weight", NKV * DH, D),
            self.mat(layer, sa + "o_proj.weight", D, NQ * DH),
            self.vec(layer, sa + "q_norm.weight"),
            self.vec(layer, sa + "k_norm.weight"),
        )

    def norms(self, layer):
        return (
            self.vec(layer, "input_layernorm.weight"),
            self.vec(layer, "post_attention_layernorm.weight"),
        )

    def mlp(self, layer):
        """(gate, up [I, D], down [D, I])."""
        m = "mlp."
        return (
            self.mat(layer, m + "gate_proj.weight", I, D),
            self.mat(layer, m + "up_proj.weight", I, D),
            self.mat(layer, m + "down_proj.weight", D, I),
        )

    def mtp_head(self):
        """(fc [D, 2D], embedding norm, hidden norm, final norm) of the MTP
        head; its decoder layer is layer "mtp" for attn / norms / mlp."""
        return (
            self.dequant("model.mtp.fc.weight", D, 2 * D),
            self.bf16("model.mtp.pre_fc_norm_embedding.weight"),
            self.bf16("model.mtp.pre_fc_norm_hidden.weight"),
            self.bf16("model.mtp.norm.weight"),
        )

    def embed(self, ids):
        """Embedding rows of `ids` (the Q8 table, exact), float32
        [len(ids), D]."""
        name = "model.embed_tokens.weight"
        return np.stack(
            [self.dequant_rows(name, int(t), int(t) + 1, D)[0] for t in ids]
        )

    def final_norm(self):
        return self.bf16("model.norm.weight")

    def lm_head_rows(self, r0, r1):
        """lm_head rows r0..r1, float32 [r1 - r0, D] (Q8 values, exact)."""
        return self.dequant_rows("lm_head.weight", r0, r1, D)
