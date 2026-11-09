# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""FastFlowLM's Qwen3.6-35B-A3B `model.q4nx` (FastFlowLM/Qwen3.6-35B-A3B-NPU2,
branch flm_q4k_high_precision), text weights only.

The block formats are Qwen3.5's (q35_weights): every projection, the experts
included, is Q4K in 32 x 256 blocks; lm_head is Q8. Norm weights already hold
the (1 + w) of the model's RMSNorm, and ssm_a holds -exp(A_log).

Layer l is a full-attention layer when l % 4 == 3, else a Gated DeltaNet
layer. Per layer:

  GDN        qkv_proj [q 2048 | k 2048 | v 4096], z (stored as
             self_attn.gate_proj) 4096, ssm_out 2048 x 4096, alpha / beta
             [2048, 32] bf16 (transposed), conv1d [4, 8192], ssm_norm [128]
  attention  q_proj [q 4096 | gate 4096], k / v 512, o 2048 x 4096,
             q_norm / k_norm [256]
  MoE        up / gate experts [256 x 512, 2048], down experts
             [256 x 2048, 512], shared up / gate 512, shared down 2048 x 512,
             router [2048, 256] bf16 (transposed), shared-expert gate [2048]
"""

import functools
import os
import sys
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# llms/ is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]

from air_examples.llms.qwen35_0_8b_q4nx import q35_weights  # noqa: E402

REPO = "FastFlowLM/Qwen3.6-35B-A3B-NPU2"
REVISION = "18adb6adc1338ba29711e58b787df609fc55cd3a"  # branch flm_q4k_high_precision
SHA256 = "a763f82ad34f09af85615c600265fa26be5723176e483ef22cd1ec735f12d175"

N_LAYERS, D, VOCAB, EPS = 40, 2048, 248320, 1e-6
# Gated DeltaNet
NK, NVH, DK, DV, CONV = 16, 32, 128, 128, 4
QKV = 2 * NK * DK + NVH * DV
# full attention
NQ, NKV, DH, ROPE_DIM, THETA = 16, 2, 256, 64, 1e7
# MoE
N_EXPERTS, TOP_K, I_EXP, I_SHARED = 256, 8, 512, 512


def is_attention(layer):
    return layer % 4 == 3


def bundle_path(source=None):
    """`source` (else $Q4NX_MODEL_SOURCE, else $QWEN36_FLM_DIR) when it is
    model.q4nx or a directory holding it, else the Hub download of REPO at
    REVISION. A downloaded file is checked against SHA256."""
    source = (
        source
        or os.environ.get("Q4NX_MODEL_SOURCE")
        or os.environ.get("QWEN36_FLM_DIR")
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


class Qwen36Q4nx(q35_weights.Q4nx):
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
        return f"model.layer.{layer}.{part}"

    def mat(self, layer, part, rows, K):
        """A whole projection, dequantized: float32 [rows, K]."""
        return self.dequant(self.name(layer, part), rows, K)

    def vec(self, layer, part):
        return self.bf16(self.name(layer, part))

    def gdn(self, layer):
        """(qkv, z, out, alpha [32, D], beta [32, D], conv [8192, 4], norm,
        a, dt_bias) of a Gated DeltaNet layer."""
        la = "linear_attn."
        return (
            self.mat(layer, la + "qkv_proj.weight", QKV, D),
            self.mat(layer, "self_attn.gate_proj.weight", NVH * DV, D),
            self.mat(layer, la + "ssm_out_proj.weight", D, NVH * DV),
            self.vec(layer, la + "ssm_alpha_proj.weight").T,
            self.vec(layer, la + "ssm_beta_proj.weight").T,
            self.vec(layer, la + "ssm_conv1d.weight").T,
            self.vec(layer, la + "ssm_norm.weight"),
            self.vec(layer, la + "ssm_a"),
            self.vec(layer, la + "ssm_dt.bias"),
        )

    def attn(self, layer):
        """(q [4096, D], gate [4096, D], k, v [512, D], o [D, 4096], q_norm,
        k_norm) of a full-attention layer."""
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

    def router(self, layer):
        """(router [N_EXPERTS, D], shared-expert gate [D])."""
        return (
            self.vec(layer, "moe_router.weight").T,
            self.vec(layer, "shared_expert_gate.weight"),
        )

    def shared(self, layer):
        """(gate, up [I_SHARED, D], down [D, I_SHARED])."""
        m = "mlp."
        return (
            self.mat(layer, m + "share_gate_exps_proj.weight", I_SHARED, D),
            self.mat(layer, m + "share_up_exps_proj.weight", I_SHARED, D),
            self.mat(layer, m + "share_down_exps_proj.weight", D, I_SHARED),
        )

    @functools.lru_cache(maxsize=4096)
    def expert(self, layer, e):
        """(gate, up [I_EXP, D], down [D, I_EXP]) of expert e."""
        m = "mlp."
        rows = lambda part, n, K: self.dequant_rows(  # noqa: E731
            self.name(layer, m + part), e * n, (e + 1) * n, K
        )
        return (
            rows("gate_exps_proj.weight", I_EXP, D),
            rows("up_exps_proj.weight", I_EXP, D),
            rows("down_exps_proj.weight", D, I_EXP),
        )

    def embed(self, ids):
        """Embedding rows of `ids`, float32 [len(ids), D]."""
        t = np.asarray(self.raw("model.embed_tokens.weight")).view(bfloat16)
        return t.reshape(VOCAB, D)[np.asarray(ids)].astype(np.float32)

    def final_norm(self):
        return self.bf16("model.norm.weight")

    @functools.cached_property
    def lm_head(self):
        """lm_head, float32 [VOCAB, D] (Q8 values, exact)."""
        return self.dequant("lm_head.weight", VOCAB, D)
