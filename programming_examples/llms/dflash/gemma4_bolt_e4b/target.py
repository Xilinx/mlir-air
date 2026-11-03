# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""gemma4_e4b_q4nx's fused prefill as this example's prompt prefill.

forward(ids, base) runs the tokens at positions [base, base + len(ids)) over
the KV already written for [0, base), and returns their logits and the
hidden states after the drafter's tap layers. With keep_kv set it also keeps
a host copy of the K/V, which kv_stack() returns to seed the decoders.
"""

import sys
import types
from pathlib import Path

import numpy as np

LLMS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLMS / "gemma4_e4b_q4nx"))
import gemma4_e4b_variant  # noqa: E402,F401  (sets GEMMA4_VARIANT=e4b)

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(LLMS.parent)
]

import gemma4_e2b_q4nx_weights as gw  # noqa: E402
from air_examples.llms.gemma4_e2b_q4nx.fused_prefill.runtime import (  # noqa: E402
    FusedPrefill,
)

# KV block of the sliding layers; the full layers' (16) divides it
ALIGN = 32


class Target(FusedPrefill):
    keep_kv = False

    def load(self, model):
        self.load_weights(model)
        self.x = np.zeros((self.max_len, gw.D), np.float32)
        self.pli = np.zeros((self.max_len, gw.NUM_LAYERS, gw.PLI_D), np.float32)

    def _kv_append(self, L, r0, k, v):
        if self.keep_kv:
            super()._kv_append(L, r0, k, v)
        else:
            self.kv_append(self.kvb[L], r0, k, v)

    def embed(self, ids):
        return self.model.embed_rows("model.embed_tokens.weight", ids) * gw.EMBED_SCALE

    def lm_logits(self, h):
        """Logits [t, VOCAB] of final-norm hidden states h [t, D], no softcap."""
        h = np.ascontiguousarray(h, np.float32)
        return np.concatenate(
            [self.mm(f"lm.{i}", h.copy()) for i in range(len(self.lm))], axis=1
        )

    def forward(self, ids, base, taps, n_logits):
        """Logits of the last n_logits tokens and, per token, the hidden states
        after layers `taps` concatenated [len(ids), len(taps) * D]."""
        assert base % ALIGN == 0, base
        self.resume()
        ids = [int(t) for t in ids]
        n = len(ids)
        end = base + n
        assert end <= self.max_len, (end, self.max_len)
        qm = self.model
        x = self.x[:end]
        x[base:] = self.embed(ids)
        tbl = qm.embed_rows("model.per_layer_token_embd.weight", ids)
        tbl = tbl.reshape(n, gw.NUM_LAYERS, gw.PLI_D) * gw.PLE_EMBED_SCALE
        chunks = [(base + r0, t) for r0, t in self.chunk_rows(n)]
        pend = []
        for c, (r0, t) in enumerate(chunks):
            self._tile(x[r0 : r0 + t], gw.D, c)
            pend.append(self.gemm_start("mp", c))
        pli = self.pli[:end]
        for c, (r0, t) in enumerate(chunks):
            proj = self._f32(pend[c], t).reshape(t, gw.NUM_LAYERS, gw.PLI_D)
            proj = proj * gw.PLE_MODEL_PROJ_SCALE
            pli[r0 : r0 + t] = (
                gw._rmsnorm(proj, self.glob["ple_proj_norm"])
                + tbl[r0 - base : r0 - base + t]
            ) * gw.PLE_INPUT_SCALE
        ropes = [
            {gw.head_dim(L): self._rope(L, r0, t) for L in self.rope_layers}
            for r0, t in chunks
        ]
        feats = []
        for L in range(gw.NUM_LAYERS):
            self._layer_all(L, x, pli, chunks, ropes)
            if L in taps:
                feats.append(x[base:].copy())
        self.current_context_length = end
        last = gw._rmsnorm(x[end - n_logits :], self.glob["final_norm"])
        return self.lm_logits(last), np.concatenate(feats, axis=1)
