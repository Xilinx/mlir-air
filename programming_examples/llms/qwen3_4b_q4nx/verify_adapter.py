# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Q4NX Qwen3-4B example (FLM-faithful prefill + decode).

Qwen3-4B is the DFlash TARGET model (docs/DFlashFeasibility.md). This mirrors
qwen3_8b_q4nx/verify_adapter.py exactly except for the model source and TIED
lm-head (handled transparently by qwen3_4b_q4nx_weights.Q4nxModel).

Drives the SAME on-device path `make run` uses: the batched AIR prefill for the
prompt, then the fused_decode superkernel (36 layers + the tied LM head in one
dispatch, attention + KV cache on the AIE array) for each new token.

REFERENCE. Qwen3 ships a single ungated checkpoint (no separate base/instruct
split), so both MODEL_CHOICES keys map to it, and the gate compares NPU-q4nx
against that HF bf16 forward.

`prefill()` runs `Qwen3Q4nxPrefill` (full logits + per-layer roped-K / raw-V) and
seeds the decode's device KV cache from it; each `decode_step()` then dispatches
one fused token.

Pointed at via `--runner=qwen3_4b_q4nx.verify_adapter`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

# Qwen3-4B geometry. Taken from qwen3_4b_q4nx_weights rather than restated, so
# it cannot drift from the loader (and so a clone of this file for another model
# fails loudly instead of silently describing the wrong one). The shared runner
# reads only n_layers today; the rest are here for diagnosis/probing.
import qwen3_4b_q4nx_weights as _w  # noqa: E402

_N_LAYERS = _w.NUM_LAYERS


@dataclass
class Qwen3Config:
    n_layers: int = _N_LAYERS
    emb_dim: int = _w.D
    n_heads: int = _w.N_Q_HEADS
    head_dim: int = _w.DH
    n_kv_heads: int = _w.N_KV_HEADS
    hidden_dim: int = _w.INTER
    vocab_size: int = _w.VOCAB


# Qwen3-4B ships instruction-tuned only; there is no -pt variant in scope here.
MODEL_CHOICES = {
    "base": "Qwen/Qwen3-4B",
    "instruct": "Qwen/Qwen3-4B",
}
DEFAULT_MODEL = "instruct"

# NPU weight source: the FastFlowLM Q4NX bundle (NOT the bf16 reference above).
# Same default as the inference driver / Makefile.
Q4NX_MODEL_SOURCE = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Qwen3-4B-NPU2")

# Padded prefill length. The Qwen3-4B GEMM registry shapes exist at M=2048, so the
# whole prefill runs there (same constraint as the other Q4NX examples).
_SEQ_LEN = int(os.environ.get("Q4NX_SEQ_LEN", "2048"))


def build_config():
    return Qwen3Config()


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """The bf16 reference. Unlike Qwen (which quantizes the reference itself),
    the NPU weights come from a separate Q4NX bundle, so the two differ."""
    return npu_model_name


def build_runner(
    model_name: str,
    config,
    max_seq: int,
    tokenizer,
    *,
    npu_attn: bool = True,
    lite_mode: bool = False,
):
    """Compile/load the Q4NX prefill + the fused Qwen3-4B decode, return an NpuRunner."""
    return NpuRunner(tokenizer=tokenizer, lite_mode=lite_mode)


class NpuRunner:
    """Adapter over the Q4NX batched prefill + one-xclbin fused Qwen3-4B decode.

    prefill() runs `Qwen3Q4nxPrefill` (full logits + per-layer roped-K / raw-V)
    and seeds the decoder's KV cache; decode_step() dispatches one fused token
    (36 layers + the tied LM head) through `FusedDecoder`. Mirrors the loop in
    `qwen3_4b_q4nx_inference.generate`."""

    name = "npu_q4nx"

    def __init__(self, tokenizer, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self._tokenizer = tokenizer

        from qwen3_4b_q4nx_prefill import Qwen3Q4nxPrefill
        from qwen3_4b_q4nx_inference import FusedDecoder

        self.prefiller = Qwen3Q4nxPrefill(
            seq_len=_SEQ_LEN, cache_dir=os.environ.get("Q4NX_CACHE_DIR") or None
        )
        self.prefiller.load_weights(model=Q4NX_MODEL_SOURCE)
        self.dec = FusedDecoder(model=Q4NX_MODEL_SOURCE)
        self.attn_maxl = self.dec.gen.attn_maxl
        self._P = 0

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        ids = [int(t) for t in prompt_tokens]
        if len(ids) > self.attn_maxl:
            raise SystemExit(
                f"prompt is {len(ids)} tokens but the fused Qwen3-4B decode KV cache "
                f"caps at {self.attn_maxl}. Rebuild with a larger cap "
                f"(`make compile-decode LBUILD=...`)."
            )
        self.prefiller.clear_context()
        logits = np.asarray(self.prefiller.prefill(ids), np.float32)
        first = int(logits.argmax())
        Kc, Vc = self.prefiller.kv_stack()
        self._P = Kc.shape[1]
        self.dec.seed_kv(Kc, Vc, self._P)

        # The prefill produces the first token; the decode continues from it, so
        # (unlike the Qwen hand-off) no priming dispatch is needed here -- the
        # runner's first decode_step is that token at position P.

        # The Q4NX prefill does not expose per-layer ffn_out, and the decode runs
        # all 36 layers inside one dispatch, so the diagnosis lens has no
        # per-layer data. Return one empty dict per layer (len == n_layers) rather
        # than an empty list: the diagnosis path indexes layer_intermediates[li]
        # per layer, and `.get("ffn_out")` on an empty dict yields None, which the
        # shared runner skips gracefully (an empty list would IndexError).
        empty = np.empty((0,), dtype=np.float32)
        return PrefillRecord(
            layer_intermediates=[{} for _ in range(_N_LAYERS)],
            final_hidden_normed=empty,
            logits_at_pred=logits,
            top1_token=first,
        )

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        # `current_pos` is the absolute position of `input_token` (the runner
        # passes len(prompt) for the first generated token), which is what
        # dispatch() wants.
        logits = np.asarray(
            self.dec.dispatch(int(input_token), int(current_pos)), np.float32
        )
        return DecodeStepRecord(
            lm_head_logits=logits,
            top1_token=int(logits.argmax()),
        )
