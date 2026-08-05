# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Q4NX Llama-3.2-3B example (FLM-faithful decode).

Drives the SAME on-device path `make run` uses: the fused_decode superkernel with
attention + KV cache on the AIE array (no CPU attention, no host KV). The
reference is the HF **bf16** checkpoint, so the shared gate compares NPU-q4nx vs
HF-bf16 top-k token-set inclusion — the closest proxy for "matches FLM's 3B".

`prefill()` mirrors the production driver: prompts of at least PREFILL_MIN_TOKENS
run through the batched NPU prefill (llama32_3b_q4nx_prefill), whose per-layer
roped-K / raw-V seed the decode's device KV cache; shorter prompts are replayed
through the decode (cheaper than a padded prefill). `decode_step()` then dispatches
one fused token. Q4NX_VERIFY_FORCE_PREFILL=1 forces the prefill path for handoff
coverage; Q4NX_VERIFY_NO_PREFILL=1 forces the replay path. `layer_intermediates` / `final_hidden_normed` are not produced --
the fused decode runs all 28 layers in one dispatch, so the per-layer `diagnosis`
lens is unavailable here; the token-set gate is the PASS/FAIL signal.

Pointed at via `--runner=llama32_3b_q4nx.verify_adapter`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
_LLAMA3B = _LLMS_DIR / "llama32_3b"
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_LLAMA3B), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from llama32_3b_weights import LlamaConfig  # noqa: E402
from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402
from q4nx_decode_3b import FusedDecode3B  # noqa: E402
from llama32_3b_q4nx_inference import PREFILL_MIN_TOKENS  # noqa: E402

MODEL_CHOICES = {
    "base": "meta-llama/Llama-3.2-3B",
    "instruct": "meta-llama/Llama-3.2-3B-Instruct",
}
DEFAULT_MODEL = "instruct"

# NPU weight source (model.q4nx bundle). `model_name` selects tokenizer + HF bf16
# reference; the NPU weights always come from here.
Q4NX_MODEL_SOURCE = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Llama-3.2-3B-NPU2")
# Templates are built into this example's directory by `make compile-decode`, the
# same place the production driver loads them from.
DECODE_TEMPLATES = os.environ.get("DECODE_TEMPLATES", str(_THIS_DIR))
# Padded prefill length (the prefill ELFs are compiled for it).
PREFILL_SEQ_LEN = int(os.environ.get("Q4NX_SEQ_LEN", "2048"))
# Same crossover the production driver uses (see llama32_3b_q4nx_inference).
PREFILL_MIN_TOKENS = int(os.environ.get("Q4NX_PREFILL_MIN", "96"))


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """Reference is the HF bf16 checkpoint (NPU q4nx vs HF bf16)."""
    return npu_model_name


def build_config():
    return LlamaConfig()


class FusedDecodeRunner:
    """Runner over the FLM-faithful fused decode (28 layers + LM head per dispatch)."""

    name = "npu-q4nx-3b-fused-decode"

    def __init__(self, model_source, templates, max_seq):
        self._dec = FusedDecode3B(model_source, templates)
        if max_seq > self._dec.ATTN_MAXL:
            raise RuntimeError(
                f"max_seq {max_seq} exceeds decode ATTN_MAXL {self._dec.ATTN_MAXL}"
            )
        # Mirror the production driver: it only uses the batched prefill for prompts
        # at least PREFILL_MIN_TOKENS long (a padded prefill costs more than replaying
        # a short prompt through the decode). Building it unconditionally here would
        # gate a path users never hit for short prompts. Q4NX_VERIFY_FORCE_PREFILL=1
        # forces it on (covers the prefill->decode KV handoff); NO_PREFILL=1 forces off.
        self._force_pf = os.environ.get("Q4NX_VERIFY_FORCE_PREFILL") == "1"
        self._no_pf = os.environ.get("Q4NX_VERIFY_NO_PREFILL") == "1"
        self._model_source = model_source
        self._pf = None  # built lazily: only a long enough prompt needs it

    def _prefiller(self):
        if self._pf is None:
            from llama32_3b_q4nx_prefill import LlamaQ4nxPrefill

            self._pf = LlamaQ4nxPrefill(
                seq_len=int(os.environ.get("Q4NX_SEQ_LEN", "2048")),
                n_layers=self._dec.N_LAYERS,
            )
            self._pf.load_weights(model=self._model_source)
        return self._pf

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        toks = [int(t) for t in np.asarray(prompt_tokens).reshape(-1)]
        if not self._no_pf and (self._force_pf or len(toks) >= PREFILL_MIN_TOKENS):
            pf = self._prefiller()
            pf.clear_context()
            logits = np.asarray(pf.prefill(toks), np.float32)
            kvs = [pf.kv_view(L) for L in range(self._dec.N_LAYERS)]
            self._dec.reset_kv()
            self._dec.seed_kv([k for k, _ in kvs], [v for _, v in kvs])
        else:
            self._dec.reset_kv()
            logits = None
            for p, t in enumerate(toks):
                logits = self._dec.dispatch(t, p)
        return PrefillRecord(
            layer_intermediates=[],  # single-dispatch superkernel: not observable
            final_hidden_normed=np.empty(0, np.float32),
            logits_at_pred=logits,
            top1_token=int(np.argmax(logits)),
        )

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        logits = self._dec.dispatch(int(input_token), int(current_pos))
        return DecodeStepRecord(
            lm_head_logits=logits, top1_token=int(np.argmax(logits))
        )


def build_runner(
    model_name: str,
    config,
    max_seq: int,
    tokenizer,
    *,
    npu_attn: bool = True,
    lite_mode: bool = False,
):
    """Attention is ALWAYS on the NPU here (that is the point of this example)."""
    if not npu_attn:
        raise ValueError(
            "llama32_3b_q4nx runs attention on the NPU by design; npu_attn=False "
            "is not supported (that was the unfaithful CPU-attention path)."
        )
    return FusedDecodeRunner(Q4NX_MODEL_SOURCE, DECODE_TEMPLATES, max_seq)
