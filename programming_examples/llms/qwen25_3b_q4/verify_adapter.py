# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Q4_0 Qwen2.5-3B example.

The NPU runs Q4_0 weights quantized straight from the HF bf16 checkpoint (no
pre-quantized Qwen bundle exists); the reference is that same HF **bf16**
checkpoint. The shared verify gate therefore compares NPU-q4 vs HF-bf16 (top-k
token-set inclusion), the same NPU-vs-bf16 proxy the Llama Q4NX adapters use.

Like `llama32_1b_q4nx`, the decode here is a fused superkernel rather than a
per-op pipeline: prefill runs through `Qwen25Q4Prefill` and each decode step
through the one-xclbin `QwenFusedDecoder` (extracted from
`fused_decode/qwen_prefill_to_decode.py`), so this adapter drives those two
components directly to satisfy the shared `prefill()` / `decode_step()` contract.

PROMPT LENGTH. The fused Qwen decode grows a KV cache inside a build-time cap
(`LBUILD`, 32 by default): the device attends over the real context and appends
at the real position, so any prompt that fits the cap works and there is no
minimum. This example still ships its own `verify_prompts.txt` (it predates the
growing cache, when every line had to be at least `ATTN_L` tokens) so the gate's
prompt set stays comparable across runs.

Pointed at via `--runner=qwen25_3b_q4.verify_adapter`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent
_VERIFY = _LLMS_DIR / "verify"
_QWEN3B = _LLMS_DIR / "qwen25_3b"  # shared prefill builders + config
_FUSED = _LLMS_DIR.parent / "fused_decode"  # QwenFusedDecoder
for _p in (str(_LLMS_DIR), str(_VERIFY), str(_QWEN3B), str(_FUSED), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

# The Q4_0 example is a requantization of the bf16 qwen25_3b architecture (36
# layers, emb=2048, 16Q/2KV, head_dim=128); reuse its config so the shared
# HfRunner sees the dims it needs.
from qwen25_3b_weights import LlamaConfig  # noqa: E402

MODEL_CHOICES = {
    "base": "Qwen/Qwen2.5-3B",
    "instruct": "Qwen/Qwen2.5-3B-Instruct",
}
DEFAULT_MODEL = "instruct"

# Weight source for the NPU side. Unlike the Llama Q4NX examples this is not a
# separate pre-quantized bundle: it is the same HF checkpoint, Q4_0-quantized on
# the host by qwen25_3b_q4_weights (bit-identical to the codec the fused decode
# packs with, which is what makes the KV hand-off exact).
QWEN_Q4_MODEL_SOURCE = os.environ.get("QWEN_Q4_MODEL_SOURCE", "")

_N_LAYERS = 36
# Padded prefill length. The qwen25_3b GEMM registry shapes exist at M=2048, so
# the whole prefill runs there (same constraint as the Llama Q4NX example).
_SEQ_LEN = int(os.environ.get("QWEN_Q4_SEQ_LEN", "2048"))

# fused_decode_qwen reads its geometry from the environment AT IMPORT, and
# QWEN_NLAYERS defaults to 1 (a fast lowering check, not a real model). Left
# unset, the decoder would pack a 1-layer weight stream for the 36-layer xclbin
# `make compile-decode` builds -- which is silent garbage, not an error. The
# Makefile targets export this; set it here too so a bare `verify_runner
# --runner=qwen25_3b_q4.verify_adapter` cannot hit that trap.
os.environ.setdefault("QWEN_NLAYERS", str(_N_LAYERS))


def build_config():
    return LlamaConfig(n_layers=_N_LAYERS)


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """Reference is the HF bf16 checkpoint (NPU q4 vs HF bf16)."""
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
    """Compile/load the Q4_0 prefill + the fused Qwen decode, return an NpuRunner."""
    return NpuRunner(model_name=model_name, tokenizer=tokenizer, lite_mode=lite_mode)


class NpuRunner:
    """Adapter over the Q4_0 prefill + one-xclbin fused Qwen decode.

    prefill() runs `Qwen25Q4Prefill` (full logits + per-layer roped-K / biased-V)
    and seeds the decoder's sliding KV window; decode_step() dispatches one fused
    token through `QwenFusedDecoder`. Mirrors the loop in
    `fused_decode/qwen_prefill_to_decode.py:main`."""

    name = "npu_q4"

    def __init__(self, model_name: str, tokenizer, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self._tokenizer = tokenizer

        from qwen25_3b_q4_prefill import Qwen25Q4Prefill
        import fused_decode_qwen as _fd
        from qwen_prefill_to_decode import QwenFusedDecoder

        # The prefill and the decode must agree on layer count or the KV
        # hand-off is meaningless. Catch it here with a message that names the
        # variable, rather than deep inside seed_kv's shape assert.
        if _fd.NLAYERS != _N_LAYERS:
            raise SystemExit(
                f"fused decode was configured for {_fd.NLAYERS} layers but this "
                f"adapter prefills {_N_LAYERS}. Export QWEN_NLAYERS={_N_LAYERS} "
                f"(what the Makefile targets do) and make sure the xclbin was "
                f"built with the same value."
            )

        self.prefiller = Qwen25Q4Prefill(seq_len=_SEQ_LEN, n_layers=_N_LAYERS)
        self.prefiller.load_weights(model=QWEN_Q4_MODEL_SOURCE or model_name)
        # `make compile-decode` builds the decode_L<N> template pair into THIS
        # example's directory; anchor it here rather than depending on where the
        # runner was launched from.
        self.dec = QwenFusedDecoder(templates=str(_THIS_DIR))
        self.attn_maxl = self.dec.ATTN_MAXL
        self._P = 0

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        ids = [int(t) for t in prompt_tokens]
        if len(ids) > self.attn_maxl:
            raise SystemExit(
                f"prompt is {len(ids)} tokens but the fused Qwen decode KV cache "
                f"caps at {self.attn_maxl}. Rebuild with a larger cap "
                f"(`make compile-decode LBUILD=...`)."
            )
        self.dec.reset_kv()
        self.prefiller.clear_context()
        logits = np.asarray(self.prefiller.prefill(ids), np.float32)
        first = int(logits.argmax())
        K = np.stack(
            [
                np.asarray(self.prefiller.kv_view(l)[0], np.float32)
                for l in range(_N_LAYERS)
            ]
        )
        V = np.stack(
            [
                np.asarray(self.prefiller.kv_view(l)[1], np.float32)
                for l in range(_N_LAYERS)
            ]
        )
        self._P = K.shape[1]
        self.dec.seed_kv(K, V, self._P)

        # Re-run the last prompt token at its own absolute position (what the CLI
        # loop's first iteration does): the device rewrites that slot with the
        # same content, so after this decode_step(token, P) lines up with the
        # runner's `cur`.
        prime = int(np.asarray(self.dec.dispatch(ids[-1], self._P - 1)).argmax())
        if prime != first:
            # The priming step recomputes the prefill's own last position, so the
            # two should agree -- that is the documented cross-check that the KV
            # hand-off is exact. Both sides now see the same context for any
            # prompt that fits the cap, so this comparison is always valid.
            # Not fatal either way; the verify gate itself is the real check.
            print(
                f"[verify] WARN: fused decode re-ran the last prompt token and "
                f"chose {prime}, but the prefill chose {first}; the "
                f"prefill->decode KV hand-off may be inexact.",
                file=sys.stderr,
            )

        # The Q4_0 prefill doesn't expose per-layer ffn_out, so the diagnosis lens
        # has no data. Return one empty dict per layer (len == n_layers) rather
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
        # passes len(prompt) for the first generated token), which is exactly
        # what dispatch() wants. prefill() primed the window for it.
        logits = np.asarray(
            self.dec.dispatch(int(input_token), int(current_pos)), np.float32
        )
        return DecodeStepRecord(
            lm_head_logits=logits,
            top1_token=int(logits.argmax()),
        )
