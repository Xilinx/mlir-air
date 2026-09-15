# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Q4NX Gemma4-E2B example.

The NPU runs FastFlowLM's 4-bit `model.q4nx` bundle; the reference is the bf16
HF checkpoint that bundle was built from. The shared verify gate compares
NPU-q4nx vs HF-bf16 (top-k token-set inclusion), the same NPU-vs-bf16 proxy the
Llama, Qwen and Gemma3 Q4NX adapters use.

REFERENCE CHECKPOINT. `unsloth/gemma-4-E2B-it` mirrors `google/gemma-4-E2B-it`,
which is what FastFlowLM's converter builds the bundle from. Both are ungated at
the time of writing; the unsloth mirror is used for the same reason the Gemma3
adapter uses one, so a future license gate on the google repo cannot redden CI.

Gemma4-E2B is multimodal (`Gemma4ForConditionalGeneration`) and only the text
half is compared -- the NPU example implements only that. `AutoModelForCausalLM`
loads the repo whole, as it does for Gemma3-4B, so the shared `HfRunner` needs no
special case.

Like the other fused Q4NX examples the decode is a superkernel rather than a
per-op pipeline: prefill runs through `Gemma4Q4nxPrefill` and each decode step
through the one-xclbin `FusedDecoder`, so this adapter drives those two
components directly to satisfy the shared `prefill()` / `decode_step()` contract.

Pointed at via `--runner=gemma4_e2b_q4nx.verify_adapter`.
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

_N_LAYERS = 35


@dataclass
class Gemma4Config:
    """The four fields the shared HfRunner reads, from gemma4_e2b_q4nx_weights.

    head_dim and hidden_dim are PER LAYER CLASS on this model (256/512 and
    6144/12288); the sliding-class values are carried here because HfRunner only
    uses n_layers materially -- it slices hidden_states by it. A single number
    would be wrong for any other purpose.
    """

    n_layers: int = _N_LAYERS
    emb_dim: int = 1536
    n_heads: int = 8
    head_dim: int = 256
    n_kv_heads: int = 1
    hidden_dim: int = 6144
    vocab_size: int = 262144


# Gemma4-E2B ships instruction-tuned only in the shape this example targets.
MODEL_CHOICES = {
    "instruct": "unsloth/gemma-4-E2B-it",
}
DEFAULT_MODEL = "instruct"

# NPU weight source: the FastFlowLM Q4NX bundle (NOT the bf16 reference above).
# Same default as the inference driver / Makefile.
Q4NX_MODEL_SOURCE = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma4-E2B-IT-NPU2")

# Padded prefill length. Every prefill ELF is built for it, so it is a rebuild
# rather than a runtime knob (same constraint as the other Q4NX examples).
_SEQ_LEN = int(os.environ.get("Q4NX_SEQ_LEN", "2048"))


def build_config():
    return Gemma4Config()


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
    """Compile/load the Q4NX prefill + the fused Gemma4 decode, return an NpuRunner."""
    return NpuRunner(tokenizer=tokenizer, lite_mode=lite_mode)


class NpuRunner:
    """Adapter over the Q4NX batched prefill + one-xclbin fused Gemma4 decode.

    prefill() runs `Gemma4Q4nxPrefill` (full logits + the per-layer roped-K /
    normed-V seed) and seeds the decoder's KV cache; decode_step() dispatches one
    fused token (35 layers + the PLE branch + the LM head) through `FusedDecoder`.
    Mirrors the loop in `gemma4_e2b_q4nx_inference.generate`.
    """

    name = "npu_q4nx"

    def __init__(self, tokenizer, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self._tokenizer = tokenizer

        from gemma4_e2b_q4nx_prefill import Gemma4Q4nxPrefill
        from gemma4_e2b_q4nx_inference import FusedDecoder

        self.prefiller = Gemma4Q4nxPrefill(
            seq_len=_SEQ_LEN, cache_dir=os.environ.get("Q4NX_CACHE_DIR") or None
        )
        self.prefiller.load_weights(model=Q4NX_MODEL_SOURCE)
        self.dec = FusedDecoder(model=Q4NX_MODEL_SOURCE)
        self.attn_maxl = self.dec.ATTN_MAXL
        self._P = 0

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        ids = [int(t) for t in prompt_tokens]
        if len(ids) > self.attn_maxl:
            raise SystemExit(
                f"prompt is {len(ids)} tokens but the fused Gemma4 decode KV cache "
                f"caps at {self.attn_maxl}. Rebuild with a larger cap "
                f"(`make compile-decode-full RUN_LBUILD=...`)."
            )
        self.prefiller.clear_context()
        logits = np.asarray(self.prefiller.prefill(ids), np.float32)
        first = int(logits.argmax())
        # Per-layer LISTS, not a stacked array: head_dim differs by layer class.
        ks, vs = self.prefiller.kv_stack()
        self._P = ks[0].shape[0]
        self.dec.seed_kv(ks, vs, self._P)

        # The prefill produces the first token and the decode continues from it,
        # so no priming dispatch is needed: the runner's first decode_step is that
        # token at position P.

        # The Q4NX prefill does not expose per-layer ffn_out, and the decode runs
        # all 35 layers inside one dispatch, so the diagnosis lens has no per-layer
        # data. Return one empty dict per layer (len == n_layers) rather than an
        # empty list: the diagnosis path indexes layer_intermediates[li] per layer,
        # and `.get("ffn_out")` on an empty dict yields None, which the shared
        # runner skips gracefully (an empty list would IndexError).
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
