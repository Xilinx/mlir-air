# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Q4_0 LFM2-1.2B example.

The NPU runs Q4_0 weights quantized from the HF checkpoint on load; the
reference is that same checkpoint in bf16. The shared verify gate compares
NPU-q4_0 vs HF-bf16 by top-k token-set inclusion -- the same NPU-vs-bf16 proxy
every other q4/q4nx adapter uses.

LFM2 publishes ONE checkpoint, so unlike the Llama adapters there is no
base/instruct split to resolve: the tokenizer, the NPU weight source and the
bf16 reference are all the same repo.

Drives `Lfm2Q4nxPrefill` and the one-xclbin `FusedDecoder` directly to satisfy
the shared `prefill()` / `decode_step()` Runner contract, mirroring the resident
pipeline in `lfm2_1_2b_q4nx_inference.Session`.

Pointed at via `--runner=lfm2_1_2b_q4nx.verify_adapter`.
"""

from __future__ import annotations

import os
import sys
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

from lfm2_1_2b_q4nx_weights import Lfm2Q4nxConfig  # noqa: E402

# LFM2 ships a single checkpoint; `model_choice` is accepted and ignored so the
# shared runner's --model flag works uniformly across examples.
MODEL_SOURCE = os.environ.get("LFM2_MODEL_SOURCE") or "LiquidAI/LFM2-1.2B"
MODEL_CHOICES = {"base": MODEL_SOURCE, "instruct": MODEL_SOURCE}
DEFAULT_MODEL = "instruct"

_N_LAYERS = 16


def build_config():
    """Config the shared HfRunner reads dims off (n_layers / emb_dim / ...).

    LFM2's conv layers have no KV, but the shared runner only uses these to size
    reference buffers, so reporting the attention dims uniformly is correct.
    """
    return Lfm2Q4nxConfig()


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
    """Reference is the same checkpoint in bf16 (NPU q4_0 vs HF bf16)."""
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
    return NpuRunner(max_seq=max_seq, tokenizer=tokenizer, lite_mode=lite_mode)


def _stair_on():
    return os.environ.get("DECODE_STAIRCASE") == "1"


class NpuRunner:
    """Adapter over the LFM2 Q4_0 prefill + one-xclbin hybrid decode."""

    name = "npu_lfm2_q4_0"

    def __init__(self, max_seq: int, tokenizer, lite_mode: bool = False):
        self.max_seq = max_seq
        self.lite_mode = lite_mode
        self._tokenizer = tokenizer

        from lfm2_1_2b_q4nx_prefill import Lfm2Q4nxPrefill
        from lfm2_1_2b_q4nx_inference import FusedDecoder

        self.prefiller = Lfm2Q4nxPrefill(seq_len=max_seq)
        self.prefiller.compile()
        self.prefiller.load_weights(model=MODEL_SOURCE)
        self.cfg = self.prefiller.config
        self.dec = FusedDecoder(staircase=_stair_on())
        self.attn_maxl = self.dec.ATTN_MAXL
        self._P = 0

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        ids = [int(t) for t in prompt_tokens]
        self.prefiller.clear_context()
        logits = np.asarray(self.prefiller.prefill(ids), np.float32)
        first = int(logits.argmax())

        cfg = self.cfg
        P = self.prefiller.get_current_context_length()
        halo = cfg.conv_L_cache - 1
        # Both prefill-fed regions of arg4, indexed by MODEL layer. A layer is
        # of exactly one kind, so the other array's rows stay zero and are never
        # read -- see the arg4 note in lfm2_1_2b_q4nx_inference.
        K = np.zeros((_N_LAYERS, P, cfg.kv_dim), np.float32)
        V = np.zeros((_N_LAYERS, P, cfg.kv_dim), np.float32)
        S = np.zeros((_N_LAYERS, halo, cfg.conv_dim), np.float32)
        for li in range(_N_LAYERS):
            if cfg.is_attn_layer(li):
                K[li] = np.asarray(self.prefiller.get_k_cache(li), np.float32)
                V[li] = np.asarray(self.prefiller.get_v_cache(li), np.float32)
            else:
                S[li] = np.asarray(self.prefiller.get_conv_state(li), np.float32)
        self._P = P
        self.dec.KVC[:] = 0
        self.dec.seed_state(K, V, S, P)

        # The prefill does not expose per-layer ffn_out, so the diagnosis lens
        # has no data. Return one EMPTY DICT per layer rather than an empty
        # list: that path indexes layer_intermediates[li], and `.get(...)` on an
        # empty dict yields None (which the shared runner skips) where an empty
        # list would IndexError.
        empty = np.empty((0,), dtype=np.float32)
        return PrefillRecord(
            layer_intermediates=[{} for _ in range(_N_LAYERS)],
            final_hidden_normed=empty,
            logits_at_pred=logits,
            top1_token=first,
        )

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        logits = np.asarray(
            self.dec.dispatch(int(input_token), int(current_pos)), np.float32
        )
        return DecodeStepRecord(
            lm_head_logits=logits,
            top1_token=int(logits.argmax()),
        )
