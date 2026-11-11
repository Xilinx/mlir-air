# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Qwen3-Coder-30B-A3B example.

Drives the decode `make run` uses: the 48 layers in one dispatch and the LM
head in a second, per token. The prompt goes through the decode a token at a
time. The NPU runs this example's Q4_K bundle (quantized from the checkpoint
below); the gate compares it against that checkpoint's bf16 forward.

Pointed at via `--runner=qwen3_coder_30b_a3b_q4k.verify_adapter`.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_VERIFY = _THIS_DIR.parent / "verify"
for _p in (str(_VERIFY), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

import qwen3_coder_30b_a3b_q4k_weights as _w  # noqa: E402


@dataclass
class Qwen3CoderConfig:
    n_layers: int = _w.N_LAYERS
    emb_dim: int = _w.D
    n_heads: int = _w.NQ
    head_dim: int = _w.DH
    n_kv_heads: int = _w.NKV
    vocab_size: int = _w.VOCAB


# one instruction-tuned checkpoint
MODEL_CHOICES = {"base": _w.REPO, "instruct": _w.REPO}
DEFAULT_MODEL = "instruct"
# the decode's KV cache: the gate's prompts and their new tokens
_MAXL = int(os.environ.get("QWEN3C_VERIFY_MAXL", "256"))


def build_config():
    return Qwen3CoderConfig()


def resolve_model(model_choice_or_id: str) -> str:
    return MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)


def hf_reference(npu_model_name: str) -> str:
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
    return NpuRunner()


class NpuRunner:
    name = "npu_q4k"

    def __init__(self):
        import filelock

        import decode

        self._w = _w.Qwen3CoderQ4k()
        self._lock = filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock"))
        self._lock.acquire()
        build = Path(os.environ.get("QWEN3C_VERIFY_DIR", "build_verify")).resolve()
        build.mkdir(parents=True, exist_ok=True)
        here = os.getcwd()
        os.chdir(build)
        try:
            self.dec = decode.load(self._w, _MAXL)
        finally:
            os.chdir(here)

    def __del__(self):
        lock = getattr(self, "_lock", None)
        if lock is not None and lock.is_locked:
            lock.release()

    def _step(self, tok, pos):
        x = np.asarray(self._w.embed([int(tok)])[0]).astype(np.float32)
        from ml_dtypes import bfloat16

        return np.asarray(self.dec.step(x.astype(bfloat16), pos + 1), np.float32)

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        ids = [int(t) for t in prompt_tokens]
        if len(ids) >= _MAXL:
            raise SystemExit(
                f"prompt is {len(ids)} tokens; the decode holds {_MAXL}"
                " (QWEN3C_VERIFY_MAXL)"
            )
        for pos, t in enumerate(ids):
            logits = self._step(t, pos)
        return PrefillRecord(
            layer_intermediates=[{} for _ in range(_w.N_LAYERS)],
            final_hidden_normed=np.empty((0,), np.float32),
            logits_at_pred=logits,
            top1_token=int(logits.argmax()),
        )

    def decode_step(self, input_token: int, current_pos: int) -> DecodeStepRecord:
        logits = self._step(input_token, current_pos)
        return DecodeStepRecord(lm_head_logits=logits, top1_token=int(logits.argmax()))
