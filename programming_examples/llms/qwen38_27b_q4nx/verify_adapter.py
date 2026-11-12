# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify adapter for the Qwen3.8-27B example.

Drives the path `make run` uses: the prompt prefilled on the fused prefill
device (FUSED_PREFILL_DIR, which the Makefile exports) and handed to the
decode, whose LM head gives the first token; then the 64 layers in one
dispatch and the LM head in a second, per token. A prompt past the fused
build's length goes through the decode a token at a time. The NPU runs
FastFlowLM's Q4NX bundle; the gate compares it against the bf16 checkpoint
below.

Pointed at via `--runner=qwen38_27b_q4nx.verify_adapter`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_THIS_DIR = Path(__file__).resolve().parent
_VERIFY = _THIS_DIR.parent / "verify"
for _p in (str(_VERIFY), str(_THIS_DIR)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from runners._records import DecodeStepRecord, PrefillRecord  # noqa: E402

import qwen38_27b_q4nx_weights as _w  # noqa: E402

# the checkpoint FastFlowLM's bundle is quantized from
HF_REPO = "Qwen/Qwen3.8-27B"
HF_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"


@dataclass
class Qwen38Config:
    n_layers: int = _w.N_LAYERS
    emb_dim: int = _w.D
    n_heads: int = _w.NQ
    head_dim: int = _w.DH
    n_kv_heads: int = _w.NKV
    vocab_size: int = _w.VOCAB


# one instruction-tuned checkpoint
MODEL_CHOICES = {"base": HF_REPO, "instruct": HF_REPO}
DEFAULT_MODEL = "instruct"
# the decode's KV cache: the gate's prompts and their new tokens
_MAXL = int(os.environ.get("QWEN38_VERIFY_MAXL", "256"))


def build_config():
    return Qwen38Config()


def _snapshot(full):
    from huggingface_hub import snapshot_download

    # the tokenizer phase takes the small files only
    pats = None if full else ["*.json", "*.jinja", "*.txt"]
    return snapshot_download(HF_REPO, revision=HF_REVISION, allow_patterns=pats)


def resolve_model(model_choice_or_id: str) -> str:
    """The checkpoint at its pinned revision, as a local snapshot, so the
    tokenizer loads that one."""
    name = MODEL_CHOICES.get(model_choice_or_id, model_choice_or_id)
    return _snapshot(False) if name == HF_REPO else name


def hf_reference(npu_model_name: str) -> str:
    """The bf16 reference: the same snapshot with its weights."""
    return _snapshot(True)


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
    name = "npu_q4nx"

    def __init__(self):
        import filelock

        import decode

        self._w = _w.Qwen38Q4nx()
        self._lock = filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock"))
        self._lock.acquire()
        build = Path(os.environ.get("QWEN38_VERIFY_DIR", "build_verify")).resolve()
        build.mkdir(parents=True, exist_ok=True)
        here = os.getcwd()
        os.chdir(build)
        try:
            self.dec = decode.load(self._w, _MAXL)
        finally:
            os.chdir(here)
        self.pf, self.pf_len = None, 0
        fused = os.environ.get("FUSED_PREFILL_DIR")
        if fused:
            import prefill

            man = json.loads((Path(fused) / "manifest.json").read_text())
            self.pf = prefill.Qwen38Prefill(fused, self._w)
            self.pf_len = man["max_len"]

    def __del__(self):
        lock = getattr(self, "_lock", None)
        if lock is not None and lock.is_locked:
            lock.release()

    def _check(self, pos):
        # position pos writes cache row pos
        if pos >= _MAXL:
            raise SystemExit(
                f"position {pos} is past the decode's {_MAXL}-position cache"
                " (QWEN38_VERIFY_MAXL)"
            )

    def _step(self, tok, pos):
        self._check(pos)
        x = np.asarray(self._w.embed([int(tok)])[0]).astype(bfloat16)
        return np.asarray(self.dec.step(x, pos + 1), np.float32)

    def prefill(self, prompt_tokens: np.ndarray) -> PrefillRecord:
        ids = [int(t) for t in prompt_tokens]
        if self.pf is not None and len(ids) <= self.pf_len:
            self._check(len(ids) - 1)
            x = self.pf.prefill(ids)
            self.dec.seed(self.pf)
            logits = np.asarray(
                self.dec.head(x[-1].astype(bfloat16), len(ids)), np.float32
            )
        else:
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
