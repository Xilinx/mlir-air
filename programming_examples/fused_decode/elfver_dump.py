#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Dump one decode step's BO contents and the xclbin's logits, for elfver_run.py
# to replay through the full-ELF build.
#
#   DEC_P=<prefill positions> python3 elfver_dump.py <outdir>

import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# Resolved from this file, not hardcoded, so the utility runs from any checkout.
_MODEL_DIR = Path(
    os.environ.get(
        "ELFVER_MODEL_DIR",
        Path(__file__).resolve().parent.parent / "llms" / "llama32_1b_q4nx",
    )
)
sys.path.insert(0, str(_MODEL_DIR))
from llama32_1b_q4nx_inference import FusedDecoder  # noqa: E402

TOK = 3681  # the token the Paris prompt lands on; any fixed id works


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/elfver"
    os.makedirs(out, exist_ok=True)
    p = int(os.environ.get("DEC_P", "2047"))

    dec = FusedDecoder()
    L = p + 1

    # A fixed pseudo-random KV, so the attention actually depends on the context
    # length: an all-ones cache would score identically at every position and
    # could not tell a right mask threshold from a wrong one.
    rng = np.random.default_rng(0)
    dec.KV[:] = 0
    RS = dec.ATTN_MAXL * dec.REGION_W
    for g in range(2 * dec.NGRP):
        sl = dec.KV[:, g * RS : g * RS + p * dec.REGION_W]
        sl[:] = (rng.standard_normal(sl.shape) * 0.05).astype(bfloat16)
    dec._kv_dirty = True
    kv = np.ascontiguousarray(dec.KV).reshape(-1).view(np.int16)

    ref = np.asarray(dec.dispatch(TOK, p), np.float32)

    lut = np.empty(64, dtype=bfloat16)
    lut[:32] = dec.rope_cos[p][:32].astype(bfloat16)
    lut[32:] = dec.rope_sin[p][:32].astype(bfloat16)
    rms = np.concatenate([dec.rms_slabs, lut, dec.final_norm]).view(np.int16)
    x = np.asarray(dec.embed[TOK], bfloat16).view(np.int16)

    np.asarray(x, np.int16).tofile(f"{out}/x.bin")
    np.asarray(rms, np.int16).tofile(f"{out}/rms.bin")
    np.asarray(kv, np.int16).tofile(f"{out}/kv.bin")
    ref.tofile(f"{out}/ref.bin")

    wpath = f"{out}/W.bin"
    if not os.path.exists(wpath):
        W = np.load(os.environ["Q4NX_DECODE_WEIGHTS_NPZ"])["W"]
        W = W.view(np.int16) if W.dtype != np.int16 else W
        W.tofile(wpath)

    with open(f"{out}/meta.txt", "w") as f:
        for k, v in [
            ("L", L),
            ("ny", dec.ny),
            ("decode_y", dec.decode_y),
            ("voc_n", dec.UNI_LM * dec.VP),
            ("vocab", dec.VOCAB_SIZE),
            ("k", dec.K),
            ("rms_size", dec._RMS_SIZE),
            ("kv_elems", kv.size),
            ("w_elems", os.path.getsize(wpath) // 2),
            ("region_w", dec.REGION_W),
        ]:
            f.write(f"{k} {int(v)}\n")

    print(
        f"[dump] L={L} argmax={int(np.argmax(ref))} "
        f"top5={np.argsort(ref)[-5:][::-1].tolist()} -> {out}"
    )


if __name__ == "__main__":
    main()
