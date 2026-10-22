# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reorder existing 8B Q4NX blocks without dequantization or requantization.

Writes mmap-friendly arrays. Every packed phase is inverted and compared
against every original input word before it is committed to the cache.
The untied vocabulary head comes from lm_head.weight, never the embedding.
"""

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16
from paths import HERE
from model_paths import TARGET_MODEL, resolve_target_model

from target_weights import Q4nxModel
import model_spec as spec


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cascade(raw):
    """[output/32,K/256,2560] -> [col,half,wave,Kchunk,core,2560]."""
    n, k, w = raw.shape
    assert n % 16 == 0 and w == 2560
    return np.ascontiguousarray(
        raw.reshape(n // 16, 4, 2, 2, k, w).transpose(1, 2, 0, 4, 3, 5)
    ).reshape(-1)


def inverse(packed, n, k):
    return np.ascontiguousarray(
        packed.reshape(4, 2, n // 16, k, 2, 2560).transpose(2, 0, 1, 4, 3, 5)
    ).reshape(n, k, 2560)


def layer_blocks(model, layer):
    blocks = {}
    for name, (suffix, m, k) in model._PROJ.items():
        blocks[name] = (
            model._raw_i16(f"model.layers.{layer}.{suffix}.weight")
            .view(np.uint16)
            .reshape(m // 32, k // 256, 2560)
        )
    up, gate = blocks["up"], blocks["gate"]
    interleaved = np.stack(
        [
            up.reshape(spec.TARGET_FFN // 512, 16, 16, 2560),
            gate.reshape(spec.TARGET_FFN // 512, 16, 16, 2560),
        ],
        axis=1,
    ).reshape(2 * spec.TARGET_FFN // 32, 16, 2560)
    return [
        np.concatenate([blocks[n] for n in ("q", "k", "v")]),
        blocks["o"],
        interleaved,
        blocks["down"],
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=TARGET_MODEL)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=spec.TARGET_LAYERS)
    args = ap.parse_args()
    assert 1 <= args.layers <= spec.TARGET_LAYERS
    args.model = Path(resolve_target_model(args.model))
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(
        status="preparing",
        model=str(args.model.resolve()),
        layers=args.layers,
        source_sha256=sha(__file__),
        quantization="existing Q4NX bytes preserved",
        head="independent lm_head.weight",
        phases=[],
        files={},
    )

    def save():
        (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        dims = [
            (6144, 4096),
            (4096, 4096),
            (2 * spec.TARGET_FFN, 4096),
            (4096, spec.TARGET_FFN),
        ]
        layer_words = sum(m * k // 4 * 5 // 4 for m, k in dims)  # .625 bytes/weight /2
        head_words = spec.VOCAB_PADDED * 4096 * 5 // 16
        w = np.lib.format.open_memmap(
            args.output / "W.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(args.layers * layer_words + head_words,),
        )
        model = Q4nxModel(str(args.model))
        norms = []
        offset = 0
        for layer in range(args.layers):
            phases = layer_blocks(model, layer)
            for phase, raw in enumerate(phases):
                packed = cascade(raw)
                assert np.array_equal(inverse(packed, *raw.shape[:2]), raw)
                w[offset : offset + packed.size] = packed
                record["phases"].append(
                    dict(
                        layer=layer,
                        phase=phase,
                        offset_words=offset,
                        words=int(packed.size),
                        inverse_bit_exact=True,
                        source_words_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    )
                )
                offset += packed.size
            norms.append(model.layer_rms(layer))
            print(f"Packed target layer {layer+1}/{args.layers}", flush=True)
            save()
            # Flush and release completed slabs: keep preparation memory bounded
            # instead of accumulating both source and destination mapped pages.
            del phases, raw, packed
            w.flush()
            w._mmap.madvise(mmap.MADV_DONTNEED)
            model._mm._mmap.madvise(mmap.MADV_DONTNEED)
            with (args.output / "W.npy").open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, offset * 2, os.POSIX_FADV_DONTNEED)
        raw = (
            model._raw_i16("lm_head.weight")
            .view(np.uint16)
            .reshape(spec.VOCAB_SIZE // 32, 16, 2560)
        )
        padded = np.zeros((spec.VOCAB_PADDED // 32, 16, 2560), np.uint16)
        padded[: len(raw)] = raw
        for wave in range(spec.VOCAB_PADDED // 8192):
            source = padded[wave * 256 : (wave + 1) * 256]
            packed = cascade(source)
            assert np.array_equal(inverse(packed, *source.shape[:2]), source)
            w[offset : offset + packed.size] = packed
            offset += packed.size
        assert offset == w.size
        w.flush()
        np.save(args.output / "RMS.npy", np.asarray(norms, bfloat16).view(np.uint16))
        record.update(
            status="prepared",
            all_phase_inverse_bit_exact=True,
            head_inverse_bit_exact=True,
            source_model_sha256=sha(args.model),
            layer_words=layer_words,
            head_words=head_words,
            total_weight_bytes=int(w.nbytes),
            files={p.name: sha(p) for p in args.output.glob("*.npy")},
        )
    except BaseException as e:
        record.update(status="failed", error=repr(e))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
