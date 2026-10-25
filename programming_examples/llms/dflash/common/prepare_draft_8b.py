# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Quantize official drafter body to the existing group32 Q4NX B8 path.

The target's untied packed LM head is copied exactly. FC/context projections
remain official BF16 inputs for the separate prepass. Packing is checked by
recovering every quantized nibble, scale and minimum; no model accuracy claim
is inferred from these storage checks.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16
from draft_weights import DraftWeights, SNAPSHOT
from common.prepare_target_8b import cascade, inverse, sha
import model_spec as spec
from paths import HERE
from common.q4nx_blocks import pack_blocks


def quantized_blocks(weight):
    weight = np.asarray(weight, np.float32)
    m, k = weight.shape
    assert m % 32 == 0 and k % 256 == 0
    # Same group32 min/max quantization as the selected 4B draft path.
    groups = weight.reshape(m, k // 32, 32)
    minimum = groups.min(-1)
    scale = (groups.max(-1) - minimum) / 15
    scale = np.where(scale <= 0, 1, scale).astype(np.float32)
    q = np.clip(
        np.round((groups - minimum[..., None]) / scale[..., None]), 0, 15
    ).astype(np.uint8)
    q = q.reshape(m // 32, 32, k // 256, 256).transpose(0, 2, 1, 3)
    sc = scale.reshape(m // 32, 32, k // 256, 8).transpose(0, 2, 3, 1).astype(bfloat16)
    mn = (
        minimum.reshape(m // 32, 32, k // 256, 8).transpose(0, 2, 3, 1).astype(bfloat16)
    )
    raw = pack_blocks(q, sc, mn)
    # Decode with the independent selected reader, covering every packed word.
    from common.q4nx_blocks import unpack_blocks

    uq, us, um = unpack_blocks(raw)
    assert np.array_equal(uq.reshape(q.shape), q)
    assert np.array_equal(us.reshape(*q.shape[:-2], 32, 8).swapaxes(-1, -2), sc)
    assert np.array_equal(um.reshape(*q.shape[:-2], 32, 8).swapaxes(-1, -2), mn)
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--target-cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    weights = DraftWeights(snapshot=a.snapshot)
    target = json.loads((a.target_cache / "manifest.json").read_text())
    assert target["status"] == "prepared" and target["layers"] == spec.TARGET_LAYERS
    assert sha(a.target_cache / "W.npy") == target["files"]["W.npy"]
    layer_words = (6144 * 4096 + 4096 * 4096 + 3 * spec.DRAFT_FFN * 4096) * 5 // 16
    head_words = target["head_words"]
    record = dict(
        status="preparing",
        layers=5,
        role="drafter",
        model=spec.DRAFT_ID,
        snapshot=str(a.snapshot.resolve()),
        source_sha256=sha(Path(__file__)),
        source_model_sha256=sha(weights.path),
        target_cache_manifest_sha256=sha(a.target_cache / "manifest.json"),
        quantization="group32 min/max Q4NX body from official BF16",
        head="exact target untied Q4NX LM head",
        phases=[],
    )
    if hasattr(spec, "DRAFT_SHA256"):
        assert record["source_model_sha256"] == spec.DRAFT_SHA256

    def save():
        (a.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        packed = np.lib.format.open_memmap(
            a.output / "W.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(5 * layer_words + head_words,),
        )
        offset = 0
        norms = []
        for layer in range(5):

            def matrix(stem):
                return quantized_blocks(weights.tensor(f"layers.{layer}.{stem}.weight"))

            qkv = np.concatenate(
                [matrix("self_attn." + name + "_proj") for name in ("q", "k", "v")]
            )
            o = matrix("self_attn.o_proj")
            up, gate = matrix("mlp.up_proj"), matrix("mlp.gate_proj")
            glu = np.stack(
                [up.reshape(24, 16, 16, 2560), gate.reshape(24, 16, 16, 2560)], axis=1
            ).reshape(768, 16, 2560)
            down = matrix("mlp.down_proj")
            begin = offset
            for phase, raw in enumerate((qkv, o, glu, down)):
                words = cascade(raw)
                assert np.array_equal(inverse(words, *raw.shape[:2]), raw)
                packed[offset : offset + words.size] = words
                record["phases"].append(
                    dict(
                        layer=layer,
                        phase=phase,
                        words=int(words.size),
                        inverse_exact=True,
                    )
                )
                offset += words.size
            assert offset - begin == layer_words
            norms.append(weights.layer_rms(layer))
            print(f"Packed drafter layer {layer+1}/5", flush=True)
            save()
        head = np.load(a.target_cache / "W.npy", mmap_mode="r")[-head_words:]
        packed[offset:] = head
        assert np.array_equal(packed[offset:], head)
        packed.flush()
        np.save(a.output / "RMS.npy", np.asarray(norms, bfloat16).view(np.uint16))
        record.update(
            status="prepared",
            all_nibbles_scales_mins_exact=True,
            all_phase_inverse_bit_exact=True,
            head_bit_exact=True,
            layer_words=layer_words,
            head_words=head_words,
            total_weight_bytes=int(packed.nbytes),
            files={p.name: sha(p) for p in a.output.glob("*.npy")},
        )
    except BaseException as e:
        record.update(status="failed", error=repr(e))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
