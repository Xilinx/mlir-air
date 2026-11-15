# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Pack the z-lab Qwen3-Coder-30B-A3B DFlash drafter for the B8 decoder.

The body is quantized group32 Q4NX from the official BF16, as the 8B drafter;
the vocabulary head is the target's, quantized the same way.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

from paths import HERE
from drafter import Safetensors
from model_paths import default_draft_cache, draft_snapshot
from common.prepare_draft_8b import quantized_blocks
from common.prepare_target_8b import cascade, inverse

K, DQ, DKV, FFN, LAYERS = 2048, 4096, 1024, 6144, 8
VOCAB, VOCAB_PADDED, HEAD_WAVE_ROWS = 151936, 153600, 20 * 512
LAYER_WORDS = ((DQ + 2 * DKV) * K + K * DQ + 3 * FFN * K) * 5 // 16
HEAD_WORDS = VOCAB_PADDED * K * 5 // 16


def target_head():
    sys.path.insert(0, str(HERE.parents[1] / "qwen3_coder_30b_a3b_q4k"))
    import qwen3_coder_30b_a3b_q4k_weights as W

    return np.asarray(W.Qwen3CoderQ4k().lm_head, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=Path, help="drafter snapshot dir")
    ap.add_argument("--output", type=Path, default=default_draft_cache())
    a = ap.parse_args()
    a.draft = draft_snapshot(a.draft)
    done = a.output / "manifest.json"
    if done.is_file() and json.loads(done.read_text()).get("status") == "prepared":
        print(f"{a.output} is prepared")
        return
    a.output.mkdir(parents=True, exist_ok=False)
    st = Safetensors(a.draft / "model.safetensors")
    packed = np.lib.format.open_memmap(
        a.output / "W.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(LAYERS * LAYER_WORDS + HEAD_WORDS,),
    )
    offset, norms, qk = 0, [], []
    for layer in range(LAYERS):

        def matrix(stem):
            return quantized_blocks(st.tensor(f"layers.{layer}.{stem}.weight"))

        def twice(stem):
            # each KV head twice: q head h reads slot h // 4, a copy of h // 8
            t = np.asarray(st.tensor(f"layers.{layer}.{stem}.weight"), np.float32)
            return quantized_blocks(
                np.repeat(t.reshape(4, 128, K), 2, 0).reshape(-1, K)
            )

        qkv = np.concatenate(
            [
                matrix("self_attn.q_proj"),
                twice("self_attn.k_proj"),
                twice("self_attn.v_proj"),
            ]
        )
        o = matrix("self_attn.o_proj")
        up, gate = matrix("mlp.up_proj"), matrix("mlp.gate_proj")
        glu = np.stack(
            [
                up.reshape(FFN // 512, 16, K // 256, 2560),
                gate.reshape(FFN // 512, 16, K // 256, 2560),
            ],
            axis=1,
        ).reshape(2 * FFN // 32, K // 256, 2560)
        down = matrix("mlp.down_proj")
        begin = offset
        for raw in (qkv, o, glu, down):
            words = cascade(raw)
            assert np.array_equal(inverse(words, *raw.shape[:2]), raw)
            packed[offset : offset + words.size] = words
            offset += words.size
        assert offset - begin == LAYER_WORDS, (offset - begin, LAYER_WORDS)
        norms.append(
            [
                st.tensor(f"layers.{layer}.{k}.weight")
                for k in ("input_layernorm", "post_attention_layernorm")
            ]
        )
        qk.append(
            [
                st.tensor(f"layers.{layer}.self_attn.{k}.weight")
                for k in ("q_norm", "k_norm")
            ]
        )
        print(f"packed drafter layer {layer + 1}/{LAYERS}", flush=True)
    raw = quantized_blocks(target_head())
    padded = np.zeros((VOCAB_PADDED // 32, K // 256, 2560), np.uint16)
    padded[: len(raw)] = raw
    for wave in range(VOCAB_PADDED // HEAD_WAVE_ROWS):
        source = padded[wave * HEAD_WAVE_ROWS // 32 : (wave + 1) * HEAD_WAVE_ROWS // 32]
        words = cascade(source)
        packed[offset : offset + words.size] = words
        offset += words.size
    assert offset == packed.size
    packed.flush()
    np.save(a.output / "RMS.npy", np.asarray(norms, bfloat16).view(np.uint16))
    np.save(a.output / "QK.npy", np.asarray(qk, bfloat16).view(np.uint16))
    (a.output / "manifest.json").write_text(
        json.dumps(
            dict(
                status="prepared",
                role="drafter",
                layers=LAYERS,
                layer_words=LAYER_WORDS,
                head_words=HEAD_WORDS,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
