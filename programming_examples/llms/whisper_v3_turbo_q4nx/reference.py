# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CPU BF16 reference of the gate's clips: openai/whisper-large-v3-turbo at
its pinned revision (transformers, bfloat16), decoded as the NPU decodes
(whisper_npu.Decoding), each step's full logits checked finite. Writes
REF_DIR/reference.json: per clip the tokens and each step's top-5.

  python3 reference.py --out REF_DIR [--clips N]
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

TOPK = 5


def main():
    import torch
    from transformers import WhisperForConditionalGeneration

    from air_examples.llms.whisper_v3_turbo_q4nx import inputs
    from air_examples.llms.whisper_v3_turbo_q4nx.whisper_npu import (
        GEN_REPO,
        GEN_REVISION,
        MAX_NEW,
        Decoding,
        log_mel,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = WhisperForConditionalGeneration.from_pretrained(
        GEN_REPO, revision=GEN_REVISION, torch_dtype=torch.bfloat16
    ).eval()
    dec = Decoding()
    recs = []
    for cid, audio, _ in inputs.clips(a.clips):
        t0 = time.perf_counter()
        mel = torch.from_numpy(log_mel(audio))[None].bfloat16()
        toks, tops = list(dec.prompt), []
        with torch.no_grad():
            enc = model.model.encoder(mel).last_hidden_state
            past = None
            ids = torch.tensor([toks])
            for i in range(MAX_NEW):
                o = model(
                    encoder_outputs=(enc,),
                    decoder_input_ids=ids,
                    past_key_values=past,
                    use_cache=True,
                )
                past = o.past_key_values
                lg = o.logits[0, -1].float().numpy()
                if not np.isfinite(lg).all():
                    raise RuntimeError(f"{cid}: non-finite logits at step {i}")
                z = dec.mask(lg, i == 0)
                tops.append(np.argsort(-z, kind="stable")[:TOPK].tolist())
                t = int(np.argmax(z))
                toks.append(t)
                ids = torch.tensor([[t]])
                if t == dec.eos:
                    break
        out = toks[len(dec.prompt) :]
        recs.append(dict(id=cid, tokens=out, top=tops))
        print(f"{cid}: {len(out)} tokens, {time.perf_counter() - t0:.1f} s", flush=True)
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / "reference.json").write_text(
        json.dumps(
            dict(
                model=GEN_REPO,
                revision=GEN_REVISION,
                precision="bf16",
                dataset=inputs.DATASET,
                dataset_revision=inputs.REVISION,
                clips=recs,
            )
        )
    )
    print(f"wrote {d / 'reference.json'} ({len(recs)} clips)")


if __name__ == "__main__":
    main()
