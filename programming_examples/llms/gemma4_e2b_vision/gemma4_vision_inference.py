#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E2B image + text -> text on NPU2.

  vision: gemma4_vision_encoder (this directory) -- the 16-layer tower and the
    multimodal projection, pixels -> soft tokens in the text model's space.
  prefill: ../gemma4_e2b_q4nx/fused_prefill -- the text model's fused prefill,
    with the soft tokens in place of the image placeholder's embeddings.
  decode: ../gemma4_e2b_q4nx's full-ELF fused decode, seeded with the prefill's
    KV.

Run:
  python3 gemma4_vision_inference.py --gate          # shapes image, keyword gate
  python3 gemma4_vision_inference.py --image photo.jpg --prompt "What is this?"
  python3 gemma4_vision_inference.py --repeat 4      # TTFT + decode tok/s

Needs the text model's fused prefill and decode (`make compile`).
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TEXT = _HERE.parent / "gemma4_e2b_q4nx"
_PE = _HERE.parent.parent  # programming_examples
for _p in (str(_HERE), str(_TEXT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(_PE)
]

import gemma4_vision_reference as R  # noqa: E402
from gemma4_vision_encoder import NpuOps, VisionEncoder  # noqa: E402
from gemma4_vision_weights import VisionWeights, preprocess  # noqa: E402

Q4NX_MODEL = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma4-E2B-IT-NPU2")
DEFAULT_PROMPT = "Describe the shapes and their colors and positions."


class TokenIds:
    """The multimodal token ids, from the checkpoint's config.json."""

    def __init__(self, cfg):
        self.image = cfg["image_token_id"]
        self.boi = cfg["boi_token_id"]
        self.eoi = cfg["eoi_token_id"]
        self.pad = cfg["text_config"]["pad_token_id"]
        self.bos = cfg["text_config"]["bos_token_id"]


def load_tokenizer():
    from transformers import AutoTokenizer

    d = VisionWeights().path.parent
    cfg = json.loads((d / "config.json").read_text())
    return AutoTokenizer.from_pretrained(str(d)), TokenIds(cfg)


def build_prompt_ids(tok, ids_cfg, text, n_soft):
    """<bos> + the chat-templated user turn with its image placeholder expanded
    to boi + image * n_soft + eoi, as Gemma4Processor expands it."""
    rendered = tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    if not ids or ids[0] != ids_cfg.bos:
        ids = [ids_cfg.bos] + ids
    hits = [i for i, t in enumerate(ids) if t == ids_cfg.image]
    assert len(hits) == 1, f"expected one image placeholder, got {len(hits)}"
    i = hits[0]
    return (
        ids[:i]
        + [ids_cfg.boi]
        + [ids_cfg.image] * n_soft
        + [ids_cfg.eoi]
        + ids[i + 1 :]
    )


class VisionLanguageModel:
    """Vision tower, fused prefill and fused decode, loaded once."""

    def __init__(self, fused_dir, max_len=2048, verbose=False):
        from air_examples.llms.gemma4_e2b_q4nx.fused_prefill.runtime import (
            FusedPrefill,
        )

        self.verbose = verbose
        self.tok, self.ids = load_tokenizer()
        self.enc = VisionEncoder(NpuOps(verbose=verbose))
        self.pf = FusedPrefill(fused_dir, max_len=max_len)
        self.pf.load_weights(model=Q4NX_MODEL)

    def prompt(self, image, text, max_soft_tokens=None):
        """-> (pixel_values, position_ids, prompt ids)."""
        kw = {"max_soft_tokens": max_soft_tokens} if max_soft_tokens else {}
        pv, pid, n_soft = preprocess(image, **kw)
        return pv, pid, build_prompt_ids(self.tok, self.ids, text, n_soft)

    def prefill(self, pv, pid, ids):
        """-> (logits, timings). The image slots carry the pad id into the
        prefill, which the per-layer token table is indexed by."""
        t0 = time.perf_counter()
        soft = self.enc.encode(pv, pid)
        t_vis = time.perf_counter() - t0
        pos = [i for i, t in enumerate(ids) if t == self.ids.image]
        assert len(pos) == len(soft), (len(pos), len(soft))
        llm_ids = [self.ids.pad if t == self.ids.image else t for t in ids]
        self.pf.current_context_length = 0
        t1 = time.perf_counter()
        logits = self.pf.prefill(llm_ids, soft_embeds=soft, soft_positions=pos)
        t_pre = time.perf_counter() - t1
        return np.asarray(logits, np.float32), dict(vision=t_vis, prefill=t_pre)

    def generate(self, ids, first_logits, n_tokens, stop=True, topk=None):
        """Greedy decode after a prefill of `ids`. -> (tokens, decode seconds,
        per-step top-k ids if `topk`)."""
        from gemma4_e2b_q4nx_inference import EOS_IDS, FusedDecoder
        from air_examples.llms.verify.comparators import topk_token_ids

        P = len(ids)
        ks, vs = self.pf.kv_stack()
        self.pf.suspend()  # release its hw_contexts for the decoder
        dec = FusedDecoder(model=Q4NX_MODEL, max_L=P + n_tokens, verbose=self.verbose)
        try:
            dec.seed_kv(ks, vs, P)
            tok = int(first_logits.argmax())
            gen, tops = [tok], []
            if topk:
                tops.append(topk_token_ids(first_logits, topk))
            t0 = time.perf_counter()
            for p in range(P, min(P + n_tokens - 1, dec.maxL)):
                if stop and tok in EOS_IDS:
                    break
                z = np.asarray(dec.dispatch(tok, p), np.float32)
                tok = int(z.argmax())
                gen.append(tok)
                if topk:
                    tops.append(topk_token_ids(z, topk))
            t_dec = time.perf_counter() - t0
            if len(gen) < n_tokens and not (stop and tok in EOS_IDS):
                print(
                    f"[vlm] stopped at the decode build's context limit "
                    f"({dec.maxL}); rebuild with a larger RUN_LBUILD",
                    flush=True,
                )
        finally:
            dec.close()
        if stop and gen[-1] in EOS_IDS:
            gen = gen[:-1]
        return gen, t_dec, tops


# Each group must appear in the shapes image's description. Position words
# are one group because the model phrases positions freely.
_GATE_TERMS = [
    ("red", ("red",)),
    ("square", ("square", "rectangle")),
    ("blue", ("blue",)),
    ("circle", ("circle", "round")),
    ("position", ("left", "right", "upper", "lower", "top", "bottom")),
]


def _gate(text):
    low = text.lower()
    missing = [name for name, alts in _GATE_TERMS if not any(a in low for a in alts)]
    if missing:
        print(f"FAIL: the description does not mention {missing}")
        return 1
    print("PASS!")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=None, help="default: the shapes image")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--n-tokens", type=int, default=100)
    ap.add_argument("--max-soft-tokens", type=int, default=None)
    ap.add_argument(
        "--fused-dir", default=str(_TEXT / "_fused_prefill"), help="fused prefill build"
    )
    ap.add_argument("--repeat", type=int, default=1, help="timed turns, model resident")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t_setup = time.perf_counter()
    vlm = VisionLanguageModel(args.fused_dir, verbose=args.verbose)
    pv, pid, ids = vlm.prompt(
        R.load_image(args.image), args.prompt, args.max_soft_tokens
    )
    print(
        f"[vlm] prompt {len(ids)} tokens ({sum(t == vlm.ids.image for t in ids)} "
        f"image); setup {time.perf_counter() - t_setup:.1f}s",
        flush=True,
    )

    rounds, text = [], ""
    for r in range(args.repeat + (1 if args.repeat > 1 else 0)):
        logits, t = vlm.prefill(pv, pid, ids)
        gen, t_dec, _ = vlm.generate(ids, logits, args.n_tokens)
        text = vlm.tok.decode(gen)
        n = max(len(gen) - 1, 1)
        t.update(ttft=t["vision"] + t["prefill"], tok_s=n / t_dec, n=len(gen))
        warm = args.repeat > 1 and r == 0
        print(
            f"[vlm] {'warm-up' if warm else f'turn {len(rounds) + 1}'}: "
            f"vision {t['vision'] * 1e3:.0f} ms + prefill {t['prefill'] * 1e3:.0f} ms"
            f" = TTFT {t['ttft'] * 1e3:.0f} ms | decode {t['tok_s']:.2f} tok/s "
            f"({len(gen)} tokens)",
            flush=True,
        )
        if not warm:
            rounds.append(t)

    print("\n--- generated ---\n" + text + "\n")
    med = {k: float(np.median([r[k] for r in rounds])) for k in rounds[0]}
    print(f"Time to first token (TTFT): {med['ttft']:.3f}s")
    print(f"Decode throughput: {med['tok_s']:.2f} tok/s")
    return _gate(text) if args.gate else 0


if __name__ == "__main__":
    raise SystemExit(main())
