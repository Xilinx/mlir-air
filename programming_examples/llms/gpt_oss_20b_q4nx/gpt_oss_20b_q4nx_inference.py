# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS-20B on NPU2, end to end: the prompt through the fused prefill
# (FUSED_PREFILL_DIR) or through the decode a token at a time, then the decode
# (gpt_oss_20b_q4nx_decode.py). Prints the TTFT and the decode tok/s;
# --compile-only builds the decode.
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import filelock
import numpy as np
import torch

# first: importing the decode module sets up sys.path
from gpt_oss_20b_q4nx_decode import DH, NpuDecoder, build_module, compile_decode
import gpt_oss_20b_q4nx_weights as g
import prefill


def cos_sim(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


class ChainRef:
    """The CPU reference through an arbitrary sequence of layers."""

    def __init__(self, w, layers):
        self.w, self.layers = w, layers
        self.kv = [[torch.zeros(g.N_KV_HEADS, 0, DH)] * 2 for _ in layers]

    def step(self, token, pos):
        x = self.w.embed(torch.tensor([token]))
        for j, l in enumerate(self.layers):
            n1, n2 = self.w.norms(l)
            x = x + g.attention(self.w.attn(l), l, g.rmsnorm(x, n1), self.kv[j], [pos])
            x = x + g.moe(self.w, l, g.rmsnorm(x, n2))[0]
        return x[0]


def prompt_ids(tok, prompt, n):
    """The prompt's tokens, repeated to n tokens when n is given."""
    ids = tok.encode(prompt).ids
    return (ids * (n // len(ids) + 1))[:n] if n else ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--maxl", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=g.N_LAYERS)
    ap.add_argument(
        "--layers", type=int, nargs="*", help="model layer of each wave (debug)"
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument(
        "--prompt-len", type=int, default=0, help="repeat the prompt to N tokens"
    )
    ap.add_argument("--steps", type=int, default=8, help="greedy tokens")
    ap.add_argument("--build-dir", type=Path, default=Path("build_decode"))
    ap.add_argument("--no-lm", action="store_true", help="skip the LM head")
    ap.add_argument(
        "--no-ref", action="store_true", help="time only: no CPU reference check"
    )
    ap.add_argument(
        "--no-prefill", action="store_true", help="the prompt goes through decode"
    )
    ap.add_argument(
        "--warm", action="store_true", help="prefill once untimed before timing"
    )
    ap.add_argument(
        "--fused-dir",
        type=Path,
        default=os.environ.get("FUSED_PREFILL_DIR") or None,
        help="prefill on this fused prefill build (fused_build.py), not per-op",
    )
    ap.add_argument(
        "--first-token",
        type=Path,
        help="a file holding the reference's first token (prefill_fused.py "
        "--token-out), checked without the CPU reference in memory",
    )
    ap.add_argument("--bench", type=int, default=0, help="time N dispatches")
    ap.add_argument(
        "--bench-ctx", type=int, default=0, help="context of the timed dispatches"
    )
    a = ap.parse_args()
    layers = a.layers or list(range(a.n_layers))
    if a.print_module_only:
        print(build_module(a.maxl, len(layers), not a.no_lm))
        return 0
    elf, kernel = compile_decode(
        a.maxl, len(layers), a.build_dir.resolve(), lm=not a.no_lm
    )
    if a.compile_only:
        print("Compilation passed.")
        return 0
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(g.flm_model_dir() / "tokenizer.json"))
    ids = prompt_ids(tok, a.prompt, a.prompt_len)
    full = layers == list(range(g.N_LAYERS))
    # FLM prefills prompts of MIN_PREFILL tokens or more and decodes shorter
    # ones; the fused prefill takes prompts up to the length it was built for
    use_prefill = full and not a.no_prefill and not a.bench
    use_prefill &= prefill.MIN_PREFILL <= len(ids) <= a.maxl
    use_fused = use_prefill and a.fused_dir is not None
    if use_fused:
        manifest = a.fused_dir / "manifest.json"
        if not manifest.is_file():
            sys.exit(f"{a.fused_dir}: no fused prefill build (fused_build.py)")
        use_fused = len(ids) - 1 <= json.loads(manifest.read_text())["max_len"]
    w = g.GptOssQ4nx()
    ref = None
    if not a.no_ref:
        ref = g.Reference(w) if full else ChainRef(w, layers)
    ok, head = True, None
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        npu = NpuDecoder(elf, kernel, a.maxl, layers, w)
        if a.bench:
            ctx = a.bench_ctx or len(ids)
            npu.step(ids[0], 1)
            t = np.array(npu.bench(ctx, a.bench)) * 1e3
            med = float(np.median(t))
            print(
                f"[bench] decode ctx={ctx} median {med:.2f} ms min {t.min():.2f} ms "
                f"({1e3 / med:.2f} tok/s)"
            )
            return 0
        seq, out, times = list(ids), [], []
        # the last prompt token takes a decode step, whose LM head gives the
        # first token
        first, t_prefill = 0, 0.0
        if use_fused:
            import prefill_fused

            pf = prefill_fused.GptOssFused(a.fused_dir, w, we=(npu.bo["WE"], 0))
            if a.warm:
                pf.prefill(ids[:-1])
            pf.by_op.clear()
            t0 = time.perf_counter()
            kv = pf.prefill(ids[:-1])
            npu.seed_kv(kv)
            t_prefill = time.perf_counter() - t0
            first = len(ids) - 1
            steps = ", ".join(f"{k} {v:.2f}s" for k, v in sorted(pf.by_op.items()))
            print(f"prefill of {first} tokens: {t_prefill:.3f}s (fused; {steps})")
        elif use_prefill:
            ops = prefill.NpuOps(str(a.build_dir / "prefill"), experts=npu.experts)
            ops.preload(w)
            for j in range(len(layers)):
                npu.experts(j)
            pf = prefill.Prefill(w, ops)
            if a.warm:
                pf.forward(ids[:-1], lm=False)
            prefill.TIMES.clear()
            t0 = time.perf_counter()
            _, kv = pf.forward(ids[:-1], lm=False)
            npu.seed_kv(kv)
            t_prefill = time.perf_counter() - t0
            first = len(ids) - 1
            steps = ", ".join(f"{k} {v:.2f}s" for k, v in prefill.TIMES.items())
            print(f"prefill of {first} tokens: {t_prefill:.3f}s ({steps})")
        for t in range(first, min(len(ids) + a.steps - 1, a.maxl)):
            t0 = time.perf_counter()
            x, logits = npu.step(seq[t], t + 1)
            dt = time.perf_counter() - t0
            times.append(dt)
            want = None
            if ref is not None and not full:
                # a debug build: its last wave's output, and the LM head on it
                xr = ref.step(seq[t], t)
                c = cos_sim(x, xr)
                ok &= c > 0.99
                print(f"L={t + 1}: last wave out cos {c:.5f} ({dt*1e3:.0f} ms)")
                if not a.no_lm:
                    if head is None:
                        head = w.lm_head()
                    want = g.rmsnorm(xr[None], w.final_norm())[0] @ head.T
            elif ref is not None:
                want = ref.forward(seq[: t + 1] if t == first else [seq[t]])
            if t + 1 < len(ids):
                if ref is not None and full:
                    print(f"L={t + 1}: prompt ({dt*1e3:.0f} ms)")
                continue
            if a.no_lm:
                break  # no logits to pick the next token from
            nxt = int(logits.argmax())
            msg = f"L={t + 1}: npu {nxt} {tok.decode([nxt])!r}"
            if want is not None:
                c = cos_sim(logits, want)
                msg += f" vs ref {int(want.argmax())}  logits cos {c:.5f}"
                # a prefilled cache carries the NPU GEMMs' bfp16 products
                ok &= nxt == int(want.argmax()) and c > (0.98 if first else 0.99)
            if a.first_token and not out:
                want_tok = int(a.first_token.read_text())
                msg += f" vs reference first token {want_tok}"
                ok &= nxt == want_tok
            print(f"{msg} ({dt*1e3:.0f} ms)")
            seq.append(nxt)
            out.append(nxt)
    if out:
        print(f"{tok.decode(ids)!r} -> {tok.decode(out)!r}")
        # the first token is ready after the prompt's last decode step
        n_prompt = len(ids) - first
        gen = np.array(times[n_prompt:]) * 1e3
        print(f"Inference: prompt_len={len(ids)}, n_tokens={len(out)}")
        ttft = t_prefill + sum(times[:n_prompt])
        print(f"Time to first token (TTFT): {ttft:.3f}s")
        if len(gen):
            med = float(np.median(gen))
            print(
                f"Decode: median {med:.2f} ms min {gen.min():.2f} ms "
                f"({1e3 / med:.2f} tok/s)"
            )
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
