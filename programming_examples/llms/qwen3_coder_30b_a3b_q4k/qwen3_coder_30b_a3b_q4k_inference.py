# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-Coder-30B-A3B on NPU2, end to end: the prompt through the fused prefill
# (--fused-dir / FUSED_PREFILL_DIR) or through the decode a token at a time, then
# the decode (qwen3_coder_30b_a3b_q4k_decode.py). Prints the TTFT and the decode
# tok/s; --compile-only builds the decode.
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import filelock
import numpy as np
from ml_dtypes import bfloat16

# first: importing the decode module sets up sys.path
from qwen3_coder_30b_a3b_q4k_decode import PER_WAVE, build_module, compile_all, load
import common
import qwen3_coder_30b_a3b_q4k_weights as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--prompt", default="The capital of France")
    ap.add_argument(
        "--prompt-len", type=int, default=0, help="repeat the prompt to N tokens"
    )
    ap.add_argument(
        "--fused-dir",
        type=Path,
        default=os.environ.get("FUSED_PREFILL_DIR") or None,
        help="prefill on this fused prefill build (fused_build.py)",
    )
    ap.add_argument(
        "--warm", action="store_true", help="prefill once untimed before timing"
    )
    ap.add_argument("--gen", type=int, default=8, help="tokens to generate")
    ap.add_argument("--maxl", type=int, default=64, help="context the build holds")
    ap.add_argument(
        "--reads",
        default="",
        help="also layers' ELFs reading only these many positions, e.g. 1024,4096",
    )
    ap.add_argument("--waves", type=int, default=W.N_LAYERS // PER_WAVE)
    ap.add_argument("--no-ref", action="store_true", help="skip the CPU reference")
    ap.add_argument("--bench", type=int, default=0, help="then time n tokens")
    ap.add_argument("--bench-l", type=int, default=0, help="at this context length")
    a = ap.parse_args()
    reads = [int(v) for v in a.reads.split(",") if v]
    if a.print_module_only:
        print(build_module(a.waves, a.maxl))
        return 0
    if a.compile_only:
        compile_all(a.waves, a.maxl, reads)
        print("Compilation passed.")
        return 0
    from reference import Reference

    w = W.Qwen3CoderQ4k()
    tok = w.tokenizer()
    ids = tok.encode(a.prompt).ids
    if a.prompt_len:
        ids = (ids * (a.prompt_len // len(ids) + 1))[: a.prompt_len]
    layers = list(range(PER_WAVE * a.waves))
    assert len(ids) + a.gen <= a.maxl
    # the prefill takes the whole prompt and the first token comes from the LM
    # head on its last row; a bench times the decode alone
    fused = a.fused_dir is not None and len(layers) == W.N_LAYERS and not a.bench
    if fused:
        import prefill as P

        manifest = a.fused_dir / "manifest.json"
        if not manifest.is_file():
            sys.exit(f"{a.fused_dir}: no fused prefill build (fused_build.py)")
        fused = len(ids) <= json.loads(manifest.read_text())["max_len"]
    # the device is held from the first context to the last dispatch
    lock = filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock"))
    with lock:
        t0 = time.time()
        npu = load(w, a.maxl, a.waves, P.WE_SLACK if fused else 0, reads)
        print(f"compiled and loaded in {time.time() - t0:.0f}s")

        def embed(t):
            return np.asarray(w.embed([t])[0]).astype(bfloat16)

        ref = None if a.no_ref else Reference(w, layers=layers)
        toks, ok, times = list(ids), True, []

        def check(i, got, new):
            nonlocal ok
            nxt = int(np.argmax(got))
            if ref is not None:
                want = ref.forward(new).float().numpy()
                ok &= nxt == int(np.argmax(want))
                print(
                    f"L={i + 1}: next {nxt} (reference {int(np.argmax(want))}),"
                    f" logits cos {common.compare(got, want)[0]:.4f}"
                )
            return nxt

        first, t_prefill = 0, 0.0
        if fused:
            pf = P.Q3CPrefill(a.fused_dir, w, layers)
            if a.warm:
                pf.prefill(ids)
            t0 = time.perf_counter()
            x = pf.prefill(ids)
            npu.seed(pf.kv_rows(), len(ids))
            got = npu.head(x[-1].astype(bfloat16), len(ids))
            t_prefill = time.perf_counter() - t0
            # its hw contexts slow the decode's dispatches; prefill() resumes
            pf.suspend()
            first = len(ids)
            print(f"prefill of {first} tokens: {t_prefill:.3f}s")
            nxt = check(first - 1, got, ids)
            if a.gen > 0:
                toks.append(nxt)
        # without the prefill the prompt goes through the decode a token at a time
        for i in range(first, len(ids) + a.gen - 1):
            t0 = time.perf_counter()
            got = npu.step(embed(toks[i]), i + 1)
            times.append(time.perf_counter() - t0)
            nxt = check(i, got, [toks[i]])
            if i + 1 >= len(ids):
                toks.append(nxt)
        print(repr(a.prompt), "->", repr(tok.decode(toks[len(ids) :])))
        n_prompt = max(0, len(ids) - first)
        print(f"Time to first token (TTFT): {t_prefill + sum(times[:n_prompt]):.3f}s")
        if len(times) > n_prompt:
            gen = np.array(times[n_prompt:]) * 1e3
            print(
                f"Decode: median {np.median(gen):.2f} ms ({1e3 / np.median(gen):.2f} tok/s)"
            )
        if a.bench:
            npu.bench(embed(toks[-1]), a.bench_l or len(toks), a.bench)
        del npu
        if fused:
            del pf
    if ref is not None:
        print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
