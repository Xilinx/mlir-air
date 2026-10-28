# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""lm_eval_npu.py -- standard task scores (lm-evaluation-harness) of an NPU
model, next to the same items on its HF reference.

`NpuLM` is an in-process lm_eval model over a verify adapter's runner:

    loglikelihood    one prefill of context + continuation, scored from its
                     logits (the runner's prefill_logits). Single-token
                     continuations of one context, such as the letters of a
                     multiple-choice question, share one prefill.
    generate_until   prefill of the prompt, then greedy decode steps until a
                     stop string, EOS or max_gen_toks.

The reference runs the same task configuration through lm_eval's HF model on
the CPU and is cached under LLM_ACCURACY_CACHE, like accuracy.py's. Both sides
keep per-item scores, so the report counts the items each side gets right and
the other gets wrong, with an exact McNemar p-value.

Example:
    python lm_eval_npu.py --runner=lfm2_1_2b_q4nx.verify_adapter \\
        --tasks gsm8k --limit 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from accuracy import DEFAULT_CACHE, _revision, model_bos  # noqa: E402
from verify_runner import _get_tokenizer, _load_adapter  # noqa: E402

# The metric each task is compared on, and the lm_eval filter it comes from.
TASK_METRICS = {
    # flexible-extract takes the last number in the answer. strict-match
    # wants lm_eval's "The answer is N" format, which instruct models that
    # phrase their answers differently fail regardless of being right.
    "gsm8k": ("exact_match", "flexible-extract"),
    "mmlu": ("acc", "none"),
    "arc_challenge": ("acc_norm", "none"),
    "hellaswag": ("acc_norm", "none"),
}


def _lm_class():
    from lm_eval.api.model import LM

    class NpuLM(LM):
        def __init__(self, runner, tokenizer, bos, prefill_len, max_seq):
            super().__init__()
            self.runner = runner
            self.tok = tokenizer
            self.bos = bos
            self.prefill_len = prefill_len
            self.max_seq = max_seq

        def _encode(self, text):
            return self.tok.encode(text, add_special_tokens=False)

        def _encode_pair(self, context, continuation):
            # As lm_eval's HF model: an empty context becomes one prefix token,
            # trailing context spaces move into the continuation, and the
            # continuation is cut from the joint encoding.
            if not context:
                prefix = self.tok.bos_token_id
                if prefix is None:
                    prefix = self.tok.eos_token_id
                return [prefix], self._encode(continuation)
            n = len(context) - len(context.rstrip())
            if n:
                context, continuation = context[:-n], context[-n:] + continuation
            whole = self._encode(context + continuation)
            ctx = self._encode(context)
            ctx_ids = ([self.bos] if self.bos is not None else []) + ctx
            return ctx_ids, whole[len(ctx) :]

        def loglikelihood(self, requests):
            pairs = [self._encode_pair(*r.args) for r in requests]
            out = [None] * len(pairs)
            # Group single-token continuations of one context into one prefill.
            groups = {}
            for i, (ctx, cont) in enumerate(pairs):
                key = tuple(ctx) if len(cont) == 1 else ("single", i)
                groups.setdefault(key, []).append(i)
            for n, members in enumerate(groups.values()):
                ctx, cont = pairs[members[0]]
                full = ctx + cont
                # Keep the continuation whole; drop the oldest context.
                keep = self.prefill_len + 1
                full = full[-keep:]
                start = len(full) - len(cont)
                rows = self.runner.prefill_logits(
                    np.asarray(full[:-1], np.int64), start - 1
                )
                logp = rows - _logsumexp(rows)
                for i in members:
                    c = pairs[i][1]
                    if len(members) > 1:  # shared prefill: one row, 1-token conts
                        lp = logp[0]
                        out[i] = (float(lp[c[0]]), bool(lp.argmax() == c[0]))
                    else:
                        idx = np.arange(len(c))
                        out[i] = (
                            float(logp[idx, c].sum()),
                            bool((logp.argmax(1) == np.asarray(c)).all()),
                        )
                if n % 50 == 0:
                    print(
                        f"[lm_eval_npu] loglikelihood {n + 1}/{len(groups)}", flush=True
                    )
            return out

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError("use accuracy.py for perplexity")

        def generate_until(self, requests):
            eos = self.tok.eos_token_id
            out = []
            for n, r in enumerate(requests):
                context, kw = r.args
                until = kw.get("until") or []
                until = [until] if isinstance(until, str) else list(until)
                max_gen = int(kw.get("max_gen_toks", 256))
                ids = ([self.bos] if self.bos is not None else []) + self._encode(
                    context
                )
                # Leave room to generate, but never cut the prompt below half
                # the sequence; generation then stops at the end of the KV.
                keep = min(
                    self.prefill_len, max(self.max_seq - max_gen, self.max_seq // 2)
                )
                ids = ids[-keep:]
                max_gen = min(max_gen, self.max_seq - len(ids))
                pf = self.runner.prefill(np.asarray(ids, np.int64))
                gen, tok, pos = [], int(pf.top1_token), len(ids)
                while True:
                    if tok == eos:
                        break
                    gen.append(tok)
                    text = self.tok.decode(gen)
                    if len(gen) >= max_gen or any(s in text for s in until):
                        break
                    tok = int(self.runner.decode_step(tok, pos).top1_token)
                    pos += 1
                text = self.tok.decode(gen)
                for s in until:
                    text = text.split(s)[0]
                out.append(text)
                if n % 10 == 0:
                    print(f"[lm_eval_npu] generate {n + 1}/{len(requests)}", flush=True)
            return out

    return NpuLM


def _logsumexp(x):
    m = x.max(axis=-1, keepdims=True)
    return m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def run_tasks(lm, args):
    import lm_eval

    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=True,
        bootstrap_iters=0,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    return per_sample(res), res["results"], res.get("n-shot", {})


def _top_task(sub):
    """The TASK_METRICS task a (sub)task belongs to: mmlu_anatomy -> mmlu."""
    return next((t for t in TASK_METRICS if sub == t or sub.startswith(t + "_")), sub)


def group_shots(nshot):
    """{task: shots} with subtasks folded into their group when they agree."""
    shots = {}
    for sub, n in nshot.items():
        shots.setdefault(_top_task(sub), set()).add(n)
    return {t: next(iter(s)) if len(s) == 1 else None for t, s in shots.items()}


def per_sample(res):
    """{task: {doc_id: score}} for each top-level task's comparison metric,
    subtasks (mmlu's subjects) folded into their group."""
    out = {}
    for sub, samples in res["samples"].items():
        task = _top_task(sub)
        metric, filt = TASK_METRICS.get(task, ("acc", "none"))
        for s in samples:
            if s.get("filter", "none") != filt or metric not in s:
                continue
            out.setdefault(task, {})[f"{sub}:{s['doc_id']}"] = float(s[metric])
    return out


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(npu, ref):
    """Per task: both scores over the items both sides ran, and the flips."""
    rows = {}
    for task, a in npu.items():
        b = ref.get(task, {})
        common = sorted(set(a) & set(b))
        if not common:
            continue
        npu_only = sum(1 for k in common if a[k] > 0.5 >= b[k])
        ref_only = sum(1 for k in common if b[k] > 0.5 >= a[k])
        rows[task] = {
            "n": len(common),
            "npu": sum(a[k] for k in common) / len(common),
            "ref": sum(b[k] for k in common) / len(common),
            "npu_only_right": npu_only,
            "ref_only_right": ref_only,
            "mcnemar_p": mcnemar_p(npu_only, ref_only),
        }
    return rows


def ref_phase(args, hf_ref_model, bos, cache_path):
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=hf_ref_model,
        dtype="bfloat16",
        device="cpu",
        batch_size=args.ref_batch,
        add_bos_token=bos is not None,
        max_length=args.prefill_len + 1,
    )
    samples, results, nshot = run_tasks(lm, args)
    cache_path.write_text(
        json.dumps({"samples": samples, "results": results, "n-shot": nshot})
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runner", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--tasks", default="gsm8k", help="comma-separated lm_eval tasks")
    p.add_argument("--limit", type=int, default=None, help="items per (sub)task")
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--prefill-len", type=int, default=2048)
    p.add_argument("--ref-batch", type=int, default=1)
    p.add_argument("--phase", choices=["auto", "ref", "npu"], default="auto")
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("LLM_ACCURACY_CACHE", str(DEFAULT_CACHE)),
    )
    p.add_argument("--report-dir", default=str(HERE / "reports"))
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    args.tasks = [t for t in args.tasks.split(",") if t]

    adapter = _load_adapter(args.runner)
    model_name = adapter.resolve_model(args.model or adapter.DEFAULT_MODEL)
    hf_ref_model = (
        adapter.hf_reference(model_name)
        if hasattr(adapter, "hf_reference")
        else model_name
    )
    tokenizer = _get_tokenizer(model_name)
    bos = model_bos(adapter, tokenizer)
    settings = {
        "ref": hf_ref_model,
        "revision": _revision(hf_ref_model),
        "tasks": args.tasks,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "prefill_len": args.prefill_len,
        "bos": bos,
        # The cached reference holds per-item scores for these metrics only.
        "metrics": {t: TASK_METRICS.get(t) for t in args.tasks},
    }
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    cache = Path(args.cache_dir) / (
        f"{hf_ref_model.replace('/', '--')}.lm_eval.{'+'.join(args.tasks)}.{key}.json"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)

    if args.phase == "ref":
        ref_phase(args, hf_ref_model, bos, cache)
        return
    if args.phase == "auto" and not cache.exists():
        print("[lm_eval_npu] reference phase (subprocess)...", flush=True)
        cmd = [sys.executable, sys.argv[0]] + sys.argv[1:] + ["--phase", "ref"]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(rc)
    ref = json.loads(cache.read_text())

    print(f"[lm_eval_npu] building NPU runner ({args.runner})...", flush=True)
    runner = adapter.build_runner(
        model_name=model_name,
        config=adapter.build_config(),
        max_seq=args.prefill_len,
        tokenizer=tokenizer,
        npu_attn=True,
        lite_mode=True,
    )
    t0 = time.time()
    lm = _lm_class()(runner, tokenizer, bos, args.prefill_len, args.prefill_len)
    samples, results, nshot = run_tasks(lm, args)
    elapsed = time.time() - t0
    table = compare(samples, ref["samples"])

    record = {
        "model": args.runner.rsplit(".", 1)[0].replace(".", "/"),
        "model_name": model_name,
        "reference": {"id": hf_ref_model, "revision": settings["revision"]},
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": args.limit,
        "n_shot": group_shots(nshot),
        "seconds": elapsed,
        "tasks": table,
    }
    for task, r in table.items():
        print(
            f"[lm_eval_npu] {task}: NPU {100 * r['npu']:.1f}%  ref {100 * r['ref']:.1f}%"
            f"  (n={r['n']}; NPU-only right {r['npu_only_right']}, ref-only right "
            f"{r['ref_only_right']}, McNemar p={r['mcnemar_p']:.3f})"
        )
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"lm_eval_{stamp}.json").write_text(json.dumps(record, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
