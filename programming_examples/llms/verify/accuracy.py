# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""accuracy.py — teacher-forced PPL / KL divergence of an NPU model vs a reference.

The method is llama.cpp's `llama-perplexity --kl-divergence` (see kld.py):
wikitext-2 test is cut into chunks of --ctx tokens, and in each chunk the
predictions for positions ctx/2 .. ctx-2 are scored, so every scored token has
at least ctx/2 tokens of context. Both sides are fed the true text, never their
own picks, so one early disagreement does not end the comparison the way it
does in the `make verify` gate.

Two phases, run as separate processes by --phase auto so the reference and NPU
weights never share host RAM:

    ref   Load the reference (HF bf16, or --ref dequant: the HF model with
          the weights the NPU runs, dequantized, from the adapter's
          build_dequant_hf_model), run each chunk through it and save the
          scored log-probs to a base file. Skipped when a base file for the
          same settings and weights already exists.
    npu   Build the adapter's NPU runner and score it against the base file.

--path decode drives the NPU the way generation does: prefill positions
0 .. ctx/2, then one decode_step per remaining position, fed the true token.
--path prefill runs the chunk through the NPU prefill once and takes the
logits of every scored row from it (the adapter's prefill_logits). Both paths
score the same rows, so they share one base file.
--ctx N --score S scores only the last S positions of a chunk of N; with the
decode path the chunk is reached through the decode's own KV cache, which
measures accuracy at depth.

Example:
    python accuracy.py --runner=qwen3_0_6b.verify_adapter --chunks 4
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from kld import (  # noqa: E402
    BaseLogitsReader,
    BaseLogitsWriter,
    KldStats,
    format_summary,
    scored_per_chunk,
)
from verify_runner import _get_tokenizer, _load_adapter  # noqa: E402

DATASET = "wikitext-2-test"
_WIKITEXT_REPO = "ggml-org/ci"  # what llama.cpp's scripts/get-wikitext-2.sh fetches
_WIKITEXT_ZIP = "wikitext-2-raw-v1.zip"
_WIKITEXT_MEMBER = "wikitext-2-raw/wiki.test.raw"
DEFAULT_CACHE = Path.home() / ".cache" / "mlir-air" / "accuracy"


def load_wikitext() -> tuple[str, str]:
    """wikitext-2 raw test text and its sha256, via the HF hub cache, so it
    works offline once cached."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=_WIKITEXT_REPO, repo_type="dataset", filename=_WIKITEXT_ZIP
    )
    raw = zipfile.ZipFile(path).read(_WIKITEXT_MEMBER)
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _revision(source: str) -> str:
    """An identity for a weight source that changes when its weights do: the
    cached snapshot commit of an HF repo id, or for a local file or directory
    the size and modification time of every file in it. Read from one cached
    file rather than snapshot_download, which rejects a snapshot fetched with
    allow_patterns as incomplete."""
    if os.path.exists(source):
        root = Path(source).resolve()
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        stats = [
            f"{f.relative_to(root.parent)}:{f.stat().st_size}:{f.stat().st_mtime_ns}"
            for f in files
            if f.is_file()
        ]
        return hashlib.sha256("\n".join(stats).encode()).hexdigest()
    from huggingface_hub import try_to_load_from_cache

    for name in ("config.json", "model.q4nx"):
        path = try_to_load_from_cache(source, name)
        if isinstance(path, str):
            return Path(path).parent.name
    return "uncached"


def chunk_tokens(tokenizer, text: str, ctx: int, n_chunks: int, bos) -> np.ndarray:
    """First `n_chunks` chunks of `ctx` tokens; position 0 of each is replaced
    by `bos` when the model expects one, as llama-perplexity does."""
    ids = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int64)
    available = ids.size // ctx
    if available < n_chunks:
        raise SystemExit(f"text has {available} chunks of {ctx}; asked for {n_chunks}")
    chunks = ids[: n_chunks * ctx].reshape(n_chunks, ctx).copy()
    if bos is not None:
        chunks[:, 0] = bos
    return chunks


def model_bos(adapter, tokenizer):
    bos = getattr(adapter, "PROMPT_BOS", None)
    if bos is not None:
        return int(bos)
    probe = tokenizer.encode("a")
    if tokenizer.bos_token_id is not None and probe[:1] == [tokenizer.bos_token_id]:
        return int(tokenizer.bos_token_id)
    return None


def _dequant_builder(adapter):
    """The adapter's dequant-reference hook. build_dequant_hf_model is
    accuracy-only; an adapter's build_hf_model also replaces the `make verify`
    reference, so it serves here too."""
    build = getattr(adapter, "build_dequant_hf_model", None) or getattr(
        adapter, "build_hf_model", None
    )
    if build is None:
        raise SystemExit(
            "--ref dequant needs the adapter to provide build_dequant_hf_model "
            "or build_hf_model (the reference with the NPU's quantized weights)"
        )
    return build


def _path_dependent(build) -> bool:
    """Whether a dequant hook builds a different reference per NPU path (a tied
    head the prefill runs in bf16 and the decode requantized)."""
    return "path" in inspect.signature(build).parameters


def build_reference(adapter, kind, model_name, hf_ref_model, config, path):
    import torch
    from transformers import AutoModelForCausalLM

    if kind == "dequant":
        build = _dequant_builder(adapter)
        kwargs = {"path": path} if _path_dependent(build) else {}
        model = build(
            npu_model_name=model_name,
            hf_ref_model=hf_ref_model,
            config=config,
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_ref_model, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    return model.eval()


def run_ref_phase(args, adapter, model_name, hf_ref_model, base_path, meta):
    import torch

    tokenizer = _get_tokenizer(model_name)
    text, _ = load_wikitext()
    chunks = chunk_tokens(tokenizer, text, args.ctx, args.chunks, meta["bos"])
    print(f"[accuracy] reference {hf_ref_model} ({args.ref}): {args.chunks} chunks")
    model = build_reference(
        adapter, args.ref, model_name, hf_ref_model, adapter.build_config(), args.path
    )
    meta["ref_revision"] = getattr(model.config, "_commit_hash", None)
    first = args.first
    writer = None
    tmp = base_path.with_suffix(".tmp")
    for ci, toks in enumerate(chunks):
        t0 = time.time()
        with torch.no_grad():
            scored = _reference_logits(model, toks, first)
        if writer is None:
            meta["n_vocab"] = scored.shape[1]
            writer = BaseLogitsWriter(
                tmp, args.ctx, scored.shape[1], chunks.ravel(), first
            )
        writer.add_chunk(scored)
        print(f"[accuracy]   ref chunk {ci + 1}/{args.chunks}: {time.time() - t0:.1f}s")
    writer.close()
    tmp.replace(base_path)
    base_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[accuracy] base file: {base_path}")
    for old in drop_superseded(base_path):
        print(f"[accuracy] removed superseded {old.name}")


def drop_superseded(base_path: Path) -> list:
    """Delete this reference's files from earlier weights or hooks: the same
    name up to the fingerprint, with another fingerprint. A base file is
    large, so the cache would otherwise grow with every revision."""
    head, _, fp = base_path.stem.rpartition(".")
    removed = []
    for p in sorted(base_path.parent.glob(f"{head}.*")):
        if p.suffix not in (".kld", ".json"):
            continue
        h, _, f = p.stem.rpartition(".")
        if h == head and f != fp and len(f) == len(fp):
            p.unlink()
            removed.append(p)
    return removed


# Past this length the reference runs in segments through its KV cache:
# attention over the whole chunk at once would need memory quadratic in it.
_REF_SEGMENT = 4096


def _longrope(model):
    """Whether the model uses LongRoPE. HF picks its RoPE table per call from
    that call's positions, so a chunk past the original context run in segments
    would rotate its first keys with one table and its later queries with the
    other; such a chunk has to run in one pass."""
    c = model.config.get_text_config()
    rope = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None)
    return (
        isinstance(rope, dict) and rope.get("rope_type", rope.get("type")) == "longrope"
    )


def _reference_logits(model, toks, first):
    """[n - 1 - first, vocab] float32 reference logits for rows first..n-2."""
    import torch

    n = len(toks)
    ids = torch.tensor(toks).unsqueeze(0)
    if n <= _REF_SEGMENT or _longrope(model):
        # Logits only for rows first..n-1; the last is not scored.
        out = model(ids, use_cache=False, logits_to_keep=n - first)
        return out.logits[0, :-1].float().numpy()
    rows, past, seg = [], None, 1024
    for s in range(0, n, seg):
        e = min(s + seg, n)
        keep = e - max(s, first) if e > first else 1
        out = model(
            ids[:, s:e], past_key_values=past, use_cache=True, logits_to_keep=keep
        )
        past = out.past_key_values
        if e > first:
            rows.append(out.logits[0].float().numpy())
    return np.concatenate(rows)[:-1]


def decode_path_logits(
    npu, toks: np.ndarray, ctx: int, first: int, prefill_len: int
) -> np.ndarray:
    """NPU logits for positions first .. ctx-2. The prefill covers up to
    `prefill_len` tokens; teacher-forced decode steps take it from there, so a
    long chunk reaches its scored tail through the decode's own KV cache."""
    n = min(first + 1, prefill_len)
    pf = npu.prefill(toks[:n])
    rows = [np.asarray(pf.logits_at_pred, dtype=np.float32)] if n == first + 1 else []
    for pos in range(n, ctx - 1):
        ds = npu.decode_step(int(toks[pos]), pos)
        if pos >= first:
            rows.append(np.asarray(ds.lm_head_logits, dtype=np.float32))
        elif (pos - n) % 1024 == 1023:
            print(f"[accuracy]     filled KV to {pos + 1}/{first}", flush=True)
    return np.stack(rows)


def prefill_path_logits(
    npu, toks: np.ndarray, ctx: int, first: int, prefill_len: int
) -> np.ndarray:
    """NPU logits for positions first .. ctx-2 from one prefill of the chunk."""
    if not hasattr(npu, "prefill_logits"):
        raise SystemExit("--path prefill needs the adapter runner's prefill_logits")
    if ctx - 1 > prefill_len:
        raise SystemExit(f"--path prefill needs --ctx <= {prefill_len + 1}")
    return npu.prefill_logits(toks[: ctx - 1], first)


def run_npu_phase(args, adapter, model_name, base_path, meta):
    base = BaseLogitsReader(base_path, args.first)
    if base.n_ctx != args.ctx or base.n_chunk < args.chunks:
        raise SystemExit(f"{base_path} does not match --ctx/--chunks")
    tokenizer = _get_tokenizer(model_name)
    print(f"[accuracy] building NPU runner ({args.runner})...")
    npu = adapter.build_runner(
        model_name=model_name,
        config=adapter.build_config(),
        max_seq=args.ctx,
        tokenizer=tokenizer,
        npu_attn=(args.npu_attn == "on"),
        lite_mode=True,
    )
    stats = KldStats()
    per_chunk = []
    first = args.first
    for ci in range(args.chunks):
        toks = base.chunk_tokens(ci).astype(np.int64)
        t0 = time.time()
        path_logits = (
            decode_path_logits if args.path == "decode" else prefill_path_logits
        )
        logits = path_logits(npu, toks, args.ctx, first, args.prefill_len)
        elapsed = time.time() - t0
        if logits.shape[1] != base.n_vocab:
            raise SystemExit(
                f"NPU vocab {logits.shape[1]} != reference vocab {base.n_vocab}"
            )
        chunk_stats = KldStats()
        targets = toks[first + 1 :]
        chunk_stats.add(logits, base.chunk(ci), targets)
        stats.merge(chunk_stats)
        # Per-chunk numbers are diagnostics; the token minimum applies to the
        # total, so short scored tails still combine into a valid run.
        c = chunk_stats.summary(min_tokens=2)
        per_chunk.append(
            {
                "chunk": ci,
                "seconds": elapsed,
                "ppl": c["ppl"],
                "kld_mean": c["kld_mean"],
                "same_top": c["same_top"],
            }
        )
        print(
            f"[accuracy]   chunk {ci + 1}/{args.chunks}: {elapsed:.1f}s  "
            f"PPL {c['ppl']:.4f}  KLD {c['kld_mean']:.5f}  "
            f"same top {100 * c['same_top']:.2f}%"
        )
    summary = stats.summary()
    print(format_summary(summary))
    return summary, per_chunk


def write_report(args, model_name, meta, summary, per_chunk):
    record = {
        # The example directory, the same key as <model>.perf.json's.
        "model": args.runner.rsplit(".", 1)[0].replace(".", "/"),
        "model_name": model_name,
        "runner": args.runner,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reference": {
            "id": meta["ref_model"],
            "kind": meta["ref_kind"],
            "revision": meta.get("ref_revision"),
        },
        "dataset": meta["dataset"],
        "dataset_sha256": meta["dataset_sha256"],
        "ctx": args.ctx,
        "chunks": args.chunks,
        "scored_from": args.first,
        "path": args.path,
        "npu_attn": args.npu_attn == "on",
        "metrics": summary,
        "per_chunk": per_chunk,
    }
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"accuracy_{args.path}_{args.ref}_{stamp}.json"
    json_path.write_text(json.dumps(record, indent=2))
    md_path = json_path.with_suffix(".md")
    md_path.write_text(
        f"# Accuracy: {model_name}\n\n"
        f"Reference `{meta['ref_model']}` ({meta['ref_kind']}), {meta['dataset']}, "
        f"ctx={args.ctx}, {summary['n_tokens']} scored tokens, {args.path} path.\n\n"
        "```\n" + format_summary(summary) + "\n```\n"
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(record, indent=2))
    print(f"[accuracy] report: {md_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runner", required=True, help="e.g. qwen3_0_6b.verify_adapter")
    p.add_argument("--model", default=None, help="adapter MODEL_CHOICES key or HF id")
    p.add_argument(
        "--ref",
        choices=["bf16", "dequant"],
        default="bf16",
        help="bf16: the original HF checkpoint. dequant: the adapter's "
        "build_hf_model, i.e. the same quantized weights the NPU runs, "
        "dequantized, so only implementation error remains.",
    )
    p.add_argument("--path", choices=["decode", "prefill"], default="decode")
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument(
        "--score",
        type=int,
        default=None,
        help="positions scored at the end of each chunk (default: the second "
        "half, as llama-perplexity). A long --ctx with a short --score "
        "measures accuracy at depth.",
    )
    p.add_argument(
        "--prefill-len",
        type=int,
        default=2048,
        help="longest prompt the NPU prefill takes; decode steps cover the "
        "rest of a longer chunk",
    )
    p.add_argument("--npu-attn", choices=["on", "off"], default="on")
    p.add_argument("--phase", choices=["auto", "ref", "npu"], default="auto")
    p.add_argument(
        "--cache-dir",
        default=os.environ.get("LLM_ACCURACY_CACHE", str(DEFAULT_CACHE)),
        help="where base (reference log-prob) files are kept",
    )
    p.add_argument("--report-dir", default=str(HERE / "reports"))
    p.add_argument("--json-out", default=None, help="also write the record here")
    args = p.parse_args()
    args.first = args.ctx // 2 if args.score is None else args.ctx - 1 - args.score

    adapter = _load_adapter(args.runner)
    model_name = adapter.resolve_model(args.model or adapter.DEFAULT_MODEL)
    hf_ref_model = (
        adapter.hf_reference(model_name)
        if hasattr(adapter, "hf_reference")
        else model_name
    )
    tokenizer = _get_tokenizer(model_name)
    _, dataset_sha = load_wikitext()
    meta = {
        "ref_model": hf_ref_model,
        "ref_kind": args.ref,
        "npu_model": model_name,
        "tokenizer": model_name,
        "dataset": DATASET,
        "dataset_sha256": dataset_sha,
        "ctx": args.ctx,
        "chunks": args.chunks,
        "first": args.first,
        "bos": model_bos(adapter, tokenizer),
    }
    # A dequant reference depends on the quantized weights, so key it on the
    # NPU model as well.
    key_model = model_name if args.ref == "dequant" else hf_ref_model
    # The cache is keyed on the weights too: the reference's revision and, for
    # a dequant reference, the quantized weights' source (the adapter's
    # *MODEL_SOURCE), so changed weights get a new base file, not a stale one.
    sources = [hf_ref_model]
    if args.ref == "dequant":
        sources += sorted(
            {
                v
                for k, v in vars(adapter).items()
                if k.endswith("MODEL_SOURCE") and isinstance(v, str)
            }
        )
    meta["weights"] = {src: _revision(src) for src in sources}
    if args.ref == "dequant":
        # Which tensors a dequant reference replaces is code, so a changed
        # hook gets a new base file too.
        hook = inspect.getsource(_dequant_builder(adapter))
        meta["hook_sha256"] = hashlib.sha256(hook.encode()).hexdigest()
    # The code that computes the reference is part of the key as well.
    ref_code = "".join(inspect.getsource(f) for f in (_longrope, _reference_logits))
    meta["reference_sha256"] = hashlib.sha256(ref_code.encode()).hexdigest()
    fingerprint = hashlib.sha256(
        json.dumps(
            {k: meta.get(k) for k in ("weights", "hook_sha256", "reference_sha256")},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:8]
    kind = args.ref
    if args.ref == "dequant" and _path_dependent(_dequant_builder(adapter)):
        kind = f"dequant-{args.path}"
    window = "" if args.first == args.ctx // 2 else f".s{args.ctx - 1 - args.first}"
    stem = (
        f"{key_model.replace('/', '--')}.{kind}.{DATASET}"
        f".c{args.ctx}{window}.n{args.chunks}.{fingerprint}"
    )
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_path = cache_dir / f"{stem}.kld"

    if args.phase in ("auto", "ref"):
        if base_path.exists() and args.phase == "auto":
            print(f"[accuracy] reusing base file {base_path}")
        elif args.phase == "ref":
            run_ref_phase(args, adapter, model_name, hf_ref_model, base_path, meta)
            return
        else:
            cmd = [sys.executable, sys.argv[0]] + sys.argv[1:] + ["--phase", "ref"]
            print("[accuracy] reference phase (subprocess)...")
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                sys.exit(rc)
    saved = json.loads(base_path.with_suffix(".json").read_text())
    saved.setdefault("first", saved["ctx"] // 2)  # files from before --score
    for k in ("dataset_sha256", "bos", "ref_model", "ref_kind", "first"):
        if saved.get(k) != meta[k]:
            raise SystemExit(
                f"{base_path}: {k} is {saved.get(k)!r}, expected {meta[k]!r}"
            )
    meta["ref_revision"] = saved.get("ref_revision")
    summary, per_chunk = run_npu_phase(args, adapter, model_name, base_path, meta)
    write_report(args, model_name, meta, summary, per_chunk)


if __name__ == "__main__":
    main()
