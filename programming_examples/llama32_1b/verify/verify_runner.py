# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""verify_runner.py — orchestrate the verify gate and the diagnosis lens.

Two modes selected by --prompts:

    --prompts topk_token  `make verify`   token-level top-k inclusion gate.
                                          NPU + HF bf16 only, lite mode
                                          runners, prompts × 32 greedy
                                          tokens, top-5 set inclusion.
                                          Method mirrors vLLM's
                                          check_logprobs_close. `make verify`
                                          caps at 2 prompts (~2 min, CI gate);
                                          `make verify-full` runs all prompts
                                          in the file (~6 min).

    --prompts single      `make diagnosis` inside-probing microscope. NPU + HF
                                          bf16 only, full-capture runners,
                                          one prompt's prefill, per-layer
                                          ffn_out cosine + max_abs (NPU vs
                                          HF) for layers 0..n_layers-2 plus
                                          the post-final-norm hidden as the
                                          L15 cell. No decode loop, no
                                          logits gate, no token match —
                                          `verify` already checks the
                                          user-visible output.
"""

from __future__ import annotations

import argparse
import functools
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure programming_examples (for `llama_kernel_builder`), project, and
# verify dirs are importable.
HERE = Path(__file__).parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT.parent))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))

from comparators import (
    compare_pair,
    compute_topk_set_check,
    topk_token_ids,
)
from report import Report
from runners.npu_runner import NpuRunner

DEFAULT_PROMPT = "The capital of France is"

# Same architecture (16 layers, emb=2048, n_heads=32, n_kv_heads=8,
# head_dim=64, vocab=128256) — only the weight tensors and tokenizer
# differ. base = original pretraining checkpoint (text continuation);
# instruct = what vLLM and other production stacks deploy.
MODEL_CHOICES = {
    "base": "meta-llama/Llama-3.2-1B",
    "instruct": "meta-llama/Llama-3.2-1B-Instruct",
}
BLOCK_PROBE = "ffn_out"

# Token-level top-k inclusion gate constants. Values mirror vLLM's
# check_logprobs_close defaults (max_tokens=32, num_logprobs=5).
PROMPTS_DIR = HERE / "prompts"
DEFAULT_PROMPTS_FILE = {
    "base": PROMPTS_DIR / "base.txt",
    "instruct": PROMPTS_DIR / "instruct.txt",
}
GATE_N_TOKENS = 32  # greedy tokens decoded per prompt
GATE_K = 5  # top-k inclusion threshold


def _load_weights(weights_mode: str, config, seed: int, model_name: str):
    from llama32_1b_weights import synthetic_weights, load_weights

    if weights_mode == "synthetic":
        return synthetic_weights(config, seed=seed)
    return load_weights(model_name, config=config)


@functools.lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    """Cached tokenizer loader. AutoTokenizer.from_pretrained is ~50 ms even
    when the files are local — pre-cache, we paid that 8 times per verify run."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _tokenize(prompt: str, model_name: str):
    tok = _get_tokenizer(model_name)
    ids = tok.encode(prompt)
    return np.array(ids, dtype=np.int64), tok


def _load_prompts(path: Path) -> list[str]:
    """Load prompts from a file; skip blank and '#' comment lines."""
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _md_escape(text: str) -> str:
    """Escape a tokenizer-decoded string for safe markdown-table embedding.
    Escapes the four sequences that would otherwise break the rendered
    cell: backslash, pipe (column separator), newline / cr / tab."""
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _decode_token_for_display(tokenizer, token_id: Optional[int]) -> Optional[str]:
    """Render one token ID as a quoted, escape-safe string for the report.
    Quoting keeps leading whitespace visible (most LLM tokens carry one)."""
    if token_id is None:
        return None
    return f'"{_md_escape(tokenizer.decode([int(token_id)]))}"'


def _generate_with_topk(runner, prompt_tokens: np.ndarray, n_tokens: int, k: int):
    """Free-run greedy decode capturing chosen token + top-k token IDs per step.

    Returns (chosen_tokens, topk_per_step) — both length n_tokens. The first
    entry is the prefill prediction; subsequent entries are decode-step
    predictions, each fed as input to the next step.

    Sanity check: each step's chosen token MUST equal the first entry of
    that step's top-k. If it does not, one of the runner's logit fields has
    been mutated between top1_token computation and the field being read
    here — print a loud warning so the rendered report is not misinterpreted
    as a real model disagreement.
    """

    def _check(step_idx, chosen_id, topk_ids, tag):
        if topk_ids and chosen_id != topk_ids[0]:
            print(
                f"[verify] WARN: {tag} step {step_idx} top1_token={chosen_id} "
                f"!= topk[0]={topk_ids[0]} (full top-{k}={topk_ids}). "
                "Indicates runner-side logit mutation between top1_token "
                "and lm_head_logits/logits_at_pred capture.",
                file=sys.stderr,
            )

    runner_tag = getattr(runner, "name", type(runner).__name__)
    pf = runner.prefill(prompt_tokens)
    chosen = [pf.top1_token]
    topk = [topk_token_ids(np.asarray(pf.logits_at_pred), k)]
    _check(0, pf.top1_token, topk[0], runner_tag)
    cur = len(prompt_tokens)
    next_tok = pf.top1_token
    for step_i in range(1, n_tokens):
        ds = runner.decode_step(next_tok, cur)
        chosen.append(ds.top1_token)
        step_topk = topk_token_ids(np.asarray(ds.lm_head_logits), k)
        topk.append(step_topk)
        _check(step_i, ds.top1_token, step_topk, runner_tag)
        cur += 1
        next_tok = ds.top1_token
    return chosen, topk


def _run_diagnosis(npu, hf, prompt_tokens, report, n_layers):
    """Diagnosis lens: per-layer ffn_out (NPU vs HF bf16) for one prompt.

    For layers 0..n_layers-2 we compare each runner's raw layer output
    (npu.layer_intermediates[li]['ffn_out'] vs hf.layer_intermediates[li]
    ['ffn_out']). For the last layer we compare each runner's
    final_hidden_normed (the post-final-RMSNorm hidden state that feeds
    LM-head) — HF's hidden_states[n_layers] is post-norm by HF v5.3
    convention, and NPU exposes the equivalent via the same final_norm
    application it does inside the production LM-head GEMV path.

    Diagnosis is informational only — no thresholds, no pass/fail. Inspect
    the cosine table by hand; the verify gate is the actual correctness
    signal.
    """
    print("[diagnosis] prefill: NPU + HF...")
    npu_pf = npu.prefill(prompt_tokens)
    hf_pf = hf.prefill(prompt_tokens)
    print("[diagnosis] comparing per-layer ffn_out (NPU vs HF bf16)...")
    for li in range(n_layers - 1):
        report.add(
            compare_pair(
                name=BLOCK_PROBE,
                npu=npu_pf.layer_intermediates[li][BLOCK_PROBE],
                hf=hf_pf.layer_intermediates[li][BLOCK_PROBE],
                layer=li,
            )
        )
    report.add(
        compare_pair(
            name=BLOCK_PROBE,
            npu=npu_pf.final_hidden_normed,
            hf=hf_pf.final_hidden_normed,
            layer=n_layers - 1,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npu-attn", choices=["on", "off"], default="on")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--weights", choices=["hf", "synthetic"], default="hf")
    p.add_argument(
        "--model",
        choices=list(MODEL_CHOICES),
        default="instruct",
        help="Llama-3.2-1B checkpoint. Default 'instruct' matches what "
        "production stacks deploy. 'base' is the original pretraining "
        "checkpoint (text continuation).",
    )
    p.add_argument("--report-dir", default=str(HERE / "reports"))
    p.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable hard exit on FAIL (default: exit 1 on FAIL)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompts",
        choices=["single", "topk_token"],
        default="single",
        help="'single' (used by `make diagnosis`) probes per-layer ffn_out "
        "for one prompt. 'topk_token' (used by `make verify`) runs the "
        "top-k token-level inclusion gate over the prompts file (capped "
        "by --max-prompts). The two modes are exclusive.",
    )
    p.add_argument(
        "--prompts-file",
        default=None,
        help="Override the prompt file used by --prompts topk_token. "
        "Defaults to verify/prompts/{model}.txt.",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Cap the number of prompts run in --prompts topk_token mode. "
        "Default: run all prompts in the file. `make verify` uses 2 (fast "
        "CI gate); `make verify-full` uses the full set.",
    )
    args = p.parse_args()

    # The HF reference runner always loads the real checkpoint, so verifying
    # against synthetic NPU weights compares apples to oranges and would
    # always FAIL. Reject up front rather than emit meaningless reports.
    if args.weights == "synthetic":
        print(
            "[verify] --weights synthetic is not supported: the HF reference "
            "runner only loads real checkpoints, so the comparison would be "
            "meaningless. Use --weights hf.",
            file=sys.stderr,
        )
        sys.exit(2)

    from llama32_1b_weights import LlamaConfig

    config = LlamaConfig()
    model_name = MODEL_CHOICES[args.model]
    weights = _load_weights(args.weights, config, args.seed, model_name)
    # Production prefill kernels are tiled for seq_len=2048; NpuRunner pads
    # short prompts internally.
    max_seq = 2048

    in_verify_mode = args.prompts == "topk_token"
    report = Report(
        config={
            "mode": "verify" if in_verify_mode else "diagnosis",
            "weights": args.weights,
            "model": args.model,
            "model_name": model_name,
            "npu_attn": args.npu_attn == "on",
            "prompt": args.prompt if not in_verify_mode else None,
        }
    )

    # ---- Build runners ----
    # Both modes use NPU + HF bf16 only. Verify runs lite (no per-layer
    # capture); diagnosis runs full-capture for the per-layer probe.
    lite = in_verify_mode
    print(f"[verify] mode = {report.config['mode']}, lite={lite}")
    print("[verify] building NPU runner...")
    npu = NpuRunner(
        weights,
        config,
        max_seq=max_seq,
        tokenizer=_get_tokenizer(model_name),
        npu_attn=(args.npu_attn == "on"),
        lite_mode=lite,
    )
    from runners.hf_runner import HfRunner

    print(f"[verify] building HF runner ({model_name}, lite={lite}, may download)...")
    try:
        hf = HfRunner(
            model_name=model_name,
            config=config,
            max_seq=max_seq,
            lite_mode=lite,
        )
    except Exception as e:
        print(f"[verify] HF runner unavailable: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Diagnosis path: single prompt, per-layer ffn_out only ----
    if not in_verify_mode:
        prompt_tokens, _ = _tokenize(args.prompt, model_name)
        # NpuRunner truncates to max_seq internally; truncate here too so the
        # HF runner sees the same context and the per-layer diagnosis stays
        # apples-to-apples on long prompts.
        if len(prompt_tokens) > max_seq:
            prompt_tokens = prompt_tokens[:max_seq]
        _run_diagnosis(npu, hf, prompt_tokens, report, config.n_layers)
        Path(args.report_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        json_path = Path(args.report_dir) / f"diagnosis_{stamp}.json"
        md_path = Path(args.report_dir) / f"diagnosis_{stamp}.md"
        report.dump_json(json_path)
        report.dump_markdown(md_path)
        print(f"\n[verify] Report: {md_path}")
        print(f"[verify] JSON:   {json_path}")
        print(f"[verify] Summary: {report.summary()}")
        if report.has_failure() and not args.no_strict:
            print("[verify] FAIL — see report for details.", file=sys.stderr)
            sys.exit(1)
        print("[verify] PASS")
        return

    # ---- Verify path: top-k token-level inclusion gate over prompts file ----
    prompts_path = (
        Path(args.prompts_file)
        if args.prompts_file
        else DEFAULT_PROMPTS_FILE[args.model]
    )
    prompts = _load_prompts(prompts_path)
    if args.max_prompts is not None and args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    report.set_prompts(prompts)
    report.config["prompts_file"] = str(prompts_path)
    report.config["max_prompts"] = args.max_prompts
    print(
        f"[verify] top-k token gate: {len(prompts)} prompts × "
        f"{GATE_N_TOKENS} tokens, k={GATE_K} (from {prompts_path.name})"
    )
    for pi, prompt in enumerate(prompts):
        short = (prompt[:60] + "…") if len(prompt) > 60 else prompt
        print(f"[verify] prompt {pi + 1}/{len(prompts)}: {short!r}")
        ptoks, tokenizer = _tokenize(prompt, model_name)
        # Same context for both runners — see the diagnosis path above.
        if len(ptoks) > max_seq:
            ptoks = ptoks[:max_seq]
        print(f"[verify]   NPU greedy decode ({GATE_N_TOKENS} tokens)...")
        npu_chosen, npu_topk = _generate_with_topk(npu, ptoks, GATE_N_TOKENS, GATE_K)
        print(f"[verify]   HF greedy decode ({GATE_N_TOKENS} tokens)...")
        hf_chosen, hf_topk = _generate_with_topk(hf, ptoks, GATE_N_TOKENS, GATE_K)

        def _decorate(rec, test_seq):
            """Inject decoded text into the record:
            - the two chosen tokens at divergence (with rank context)
            - the agreed prefix (the tokens both runners produced
              identically before divergence) — empty string when
              divergence_step == 0.
            """
            rec.test_chosen_text_at_div = _decode_token_for_display(
                tokenizer, rec.test_chosen_at_div
            )
            rec.ref_chosen_text_at_div = _decode_token_for_display(
                tokenizer, rec.ref_chosen_at_div
            )
            if rec.divergence_step is not None and rec.divergence_step > 0:
                prefix_ids = [int(t) for t in test_seq[: rec.divergence_step]]
                rec.agreed_prefix_text = f'"{_md_escape(tokenizer.decode(prefix_ids))}"'
            elif rec.divergence_step == 0:
                rec.agreed_prefix_text = '""'
            return rec

        rec = compute_topk_set_check(
            test_chosen=npu_chosen,
            test_topk=npu_topk,
            ref_chosen=hf_chosen,
            ref_topk=hf_topk,
            k=GATE_K,
            prompt_idx=pi,
            prompt_text=short,
        )
        report.add_topk(pair="npu_vs_hf", record=_decorate(rec, npu_chosen))

    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = Path(args.report_dir) / f"verify_topk_token_{stamp}.json"
    md_path = Path(args.report_dir) / f"verify_topk_token_{stamp}.md"
    report.dump_json(json_path)
    report.dump_markdown(md_path)
    print(f"\n[verify] Report: {md_path}")
    print(f"[verify] JSON:   {json_path}")
    print(f"[verify] Summary: {report.summary()}")
    if report.has_failure() and not args.no_strict:
        print("[verify] FAIL — see report for details.", file=sys.stderr)
        sys.exit(1)
    print("[verify] PASS")


if __name__ == "__main__":
    main()
