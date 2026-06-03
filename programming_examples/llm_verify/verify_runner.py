# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""verify_runner.py — shared verify gate + diagnosis lens for any LLM example.

Adapter-driven: each model's `verify_adapter.py` (under its programming_example
dir) exports `build_runner(...)`, `build_config()`, `MODEL_CHOICES`, and
`resolve_model(...)`. Pick one with `--runner=<dotted.module.path>`; the rest
of this driver is model-agnostic and reuses the same HF reference runner,
comparators, and report code across all LLMs.

Two modes (mutually exclusive, selected by --prompts):

    --prompts topk_token  `make verify`   token-level top-k inclusion gate.
                                          Multi-prompt × 32 greedy tokens.
                                          Lite-mode runners (no per-layer
                                          capture). Method mirrors vLLM's
                                          check_logprobs_close.

    --prompts single      `make diagnosis` inside-probing microscope. Single
                                          prompt's prefill, per-layer
                                          ffn_out cosine + max_abs (NPU vs
                                          HF) for layers 0..n_layers-2 plus
                                          the post-final-norm hidden as the
                                          last-layer cell. No decode loop.

Example invocations:
    python verify_runner.py --runner=llama32_1b.verify_adapter \\
        --prompts topk_token --max-prompts 2
    python verify_runner.py --runner=llama32_1b_int4.verify_adapter \\
        --prompts topk_token --max-prompts 2
"""

from __future__ import annotations

import argparse
import functools
import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).parent
_PROG_EXAMPLES = HERE.parent
sys.path.insert(0, str(_PROG_EXAMPLES))
sys.path.insert(0, str(HERE))

from comparators import (  # noqa: E402
    compare_pair,
    compute_topk_set_check,
    topk_token_ids,
)
from report import Report  # noqa: E402

DEFAULT_PROMPT = "The capital of France is"
BLOCK_PROBE = "ffn_out"

# Prompt sets live under llm_verify/prompts/; pick by `--prompt-style`
# (`instruct` for chat-tuned models, `base` for completion checkpoints).
PROMPTS_DIR = HERE / "prompts"
DEFAULT_PROMPTS = {
    "instruct": PROMPTS_DIR / "instruct.txt",
    "base": PROMPTS_DIR / "base.txt",
}
GATE_N_TOKENS = 32  # greedy tokens decoded per prompt
GATE_K = 5  # top-k inclusion threshold


@functools.lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    """Cached tokenizer loader. AutoTokenizer.from_pretrained is ~50 ms even
    when local."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _tokenize(prompt: str, model_name: str):
    tok = _get_tokenizer(model_name)
    ids = tok.encode(prompt)
    return np.array(ids, dtype=np.int64), tok


def _load_prompts(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _md_escape(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def _decode_token_for_display(tokenizer, token_id: Optional[int]) -> Optional[str]:
    if token_id is None:
        return None
    return f'"{_md_escape(tokenizer.decode([int(token_id)]))}"'


def _generate_with_topk(runner, prompt_tokens: np.ndarray, n_tokens: int, k: int):
    """Free-run greedy decode capturing chosen token + top-k token IDs per
    step. The runner-integrity check below catches accidental mutation of
    logit arrays between top1_token computation and the field being read."""

    def _check(step_idx, chosen_id, topk_ids, tag):
        if topk_ids and chosen_id != topk_ids[0]:
            print(
                f"[verify] WARN: {tag} step {step_idx} top1_token={chosen_id} "
                f"!= topk[0]={topk_ids[0]} (full top-{k}={topk_ids}). "
                "Indicates runner-side logit mutation between top1_token "
                "and lm_head_logits capture.",
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
    """Per-layer ffn_out (NPU vs HF) for one prompt. Informational only —
    inspect the cosine table by hand; the verify gate is the actual
    correctness signal."""
    print("[diagnosis] prefill: NPU + HF...")
    npu_pf = npu.prefill(prompt_tokens)
    hf_pf = hf.prefill(prompt_tokens)
    print("[diagnosis] comparing per-layer ffn_out (NPU vs HF bf16)...")
    for li in range(n_layers - 1):
        npu_li = npu_pf.layer_intermediates[li].get(BLOCK_PROBE)
        hf_li = hf_pf.layer_intermediates[li].get(BLOCK_PROBE)
        if npu_li is None or hf_li is None or npu_li.size == 0 or hf_li.size == 0:
            # Adapter doesn't expose this layer's intermediate — skip rather
            # than poison the cosine table.
            continue
        report.add(compare_pair(name=BLOCK_PROBE, npu=npu_li, hf=hf_li, layer=li))
    if npu_pf.final_hidden_normed.size and hf_pf.final_hidden_normed.size:
        report.add(
            compare_pair(
                name=BLOCK_PROBE,
                npu=npu_pf.final_hidden_normed,
                hf=hf_pf.final_hidden_normed,
                layer=n_layers - 1,
            )
        )


def _load_adapter(dotted_path: str):
    """Import the model's verify_adapter module by dotted path.

    Adapter modules live next to each LLM example; we make
    `programming_examples/` importable above so `<example_dir>.verify_adapter`
    resolves natively.
    """
    return importlib.import_module(dotted_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runner",
        required=True,
        help="Dotted path to the model's verify adapter (e.g. "
        "'llama32_1b.verify_adapter', 'llama32_1b_int4.verify_adapter').",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Adapter's MODEL_CHOICES key (e.g. 'instruct', 'base') OR a "
        "raw HF id / local path. Defaults to adapter.DEFAULT_MODEL.",
    )
    p.add_argument(
        "--prompt-style",
        choices=list(DEFAULT_PROMPTS),
        default="instruct",
        help="Which prompts file to use (selects from llm_verify/prompts/).",
    )
    p.add_argument("--npu-attn", choices=["on", "off"], default="on")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
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
        "multi-prompt top-k token-level inclusion gate.",
    )
    p.add_argument(
        "--prompts-file",
        default=None,
        help="Override the prompt file used by --prompts topk_token. "
        "Defaults to llm_verify/prompts/{prompt-style}.txt.",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Cap the number of prompts run in --prompts topk_token mode. "
        "Default: run all prompts in the file.",
    )
    args = p.parse_args()

    adapter = _load_adapter(args.runner)
    model_choice = args.model if args.model is not None else adapter.DEFAULT_MODEL
    model_name = adapter.resolve_model(model_choice)
    hf_ref_model = (
        adapter.hf_reference(model_name)
        if hasattr(adapter, "hf_reference")
        else model_name
    )

    config = adapter.build_config()
    max_seq = 2048  # Production prefill kernels are tiled for seq_len=2048.

    in_verify_mode = args.prompts == "topk_token"
    report = Report(
        config={
            "mode": "verify" if in_verify_mode else "diagnosis",
            "runner": args.runner,
            "model": model_choice,
            "model_name": model_name,
            "npu_attn": args.npu_attn == "on",
            "prompt": args.prompt if not in_verify_mode else None,
        }
    )

    lite = in_verify_mode
    print(
        f"[verify] adapter={args.runner}, model={model_name}, "
        f"hf_ref={hf_ref_model}, mode={report.config['mode']}, lite={lite}"
    )
    tokenizer = _get_tokenizer(model_name)
    # The HF runner needs its own tokenizer instance (matched to the HF ref
    # model). For bf16 this is the same checkpoint as model_name; for int4
    # adapter overrides hf_reference() to the upstream un-quantized model.
    hf_tokenizer = _get_tokenizer(hf_ref_model)
    print("[verify] building NPU runner via adapter...")
    npu = adapter.build_runner(
        model_name=model_name,
        config=config,
        max_seq=max_seq,
        tokenizer=tokenizer,
        npu_attn=(args.npu_attn == "on"),
        lite_mode=lite,
    )
    from runners.hf_runner import HfRunner

    # Optional adapter hook: produce a custom HF model (e.g. meta-llama
    # architecture with AWQ-dequantized weights patched in) so the verify
    # gate isolates NPU drift from quantization error. When absent, fall
    # back to plain `from_pretrained(hf_ref_model)`.
    hf_model = None
    if hasattr(adapter, "build_hf_model"):
        print(f"[verify] adapter is building custom HF reference model...")
        hf_model = adapter.build_hf_model(
            npu_model_name=model_name,
            hf_ref_model=hf_ref_model,
            config=config,
        )
    print(f"[verify] building HF runner ({hf_ref_model}, lite={lite}, may download)...")
    try:
        hf = HfRunner(
            model_name=hf_ref_model,
            config=config,
            max_seq=max_seq,
            lite_mode=lite,
            model=hf_model,
        )
    except Exception as e:
        print(f"[verify] HF runner unavailable: {e}", file=sys.stderr)
        sys.exit(1)

    if not in_verify_mode:
        prompt_tokens, _ = _tokenize(args.prompt, model_name)
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

    prompts_path = (
        Path(args.prompts_file)
        if args.prompts_file
        else DEFAULT_PROMPTS[args.prompt_style]
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
        if len(ptoks) > max_seq:
            ptoks = ptoks[:max_seq]
        print(f"[verify]   NPU greedy decode ({GATE_N_TOKENS} tokens)...")
        npu_chosen, npu_topk = _generate_with_topk(npu, ptoks, GATE_N_TOKENS, GATE_K)
        print(f"[verify]   HF greedy decode ({GATE_N_TOKENS} tokens)...")
        hf_chosen, hf_topk = _generate_with_topk(hf, ptoks, GATE_N_TOKENS, GATE_K)

        def _decorate(rec, test_seq):
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
