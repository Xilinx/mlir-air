# Llama-3.2-1B verification

Two ways to look at the production Llama-3.2-1B NPU2 inference pipeline,
both comparing against HuggingFace transformers in **bf16** (same dtype
as NPU — fair fight). Companion doc: `../docs/VERIFICATION.html`.

Targets live in the parent Makefile (`programming_examples/llama32_1b/Makefile`):

```
cd programming_examples/llama32_1b

make verify [MODEL=instruct|base]                # ~4 min — top-k token-level correctness gate
make diagnosis [MODEL=...] [PROMPT="..."]        # ~3 min — per-layer cosine, informational
make clean                                       # rm build_*/ + verify/reports/
```

## `make verify` — the correctness gate

Top-k token-level inclusion check (mirrors vLLM's
`check_logprobs_close` in `tests/models/utils.py`). For each prompt in the
selected set:

1. NPU and HF each greedy-decode 32 tokens, capturing top-5 token IDs per step.
2. Walk in lockstep. On the first step where chosen tokens differ, both
   sides' chosen tokens must appear in the OTHER side's top-5; otherwise
   FAIL. Stop walking after first divergence.
3. All prompts in the run must pass. `verify_runner.py` exits 1 on any FAIL,
   exit 0 on PASS.

`make verify` runs **2 prompts** (fast CI gate); `make verify-full` runs the
full set (currently 8). Both are pass/fail; use `verify-full` locally for
exhaustive validation.

This is the only correctness signal. The discrete top-k judgment is
robust to the bf16 ULP noise that fluctuates continuous metrics like
cosine, while still catching every real implementation regression.

Configuration:
- **NPU FlashAttention is on** (`--npu-attn on` is the default) — verify
  exercises the full NPU end-to-end production path: GEMV + RMSNorm +
  RoPE + FlashAttention + LM-head GEMV.
- **Lite-mode runners**: skip per-layer intermediate capture, KV-cache
  copies, and the CPU-side full-sequence LM-head recompute. Only the
  per-step top-1 token + top-5 logits are read.
- **Tokenizer cached** via `functools.lru_cache` (no per-prompt reload).
- **MODEL=instruct** (default) uses `meta-llama/Llama-3.2-1B-Instruct`
  with `prompts/instruct.txt` (instruction-style prompts).
- **MODEL=base** uses `meta-llama/Llama-3.2-1B` with `prompts/base.txt`
  (continuation-style prompts matched to the base checkpoint's behavior).

## `make diagnosis` — the inside-probing lens

Reach for this when verify flags an issue and you need to localize.

For one prompt, runs prefill on NPU + HF and reports per-position cosine
+ element-wise abs error for each layer's `ffn_out` (the block output).
Layers 0..n_layers-2 use each runner's raw layer output; the last layer
uses each runner's post-final-RMSNorm hidden state (HF exposes
`hidden_states[n_layers]` as post-norm by HF v5.3 convention; NPU
produces the equivalent via the final_norm step inside its production
LM-head GEMV path).

**Diagnosis is informational only — it never fails the run.** The
verify gate is the correctness signal. The cosine table tells you where
the NPU implementation drifts most from HF (which layer, by how much),
which is what you want when triaging a real verify failure or weighing
a kernel-side optimization. Inspect the table by hand.

Defaults to `--npu-attn on` so the inside-probing exercises the same
end-to-end NPU production path verify gates against. Diagnosis only
probes `ffn_out` (the block output), not `attn_out`, so the previous
runner-side per-layer attn_out reshape bug under `--npu-attn on` does
not affect this lens.

## Output

Each run writes a timestamped pair of files in `reports/`:

- **verify**: `verify_topk_token_YYYYMMDD-HHMMSS.{json,md}` — Prompts table +
  per-prompt top-k inclusion table with agreed-prefix sub-lines.
- **diagnosis**: `diagnosis_YYYYMMDD-HHMMSS.{json,md}` — single
  per-layer cosine + max_abs table.

`reports/` is gitignored.

## Memory

Real-weight runs need ~5 GB for the HF model + project numpy weights
shared by the NPU runner. Plan for ~6-8 GB working set.

## File map

| File | What |
|---|---|
| `verify_runner.py` | CLI orchestrator — picks `verify` vs `diagnosis` by `--prompts` |
| `comparators.py` | `compare_pair` (cosine + max_abs), `compute_topk_set_check` (top-k token-level), `topk_token_ids` |
| `report.py` | `Report` accumulator + JSON / markdown dumpers |
| `runners/npu_runner.py` | NPU production prefill + decode wrapper |
| `runners/hf_runner.py` | HuggingFace transformers bf16 wrapper |
| `runners/_records.py` | `PrefillRecord` / `DecodeStepRecord` dataclasses |
| `prompts/instruct.txt` | 8 instruction-style prompts (verify MODEL=instruct) |
| `prompts/base.txt` | 8 continuation-style prompts (verify MODEL=base) |
