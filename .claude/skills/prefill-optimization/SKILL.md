---
name: prefill-optimization
description: Phase 4 of LLM deployment — apply known prefill optimization patterns to a Phase-3-correct pipeline (multi-launch merge, BO pre-loading, intermediate buffer reuse, seq-first layout, CPU→NPU op promotion). Each step preserves correctness by re-running the Phase 3 gate — `make verify` (token-set vs HF bf16) is the PASS/FAIL gate; `make diagnosis` per-layer cosine is the informational lens used to localize a regression. Invoked after Phase 3 PASS.
---

## Purpose

Phase 1-3 produced a functionally correct NPU pipeline (kernel-by-kernel
in Phase 1, layer-by-layer in Phase 2-3, all numerically aligned with
the HF bf16 reference). Phase 4 keeps that correctness while applying
optimization patterns to **reduce prefill latency**. These patterns —
ELF-merging, host↔NPU runtime-overhead reduction, host-side layout — are
things you **compose from the kernels**, not behaviors inherited from a
reference; the reference's builders are worked examples of each. Every
pattern is an experiment: apply, re-measure, re-run Phase 3 gate; revert
if correctness regresses.

For scale, the reference deployment llama3.2-1B took prefill from
18.67 s → 1.30 s (14×) by composing the 4 categories below — an
illustrative datapoint for what the patterns buy on one model, not a
target every deployment must hit.

## Phase 4 PASS criteria (HARD GATES)

1. **Correctness preserved**: after every applied pattern, **`make verify`
   (the token-set gate vs HF bf16) still PASSES** — this is the Phase 3
   correctness gate, re-run between patterns. `make diagnosis` per-layer
   cosine is NOT a gate (the verify subsystem retired threshold-based
   diagnosis; `compare_pair` reports cosine with no pass/fail); run it only
   to **localize** a regression when verify breaks (which pattern / which
   layer the cosine cliff appears at). If `make verify` regresses to FAIL,
   **revert the pattern** and document why it doesn't apply to this model.
2. **Prefill kernel time strictly < Phase 3 baseline**, measured at
   the same canonical prompt + seq_len with 5-warmup + 20-iter
   profile. Recording wall time too (kernel + host overhead) is good
   but the gate is on kernel time (host overhead optimization is a
   separate concern).
3. **Per-pattern outcome documented** in
   `<model>/docs/development_progress/phase4_prefill.md`: for each of
   the 4 patterns, record `applied / skipped / reverted`, the latency
   delta, and a one-line reason.

The "≥ N patterns applied" check is NOT a gate — some models
legitimately need only 1-2 patterns (e.g., the model is already
seq-first by construction → Pattern C N/A; no CPU fallback exists →
Pattern D N/A). The gate is the outcome (perf improved + correctness
preserved), not the process count.

## Knowledge base references

PRIMARY:

- `programming_examples/llms/llama32_1b/docs/profile.md` — the reference
  deployment's profiling breakdown; the reference for what "good" looks like
- `programming_examples/llms/<model>/docs/development_progress/phase3_full.md`
  — Phase 3 baseline timings + cosine numbers (the "before" state to
  measure against and preserve)
- `programming_examples/llms/llama_kernel_builder/cache.py` —
  `static_input_indices`, `intermediate_indices` mechanics (the
  `KernelCache` host-optimization knobs)

REFERENCE EXEMPLARS (read/mirror to compose your own fused ELFs; import
directly only on a bit-for-bit kernel-sequence match):

- `programming_examples/llms/llama32_1b/multi_launch_builder/` — the full set
  of fused-ELF builders, the worked example of how registry leaf kernels
  stitch into multi-launch ELFs. Mirror these for your model's kernel
  sequence. Two representative ones:
  - `rms_gemms_rope_multi.py` — fused 6-launch ELF for RMSNorm + Q/K/V GEMM + RoPE Q/K
  - `o_ffn_multi.py` — fused 8-launch ELF for O + add + RMSNorm + Gate/Up + SwiGLU + Down + add
- `programming_examples/llms/llama_kernel_builder/` — the shared toolkit
  (KernelCache, stitching, external_kernels) every fused-ELF build uses,
  inheritance or kernel-first alike.

## Workflow

### Step 1: Measure Phase 3 baseline

Before applying any pattern, capture the baseline prefill kernel time:

```bash
cd programming_examples/llms/<model>
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile
```

Record: kernel time (ms) + wall time. Also note the Phase 3 per-layer
cosine table as a baseline to compare against *if* a later pattern breaks
verify (it's the localization reference, not a gate).

### Step 2: Apply optimization patterns

Apply A → B → C → D. Between each, re-run the Phase 3 gate (`make verify`)
and re-measure profile. Skip a pattern if the trigger condition isn't met
(skip reasons go in the log).

#### Pattern A — Multi-launch merge (the dominant win)

Stitch each leaf kernel's `air.launch` into a single `func.func` so
the host issues ONE `xrt.run()` per group instead of one per kernel.
Llama3 prefill went from 16-XRT-call/layer to 3-XRT-call/layer here.

Two paths — the **same kernel-first-vs-inheritance decision Phase 2 already
made** (kernel-first is the default; the bit-for-bit match rule + its a/b/c
breakers are defined once in `single-block-validation` Step 1 — don't
restate them, reuse Phase 2's verdict):

| Path | When | What to do |
|---|---|---|
| **Build fused ELF (default)** | Phase 2 integrated kernel-first (the general path) | Write the model's builders in `<model>/multi_launch_builder/`, mirroring `llama32_1b/multi_launch_builder/*` as the worked exemplar. Invoke `merge-multi-launch-kernels` skill for the procedural recipe; on compile failure invoke `debug-multi-launch-merge`. |
| **Reuse existing fused ELF (shortcut)** | Phase 2 took the inheritance shortcut (bit-for-bit llama match) | Import `llama32_1b/multi_launch_builder/{rms_gemms_rope_multi, o_ffn_multi}` directly; pass new shape parameters. Zero ELF-building work — the llama-variant fast path. |

Either way, expected gain: 2–4× XRT-call reduction → corresponding
wall-clock improvement.

**Sub: registry-driven GEMM config (no hardcoded tiles).** When your fused
ELF contains GEMMs, do NOT hand-copy tile sizes into the builder — look
them up from the registry so there is one source of truth (this removed a
real drift bug where Gate/Up `tile_k_l2` was stale at 64 vs the
registry-best 256). Call
`kernel_registry/registry_lookup.gemm_config(M, K, N, output_dtype,
precision)` at build time; it returns `{method, tile, gflops,
mean_rel_L1}` for the best measured config and **raises for an unmeasured
shape** (run a sweep + record it in Phase 1 first — no silent guess). For
BF16-out prefill GEMMs use `precision="high"` (the FP32-accumulate
fused-cast / drain path, GPU-standard ~9.3e-3); `gemm_config` picks
fused-cast vs drain per shape. `llama32_1b`'s `gemm_builder.py` +
`multi_launch_builder/{o_ffn_multi, rms_gemms_rope_multi}.py` are the worked
example (mixed fused-cast tile_m=64 + drain tile_m=32 co-linked in one ELF
via distinct symbol suffixes).

#### Pattern B — Eliminate redundant host↔NPU data movement

**B1. Per-layer BO pre-loading** — weights are written ONCE during
setup, reused on every layer call:

- Allocate per-layer BOs using `bo_key=f"kernel_L{layer_idx}"`
- Write weights in `prepare_runtime()` (not on every prefill call)
- Pass `static_input_indices=[<weight_indices>]` on every
  `cache.load_and_run()` to skip re-write

**B2. Intermediate buffer reuse** — for buffers the kernel fully
overwrites (its outputs and scratch):

- Pass `intermediate_indices=[<output_indices>]` on
  `cache.load_and_run()` to skip the initial host write

Llama3 observed multi-second prefill savings from B1 alone.

#### Pattern C — Seq-first activation layout

If the kernels in your block accept seq-first `(seq, n_heads*head_dim)`
natively, eliminate any host-side transposes between RoPE / FlashAttention.

- RoPE: accept seq-first input
- FlashAttention: accept seq-first Q, K, V (this is `attn_npu2_seqfirst.py`
  for head_dim ≤ 64; for head_dim ≥ 128, a head-first wrapper does the
  host transpose precisely so the rest of the pipeline stays seq-first —
  see `debug-fa-runtime-failure` for why head_dim ≥ 128 must route this way)

Skip Pattern C if the model already runs seq-first end-to-end (most
inheritance deployments do).

Expected gain: eliminate 1–4 host transposes per layer.

#### Pattern D — CPU→NPU op promotion

If anything in the prefill pipeline still falls back to CPU (e.g.,
residual add wrapped in `np.add`, intermediate small-shape RMSNorm,
host-side `softmax`), move it to NPU using the standalone harness
that Phase 1 already validated.

Triggering condition: profile shows host overhead per layer ≥ kernel
time per layer. If kernel time dominates, skip.

> **head_dim ≥ 128 FA caveat**: NPU FlashAttention has known compile-
> flag and runtime quirks at head_dim ≥ 128. The head-first wrapper
> auto-routes the attention call through the head-first kernel via host
> transposes. If FA hangs or produces NaN, invoke
> **`debug-fa-runtime-failure`** — it bisects the three known root
> causes. Do NOT debug FA inline in this skill.

### Step 3: Re-run Phase 3 gate after each pattern

After every applied (or attempted) pattern, re-run the gate:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify      # GATE: token-set, exit 1 on FAIL
```

`make verify` PASS is the correctness gate. If it regresses to FAIL,
revert the pattern and document why.

Only when verify FAILs, run diagnosis to **localize** the break:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make diagnosis   # informational: per-layer cosine table
```

A cosine cliff at layer *i* points at the pattern's broken assumption
there (e.g. a transpose Pattern C removed that this model actually
needed). diagnosis does not PASS/FAIL — it is the microscope, verify is
the gate.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Multi-launch merge compile fails (BD exhaustion, channel routing, herd shape conflict) | non-1024-aligned dim (see the kernel's `details/<Kernel>_bf16.md`) OR wrong stitching boundary | Invoke `debug-multi-launch-merge` — it discriminates the 4 known compile blockers |
| Output corruption after BO pre-loading (correct first call, NaN/garbage on subsequent calls) | Per-layer BO key collision OR `static_input_indices` set wrong | Invoke `debug-bo-corruption` |
| FA hang (`ERT_CMD_STATE_TIMEOUT`) at head_dim ≥ 128 | Seq-first `dk_chunks > 1` path bug | Invoke `debug-fa-runtime-failure`; head-first wrapper is the workaround (see Pattern D caveat) |
| FA all-NaN at runtime | Compile-flag mismatch on `attn_npu2.cc` macros (LESSON 3 — `-Dlqp` must be per-tile, not per-launch) | Invoke `debug-fa-runtime-failure`; `compile_attn_npu2_split` derives correct flags |
| Cosine drops after Pattern X | Pattern X has a layout/type assumption your model violates | Revert Pattern X; check whether the assumption (e.g., seq-first only, all weights pre-transposed) holds |
| Latency unchanged after Pattern A | Multi-launch ELF compiled but XRT call count didn't drop | Check `xrt-smi top` for actual call count; verify the new fused ELF is what `_run_cached` actually invokes (not falling back to per-kernel path) |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 4 PASS:

- `<model>/docs/development_progress/phase4_prefill.md`: per-pattern
  table with `applied / skipped / reverted`, latency delta, reason
- `<model>/TODO.md`: mark Phase 4, append final prefill kernel time +
  speedup vs Phase 3 baseline
- If a new fused ELF was built (kernel-first path of Pattern A),
  surface to Phase 6 for potential promotion to a shared location if a
  second deployment validates the same pattern
