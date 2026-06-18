---
name: phase-4-prefill-optimization
description: Phase 4 of LLM deployment — apply the shared optimization skillset to a Phase-3-correct prefill pipeline (multi-launch merge, BO pre-loading + intermediate buffer reuse, seq-first layout). Thin orchestrator that dispatches `merge-multi-launch-kernels`, `buffer-object-reuse`, and `layout-alignment`. Each step preserves correctness by re-running the Phase 3 gate — `make verify` (token-set vs HF bf16) is the PASS/FAIL gate; `make diagnosis` per-layer cosine is the informational lens used to localize a regression. Invoked after Phase 3 PASS.
---

## Purpose

Phase 1-3 produced a functionally correct NPU pipeline (kernel-by-kernel
in Phase 1, layer-by-layer in Phase 2-3, all numerically aligned with
the HF bf16 reference). Phase 4 keeps that correctness while applying
the shared optimization skillset to **reduce prefill latency**. This phase
is a thin orchestrator: it dispatches three now-independent optimization
skills — `merge-multi-launch-kernels` (ELF-merging),
`buffer-object-reuse` (host↔NPU runtime-overhead reduction), and
`layout-alignment` (host-side layout) — each of which owns its own recipe
and failure modes. These wins are things you **compose from the kernels**,
not behaviors inherited from a reference; the reference's builders are
worked examples. Every optimization is an experiment: apply, re-measure,
re-run Phase 3 gate; revert if correctness regresses.

For scale, the reference deployment llama3.2-1B took prefill from
18.67 s → 1.30 s (14×) by composing these optimizations — an
illustrative datapoint for what they buy on one model, not a target every
deployment must hit.

## Phase 4 PASS criteria (HARD GATES)

1. **Correctness preserved**: after every applied optimization, **`make verify`
   (the token-set gate vs HF bf16) still PASSES** — this is the Phase 3
   correctness gate, re-run between optimization skills. `make diagnosis`
   per-layer cosine is NOT a gate (the verify subsystem retired
   threshold-based diagnosis; `compare_pair` reports cosine with no
   pass/fail); run it only to **localize** a regression when verify breaks
   (which optimization / which layer the cosine cliff appears at). If
   `make verify` regresses to FAIL, **revert the change** and document why
   it doesn't apply to this model.
2. **Prefill kernel time strictly < Phase 3 baseline**, measured at
   the same canonical prompt + seq_len with 5-warmup + 20-iter
   profile. Recording wall time too (kernel + host overhead) is good
   but the gate is on kernel time (host overhead optimization is a
   separate concern).
3. **Per-skill outcome documented** in
   `<model>/docs/development_progress/phase4_prefill.md`: for each
   optimization skill it invoked, record `applied / skipped / reverted`,
   the latency delta, and a one-line reason.

The "≥ N optimization skills applied" check is NOT a gate — some models
legitimately need only 1-2 (e.g., the model is already seq-first by
construction → `layout-alignment` N/A). The gate is the outcome (perf
improved + correctness preserved), not the process count.

## Knowledge base references

PRIMARY:

- `programming_examples/llms/llama32_1b/docs/profile.md` — the reference
  deployment's profiling breakdown; the reference for what "good" looks like
- `programming_examples/llms/<model>/docs/development_progress/phase3_full.md`
  — Phase 3 baseline timings + cosine numbers (the "before" state to
  measure against and preserve)
- `programming_examples/llms/llama_kernel_builder/cache.py` — the
  `KernelCache` host-optimization knobs (`static_input_indices`,
  `intermediate_indices`); the `buffer-object-reuse` skill owns how to
  wire them, this is just the source file it touches

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

Before invoking any optimization skill, capture the baseline prefill time —
this is the number every skill must beat:

```bash
cd programming_examples/llms/<model>
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile
```

Record: kernel time (ms) + wall time as the baseline. Also note the Phase 3
per-layer cosine table as a baseline to compare against *if* a later
optimization breaks verify (it's the localization reference, not a gate).

### Step 2: Apply optimization skills

prefill draws on the shared optimization skillset. For each, decide if it
applies, invoke the skill, then re-run the gate (Step 3). Skip with a logged
reason if the trigger condition isn't met — "≥ N applied" is NOT the gate;
the gate is the outcome (faster + `make verify` still PASSES).

| Optimization skill | When it applies to prefill | What it does |
|---|---|---|
| `merge-multi-launch-kernels` | almost always (the dominant win) | stitch each leaf kernel's `air.launch` into one fused ELF per kernel-group → one `xrt.run()` per group instead of per kernel (llama3: 16→3 calls/layer). Build the model's `multi_launch_builder/` (kernel-first) or reuse llama's fused ELFs (bit-for-bit inheritance — the verdict made in `phase-2-single-block-validation` Step 1). |
| `buffer-object-reuse` | always | pre-load per-layer weight BOs once (`static_input_indices`) + reuse intermediate BOs (`intermediate_indices`); removes redundant host↔NPU uploads. |
| `layout-alignment` | only if a host transpose still sits between two kernels | choose seq-first layouts so RoPE/FA/O-proj hand off on-device; skip if the model already runs seq-first end-to-end (most inheritance deployments do). |

Each skill owns its own recipe, success self-check, and failure modes —
this phase does not restate them. Invoke the skill, read its result, then
gate.

### Step 3: Re-run Phase 3 gate after each optimization skill

After every applied (or attempted) optimization skill, re-run the gate:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify      # GATE: token-set, exit 1 on FAIL
```

`make verify` PASS is the correctness gate. If it regresses to FAIL,
revert the change and document why.

Only when verify FAILs, run diagnosis to **localize** the break:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make diagnosis   # informational: per-layer cosine table
```

A cosine cliff at layer *i* points at the broken assumption there (e.g. a
transpose `layout-alignment` removed that this model actually needed).
diagnosis does not PASS/FAIL — it is the microscope, verify is the gate.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Multi-launch merge compile fails (BD exhaustion, channel routing, herd shape conflict, bare-herd, DMA stride, IR/compile blowup) | non-1024-aligned dim (see the kernel's `details/<Kernel>_bf16.md`) OR wrong stitching boundary | Invoke `debug-multi-launch-merge` — it discriminates the 6 known compile blockers |
| Output corruption after BO pre-loading (correct first call, NaN/garbage on subsequent calls) | Per-layer BO key collision OR `static_input_indices` set wrong | Invoke `debug-bo-corruption` |
| FA hang (`ERT_CMD_STATE_TIMEOUT`) at head_dim ≥ 128 | Seq-first `dk_chunks > 1` path bug | Invoke `debug-fa-runtime-failure`; the head-first wrapper (routed by `layout-alignment`) is the workaround |
| FA all-NaN at runtime | Compile-flag mismatch on `attn_npu2.cc` macros (LESSON 3 — `-Dlqp` must be per-tile, not per-launch) | Invoke `debug-fa-runtime-failure`; `compile_attn_npu2_split` derives correct flags |
| Cosine drops after an optimization skill | the skill has a layout/type assumption your model violates | Revert the change; check whether the assumption (e.g., seq-first only, all weights pre-transposed) holds |
| Latency unchanged after `merge-multi-launch-kernels` | Multi-launch ELF compiled but XRT call count didn't drop | Check `xrt-smi top` for actual call count; verify the new fused ELF is what `_run_cached` actually invokes (not falling back to per-kernel path) |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 4 PASS:

- `<model>/docs/development_progress/phase4_prefill.md`: per-skill
  table with `applied / skipped / reverted`, latency delta, reason
- `<model>/TODO.md`: mark Phase 4, append final prefill kernel time +
  speedup vs Phase 3 baseline
- If a new fused ELF was built (kernel-first path of
  `merge-multi-launch-kernels`), surface to Phase 6 for potential
  promotion to a shared location if a second deployment validates the
  same pattern
