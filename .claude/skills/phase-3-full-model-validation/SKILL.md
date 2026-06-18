---
name: phase-3-full-model-validation
description: Phase 3 of LLM deployment — wire all N layers and verify NPU matches the HF bf16 reference end-to-end (per-layer cosine via the shared `llms/verify/` diagnosis lens + token-level top-5 set-inclusion via its token-set gate) at canonical prompts. Catches accumulated drift, KV cache bugs, layer-indexed weight loading errors. Invoked after Phase 2 gate.
---

## Purpose

Phase 2 verified one transformer block. Phase 3 scales to all N layers
and confirms the full prefill stays numerically aligned with the **HF bf16
reference** end-to-end. Catches: accumulated BF16 drift across deep stacks,
KV cache layout bugs, layer-indexed weight loading errors, LM head
precision drops.

Phase 3 reuses the `verify/` subsystem's two lenses:

- **`make diagnosis`** — per-layer `ffn_out` cosine (NPU vs HF bf16) for
  ALL layers. Informational by default; Phase 3 promotes it to a gate.
- **`make verify`** — token-level top-5 set-inclusion (NPU vs HF bf16
  greedy sequences). Already a hard gate (exit 0/1).

There is no hand-written CPU full-model forward — the reference is HF
transformers bf16 throughout.

## Phase 3 PASS criteria (HARD GATES)

Run `make diagnosis` (per-layer cosine, all layers) and `make verify`
(token-set gate) on the canonical prompts. All must hold:

**Numerical correctness (per-layer, vs HF bf16 — the diagnosis lens)**

1. **Per-layer cosine ≥ 0.85** at EVERY layer (NPU layer-i `ffn_out` vs
   HF bf16 layer-i `ffn_out`, plus the final post-norm hidden). 0.85 is
   loose enough to admit legitimate BF16 drift in deep models (a 28-layer
   stack hits ~0.88 at the last layer with no kernel bug) but tight enough
   to catch gross corruption.
2. **No sudden cliff** between consecutive layers:
   `|cos[i+1] - cos[i]| < 0.05`. Catches layer-indexed bugs (wrong
   `bo_key=f"kernel_L{i}"`, weight-load shifted by a layer) that would
   slip through (1) if the absolute value stays above threshold but the
   trend has a discontinuity.
3. **Record `max_abs` and `max_rel` error** alongside the cosines
   (informational, not gated — the diagnosis `error_metrics` reports
   them). Future deployments use these as a regression baseline (e.g.
   "new deployment's per-layer max_abs ≤ 1.5× the value logged here at
   the same layer signals no perf-related correctness drop").

**Semantic correctness (token-level, vs HF bf16 — the verify gate)**

4. **`make verify` PASSES**: at the first divergence between NPU and HF
   greedy sequences, NPU's chosen token is in HF's top-5 AND HF's chosen
   token is in NPU's top-5 (the `compute_topk_set_check` gate, GATE_K=5,
   GATE_N_TOKENS=32). This replaces the old hand-written top-1/top-5 vs
   CPU-reference check — it is the same idea (semantic agreement within a
   top-k band) but measured against HF bf16 and on generated tokens, not
   just the prefill prediction position. bf16 noise can flip top-1 even
   between mathematically equivalent implementations, but almost never
   displaces a token out of the top-5. This top-k token-set gate **mirrors
vLLM's correctness check** — it is the GPU/industry-standard end-to-end
signal, and the same gate Phase 6/7 re-run.

**Hygiene**

5. No NaN anywhere in the stack.
6. Per-layer cos table (from diagnosis) + `make verify` PASS/FAIL for
   every canonical prompt documented in
   `<model>/docs/development_progress/phase3_full.md`.

> **Standardization note (interim gate).** The per-layer cosine (criteria
> 1-3) is the interim correctness lens for *intermediate activations* —
> there is no token to score mid-stack. The token-set gate (criterion 4)
> is already the GPU/industry-standard (vLLM-aligned) end-to-end signal.
> The standardization goal is to bring the mid-layer activation gates onto
> the same footing over time; thresholds unchanged for now.

## Knowledge base references

- `programming_examples/llms/llama32_1b/llama32_1b_inference.py:run_npu_prefill`
  — reference full-stack pipeline (loops Phase 2's per-layer block)
- `programming_examples/llms/verify/verify_runner.py` — both lenses:
  `_run_diagnosis` (per-layer cosine) and the `topk_token` gate
  (`compute_topk_set_check`)
- `programming_examples/llms/verify/README.md` — the verify
  methodology (HF bf16 reference, top-k token-set gate, cosine as
  diagnosis)
- `<model>/docs/development_progress/` + the `kernel_registry`
  "tested shapes" rows with Used by = `<model>`
  — Phase 1 verified kernels (Phase 3 confirms they compose at scale)

## Workflow

### Step 1: Wire all N layers

Implement `run_full_prefill(input_ids, weights, config)` (or reuse the
deployment's `run_npu_prefill`):

- **Kernel-first path (default)**: loop `run_transformer_block_<model>(...)`
  (the per-model block runner Phase 2 produced) for
  `layer_idx in range(config.n_layers)`, then apply final RMSNorm + LM head.
- **Inheritance path (shortcut, bit-for-bit match only)**: loop
  `llama32_1b_prefill.run_transformer_block(...)` the same way.

Either way, this is just iterating the Phase 2 block runner N times plus
head — no new kernels.

### Step 2: Canonical prompts

Use the deployment's verify prompt set (`verify/prompts/{base,instruct}.txt`
— `base` for base checkpoints, `instruct` for instruct/chat checkpoints).
`make verify` runs 2 prompts × 32 tokens by default; `make verify-full`
runs the full set. `make diagnosis` runs a single prompt for the per-layer
cosine table.

The token-set gate handles the old "decisive vs competitive prompt"
distinction automatically: a top-1 flip within the top-5 band is treated
as benign drift (not a failure), while a token leaving the top-5 entirely
is a failure. No manual per-prompt probability classification is needed.

### Step 3: Run both lenses + collect metrics

```bash
cd programming_examples/llms/<model>
flock -x -w 1800 /tmp/mlir-air-npu.lock make diagnosis    # per-layer cosine, all layers
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify       # token-set gate (exit 0/1)
```

From diagnosis: read the per-layer cosine table (gate on criteria 1-2).
From verify: PASS/FAIL (criterion 4); the report under `verify/reports/`
records the first divergence and the top-5 sets on each side.

### Step 4: Bisect on FAIL

If the per-layer cosine gate fails, the diagnosis table localizes where:

- **Sudden cliff at layer i** (cos[i] >> cos[i+1]) → layer-indexed bug
  at i+1 (weight load shifted, wrong `bo_key`, wrong `wq` for that layer)
- **Gradual drift** but a layer < 0.85 → BF16 accumulator saturating;
  check whether the production GEMM path uses F32 internal accumulate;
  cross-check `make verify` — if the token-set gate still PASSES, the
  drift is geometric not a bug; if it FAILS, real bug
- **Layer 0 already low** → integration error in the single-block runner
  itself; revisit Phase 2

If `make verify` fails but per-layer cosine looks fine, the divergence is
likely in the decode path / KV-cache (verify generates 32 tokens; the
per-layer cosine only probes prefill). Validate KV cache values at end of
prefill against HF. Within an offending layer, bisect kernel-by-kernel
using Phase 2's CPU-fallback technique.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Per-layer cos cliff at one layer | Layer-indexed weight load bug or wrong `bo_key=f"kernel_L{i}"` | Print weight shapes per layer; compare K/V cache layout at boundary |
| Per-layer cos drifts gradually < 0.85 by last layer | Real BF16 saturation, OR Down GEMM missing F32 accumulator | Run `make verify` — if token-set gate PASSES, drift is geometric not a bug; otherwise check F32 accumulator pattern |
| Per-layer cos OK but `make verify` FAILS at prediction | LM head precision drop (BF16 truncation in vocab projection) | Apply F32 accumulator pattern to LM head GEMM |
| `make verify` FAILS: NPU top-1 NOT in HF top-5 at all | Real correctness bug | Step 4 bisect |
| NaN in output | Uninitialized BO / reused stale buffer | Invoke `debug-bo-corruption` |
| Per-layer cos OK, prefill prediction OK, but multi-token generation diverges quickly | KV cache update bug at decode time (per-layer cosine only probes prefill; verify's 32-token generation catches it) | Validate KV cache values at end of prefill match HF |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 3 PASS, this is the **end-to-end correctness milestone** — NPU is
now numerically faithful to the HF bf16 reference at full N-layer scale.
Update:

- `<model>/docs/development_progress/phase3_full.md`: per-layer cos table
  (diagnosis) + `make verify` PASS per prompt
- `<model>/docs/development_progress/progress.md`: Phase 3 summary
- `<model>/TODO.md`: mark Phase 3, advance to perf phases

Phase 4 (prefill perf) and Phase 5 (decode perf) MUST preserve these gates
(`make diagnosis` per-layer cosine + `make verify` token-set) after every
optimization — perf cannot trade correctness.
