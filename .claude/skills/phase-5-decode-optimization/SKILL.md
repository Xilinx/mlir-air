---
name: phase-5-decode-optimization
description: Phase 5 of LLM deployment — apply the shared optimization skillset to a Phase-4-correct decode pipeline (multi-launch merge with N-way extern rename, static weight BOs, on-device layout). Thin orchestrator that dispatches `opt-merge-multi-launch-kernels`, `opt-buffer-object-reuse`, and `opt-layout-alignment`. Each step preserves correctness by re-running the Phase 3 gate — `make verify` (token-set vs HF bf16) is the PASS/FAIL gate; `make diagnosis` per-layer cosine is the informational lens used to localize a regression. Invoked after Phase 4 PASS.
---

## Purpose

Phase 4 optimized prefill while preserving Phase 3 correctness. Phase 5
does the same for decode — but the dominant optimizations differ because
decode runs at M=1 per token, calling all N layers once per generated
token. This phase is a thin orchestrator: it dispatches the same shared
optimization skillset as Phase 4 — `opt-merge-multi-launch-kernels`
(ELF-merging), `opt-buffer-object-reuse` (host↔NPU runtime-overhead
reduction), and `opt-layout-alignment` (host-side layout) — each of which
owns its own recipe and failure modes. Decode amplifies the value of
static weight BOs (weights are loaded once but read on every token).
These wins are things you **compose from the kernels**, not behaviors
inherited from a reference; the reference's builders are worked examples.
Every optimization is an experiment: apply, re-measure, re-run Phase 3
gate; revert if correctness regresses.

For scale, the reference deployment llama3.2-1B took decode from ~500
ms/token → 92 ms/token (5.4×) by composing these optimizations — an
illustrative datapoint for what they buy on one model, not a target every
deployment must hit.

## Phase 5 PASS criteria (HARD GATES)

1. **Correctness preserved**: after every applied optimization, **`make verify`
   (the token-set gate vs HF bf16) still PASSES** — this is the Phase 3
   correctness gate, re-run between optimization skills. `make diagnosis`
   per-layer cosine is NOT a gate (the verify subsystem retired
   threshold-based diagnosis; `compare_pair` reports cosine with no
   pass/fail), and note it only probes *prefill* — decode regressions (KV
   cache, per-token BO reuse) surface in `make verify`'s 32-token
   generation, not in diagnosis. If `make verify` regresses to FAIL,
   **revert the change** and document why.
2. **Decode time/token strictly < Phase 4 baseline**, measured with
   `make profile` at the same canonical prompt.
3. **Per-skill outcome documented** in
   `<model>/docs/development_progress/phase5_decode.md`: for each
   optimization skill it invoked, record `applied / skipped / reverted`,
   the latency delta, and a one-line reason.

The "≥ N optimization skills applied" check is NOT a gate — some models
legitimately need only merge + static weight BOs. The gate is the
outcome (decode time improved + correctness preserved), not the process
count.

## Knowledge base references

PRIMARY:

- `programming_examples/llms/llama32_1b/docs/profile.md` — the reference
  deployment's profiling breakdown; the reference for what "good" looks like
- `programming_examples/llms/<model>/docs/development_progress/phase4_prefill.md`
  — Phase 4 baseline (prefill numbers + the integration path used)
- `programming_examples/llms/llama32_1b/multi_launch_builder/o_gemv_ffn_multi.py`
  — decode-specific merge pattern + 2-K extern kernel rename, in code
- `programming_examples/kernel_registry/details/GEMV_bf16.md` — per-kernel
  constraints / placeability notes (the authority for that kernel's hard
  limits): K_max=8160, combined channel reads ≤ 255, L2 cap — relevant
  when GEMV K > 8160 or M is large

REFERENCE EXEMPLARS (read/mirror to compose your own decode ELFs; import
directly only on a bit-for-bit kernel-sequence match):

- `programming_examples/llms/llama32_1b/multi_launch_builder/rms_gemv_rope_multi.py`
  — fused 6-launch decode ELF for RMSNorm + Q/K/V GEMV + RoPE Q/K
- `programming_examples/llms/llama32_1b/multi_launch_builder/o_gemv_ffn_multi.py`
  — fused 8-launch decode ELF (with 2-K extern rename for K=8192 Down)
- `programming_examples/llms/llama32_1b/multi_launch_builder/lm_head_gemv_multi.py`
  — vocab-partitioned LM Head GEMV (part of the model's decode assembly,
  built in Phase 3/finalize; profiled here, not a separate optimization)
- `programming_examples/llms/llama_kernel_builder/` — the shared toolkit
  every decode-ELF build uses (KernelCache, stitching, external_kernels).

## Workflow

### Step 1: Measure Phase 4 baseline

Capture the decode time/token before invoking any optimization skill —
this is the number every skill must beat:

```bash
cd programming_examples/llms/<model>
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile
```

Record: ms/token + per-layer + LM head time breakdown (where the
budget goes today) as the baseline. These numbers gate every
optimization below.

### Step 2: Apply optimization skills

decode draws on the same shared optimization skillset; the dominant patterns
differ because decode runs at M=1 per token, calling all N layers per token.

| Optimization skill | When it applies to decode | What it does (decode flavor) |
|---|---|---|
| `opt-merge-multi-launch-kernels` | almost always | stitch decode kernel groups (GEMV instead of GEMM) into fused ELFs (10 launches/layer/token → 2–3). Build the model's `multi_launch_builder/` (kernel-first) or reuse llama's fused decode ELFs (bit-for-bit inheritance — the verdict made in `phase-2-single-block-validation` Step 1). **Decode specifics handled by the skill**: N-way extern kernel rename when multiple GEMV K values co-link in one ELF (2-K for llama: `mv.o` K=2048 + `mv_k8192.o`; add a 3rd renamed `.o` when `n_heads·head_dim ≠ emb_dim`), and K-split (`down_k_split`) for K > 8160 (`details/GEMV_bf16.md`). |
| `opt-buffer-object-reuse` | always — biggest decode win | static weight BOs: weights allocated once, `bo.map()` zero-copy, `static_input_indices` skips re-write on every token. With 16+ layers × ~7 weights × 100 tokens, this is the dominant pre-optimization decode host cost. |
| `opt-layout-alignment` | usually N/A | only if decode introduced a transpose Phase 4 didn't already fix. |

Each skill owns its recipe + success self-check + failure modes. The LM Head
GEMV (vocab-partitioned) is part of the model's decode assembly built in
Phase 3/finalize, profiled here — not a separate optimization skill.

### Step 3: Re-run Phase 3 gate after each optimization skill

After every applied (or attempted) optimization skill, re-run the gate:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify      # GATE: token-set, exit 1 on FAIL
```

`make verify` PASS is the correctness gate (its 32-token generation is
what catches decode-only bugs — KV cache, per-token static-BO reuse). If
it regresses to FAIL, revert the change and document why. diagnosis only
probes prefill, so it cannot localize a decode regression — bisect by
reverting optimization skills instead.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Multi-launch merge compile fails (BD exhaustion, channel routing, herd shape conflict, bare-herd, DMA stride, IR/compile blowup) | BD/placeability limit or wrong stitching boundary | Invoke `debug-multi-launch-merge` — it discriminates the 6 known compile blockers |
| Extern kernel rename collision (link error, symbol redefined) | Two `.o` files exporting same symbol | Check `-D` mapping uniqueness; each `.o` must export distinct `<group>_matvec_*` names |
| `'aiex.npu.push_queue' op Repeat count exceeds [0:255]` (`opt-merge-multi-launch-kernels`) | GEMV K > 8160, or combined channel reads > 255 (see `details/GEMV_bf16.md`) | For K > 8160 → set `k_split` / `down_k_split`; for large M → set `tile_m == m_input` and grow `tile_m × herd_m` |
| `L2 capacity exceeded` (matvec.py builder assert) | GEMV staged buffer `K × herd_m × tile_m × 2 > 512 KiB` (see `details/GEMV_bf16.md`) | Reduce `tile_m` (e.g., 8 → 2 for K=8192) |
| Output corruption after static weight BO conversion (correct first call, NaN/garbage on subsequent) | Per-layer BO key collision OR `static_input_indices` wrong | Invoke `debug-bo-corruption` |
| Cosine drops after an optimization skill | the skill has a layout/type assumption this model violates | Revert the change; check the assumption (e.g., decode already seq-first, weights already pre-transposed) |
| ms/token unchanged after `opt-merge-multi-launch-kernels` | Per-call XRT overhead dominates; fusion alone insufficient | `opt-buffer-object-reuse` (static weight BOs) is likely the missing piece — apply it next |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 5 PASS:

- `<model>/docs/development_progress/phase5_decode.md`: per-skill
  table with `applied / skipped / reverted`, latency delta, reason
- `<model>/TODO.md`: mark Phase 5, append final ms/token + speedup
  vs Phase 4 baseline
- If a new fused decode ELF was built (kernel-first path of
  `opt-merge-multi-launch-kernels`), surface to Phase 6 for potential
  promotion to a shared location if a second deployment validates the
  same pattern
