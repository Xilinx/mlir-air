---
name: buffer-object-reuse
description: Optimization skill — reuse NPU BufferObjects across calls instead of re-allocating/re-writing them. Two mechanics in one class: (B1) per-layer weight BOs pre-loaded once and skipped via static_input_indices, and (B2) intermediate BOs the kernel overwrites, skipped via intermediate_indices. Invoked by phase-4-prefill-optimization and phase-5-decode-optimization to cut redundant host↔NPU data movement. Decode amplifies the weight-BO win (weights reused on every token).
---

## Purpose

NPU kernels re-allocate and re-upload BufferObjects (BOs) on every call by
default. For an N-layer transformer that re-runs the same kernels per layer
(prefill) and per token (decode), this is pure redundant host↔NPU traffic.
This skill removes it. It is the same optimization the reference ablation
isolated as cells A→B (weight BOs) and B→C (intermediate BOs); together they
were a multi-second prefill saving and the dominant decode host-side cost.

Two mechanics, one class:

- **B1 — per-layer weight BOs** (`static_input_indices`): allocate each
  layer's weight BOs once during setup, write them once, and pass
  `static_input_indices=[<weight slots>]` on every `cache.load_and_run()` so
  the runtime skips re-writing them.
- **B2 — intermediate BOs** (`intermediate_indices`): for buffers the kernel
  fully overwrites (its own outputs / scratch), pass
  `intermediate_indices=[<output slots>]` so the host does not write them
  before the call.

## Success criteria

Applying this skill is "successful" when ALL hold:

1. Output cosine ≥ 0.99 vs the pre-optimization baseline (BO reuse must not
   change the math — same kernels, same inputs, only the upload is skipped).
   Log `max_abs / max_rel` informational (match the BF16 convention from
   `phase-1-kernel-validation`; do not use a tight `rtol`).
2. `make verify` still PASSES (the end-to-end gate; token-set top-k vs HF bf16).
3. Measured host/wall time is strictly lower than the baseline.

If (1)/(2) regress (NaN or garbage on the 2nd+ call, correct on the 1st) →
the BO bookkeeping is wrong; invoke `debug-bo-corruption`.
If (3) shows no gain → the per-call upload wasn't the bottleneck here;
document and keep or revert.

## Knowledge base references

- `programming_examples/llms/llama_kernel_builder/cache.py` —
  `KernelCache.load_and_run`, the `static_input_indices` /
  `intermediate_indices` mechanics this skill drives.
- `programming_examples/llms/llama32_1b/multi_launch_builder/*` — the worked
  example of weight + intermediate BO slots passed to a fused ELF.
- `programming_examples/llms/llama32_1b/llama32_1b_inference.py` —
  `prepare_runtime` / setup where per-layer BOs are allocated once.

## Workflow

### Step 1: Identify the BO slots

From the kernel group's argument signature, classify each BO slot:

- **weight / LUT slots** — written once, read every call → B1 candidates
  (`static_input_indices`).
- **kernel-overwritten slots** (the kernel's outputs and scratch) → B2
  candidates (`intermediate_indices`).
- **genuine per-call inputs** (the activation that changes each call) — leave
  as normal host-written inputs.

### Step 2: Pre-load weight BOs once (B1)

- Allocate per-layer weight BOs in `prepare_runtime()` (setup), keyed per
  layer (e.g. `bo_key=f"kernel_L{layer_idx}"`), and write the weights ONCE.
- On every `cache.load_and_run()`, pass `static_input_indices=[<weight slots>]`
  so the runtime does not re-upload them.

### Step 3: Reuse intermediate BOs (B2)

- For each slot the kernel fully overwrites, pass
  `intermediate_indices=[<output slots>]` on `cache.load_and_run()` so the
  host does not write that buffer before the call.

### Step 4: prefill vs decode

Same mechanic, two contexts (pass which one as the caller's parameter):

- **prefill**: weights re-used across the 16 per-layer calls within one pass.
- **decode**: weights re-used across every generated token × every layer —
  the win is much larger (16 layers × ~7 weight tensors × N tokens of upload
  removed). Static weight BOs are the dominant decode host-side optimization.

### Step 5: Validate + measure

- Run with BO reuse; compare output to the pre-reuse baseline → cosine ≥ 0.99.
- Re-run `make verify` → must still PASS.
- Profile host/wall time → must be strictly lower.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Correct on 1st call, NaN/garbage on 2nd+ call | per-layer BO key collision OR `static_input_indices` slot list wrong | Invoke `debug-bo-corruption` |
| Output mismatch on the very 1st call | a slot marked `intermediate` is actually read before being written | Re-classify that slot as a real input (drop it from `intermediate_indices`) |
| No host-time reduction | the per-call upload wasn't the bottleneck (kernel-bound) | Document; the merge skill (dispatch) or layout-alignment may be the bigger win |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

Append to `<model>/docs/development_progress/phase{4,5}_*.md`:

```
## Buffer-object reuse
- B1 weight BOs: applied / skipped (reason)
- B2 intermediate BOs: applied / skipped (reason)
- Host/wall time before: X ms
- Host/wall time after:  Y ms
- Cosine vs baseline: <value>  | make verify: PASS/FAIL
```
