---
name: debug-bo-corruption
description: Use when an NPU kernel passes its standalone shape test but produces NaN, garbage, or stale values when invoked as part of a larger pipeline. Common symptoms: correct first invocation but wrong on subsequent calls; correct in isolation but wrong when chained with other kernels.
---

## Purpose

Diagnose Buffer Object (BO) corruption — bugs that surface only at
integration time, characterized by a kernel that passes standalone
correctness but produces wrong output in a larger pipeline. The 4
hypotheses below cover every BO-corruption case observed across the
6 LLM deployments to date. Use this skill as a **diagnostic checklist**,
not a mechanical fix-applier — understand which hypothesis fits before
applying the corresponding remedy.

## Knowledge base references

- `programming_examples/llms/llama_kernel_builder/cache.py`
  — `static_input_indices`, `intermediate_indices` mechanics (the
  `KernelCache` host-optimization knobs)
- `programming_examples/llms/llama_kernel_builder/stitching.py`
  — `_wrap_ir_in_launch` is the helper that wraps a bare herd in
  `air.launch + air.segment` (see hypothesis 4 below); also where
  `airrt.herd_load` vs `airrt.segment_load` semantics matter

## Trigger pattern

This skill matches when ANY of these apply:

- Output tensor contains NaN despite kernel passing standalone XRTRunner test
- Kernel produces correct output on first invocation but wrong on second+
  invocation in a loop
- Kernel produces correct output standalone but wrong when chained with
  another kernel
- Output tensor has correct shape but values match a *previous*
  layer/iteration

## Diagnostic hypothesis tree

For each hypothesis: check the symptom-fit, then apply the listed
remedy ONLY if the diagnostic matches. Don't apply remedies blindly.

### Hypothesis 1: Stale BO state from a prior call

**Symptom-fit**: Output is correct on first call, wrong on subsequent
calls. Wrong values match a different invocation.

**Diagnostic**: Find the kernel's `cache.load_and_run(...)` call.
Inspect whether `static_input_indices=[...]` is passed. Cross-reference
the kernel's MLIR signature: which input indices are weights (written
once, reused) vs activations (overwritten each call)?

**Remedy**: Add the weight indices to `static_input_indices` so
`load_and_run` skips re-writing them every call. Re-test.

### Hypothesis 2: Intermediate buffer not marked

**Symptom-fit**: Output buffer has stale values from previous call;
NaN on first call (uninitialized) but a "successful" cosine number
later.

**Diagnostic**: Identify buffers in the kernel that are *outputs* the
kernel will fully overwrite. Check whether `intermediate_indices=[...]`
lists these.

**Remedy**: Add the output indices to `intermediate_indices` so
`load_and_run` skips the initial host write. Re-test.

### Hypothesis 3: Buffer aliasing across layers

**Symptom-fit**: Per-layer kernel call works in isolation; multiple
layers chained produce wrong output. Different layers are clobbering
each other's BO state.

**Diagnostic**: Search for `bo_key` strings across layers. Confirm
each layer uses a unique key (the convention is
`f"kernel_L{layer_idx}"`).

**Remedy**: Parameterize on layer index. Re-test after fix in a
multi-layer call (e.g., 2-layer test before going to N-layer).

### Hypothesis 4: Bare `air.herd` missing the `air.launch + air.segment` wrapper

**Symptom-fit**: Output is all-zero (or undefined) — the kernel
"succeeds" silently but never actually executes the DMA path (a
silent-corruption pattern: cosine ≈ 0 / NaN).

**Diagnostic**: Search the multi-launch builder (or the standalone
test harness) for `air.herd` ops. Check each is wrapped in `air.launch
+ air.segment`. Reference: the `airrt-to-npu` lowering pass needs
`airrt.segment_load` (not `airrt.herd_load`) to attach the launch
region to the `aie.device` op. A bare herd without the segment wrapper
gets silently dropped.

**Remedy**: Wrap via `_wrap_ir_in_launch(mlir_text)` from
`programming_examples/llms/llama_kernel_builder/stitching.py`. The fused multi-launch
builders in `llama32_1b/multi_launch_builder/` use this wrapper around
every bare herd (e.g. the RMSNorm and Eltwise-Add `herd_x=8` builders,
which emit bare herds) — mirror that pattern.

Re-test.

## Verification

After applying any fix:

1. Re-run the failing test (the standalone XRT runner test or the
   integration test that surfaced the bug)
2. Confirm output cosine ≥ 0.99 vs CPU reference (BF16 convention; do
   NOT use `rtol=1e-3` — too tight for K ≥ 1024 BF16)
3. Run the test 3 times in a loop to confirm consistency across
   invocations (NaN on iter 2+ is a different bug than failure on iter 1)

If fix succeeded: record `recovered_via=debug-bo-corruption` and
which hypothesis fired in
`<model>/docs/development_progress/debug_log.md`.

If no hypothesis fits the symptom OR no remedy works: this is a new
failure mode. Escalate via `<model>/TODO.md` "Active blockers" with
the failing test, the hypotheses tried, and the unchanged failure
output. Don't apply remedies speculatively.

## Failure modes (when this skill itself can't resolve)

| Symptom doesn't fit any of 4 hypotheses | What to check |
|---|---|
| Output is wrong on iter 1 too (not just stale across calls) | Not BO corruption; this is a kernel correctness bug — re-run Phase 1 standalone test for that kernel |
| Output is wrong only at specific seq_len / layer index | Could be KV cache layout bug; invoke `superpowers:systematic-debugging` |
| `bo_key` is unique per layer but still aliasing | `KernelCache` may be reusing artifact across `bo_key` — inspect `cache.artifacts` dict at runtime |

## Update protocol

On success: append to `<model>/docs/development_progress/debug_log.md`:

```
## debug-bo-corruption recovery (YYYY-MM-DD)
- Failing item: <kernel/test name>
- Hypothesis fired: 1 / 2 / 3 / 4
- Fix applied: <one-line description>
- Verified: 3/3 consistent runs, cosine ≥ 0.99
```

On escalation: append to `<model>/TODO.md` Active blockers with full
failure context and which hypotheses were ruled out.
