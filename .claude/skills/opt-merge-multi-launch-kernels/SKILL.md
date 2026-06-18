---
name: opt-merge-multi-launch-kernels
description: Procedural recipe for fusing multiple `air.launch` kernels into one multi-launch ELF (single XRT invocation). Invoked by phase-4-prefill-optimization and phase-5-decode-optimization to fuse kernel groups when building NEW model-specific fused ELFs (kernel-first path). Reduces XRT dispatch overhead (~50–200 µs per call on NPU2).
---

## Purpose

When a deployment goes the **kernel-first path** in Phase 4/5 (model
has new ops or stitching boundaries differ from llama), build new
model-specific fused ELFs by stitching multiple `air.launch` kernels
into a single `func.func`. This skill is the procedural recipe.

The reference llama3 deployment went 10 launches/layer → 3
launches/layer using this recipe — the dominant prefill perf win.

## Success criteria

A merge is "successful" when ALL hold:

1. Merged ELF compiles without errors
2. Output cosine ≥ 0.99 vs the unmerged baseline (run the same kernels
   as separate XRT calls, compare output). Match the BF16 tolerance
   convention from the kernel registry — `rtol=1e-3` is too tight for
   K=2048 BF16; use cosine + `max_abs/max_rel` informational logging
   (see `phase-1-kernel-validation` PASS criteria for the convention).
3. Wall-clock time is lower than the unmerged baseline

If (1) fails → invoke `debug-multi-launch-merge`.
If (2) fails → bisect by un-merging the last-added kernel.
If (3) fails (compile slower than unmerged) → not a correctness bug;
document and accept or revert.

## Knowledge base references

- `programming_examples/llms/llama_kernel_builder/stitching.py` —
  the helpers this recipe uses (`_rename_all`, `_fix_launch_func_args`,
  `_wrap_ir_in_launch`, `_rename_all_with_externs`, `_rename_all_gemv`)
- `programming_examples/llms/llama32_1b/multi_launch_builder/rms_gemms_rope_multi.py`
  — reference 6-launch prefill merge (RMSNorm + Q/K/V GEMM + RoPE Q/K)
- `programming_examples/llms/llama32_1b/multi_launch_builder/o_ffn_multi.py`
  — reference 8-launch prefill merge (O + add + RMSNorm + Gate/Up + SwiGLU + Down + add)
- `programming_examples/llms/llama32_1b/multi_launch_builder/o_gemv_ffn_multi.py`
  — decode merge with 2-K extern rename (the pattern to extend to 3-K
  when `n_heads*head_dim != emb_dim`)
- `programming_examples/kernel_registry/supported_kernels.md`
  — per-kernel constraints; the FA section notes FlashAttention stays a
  separate XRT call (does NOT merge into the fused block)

## Workflow

### Step 1: Identify merge candidates

List the per-layer kernel sequence. Mark each as one of:

- **Mergeable** — pure compute (RMSNorm, GEMM, GEMV, RoPE, eltwise)
- **Hard-stop** — has data-dependent control flow or unsupported merge
  pattern. FlashAttention is the canonical hard-stop (see `full_block.md`)
- **Conditional** — mergeable but check herd-shape compatibility with
  neighbors (e.g., GEMM uses 8×4 herd; eltwise uses 8×1; coexisting in
  one segment is OK but verify)

### Step 2: Pick merge boundaries

Group consecutive mergeable launches between hard-stops. For a
typical decoder-only LLM:

- **Group A**: RMSNorm + Q/K/V proj + RoPE Q + RoPE K (6 launches —
  `rms_gemms_rope_multi.py` shape)
- **Hard-stop**: FlashAttention (separate XRT call)
- **Group B**: O proj + Add + RMSNorm + Gate + Up + SiLU+Mul + Down +
  Add (8 launches — `o_ffn_multi.py` shape)

For decode, replace GEMM with GEMV throughout.

### Step 3: Author the multi-launch builder

For each group, create `<model>/multi_launch_builder/<group_name>_multi.py`
that:

1. Builds each sub-kernel's IR via `@module_builder`
2. Imports stitching helpers: `from llama_kernel_builder.stitching import (_rename_all, _fix_launch_func_args, _wrap_ir_in_launch, ...)` (resolved against the shared `llms/llama_kernel_builder/` via sys.path)
3. For each sub-kernel: extract its function body via `_extract_between_func_and_return(ir_text)`
4. Rename SSA values with a per-kernel prefix via `_rename_all(body, prefix=...)` to avoid collisions across the merged module
5. Remap function arguments to the merged module's args via `_fix_launch_func_args(body, prefix, arg_map)`
6. Concatenate the renamed bodies into a single `func.func`. If any
   sub-kernel emits a bare `air.herd` (RMSNorm `herd_x>1`, Eltwise
   Add at `herd_x>1`), wrap each via `_wrap_ir_in_launch(...)` BEFORE
   stitching — otherwise the lowering's `airrt-to-npu` pass drops the
   bare herd
7. For multi-K GEMV in one ELF (decode kernel-first): use
   `_rename_all_with_externs` with per-launch extern allowlists to keep
   different `mv_*.o` symbols distinct (extend the 2-K rename in
   `o_gemv_ffn_multi.py` to 3-K when `n_heads*head_dim != emb_dim`)

The canonical reference is `rms_gemms_rope_multi.py` — copy from there.

### Step 4: Compile via KernelCache

Use `KernelCache.compile_and_cache(name=<group_name>, builder=<your_builder_function>, ...)`.

If compile fails → invoke `debug-multi-launch-merge`.

### Step 5: Validate output vs unmerged baseline

Run the merged ELF on a fixed input. Run the same kernels as separate
XRT calls (the unmerged baseline). Compare the merged output to the
unmerged output:

- cosine ≥ 0.99 (gate)
- log `max_abs / max_rel` informational

If cosine fails: bisect by un-merging the last-added kernel from
Step 3. If still fails after un-merging that one, the bug is in an
EARLIER merge addition — bisect further.

### Step 6: Measure perf gain

Profile the merged version vs unmerged. Expected at NPU2 dispatch
overhead levels: ≥ 20% reduction per merged group at moderate scale
(llama3 saw 16-XRT-call/layer prefill → 3-XRT-call/layer = much
larger reduction). If the gain is much smaller than expected, the
per-call XRT overhead may not have been the bottleneck — the
`opt-buffer-object-reuse` skill (static weight BOs) is often the missing piece
in that case.

Record gain in `<model>/docs/development_progress/phase{4,5}_*.md`.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Compile fails (BD exhaustion, channel routing, herd shape conflict, IR validation) | Multi-launch hardware-resource collision | Invoke `debug-multi-launch-merge` for the diagnostic decision tree |
| Output mismatch vs unmerged baseline | One of the per-kernel stitch operations corrupted SSA names, arg mapping, or layout boundary | Bisect by un-merging the last-added kernel; the first un-merge that restores correctness identifies the offender |
| BO corruption after merge (NaN on 2nd+ call, stale values) | `static_input_indices` / `intermediate_indices` not propagated to the merged ELF's `cache.load_and_run()` | Invoke `debug-bo-corruption` |
| Bare-herd kernel produces all-zero in merged ELF | RMSNorm or Eltwise at `herd_x>1` emits bare `air.herd`; needs `_wrap_ir_in_launch` BEFORE stitching | Wrap in Step 3 step 6 |
| Merged compile slower than unmerged baseline | Compile-time scaling with ELF size; not a correctness bug | Document in TODO; accept or drop the most-recent merge addition |

## Update protocol

Append to `<model>/docs/development_progress/phase{4,5}_*.md`:

```
## Multi-launch merge: <group_name>
- Sub-kernels merged: N
- Latency before: X ms
- Latency after: Y ms
- Gain: Z%
- Cosine vs unmerged: <value>
- max_abs / max_rel: <value> / <value>
```
