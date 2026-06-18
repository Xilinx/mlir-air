---
name: debug-multi-launch-merge
description: Use when stitching kernels into a multi-launch ELF and the AIE compiler rejects the merged module (BD exhaustion, channel routing, herd shape conflict, IR validation error, DMA stride limitation). Discriminates the 6 known compile blockers via a symptom-classification table.
---

## Purpose

When `merge-multi-launch-kernels` produces a fused ELF and `aircc.py`
rejects it for hardware-resource reasons, this skill identifies which
of the 6 documented constraints was hit. Use as a **diagnostic
decision tree**, not a mechanical fix-applier — confirm the trigger
matches your stderr before applying the corresponding remedy.

## Knowledge base references

- `programming_examples/kernel_registry/details/<Kernel>_bf16.md` —
  per-kernel constraints / placeability notes (the authority for that
  kernel's hard limits): BD inner-dim 1024, GEMV K-DMA repeat, combined
  channel reads, L2 cap — the constraints the merge blockers below map
  back to (for GEMV-specific limits see `details/GEMV_bf16.md`)
- `programming_examples/kernel_registry/supported_kernels.md`
  — per-kernel constraints + silent-corruption traps (the merge
  constraints in context of each leaf kernel)
- `programming_examples/llms/llama32_1b/multi_launch_builder/` — working
  fused-ELF builders to diff a failing merge against (what DOES merge,
  and the FA-stays-separate boundary)

## Trigger pattern

This skill matches when ANY of these appear in `aircc.py` stderr:

- `buffer descriptor` / `BD` / `out of buffer descriptors` (BD exhaustion)
- `channel routing` / `cannot route` / `channel allocation failed`
- `herd shape mismatch` / `herd dimension`
- `aie.tile` location conflict
- `airrt.herd_load` not found / undefined symbol
- `stride must be 1` for BF16 / sub-32b types

If your error doesn't match any of these patterns, the issue is likely
NOT a multi-launch resource collision — invoke
`superpowers:systematic-debugging` instead.

## Diagnostic hypothesis tree

For each hypothesis: confirm the symptom-fit, then apply the listed
remedy. Don't apply remedies speculatively.

### Hypothesis 1: BD exhaustion

**Symptom-fit**: stderr mentions `buffer descriptor` / `BD` / `out of
buffer descriptors`. Often triggered by a non-1024-aligned dim (see the
kernel's `details/<Kernel>_bf16.md` placeability notes) because the DMA
auto-splits into many shim BDs that exhaust the pool.

**Diagnostic**: identify which model dim is non-1024-aligned (e.g.
`emb_dim=1536`). Check the merged ELF's launch count and whether each
launch's DMA pattern is BD-friendly (see the kernel's
`details/<Kernel>_bf16.md`).

**Remedy** (in order of preference):

1. Pad the offending dim to 1024-aligned via GQA-aware reindexed padding
   (technique in `phase-2-single-block-validation` Step 2 + the kernel's `details/<Kernel>_bf16.md` placeability notes)
2. If padding is infeasible: un-merge the most recently added launch
   from the merged ELF; recompile
3. Switch to kernel-first split-ELF path (more, smaller ELFs)

### Hypothesis 2: Channel routing congestion

**Symptom-fit**: stderr mentions `channel routing` / `cannot route` /
`channel allocation failed`. Adjacent launches use overlapping channel
IDs that cannot coexist physically on the AIE2P fabric.

**Diagnostic**: print the merged MLIR (`make print` or
`--print-module-only`); search for `air.channel` declarations and
check IDs across the merged launches.

**Remedy**: rename channels in one of the offending launches. The
`_rename_all(text, prefix=...)` helper in
`llms/llama_kernel_builder/stitching.py` already prefixes every SSA
name (including channels) per-kernel — confirm your stitching code
used distinct prefixes per sub-kernel. If two sub-kernels were
stitched with the same prefix, that's the bug. Re-run stitching with
distinct prefixes; recompile.

### Hypothesis 3: Herd shape conflict

**Symptom-fit**: stderr mentions `herd shape mismatch` / `herd
dimension`. Two launches need different herd shapes (e.g., `[8,4]` for
GEMM and `[8,1]` for RMSNorm) that the placement pass can't reconcile
in one segment.

**Diagnostic**: identify the herd `sizes=[N, M]` of each sub-kernel.
GEMM is typically `[8, 4]`, GEMV/RMSNorm/RoPE/Eltwise are `[8, 1]`.

**Remedy**: in practice these CAN coexist in one segment when both fit
the chip's 8 columns (8 cols × max(M_i) = chip rows). If the compiler
still rejects, the cleanest path is to **keep the offending kernels
as separate XRT calls** (don't merge them). This isn't a regression —
it's a scoping decision. Document in TODO and accept the slightly
higher dispatch overhead.

### Hypothesis 4: Bare `air.herd` missing the launch+segment wrapper

**Symptom-fit**: stderr mentions `airrt.herd_load not found` /
`undefined symbol` / `failed to legalize airrt.dma_memcpy_nd`.

**Diagnostic**: same as `debug-bo-corruption` Hypothesis 4 — search
the multi-launch builder for `air.herd` ops not wrapped in
`air.launch + air.segment`.

**Remedy**: wrap via `_wrap_ir_in_launch(mlir_text)` from
`llms/llama_kernel_builder/stitching.py`. The fused builders in
`llama32_1b/multi_launch_builder/` apply this wrapper around every bare
herd (e.g. the RMSNorm and Eltwise-Add `herd_x=8` builders).

### Hypothesis 5: DMA stride limitation (sub-32b)

**Symptom-fit**: stderr mentions `stride must be 1` for BF16 or other
sub-32b types.

**Diagnostic**: BF16 DMA on AIE2P requires `stride=1` for the inner
dim. A producer kernel emitting an output with stride > 1 (e.g., a
transpose layout with non-contiguous BF16 elements) will hit this.

**Remedy**: restructure the data layout so the offending DMA has
`stride=1`. Often requires changing memref shape or transpose order
in the producer kernel. See
`compiler_issues/weight_broadcast_dma.md` for examples.

### Hypothesis 6: Compile-time blowup

**Symptom-fit**: compile doesn't fail with an error — it just takes
> 5 minutes (per `compiler_scaling.md`).

**Diagnostic**: not a correctness bug; a workflow blocker.

**Remedy**: reduce the merge scope by dropping the most-recently
added launch from the merged set. Document the soft cap in TODO so
future deployments don't push past it.

## Verification

After applying a fix:

1. Recompile the merged ELF — should succeed within reasonable time
   (< 5 min)
2. Run the merged ELF on the same input as a single XRT call
3. Compare output against the unmerged baseline (cosine ≥ 0.99; do
   NOT use `rtol=1e-3` — too tight for K ≥ 1024 BF16); log
   `max_abs/max_rel` informational

If output mismatch: invoke `merge-multi-launch-kernels` Step 5 bisect
(un-merge the last-added kernel to localize the offender).

## Failure mode (when this skill itself can't resolve)

If your stderr doesn't match any of the 6 hypothesis trigger patterns,
this is a new compile blocker. Escalate via `<model>/TODO.md` "Active
blockers" with:

- Full stderr
- The merged MLIR (`make print` output)
- Which sub-kernels were being stitched
- Hypotheses tried (and ruled out, with diagnostic evidence)

## Update protocol

On success: append to `<model>/docs/development_progress/debug_log.md`:

```
## debug-multi-launch-merge recovery (YYYY-MM-DD)
- Failing merge: <group_name>
- Hypothesis fired: 1 / 2 / 3 / 4 / 5 / 6
- Fix applied: <one-line description>
- Verified: merged ELF compiles, output cosine ≥ 0.99 vs unmerged baseline
```
