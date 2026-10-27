---
name: opt-buffer-object-reuse
description: "Reduce redundant NPU host allocation and transfers when BO input generations, device overwrite coverage and lifetimes permit reuse; validate first-use, repeat and state behavior."
---

# Reuse BOs conditionally

Use when [Optimization](../npu-optimization/SKILL.md) observes repeated allocation/upload or host handoff. Identify the actual runtime first: the inspected AIR `KernelCache` is in `llms/shared/infra/cache.py`; other megakernels may use another ABI. Read [project integration](../../references/project-integration.md) and the current implementation before selecting slot options.

## Classify slots by value and lifetime

| Slot | Requirement before skipping a host transfer |
|---|---|
| Weight or LUT | Contents are invariant for this BO identity across calls; correct generation has actually been uploaded |
| Kernel output/scratch | Every subsequently read element is initialized by the device before that read; preserved prefix/tail/state is accounted for |
| Changing activation/position/state | Upload or supply the new generation through the actual producer; shape equality is not invariance |
| Shared producer-output/consumer-input | Producer completion, exact layout/size, consumer read interval and final reader permit aliasing |

Build the slot map from the generated function signature and runtime binding, not an old argument list. Per-layer weights need distinct keys/generations; varying shape/precision/artifact also requires a compatible BO set. Do not assume a cached key resizes itself.

## Current AIR mechanics

- `load_and_run(..., static_input_indices=..., bo_key=...)` uploads static inputs on first use of the BO key and skips subsequent host writes/syncs. Allocate/populate during setup or first use, and account for first-use cost separately. A weight/config change needs a fresh compatible key or explicit supported refresh.
- `intermediate_indices` skips later writes for device-overwritten buffers. **In the inspected implementation it still writes on first use**, despite the shorter docstring wording. Check initialization and aliasing on the first call; do not assume this option eliminates every write or supplies an unwritten value.
- `shared_nonstatic` can pool transient buffers across per-layer keys **within the same KernelCache instance** (`self._shared_bos`); cross-kernel `shared_alias` handoff requires that same instance. This relies on compatible sizes and sequential nonoverlapping use. `shared_alias` gives producer and consumer slots a shared pool name. Static indices cannot be aliased by this API. Initial host writes can clobber a producer's value on a first consumer call, so inspect the actual initialization/supply sequence before selecting zero-copy handoff.
- `output_indices` controls device-to-host readback (default is the last argument). Returned arrays are views into reusable BO memory; copy a result when it must survive a later overwrite for comparison or application use.

Keep changed activations dynamic, preserve state not wholly overwritten, and check all shared consumers/async completion before extending the sequential-use assumptions. Skipping writes is not a generic corruption remedy.

A host-only BO change can reuse unchanged NPU compilation with the [dependency and ABI evidence](../../references/staged-checks.md) recorded; a Python builder or changed binary is not covered by that assumption. BO identity, initialization, layer/weight generation, aliases and dynamic supply still invalidate affected runtime evidence. Apply the relevant kernel/full-model gate. Probe first use, repeated calls, changed inputs/layers and reset/carry transitions when needed for diagnosis or an explicit task/interface requirement. Use [BO debug](../debug-bo-corruption/SKILL.md) on stale/alias failures. Measure actual host and full-call time against the valid baseline; fewer transfers need not improve the target. Return applied/skipped/reverted, identities, coverage and comparable timing to Optimization.
