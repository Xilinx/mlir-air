---
name: debug-bo-corruption
description: "Diagnose stale, aliased, incorrectly initialized or bound NPU buffers when a kernel fails in a pipeline or across calls, including failures on the first invocation."
---

# Diagnose BO identity and contents

Preserve the raw failure, actual input/state and call sequence. First-call failure does not exclude BO/address defects; repeat-only failure is a strong clue about identity or lifetime. Read the runtime's real transfer and allocation logic, using [project integration](../../references/project-integration.md) for current AIR paths.

| Hypothesis | Distinguishing evidence | Conditional action |
|---|---|---|
| Wrong static generation or key collision | Print/inspect layer, artifact, shape and BO key; trace first upload and later skipped writes; compare wrong values with prior calls | Separate keys or refresh the intended generation through supported runtime behavior; keep changing inputs dynamic |
| Incomplete initialization or overwrite | Trace every element read before device write, including tail/prefix/state; inspect first and subsequent host writes | Restore required initialization; mark intermediate only if write-before-read/valid producer supply is established |
| Aliased live buffers | Trace physical BO identity, producer completion and last consumer; observe when saved returned views change | Separate incompatible lifetimes or establish completion; copy host-visible results that must outlive reuse |
| ABI/segment/address mismatch | Compare signature slot to bound BO, packed source range, offset units and accessed end | Repair the complete packing/offset/binding relation; equal sizes alone do not establish provenance |
| Lowered execution path absent | Inspect retained IR and wrapper/launch structure; distinguish input/binding error from missing DMA/compute | Use the selected backend's supported wrapper/lowering path, then rebuild and verify actual artifact |

For the inspected KernelCache, static and intermediate lists skip writes only after first use; output arrays can alias reused BO memory. Adding these lists indiscriminately can hide or create an initialization defect. [BO reuse](../opt-buffer-object-reuse/SKILL.md) gives exact mechanics.

Large weight BOs also require coverage and address representation checks: group-local offsets, each segment's packed source, padding/DMA footprint, host argument order and consistent grouping across the supplier schedule. The historical 32-bit shim-offset limitation constrains one base-relative encoded offset, not total DDR capacity. Recheck the current backend rather than assuming a universal 4 GiB device limit. A sizing helper can pass when two equal-length source segments are swapped.

Choose the smallest source/host check or authorized observation separating plausible causes. For repeat/state/slot transitions, use [correctness diagnosis](../megakernel-correctness-diagnosis/SKILL.md); for receive ordering use [resource diagnosis](../megakernel-resource-diagnosis/SKILL.md). After the coupled fix rerun the failing sequence, relevant repeat/reset/layer/input transitions and fixed production gates. Record observations and unresolved properties in the shared task record. Do not replace a fixed exact/state gate with a universal cosine test.
