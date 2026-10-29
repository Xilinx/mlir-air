---
name: opt-layout-alignment
description: "Align producer and consumer layouts to remove measured host conversions while preserving semantic indices, packing, tails and state in the actual NPU implementation."
---

# Align a producer-consumer layout

Use when an actual host transpose/repacking/conversion lies on the measured path. Profile or inspect boundaries and classify each conversion: necessary semantic transformation, API/layout mismatch, or redundant copy. A transpose view and its materializing contiguous copy can have different costs; confirm the target cost is really paid.

Trace logical indices from producer output through memory shape/strides, DMA addressing and consumer access. Choose a common supported layout and update the coupled builder, packing, kernel flags, runtime view, buffer sizes and state/cache interpretation. Check real/tail/padded lanes and reduction order. Delete a conversion only when the consumer truly accepts the producer layout; moving it elsewhere does not remove its cost.

The workshop seq-first `(sequence, heads × head_dim)` path is one useful option for matching RMSNorm/projection/RoPE/attention boundaries. It is not a universal preferred layout. In the inspected profile, FA builders are under `programming_examples/flash_attention/kernel_fusion_based`; some historical seq-first `dk_chunks > 1` paths failed while head-first worked. Use [FA debug](../debug-fa-runtime-failure/SKILL.md) to recheck version/shape/flags before choosing a wrapper. A head-first wrapper may retain host conversion to keep a valid NPU path.

If padding changes GQA grouping or head indexing, use [optional integration diagnosis](../../references/integration-diagnostics.md); if framing/owner/receive order changes, use [residency and supply](../../references/residency-and-supply.md). A layout change that also changes arithmetic uses its calibrated task gate, not an assumed bitwise comparison.

Apply the affected kernel/full-model gate. Use boundary/state/repeat probes when diagnosing buffer reuse or when the task/interface requires them. Measure the complete target path and record removed or retained conversions. Revert an invalid mapping; preserve a valid conversion when its alternative is unsupported or slower. Return the layout relation, coupled changes, results and timing to [Optimization](../npu-optimization/SKILL.md).
