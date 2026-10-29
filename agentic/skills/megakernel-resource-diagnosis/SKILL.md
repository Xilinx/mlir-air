---
name: megakernel-resource-diagnosis
description: Diagnose storage, placement, routing, data-supply, synchronization, and progress constraints in spatial NPU megakernels. Use for compile or route failures, runtime stalls, and mapping changes that alter ownership, partitioning, buffering, or concurrent data lifetimes.
---

# Diagnose resources and data supply

Identify the failing stage and bind its source, configuration, toolchain and artifact/log. Compilation, routing, loading and execution failures need different evidence. Treat a compiler diagnostic as an observation about that stage; do not infer a hardware impossibility or a runtime deadlock from it alone.

First select the actual device, partition and toolchain in [hardware facts](../../references/hardware/README.md). Distinguish per-tile totals from usable allocation budgets and implementation/toolchain limits; do not transfer a compute-tile profile to MemTile or shim.

## Trace the constrained path

Starting at the affected consumer, identify the semantic value and required order, then trace its producer, owner, transfers, staging and synchronization. Use [residency and supply](../../references/residency-and-supply.md) when choosing a new organization. A source-only result should identify what final allocation or execution evidence is still missing.

| Concern | Read or derive | Decision-changing evidence |
|---|---|---|
| Storage | Allocations, aliases, simultaneous lifetimes, staging, stack/spills, constants and alignment | Peak live storage is a necessary budget; actual allocation/address ranges and placement establish a different, still limited fact |
| Layout and transport | Element addresses with units, bounds, stride/offset, tail, number/order of frames and consumer count | Equal payload sizes can still carry the wrong elements; identify the first differing source/consumer element |
| Buffer rotation and progress | Fill/read/release, ping-pong generation, BD chain/loop, lock initial state, waits and completion | Trace one transition and the next reuse; check read-before-fill, overwrite-before-last-read, unmatched counts or cyclic wait |
| Placement and capacity | Logical versus physical owner, accessible banks, DMA channels/BDs/locks, route endpoints | Check final relevant IR/ELF/route stage and versioned limits; summed bytes alone do not prove availability |
| Parallel supply | Fanout/gather, shared supplier, per-stage service rate and backpressure | More compute tiles help only if supply and output paths sustain them; a drain-only probe may not represent the consumer |
| Toolchain restriction | Earliest failing pass, emitted operation and known version-specific handling | Classify hardware fact, implementation restriction, or workaround with a source/version; retain UNKNOWN if attribution is unsupported |

For an unfamiliar organization, manually model one bounded producer-consumer path from the actual code. State unmodeled aliases, branches and completion events. Do not execute arbitrary builder imports just to inspect configuration; use retained facts or an explicitly permitted build path.

## Choose alternatives conditionally

If simultaneous storage exceeds a supported limit, consider shorter lifetimes, smaller chunks, fewer stages, partitioned/remote ownership, or recomputation. Compare their added communication and synchronization before selecting one; [reuse or recompute](../../references/reuse-or-recompute.md) explains when keeping production is reasonable. If bytes fit but supply or progress fails, changing storage capacity alone is not a remedy: investigate element order, wait dependencies and rate mismatch.

A coupled plan identifies the producer/consumer partition, owner/layout, allocation lifetime, transfer/BD sequence, synchronization and runtime binding that move together. Separate conditions established by source checks from those requiring compilation, routing and device observations. Before execution, satisfy known prerequisites and preserve the task's resource authorization; a skill is not a device lease.

Return the violated or unresolved relationship, source/artifact anchors, viable alternatives and the cheapest discriminating check. Use [tools and evidence](../../references/tools-and-evidence.md) for actual finite inspectors and their exclusions. Record PASS only for the checked property; no generic liveness/alias/route proof is supplied. Reuse the optional [task record](../../references/task-record.md), then route output/state failures to [correctness diagnosis](../megakernel-correctness-diagnosis/SKILL.md) or mapping-specific cost questions to [performance diagnosis](../megakernel-performance-diagnosis/SKILL.md). General strategy selection and complete candidate comparison belong to [Optimization](../npu-optimization/SKILL.md).
