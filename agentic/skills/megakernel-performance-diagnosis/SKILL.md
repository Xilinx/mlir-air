---
name: megakernel-performance-diagnosis
description: Diagnose performance tradeoffs in a spatial NPU megakernel's partitioning, residency, repeated production and data supply. Use to return a mapping candidate or discriminating measurement to an existing optimization task, rather than to manage general runtime optimization.
---

# Diagnose a mapping's performance relationships

Use the caller's baseline, performance target, workload, arithmetic, source/artifact identity and fixed gate at the stated scope: a locally validated kernel during development or a current-valid deployment after integration. This is the mapping-specific part of [Optimization](../npu-optimization/SKILL.md), not another optimization loop. If validity or an acceptance property is unresolved, report it to the caller; scoped source diagnosis can still proceed without declaring a performance-valid candidate. Return coupled checks through [static, compile-time and runtime evidence](../../references/staged-checks.md); local performance remains separate from complete-deployment adoption.

## Account for the constrained mapping path

Read the loops and supply path for the suspected value. Separate unique semantic values, production repetitions, fill/copy/replay, retained consumer receives and synchronization. Trace producer → owner/storage → consumers with actual partitions, layout and live intervals. Source work counts, semantic bytes and physical traffic are different quantities; label their domains and unknowns.

| Mapping hypothesis | Alternatives worth comparing | Evidence that can change the recommendation |
|---|---|---|
| An invariant value is repeatedly produced | Recompute, retain locally, replay remotely, reuse only chunks | Reuse count, changed live storage, fill/copy/synchronization and actual supply cost; [reuse or recompute](../../references/reuse-or-recompute.md) |
| Consumers starve or stages serialize | Change owner/partition, buffering, transfer order, or retain mapping | Supply/consumer service rates, backpressure, actual overlap and resource feasibility; [residency and supply](../../references/residency-and-supply.md) |
| Compute is uneven across stages | Change work assignment, chunking or parallelism | Stage work/timing with matched inputs and communication; extra compute can expose a new supplier or output bottleneck |
| A residency change adds movement or waiting | Shorten lifetime, change stage/owner, stream chunks, or keep original | Removed production versus added transport, synchronization and contention along the full affected path |

Do not assume weights dominate, decode is memory-bound, or more tiles guarantee more throughput. If the dominant evidence instead points to generic host allocation/upload/dispatch, return that finding to Optimization for the corresponding standard recipe; this module need not own it.

## Return a candidate, a discriminating probe, or abstention

Compare plausible savings with new production, storage, movement and schedule costs. State competing hypotheses, source objects/coupled edits, expected direction or interval, assumptions and an observation that would reverse the choice. A missing path cost or overlap may make a probe more useful than an edit; keep-current and ABSTAIN are legitimate outputs.

Match any probe to endpoints, memory level, layout/size, repetition, fanout, consumer demand, dispatch boundary and device/toolchain. Distinguish theory, matched path references, full-candidate observations and predictions. Independent probe times cannot be summed into a fused breakdown without a justified schedule/overlap model; one path's throughput is not universal bandwidth.

Return the diagnosis and next action with scope/uncertainty, affected resource and semantic checks, and the caller's full-boundary measurement obligation in the optional [task record](../../references/task-record.md). [Resource diagnosis](../megakernel-resource-diagnosis/SKILL.md) can resolve placement/supply feasibility; [correctness diagnosis](../megakernel-correctness-diagnosis/SKILL.md) can resolve arithmetic/state implications. Use only applicable [tools and evidence](../../references/tools-and-evidence.md).

Optimization retains responsibility for execution authorization, fixed gates, comparable complete measurements, best-valid tracking and stopping. Freeze predictions before unseen results only when prospective decision support is being evaluated; a historical explanation is not evidence of predictive value.
