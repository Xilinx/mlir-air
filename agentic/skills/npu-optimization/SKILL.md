---
name: npu-optimization
description: Optimize NPU kernels during development or complete deployments after model integration, using workload-specific gates, staged evidence and applicable kernel/runtime recipes. Use for prefill, decode, verification or another mode without assuming a fusion scheme or bottleneck.
---

# Optimize at the measured boundary

Own the current task's optimization loop. Read seed/source/artifact identity, workload, reference, fixed kernel/full-model verification criteria, allowed changes/resources, budget and stop rule. An old-valid seed is not a valid baseline for a new requirement. [Workflow](../../WORKFLOW.md) defines development, integration and return paths.

## Select the optimization level

| Level | Candidate decisions | Validity and comparison |
|---|---|---|
| Kernel development | Existing implementation family and applicable registry configuration; computation, tiling, layout, storage, supply and spatial mapping | Use the applicable full-output kernel gate before ranking or validated handoff. Complete-model validity is not a prerequisite for local tuning. Carry integration obligations with the kernel. |
| End-to-end deployment | Host processing, BO lifetime, data handoff, synchronization and submission; return to kernels when evidence points there | Begin from the current-valid integrated model, preserve its complete task gate and compare the full-call/request boundary. Changed kernels return through local and affected integration validation. |

Candidates may be explored before validity is established, but missing required evidence excludes them from the corresponding best-valid ranking. Keep local kernel and complete-deployment validity separate. If no current-valid candidate exists, use scoped diagnosis/development to establish one; do not rank failures as successful optimizations.

## Fix the comparison

Record arithmetic, shapes, live state/capacity, host/device work, warmup/reset/order, preceding device state/program, sampling and uncertainty. Reuse evidence only when its dependencies and coverage apply. Prefill, AR decode and verification share this organization, not their inputs or state semantics. A fixed-input verifier call omits drafting, acceptance, host work and emitted-token accounting needed for speculative throughput. Do not assume M=1, memory-bound decode or a universal numerical threshold.

Identify the leading cost and a competing explanation on the actual timed path. When a total latency cannot distinguish them, select a discriminating probe instead of fabricating a breakdown.

| Evidence points to | Direction |
|---|---|
| Primitive shape/implementation or tiling | Trace the caller; use [registry hit/miss](../../references/kernel-registry.md) where it supplies the implementation, then [kernel development](../verify-kernel/SKILL.md) |
| Custom Q4NX fused-decode implementation | Start with its existing builder/C++/packing/runtime and [megakernel change analysis](../megakernel-change-analysis/SKILL.md); check affected kernels and compare the actual fused boundary |
| Repeated host allocation/upload, conversion or dispatch | [BO reuse](../opt-buffer-object-reuse/SKILL.md), [layout alignment](../opt-layout-alignment/SKILL.md), [multi-launch merge](../opt-merge-multi-launch-kernels/SKILL.md) |
| Partition, residency, repeated production, supply or balance | Optional [megakernel module](../../modules/megakernel/README.md) for coupled candidates or probes |

Standalone registry timing does not predict the embedded operator or complete fused-call time: shared buffers, streaming, synchronization and surrounding work change the boundary. Introducing a registry primitive into a custom fused path is an integration candidate requiring compatible arithmetic/ABI/packing, resource and supply checks, affected verification and measurement. An absent row does not by itself justify abandoning the existing mapping.

Check each recipe's assumptions against the workload. Ordinary recipes need no megakernel analysis. Keep-current or declining to rank is valid when evidence is insufficient. Freeze predictions before observing candidate results only when prospective guidance is being evaluated.

## Check, compare and return

Use the same [static → compile-time → runtime checks](../../references/staged-checks.md) for operator and megakernel candidates, and as applicable at the complete-deployment level: inspect known necessary conditions; inspect matching build diagnostics/artifacts when required; establish the applicable kernel or full-model gate before valid performance comparison. Return object, condition/verdict, evidence/coverage and next action to the same record. `UNKNOWN` creates an evidence obligation; an unsupported optional tool is not an extra gate.

Host-only changes can reuse unchanged NPU compilation with a recorded dependency/ABI/artifact basis. BO initialization, layer/weight generations, aliasing and state changes still require affected runtime checks. Builder, flags or stitching changes may require new NPU artifacts even when edited in Python.

Apply coupled changes and the fixed gate without relaxing acceptance for speed. Measure the complete selected boundary against a comparable valid baseline. Kernel-only improvement can vanish after integration; do not replace the best-valid deployment until its complete gate and timing pass. A submit-and-wait timer is not pure device time. Label differences below resolution inconclusive.

Keep candidate/build/artifact/gate/measurement identities, failures, timeouts, slow variants and interventions in the [task record](../../references/task-record.md). Stop at the target, budget or lack of useful authorized actions. Kernel work returns the best locally valid implementation, ABI/constraints, scoped timing and integration obligations to [kernel verification](../verify-kernel/SKILL.md); deployment work returns the best complete valid result to [finalization](../finalize-deployment/SKILL.md). Neither result proves optimality.
