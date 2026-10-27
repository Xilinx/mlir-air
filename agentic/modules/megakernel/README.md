# Megakernel module: computation, residency and supply

Use this module to identify what a spatial NPU mapping change affects and which check, candidate or measurement is useful next. It is a specialized part of the [development and deployment workflow](../../WORKFLOW.md), available during kernel development, adaptation and verification, or from either scope of [Optimization](../../skills/npu-optimization/SKILL.md). Both scopes use [static, compile-time and runtime checks](../../references/staged-checks.md); the caller keeps kernel verification distinct from full-model verification.

For current AIR Q4NX fused-decode adaptation, the matching existing implementation is the default starting point: trace the model runtime into the fused builder, linked C++ kernels and weight packing before changing its mapping. Use the registry for connected primitive/shared-builder paths or as applicable reference material; a row does not configure custom decode, and a miss does not require standalone composition. [Registry scope and source anchors](../../references/kernel-registry.md) distinguish these routes, including registry-backed prefill within the same model.

| Current question | Skill | Returned evidence |
|---|---|---|
| Which assumptions does a new requirement or structural change invalidate? | [Change analysis](../../skills/megakernel-change-analysis/SKILL.md) | Affected objects, coupled relationships, reusable evidence and required rechecks |
| Why are outputs, positions, masks or cross-call state wrong? | [Correctness diagnosis](../../skills/megakernel-correctness-diagnosis/SKILL.md) | Competing causes, earliest distinguishing boundary, useful checks and remaining obligations |
| Why does a candidate exceed capacity, fail supply/compilation/routing or stop progressing? | [Resource and supply diagnosis](../../skills/megakernel-resource-diagnosis/SKILL.md) | Constrained objects, feasible alternatives and unproven conditions |
| Where does a valid mapping repeat production, lack supply or distribute work unevenly? | [Mapping performance diagnosis](../../skills/megakernel-performance-diagnosis/SKILL.md) | Work/traffic/supply evidence, candidates or distinguishing measurements, abstention and scope |

Read only the entry needed for the current question. The same agent continues the same task record; these entries do not require four agents or a fixed sequence.

For a design choice, read [reuse or recompute](../../references/reuse-or-recompute.md) or [joint residency and supply changes](../../references/residency-and-supply.md) as needed. Keeping the current design, storing locally or elsewhere, chunking and recomputation can each be valid choices. Actual lifetimes, supply, arithmetic/state requirements and cost determine the choice. Public source examples illustrate the reasoning; inspect the selected implementation before applying it to a new shape.

The module returns supported judgments and next actions. The caller continues adaptation or optimization, applies the kernel gate and scoped timing for kernel work, and the full-model gate and complete timing for deployment work. Mapping performance diagnosis does not establish another baseline, gate or budget, or take over ordinary BO/layout/kernel tuning. If new execution evidence is needed, return the specific measurement and its prerequisites to the authorized task.

Inputs are the existing task, source and evidence; no new universal IR or schema is required. The bounded [optional tools](../../references/tools-and-evidence.md) offer support. Unsupported organizations can receive scoped source analysis with unknowns stated explicitly. A missing optional inspector is not automatic candidate failure; required task properties and execution prerequisites still need evidence. Intermediate or state comparisons are optional diagnosis unless the kernel interface or task requires them; they do not create additional global verification levels.

The four skills remain in the package's `skills/` directory. Guidance and source checks do not establish method effectiveness. Match the actual target, partition and toolchain before using [versioned hardware facts](../../references/hardware/README.md). Ordinary failures use [debug routing](../../references/debugging.md).
