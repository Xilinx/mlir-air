---
name: finalize-deployment
description: "Finalize a validated NPU implementation with a production-linked evaluator, reproducible source and artifact identities, scoped performance results and conditional development lessons."
---

# Finalize the validated implementation

Take the best candidate valid for the current requirement, its gate/measurement evidence and remaining obligations. State whether the deliverable is a locally validated kernel or a complete deployment. Kernel-only work carries its ABI/constraints and pending integration obligations; a model task requires its complete production gate. Preserve both scopes when end-to-end optimization returned through kernel development. Keep [project integration](../../references/project-integration.md) and the actual task contract at hand. Finalization consolidates already validated work; it is not the first time a needed evaluation adapter appears.

1. Consolidate setup, runtime calls and state transitions into the intended production entry. Distinguish compile/load/preprocess/BO initialization, warmup and actual request work. Keep the timed boundary honest: measure setup separately when reporting warm steady state, include it when the task targets cold/request latency. Do not rename prefill kernel time TTFT.
2. Complete the shared-evaluator adapter and project command targets. Confirm the adapter invokes this candidate's actual production functions and current config/weights, not an exemplar or stale path. Inspect arguments, fallback branches, actual artifact loading and report exit behavior. The supported AIR interface is described in project integration; another implementation uses its supplied fixed evaluator.
3. After cleanup or integration edits, rerun affected fixed gates and the task's final production validation. Recheck the full target timing boundary when its dependencies changed; use repeat/reset/state probes for diagnosis or explicit task/interface requirements. Preserve raw data, sample order, warmup/state and uncertainty; do not copy optimization-time numbers onto a changed final binary.
4. Deliver source revision plus local diff/snapshot, build configuration and commands, toolchain/driver/device identity, packed-weight provenance, final artifact hashes, runner/evaluator identities, exact workload, results and reproducible command locations. State missing artifacts or unavailable checks explicitly.
5. Record lessons as trigger → observed evidence → interpretation → coupled remedy → validation scope → counterexample/unknown. Mark historical, reconstructed and newly observed evidence separately. Propose registry/skill/shared-code updates with prerequisites and source versions; repeated reuse alone does not establish universality. Keep shared guidance and gates frozen during controlled comparisons.

The handoff should let another reader reproduce the stated checks without private development chat. Include model/config and key file map (for example `ARCHITECTURE.md`), performance inclusions/exclusions, failed or discarded candidates, unmet target, cost/usage availability and next action. Documentation completeness cannot turn an invalid candidate valid.

Use [evaluation](../evaluate-deployment/SKILL.md) for final production-path audit and applicable checks. Independent review is available through [handoffs](../../references/review-and-handoff.md); a self-check is labeled self-validation when no independent evaluator runs.
