---
name: debug-multi-launch-merge
description: "Diagnose AIR multi-launch composition failures using emitted IR, symbol/argument mappings and versioned resource evidence, including BD pressure, routing, wrappers and compile scaling."
---

# Diagnose a merged AIR module

Preserve source, emitted IR stage, compile command/configuration, external object identities and the earliest failing pass/error. Use these observations to distinguish malformed composition from physical allocation or unsupported lowering. The [merge recipe](../opt-merge-multi-launch-kernels/SKILL.md) gives the construction procedure.

| Observation | Evidence to inspect | Candidate remedy after confirmation |
|---|---|---|
| BD pool exhaustion | Actual DMA splitting/iteration and descriptor allocation, contributing launches and backend limits | Reduce unnecessary splitting or merge scope; consider semantic-preserving layout/padding only after checking real constraint |
| Channel/route allocation failure | Symbol/channel uniqueness, physical endpoint demand, allocator/route stage and competing paths | Fix true naming collisions; for physical congestion change mapping/scope. Renaming alone does not create hardware channels |
| Herd/tile conflict | Logical herd shapes versus final assigned resources and overlap | Use compatible placement or separate the conflicting launch; different shapes alone need not prevent coexistence |
| Undefined herd-load / missing lowered DMA / zero output | Bare herd wrapper structure, entrypoint and retained lowered path | Apply the selected backend's required launch/segment wrapper and confirm lowered execution |
| Stride/subword diagnostic | Actual element units, contiguity, offset/stride encoding and emitted transfer | Restructure supported layout or split transfer; keep host packing and consumer indices consistent |
| Wrong merged output despite build success | SSA/external symbols, argument map, output slots, layout and scheduling/state | Isolate the recently added boundary and repair the coupled mapping |
| Compile scaling beyond budget | Pass timing, module/launch size and progress | Bound the attempt; keep smaller groups or separate dispatches and record compile cost |

Read [hardware facts](../../references/hardware/README.md) for the selected target; locate the exact backend/implementation restriction in source or diagnostic. Historical 1024 alignment, fixed tile counts, fixed compile-minute cutoffs and particular subword stride rules do not define every AIR revision. A compile/resource PASS does not prove runtime supply or state correctness.

If a proposed padding changes head grouping, use [optional integration diagnosis](../../references/integration-diagnostics.md). If bytes fit but producers/consumers cannot progress, use [residency and supply](../../references/residency-and-supply.md). More channels or buffering may be infeasible; preserving a split implementation is a legitimate candidate.

Rebuild the corrected composition, rerun matched unmerged/merged output and relevant state checks under the frozen task gate, then validate production integration and full target time. Record failed hypotheses and unresolved evidence at the budget boundary; no external debugging skill or unbounded retry is required.
