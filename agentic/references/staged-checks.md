# Staged checks and evidence reuse

Use this reference when selecting a check, interpreting feedback or deciding whether earlier evidence still applies. Both kernel development and end-to-end optimization use these stages. The task fixes required properties; stage names do not impose an unconditional rebuild or new gate.

## Check the candidate at the supported level

| Stage | Inputs and useful checks | Limits and next evidence |
|---|---|---|
| Static | Source/configuration, interface/packing, declared independent allocations, supported producer-consumer relations and matching [hardware facts](hardware/README.md). Check known divisibility/tails, address units/bounds, transfer counts and storage necessary conditions. Label extracted, derived and declared inputs. | A violation can reject the current allocation/protocol, not every possible mapping. No violation found does not establish compiler allocation, route, runtime progress or semantics. Unmodeled aliases/lifetimes/branches need explicit follow-up. |
| Compile-time | Exact source/configuration/toolchain identity, logs and retained IR/ELF/manifest at a named stage. Inspect actual visible allocations, channel/BD/lock use and placement; inspect routing only when that evidence is retained. | An early IR property does not prove later lowering. Missing stack/bank/program/route/liveness evidence remains unknown. A compile error does not alone establish hardware impossibility, and successful compilation does not establish output/state correctness. |
| Runtime | Matching binary and host ABI/binding, fixed inputs and initial state, actual production path and reference. Check completion and the applicable kernel or full-model gate, then measure valid performance. | Local kernel results cover their listed cases. Model acceptance uses the production top-5 token gate with its actual capture/comparison coverage. Block/layer/state probes remain optional diagnosis or explicit task requirements. Timing requires comparable boundaries and samples; a timeout is an observation, not a proven root cause. |

Count independently allocated data buffers against the selected tile's capacity; do not assume the compiler aliases them because their mathematical lifetimes differ. Keep stack, alignment and actual placement separately accounted. A known capacity overflow rejects that candidate allocation; smaller chunks, different output conversion or ownership may remain alternatives. Similarly, an odd output-block count is not a hardware prohibition: check the actual loop/tail and send/receive protocol. Logical arrows do not directly count physical DMA channels.

These are bounded checks, often requiring source analysis by the agent. Available program checks and their exact coverage are in [tools and evidence](tools-and-evidence.md). The [host profile helper](../profiles/README.md) checks source symbol presence and records hashes only; its source status supplies none of the static-resource, compile or runtime verdicts above.

## Return one actionable feedback item

Append to the existing [candidate record](task-record.md): affected object and source anchor; condition; verdict; evidence identity and stage; checked coverage/exclusions; next action and required prerequisite. Preserve native results rather than forcing every tool into a single success field.

- `PASS`: this property passed over the stated cases and evidence; unrelated conditions remain open.
- `FAIL`: name the violated property and candidate scope, then revise or diagnose it.
- `UNKNOWN`: name missing evidence and a check that can resolve it. Continue useful analysis or authorized later checks; do not accept an unknown required property.
- `UNSUPPORTED`: this tool does not handle the representation. Use an applicable alternative or bounded manual analysis; tool abstention is not a new candidate gate.
- `NOT_RUN` / `NOT_EVALUATED`: no relevant execution/evaluation occurred. `INCONCLUSIVE` preserves an unresolved observation or timing comparison.

For a diagnostic compiler failure, preserve the raw error even when source attribution is unknown. For an unimplemented checker, describe the analysis to perform and missing capability; do not invent a command or report a program-checked result.

## Reuse compilation by dependency, rerun changed behavior

Record the old evidence identity, the new candidate identity, unchanged dependencies and coverage, and the evidence invalidated by the change. A base commit or familiar artifact filename is insufficient; bind relevant dirty/untracked content, build configuration, external objects and actual binary.

NPU compilation can be reused for a host-only change if kernel/builder inputs and flags, generated artifact identity, ABI, and applicable device/toolchain remain compatible. Check the selected runtime still loads that binary. Editing a Python builder, stitching code, shape/precision or compile option is not automatically host-only.

BO key/layer/weight generation, first-use initialization, aliasing, overwrite/completion, dynamic inputs and state changes can leave the binary unchanged while invalidating runtime evidence. Reuse the compile receipt with its reason, then run affected boundary/transition checks and the required complete task gate. Reuse runtime or reference evidence only when its own dependencies and the contract permit it.

## Choose the useful next cost

Prefer an existing source/host check that resolves the current question before a new build or device run. If source cannot settle a required property, obtain the appropriate compile/runtime evidence within the task's resources. No universal cost ratio, automatic early-rejection rate or time/token saving is established. Stop at the agreed budget with the best valid result and remaining obligations.
