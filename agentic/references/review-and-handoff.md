# Optional review and evaluation handoffs

One implementer owns the candidate and task record. Single-agent execution includes implementing, diagnosing, validating and documenting the final result. Add a reviewer only when requested or when an independent focused check has enough value within the authorized task. No stage requires three reviewers or parallel device runs.

Review either optimization level using its actual scope: a kernel review checks local gate/interface/timing and integration obligations; a deployment review checks complete-task validity and the full target boundary. Review the [staged feedback](staged-checks.md) and any compile-evidence reuse basis. Source binding or a local gate does not substitute for complete model acceptance.

## Development review roles

| Optional focus | Inputs | Concrete output |
|---|---|---|
| Numerical and state | Fixed contract, candidate diff/identity, reference arithmetic, input/state coverage, raw failures and results | Supported semantic claims; missing/incorrect mask, position, row, state or numerical evidence; a discriminating check |
| Hardware and supply | Selected hardware profile, source and retained IR stage, allocations/lifetimes, producer-consumer path and build/route diagnostics | Confirmed/unknown resource or protocol relationships; coupled edits or inspection needed; source anchors |
| Performance evidence | Validity receipts, workload/timing boundary, raw samples/order/state, work/path ledger and proposed choice | Comparability and uncertainty assessment; supported benefit or ABSTAIN; next measurement that could change the recommendation |

A review request names one question, allowed source/artifacts, output path, time/token/device budget, editing ownership and stopping rule. Supply evidence needed to judge that question, not a request to endorse the implementer's conclusion. Reviewers normally read and report; the candidate owner integrates edits to avoid conflicting changes. Shared build artifacts and all device timing/reset operations are serialized under the task's resource arrangement.

The reviewer returns source-specific findings, evidence level, affected acceptance obligations, suggested rework and unresolved items. The implementer records which findings were addressed or rejected with reasons, produces a new candidate identity for changes, and reruns the affected fixed gates. Re-review only the changed or unresolved finding when warranted. A disagreement is resolved by evidence or recorded uncertainty; a review vote cannot pass a device/numerical gate.

## Final evaluation and independence

The [evaluation skill](../skills/evaluate-deployment/SKILL.md) audits the production path and the fixed task evaluator, then establishes the contract's remaining gate and measurement evidence. It can guide a single-agent final self-check; label that SELF-VALIDATED. If the contract requires an independent evaluator, provide a fresh evaluator the final source/build/artifact identities, task contract, raw inputs, commands, environment and budget. They form their audit and observations before reading the developer's claimed conclusions; report their own evidence and any later comparison.

No independent evaluator available means independence is unestablished, not that a self-rerun becomes independent. When independence is contract-required, final independent acceptance remains pending. When it is optional, complete the authorized implementation and report the limitation. Never imply that an optional development reviewer substitutes for final independent execution.

## Future comparisons and aggregate cost

Freeze organization across with/without-skills groups: same number/roles of agents, model/settings, access to base source/hardware facts/tools, total budget and evaluator. A multi-agent effect requires its own contrast. Record all role invocation IDs, start/end and active time, usage receipts, tool/compile/device time, wait time, handoffs, failures and human interventions. Report wall elapsed and aggregate agent effort separately so parallel time is not added as elapsed time. Cached/reasoning usage may be subsets; preserve provider definitions and unknown values. No behavioral evaluation or ablation is executed by this package-authoring task.
