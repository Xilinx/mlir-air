# Shared task and relationship record

Use the task's existing record if it already covers these facts. This is a small optional shape for one evolving record, shared by Optimization and its optional module calls, not a separate report per skill or a new schema/tool requirement. Omit irrelevant fields and refer to existing evidence rather than copying it.

## Establish once, update only when the task changes

- **Task:** new deployment / valid-seed adaptation / valid-seed optimization / recovery; seed source/artifact and old/new requirement; allowed edits and resources; active budget and stop condition.
- **Contract:** workload/execution mode, shapes and live context/capacity, input/state, reference and actual arithmetic; fixed kernel/full-model verification criteria and applicable stages; target and timing boundary; repeat/reset/stability rules. Mark missing definitions and who owns acceptance. Never fill missing standards with an old recipe's defaults.
- **Active scope:** kernel development/optimization, model integration or optional block diagnosis, or end-to-end optimization; local handoff evidence and complete-task validity stay separate. Record the performance boundary and any return from deployment to kernel work.
- **Reuse:** which seed evidence still applies, why its relevant dependencies/settings are unchanged, and which evidence the change invalidates. Record validity separately for the old and new requirement: an old-valid seed is a rollback/reference, not completion of the new task. If no candidate meets the new requirement, say so.

## One entry per meaningful decision

| Field | Minimum useful content |
|---|---|
| Observation | Concrete change, failing boundary or timing; source/artifact/input/run identity and evidence level |
| Hypotheses and evidence | Leading explanations, supporting and contrary observations; separate extracted/derived facts from declarations |
| Affected objects | Semantic value including phase/state generation; producer → owner/storage → consumer; layout/addresses, lifetime and synchronization only where relevant; source locations |
| Candidates and coupled edits | Keep current design as an option; identify code/config/runtime objects that must change together and prerequisites |
| Next evidence | One check/experiment or candidate validation with expected observations that distinguish explanations; actual available tool or explicit missing capability |
| Unknowns and acceptance | What remains unknown, applicable kernel/full-model gates, result/next action; finite PASS scope, failure or abstention and stop condition |
| Staged feedback | Object/source anchor → condition/verdict → static/compile-time/runtime evidence and coverage → next action; mark tool-executed, agent-derived, declared or not checked |

Keep source count, emitted artifact property, measured time and prediction distinguishable. For a new candidate bind source snapshot, build/config/toolchain, actual artifact and gate/measurement identities; a base commit alone may omit local source changes. If no build or execution occurred, say so.

Reference the [host helper output](../profiles/README.md) when used; retain its source status, prerequisites and described entrypoints without promoting them to gate results. Reuse compilation according to [dependency evidence](staged-checks.md), naming invalidated runtime properties even when the binary stays unchanged.

Update best **valid** performance separately for locally valid kernels and complete deployments, only for candidates meeting that current/new requirement and its relevant gates; do not include an old-only valid rollback seed. Preserve attempted candidates, failures/timeouts, negative results and interventions with available time/usage provenance; unknown cost stays unknown. Independent acceptance and final remeasurement remain separate from the developer's diagnostic result.

Evidence interpretation and optional local integrations: [tools and evidence](tools-and-evidence.md). Main entry: [workflow](../WORKFLOW.md).

## Handoffs and costs when relevant

Record implementer/reviewer/evaluator roles and candidate ownership, questions/findings/rework, whether observations were independently obtained, and remaining contract obligations. Preserve per-role invocation/usage provenance, active/wall/wait/compile/device time, failures and interventions. Aggregate agent effort and elapsed time are different; cached/reasoning fields may be subsets and missing fields remain unknown. See [review and handoff](review-and-handoff.md).
