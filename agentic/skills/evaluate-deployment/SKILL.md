---
name: evaluate-deployment
description: "Audit the final NPU production path and fixed evaluator, rerun the contract-required checks and measurements, and report whether results were independently obtained or self-validated."
---

# Evaluate the final production path

Receive the final source/build/artifact identities, model/input/state, fixed task contract, actual commands and environment/resources. This procedure supports a final self-check or an independent evaluator. Label who performed it and what context they received; self-validation cannot be called independent. [Review and handoff](../../references/review-and-handoff.md) defines when independence is required by the task and how to handle pending evaluation.

## Audit before interpreting a PASS

Trace the real command through Makefile/entrypoint → model adapter → production runtime functions → actual artifact and weight/state binding → reference → comparator/report/exit status. Confirm:

- The evaluated function and loaded artifact are the candidate being delivered, with declared CPU/device placement and fallbacks.
- Reference identity, arithmetic and preprocessing match the task's intended comparison; model/config/tokenizer and initial state are correct.
- Kernel reference/tolerances, full-model top-5 semantics, prompt/token capture and error exit behavior implement the frozen contract. No missing/empty/truncated result is accepted as complete evidence.
- Cached outputs and old reports are labeled saved evidence, not a fresh execution. The diagnostic path and acceptance gate have their declared roles.

For the inherited AIR profile, inspect `verify_adapter.py`, shared `verify_runner.py`, `runners/hf_runner.py`, `comparators.py` and `report.py`; current locations and first-divergence top-k limits are in [source integration](../../references/project-integration.md) and [validation contracts](../../references/validation-contracts.md). Do not force this HF BF16 profile on a different task. A misbinding or ineffective comparator is a concrete audit defect; do not infer intent from it.

## Apply the same two verification levels

Using the task's authorized resource window, execute the required final gates and performance checks on that audited path. Reuse still-applicable evidence only if the contract permits it, identifying it as reused. Record raw outputs, exact coverage, candidate/reference identities and fresh/saved status. Block/layer/logit/state comparisons are diagnostic unless the task explicitly requires them; do not add a mandatory numerical suite. Diagnostic cosine is informational unless this contract explicitly gates it. Reviewer opinion cannot replace a runtime/state measurement.

An independent evaluator forms their audit and observations before reading the developer's claimed conclusions, then compares with the claims. Apply kernel verification and full-model verification as appropriate to the deliverable; evaluation is a role, not a third numerical gate. Rerun affected or required properties and justify reuse. In a performance task, final comparable measurement and uncertainty must support the delivered performance claim.

Return a concise evaluation record: audit findings with source anchors; checks executed/reused/not executed and values/verdicts; performance boundary and evidence; SELF-VALIDATED or INDEPENDENT; remaining limitations and exact reproduce commands. Required failure yields FAIL; unavailable required evidence yields NOT_EVALUATED/INCOMPLETE rather than PASS. Send concrete defects back to the candidate owner for a new identity and affected-gate recheck; do not edit the same candidate concurrently. Stop at the evaluation budget and report what remains.
