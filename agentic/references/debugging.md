# Debug routing and bounded diagnosis

Preserve the failing candidate, raw error/output, input/state, call order, build configuration and artifact identity before modifying it. Determine whether the first failure is in configuration/reference loading, build/lowering, routing/loading, first execution, repeated execution or a performance comparison. Read the earliest failing stage instead of interpreting every symptom as a hardware resource limit.

| Observation | Starting capability |
|---|---|
| Model/config/weight/tokenizer mismatch | [Reference](../skills/build-cpu-reference/SKILL.md) |
| Leaf output or shape mismatch | [Component validation](../skills/verify-kernel/SKILL.md) |
| First/repeat/integration BO corruption | [BO debug](../skills/debug-bo-corruption/SKILL.md) |
| FA NaN, garbage or timeout | [FA debug](../skills/debug-fa-runtime-failure/SKILL.md) |
| Merged AIR build, name/argument or resource issue | [Multi-launch debug](../skills/debug-multi-launch-merge/SKILL.md) |
| Wrong element order, position, mask or state | [Megakernel correctness](../skills/megakernel-correctness-diagnosis/SKILL.md) |
| Capacity, wait, route or supplier/receiver problem | [Megakernel resources](../skills/megakernel-resource-diagnosis/SKILL.md) |
| Valid output, unexpected time | [Optimization](../skills/npu-optimization/SKILL.md) |

The recipes are hypotheses from real development, not exhaustive fault classifiers. When the cause is not established, compare plausible explanations and choose the smallest source/artifact check or authorized observation that predicts different results. A decisive source defect need not be padded with an artificial competing hypothesis. Trace one semantic value from runtime argument through packing, transfer, storage and consumer. Compare against a known-valid configuration with one meaningful difference; preserve shape, input/state and measurement boundary wherever possible. Source checks may settle element identities; ordering and runtime progress may require later observations.

Change the related set of objects together, then rerun the original discriminating case and affected contract gates. A successful small test establishes that boundary only. If the result does not distinguish the hypotheses, say INCONCLUSIVE and select a better observation or stop at the budget. Record attempted hypotheses, contrary evidence, failed remedies and the next specific prerequisite; no unbounded retry loop or external general-debug skill is required.

A timeout at a larger shape is evidence of noncompletion, not by itself proof of L1 overflow. All-zero output can be a missing lowered path, wrong binding, input issue or arithmetic defect. Finite capacity/BD/lock checks do not prove receive order, liveness, valid state or speed.
