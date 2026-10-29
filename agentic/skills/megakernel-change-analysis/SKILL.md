---
name: megakernel-change-analysis
description: Trace which computation, data ownership, lifetime, layout, and supply assumptions change when adapting a spatial NPU megakernel to a new model, shape, execution mode, mapping, or toolchain. Use to select affected objects and validation scope before edits or after a structural change.
---

# Analyze a megakernel change

As part of the [megakernel module](../../modules/megakernel/README.md), produce a source-grounded impact analysis and the next useful check for adaptation, validation or an optimization candidate. Start with the task's seed identity, old/new requirement, allowed edits, fixed acceptance gates, and budget. If already supplied, reuse them. The [workflow](../../WORKFLOW.md) distinguishes new deployment, valid-seed adaptation, optimization, and recovery; do not reset a valid seed to an unnecessary full deployment.

## Find relationships in unfamiliar source

Locate the actual runtime call and its configuration/build entrypoint first. Follow one changed value through its producer, packing/conversion, transfer, storage, and consumer; inspect generated IR or retained artifacts when source cannot settle a question. Search by a called symbol, argument, shape expression, or buffer use rather than assuming file names or tile coordinates.

For each affected relationship, record only what matters:

- **Meaning:** model/layer/phase/token/element identity, arithmetic, mask/position and state generation. Equal shape or byte count does not establish equal values.
- **Organization:** producer/consumer partitions, element order, tails and padding, replication or reduction; relate host packing to consumer indices and transfer addresses.
- **Residence:** owner, aliases, fill, last reader, overwrite/invalidation and reset; check whether newly concurrent consumers extend the live interval.
- **Supply:** transfer count/order, endpoints, staging, reuse slots, DMA/BD and synchronization dependencies; a logical tile name is not final placement.

Bind conclusions to source locations and version/artifact identity. Label facts as extracted, derived, declared, or unknown. Stop tracing unaffected regions once their boundary assumptions and reused evidence are justified; expand across aliases and shared state when those assumptions change.

## Decide what must change or be checked

| Change observed | Follow the dependency | Next evidence |
|---|---|---|
| Dimension, partition or layout | Producer framing → packing → consumer indices → transfer addresses/slot phase | Compare symbolic element identity at boundaries, including tail and transition cases; check compiled transfer organization if unresolved |
| Batch, overlap or owner | Concurrent live data, last-use, fill/invalidation, fanout and backpressure | [Residency and supply](../../references/residency-and-supply.md), then the affected resource/protocol evidence |
| Repeated computation is exposed | Input/parameter/state invariance and number of consumers | [Reuse or recompute](../../references/reuse-or-recompute.md); retain recomputation as a candidate |
| Arithmetic or execution semantics | Reference path, reduction/rounding, mask/position and state updates | [Correctness diagnosis](../megakernel-correctness-diagnosis/SKILL.md); identify existing gate applicability before choosing new tests |
| Device, toolchain or runtime changes | Resource facts, lowering assumptions, artifact/ABI binding | [Resource diagnosis](../megakernel-resource-diagnosis/SKILL.md); separate hardware facts from version-specific restrictions/workarounds |

Return the coupled edit/check set, still-applicable seed evidence, invalidated evidence, and earliest validation stage needed. If a source pattern is unsupported, state the missing relationship and the file/artifact or observation needed to resolve it; a checker abstention is not PASS. Existing tools and their limits are in [tools and evidence](../../references/tools-and-evidence.md).

Use the shared [task record](../../references/task-record.md) only if helpful: observation; hypotheses/evidence; affected objects; candidates/coupled edits; next evidence and expected discrimination; unknowns/acceptance. Do not prescribe an optimization merely because a historical case had a similar shape.

For the AIR Q4NX fused path, read [fused-decode compatibility](../../references/fused-decode-compatibility.md) when changing compiler flags, packed weights, instruction templates, or KV ownership. Its source-specific constraints and example complement this skill; they do not choose a new model's implementation.
