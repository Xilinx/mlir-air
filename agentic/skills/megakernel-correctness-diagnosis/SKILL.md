---
name: megakernel-correctness-diagnosis
description: Diagnose numerical, layout, position, mask, or state mismatches in spatial NPU megakernels, including failures that appear only across repeated or integrated calls. Use to select a discriminating test or establish task-appropriate validation coverage.
---

# Diagnose numerical and state behavior

Read the fixed task gate and exact failing candidate/reference identities first. Preserve the raw failure, input/state, call order, position and observation boundary. A reference using different quantization, accumulation, rounding or masking may answer a different question; document that difference before interpreting an error metric. Never relax a frozen gate to pass the candidate.

## Narrow the first supported mismatch

Follow the production runtime's argument binding into packing, transfer and the consumer. Compare matching semantic elements at the earliest observable boundary before scanning later errors. Look for systematic row/chunk/half/phase permutations, stale generations, padding writes and nonfinite values as well as magnitude. Do not classify a mismatch as precision loss merely because arithmetic is low precision.

| Observation | Competing explanations to distinguish | Useful next evidence |
|---|---|---|
| First call differs with correct total sizes | Wrong element order/address/argument, missing computation, or arithmetic difference | Trace a small set of tagged logical elements through packer → transfer → consumer; compare actual arithmetic and compiled path |
| First call passes, repeat or layer transition fails | Stale data, aliasing, incomplete overwrite, slot phase or missing completion | Compare reset/replay/state-carry transitions; inspect per-call BO identity, last reader, overwrite and completion order |
| Error begins at tail, position or one row | Mask/index/padding bounds or incorrectly shared state | Inspect the affected address/position calculation and an adjacent boundary case; check per-row state isolation and preserved regions |
| Local boundary passes, integrated path fails | Upstream state/packing/binding or scheduling effects | Hold input/state constant and inspect the first changed boundary; a CPU substitution can alter scheduling and is supporting evidence, not unique root-cause proof |
| Only reduction/precision changes are implicated | Intended rounding change or an additional semantic defect | Compare against the calibrated arithmetic reference, with separate exact index/mask/state checks |

Choose the smallest check that separates live hypotheses, with predicted observations for each. A host symbolic check can settle element order without a device. Runtime ordering, state carry or compiler behavior may remain unknown until the appropriate authorized observation is available. For ownership, overwrite or wait-order concerns, use [resource diagnosis](../megakernel-resource-diagnosis/SKILL.md) and [residency and supply](../../references/residency-and-supply.md).

## Select coverage without changing the acceptance rule

For an arithmetic-preserving storage/supply change, exact output/state and replay/reset comparisons can distinguish causes. If arithmetic or reduction changes, use the agreed numerical criteria and targeted diagnostics. These probes support the kernel/full-model gates; they are not additional global acceptance levels. If the contract does not cover the intended change, identify the gap for task definition before an acceptance run.

Include state the computation must preserve as well as state it returns: prefix/tail/padding, positions, per-row independence, initialized versus carried state, and repeat/reset behavior where applicable. Component equivalence and same-path equality do not establish full-model quality; apply the full-model token gate and any additional quality requirement only when the task explicitly specifies it.

After a candidate fix, rerun the discriminating failure and the affected fixed gates through the actual production path. A successful local test supports that boundary only. Return supported/excluded/unknown hypotheses, coupled changes and remaining acceptance obligations in the optional [shared record](../../references/task-record.md). Consult [tools and evidence](../../references/tools-and-evidence.md) for finite checker coverage; do not invent a general correctness checker.
