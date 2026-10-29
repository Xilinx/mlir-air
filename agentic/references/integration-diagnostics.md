# Optional integration diagnosis

Use a bounded block, layer or state comparison when a kernel or full-model failure needs localization. This is a diagnostic technique, not a separate acceptance stage. A passing block does not replace either [verification level](validation-contracts.md), and a deployment need not run a mandatory block test before full-model verification.

## Inspect the affected composition

Read the kernel handoffs: actual ABI, packing/weight assumptions, state/lifetime requirements and unverified properties. Reuse a seed block only when computation order, shapes, biases/norms, arithmetic, layouts and state agree. A model-family label is insufficient. For a new composition, use a matching shared builder as an example and compose only the needed kernels.

For each affected intermediate, identify semantic axes, representation, strides, producer, consumer, storage owner, lifetime and state generation. Bind host arguments to the built ABI; inspect transfers alongside kernel indices, tails and padded lanes. Use the production functions for captures or substitutions so the diagnosis reaches the actual failing path. [Project integration](project-integration.md) lists current AIR locations. Apply [staged checks](staged-checks.md) to changed composition, with justified reuse of unchanged build evidence.

A Q/K norm between projection and RoPE may require splitting an old fused group. A per-head norm must use the correct head axis and shared weights. Reuse registry operations after matching shape/arithmetic rather than assuming new kernels are necessary.

## Conditional techniques

- **GQA-aware padding:** when a lowering benefits from padding, preserve query-to-KV group membership. Inserting phantom Q heads within each KV group differs from appending heads. Track zero/ignored lanes through output projection and cache; a mapped real-output comparison with an unpadded reference can diagnose mistakes. Recheck alignment in the chosen backend instead of treating a fixed 1024 alignment as an architectural requirement.
- **Bias around a bias-free fused path:** RoPE linearity can allow correctly positioned `RoPE(bq)`/`RoPE(bk)` after a bias-free RoPE to represent bias-before-RoPE. Follow V bias through its actual path. Intervening nonlinear operations, such as Q/K norm on biased vectors, invalidate that equivalence. Derive the transformation from the model's computation order and position/head/padding indexing; finite-precision movement can change rounding, so retain the applicable kernel tolerance.

## Compare only what distinguishes the hypothesis

Use matched inputs and initial state at the selected boundary. Capture relevant outputs, semantic intermediates, state or preserved regions; inspect per-position/row magnitude and direction as useful. Cosine is diagnostic and can hide scale errors. Keep these observations separate from kernel elementwise PASS and the full-model token gate.

A shape-preserving permutation suggests packing/layout/address; first-call success followed by failure suggests identity/lifetime/scheduling; a tail-only defect suggests masking/padding or bounds. Capture or replace one boundary at a time, predicting how competing causes differ. CPU substitution changes scheduling/transport as well as arithmetic, so improvement does not uniquely identify a guilty kernel.

A sudden layer cliff motivates checking layer indexing/binding; gradual drift motivates checking arithmetic and error magnitudes. If prefill appears correct and generation fails later, inspect cache layout, offsets, updates and the prefill-to-decode transition. If intermediate states agree but output tokens differ, inspect final norm/head and token mapping. State cases such as repeat/reset, independent rows, append/commit/rollback or untouched prefix are selected for a concrete failure or required interface/task, not imposed as a universal extra suite.

Return source/artifact/input identities, the boundary actually compared, observations, hypotheses resolved and remaining unknowns. Continue through [debug routing](debugging.md), [megakernel correctness](../skills/megakernel-correctness-diagnosis/SKILL.md) or [residency and supply](residency-and-supply.md) as warranted. After a fix, rerun the affected kernel/full-model gate; diagnostic improvement alone is not acceptance.
