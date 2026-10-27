# Deployment workflow

The entry is [deploy-new-llm](skills/deploy-new-llm/SKILL.md). Start from the HF ID or supplied task, the current source/seed, requested workloads and fixed acceptance criteria. For a new task, use [task preparation](references/project-integration.md); for a follow-up, preserve the existing task and identify the changed dependencies.

## Kernel development loop

Choose a compatible registry configuration, primitive implementation or fused-decode seed. Read [registry scope](references/kernel-registry.md): standalone measured tiles and custom megakernel seeds are different reuse routes. A registry miss does not require decomposing a working fused implementation.

Use [change analysis](skills/megakernel-change-analysis/SKILL.md) to trace affected arithmetic, partitions, packing, buffers, transfers and synchronization. Revise coupled objects together. Apply [staged checks](references/staged-checks.md): source inspection first, then compiler/artifact evidence, then runtime correctness. Reuse unchanged evidence when its dependencies and coverage still match. A tool's UNKNOWN means further evidence is needed for that property.

Apply [kernel verification](skills/verify-kernel/SKILL.md) using fixed operator-specific elementwise tolerances. Optimize valid candidates with [npu-optimization](skills/npu-optimization/SKILL.md); retain the fastest verified candidate under the same measured workload and boundary. A valid slower candidate remains valid. Record isolated coverage separately from complete-model acceptance.

## Model composition loop

[Prepare the reference](skills/build-cpu-reference/SKILL.md), weight packing and model-specific semantics. For the supported affine-Q4 path, prefill GEMM defaults to BFP16; attention, normalization and head precision follow the model contract. Connect production buffers, state handoff and calls, then apply [model verification](skills/verify-model/SKILL.md) to complete finite full-vocabulary captures from the actual runtime.

After the fixed gate passes, profile the complete request. Distinguish TTFT, decode and complete-request time; retain warmup and raw samples. [BO reuse](skills/opt-buffer-object-reuse/SKILL.md), [layout alignment](skills/opt-layout-alignment/SKILL.md) and [launch merging](skills/opt-merge-multi-launch-kernels/SKILL.md) address host/runtime overhead. Return to kernel development when the bottleneck is in device work. Block/layer checks are targeted diagnosis rather than a mandatory extra stage.

## Completion and feedback

[Finalize](skills/finalize-deployment/SKILL.md) the verified production source/artifacts, reference identity, commands and results. [Evaluate](skills/evaluate-deployment/SKILL.md) with the same contracts; label self-validation and independent review accurately. One primary agent can complete the workflow. Additional reviewers require the user's delegation authorization.

Run affected checks after a change and stop after the required successful gates and timing batch. Add probes for a concrete failure, changed dependency or explicit task requirement. Preserve failed attempts and the best valid result if the task stops early. Do not restart an accepted deployment solely because a skill names a preparation step.

Keep one concise [task record](references/task-record.md), linking raw evidence. [Propose reusable kernel records and lessons](references/maintenance.md) with their source/validation scope. Review shared updates normally; freeze shared guidance and assets during controlled experiments. Study-specific A/B protocols, provider accounting and budgets belong to their task contract.
