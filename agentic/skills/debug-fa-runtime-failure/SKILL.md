---
name: debug-fa-runtime-failure
description: "Diagnose NPU FlashAttention NaNs, garbage or timeouts by checking object compile flags, actual layout/chunk paths, resource allocation and supply before choosing a workaround."
---

# Diagnose FlashAttention failure

Bind failing shape, q/kv head grouping, live lengths, mask, layout, precision, input distribution, build flags, object/artifact identity and raw error. A historical head-dimension failure is a hypothesis for the selected implementation, not proof of a current root cause.

| Candidate cause | Check | Next action if supported |
|---|---|---|
| Object flags disagree with IR | Compare actual `.o` flags to builder per-tile geometry and canonical Makefile, including q tiles and full versus chunk dimensions | Rebuild the matching object and dependent artifact in a fresh candidate/cache identity |
| Unsupported/broken seq-first chunk path | Compare seq-first and head-first at the same shape/input/arithmetic; vary only the relevant chunk/layout parameter | Retain a validated head-first wrapper or another supported path, accounting for conversions and state |
| Allocation/resource pressure | Inspect final allocations, banks, staging/stack and resource diagnostics for the failing owner; consult versioned limits | Reduce a supported tile/chunk/buffer configuration or remap ownership with its supply consequences |
| Supply, mask or state defect | Inspect receive order, tails, causal masks, initial state and bound data; compare first/repeated call | Fix the affected protocol or semantics; smaller-shape success does not rule these out |

In the inspected AIR profile, `shared/infra/external_kernels.py:compile_attn_npu2` accepts `head_dim`, `lkp` and `lqp_tile`. Its object flags use per-tile `lqp`, chunk `dk`/`dv` and full `dk_full`/`dv_full`; the caller must derive the q tile size correctly. Read the actual signature and `flash_attention/kernel_fusion_based/Makefile` rather than assuming every helper derives per-launch geometry automatically. Invalidate/rebuild stale dependent outputs when flags change; preserve old failure evidence.

The historic workaround routes a seq-first activation through a head-first FA kernel with explicit reshape/transpose on input/output. Check head/KV grouping and output layout, then validate the target shape and required state. Do not delete those conversions just because a different builder is named seq-first. Whether the issue persists in a new revision requires new evidence.

A hang at one large shape does not establish L1 overflow. Use [hardware facts](../../references/hardware/README.md) and [resource/supply diagnosis](../megakernel-resource-diagnosis/SKILL.md); distinguish necessary storage budgets from actual allocation and liveness. Avoid substituting rough historic FA buffer estimates for the compiled mapping.

An authorized bounded bisect may vary length, head grouping, chunk/layout or q tiling one axis at a time and compare with the appropriate independent attention reference. Classify numerical failure, timeout and unsupported configuration separately. Stop at the task budget with the failing configuration, flags, artifacts and hypotheses if unresolved. After a fix rerun the original failure, affected state/semantic gates and integration; timing the workaround is a separate [Optimization](../npu-optimization/SKILL.md) decision.
