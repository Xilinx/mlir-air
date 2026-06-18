# Optimization Skillset Refactor — Design

**Date**: 2026-06-18
**Worktree**: `.claude/worktrees/skills-upstream` (branch `skills-upstream-v2`)
**Touches**: `.claude/skills/{phase-4-prefill-optimization, phase-5-decode-optimization, merge-multi-launch-kernels}` + 2 new skills

## Motivation

The paper (Table I + Fig 2) describes optimization as a **shared skillset of
independent skills, each solving one class of optimization problem**, that
Phase 4/5 *draw on*. The current implementation only realizes one of the
three: `merge-multi-launch-kernels` is a standalone skill, but the other two
optimization classes (buffer-object reuse, layout alignment) are inlined as
Patterns B/C inside phase-4 and phase-5, plus a fourth inlined Pattern D
(CPU→NPU promotion) that appears neither in the paper's Table I nor in the
ablation study.

Consequences of the current shape:
- phase-4/5 are bloated (~200+ lines each), and the same optimization class
  (BO reuse) is written twice (once per phase) — duplication that can drift.
- Adding a new optimization class means editing a phase skill body, not
  adding an independent skill — the opposite of the paper's "expected to grow
  gradually" design.
- The `.claude/skills/` tree does not match the paper's Table I.

## Evidence base

Two sources ground the skill boundaries:

1. **Paper Table I / Fig 2**: three optimization skills — *Multi-kernel
   merging*, *Buffer object reuse*, *Layout alignment*. Phase 4/5 "draw on"
   them; any phase can auto-invoke a debug skill.
2. **Ablation study** (`llama-3.2-1B-devel` branch,
   `programming_examples/llama32_1b/ablation/`): a 4-cell ladder
   (A naive → B → C → D production) that isolated three dispatch/host
   optimizations and measured each one's contribution:
   - **#2** per-layer weight BOs (`static_input_indices`): A→B
   - **#3** shared intermediate BOs (`intermediate_indices`): B→C
   - **#1** multi-launch ELF stitching (15→3 dispatch/layer): C→D
   - Headline: prefill A→D = **1.56×**, decode A→D = **2.83×**.

   The ablation covers merge (#1) and BO-reuse (#2+#3). Layout alignment is
   in the paper's Table I and §III-B (Step 9) but was not isolated in the
   ablation. CPU→NPU promotion is in neither.

## Design

### Three independent optimization skills (shared)

```
optimization skills (each = one problem class, each self-checks correctness):
├── merge-multi-launch-kernels   (exists)  — dispatch fusion;  ablation #1 (C→D)
├── buffer-object-reuse          (NEW)     — weight + intermediate BO reuse; ablation #2+#3 (A→C)
└── layout-alignment             (NEW)     — seq-first / drop host transposes; Table I, §III-B Step 9
```

Each follows the existing `merge-multi-launch-kernels` structure: frontmatter
`description` (what + when, "invoked by phase-4/5 ..."), **Purpose**,
**Success criteria** (self-check), **Knowledge base references**,
numbered **Workflow** steps, **Failure modes** table, **Update protocol**.

### `buffer-object-reuse` (new skill)

- Merges ablation #2 (per-layer weight BOs, `static_input_indices`) and #3
  (intermediate BOs, `intermediate_indices`) — the ablation showed both are
  the same BO-management class, and Table I lists them as one skill.
- Parameterized for prefill vs decode: prefill pre-loads per-layer weight BOs
  once in `prepare_runtime`; decode amplifies the win (weights reused on every
  token), so static weight BOs are the dominant decode host-side optimization.
- **Success criteria**: after applying, output cosine ≥ 0.99 vs the
  pre-optimization baseline (BO reuse must not change the math) + `make verify`
  still PASSES + measured host/wall time strictly lower.
- **Failure modes**: NaN/garbage on 2nd+ call (per-layer BO key collision, or
  `static_input_indices` wrong) → invoke `debug-bo-corruption`.
- **Knowledge base**: `llama_kernel_builder/cache.py`
  (`static_input_indices`/`intermediate_indices` mechanics);
  `llama32_1b/multi_launch_builder/*` BO usage as the worked example.

### `layout-alignment` (new skill)

- Choose activation layouts so consecutive kernels hand off without a host
  transpose. The canonical case: seq-first `(seq, n_heads·head_dim)` so RoPE →
  FlashAttention → O-proj stay on-device.
- **head_dim ≥ 128 caveat**: the seq-first `dk_chunks > 1` path has known
  runtime issues; route head_dim ≥ 128 through the head-first wrapper (host
  transpose). For the *why*, defer to `debug-fa-runtime-failure`; the skill
  states only the *what*.
- **Success criteria**: cosine ≥ 0.99 vs baseline + `make verify` PASSES +
  the targeted host transpose(s) removed (fewer host ops, lower wall time).
- **Failure modes**: cosine drops after switching layout (a kernel didn't
  actually accept seq-first) → revert; FA hang/NaN at head_dim ≥ 128 →
  `debug-fa-runtime-failure`.
- **Knowledge base**: `flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py`;
  the head-first wrapper path.

### phase-4 / phase-5 slimmed to thin orchestration

Both phases keep existing (paper has Phase 4 + Phase 5) but shrink to:

1. **Measure the Phase 3 baseline** (kernel + wall time; `make profile`).
2. **Decide which optimization classes apply** — prefill: all three may
   apply; decode: merge + BO reuse, layout usually N/A (skip reasons logged).
3. **Trigger the matching optimization skill(s)** from the shared set,
   passing the prefill/decode difference as a parameter (GEMM vs GEMV; which
   classes apply). The phase does not restate the recipe — the skill owns it.
4. **Re-run `make verify` after each applied skill** (the gate; per the
   verify-subsystem alignment, `make verify` token-set is PASS/FAIL,
   `make diagnosis` is localization only).
5. **Record** per-skill `applied / skipped / reverted` + latency delta in
   `<model>/docs/development_progress/phase{4,5}_*.md`.

The "≥ N skills applied" count is NOT a gate — some models legitimately need
only a subset. The gate is the outcome (perf improved + `make verify` PASSES).

### Removed: Pattern D (CPU→NPU promotion)

Deleted from phase-4 and phase-5. It has no ablation backing and is not in the
paper's Table I; the kernel-first default already runs ops on NPU, so there is
no standing "residual CPU op to promote" class worth a pattern.

## Out of scope (this refactor)

- No new ablation runs / no NPU measurement — this is a skill-doc refactor.
- The verify-subsystem code is untouched (gate semantics already aligned).
- Future optimization classes (quantization, dataflow) the paper lists as
  "open directions" are not added now — but the new structure makes adding one
  = adding a skill file, which is the point.

## Consistency with the paper

- ✅ Table I's three optimization skills all exist as standalone skill files.
- ✅ Fig 2 "Phase 4 and 5 draw on optimization skills" becomes literally true
  (the phases are thin orchestrators that dispatch the shared skillset).
- ✅ ablation #1 / #2+#3 back merge + buffer-object-reuse with measured deltas.

## Risks / notes

- `buffer-object-reuse` must parameterize prefill vs decode cleanly without
  becoming two skills in a trenchcoat — the ablation confirms they're one
  class, so the skill body should lead with the shared mechanic and note the
  two deployment contexts.
- Cross-references: slimming phase-4/5 must not leave dangling pointers; the
  failure-mode rows that referenced inlined Patterns B/C/D need updating to
  point at the new skills (or be removed for D).
- Phase 4/5 still own the kernel-first-vs-inheritance decision *reference*
  (defined in phase-2 Step 1) — unchanged; the optimization skills inherit
  that verdict, they don't restate it.
