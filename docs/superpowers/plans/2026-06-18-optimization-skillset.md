# Optimization Skillset Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the inlined phase-4/5 optimization patterns into a shared skillset of three independent optimization skills (merge / buffer-object-reuse / layout-alignment), and slim phase-4/5 into thin orchestrators that dispatch them.

**Architecture:** Two new SKILL.md files mirror the structure of the existing `merge-multi-launch-kernels` skill (Purpose / Success criteria with self-check / Knowledge base / Workflow / Failure modes / Update protocol). phase-4 and phase-5 are rewritten to: measure baseline → decide which optimization classes apply → trigger the matching shared skill → re-run `make verify` → record. Pattern D (CPU→NPU promotion) is deleted.

**Tech Stack:** Markdown SKILL.md files under `.claude/skills/`. No code, no NPU runs. Verification is by grep/consistency inspection (these are agent instruction docs, not executable code).

**Reference (read before starting):**
- Spec: `docs/superpowers/specs/2026-06-18-optimization-skillset-design.md`
- Template skill: `.claude/skills/merge-multi-launch-kernels/SKILL.md` (the structure all optimization skills follow)
- Current phase-4: `.claude/skills/phase-4-prefill-optimization/SKILL.md` (source of Pattern B/C content to extract)
- Current phase-5: `.claude/skills/phase-5-decode-optimization/SKILL.md`

---

### Task 1: Create the `buffer-object-reuse` optimization skill

**Files:**
- Create: `.claude/skills/buffer-object-reuse/SKILL.md`

This skill merges what phase-4 Pattern B (per-layer weight BOs + intermediate
BO reuse) and phase-5 Pattern B (static weight BOs) describe today. The
ablation isolated these as #2 (`static_input_indices`, A→B) and #3
(`intermediate_indices`, B→C) — one BO-management class.

- [ ] **Step 1: Read the source material**

Read these to copy exact mechanics (do not invent API names):
- `.claude/skills/phase-4-prefill-optimization/SKILL.md` Pattern B (B1 per-layer BO pre-loading, B2 intermediate buffer reuse)
- `.claude/skills/phase-5-decode-optimization/SKILL.md` Pattern B (static weight BOs)
- `.claude/skills/merge-multi-launch-kernels/SKILL.md` (structure to mirror)

- [ ] **Step 2: Write the skill file**

Create `.claude/skills/buffer-object-reuse/SKILL.md` with this exact content:

```markdown
---
name: buffer-object-reuse
description: Optimization skill — reuse NPU BufferObjects across calls instead of re-allocating/re-writing them. Two mechanics in one class: (B1) per-layer weight BOs pre-loaded once and skipped via static_input_indices, and (B2) intermediate BOs the kernel overwrites, skipped via intermediate_indices. Invoked by phase-4-prefill-optimization and phase-5-decode-optimization to cut redundant host↔NPU data movement. Decode amplifies the weight-BO win (weights reused on every token).
---

## Purpose

NPU kernels re-allocate and re-upload BufferObjects (BOs) on every call by
default. For an N-layer transformer that re-runs the same kernels per layer
(prefill) and per token (decode), this is pure redundant host↔NPU traffic.
This skill removes it. It is the same optimization the reference ablation
isolated as cells A→B (weight BOs) and B→C (intermediate BOs); together they
were a multi-second prefill saving and the dominant decode host-side cost.

Two mechanics, one class:

- **B1 — per-layer weight BOs** (`static_input_indices`): allocate each
  layer's weight BOs once during setup, write them once, and pass
  `static_input_indices=[<weight slots>]` on every `cache.load_and_run()` so
  the runtime skips re-writing them.
- **B2 — intermediate BOs** (`intermediate_indices`): for buffers the kernel
  fully overwrites (its own outputs / scratch), pass
  `intermediate_indices=[<output slots>]` so the host does not write them
  before the call.

## Success criteria

Applying this skill is "successful" when ALL hold:

1. Output cosine ≥ 0.99 vs the pre-optimization baseline (BO reuse must not
   change the math — same kernels, same inputs, only the upload is skipped).
   Log `max_abs / max_rel` informational (match the BF16 convention from
   `phase-1-kernel-validation`; do not use a tight `rtol`).
2. `make verify` still PASSES (the end-to-end gate; token-set top-k vs HF bf16).
3. Measured host/wall time is strictly lower than the baseline.

If (1)/(2) regress (NaN or garbage on the 2nd+ call, correct on the 1st) →
the BO bookkeeping is wrong; invoke `debug-bo-corruption`.
If (3) shows no gain → the per-call upload wasn't the bottleneck here;
document and keep or revert.

## Knowledge base references

- `programming_examples/llms/llama_kernel_builder/cache.py` —
  `KernelCache.load_and_run`, the `static_input_indices` /
  `intermediate_indices` mechanics this skill drives.
- `programming_examples/llms/llama32_1b/multi_launch_builder/*` — the worked
  example of weight + intermediate BO slots passed to a fused ELF.
- `programming_examples/llms/llama32_1b/llama32_1b_inference.py` —
  `prepare_runtime` / setup where per-layer BOs are allocated once.

## Workflow

### Step 1: Identify the BO slots

From the kernel group's argument signature, classify each BO slot:

- **weight / LUT slots** — written once, read every call → B1 candidates
  (`static_input_indices`).
- **kernel-overwritten slots** (the kernel's outputs and scratch) → B2
  candidates (`intermediate_indices`).
- **genuine per-call inputs** (the activation that changes each call) — leave
  as normal host-written inputs.

### Step 2: Pre-load weight BOs once (B1)

- Allocate per-layer weight BOs in `prepare_runtime()` (setup), keyed per
  layer (e.g. `bo_key=f"kernel_L{layer_idx}"`), and write the weights ONCE.
- On every `cache.load_and_run()`, pass `static_input_indices=[<weight slots>]`
  so the runtime does not re-upload them.

### Step 3: Reuse intermediate BOs (B2)

- For each slot the kernel fully overwrites, pass
  `intermediate_indices=[<output slots>]` on `cache.load_and_run()` so the
  host does not write that buffer before the call.

### Step 4: prefill vs decode

Same mechanic, two contexts (pass which one as the caller's parameter):

- **prefill**: weights re-used across the 16 per-layer calls within one pass.
- **decode**: weights re-used across every generated token × every layer —
  the win is much larger (16 layers × ~7 weight tensors × N tokens of upload
  removed). Static weight BOs are the dominant decode host-side optimization.

### Step 5: Validate + measure

- Run with BO reuse; compare output to the pre-reuse baseline → cosine ≥ 0.99.
- Re-run `make verify` → must still PASS.
- Profile host/wall time → must be strictly lower.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Correct on 1st call, NaN/garbage on 2nd+ call | per-layer BO key collision OR `static_input_indices` slot list wrong | Invoke `debug-bo-corruption` |
| Output mismatch on the very 1st call | a slot marked `intermediate` is actually read before being written | Re-classify that slot as a real input (drop it from `intermediate_indices`) |
| No host-time reduction | the per-call upload wasn't the bottleneck (kernel-bound) | Document; the merge skill (dispatch) or layout-alignment may be the bigger win |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

Append to `<model>/docs/development_progress/phase{4,5}_*.md`:

\`\`\`
## Buffer-object reuse
- B1 weight BOs: applied / skipped (reason)
- B2 intermediate BOs: applied / skipped (reason)
- Host/wall time before: X ms
- Host/wall time after:  Y ms
- Cosine vs baseline: <value>  | make verify: PASS/FAIL
\`\`\`
```

- [ ] **Step 3: Verify structure + naming**

Run: `grep -E "^name: buffer-object-reuse$|^## (Purpose|Success criteria|Knowledge base|Workflow|Failure modes|Update protocol)" .claude/skills/buffer-object-reuse/SKILL.md`
Expected: the name line + all 6 section headers present.

Run: `grep -c "static_input_indices\|intermediate_indices" .claude/skills/buffer-object-reuse/SKILL.md`
Expected: ≥ 4 (both mechanics referenced multiple times).

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/buffer-object-reuse/SKILL.md
git commit -m "[.claude/skills] add buffer-object-reuse optimization skill"
```

---

### Task 2: Create the `layout-alignment` optimization skill

**Files:**
- Create: `.claude/skills/layout-alignment/SKILL.md`

Extracts phase-4 Pattern C (seq-first layout / drop host transposes).

- [ ] **Step 1: Read the source material**

Read to copy exact mechanics:
- `.claude/skills/phase-4-prefill-optimization/SKILL.md` Pattern C (seq-first layout) + the head_dim ≥ 128 FA caveat
- `.claude/skills/debug-fa-runtime-failure/SKILL.md` (the skill that owns the head_dim ≥ 128 "why")

- [ ] **Step 2: Write the skill file**

Create `.claude/skills/layout-alignment/SKILL.md` with this exact content:

```markdown
---
name: layout-alignment
description: Optimization skill — choose activation layouts so consecutive kernels hand off on-device without a host-side transpose. Canonical case: seq-first (seq, n_heads·head_dim) so RMSNorm → RoPE → FlashAttention → O-proj stay seq-first, eliminating 1–4 host transposes per layer. Invoked by phase-4-prefill-optimization (and phase-5 when decode introduces a transpose phase-4 didn't fix).
---

## Purpose

When two consecutive kernels disagree on activation layout, the host inserts a
transpose between them — a data round-trip that adds up across many per-layer
calls. This skill removes those transposes by choosing layouts that let
consecutive kernels hand off directly on-device. The canonical alignment is
**seq-first** activations `(seq, n_heads·head_dim)`, which keeps RoPE,
FlashAttention, and the O projection on the same layout the GEMMs/RMSNorm
already produce — no host transpose between them.

Most inheritance deployments already run seq-first end-to-end (nothing to do —
skip). This skill applies when the deployment still has a host transpose
between two kernels.

## Success criteria

Applying this skill is "successful" when ALL hold:

1. Output cosine ≥ 0.99 vs the pre-alignment baseline (changing layout must
   not change the math). Log `max_abs / max_rel` informational.
2. `make verify` still PASSES (end-to-end gate).
3. The targeted host transpose(s) are gone — fewer host ops, lower wall time.

If (1)/(2) regress → a kernel did not actually accept the new layout; revert.

## Knowledge base references

- `programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py`
  — the seq-first FlashAttention variant (head_dim ≤ 64).
- `programming_examples/flash_attention/kernel_fusion_based/` — the
  head-first kernel + wrapper used for head_dim ≥ 128 (see caveat below).
- `.claude/skills/debug-fa-runtime-failure` — owns the *why* of the
  head_dim ≥ 128 routing.

## Workflow

### Step 1: Find the host transposes

Profile / read the per-layer host code. Each `np.transpose` /
`ascontiguousarray` between two NPU kernel calls is a candidate. Note which
kernel boundary it bridges (typically RoPE→FA or FA→O-proj).

### Step 2: Make the kernels accept seq-first

- RoPE: accept seq-first input.
- FlashAttention: accept seq-first Q, K, V — this is `attn_npu2_seqfirst.py`
  for head_dim ≤ 64.
- Verify the producer kernel emits the layout the consumer expects, so the
  transpose can be deleted (not just moved).

### Step 3: head_dim ≥ 128 caveat

The seq-first `dk_chunks > 1` path has known runtime issues at head_dim ≥ 128.
Route head_dim ≥ 128 attention through the **head-first wrapper** (it does the
host transpose precisely so the rest of the pipeline stays seq-first). Do NOT
debug FA inline here — for the *why* and the discrimination of the failure
modes, invoke `debug-fa-runtime-failure`.

### Step 4: Validate + measure

- Compare output to the pre-alignment baseline → cosine ≥ 0.99.
- Re-run `make verify` → must still PASS.
- Confirm the transpose is gone and wall time dropped.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Cosine drops after switching layout | a kernel didn't actually consume seq-first; the transpose was masking a real layout mismatch | Revert; confirm each kernel's accepted layout before deleting the transpose |
| FA hang (`ERT_CMD_STATE_TIMEOUT`) or NaN at head_dim ≥ 128 | seq-first `dk_chunks > 1` path bug | Route through the head-first wrapper; invoke `debug-fa-runtime-failure` |
| Transpose removed but no wall-time gain | the transpose wasn't on the hot path | Document; revert or keep for cleanliness |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

Append to `<model>/docs/development_progress/phase{4,5}_*.md`:

\`\`\`
## Layout alignment
- Transposes removed: <which boundaries>
- head_dim ≥ 128 routed head-first: yes/no/N-A
- Wall time before: X ms
- Wall time after:  Y ms
- Cosine vs baseline: <value>  | make verify: PASS/FAIL
\`\`\`
```

- [ ] **Step 3: Verify structure + naming**

Run: `grep -E "^name: layout-alignment$|^## (Purpose|Success criteria|Knowledge base|Workflow|Failure modes|Update protocol)" .claude/skills/layout-alignment/SKILL.md`
Expected: the name line + all 6 section headers.

Run: `grep -c "seq-first\|debug-fa-runtime-failure" .claude/skills/layout-alignment/SKILL.md`
Expected: ≥ 4.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/layout-alignment/SKILL.md
git commit -m "[.claude/skills] add layout-alignment optimization skill"
```

---

### Task 3: Rewrite phase-4-prefill-optimization as a thin orchestrator

**Files:**
- Modify: `.claude/skills/phase-4-prefill-optimization/SKILL.md`

Replace the inlined Pattern A/B/C/D with a thin dispatch over the three shared
optimization skills. Delete Pattern D. Keep: PASS criteria (verify-gate),
baseline measurement, per-step re-verify, recording, failure-mode table
(updated to point at the skills).

- [ ] **Step 1: Read the current file**

Read `.claude/skills/phase-4-prefill-optimization/SKILL.md` in full to know
exactly which lines are Pattern A/B/C/D and what to preserve.

- [ ] **Step 2: Replace the "Step 2: Apply optimization patterns" section**

Replace the entire `### Step 2: Apply optimization patterns` block (Patterns
A, B, C, D and their sub-bullets) with this:

```markdown
### Step 2: Apply optimization skills

prefill draws on the shared optimization skillset. For each, decide if it
applies, invoke the skill, then re-run the gate (Step 3). Skip with a logged
reason if the trigger condition isn't met — "≥ N applied" is NOT the gate;
the gate is the outcome (faster + `make verify` still PASSES).

| Optimization skill | When it applies to prefill | What it does |
|---|---|---|
| `merge-multi-launch-kernels` | almost always (the dominant win) | stitch each leaf kernel's `air.launch` into one fused ELF per kernel-group → one `xrt.run()` per group instead of per kernel (llama3: 16→3 calls/layer). Build the model's `multi_launch_builder/` (kernel-first) or reuse llama's fused ELFs (bit-for-bit inheritance — the verdict made in `phase-2-single-block-validation` Step 1). |
| `buffer-object-reuse` | always | pre-load per-layer weight BOs once (`static_input_indices`) + reuse intermediate BOs (`intermediate_indices`); removes redundant host↔NPU uploads. |
| `layout-alignment` | only if a host transpose still sits between two kernels | choose seq-first layouts so RoPE/FA/O-proj hand off on-device; skip if the model already runs seq-first end-to-end (most inheritance deployments do). |

Each skill owns its own recipe, success self-check, and failure modes —
this phase does not restate them. Invoke the skill, read its result, then
gate.
```

- [ ] **Step 3: Update Step 1 (baseline) and Step 3 (re-verify) to drop Pattern references**

In `### Step 1: Measure Phase 3 baseline`, ensure it says to record kernel +
wall time as the baseline every skill must beat (no Pattern-specific text).

In `### Step 3: Re-run Phase 3 gate after each pattern`, change "after each
pattern" → "after each optimization skill"; keep `make verify` as the gate and
`make diagnosis` as localization (already aligned from the verify-gate work).

- [ ] **Step 4: Update the PASS criteria #3 wording**

In `## Phase 4 PASS criteria`, criterion #3 currently says "for each of the 4
patterns, record applied/skipped/reverted". Change "the 4 patterns" → "each
optimization skill it invoked". The "≥ N patterns applied is NOT a gate"
paragraph: change "patterns" → "optimization skills".

- [ ] **Step 5: Update the failure-modes table**

In `## Failure modes`, remove any row that referenced Pattern D / CPU→NPU
promotion. Keep the multi-launch-merge, BO-corruption, and FA rows (they now
correspond to the three skills). Ensure the merge row still points at
`debug-multi-launch-merge` and says "6 known compile blockers" (from the
chain-review fix).

- [ ] **Step 6: Verify no Pattern A/B/C/D or Pattern-D content remains**

Run: `grep -nE "Pattern [ABCD]|CPU→NPU|CPU->NPU|Pattern D" .claude/skills/phase-4-prefill-optimization/SKILL.md`
Expected: no matches (all inlined patterns removed). If "Pattern A" etc. still
appears, finish the replacement.

Run: `grep -c "merge-multi-launch-kernels\|buffer-object-reuse\|layout-alignment" .claude/skills/phase-4-prefill-optimization/SKILL.md`
Expected: ≥ 3 (all three skills referenced).

- [ ] **Step 7: Commit**

```bash
git add .claude/skills/phase-4-prefill-optimization/SKILL.md
git commit -m "[.claude/skills] phase-4 thin orchestrator over optimization skillset; drop Pattern D"
```

---

### Task 4: Rewrite phase-5-decode-optimization as a thin orchestrator

**Files:**
- Modify: `.claude/skills/phase-5-decode-optimization/SKILL.md`

Same shape as Task 3, decode flavor. Decode uses merge (GEMV variants, with
extern rename) + buffer-object-reuse (static weight BOs — amplified win);
layout-alignment usually N/A. Delete Pattern D.

- [ ] **Step 1: Read the current file**

Read `.claude/skills/phase-5-decode-optimization/SKILL.md` in full. Note the
decode-specific content that must be PRESERVED inside the merge dispatch: the
N-way extern kernel rename (2-K / 3-K) and K-split for K > 8160 — these are
GEMV-merge specifics that belong with `merge-multi-launch-kernels` invocation.

- [ ] **Step 2: Replace the "Step 2: Apply optimization patterns" section**

Replace the Pattern A/B/D block with:

```markdown
### Step 2: Apply optimization skills

decode draws on the same shared optimization skillset; the dominant patterns
differ because decode runs at M=1 per token, calling all N layers per token.

| Optimization skill | When it applies to decode | What it does (decode flavor) |
|---|---|---|
| `merge-multi-launch-kernels` | almost always | stitch decode kernel groups (GEMV instead of GEMM) into fused ELFs (10 launches/layer/token → 2–3). **Decode specifics handled by the skill**: N-way extern kernel rename when multiple GEMV K values co-link in one ELF (2-K for llama: `mv.o` K=2048 + `mv_k8192.o`; add a 3rd renamed `.o` when `n_heads·head_dim ≠ emb_dim`), and K-split (`down_k_split`) for K > 8160 (`details/GEMV_bf16.md`). |
| `buffer-object-reuse` | always — biggest decode win | static weight BOs: weights allocated once, `bo.map()` zero-copy, `static_input_indices` skips re-write on every token. With 16+ layers × ~7 weights × 100 tokens, this is the dominant pre-optimization decode host cost. |
| `layout-alignment` | usually N/A | only if decode introduced a transpose Phase 4 didn't already fix. |

Each skill owns its recipe + success self-check + failure modes. The LM Head
GEMV (vocab-partitioned) is part of the model's decode assembly built in
Phase 3/finalize, profiled here — not a separate optimization skill.
```

- [ ] **Step 3: Update Step 1 + Step 3 wording**

`### Step 1: Measure Phase 4 baseline` — record ms/token + LM-head breakdown
as the baseline (no Pattern text). `### Step 3` — "after each optimization
skill"; `make verify` is the gate, diagnosis localizes (already aligned;
decode regressions surface in verify's 32-token generation, keep that note).

- [ ] **Step 4: Update PASS criteria #3 + the "≥ N" paragraph**

Same edit as Task 3 Step 4: "the 4 patterns" → "each optimization skill"; "≥ N
patterns" → "≥ N optimization skills".

- [ ] **Step 5: Update the failure-modes table**

Keep the rows that map to the three skills (multi-launch merge → 6 blockers;
extern rename collision; push_queue repeat > 255 / K-split; L2 cap; BO
corruption). These are all real decode failure modes and now correspond to
`merge-multi-launch-kernels` + `buffer-object-reuse`. Remove any Pattern-D
row.

- [ ] **Step 6: Verify**

Run: `grep -nE "Pattern [ABD]|CPU→NPU|CPU->NPU" .claude/skills/phase-5-decode-optimization/SKILL.md`
Expected: no matches.

Run: `grep -c "merge-multi-launch-kernels\|buffer-object-reuse\|layout-alignment" .claude/skills/phase-5-decode-optimization/SKILL.md`
Expected: ≥ 3.

- [ ] **Step 7: Commit**

```bash
git add .claude/skills/phase-5-decode-optimization/SKILL.md
git commit -m "[.claude/skills] phase-5 thin orchestrator over optimization skillset; drop Pattern D"
```

---

### Task 5: Update merge skill cross-reference + chain-wide consistency sweep

**Files:**
- Modify: `.claude/skills/merge-multi-launch-kernels/SKILL.md` (description line only, if needed)
- Verify: all skills

The merge skill's description already says "Invoked by phase-4 ... and phase-5
... Pattern A". Now that phases dispatch by skill (not "Pattern A"), fix that
phrase. Then sweep the whole chain for dangling references to deleted Patterns.

- [ ] **Step 1: Fix the merge skill description**

In `.claude/skills/merge-multi-launch-kernels/SKILL.md`, the frontmatter
description ends with "... Pattern A when building NEW model-specific fused
ELFs". Change "Pattern A" → "to fuse kernel groups". Also the Purpose line
"in Phase 4/5 (model has new ops...)" stays fine.

Run: `grep -n "Pattern A" .claude/skills/merge-multi-launch-kernels/SKILL.md`
Expected: no matches after the edit.

- [ ] **Step 2: Chain-wide dangling-Pattern sweep**

Run: `grep -rnE "Pattern [ABCD]" .claude/skills/`
Expected: no matches anywhere. (deploy-new-llm, debug-*, the phases — none
should reference inlined Patterns anymore.)

If any match appears, fix that reference to name the skill instead.

- [ ] **Step 3: Verify all three optimization skills are discoverable + consistent**

Run: `for d in merge-multi-launch-kernels buffer-object-reuse layout-alignment; do echo "== $d =="; grep -m1 "^name:" .claude/skills/$d/SKILL.md; done`
Expected: three lines, each `name:` matching its directory.

Run: `grep -rl "buffer-object-reuse\|layout-alignment" .claude/skills/phase-4-prefill-optimization/SKILL.md .claude/skills/phase-5-decode-optimization/SKILL.md`
Expected: both phase files reference the new skills.

- [ ] **Step 4: Commit (if Step 1 changed anything)**

```bash
git add .claude/skills/merge-multi-launch-kernels/SKILL.md
git commit -m "[.claude/skills] merge skill: dispatch-by-skill wording; chain Pattern-ref sweep"
```

---

## Self-Review checklist (run after all tasks)

- [ ] **Spec coverage:** three independent optimization skills exist (Task 1, 2, + existing merge); phase-4/5 are thin orchestrators (Task 3, 4); Pattern D deleted (Task 3, 4); merge/BO have self-check success criteria (Task 1, 2, existing). All spec sections map to a task.
- [ ] **Dangling refs:** `grep -rnE "Pattern [ABCD]" .claude/skills/` returns nothing (Task 5 Step 2).
- [ ] **Naming consistency:** the two new skills' `name:` == directory name; phase-4/5 reference them by exact name (Task 5 Step 3).
- [ ] **Gate semantics preserved:** phase-4/5 still gate on `make verify`, diagnosis is localization (not re-introduced as a gate by the rewrite).
