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

```
## Layout alignment
- Transposes removed: <which boundaries>
- head_dim ≥ 128 routed head-first: yes/no/N-A
- Wall time before: X ms
- Wall time after:  Y ms
- Cosine vs baseline: <value>  | make verify: PASS/FAIL
```
