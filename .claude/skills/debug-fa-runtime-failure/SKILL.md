---
name: debug-fa-runtime-failure
description: Use when NPU FlashAttention hangs (`ERT_CMD_STATE_TIMEOUT`) or produces NaN at head_dim ≥ 128. Discriminates the three known root causes (compile-flag mismatch, seq-first dk_chunks bug, true L1 overflow) via a symptom-classification table and applies the documented fix.
---

## Purpose

NPU FlashAttention failures at head_dim ≥ 128 manifest in three
distinct ways with three distinct root causes. This skill walks the
diagnosis efficiently instead of bisecting from scratch — head_dim=128
deployments hit each of these.

Use this as a **diagnostic decision tree**, not a mechanical
fix-applier — match symptom to hypothesis BEFORE applying the
documented remedy.

## Knowledge base references

- `programming_examples/flash_attention/kernel_fusion_based/Makefile`
  — canonical `-D` flag conventions (the ground truth for compile
  flags)
- `programming_examples/flash_attention/kernel_fusion_based/attn_npu2.py`
  vs `.../attn_npu2_seqfirst.py` — the two Python builders (head-first
  vs seq-first); compile from the same C++ kernel but different IR
- `programming_examples/llms/llama_kernel_builder/external_kernels.py`
  — the FA compile API that derives per-tile flags correctly (the
  `compile_attn_npu2*` helpers)

## Triggers

ALL of these route here:

- `RuntimeError: Command failed to complete successfully (ERT_CMD_STATE_TIMEOUT)`
  from `cache.load_and_run("flash_attn", ...)` or `XRTRunner.run_test`
- All-NaN output from FA when inputs are well-formed (no NaN/Inf in)
- Compile passes but runtime output is constant garbage (e.g., all
  `49.0`, `844.0`, growing magnitudes — typical of softmax-not-running)

## Workflow

### Step 1: Classify the symptom

Run a minimal repro at the failing shape. Match the symptom:

| Symptom | Most-likely root cause | Jump to |
|---|---|---|
| HANG (timeout) at all `dk_chunks > 1` configs but PASS at `dk_chunks = 1` | Seq-first `dk_chunks > 1` upstream bug | Step 3 |
| NaN at any config (including ones the lit test passes with `uniform(0,4)` inputs) | `compile_attn_npu2*` flag mismatch (per-launch sizes baked into .o) | Step 2 |
| Garbage non-NaN output (large constant values, no softmax behavior) | Same as NaN — flag mismatch, just numerically different surface | Step 2 |
| HANG at one specific shape but PASS at smaller variants | True L1 overflow at the larger shape | Step 4 |

If the symptom doesn't fit any row, this is a new failure mode —
escalate per "Failure mode" at the bottom.

### Step 2: Verify .o flag conventions (most common — fixes NaN/garbage)

The `attn_npu2.cc` kernel's `lqp/lkp/dk/dv` defines are **per-tile**,
NOT per-launch. The Makefile's convention is canonical (verify against
this):

```makefile
LQP_TILE := $(shell echo $$(($(LQP) / $(NUM_Q_TILES))))
... -Dlqp=$(LQP_TILE) -Dlkp=$(LKP) \
    -Ddk=$(LKP) -Ddk_full=$(DK) \
    -Ddv=$(LKP) -Ddv_full=$(DV) ...
```

**Diagnostic**: diff your compile call against the Makefile. The FA
compile helper in
`llms/llama_kernel_builder/external_kernels.py` derives
`lqp_tile = lqp // num_q_tiles` internally and emits the right per-tile
flags.

**Remedy**: if your `.o` was compiled with per-launch flags, delete the
stale `.o` and the cached `flash_attn.elf`, then rebuild via the
external-kernels FA compile helper (which emits per-tile flags). Re-run.

### Step 3: Workaround the seq-first `dk_chunks > 1` upstream bug (Option C)

**Diagnostic**: `attn_npu2_seqfirst.py` (the seq-first Python builder)
has an untested `dk_chunks > 1` shim-DMA path that hangs at runtime.
Verify via bisect: every `dk_chunks=2` config hangs in seq-first,
regardless of (n_heads/n_kv, lq=lk). The HEAD-first kernel
`attn_npu2.py` at the same shape PASSES (`make run DK=128 DV=128
NUM_HEADS=32 NUM_KV_HEADS=8` → corr ≈ 0.996).

**Remedy** — Option C (head-first FA + host transposes):

1. Route the attention call through the head-first kernel
   (`attn_npu2.build_module(...)`) instead of seq-first, and add host
   transposes around it: reshape seq-first `[seq, n_heads*head_dim]` ↔
   head-first `[n_heads, seq, head_dim]` on the way in and out.
2. Wrap this as a helper that intercepts the `flash_attn` cached call so
   the rest of the pipeline stays seq-first and only the FA call sees
   head-first layout. Build it once in the deployment's `setup()` /
   block-compile path.

Cost: a few ms/layer host transpose. Gain: NPU FA actually runs (a
head_dim=128 deployment that fell back to CPU attention recovers a
multi-× warm prefill speedup once NPU FA works).

### Step 4: True L1 overflow (rare)

If both Step 2 and Step 3 are clean (correct flags, head-first
variant) and the kernel STILL hangs at one specific shape but PASSES
at smaller variants, you're hitting the actual 64 KB per-core L1
limit. Per-tile budget for FA at `(tile_size_q, lkp, dk_full, dv_full)`:

```
Q tile          : tile_size_q * lkp_per_dk_chunk * 2B
K tile (per dk) : lkp * lkp * 2B            (= 8 KB at lkp=64)
V tile (per dv) : lkp * lkp * 2B            (= 8 KB at lkp=64)
Gp accumulator  : tile_size_q * dv_full * 2B
misc (up,sp,r)  : ~2 KB
```

With `lkp=64`, the per-`dk_chunk` budget is small (~8 KB each) and
shared buffers are off (`lkp != dk_full` at hd=128). Sum stays well
under 64 KB at typical `tile_size_q ≤ 64`.

**Remedy**: drop `lqp` in the Python builder (which reduces
`tile_size_q = lqp / num_q_tiles`) and recompile.

True L1 overflow is rare for the shapes LLM deployments use. If you
hit it, also document the failing shape — it's a useful data point.

## Reusable bisect harness

When the symptom doesn't immediately fit Step 1's table, bisect across
(n_heads, n_kv_heads, lq=lk, dk) one axis at a time toward your
production config. The first axis that flips PASS → HANG/NaN tells
you which dimension is the offender.

The pattern is straightforward — invoke the external-kernels FA compile
helper to build the .o at a given shape, build the module via
`attn_npu2[_seqfirst].build_module(...)`, generate random inputs +
NumPy causal-SDPA reference, run via XRTRunner. Catch `TIMEOUT` →
"HANG"; cosine < threshold → "FAIL_NUMERICAL". A `<model>_phaseN_test.py`
script that exercises FA at the production shape is the worked example to
mirror.

## Verification

This skill is "successful" when the failing FA invocation produces
correct output (cosine ≥ Phase 2's head_dim-scaled threshold) and
runs without hang. Capture the resolution path in
`<model>/docs/development_progress/debug_log.md`.

## Failure mode (when this skill itself can't resolve)

If the symptom matches one of Step 1's rows but the documented remedy
in Step 2/3/4 doesn't resolve, OR the symptom doesn't fit any row:
this is a new failure mode not covered by current knowledge.

Escalate to the user with:
- The bisect matrix (which axis was varied, which configs PASS/HANG/NaN)
- The exact compile flags used
- The kernel cache state (was `.o` rebuilt? was `flash_attn.elf` re-cached?)

Do NOT silently wrap-fix or further-bisect for hours — the 3
documented root causes are well-characterized; a real new failure
mode deserves human triage.

## Update protocol

On successful diagnosis, append to
`<model>/docs/development_progress/debug_log.md`:

```
## debug-fa-runtime-failure recovery (YYYY-MM-DD)
- Failing shape: lq=X, lk=Y, dk=Z, hd=W, n_heads=A, n_kv=B
- Symptom: HANG / NaN / garbage
- Step matched: 2 / 3 / 4
- Fix applied: <one-line description>
- Verified: cosine ≥ <threshold> at production shape
```
