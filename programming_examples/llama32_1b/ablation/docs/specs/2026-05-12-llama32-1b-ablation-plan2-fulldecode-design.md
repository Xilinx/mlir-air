# Llama-3.2-1B NPU2 Ablation Study — Plan 2 (Full Decode) Design

**Status**: Design (pending implementation plan)
**Date**: 2026-05-12
**Branch**: implementation on a fresh worktree from `llama-3.2-1B-devel`
**Scope**: `programming_examples/llama32_1b/ablation/decode/` (new self-contained subdir)

**Companion docs:**
- Master ablation spec: [`2026-05-07-llama32-1b-ablation-study-design.md`](2026-05-07-llama32-1b-ablation-study-design.md)
- Plan 0 (decode `rms_gemv_rope` pilot) plan: [`../plans/2026-05-07-llama32-1b-ablation-decode-pilot.md`](../plans/2026-05-07-llama32-1b-ablation-decode-pilot.md)
- Plan 1 (full prefill) spec: [`2026-05-07-llama32-1b-ablation-plan2-prefill-design.md`](2026-05-07-llama32-1b-ablation-plan2-prefill-design.md)
- Plan 1 (full prefill) plan: [`../plans/2026-05-07-llama32-1b-ablation-plan2-prefill.md`](../plans/2026-05-07-llama32-1b-ablation-plan2-prefill.md)
- ABLATION_STUDY.html Part 5 (Plan 2 design summary, audience-facing): `programming_examples/llama32_1b/docs/ABLATION_STUDY.html#plan2-status`
- Production profile: `programming_examples/llama32_1b/docs/profile.md`

---

## 1. Goal

Apply the proven 4-cell ablation methodology — validated by Plan 0 (decode `rms_gemv_rope` pilot, A→D = 2.75×) and Plan 1 (full prefill, A→D = 1.56×, Cell D = 1.13 s ≈ profile.md's 1.27 s) — to the **full decode** dispatch pipeline. Three decode kernel-groups are in scope:

- `rms_gemv_rope` (6 sub-launches at single-token GEMV shapes) — already pilot-tested in Plan 0
- `o_gemv_ffn` (8 sub-launches at single-token GEMV shapes) — new in this plan
- `lm_head_gemv` (8 partitions stitched in 1 ELF, 1 NPU call/token) — held INVARIANT across cells (rationale §4)

The CPU-side `decode_attention_cpu` is also held invariant (it's CPU code; nothing to ablate). FlashAttention's NPU decode path is OUT OF SCOPE — production decode uses CPU attention at head_dim=64 because the NPU FA path has overhead at single-query workloads.

**Two scopes per cell:**
1. **Per-kernel-group single-call timings** for each of `rms_gemv_rope` and `o_gemv_ffn` — fast iteration and per-launch breakdown extraction (matches Plan 0/1's reporting style).
2. **Per-token full-pipeline wall time** = 16 layers × (rms_gemv_rope + decode_attn_cpu + o_gemv_ffn) + final RMSNorm + lm_head_gemv + argmax. Headline number directly comparable to `profile.md`'s per-token decode latency.

Plan 2 produces the comprehensive end-to-end decode ablation report. After Plan 2, all three production phases (single-kernel-group decode, end-to-end prefill, end-to-end decode) have controlled measurements.

## 2. Optimizations under study

Same three optimizations as Plan 0/1, applied to the decode kernel-groups:

| ID | Optimization | Production behavior in decode |
|----|--------------|-------------------------------|
| **#1** | Multi-launch ELF | Per layer per token: 6 sub-launches stitched into `rms_gemv_rope.elf`, 8 stitched into `o_gemv_ffn.elf`. Two `xrt.run()` per layer (plus the CPU attention step). Per token: 16 × 2 + 1 (LM head) = **33 NPU calls**. |
| **#2** | Per-layer weight BOs (`static_input_indices`) | All 16 layers' decode weights pre-loaded into per-layer BOs once during `prepare_runtime`; `static_input_indices` skips re-write on subsequent calls. Same `bo_key=f"name_L{layer_idx}"` trick as production. |
| **#3** | `intermediate_indices` | Buffers the kernel will overwrite are not host-written on subsequent calls. For Cell C, intermediate BOs are also explicitly aliased across separate `xrt.run()` calls within each kernel-group via `_share_bo` (mirror Plan 0/1). |

These are the same three flags exercised in Plan 0/1; what changes is the dispatch envelope (per-token loop instead of single dispatch or 16-layer prefill loop) and the addition of `o_gemv_ffn` as a second cell-ablated kernel-group.

## 3. Experimental design — the 4-cell ladder

The ladder applies to the **decode per-layer triple** (rms_gemv_rope + decode_attn_cpu + o_gemv_ffn). The CPU attention is invariant across cells. Cells differ only in how they dispatch the within-kernel-group sub-launches of `rms_gemv_rope` and `o_gemv_ffn`. LM head is invariant (production-merged in every cell).

| Cell | Description | Marginal change | Isolates |
|------|-------------|-----------------|----------|
| **A** Naive no-merge | Each sub-launch as separate `xrt.run()`: 6 calls for `rms_gemv_rope` + 1 CPU attn + 8 calls for `o_gemv_ffn` = **14 NPU calls per layer**. Plus 8 calls for `lm_head_gemv` per token (held merged here per §4 rationale; if also un-merged, would be 22). Per token: 14 × 16 + 8 = **232 NPU calls (with LM head merged) / 232 + 7 = 239 (with LM head un-merged)**. Production-decode-uses-merged baseline: **232 calls/token in Cell A**. Host round-trip on every intermediate. Weights re-uploaded every call. | (baseline) | — |
| **B** + per-layer weight BOs | Same as A, but weights pre-loaded into per-layer BOs once; `static_input_indices` skips re-write. Same NPU call count. | +#2 | A→B = #2 alone |
| **C** + shared intermediate BOs | Same as B, but intermediate BOs are aliased across separate `xrt.run()` calls **within each kernel-group**. Cross-kernel-group transitions (rms_gemv_rope → CPU attn → o_gemv_ffn) still go through host — matches production. Same NPU call count. | +#3 (intermediate-BO sharing across separate `xrt.run()` calls within each group) | B→C = #3 alone |
| **D** Multi-launch merged | Production: `rms_gemv_rope`'s 6 stitched into one ELF, `o_gemv_ffn`'s 8 stitched into one ELF. **2 NPU calls per layer + 1 LM head per token = 33 NPU calls/token** (matches profile.md). | +#1 | C→D = pure #1 (XRT dispatch saved by group-merging) |

### Reported claims

| Reported number | What it answers |
|-----------------|------------------|
| **A→D (per-token wall)** | Total naïve→production speedup for decode |
| **C→D** | Pure multi-launch merging effect for decode |
| **A→B** | #2 contribution alone in decode |
| **B→C** | #3 contribution alone in decode |
| **Per-kernel-group medians** | Per-call wall time for each of `rms_gemv_rope` and `o_gemv_ffn` across cells (analogous to Plan 1's per-call breakdown table) |
| **Cell D per-token wall vs `profile.md`** | Confirms (or corrects) the production decode per-token number from a clean ablation |
| **Cross-comparison vs Plan 0** | Does the single-kernel-group finding (Plan 0: #2 dominates at 1.60×) hold at full per-token end-to-end scale, or shift when `o_gemv_ffn` is added to the ablation envelope? |

## 4. Invariants across all cells

To ensure cell-to-cell deltas reflect only the within-kernel-group dispatch strategy:

- **Same C++ kernels, shapes, weights, prompt seed.** Bit-exact output validated against Cell D for layer 0 (one validation gate per kernel-group: `rms_gemv_rope` and `o_gemv_ffn`).
- **`decode_attention_cpu` is the same Python/numpy function in every cell.** Its CPU work is ~constant across cells (same input shapes, same KV cache state at the timed window's start) — see §6 for state management.
- **`lm_head_gemv.elf` is held INVARIANT (production-merged) in every cell.** Rationale: it's structurally one `xrt.run()` with 8 stitched launches; production already merges; it is invariant in the same sense `flash_attn.elf` is invariant in Plan 1. Reporting it as a separate "fixed cost per token" line keeps the 4 cells comparable on the parts that DO change. If a follow-up Plan 2.5 wants to ablate LM head dispatch separately (option (b) or (c) from the ABLATION_STUDY.html design), it can be done on top of Plan 2's results.
- **Same KV cache initial state at the start of every cell's timed window.** A fixed-seed pre-fill of `prompt_len = 7` populates layer 0..15 cache slots 0..6; `current_pos = 7` at trial start. Each trial generates exactly ONE decode token. After the trial, the cache slot at position 7 is NOT preserved across trials — re-initialized per trial so each trial measures the same starting state.
- **NPU exclusive-locked**: `flock -x -w 1800 /tmp/mlir-air-npu.lock` mirrors Plan 0/1.
- **Synthetic deterministic inputs** from numpy `seed=42` (mirrors Plan 0/1 exactly).

## 5. Timing protocol

**Per cell:**
1. **Preload** (not timed): build cache state, pre-load weights into per-layer BOs (Cells B/C/D), allocate intermediate BOs (Cell C aliasing wired here).
2. **5 timed trials**, each generating exactly **1 decode token** starting from the same KV cache state (`current_pos = 7`).
3. **Drop trial 1 as warmup** (XRT context warmup, instruction-cache fill, BO-mapping cache fill).
4. **Report median + (min, max) over trials 2–5** per cell.

**Why single-token per trial (not 32-token loops):**
- Per-token decode wall time has a position-dependent component: `decode_attention_cpu` reads `[0:current_pos+1]` of the KV cache, so its CPU work scales linearly with `current_pos`. Generating 32 tokens means each token's wall time grows slightly with position, contaminating the dispatch-only comparison we care about.
- Single-token-at-fixed-position keeps the CPU attention work CONSTANT across trials and across cells.
- Trade-off: 5 trials × 1 token gives less smoothing than 5 trials × 32 tokens. Mitigation: warmup-trial-drop captures the first-call overhead; trials 2-5 should be very tight (similar to Plan 0/1's within-cell variance of <1% of mean).

## 6. KV cache state management

Each cell sees identical cache state at the start of each timed trial:

```
At trial start:
    k_cache[0..15, :, 0:7, :] = synthetic-pre-filled (seed=42)
    v_cache[0..15, :, 0:7, :] = synthetic-pre-filled (seed=42)
    k_cache[0..15, :, 7:, :] = zeros
    v_cache[0..15, :, 7:, :] = zeros
    current_pos = 7

During trial:
    For L in 0..15:
        rms_gemv_rope (NPU)            # produces q_roped, k_roped, v
        decode_attention_cpu (CPU)     # reads k/v_cache[L, :, 0:8, :]; writes k/v at slot 7
        o_gemv_ffn (NPU)               # produces next-layer x_decode
    final_rmsnorm (CPU, single row)
    lm_head_gemv (NPU)
    argmax (CPU)

At trial end:
    Reset k_cache[L, :, 7, :] = 0 and v_cache[L, :, 7, :] = 0 for all L.
    (Or more simply: reset entire cache from the saved pre-filled state.)
```

The cache reset between trials is a host-side numpy array assignment — negligible cost outside the timed window.

## 7. Validation gate

Mirror Plan 0/1: every cell must produce **byte-identical** outputs for both `rms_gemv_rope` and `o_gemv_ffn` against committed Cell D goldens, on the seed=42 synthetic input at `current_pos = 7`. Cells failing the gate have their timing suppressed in the report.

Two committed `golden_*.npz` fixtures (one per kernel-group), regenerated by Cell D's harness if production kernels change. The validation step compares all six rms_gemv_rope outputs (`normed, q, k, v, q_roped, k_roped`) and the eight o_gemv_ffn outputs (intermediate buffers + final layer output). For LM head: validate that the final argmax token id matches across all four cells (single-integer comparison; bit-exact).

## 8. File structure (proposed)

All paths under `programming_examples/llama32_1b/ablation/decode/` (new sibling of `ablation/prefill/`).

| File | Responsibility | Mirrors |
|------|----------------|---------|
| `__init__.py` | Package marker | — |
| `README.md` | Methodology, run instructions, results, reproducibility | Plan 1's `README.md` |
| `Makefile` | `make compile / regen-golden / run / report / all / clean` | Plan 1's `Makefile` |
| `specs/__init__.py` | Package marker | — |
| `specs/kernel_group.py` | `SubLaunchSpec`, `BatonLink`, `KernelGroupSpec` (or re-export from `ablation/prefill/specs/kernel_group.py` to share definitions) | Plan 1 |
| `specs/rms_gemv_rope.py` | `KernelGroupSpec` instance for the 6-launch decode attention pre-block | Plan 1's `specs/rms_gemms_rope.py` |
| `specs/o_gemv_ffn.py` | `KernelGroupSpec` instance for the 8-launch decode FFN block | Plan 1's `specs/o_ffn.py` |
| `standalone_builders/__init__.py` | Package marker | — |
| `standalone_builders/rms_gemv_rope.py` | Re-export Plan 0's existing `STANDALONES` registry (already in `ablation/standalone_builders/decode_rms_gemv_rope.py`) | Plan 0 |
| `standalone_builders/o_gemv_ffn.py` | 8 single-launch builder wrappers + `STANDALONES` registry — NEW | Plan 1's `standalone_builders/o_ffn.py` |
| `cells/__init__.py` | Package marker | — |
| `cells/common.py` | `compile_standalone_kernels` (parameterized), `_share_bo`, `_extract_public_func_name`, helpers — re-export or copy from Plan 1 | Plan 1's `cells/common.py` |
| `cells/cell_a_naive.py` | Parameterized Cell A — walks a `KernelGroupSpec` with `naive=True` | Plan 1 |
| `cells/cell_b_static.py` | Parameterized Cell B — preload weights, `static_input_indices` | Plan 1 |
| `cells/cell_c_charitable.py` | Parameterized Cell C — preload + alias intermediate BOs per `spec.baton_links` | Plan 1 |
| `cells/cell_d_merged.py` | Wraps production `build_rms_gemv_rope_module` and `build_o_gemv_ffn_module` from `multi_launch_builder/` | Plan 1 |
| `cells/decode_attn_const.py` | CPU attention invariant: same Python function in every cell | Plan 1's `flash_attn_const.py` |
| `cells/lm_head_const.py` | LM head invariant: production-merged 8-partition GEMV in every cell | NEW (Plan 1's FA invariant pattern) |
| `cells/per_token_loop.py` | Wraps a per-layer triple in a 16-layer loop, then runs final RMSNorm + LM head + argmax. **The end-to-end timed unit.** | Plan 1's `cells/multi_layer.py` |
| `golden/__init__.py` | Package marker | — |
| `golden/regen_golden.py` | One-shot Cell-D run for layer 0; dumps two npz fixtures + meta json | Plan 1 |
| `golden/golden_rms_gemv_rope_decode.npz` | Committed bit-exact reference (Cell D, layer 0, seed=42, current_pos=7) | Plan 1 |
| `golden/golden_o_gemv_ffn_decode.npz` | Committed bit-exact reference for o_gemv_ffn | Plan 1 |
| `golden/golden_meta.json` | Hashes, shapes, config, prompt_len, current_pos | Plan 1 |
| `run_ablation.py` | Orchestrator: validate → time × {per-call, per-token} × 4 cells, emit JSON | Plan 1 |
| `analyze.py` | JSON → markdown report | Plan 1 |
| `tests/__init__.py` | Package marker | — |
| `tests/conftest.py` | Pytest sys.path setup | Plan 1 |
| `tests/test_kernel_group_spec.py` | Dataclass invariants (NPU-free) | Plan 1 (or just import from Plan 1's tests) |
| `tests/test_parameterized_cells.py` | Mock-cache tests verifying each cell walks its spec correctly (NPU-free) | Plan 1 |
| `tests/test_validation_gate.py` | Tests against the two new decode goldens | Plan 1 |
| `tests/test_kv_cache_state.py` | NEW: verifies cache initialization + per-trial reset is deterministic | NEW |

**Files NOT touched** (Plan 0/1 isolation guarantee): every file under `programming_examples/llama32_1b/ablation/` outside `decode/`. Production code (`programming_examples/llama32_1b/{kernel_builder,multi_launch_builder}/`) read-only — only imported.

## 9. Open design decisions (RESOLVED)

For traceability, the 7 questions raised in `ABLATION_STUDY.html#plan2-validation` and their answers (per user discussion 2026-05-12):

| # | Question | Decision |
|---|----------|----------|
| 1 | How many tokens to generate per timed run? | **1 decode token per trial × 5 trials, drop trial 1 (warmup), median over trials 2-5.** Avoids position-dependent CPU attention growth contaminating the dispatch comparison. |
| 2 | Should LM head be its own cell ladder? | **Hold INVARIANT** (production-merged in every cell). Mirrors Plan 1's FA treatment. Defer separate LM-head ablation to a possible Plan 2.5. |
| 3 | KV cache state initialization | **Deterministic synthetic pre-fill of 7 tokens** from `seed=42`; reset between trials. |
| 4 | Where does `decode_attention_cpu` wall time get attributed? | **Counted in per-token total AND reported separately as a "CPU floor" line** (mirrors Plan 1's FA reporting). |
| 5 | Predicted findings | **Not in the spec or plan.** Forecasts become bias when running. Report only after measurement. |
| 6 | Production CPU-attention or experimental NPU FA decode? | **Production CPU-attention path only.** That's what `profile.md` reflects. |
| 7 | Where does the harness live? | **`programming_examples/llama32_1b/ablation/decode/`** (new sibling of `ablation/prefill/`). |

## 10. Out of scope

- **NPU FlashAttention decode path** (head_dim=64). Production uses CPU; this plan doesn't ablate the alternative.
- **LM Head L1/L8 mini-study** (whether to use 1-launch or 8-partition LM head). Held invariant in this plan; can be a follow-up Plan 2.5.
- **Cross-kernel-group BO aliasing** (rms_gemv_rope output BO → CPU attention input → o_gemv_ffn input). This is the C2 future-work entry in IMPLEMENTATION_GUIDE.html. Cross-group goes through host in every cell, matching production.
- **Tokens beyond a single fixed `current_pos`.** Single-token-at-fixed-position is intentional (§5).
- **Real HuggingFace weights.** Synthetic seed=42 only — same justification as Plan 0/1.
- **Numerical-precision study vs an HF / F32 reference.** That belongs to the production verify subsystem (`make verify` for the top-k token gate, `make diagnosis` for per-layer cosine), not duplicated here.

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Single-token timing has high variance because no per-token smoothing | Medium | Medium | Warmup-drop + 5 trials usually suffices (Plan 0 saw <1% within-cell variance with the same approach). If trials 2-5 spread is >5% of median, increase to 9 trials (drop 1). |
| `o_gemv_ffn` standalone builder for cell A/B/C is more complex than `rms_gemv_rope`'s (8 sub-launches incl. SwiGLU + Down GEMV at K=8192) | High | Medium | Carefully reuse Plan 1's `standalone_builders/o_ffn.py` patterns; the kernel-group structure parallels but with GEMV instead of GEMM and the special `mv_k8192.o` for the Down step. Allow extra time for this task. |
| Bit-exact validation across 32 generated tokens (if we extend later) might fail because cache state evolves identically only if every cell sees the same input bytes at every position | Low (since we use 1 token) | Low | Single-token approach sidesteps this entirely. If we later extend to multi-token, validation must hash all generated outputs, not just the first. |
| LM head's per-token wall time is non-trivial (~14 ms typical), so even though it's invariant it shifts the per-token total significantly | Low | Low | Report the LM head as a separate fixed-cost line (mirrors Plan 1's FA reporting). Doesn't bias cell-to-cell deltas. |
| Goldens become stale when production kernels are recompiled (e.g., after a Peano upgrade) | Medium | Medium | Same as Plan 0/1: `make regen-golden` documented; validation gate fails loudly so divergence is visible. |
| KV cache state between trials accidentally drifts (e.g., partial reset bug) | Low | High (would invalidate timing if cells see different input data) | `tests/test_kv_cache_state.py` verifies reset determinism BEFORE timing trials run. |

## 12. Reproducibility guarantee

```
git clone <repo> && git checkout <plan-2-branch>
cd programming_examples/llama32_1b/ablation/decode
make clean
make all   # compile (~5 min) + run (~2 min, NPU-locked) + report
```

Expected output (5 trials per cell, drop trial 1, median + range):
```
  Cell A: PASS  per-token median=~XX ms  range=[~YY, ~ZZ]ms
  Cell B: PASS  per-token median=~XX ms  range=[~YY, ~ZZ]ms
  Cell C: PASS  per-token median=~XX ms  range=[~YY, ~ZZ]ms
  Cell D: PASS  per-token median=~XX ms  range=[~YY, ~ZZ]ms
```

(Numbers TBD by implementation. Cell D per-token median should be in the ballpark of `profile.md`'s decode latency, modulo ~1-2 ms of host steps not in the timed window.)

NPU-free unit tests: `python3 -m pytest tests/ -v` should report 8+ passed.

## 13. Companion ABLATION_STUDY.html updates (post-implementation)

After Plan 2 is implemented and measured, update `programming_examples/llama32_1b/docs/ABLATION_STUDY.html`:

- Section 5.1 (status): change from "📋 PLANNED" to "✅ Implemented + measured"
- Add new Section 5.4 (Results — Plan 2: full decode), parallel to Sections 3.3 and 4.3
- Update Section 6.1 (cross-comparison): replace "decode vs. prefill (so far)" with three-way comparison (Plan 0 vs Plan 1 vs Plan 2)
- Update Quick recap at bottom
- Update sidebar nav if needed

These updates are part of the Plan 2 implementation plan, not a separate plan.
