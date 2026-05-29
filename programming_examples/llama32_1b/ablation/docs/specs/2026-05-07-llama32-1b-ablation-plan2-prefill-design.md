# Llama-3.2-1B NPU2 Ablation Study — Plan 2 (Prefill) Design

**Status**: Design (pending implementation plan)
**Date**: 2026-05-07
**Branch**: implementation on `llama32-1b-ablation-plan2-prefill` (worktree from `llama-3.2-1B-devel`)
**Scope**: `programming_examples/llama32_1b/ablation/prefill/` (new self-contained subdir)
**Companion docs**:
- Master ablation spec: [`2026-05-07-llama32-1b-ablation-study-design.md`](2026-05-07-llama32-1b-ablation-study-design.md)
- Plan 1 (decode `rms_gemv_rope` pilot): [`../plans/2026-05-07-llama32-1b-ablation-decode-pilot.md`](../plans/2026-05-07-llama32-1b-ablation-decode-pilot.md)
- Production profile: [`../../programming_examples/llama32_1b/docs/profile.md`](../../programming_examples/llama32_1b/docs/profile.md)

---

## 1. Goal

Apply the proven 4-cell ablation methodology (validated by Plan 1 on decode
`rms_gemv_rope`) to the **prefill** pipeline. Two prefill kernel-groups are in
scope: `rms_gemms_rope` (6 sub-launches at seq=2048 GEMM shapes) and `o_ffn`
(8 sub-launches at seq=2048 GEMM shapes). FlashAttention is held constant per
master-spec §5 (un-mergeable per `docs/explain.md`'s `air-opt-shim-dma-bds`
scaling note).

**Two scopes per cell:**
1. **Single-layer per-call timings** for fast iteration and per-launch
   breakdown extraction (matches Plan 1's reporting style).
2. **Full 16-layer prefill wall time** for headline numbers directly
   comparable to `profile.md`'s **1.27 s** measured production prefill.

Plan 2 produces a comprehensive prefill ablation report. Decode completion
(`o_gemv_ffn`) and the LM Head L1/L8 mini-study are explicitly **out of
scope** for this plan — they are scheduled as Plan 2-decode and Plan 2-lm-head
follow-ups.

## 2. Optimizations under study

Same three optimizations as Plan 1, applied to the prefill kernel-groups:

| ID | Optimization | Production behavior in prefill |
|---|---|---|
| **#1** | Multi-launch ELF | Per-layer: 6 sub-launches stitched into `rms_gemms_rope.elf` + 8 sub-launches stitched into `o_ffn.elf`, two `xrt.run()` per layer (plus FA). |
| **#2** | Per-layer weight BOs (`static_input_indices`) | All 16 layers' weights pre-loaded into per-layer BOs once during `prepare_runtime`; `static_input_indices` skips re-write on subsequent calls. |
| **#3** | `intermediate_indices` | Buffers the kernel will overwrite are not host-written on subsequent calls. |

These are the same flags exercised in Plan 1; what changes is the kernel
shape regime (GEMMs at seq=2048 instead of GEMVs at single-token), the launch
counts (6 + 8 instead of 6), and the multi-layer envelope.

## 3. Experimental design — the 4-cell ladder

The ladder applies to the **prefill per-layer triple** (rms_gemms_rope + FA +
o_ffn). FA is invariant across cells; the cells differ only in how they
dispatch the within-kernel-group sub-launches of rms_gemms_rope and o_ffn.

| Cell | Description | Marginal change | Isolates |
|---|---|---|---|
| **A** Naive no-merge | Each sub-launch as separate `xrt.run()`: 6 calls for rms_gemms_rope + 1 FA + 8 calls for o_ffn = **15 NPU calls per layer**. Host round-trip on every intermediate. Weights re-uploaded every call. | (baseline) | — |
| **B** + per-layer weight BOs | Same as A, but weights pre-loaded into per-layer BOs once; `static_input_indices` skips re-write. Still 15 NPU calls per layer. | +#2 | A→B = #2 alone |
| **C** + shared intermediate BOs | Same as B, but intermediate BOs are aliased across separate `xrt.run()` calls **within each kernel-group** (rms_gemms_rope's 6, and o_ffn's 8). Cross-kernel-group transitions (rms→FA, FA→o_ffn) still go through host — matches production. Still 15 NPU calls per layer. | +#3 (intermediate-BO sharing across separate `xrt.run()` calls within each group) | B→C = #3 alone |
| **D** Multi-launch merged | Production: rms_gemms_rope's 6 sub-launches stitched into one ELF, o_ffn's 8 stitched into one ELF. **3 NPU calls per layer** (rms_gemms_rope + FA + o_ffn). | +#1 | C→D = pure #1 (XRT dispatch saved by group-merging) |

### Reported claims

| Reported number | What it answers |
|---|---|
| **A→D** | Total naïve→production speedup for prefill (β baseline) |
| **C→D** | Pure multi-launch merging effect for prefill (α baseline) |
| **A→B** | #2 contribution alone in prefill |
| **B→C** | #3 contribution alone in prefill |
| **A→D × 16 layers vs `profile.md`'s 1.27 s** | Confirms (or corrects) the production headline number from a clean ablation |

## 4. Invariants across all cells

To ensure cell-to-cell deltas reflect only the within-kernel-group dispatch
strategy:

- **Same C++ kernels, shapes, weights, prompt seed.** Bit-exact output
  validated against Cell D for layer 0 (one validation gate per kernel-group).
- **FlashAttention is the same standalone ELF in every cell.**
  `rms_gemms_rope`'s outputs (`q_roped, k_roped, v`) are extracted to host →
  written to FA's BOs → `xrt.run` → `attn_out` extracted to host → written to
  o_ffn's residual-add input. This cross-kernel-group host hop happens
  identically in all cells. (Cross-group BO sharing is a potential
  Plan 2.5 — see §11.)
- **Synthetic deterministic inputs.** numpy seed=42 for layer 0; seed=42+i
  for layer i. Same RNG that Plan 1 used.
- **Decode-side optimizations untouched.** Plan 1's decode pilot files at
  `programming_examples/llama32_1b/ablation/` top-level remain frozen.
- **NPU power state.** Cells run back-to-back within one process (16-layer
  loop keeps NPU active throughout the trial).

## 5. Correctness verification (load-bearing)

Mirrors Plan 1 §9, with two adjustments:

- **Two golden fixtures**, one per kernel-group:
  `golden/golden_rms_gemms_rope_prefill.npz` and
  `golden/golden_o_ffn_prefill.npz`. Each is Cell D's layer-0 output for that
  group (numpy seed=42 inputs).
- **Validation per cell**, before any timing data is collected:
  1. Run cell on layer 0. Compare rms_gemms_rope outputs and o_ffn outputs
     bit-exactly against their respective goldens.
  2. **No multi-token decode equivalent** (prefill is single-pass).
  3. CPU reference cosine-sim sanity is logged but not gating (BF16 ≠ F32 by
     definition).
- **Cross-cell consistency re-check** after timing: re-run cell A vs D for
  layer 0 in the same process; assert byte-equal outputs. Catches BO
  recycle / lifetime bugs that surface only after long timing runs.
- Failed cells suppress their timing in the report.

The validation reuses Plan 1's `programming_examples/llama32_1b/ablation/validate.py`
unchanged (it's kernel-group-agnostic).

## 6. Per-launch breakdown — falls out of Cell C

Same mechanism as Plan 1: in Cell C, each sub-launch is its own `xrt.run()`
call → existing `KernelCache.Profiler` records `write_ms / kernel_ms / read_ms`
per call. Cell C automatically yields a 6-line breakdown for rms_gemms_rope
and an 8-line breakdown for o_ffn (in addition to the FA timing, which is
identical across cells).

D − C therefore quantifies pure dispatch-overhead reduction from merging,
**per kernel-group separately** (so we can report e.g. "merging saves X ms in
rms_gemms_rope and Y ms in o_ffn").

## 7. Host overhead — same arithmetic as Plan 1

For each cell:

```
host_overhead = wall_time − Σ(write_ms + kernel_ms + read_ms)
```

Reported per cell. The 16-layer wall-time minus 16 × per-layer NPU sum
reveals Python loop overhead in the multi-layer wrapper, distinct from
per-call host overhead.

## 8. Implementation approach

### 8.1 Self-contained subdirectory layout

All Plan 2 code lives under `programming_examples/llama32_1b/ablation/prefill/`.
Plan 1 files at `ablation/` top level are **byte-immutable** during Plan 2
development.

```
ablation/prefill/
├── README.md                           methodology, results, reproducibility
├── Makefile                            compile / run / report / regen-golden / clean
├── specs/
│   ├── kernel_group.py                 dataclass: KernelGroupSpec
│   ├── rms_gemms_rope.py               6-launch spec at prefill shapes
│   └── o_ffn.py                        8-launch spec at prefill shapes
├── standalone_builders/
│   ├── rms_gemms_rope.py               6 single-launch builder wrappers
│   └── o_ffn.py                        8 single-launch builder wrappers
├── cells/
│   ├── cell_a_naive.py                 parameterized; takes a KernelGroupSpec
│   ├── cell_b_static.py                "
│   ├── cell_c_charitable.py            " (consumes spec.baton_links)
│   ├── cell_d_merged.py                wrapper around production build_*_module
│   ├── flash_attn_const.py             FA invocation (held constant)
│   └── multi_layer.py                  wraps per-layer triple in 16-layer loop
├── golden/
│   ├── regen_golden.py                 one-shot Cell-D run, dumps both npz files
│   ├── golden_rms_gemms_rope_prefill.npz
│   └── golden_o_ffn_prefill.npz
├── run_ablation.py                     orchestrator
├── analyze.py                          JSON → markdown report
└── tests/
    ├── test_kernel_group_spec.py       dataclass validation, NPU-free
    ├── test_parameterized_cells.py     mock-cache tests, NPU-free
    └── test_validation_gate.py         re-uses Plan 1's validate.py against new goldens
```

### 8.2 KernelGroupSpec dataclass

A single concrete, grep-friendly description per kernel-group:

```python
@dataclass(frozen=True)
class SubLaunchSpec:
    name: str                          # e.g. "rmsnorm" | "q_gemm" | "rope_q"
    builder_ref: Callable              # function returning a 1-launch mlir.Module at production shape
    build_kwargs: dict                 # passed verbatim to builder_ref
    weight_slot_in_standalone: int | None  # which arg slot of the *standalone* call holds the weight (or None)
    output_slot_in_standalone: int     # which arg slot of the *standalone* call holds the output


@dataclass(frozen=True)
class BatonLink:
    producer_idx: int                  # index into sub_launches list
    producer_out_slot: int             # output slot of producer's standalone signature
    consumer_idx: int                  # index into sub_launches list (must be > producer_idx)
    consumer_in_slot: int              # input slot of consumer's standalone signature


@dataclass(frozen=True)
class KernelGroupSpec:
    name: str                          # "rms_gemms_rope" | "o_ffn"
    sub_launches: list[SubLaunchSpec]  # ordered execution sequence
    merged_arg_signature: list[str]    # ordered names matching production merged ELF args
    weight_slots: set[int]             # slots in merged signature that are weights/LUTs (for Cell D static_input_indices)
    intermediate_slots: set[int]       # slots that are kernel-overwritten intermediates (for Cell D intermediate_indices)
    output_slots_for_validation: list[int]  # slots in merged signature whose bytes go in the golden npz
    baton_links: list[BatonLink]       # Cell C uses these to alias intermediate BOs across sub-launches
```

Walking this spec gives each cell its dispatch sequence + BO-management
parameters. Adding a new kernel-group later (e.g., `o_gemv_ffn` for Plan
2-decode) = one new spec file; cell logic is unchanged.

### 8.3 Standalone (1-launch) ELFs

Same approach as Plan 1: thin wrappers around existing sub-builders in
`multi_launch_builder/rms_gemms_rope_multi.py` and
`multi_launch_builder/o_ffn_multi.py`, called with single-launch stitch
specs at production prefill shapes (seq=2048).

The wrappers should match the same `_extract_public_func_name` pattern Plan
1 settled on for `instance_name` — the standalone ELF's exported symbol
must be the actual MLIR public func name, not the cache key.

### 8.4 Cell-specific harness (parameterized)

| Cell | Implementation |
|---|---|
| **A** | Walks `spec.sub_launches` in order, invokes each via `cache.load_and_run(naive=True)` (Plan 1's `KernelCache.naive=True` mode). Per the §3 cross-group note: between rms_gemms_rope and FA, and FA and o_ffn, intermediates flow through host (extract → write to next group's input arrays). |
| **B** | Same as A, but a `preload(spec, weights_per_layer)` pass writes weights into per-layer BOs first (per-layer `bo_key`). Subsequent calls use `static_input_indices=spec.weight_slots`. |
| **C** | Same as B, but after preload, walk `spec.baton_links` and call `_share_bo` (Plan 1's helper, lifted into `prefill/cells/common.py` if needed) to alias intermediate BOs across sub-launches within each group. Use `intermediate_indices` for both producer-output and consumer-input slots. |
| **D** | Wrapper around production `build_rms_gemms_rope_module(seq_len=2048, ...)` and `build_o_ffn_module(seq_len=2048, ...)`. Two `cache.load_and_run` calls per layer (one per merged ELF). Unpacks output by slot index per Plan 1's lesson. |
| **flash_attn_const** | Compiles FA via existing `flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py:build_module` with the same kwargs production uses. Invocation is identical in every cell — same `bo_key`, same `output_indices`, same FA-input/output extraction pattern. |
| **multi_layer** | Wraps a per-layer triple in a 16-layer loop. Threads `x_in[layer_i+1] = o_ffn_output[layer_i]`. Used by both single-layer and 16-layer orchestrator scopes. |

### 8.5 Validation

Reuses Plan 1's `programming_examples/llama32_1b/ablation/validate.py`
verbatim (read-only import). Two golden npz files + per-cell validation gate
+ cross-cell consistency re-check (per §5). Failed cells suppress timing.

### 8.6 Orchestrator scopes

```
run_ablation.py supports two timing scopes:
  --scope=single-layer    5 trials × 1-layer cell call
  --scope=16-layer        5 trials × 16-layer cell call
  --scope=both (default)  both above; report both numbers
```

Both scopes run the same validation gate (layer-0 against golden) before
timing.

## 9. Statistical methodology

- **5 trials per cell × scope**, drop trial 1 (warmup), report median + (min, max).
- All `xrt.run()` invocations wrapped in `flock -x -w 1800
  /tmp/mlir-air-npu.lock` per `CLAUDE.md`.
- 16-layer trials may exhibit higher variance than single-layer (more
  opportunity for NPU jitter). Budget for 10 trials on 16-layer scope if
  median ± range > 5 %.

## 10. Deliverable: `programming_examples/llama32_1b/ablation/prefill/`

Self-contained mini-project with its own `make all` entry point:

```
make compile       # one-time, ~10-15 min (16 ELFs at seq=2048 + FA)
make regen-golden  # one-shot, after Cell D changes
make run           # all 4 cells × both scopes, emit JSON
make report        # markdown report
make all           # compile + run + report
make clean         # wipe build/
```

The auto-generated report includes:
- Validation badge table (per cell PASS/FAIL).
- Single-layer per-call timing table (per cell × per kernel-group).
- 16-layer total wall-time table (per cell, with comparison to `profile.md`'s 1.27 s).
- Marginal delta tables (per kernel-group AND aggregated).
- Per-launch breakdown extracted from Cell C (6 lines for rms_gemms_rope, 8 lines for o_ffn).
- Host-overhead share per cell.
- Comparison against `profile.md`'s "Key Optimizations" table claims.

A pointer is added to `programming_examples/llama32_1b/ablation/README.md`
(Plan 1's README) cross-linking to this study.

## 11. Out of scope (explicitly)

- **Plan 2-decode**: Decode `o_gemv_ffn` ablation (4 cells × 8 sub-launches). Same methodology; deferred to next sub-plan.
- **Plan 2-lm-head**: LM Head L1 (production 8-merged) vs L8 (8 separate `xrt.run()`) mini-study. Orthogonal homogeneous-merging characterization.
- **Plan 2.5 (potential)**: Cross-kernel-group BO sharing (rms_gemms_rope's `q_roped/k_roped/v` outputs aliased to FA's input BOs; FA's `attn_out` aliased to o_ffn's residual-add input). Production doesn't do this; could be a separate optimization study.
- **Tier A #4 / #5** from the master spec (last-token LM Head; CPU vs NPU LM Head GEMV).
- **All Tier B** (seq-first FA/RoPE; FA vs naive attention; CPU vs NPU decode attention; `omit_pingpong` toggling; LM Head partition sweep beyond {1, 8}).
- **Real HuggingFace weights.** Synthetic seed=42 only.

## 12. Isolation strategy

### 12.1 Worktree

```
git worktree add ../mlir-air-ablation-plan2 -b llama32-1b-ablation-plan2-prefill
```

The user's primary checkout at `/home/jiajli/apps/mlir-air/` (currently on
`llama-3.2-1B-devel`) is not perturbed. Plan 2 work happens in
`../mlir-air-ablation-plan2/` on its own branch. The user can review Plan 1
files in the primary checkout while Plan 2 develops.

### 12.2 File-level guarantee

Plan 2 code only **imports** from Plan 1's read-only modules
(`programming_examples/llama32_1b/ablation/cells/common.py:compile_standalone_kernels`,
`ablation/validate.py`, `ablation/cells/baton.py:_share_bo` may be lifted into
prefill/cells/common.py if needed but the original is not modified).

Production code (`programming_examples/llama32_1b/kernel_builder/cache.py`)
already has the `naive=True` mode from Plan 1; Plan 2 introduces no further
changes to it.

### 12.3 Merge plan

After Plan 2 is implemented and tested, the worktree branch is merged into
`llama-3.2-1B-devel` (or a parent branch as the user designates). Because
Plan 2 only adds files and never modifies existing ones, the merge is
fast-forward / no-conflict.

## 13. Risks

| Risk | Mitigation |
|---|---|
| 14 standalone ELFs at seq=2048 + FA = ~16 ELFs to compile, ~10–15 min one-time | Cached to disk after first compile; documented in README. |
| 16 layers × multiple weight tensors at seq=2048 ≈ 1 GB resident BO memory | Verified to fit on test machine; if not, fall back to 1-layer scope only. |
| Parameterized cell logic harder to debug than Plan 1's hardcoded form | KernelGroupSpec is a frozen dataclass; cells walk it mechanically. Unit tests on a mock cache verify each cell's call sequence per spec. |
| FA ELF first-time compile is ~46 s per `profile.md` | Compiled once, cached. Verified once via FA's own validation. |
| Cell A high BO traffic on 16-layer scope may dominate variance | Bump trial count to 10 for Cell A 16-layer if 5-run median ± range > 5 %. |
| Cross-cell consistency re-check (§5) may fail after long 16-layer runs if BO recycle has bugs | If failure occurs, suspend cell and report — don't trust timing. |

## 14. Success criteria

The study succeeds if it produces:

1. A reproducible harness (single `make all` from
   `programming_examples/llama32_1b/ablation/prefill/`).
2. Every reported cell passes the §5 correctness gate (per-cell + cross-cell
   bit-exact).
3. Numerical attribution for #1, #2, #3 in the prefill regime, per
   kernel-group AND aggregated.
4. Per-launch breakdown for both prefill kernel-groups (from Cell C).
5. Host-overhead share for each cell (single-layer and 16-layer scopes).
6. 16-layer total prefill wall-time numbers with confirmed (or corrected)
   comparison to `profile.md`'s 1.27 s headline.
7. Plan 1 files unmodified (`git diff main..plan2-branch` shows only file
   additions in `ablation/prefill/`).
