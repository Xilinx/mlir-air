# Llama-3.2-1B Prefill Ablation (Plan 2)

Bit-exact 4-cell ablation of the production **prefill** pipeline:
`rms_gemms_rope` (6 launches) + FlashAttention (held constant) + `o_ffn`
(8 launches), at seq=2048 GEMM shapes, both single-layer and full 16-layer
scopes.

Companion docs:
- Plan 2 spec: [`../docs/specs/2026-05-07-llama32-1b-ablation-plan2-prefill-design.md`](../docs/specs/2026-05-07-llama32-1b-ablation-plan2-prefill-design.md)
- Plan 1 (decode pilot): [`../README.md`](../README.md)
- Production profile: [`../../docs/profile.md`](../../docs/profile.md)

## What this measures

Four cells, identical computation, different dispatch strategy:

| Cell | What changes within each kernel-group | Adds |
|------|---------------------------------------|------|
| A | 6 + 8 separate `xrt.run()` per layer, host round-trip on every intermediate | (baseline) |
| B | + per-layer weight BOs (`static_input_indices`) | #2 |
| C | + shared intermediate BOs across separate `xrt.run()` calls | #3 |
| D | + multi-launch merging (production: 6→1 + 8→1 ELF per layer) | #1 |

FA is held constant per spec (un-mergeable). Cross-kernel-group transfers
(rms→FA, FA→o_ffn) go through host in every cell — matches production.

## Pilot measurements (final smoke run)

### 16-layer total wall — comparable to profile.md's 1.27 s

| Cell | Median (s) | Range | Δ vs prev | Speedup | vs profile.md |
|---|---|---|---|---|---|
| A — Naive | 1.754 | [1.751, 1.755] | — | (baseline) | 1.38× slower |
| B — + per-layer weight BOs (#2) | 1.589 | [1.584, 1.594] | A→B = +0.165 s | **1.10×** | 1.25× slower |
| C — + shared intermediate BOs (#3) | 1.212 | [1.212, 1.222] | B→C = +0.377 s | **1.31×** | 0.95× faster |
| D — + multi-launch merging (#1) | 1.125 | [1.124, 1.127] | C→D = +0.087 s | **1.08×** | 0.89× faster |
| | | | **A→D total** | **1.56×** | |

5 trials per cell, drop trial 1 (warmup), median + (min, max) over remaining 4.
**Cell D = 1.125 s ≈ profile.md's 1.27 s** (small overshoot from embedding lookup, KV cache extraction, etc. not in this harness).

### Single-layer per-call medians (ms)

| Cell | rms_gemms_rope | o_ffn |
|---|---|---|
| A | 14.99 | 75.05 |
| B | 12.52 | 64.67 |
| C | 9.77 | 45.01 |
| D | 7.43 | 40.99 |

Per-call speedups: rms_gemms_rope A→D = 2.02×, o_ffn A→D = 1.83×.

### Findings

- **#3 (shared intermediate BOs) dominates in prefill** at 1.31× — *opposite of decode* where #3 ≈ 1.0×. In prefill, per-launch intermediates are large (e.g. 8 MB GEMM outputs at seq=2048) and the bandwidth saved by aliasing BOs is significant.
- **#2 (per-layer weight BOs) is small in prefill** (1.10×) — weights are big but the per-call NPU compute is much bigger, so weight-transfer cost is a smaller fraction of total time. (Decode is the opposite: weights dominate because per-call compute is small.)
- **Pure multi-launch merging (#1) is small in prefill** (1.08×) — same intuition: dispatch overhead matters less when per-call work is large.
- **Total A→D = 1.56× speedup** for prefill — smaller than decode's 2.75× because per-call work is much bigger, so dispatch-related overheads are a smaller share.
- **All 4 cells produce bit-identical output bytes** (validated against committed golden fixtures from Cell D), so timing differences are purely dispatch-related.

## Quick start

```
make compile          # one-time, ~10-15 min for 14 standalone ELFs + 2 merged + FA
make run              # 5 trials × both scopes × all 4 cells (~5-10 min)
make report           # markdown report
```

## Validation gate

Every cell's per-kernel-group output must match the committed `golden/*.npz`
fixtures bit-exactly (synthetic numpy seed=42 inputs). Cells failing the
gate suppress their timing in the report.

## Reproducibility

```
cd programming_examples/llama32_1b/ablation/prefill
make clean && make all
```

The 16-layer Cell D total wall time should be in the ballpark of
`profile.md`'s **1.27 s** production headline. The marginal deltas table
attributes how much each of optimizations #1, #2, #3 contributes to that
number for prefill specifically.

Unit tests (NPU-free):

```
python3 -m pytest tests/ -v
```

Expected: 8 passed (4 KernelGroupSpec + 4 validation gate).

## Limitations of this plan (Plan 2-decode and Plan 2-lm-head will address)

- Prefill only — decode `o_gemv_ffn` and the LM Head L1/L8 mini-study are separate plans.
- FA is invariant in every cell. A potential **Plan 2.5** could ablate cross-kernel-group BO sharing (FA's input BOs aliased to rms_gemms_rope's output BOs).
- Synthetic weights only. No HuggingFace.

## File map

| Path | Purpose |
|------|---------|
| `specs/kernel_group.py` | Frozen dataclasses (SubLaunchSpec, BatonLink, KernelGroupSpec) |
| `specs/{rms_gemms_rope,o_ffn}.py` | Concrete spec instances |
| `standalone_builders/` | Re-exported STANDALONES registries |
| `cells/cell_{a,b,c,d}_*.py` | Parameterized cell harnesses |
| `cells/flash_attn_const.py` | FA invariant |
| `cells/multi_layer.py` | 16-layer wrapper |
| `cells/common.py` | Compile harness, BO baton-pass helper, public-func-name extractor |
| `golden/` | Two committed npz fixtures + regen script + meta json |
| `validate.py` | Parameterized bit-exact gate |
| `run_ablation.py` | Orchestrator |
| `analyze.py` | Report generator |
| `Makefile` | Convenience targets |
