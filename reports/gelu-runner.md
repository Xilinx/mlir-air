# GELU (tanh approximation) BF16 — kernel-runner report (Gates B/C/D/E)

SmolVLA A3 (vision) step. GELU-tanh promoted to registry kernel #8, verified on
real NPU2 (Strix / AIE2P), 2026-07-21. Pure kernel validation, no model integration.

## Files written (paths)

Committed (worktree branch `smolvla`, commit 8008e853):
- `programming_examples/gelu/gelu.py` — harness (type-(a) change, +88/−46)
- `programming_examples/gelu/Makefile` — N/TILE_N/HERD_X/HERD_Y/profile overrides
- `programming_examples/kernel_registry/details/GELU_bf16.md` — public detail page (E2/E3/E5)
- `programming_examples/kernel_registry/supported_kernels.md` — index row + scope L16 + "GELU-and-tanh — tested shapes" section (E4)
- `programming_examples/kernel_registry/README.md` — scope + file table + roadmap + methodology note

Internal triplet (gitignored, on disk at
`programming_examples/kernel_registry/details/internal_GELU_bf16/`):
- `01_implementation.md`, `02_precision.md`, `03_performance.md`, `README.md` (E1)
- `data/{sweep_results.csv, sweep_refine.csv, placement_map.csv, best_tile.csv}`
- `scripts/{gelu_sweep.sh, gelu_refine.sh}`

## Sweep command + CSV path

```bash
bash programming_examples/kernel_registry/details/internal_GELU_bf16/scripts/gelu_sweep.sh   # herd_x + herd_y probe + tile_n + per-shape @ 8x1
bash programming_examples/kernel_registry/details/internal_GELU_bf16/scripts/gelu_refine.sh  # 16-tile refine (8x2/4x4) + tile_n@16 + per-shape @ 8x2 best
```
Raw CSVs: `internal_GELU_bf16/data/sweep_results.csv` (32 rows), `sweep_refine.csv` (31 rows).
All runs flock'd on `/tmp/mlir-air-npu.lock`.

## Best config per shape (all herd_x=8 / herd_y=2 / tile_n=4096)

| N | (2-D) | latency | bandwidth | mean_rel_L1 | abs_err max |
|---|---|---|---|---|---|
| 1048576 | — | 235 µs | 17.8 GB/s | 8.4e-3 | 1.56e-2 |
| 2097152 | — | 379 µs | 22.1 GB/s | 8.4e-3 | 1.56e-2 |
| **3145728** | **1024×3072 (vision)** | **522 µs** | **24.1 GB/s** | **8.4e-3** | **1.56e-2** |
| 4194304 | 2048×2048 | 672 µs | 25.0 GB/s | 8.4e-3 | 1.56e-2 |
| 8388608 | — | 1244 µs | 27.0 GB/s | 8.4e-3 | 1.56e-2 |

## Precision command + numbers

```bash
cd programming_examples/gelu/build_peano
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 ../gelu.py --n 3145728 --tile-n 4096 --herd-x 8 --herd-y 2 --perf-iters 30
```
Output (this run):
```
Latency (us): 525.8
[precision] Output 0 (3145728 elements): mean_rel_L1=8.408e-03 | rel_err max=1.000e+00 | abs_err max=1.562e-02 | rtol=1.6e-02 atol=5.0e-02
PASS!
```
Reference confirmed **tanh approximation** (gelu.py:149-159 uses np.tanh, matching
SmolVLA GELUTanh at profile_cpu_baseline.py:98) — NOT erf-GELU. Diagnostic: ideal
per-step bf16 datapath (np.tanh) = 1.78e-3; NPU 8.4e-3 (~4.7×) → error dominated by
the hardware __builtin_aie2p_tanh LUT, not a bug. Bit-identical across all configs
(mean_rel_L1 ∈ [8.406e-3, 8.412e-3], abs_err max = 1.56e-2 everywhere).

## Knob exposed (type-(a), B6)

Exposed the hardcoded `num_tiles=2` / `sizes=[1,2]` (1×2 herd) as `--herd-x`/`--herd-y`
(default 1/2 preserves the original herd → lit test PASS, confirmed via `make run`).
Added `--perf-iters`. Upgraded the weak 100-sample stochastic check (rtol 1e-1) to a
full-output FP32 element-wise gate + `report_precision`. Compute chain (gelu.py:124-132),
tanh, and constants unchanged. `git diff --stat` = +88/−46.

## Key finding (D2/D5)

GELU is **single-input → 2 DMAs/tile** (vs SiLU/EltwiseAdd's 3), so `herd_y>1`
**places**. Best = **8×2 (16 tiles)**, twice the 8-tile ceiling of the 3-DMA
elementwise kernels; ~1.8× the bandwidth of the 8×1 config a SiLU-based assumption
would have picked. 8×4 (32 tiles) does not place; herd_y≤4 (4 AIE rows). herd_x
scales near-linearly 7.6× (1→8); tile_n nearly inert (<3% across {512…6144}). D5
recommendation recorded: SmolVLA vision integration must pass `--herd-x 8 --herd-y 2`
explicitly (builder default 1×2 exists only for the lit test).

## Ambiguity hit

None on numerics. One coordination note: a concurrent agent added a **LayerNorm**
kernel to supported_kernels.md/README.md during this session; my edits were made to
coexist (GELU row added after RoPE, scope line includes both). No conflict.

## Gate self-check

- B1-B6: 01_implementation.md — dtype layering, inline datapath with gelu.py file:line,
  codegen path count (1, direct-codegen), GELU-vs-SiLU comparison table, SmolVLA usage
  table, harness diff recorded.
- C1-C5: 02_precision.md — full-output FP32 tanh-approx ref (C1), element-wise gate no
  cosine (C2), no reduction so C3 N/A, tolerances justified vs GPU std (C4), every shape
  + config-independence shown (C5). C6 N/A (stateless; precision+perf same run is safe).
- D1-D5: 03_performance.md — herd_x=8 full width (D1), swept herd_x/herd_y/tile_n with
  the herd_y knob exposed & swept (D2), best selected from sweep (D3), bandwidth reported
  w/ arithmetic-intensity justification (D4), SmolVLA-vs-best compared (D5).
- E1-E5: internal triplet+README exist, public table zero `—`, both reproduce commands,
  index row matches, boilerplate matches SiLU page.
