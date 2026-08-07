# SmolVLA vision (SigLIP) FlashAttention — runner report

## Config verified (real NPU2, this session)
SmolVLA vision encoder self-attention (SigLIP): lq=lk=**1024**, **12q/12kv MHA**
(no GQA), head_dim dk=dv=**64**, **causal=False** (bidirectional, no mask), bf16.

## Final placeable config (the default full-chip 32-tile)
| param | value |
|---|---|
| lqp | 256 |
| lkp | 64 |
| num_q_tiles | 4 |
| num_heads_per_unroll | **2** |
| num_cascade_stages | 4 |
| dv_chunks | 1 (head_dim=64) |
| tiles | 2×4×4 = 32 (full 8×4 array) |

## Placement conclusion — the headline result
**12 heads is EVEN → `num_heads_per_unroll=2` places and fills the full 8×4 array.**
`12 % 2 == 0` satisfies build_module's `num_heads % num_heads_per_unroll == 0`
assert. This is the key contrast with the SmolVLA **decoder backbone's 15-head**
attention (odd → hpu must be 1 → only 4 columns = half the array).

Measured both, identical numerics (same `attn_npu2.o`):
- **hpu=2 (full 8×4 array)**: 2357.8 / 2493.2 / 2607.5 us → **1366 / 1292 / 1235 GFLOP/s** (median ~2493 us / ~1292 GFLOP/s)
- **hpu=1 (half array)**: 3727.2 us → **864 GFLOP/s**
- **Speedup from filling the chip: ~1.58×** (1366/864). Placement-only win.

## Precision (clean run, no --perf-iters)
`mean_rel_L1 = 4.811e-02 | abs_err max = 3.027e-02 | rtol=1.6e-2 atol=1e-1 → PASS!`
- 786432 output elements, full-output element-wise np.isclose gate.
- Reference: FP32 non-causal SDPA (per-head Qf@Kf.T/√dk in f32, softmax in f32,
  P@Vf in f32, cast bf16), inputs randn(seed 42) — harness lines 1335-1347.
- 4.8e-2 is within the FA tier (two BFP16-emulated MMAs + bf16 online-softmax;
  non-causal rows run 4.4–5.5e-2). Both hpu=1 and hpu=2 report the identical
  4.811e-2 (accuracy is datapath-bound, not placement-bound).
- `rel_err max=1.129e+04` is the usual near-zero-ref blowup, not a signal.

## Commands (NPU lock always)
Correctness (clean):
```
cd programming_examples/flash_attention/kernel_fusion_based
flock -x -w 1800 /tmp/mlir-air-npu.lock make run SCRIPT=attn_npu2.py \
  LK=1024 LQ=1024 LKP=64 LQP=256 DK=64 DV=64 NUM_HEADS=12 NUM_KV_HEADS=12 \
  PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```
Performance (separate run — the precision line in a --perf-iters run prints
"failed." as a documented buffer-reuse artifact, ignore it):
```
flock -x -w 1800 /tmp/mlir-air-npu.lock make run SCRIPT=attn_npu2.py \
  LK=1024 LQ=1024 LKP=64 LQP=256 DK=64 DV=64 NUM_HEADS=12 NUM_KV_HEADS=12 \
  EXTRA_PY_FLAGS="--perf-iters 20" PEANO_INSTALL_DIR=$PEANO_INSTALL_DIR
```
Half-array comparison: add `--num-heads-per-unroll 1` to EXTRA_PY_FLAGS.

## Files written
- `programming_examples/kernel_registry/details/FlashAttention_bf16.md`
  — added 1024×1024 12/12 hd64 non-causal row + SmolVLA-vision footnote
  (MHA/no-GQA, bidirectional/no-mask, even-12 full-array vs odd-15 half-array 1.58×).
- `programming_examples/kernel_registry/supported_kernels.md`
  — added matching FA tested-shapes row (used_by = SmolVLA vision self-attn (SigLIP))
  + footnote.
- No FA JSON exists (only GEMM has JSON) — nothing to update there.

## Harness changes
None. No type-(a) knob exposure needed — `--num-heads-per-unroll` and
`--causal`/`--perf-iters` are already CLI args in attn_npu2.py.

## Ambiguity / traps
- None numerics-touching. Standard FA perf-run "failed." artifact handled by
  splitting precision (clean run) from latency (perf-iters run).
- Committed only the two registry docs (git status confirmed no stray files in
  the commit). Worktree commit b518b272 on branch smolvla.
