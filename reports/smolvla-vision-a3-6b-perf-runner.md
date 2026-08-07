# A3-6b — SmolVLA vision encoder performance hill-climb (runner report)

## Result: gate MET

- **NPU vision (fused): 141.6 ms** median-of-10 (compile excluded), vs the
  **281 ms** unfused baseline stated in the task → **< 281 ms gate cleared**.
  On THIS machine the same harness reproduces the unfused path at 368 ms
  (machine/contention variance vs the original 281 ms A3-6a run); the
  apples-to-apples delta here is **368 → 141.6 ms (2.6x)**. Either reference,
  the fused wall is far below 281 ms.
- CPU reference (same 12-layer encoder, this machine): 4093 ms.
- **Correctness held AND improved**: `test_full_vit.py` now PASSES the strict
  gate — all 12 per-layer cosines > 0.98, **post_ln cosine 0.945 → 0.990635**.
  (A cosine RISE, not a drop; see "Why cosine improved" below — it is not a
  masked fusion bug.)

## Levers applied

| Lever | Applied? | Wall (this harness) | Dispatches | post_ln cos |
|---|---|---|---|---|
| baseline unfused | — | 368 ms | 121 | 0.9455 |
| L1 BO intermediate reuse | yes | 368 ms (≈noise standalone) | 121 | 0.9455 |
| L2 ELF fusion + L3 on-device glue | yes | **141.6 ms** | **37** | **0.9906** |

- **L1 (opt-buffer-object-reuse)**: weights were already static
  (`static_input_indices={1}`); the remaining redundant traffic was
  re-uploading the zeroed output/scratch BOs each call. Marked them
  `intermediate_indices` in every `_run_*` helper. Standalone effect was within
  noise (tiny per-op buffers vs the 1.75 ms/dispatch python+XRT overhead) but is
  retained and inherited by the fused runner. Cosine unchanged (0.9455) ✓.
- **L2 (opt-merge-multi-launch-kernels) + L3 (host glue on-device)** — dominant
  win, applied together. Two fused ELFs built in `vit_fused_builders.py` via
  `shared/infra/stitching.stitch_elf`, mirroring the backbone's
  rms_gemms_rope / o_ffn:
  - `vit_ln_qkv` (7 launches): affine LN1 + Q/K/V drain GEMM + Q/K/V on-device
    broadcast bias-add.
  - `flash_attn` (1 launch, unchanged registry ELF).
  - `vit_o_ffn` (10 launches): O GEMM + O bias + residual + affine LN2 + fc1
    GEMM + fc1 bias + GELU-tanh + fc2 GEMM + fc2 bias + residual.
  Per-layer weight/bias BOs pre-loaded once (`static_input_indices` +
  `bo_key=f"...L{i}"`); every scratch/output BO `intermediate_indices`. All
  per-Linear bias-adds + both residuals now run ON-DEVICE (no host f32 glue).
- **L3 remainder NOT done**: im2col patch-embed stays on host — it is a
  one-time pre-loop op (not per-layer hot-loop work), so it costs nothing in the
  gated per-inference number. No throughput benefit to moving it. Not a fallback.

## Gate evidence

- `test_full_vit.py` (fused, default): `PASS: all 12 per-layer cosines > 0.98;
  final gate > 0.99` — post_ln cosine = **0.990635**.
- `test_vit_fused_elfs.py` (standalone ELF vs numpy mirror of the exact
  sub-graph): `vit_ln_qkv` q/k/v cos = 0.99992/0.99993/0.99993; `vit_o_ffn` out
  cos = 0.99986. Proves the fused math is exactly as designed BEFORE the oracle
  comparison — so the end-to-end rise is not hiding a compensating error.
- `bench_vision.py --iters 10`: NPU median 141.6 ms, CPU 4093 ms.
- Fused profile: wall 136 ms = 132 ms NPU-xrt + **4 ms host-gap** (was 212 ms),
  **37 dispatches** (vit_ln_qkv x12 @2.85ms, flash_attn x12 @2.87ms,
  vit_o_ffn x12 @5.24ms, layer_norm x1 @0.62ms for post_ln).

## Why post_ln cosine improved (0.945 → 0.990)

Expected, not a bug. The unfused path did each bias-add + residual on host in
f32 with a bf16→f32→bf16 recast per op; that intermediate re-quantization was
compounding the BFP16 systematic per-channel bias (the documented 0.945 ceiling).
The fused path adds bias/residual on-device in bf16 vector lanes with no
recast, which is strictly the more-accurate arithmetic, so the ceiling lifts.
Each fused ELF was validated in isolation vs a numpy mirror (0.9999) before
wiring, ruling out a compensating-error explanation.

## NPU-execution: all ops on NPU (0 CPU fallback)

- vit_ln_qkv: **NPU** (LN1 + Q/K/V GEMM + Q/K/V bias)
- flash_attn: **NPU** (non-causal MHA)
- vit_o_ffn: **NPU** (O + bias + residual + LN2 + fc1 + bias + GELU + fc2 + bias
  + residual)
- post_layernorm: **NPU** (standalone affine layer_norm ELF, 1 dispatch, end of
  stack)
- im2col patch-embed: host (one-time, pre-loop; documented, not a hot-loop
  fallback)

## ELFs/layer: 3 (vit_ln_qkv, flash_attn, vit_o_ffn) + 1 post_ln = 37 total (was 121)

## Merge-completeness (Gate I)

Both natural fusion groups merged into single ELFs (no group left unmerged).
Fused-ELF mm.o disambiguation gotcha resolved: drain GEMMs at tile_n 96 vs 128
forced to tile_n-keyed suffixes (`_m32_n96`/`_m32_n128`) via
`_force_tile_n_suffix` so each ELF links its correctly-baked mm.o (first attempt
without this: vit_ln_qkv cos 0.07 from a stale generic mm_m32.o).

## Deliverables (committed in worktree, branch smolvla, commit 5b6cf5e9)

- `vit_fused_builders.py` (new) — the two fused-ELF builders.
- `vision_prefill.py` — `compile_all_kernels(..., fused=True)` default +
  `run_vit_block_fused` (3 dispatches/layer, static weight BOs).
- `test_vit_fused_elfs.py` (new) — standalone NPU correctness of each ELF.
- `bench_vision.py` (new) — median-of-10 NPU vs CPU bench.
- `docs/TODO.md` — A3-6b results section.

Tag `smolvla-vision-unfused-v1` preserved (untouched); `fused=False` keeps the
frozen unfused path for A/B + diagnosis.
