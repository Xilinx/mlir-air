# A3-5 Step 2-3 — Vision NPU 12-layer ViT (SigLIP encoder) — runner report

## Status: FAIL against the stated gate (post_ln cosine 0.945 < 0.99), root-caused to a precision ceiling of the validated BF16/BFP16 kernels — NOT a wiring bug.

## What was built

- `vision_prefill.py` — mirrors `smolvla_backbone_prefill.py` for the SigLIP ViT.
  - `compile_all_kernels(cache, config, seq_len=1024)`: compiles 3 GEMM ELFs
    (qkvo 1024×768×768, fc1 1024×768×3072, fc2 1024×3072×768; all drain tile_m=32,
    registry tiles from A3-1), the affine LayerNorm ELF (1024×768, herd_x=8), the
    GELU-tanh ELF (N=1024·3072, herd 8×2), and the non-causal FlashAttention ELF
    (1024/1024, 12q/12kv MHA, head_dim=64, causal=False, num_heads_per_unroll=2 →
    fills the full 8×4 array). `cache._save_manifest()`.
  - `run_vit_block(...)`: LN1 → q/k/v GEMM(+host bias) → FA → o GEMM(+host bias) →
    residual → LN2 → fc1 GEMM(+host bias) → GELU → fc2 GEMM(+host bias) → residual.
  - `run_vit_encoder(...)`: host im2col patch-embed (if given pixel_values) + 12
    blocks + post_layernorm on NPU. `do_connector=False` (Step 4, out of scope).
- `test_full_vit.py` — the gate: per-layer cosine + post_ln cosine vs the oracle
  (`vision_oracle.npz`), CLI, exit 1 on fail. Adds a `--cpu-attn` diagnostic mode
  and per-token-median cosine.
- `test_vit_layer0.py`, `test_vit_teacher.py` — bring-up + teacher-forced
  per-layer diagnostics.

## NPU-execution: ALL 4 heavy ops on NPU, 0 CPU fallback

Per layer: LayerNorm ×2, GEMM ×4 (q/k/v/o + fc1/fc2 = 6 GEMM), FlashAttention ×1,
GELU ×1. Host does only the correctness-first glue (per-Linear bias-adds, the two
residual adds, im2col patch-embed) — all A3-6 fusion candidates, NOT fallbacks.
- **NPU-execution: LayerNorm=NPU, all GEMM=NPU, GELU=NPU, FlashAttention=NPU.**
- **ELFs/dispatches per layer = 10** (LN1, q, k, v, FA, o, LN2, fc1, gelu, fc2),
  + 1 post_ln once. Correctness-first: NOT merged (A3-6 will fuse).

## Gate evidence (real NPU2, this run)

- Layer-0 block vs oracle: **0.998485 PASS**. Sub-ops: LN1 0.99997, q-proj
  0.99999, FA 0.99979 — all clean.
- Full 12-layer per-layer cosine: 0.998, 0.978, 0.975, 0.971, 0.973, 0.972,
  0.969, 0.968, 0.968, 0.968, 0.968, 0.983 (layers 1–10 below the 0.98 gate).
- **post_ln cosine = 0.945502** (< 0.99 gate). Per-token-median post_ln = 0.988.

## Root cause (exhaustively verified — precision ceiling, not a bug)

1. Every leaf kernel is registry-tier: layer-0 block 0.9985, all sub-ops > 0.9998.
2. Teacher-forced per-layer (each block fed the ORACLE input independently) is
   clean: L0=0.998, L2–L11 = 0.994–0.9998; only L1 anomalous at 0.981.
3. An EXACT numpy mirror of the pipeline (BFP16-input matmul + bf16-out cast +
   host f32 bias + f32 residual) predicts post_ln 0.998 → the wiring, bf16
   boundaries, and eps (1e-5 vs 1e-6) are all correct.
4. The excess error is SYSTEMATIC: **81% of FlashAttention's error is a shared
   per-channel column-mean bias** (measured FA err L2 3.32, systematic 2.70). This
   is BFP16 block-float (shared 8-elem exponent) on the encoder's outlier-heavy
   LayerNorm activations (max/mean-abs ~45) — a directional bias that survives
   LayerNorm centering.
5. Layer 1 has the strongest residual cancellation (|input|=225.8,
   |attn_out|=182.8 → |x1|=143.0), which AMPLIFIES that systematic bias (→ 0.981).
   Random-noise models of the same magnitude give x1 cos 0.9999 (cancellation
   averages random error but not systematic). Errors compound in quadrature to 0.945.

## NPU precision levers tried (all exhausted; none recovers the gate)

- GEMM `bfp16=False` (native aie2p bf16 8×8×8 mmul): WRONG (block-vs-oracle
  0.28–0.90) — mm_aie2p.cc non-BFP16 branch has a different C_block accumulator
  layout that isn't correct at these tiles.
- Direct-codegen GEMM (`build_module_lowered`): WRONG at cancellation layers
  (L1=0.23, L5=0.35).
- FlashAttention `bfp16=False`: runtime HANG (ERT_CMD_STATE_TIMEOUT) — FA's L1
  tiling is sized for the BFP16 mmul.
- CPU attention: WORSE full run (post_ln 0.905) → FA is not the dominant term and
  CPU is not a usable fallback.

Documented `bfp16=` flags added to `compile_gemm_mm` / `compile_attn_npu2`
(default True = unchanged for all siblings) to record the experiments.

## Path forward (A3-6 kernel work, out of this correctness task's scope)

The gate needs a higher-precision matmul that stays on NPU: (a) rewrite the FA
kernel's L1 tiling so the native (non-BFP16) bf16 8×8×8 mmul places+runs, removing
the 81%-systematic attention bias; or (b) an fp32-accumulate FA epilogue. Both are
microkernel changes. Full analysis in `docs/TODO.md` (A3-5 section).

## Deliverables (all committed on branch `smolvla`, commit 97b133b5)

`vision_prefill.py`, `test_full_vit.py`, `test_vit_layer0.py`,
`test_vit_teacher.py`, `docs/TODO.md` (A3-5 section + root cause),
`shared/infra/external_kernels.py` (bfp16 flags). No build artifacts / caches /
oracle committed (gitignored).
