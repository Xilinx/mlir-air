---
name: phase-2-single-block-validation
description: Phase 2 of LLM deployment — wire the verified Phase 1 kernels into one transformer block on NPU and verify per-layer cosine vs the HF bf16 reference (the shared `programming_examples/llms/verify/` diagnosis lens, promoted to a gate at layer 0). Catches integration bugs (layout mismatches, missing transposes, type drops between kernel boundaries) before scaling to N layers.
---

## Purpose

Phase 1 verified each kernel × shape standalone. Phase 2 wires those
kernels into a single transformer block on NPU and compares the
block-level output to the **HF bf16 reference** at the same input. This
catches integration bugs (wrong tensor layouts at kernel boundaries,
dropped biases, missing padding) without the cost of running N layers.

The comparison reuses the `verify/` subsystem's **diagnosis lens**:
`verify/verify_runner.py:_run_diagnosis` already computes per-layer
`ffn_out` cosine (NPU vs HF bf16) for every layer. Diagnosis itself is
informational (no thresholds); Phase 2's job is to promote layer 0's
result to a **hard gate** with a head_dim-scaled threshold. The reference
is HF transformers in bf16 (same dtype as the NPU — fair fight), obtained
via `HfRunner` in diagnosis mode (`lite_mode=False`), NOT a hand-written
CPU forward.

## Phase 2 PASS criteria (HARD GATES)

Run the diagnosis lens on the canonical prompt and gate on **layer 0**
(the single block under test). All four must hold:

1. **Whole-tensor cosine ≥ 0.99** between NPU layer-0 `ffn_out` and HF
   bf16 layer-0 `ffn_out` (`hf.layer_intermediates[0]['ffn_out']`).
   Catches: coarse integration breaks (wrong layout, missing op).

2. **Per-position cosine ≥ THRESHOLD(head_dim)** at each real-token
   position (`[:real_len]`, NOT padded positions — those are
   out-of-distribution and amplify BF16 noise unhelpfully). The
   diagnosis comparator already returns per-position cosine
   {min, p5, median, mean} via `per_position_cosine` + `aggregate`; gate
   on `min`. Catches: per-row dropouts the whole-tensor cosine averages over.

   | head_dim | per-position min |
   |---|---|
   | ≤ 64  | 0.99 |
   | 128   | 0.98 |
   | ≥ 256 | 0.97 |

   Threshold scales with `head_dim` because BF16 accumulation noise
   grows as `√(head_dim · K)`: hd=64 deployments hit per-pos min ≈ 0.998;
   hd=128 hits ≈ 0.980 with ~5× LOWER MAE — the larger cosine drop is
   geometric, not a bug.

3. **No NaN** anywhere in NPU output.

4. **Result documented** in
   `<model>/docs/development_progress/phase2_block.md` (cosine numbers
   + tile/integration choice + any bisect findings if Step 4 fired).

**Also record `max_abs` and `max_rel` error** alongside the cosine numbers
(informational, not gated — absolute thresholds depend on input
distribution). The diagnosis comparator's `error_metrics` reports these.
The current BF16-output GEMM production path is the registry's
high-precision tier (FP32-accumulate + a single epilogue cast, fused-cast
or drain) at ~9.3e-3 `mean_rel_L1` — the GPU-standard accuracy, single-
sourced from `details/GEMM_bf16_in_bf16_out.json`; the low-precision
direct-codegen tier (1.3e-2–1.9e-2) is the fallback. Across 7 GEMMs +
softmax + RoPE + RMSNorm the block stays in that high-precision band.
Recording them gives
future deployments a regression baseline (NPU max_abs ≤ 1.5× reference
deployment's measured value at same shape signals no regression).

> **Why cosine here, and why it's only an interim gate.** A 2026 survey of
> industry practice (vLLM, HF transformers, llama.cpp, MLPerf, TensorRT-LLM,
> the GPTQ/AWQ/SmoothQuant literature) found that **per-layer activation
> cosine is NOT a standard correctness gate** — everyone gates either
> end-to-end (vLLM token-set, llama.cpp logit-KL, MLPerf ≥99%-of-reference)
> or per-tensor on element-wise atol/rtol / SQNR vs an FP32 reference. The
> one stack that does gate on per-layer cosine (TPU-MLIR) uses **0.99**, not
> a loose value. So treat this gate honestly:
> - At Phase 2 there is **no token output yet to score**, so we still need
>   *some* numeric tripwire on the block output — cosine vs HF bf16 is that
>   tripwire. It catches gross integration bugs (layout, missing transpose,
>   dtype drop), which is all Phase 2 needs.
> - It is **bf16-vs-bf16** (NPU bf16 vs HF bf16) — there is no FP32 ground
>   truth at this layer, so a tight element-wise atol/rtol (HF's FP32-parity
>   bar) would mis-fire on benign rounding-order differences; cosine's
>   direction-only nature is why it tolerates that. That tolerance is also
>   its weakness (it can pass on magnitude errors), which is exactly why
>   it's interim, not the final word.
> - The **real correctness gate is end-to-end** (Phase 3/6 token-set top-5,
>   which mirrors vLLM `check_logprobs_close`). Once the full model exists,
>   that gate — not cosine — decides correctness. A future upgrade could add
>   SQNR (≥40 dB, PyTorch Numeric Suite's "very good alignment" bar) as a
>   more defensible per-block number; thresholds unchanged for now.

## Knowledge base references

PRIMARY:

- `programming_examples/llms/llama_kernel_builder/` — the shared toolkit
  (KernelCache, stitching, external_kernels) you compose the block FROM
  (kernel-first default).
- `programming_examples/llms/llama32_1b/multi_launch_builder/` +
  `llama32_1b_prefill.py:run_transformer_block` — the reference exemplar:
  read to see how the leaf kernels stitch into a block. On a bit-for-bit
  kernel-sequence match you may call `run_transformer_block` directly
  (inheritance shortcut).
- `programming_examples/llms/verify/verify_runner.py:_run_diagnosis`
  — the per-layer NPU-vs-HF-bf16 cosine lens Phase 2 promotes to a gate.
- `programming_examples/llms/verify/runners/hf_runner.py` — how the
  HF bf16 reference exposes per-layer `ffn_out` (`lite_mode=False`).
- `<model>/docs/development_progress/` (Phase 1 output) +
  `programming_examples/kernel_registry/supported_kernels.md` rows with
  Used by = `<model>` — Phase 1's verified (kernel, shape) list; Phase 2
  must wire ALL of them.
- `programming_examples/kernel_registry/details/<Kernel>_bf16.md`
  — kernel-by-kernel reference (datapath, tile rules, constraints, layouts).

WORKAROUNDS (apply when model config triggers them — re-derive from the HF
reference impl; the patterns below describe the technique, not a shipped file):

- **GQA-aware reindexed padding** for non-1024-aligned dims (see Step 2)
- **Host-side post-RoPE bias add** for QKV-bias models (see Step 2)

## Workflow

### Step 1: Choose integration path

**Kernel-first (default).** Derive the model's per-layer kernel sequence
from its config and build the block by composing the registry leaf kernels
(verified in Phase 1) into model-specific multi-launch ELFs under
`<model>/multi_launch_builder/`, using the shared `llama_kernel_builder`
toolkit (KernelCache, stitching, external_kernels). This is the general
path — it does not assume the model resembles llama, so it generalizes to
any decoder-only architecture in scope.

Read `llama32_1b`'s assembly (`llama32_1b_prefill.run_transformer_block`
and `llama32_1b/multi_launch_builder/*`) as a **worked exemplar** of how
the leaf kernels stitch into a block — mirror its structure, adapting the
kernel sequence and shapes to your model.

**Inheritance (shortcut).** ONLY when the model's per-layer kernel
sequence matches llama's bit-for-bit —
RMSNorm → Q/K/V GEMM → RoPE → FA → O → add → RMSNorm → Gate/Up → SwiGLU → Down → add —
you may skip writing builders and call
`llama32_1b_prefill.run_transformer_block` directly with the new shape
parameters. This is an optimization for genuine llama variants, not the
starting assumption. Any of these breaks the bit-for-bit match and forces
the kernel-first path:

  (a) NEW op type (e.g., Qwen3's Q/K Norm — per-head RMSNorm with
      `(head_dim,)` weight)
  (b) NEW op needs to land BETWEEN currently-fused launches (e.g.,
      Q/K Norm sits between Q/K projection and RoPE, but
      `rms_gemv_rope` fuses both)
  (c) Op REORDER (post-norm vs pre-norm)

Either way, don't write new C kernels speculatively — almost always the
leaf kernel exists in the registry; the trick is the right way to STITCH.
For Q/K Norm specifically, `weighted_rms_norm` with the heads-as-M trick
(M=n_heads, N=head_dim, sharing the (head_dim,) weight across rows) IS
the op.

### Step 2: Apply config-specific prereqs (only if model needs them)

Two known triggers from model config (NOT from upstream phases):

**Non-1024-aligned `emb_dim` or `hidden_dim`** → BD pool exhaustion
risk at long seq (see the kernel's `details/<Kernel>_bf16.md` placeability
notes). Use GQA-aware
reindexed padding: pad up to a 1024-aligned multiple by inserting phantom
Q heads INSIDE each KV group (not at the end — naive padding breaks GQA
semantics by changing `n_heads / n_kv_heads = group_size`). CPU-only
sanity test the padded vs orig forward FIRST (cosine should be 0.999998+)
before touching NPU.

**`qkv_bias=True`** (Qwen2 / Qwen3 family) → host-side post-RoPE bias add,
exploiting RoPE's linearity: `RoPE(q + bq) = RoPE(q) + RoPE(bq)`. The
`rms_gemms_rope` ELF stays bias-free; bias is added on host after the ELF
returns.

If both: padding determines the n_heads count the bias precompute uses.

### Step 3: Wire one block + numerical check

In `<model>/<model>_prefill.py`, implement
`run_single_block(layer_idx=0, hidden, weights, ...)`:

- **Kernel-first path (default)**: call your new
  `run_transformer_block_<model>(...)` that runs the per-model multi-launch
  ELFs in order via the shared `KernelCache`. Minimal skeleton:
  ```python
  from llama_kernel_builder.cache import KernelCache
  cache = KernelCache()                      # compile-once, run-many
  def run_transformer_block_<model>(hidden, weights, cfg, cache):
      # one _run_cached per fused ELF you built in Step 1, in order:
      x = cache._run_cached("rms_qkv_rope", hidden, weights.qkv, ...)   # RMSNorm+Q/K/V+RoPE
      x = cache._run_cached("attn",         x, ...)                     # FA
      x = cache._run_cached("o_ffn",        x, weights.o, weights.ffn, ...)  # O+add+RMSNorm+SwiGLU+Down+add
      return x
  ```
- **Inheritance path (shortcut, bit-for-bit match only)**: call
  `llama32_1b_prefill.run_transformer_block(...)` with this model's shape
  parameters

Use the canonical prompt from the deployment's verify prompt set
(`verify/prompts/{base,instruct}.txt`). Get the layer-0 reference from the
HF bf16 runner and compare:

```python
from verify.runners.hf_runner import HfRunner
hf = HfRunner(hf_model_id, config, max_seq, lite_mode=False)
hf_pf = hf.prefill(prompt_tokens)
ref_block0 = hf_pf.layer_intermediates[0]["ffn_out"]   # HF bf16, layer-0 output

npu_block0 = run_single_block(layer_idx=0, hidden=x, weights=weights, ...)
```

Compute whole-tensor cosine and per-position cosines (real-token positions
only) with the diagnosis comparator
(`verify/comparators.py:per_position_cosine` + `aggregate`). Check against
the PASS criteria above. The simplest route is to run `make diagnosis` and
read layer 0's row from the report; the manual snippet above is for when
you need to gate inside a Phase-2 test script.

### Step 4: Bisect on FAIL

If cosine fails, the integration is broken at one specific kernel
boundary. Bisect by swapping NPU kernels back to a CPU equivalent one at a
time (use `<model>_cpu_helpers.py` for the ops that have a helper —
`rms_norm`, `attention_reference` — and a small inline numpy for the rest):
walk forward through the block, replacing `npu_<kernel>(...)` with the CPU
equivalent, recompute cosine. The first replacement that pushes cosine
above threshold identifies the offender — that's where the layout / type /
argument mismatch lives. Invoke `superpowers:systematic-debugging` on it.

Record the bisect table (per-step cosine) in `phase2_block.md` so future
deployments learn from this specific failure.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Cosine drops at Q/K/V GEMM | weight loading / tensor layout (seq-first vs heads-first) | Compare NPU output shape to reference's; check `np.ascontiguousarray()` after weight load |
| Cosine drops at FlashAttention | causal masking missing / wrong dk_chunks compile flag | See `debug-fa-runtime-failure` |
| Cosine drops at Down GEMM | BF16 truncation; running the low-precision (direct-codegen) tier instead of high-precision | confirm the GEMM uses the registry's high-precision path (fused-cast / drain = FP32-accumulate + single cast), not `--high-precision false`; see `details/GEMM_bf16_in_bf16_out.md` |
| NaN in output | uninitialized BO / reused stale buffer | Invoke `debug-bo-corruption` |
| Cosine drops at residual add | bias forgotten on padded path / GQA reindex bug | If padding+bias model: re-run CPU sanity test on padded forward (Step 2) |
| Whole-tensor cosine OK but per-position min low | one bad position run; check whether last few positions diverge (causal mask edge case) | Print per-position cosine, look for contiguous bad runs |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 2 PASS:

- Append cosine + per-position min + integration-path choice to
  `<model>/docs/development_progress/phase2_block.md`
- Mark Phase 2 in `<model>/TODO.md`
- If Step 2 padding or bias workarounds were used, surface as a
  Phase 4/5 prerequisite ("perf optimization must preserve the
  padded/bias wrappers")
