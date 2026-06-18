---
name: phase-5-decode-optimization
description: Phase 5 of LLM deployment — apply decode-specific optimization patterns to a Phase-4-correct pipeline (multi-launch merge with N-way extern rename, static weight BOs, NPU LM Head GEMV, CPU→NPU promotion). Each step preserves correctness by re-running the Phase 3 gate — `make verify` (token-set vs HF bf16) is the PASS/FAIL gate; `make diagnosis` per-layer cosine is the informational lens used to localize a regression. Invoked after Phase 4 PASS.
---

## Purpose

Phase 4 optimized prefill while preserving Phase 3 correctness. Phase 5
does the same for decode — but the dominant patterns differ because
decode runs at M=1 per token, calling all N layers once per generated
token. This amplifies the value of static weight BOs (weights are
loaded once but read on every token) and LM Head GEMV (was a CPU
bottleneck dwarfing all kernel time).

The reference deployment llama3.2-1B took decode from ~500 ms/token →
92 ms/token (5.4×) by composing the patterns below.

## Phase 5 PASS criteria (HARD GATES)

1. **Correctness preserved**: after every applied pattern, **`make verify`
   (the token-set gate vs HF bf16) still PASSES** — this is the Phase 3
   correctness gate, re-run between patterns. `make diagnosis` per-layer
   cosine is NOT a gate (the verify subsystem retired threshold-based
   diagnosis; `compare_pair` reports cosine with no pass/fail), and note it
   only probes *prefill* — decode regressions (KV cache, per-token BO
   reuse) surface in `make verify`'s 32-token generation, not in diagnosis.
   If `make verify` regresses to FAIL, **revert the pattern** and document
   why.
2. **Decode time/token strictly < Phase 4 baseline**, measured with
   `make profile` at the same canonical prompt.
3. **Per-pattern outcome documented** in
   `<model>/docs/development_progress/phase5_decode.md`: for each of
   the 4 patterns, record `applied / skipped / reverted`, the latency
   delta, and a one-line reason.

The "≥ N patterns applied" check is NOT a gate — some models
legitimately need only A + B (most kernel-first deployments don't need
LM Head promotion if Phase 4 already moved it). The gate is the
outcome (decode time improved + correctness preserved).

## Knowledge base references

PRIMARY:

- `programming_examples/llms/llama32_1b/docs/profile.md` — the reference
  deployment's profiling breakdown; the reference for what "good" looks like
- `programming_examples/llms/<model>/docs/development_progress/phase4_prefill.md`
  — Phase 4 baseline (prefill numbers + the integration path used)
- `programming_examples/llms/llama32_1b/multi_launch_builder/o_gemv_ffn_multi.py`
  — decode-specific merge pattern + 2-K extern kernel rename, in code
- `programming_examples/kernel_registry/details/GEMV_bf16.md` — per-kernel
  constraints / placeability notes (the authority for that kernel's hard
  limits): K_max=8160, combined channel reads ≤ 255, L2 cap — relevant
  when GEMV K > 8160 or M is large

REFERENCE EXEMPLARS (read/mirror to compose your own decode ELFs; import
directly only on a bit-for-bit kernel-sequence match):

- `programming_examples/llms/llama32_1b/multi_launch_builder/rms_gemv_rope_multi.py`
  — fused 6-launch decode ELF for RMSNorm + Q/K/V GEMV + RoPE Q/K
- `programming_examples/llms/llama32_1b/multi_launch_builder/o_gemv_ffn_multi.py`
  — fused 8-launch decode ELF (with 2-K extern rename for K=8192 Down)
- `programming_examples/llms/llama32_1b/multi_launch_builder/lm_head_gemv_multi.py`
  — vocab-partitioned LM Head GEMV (Pattern D)
- `programming_examples/llms/llama_kernel_builder/` — the shared toolkit
  every decode-ELF build uses (KernelCache, stitching, external_kernels).

KERNEL-FIRST NOTE (the default path — compose, don't inherit):

- For `n_heads*head_dim != emb_dim` models, an N-way (3-K) extern kernel
  rename is needed (a 3rd distinct GEMV K beyond the llama 2-K pattern).
  Mirror the 2-K rename in `o_gemv_ffn_multi.py` and add one more renamed
  `.o`. Per-call XRT overhead dominates decode, so fusion + rename can give
  a large wall speedup.

## Workflow

### Step 1: Measure Phase 4 baseline

Capture the decode time/token before any Phase 5 pattern:

```bash
cd programming_examples/llms/<model>
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile
```

Record: ms/token + per-layer + LM head time breakdown (where the
budget goes today). These numbers gate every pattern below.

### Step 2: Apply optimization patterns

Apply A → B → D (skip C unless decode introduced a layout transpose
that Phase 4 didn't fix). Between each, re-run the Phase 3 gate (`make
verify`) and re-measure profile.

#### Pattern A — Multi-launch merge (decode variants)

Stitch decode kernel groups into fused ELFs (mirrors Phase 4 Pattern A
but with GEMV instead of GEMM). Two paths — the **same kernel-first-vs-
inheritance decision Phase 2 made** (the bit-for-bit match rule lives once
in `phase-2-single-block-validation` Step 1; reuse that verdict, don't restate it):

| Path | When | What to do |
|---|---|---|
| **Build fused ELF (default)** | Phase 2 integrated kernel-first (the general path) | Write decode-specific builders in `<model>/multi_launch_builder/`, mirroring `llama32_1b/multi_launch_builder/*` as the worked exemplar. Invoke `merge-multi-launch-kernels`. |
| **Reuse existing fused ELF (shortcut)** | Phase 2 took the inheritance shortcut (bit-for-bit llama match) | Import `llama32_1b/multi_launch_builder/{rms_gemv_rope_multi, o_gemv_ffn_multi, lm_head_gemv_multi}` directly — the llama-variant fast path. |

Expected: 10 launches/layer/token → 2-3 launches/layer/token.

**Sub: extern kernel rename for shape-collision.** When multiple GEMV K
values coexist in one fused ELF, they collide on the
`@matvec_vectorized_bf16_bf16` symbol. Compile `mv.cc` with `-D` symbol
renames per group, link them together. Two sizes:

- **2-K (llama pattern)**: default `mv.o` for K=2048 + `mv_k8192.o`
  (`-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16` etc.)
  for K=8192 Down. See
  `llms/llama_kernel_builder/external_kernels.py`.
- **N-K (n_heads·head_dim ≠ emb_dim)**: introduces a 3rd K. Add another
  renamed `.o` (e.g. an `og_matvec_*` for the O/Gate K and a
  `dg_matvec_*` for the Down K), route per-launch via the rename
  allowlists. Mirror the 2-K rename in `o_gemv_ffn_multi.py` and add the
  extra symbol set.

**Sub: K-split for K > 8160** (see `details/GEMV_bf16.md`):
auto-split GEMV K-DMA caps at outer = 255 → max practical K ≈ 8160.
Above this, pass `down_k_split=N` (where `K % N == 0` AND `N ≤ 255`
AND `K/N ≤ 1023`). Examples: qwen25 K=8960 → `down_k_split=70`;
llama3-8B K=14336 → `down_k_split=56`. Same `k_split` knob is exposed
on `matvec.build_module` directly (default None = back-compat).

#### Pattern B — Static weight BOs (decode amplifies the win)

Decode reuses every weight on every token. Convert weight BOs to
allocated-once with `bo.map()` zero-copy access:

- Allocate all per-layer weight BOs in `prepare_runtime()`
- Use `bo.map()` to expose host-writable views and write once
- On every decode call: pass `static_input_indices=[<weight_indices>]`
  to skip the per-token re-write

Expected: removes per-token BO write of all weights. With 16+ layers
× 7 weight tensors per layer × 100 tokens, this is the dominant
host-side decode overhead pre-Pattern-B.

#### Pattern D — CPU→NPU promotion (LM Head is the headline)

**D1. NPU LM Head GEMV (the big win).** Pre-Phase-5, LM Head is often
on CPU (`logits = hidden @ embed.T`). Replace with NPU GEMV
partitioned across vocab. llama3-1B used 8 partitions, each handling
`vocab/8` rows × `emb_dim` cols, compiled into `lm_head_gemv.elf`.
Llama3 saw **~250 ms → ~14 ms**.

Choosing partitions: pick the largest partition that fits one tile's
L2 budget (the GEMV staged-buffer limit, see `details/GEMV_bf16.md`). For
vocab=128256, 8 partitions × 16384 rows fits; for larger vocab, increase
partitions.

**Combined-channel constraint at large M** (see `details/GEMV_bf16.md`): with LM-head
partitions M ≥ 16384, the B-input shim DMA fires `launch_count ×
(tile_m/m_input)` times per GEMV; combined GEMVs sharing a channel
add up. Stay under 255: set `tile_m = m_input` (inner_loop=1) and
pick `tile_m × herd_m ≥ M / 127`. Example: M=16384, `tile_m=16,
m_input=16, herd_m=8` → `16384/(16*8) × 1 = 128` ✓.

**D2. Other decode CPU→NPU promotion.** If the decode pipeline still
falls back to CPU for any small op (e.g., residual add wrapped in
`np.add`, decode attention if not yet on NPU), promote it using the
standalone harness Phase 1 already validated.

### Step 3: Re-run Phase 3 gate after each pattern

After every applied pattern, re-run the gate:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify      # GATE: token-set, exit 1 on FAIL
```

`make verify` PASS is the correctness gate (its 32-token generation is
what catches decode-only bugs — KV cache, per-token static-BO reuse). If
it regresses to FAIL, revert the pattern and document why. diagnosis only
probes prefill, so it cannot localize a decode regression — bisect by
reverting patterns instead.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Multi-launch merge compile fails (BD exhaustion, channel routing, herd shape conflict) | BD/placeability limit or wrong stitching boundary | Invoke `debug-multi-launch-merge` |
| Extern kernel rename collision (link error, symbol redefined) | Two `.o` files exporting same symbol | Check `-D` mapping uniqueness; each `.o` must export distinct `<group>_matvec_*` names |
| `'aiex.npu.push_queue' op Repeat count exceeds [0:255]` (Pattern A or D) | GEMV K > 8160, or combined channel reads > 255 (see `details/GEMV_bf16.md`) | For K > 8160 → set `k_split` / `down_k_split`; for large M → set `tile_m == m_input` and grow `tile_m × herd_m` |
| `L2 capacity exceeded` (matvec.py builder assert) | GEMV staged buffer `K × herd_m × tile_m × 2 > 512 KiB` (see `details/GEMV_bf16.md`) | Reduce `tile_m` (e.g., 8 → 2 for K=8192) |
| Output corruption after static weight BO conversion (correct first call, NaN/garbage on subsequent) | Per-layer BO key collision OR `static_input_indices` wrong | Invoke `debug-bo-corruption` |
| LM Head GEMV NaN / argmax differs from CPU | Partition boundary off-by-one OR vocab shape padding mismatch | Check partition count divides vocab evenly; print partition outputs and concatenate manually to compare |
| Cosine drops after Pattern X | Pattern X assumption violated by this model | Revert Pattern X; check assumption (e.g., decode already seq-first, weights already pre-transposed) |
| ms/token unchanged after Pattern A | Per-call XRT overhead dominates; fusion alone insufficient | Pattern B (static weight BOs) is likely the missing piece — apply it next |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 5 PASS:

- `<model>/docs/development_progress/phase5_decode.md`: per-pattern
  table with `applied / skipped / reverted`, latency delta, reason
- `<model>/TODO.md`: mark Phase 5, append final ms/token + speedup
  vs Phase 4 baseline
- If a new fused decode ELF was built (kernel-first path of Pattern A),
  surface to Phase 6 for potential promotion to a shared location if a
  second deployment validates the same pattern
