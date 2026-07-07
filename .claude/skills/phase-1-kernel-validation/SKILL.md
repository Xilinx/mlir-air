---
name: phase-1-kernel-validation
description: Phase 1 of LLM deployment — for every leaf kernel × shape the model needs, verify numerical correctness on real NPU2 against the registry's GPU/vLLM-aligned standard. Primary gate where a standalone harness exists: the harness's full-output element-wise `np.isclose` check at that kernel's `rtol`/`atol` vs an FP32 reference (the same PASS/FAIL `make run` prints). Fallback for a kernel with no harness: `make diagnosis` per-layer cosine vs the HF bf16 reference. Record each verified (kernel, shape) as a row in that kernel's "tested shapes" table in `kernel_registry/supported_kernels.md` + `details/<Kernel>_bf16.md` (Used by = `<model>`); per-model progress stays in `<model>/docs/`. Hard gate before integration. Invoked by deploy-new-llm after Phase 0.
---

## Purpose

Enumerate every (kernel, shape) the model needs and verify each is
numerically correct on real NPU2 before integrating them into a block.
Phase 1 isolates per-kernel correctness from integration bugs and extends
the shared kernel registry with the shapes this model exercises, so Phase 2+
(and future deployments) can rely on a verified shape catalog.

The core loop is small:

1. Derive the kernel × shape list from the model's config (worked example:
   the llama-3.2-1B rows in `kernel_registry/supported_kernels.md`).
2. For each (kernel, shape) → obtain a correctness verdict (harness
   atol/rtol PASS, or diagnosis cosine) → record verdict + `mean_rel_L1`
   + perf.
3. For each **new** (kernel, shape), append a row to that kernel's "tested
   shapes" table in `kernel_registry/supported_kernels.md` and its
   `details/<Kernel>_bf16.md` (Used by = `<model>`); record the full
   per-kernel results under `<model>/docs/`.

**Two sources of the per-kernel correctness verdict** (use whichever exists):

- **Standalone harness (primary)** — a self-contained `make run` that
  compiles ONE kernel/block to an ELF and checks its **full output
  element-wise against an FP32 reference**: `np.isclose(|out−ref| ≤ atol +
  rtol·|ref|)`, every element must pass. The harness prints
  `[precision] mean_rel_L1=… rtol=… atol=… PASS!/failed.` — that PASS/FAIL
  IS the gate. This is the **GPU/vLLM-aligned standard** (`rtol=1.6e-2` is
  PyTorch/vLLM's canonical bf16 tolerance; `atol` is sized per kernel to the
  measured datapath error). The registry's `details/<Kernel>_bf16.md`
  §Tolerances documents each kernel's exact `rtol`/`atol`. Upstream ships a
  harness for the FFN block (`programming_examples/llms/llama_kernel_builder/ffn_swiglu/`) plus
  the top-level kernel examples `matrix_multiplication/bf16_in_bf16_out`
  and `matrix_multiplication/bf16_in_fp32_out` (the BF16 GEMM, split by
  output dtype — the legacy `matrix_multiplication/bf16` is kept for NPU1),
  `matrix_vector_multiplication/bf16`, `flash_attention/kernel_fusion_based`,
  `eltwise_add`, `weighted_rms_norm`, `rms_norm`, `rope_lut`.
- **In-context diagnosis (fallback)** — for a kernel with **no standalone
  harness**, there is no isolated output to run element-wise atol/rtol on,
  so fall back to `make diagnosis`: per-layer `ffn_out` **cosine vs the HF
  bf16 reference**. The per-layer cosine reflects every kernel in that
  layer. Cosine is the fallback lens only — not the gate for any kernel that
  has a harness.

## Phase 1 PASS criteria (HARD GATES)

Every (kernel, shape) the model needs must satisfy all four. Each catches
a different bug class:

1. **A correctness signal exists for the kernel × shape**: either a
   standalone harness `make run` (preferred — an isolated leaf check at the
   registry's atol/rtol) OR the kernel is covered by `make diagnosis`
   per-layer cosine. A kernel with neither is a silently-unverified
   kernel — not allowed.

2. **Numerical correctness** (the **real correctness gate** — not
   theoretical compile-time rules):
   - **Harness-backed kernel** → the harness's full-output element-wise
     `np.isclose` check PASSES at that kernel's `rtol`/`atol`
     (`details/<Kernel>_bf16.md` §Tolerances) vs the FP32 reference. This is
     the GPU/vLLM-aligned standard and the gate for every kernel that has a
     harness.
   - **No-harness kernel** → fall back to the `make diagnosis` per-layer
     cosine vs HF bf16 staying healthy (no cliff at the layer exercising
     this kernel).
   Catches: silent-corruption tile configs (e.g. GEMM
   `N % (tile_n × herd_n) != 0`, which the builder does NOT assert). Trust
   the measured atol/rtol verdict, not the rule.

   Record `mean_rel_L1` (the headline metric) plus `max_abs` / `max_rel`
   when the harness prints them. `mean_rel_L1` is the registry's per-shape
   accuracy column and a cheap regression baseline for future deployments;
   the gate itself is the harness's pass/fail at `rtol`/`atol`.

3. **Tile utilization documented**: each (kernel, shape) records its herd
   config. Targets:
   - Compute-bound (GEMM, FA): full 8×4 = 32 tiles
   - Row-parallel (RMSNorm, RoPE, GEMV-decode, SiLU+Mul, Eltwise): 8×1 = 8 tiles
   When achieved < target, justify in the catalog's Notes column (e.g.
   "M=1 decode RMSNorm uses 1 tile because batch=1 has no row-parallelism").
   Catches: silent under-utilization.

4. **Registry row written**: every verified (kernel, shape) that is new to
   the registry is appended to that kernel's "tested shapes" table in
   `programming_examples/kernel_registry/supported_kernels.md` and its
   `details/<Kernel>_bf16.md`, with Used by = `<model>`, matching the
   existing rows' column schema. The full per-kernel results
   (`mean_rel_L1` + harness PASS or diagnosis cosine, max_abs/max_rel,
   perf, tile config) also go to `<model>/docs/`.
   Catches: deployments that pass without leaving a reusable record.

Failure on ANY criterion blocks Phase 2.

## Knowledge base references

PRIMARY (read before starting):

- `programming_examples/kernel_registry/supported_kernels.md`
  — index of every supported leaf kernel + its "tested shapes" table
  (shape, tile config, perf, `mean_rel_L1`, **Used by**, status) across
  deployments. This is the menu of known-good shapes to copy from and the
  table you extend.
- `programming_examples/kernel_registry/details/<Kernel>_bf16.md`
  — per-kernel detail: the numerical datapath, **Tunable parameters**
  (knobs + hard constraints + tradeoffs), tolerances, per-shape data, and
  the reproduce commands (which harness, how to run). Read the page for
  each kernel your model needs. **GEMM is the exception**: it is split by
  output dtype into `details/GEMM_bf16_in_bf16_out.md` and
  `details/GEMM_bf16_in_fp32_out.md` (BF16-out has a `--high-precision`
  tier — fused-cast / drain = FP32-accumulate + single cast, GPU-standard
  ~9.3e-3; F32-out always FP32-accumulates).
- `programming_examples/kernel_registry/registry_lookup.py` — the
  **machine-readable** half of the registry. `gemm_config(M,K,N,
  output_dtype, precision)` returns the registry's best measured
  `{method, tile, gflops, mean_rel_L1}` for a shape from the companion
  `details/*.json`, and **raises (no silent guess) for an unmeasured
  shape** — so Phase 1 recording a new GEMM shape is what unlocks the
  programmatic lookup that Phase 4 builders consume. The `.md` tables
  mirror these `.json` files for humans.

## Workflow

### Step 1: Derive the model's shape list

Read the HF `config.json` (or the model's `<model>_weights.py:Config`
dataclass after Phase 0). Map each kernel call site to its shape using
standard transformer identities (Q proj output dim = `n_heads * head_dim`,
etc.). The llama-3.2-1B rows in `kernel_registry/supported_kernels.md`
(and `programming_examples/llms/llama32_1b/`'s prefill/decode call sites) are the worked
example of that call-site → shape mapping.

Write the working shape list — one row per (kernel, shape) with
`mean_rel_L1` + verdict + perf + status columns **empty** (Step 2 fills
them) — under `<model>/docs/`
(e.g. `<model>/docs/development_progress/phase1_kernels.md`).
This is scratch tracking for the deployment, not a registry file; the
registry rows get written in Step 3.

Before running anything, read each needed kernel's `details/<Kernel>_bf16.md`
**Tunable parameters / constraints** section and check the model's dims
against it (alignment, max-K, placeability notes) — flag likely walls
BEFORE compile time.

(If the model has unusual ops — Q/K Norm, post-norm, ops between
currently-fused launches — flag in `<model>/TODO.md` as a Phase 2
prerequisite. The actual integration decision lives in
`phase-2-single-block-validation` Step 0a.)

### Step 2: Per-kernel verification

For each (kernel, shape) in your Step 1 shape list:

**a. Pick the correctness source + initial tile config.**
Each kernel's `details/<Kernel>_bf16.md` has its reproduce commands (which
harness, how to run) and a Tunable parameters table (knobs + hard
constraints + tradeoffs). If your shape exists in that kernel's "tested
shapes" table in `supported_kernels.md` → reuse the tile config. Else →
mirror the nearest-shape entry; verify the hard constraints hold for your
shape; adjust if not (e.g., GEMM `N % (tile_n × herd_n) != 0` → pick
smaller `tile_n`).

**b. Run correctness (+ profile where a harness exists).**

If a standalone harness or top-level example covers the shape:

```bash
cd programming_examples/<harness_or_example_dir>
flock -x -w 1800 /tmp/mlir-air-npu.lock make run       # element-wise atol/rtol vs FP32 ref → PASS!/failed.
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile   # timing
```

If no standalone harness exists for the kernel, use the deployment's
per-layer diagnosis:

```bash
cd programming_examples/llms/<model>
flock -x -w 1800 /tmp/mlir-air-npu.lock make diagnosis  # per-layer ffn_out cosine vs HF bf16
```

NPU is shared on this machine — every NPU command must be `flock`-wrapped
(see project memory). Compile-only steps don't need the lock.

If the harness reports `failed.` (or, for a no-harness kernel, the
diagnosis cosine drops) → see "Failure modes" below. Bound: 1 retry per
recipe per shape; if still failing, escalate to TODO.md "Active blockers".

**c. Record results in `<model>/docs/`.** Fill the scratch row:
`mean_rel_L1` (+ harness PASS / diagnosis cosine), profile (ms / GFLOPS)
where measured, tile config used, tiles in flight, status, source (harness
vs diagnosis), and Notes (especially when tiles-in-flight is below
target — justify why).

### Step 3: Extend the kernel registry

For each (kernel, shape) verified in Step 2 that's **new** (not already
in that kernel's "tested shapes" table in `supported_kernels.md`), append
a row to both `supported_kernels.md` and the kernel's
`details/<Kernel>_bf16.md` per-shape table: shape + tile config +
tiles-in-flight, "Used by" listing your new model, `mean_rel_L1` +
profile from Step 2, status. Match the existing rows' column schema for
that kernel exactly.

This grows the registry organically: each verified deployment extends the
menu of known-good shapes future deployments can copy from. (Adding a
*new kernel* — not just a new shape of an existing one — is the heavier
`add-kernel` workflow with its own parity checklist; Phase 1 only adds
shapes to kernels the registry already covers.)

## Failure modes

When a test fails (harness `failed.` / diagnosis cosine drop, compile
error, hang), match the symptom to a likely cause. These are debug
starting points, not gates — trust the measured atol/rtol verdict, not the
rule. The relevant kernel's `details/<Kernel>_bf16.md` (constraints /
placeability section) is the authority for that kernel's hard limits.

| Symptom | Likely cause | Where to look |
|---|---|---|
| `'aiex.npu.push_queue' op Repeat count exceeds [0:255]` | GEMV K too large; auto-split outer dim ≥ 256 | Set `k_split` so `K = k_split × inner` and `k_split ≤ 255`; see `details/GEMV_bf16.md` |
| `Allocator exhausted available buffer descriptor IDs` | BD pool exhausted (non-aligned dim, or non-monotonic placeability of the iteration count) | Pad dim to 1024-aligned (GQA-aware reindexed padding) OR use kernel-first split-ELF path; see the placeability notes in the kernel's `details/` page |
| `L2 capacity exceeded` (matvec builder assert) | GEMV staged buffer > 512 KiB | Reduce `tile_m` (e.g., 8 → 2 for K=8192) or `herd_m`; see `details/GEMV_bf16.md` |
| Output all-zero / cosine = NaN | Bare-herd kernel without launch+segment wrapper | Wrap the bare herd in `air.launch`/`air.segment` (see ffn_swiglu harness for the multi-launch pattern) |
| Cosine = 0.02 or other small | GEMM `N % (tile_n × herd_n) != 0` silent corruption (builder does NOT assert) | Pick `tile_n` so divisibility holds at this N |
| FA all-NaN at runtime | Compile-flag mismatch on `attn_npu2.cc` macros; OR seq-first `dk_chunks>1` path at head_dim≥128 | see `debug-fa-runtime-failure` skill |
| Compile hangs > 10 min | Compiler scaling issue at large multi-launch | Cap and document; don't retry |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 1 PASS:

- Mark Phase 1 in `<model>/TODO.md`, append "(N/N kernels PASSED)"
- Append summary to `<model>/docs/development_progress/progress.md`
  (per-kernel `mean_rel_L1` + verdict, total time) — the per-model durable record
- `kernel_registry/supported_kernels.md` + `details/<Kernel>_bf16.md`
  carry a "tested shapes" row (Used by = `<model>`) for every new
  (kernel, shape) this model exercises — the cross-model durable record
