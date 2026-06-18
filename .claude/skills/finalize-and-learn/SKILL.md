---
name: finalize-and-learn
description: Phase 6 of LLM deployment — integrate Phase 4 prefill + Phase 5 decode into a clean `<model>_inference.py`, write the model's `verify_adapter.py` hooking into the shared `llms/verify/` subsystem + a Makefile (run / verify / verify-full / diagnosis / profile), and confirm `make verify` (top-k token-set gate vs HF bf16) PASSES. That gate is the production-readiness check. Capture lessons learned. Invoked after Phase 5 PASS.
---

## Purpose

Phase 4-5 produced an optimized prefill kernel and an optimized decode
kernel — but they may live in separate scripts with optimization-time
warmup hacks scattered through the main flow. Phase 6 integrates them
into a single clean `<model>_inference.py` and wires the deployment into
the **shared** `llms/verify/` subsystem via a per-model `verify_adapter.py`
(mirroring `llama32_1b/verify_adapter.py`) so the deployment has:

1. A clean **setup → prefill → decode** structure with no warmup hacks
   in the profiled scope (preprocess / weight pre-load happens ONCE in
   `setup()` BEFORE the timed region).
2. A `Makefile` with the same targets the reference deployment and the
   Phase 7 evaluator rely on:
   - `make run` — runs inference, prints **TTFT** (prefill ms) + **TPS** (tokens/sec)
   - `make verify` — top-k token-set gate vs HF bf16 (2 prompts × 32 tokens, k=5) — the production-readiness gate
   - `make verify-full` — same gate over the full prompt set
   - `make diagnosis` — per-layer `ffn_out` cosine vs HF bf16 (informational)
   - `make profile` — per-phase + per-key-kernel breakdown
3. New experience captured in `LESSONS.md` for future deployments.

**Why the token-set gate** (not a hand-written CPU greedy match): the
reference is HF transformers in bf16, and the gate (`compute_topk_set_check`
in `verify/comparators.py`) checks that at the first divergence between
NPU and HF greedy sequences, each side's chosen token is in the other's
top-5. This catches decode-side KV-cache bugs that prefill-only checks
miss, while tolerating benign bf16 top-1 flips within the top-5 band. It
runs the **production** prefill/decode path, so it exercises the real
deployment, not a separate verify-only code path.

This top-k token-level inclusion check **mirrors vLLM's correctness
methodology** — it is the GPU/industry-standard end-to-end signal, which
is why it (not a per-tensor cosine) is the production-readiness gate. The
whole `verify/` subsystem is **shared across all models** under
`llms/verify/`; a new deployment hooks in with a thin `verify_adapter.py`
rather than copying the runner, so every model is judged by the identical
gate. See `llms/verify/README.md`.

## Phase 6 PASS criteria (HARD GATES)

1. **`<model>_inference.py` exists** with the clean structure: `setup()`
   (one-time preprocess, weight pre-load, BO allocation) called ONCE
   before the profiled `prefill() + decode_loop()` region. No warmup
   hacks, cache prime calls, or timing resets in the main flow.
2. **`<model>/verify_adapter.py` exists**, mirroring
   `llama32_1b/verify_adapter.py`: it provides the adapter interface the
   shared `llms/verify/verify_runner.py` calls — `resolve_model`,
   `hf_reference`, `build_config`, `build_runner`, and an `NpuRunner`
   (implementing `.prefill()` / `.decode_step()`) that calls THIS model's
   production `run_npu_prefill` / `run_npu_decode_step`. The verify runner,
   comparators, report, and HF runner are NOT copied — they live once in
   `llms/verify/`.
3. **`make run` works**: invokes inference at default `--n-tokens 100`,
   prints TTFT (prefill kernel ms) + TPS (tokens/sec).
4. **`make verify` PASSES** the token-set gate (production-readiness):
   - NPU and HF bf16 each greedy-decode the prompt set; at the first
     divergence, NPU's chosen token ∈ HF top-5 AND HF's chosen token ∈
     NPU top-5 (`compute_topk_set_check`, GATE_K=5, GATE_N_TOKENS=32).
   - `report.has_failure()` is False → exit 0.
5. **`make diagnosis` runs without error** (informational, not a gate —
   the verify subsystem retired threshold-based diagnosis; `compare_pair`
   reports per-layer cosine with no pass/fail). Eyeball the per-layer
   cosine table against the Phase 3 baseline to confirm the finalized
   integration didn't perturb the numerical alignment; if criterion 4
   (`make verify`) FAILs, this table is the localization lens. The gate is
   criterion 4, not this.
6. **`make profile` works**: outputs per-phase total (setup / prefill
   total / per-token decode avg / LM head) + key-kernel ms (FA, Down
   GEMM/GEMV, LM Head — the bottleneck candidates).
7. **`LESSONS.md` updated** with any new experiences (informational; not
   gating the technical artifacts above).

If `make verify` fails, the deployment is NOT production-ready regardless
of how good Phase 4/5 perf numbers look.

## Knowledge base references

PRIMARY:

- `programming_examples/llms/llama32_1b/llama32_1b_inference.py` — reference
  clean inference structure (setup / prefill / decode_loop); copy from
- `programming_examples/llms/llama32_1b/verify_adapter.py` — reference
  adapter to mirror (the interface the shared verify runner calls)
- `programming_examples/llms/verify/` — the **shared** verify subsystem
  (runner + comparators + report + HF runner + prompts); hooked into, not copied
- `programming_examples/llms/llama32_1b/Makefile` — reference target set
  (run / verify / verify-full / diagnosis / profile / compile / clean)
- `programming_examples/llms/<model>/docs/development_progress/{phase4_prefill,phase5_decode}.md`
  — Phase 4/5 outputs: which integration path was used + the optimized
  prefill/decode runners to integrate

SECONDARY:

- `programming_examples/llms/verify/README.md` — verify methodology
- `programming_examples/kernel_registry/supported_kernels.md`
  — Phase 1's registry rows (Phase 6 confirms "Used by" reflects this model)

## Workflow

### Step 1: Integrate prefill + decode into `<model>_inference.py`

Copy `programming_examples/llms/llama32_1b/llama32_1b_inference.py` as the
starting point. The structure should be:

```python
def setup(weights, config):
    """ONE-TIME preprocess: pre-load weight BOs, allocate caches,
    install head-first FA wrapper if head_dim ≥ 128, etc. Everything
    that should NOT be inside the profiled scope."""
    ...

def run_npu_prefill(input_ids, ...):
    """Phase 4 prefill — clean of warmup hacks."""
    ...

def run_npu_decode_step(token, pos, ...):
    """Phase 5 single decode step — clean. Called by both the decode
    loop AND the NpuRunner in verify_adapter.py."""
    ...
```

Audit Phase 4/5 scripts for **warmup hacks** that crept into the main
flow — pre-warm cache calls, dummy runs, timing resets — and move them
into `setup()` (or delete if no longer needed). The profiled scope must
be ONLY `prefill + decode_loop`.

**Crucial**: `run_npu_prefill` / `run_npu_decode_step` are the SAME
functions the model's `verify_adapter.py` `NpuRunner` calls. The verify
gate exercises the production path precisely because it imports these — do
not fork a verify-only copy.

### Step 2: Write the model's `verify_adapter.py`

The verify runner/comparators/report/HF runner are **shared** in
`llms/verify/` — you do NOT copy them. You write one per-model file,
`<model>/verify_adapter.py`, mirroring `llama32_1b/verify_adapter.py`. It
provides the adapter interface the shared `verify_runner.py` loads via
`--runner=<model>.verify_adapter`:

- `resolve_model(choice)` / `hf_reference(name)` → map `--model`
  (base/instruct) to the HF checkpoint id.
- `build_config()` → return THIS model's Config.
- `build_runner(...)` → load weights, compile kernels, return the runner.
- `class NpuRunner` → `.prefill()` / `.decode_step()` calling THIS model's
  production `run_npu_prefill` / `run_npu_decode_step`.

The HF reference path, the comparators (`compute_topk_set_check`), and the
report all stay in `llms/verify/` — unchanged, model-agnostic. If the
default chat template differs, point `resolve_model` / the prompt choice
at the right `llms/verify/prompts/*.txt`.

### Step 3: Wire the Makefile

Mirror `programming_examples/llms/llama32_1b/Makefile` — note the verify
runner is the **shared** one at `../verify/`, selected via `--runner`:

```makefile
RUNNER_ADAPTER := <model>.verify_adapter
VERIFY_RUNNER  := $(srcdir)/../verify/verify_runner.py

run:
	flock -x -w 1800 /tmp/mlir-air-npu.lock \
		bash -c 'cd $(BUILD_DIR) && python3 $(srcdir)/<model>_inference.py --n-tokens 100'

verify:
	flock -x -w 1800 /tmp/mlir-air-npu.lock \
		bash -c 'cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) --prompts topk_token --model $(MODEL) --max-prompts 2'

verify-full:
	flock -x -w 1800 /tmp/mlir-air-npu.lock \
		bash -c 'cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) --prompts topk_token --model $(MODEL)'

diagnosis:
	flock -x -w 1800 /tmp/mlir-air-npu.lock \
		bash -c 'cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) --prompts single --model $(MODEL)'

profile:
	flock -x -w 1800 /tmp/mlir-air-npu.lock \
		bash -c 'cd $(BUILD_DIR) && python3 $(srcdir)/<model>_inference.py --profile --n-tokens 20'
```

`MODEL` defaults to `instruct` (matches what production stacks deploy);
`MODEL=base` selects the base prompt set.

### Step 4: Run `make verify` — the production-readiness gate

```bash
cd programming_examples/llms/<model>
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
```

The runner: both NPU and HF bf16 greedy-decode each prompt × 32 tokens,
then `compute_topk_set_check` compares the two sequences. PASS = no
`npu_vs_hf` record is FAIL → exit 0; the report under `verify/reports/`
records the first divergence and the top-5 sets on each side.

If it FAILS at token i ≥ 1 (but `make diagnosis` per-layer cosine on
prefill is fine), the divergence is in the decode path / KV-cache — print
K/V cache values after token i-1 and compare to HF.

### Step 5: Run `make run`, capture TTFT + TPS

Record final numbers in `<model>/docs/development_progress/phase6_finalize.md`:

| Metric | Value | vs reference llama32_1b |
|---|---|---|
| TTFT (prefill kernel ms) | X | Y× / Y% |
| TPS (tokens/sec) | A | B× / B% |
| Decode ms/token | T | — |

### Step 6: Run `make profile`, capture breakdown

`--profile` mode prints per-phase totals (setup / prefill / per-token
decode / LM head) + key-kernel ms (FA, Down GEMM/GEMV, LM Head — the
bottleneck candidates). Record in `phase6_finalize.md`.

### Step 7: Update LESSONS.md + flag promotion candidates

Append to `<model>/docs/development_progress/LESSONS.md` for any new
experience (debug techniques used, surprising failures, configs that
mattered).

Then audit for promotion candidates (don't promote speculatively — only
if 2+ uses):

1. Did this deployment **add a new C++ kernel** or a new fused
   multi-launch ELF under `<model>/multi_launch_builder/`
   (kernel-first path)? Cross-reference other deployments — if a 2nd
   uses the same pattern, it's a candidate for a future shared location.
2. Did this deployment hit a **new per-kernel constraint** or compiler
   quirk (placeability, alignment, max-K) worth recording in that
   kernel's `kernel_registry/details/<Kernel>_bf16.md`?
3. Did this deployment surface a **new skill-chain change**? Edit the
   relevant `.claude/skills/<phase>/SKILL.md` directly (git history is
   the change trail). Surface in `<model>/TODO.md` if not done inline.

### Step 8: Confirm kernel registry reflects this deployment

Sanity check that Phase 1's registry step completed:

- `kernel_registry/supported_kernels.md` (and each `details/<Kernel>_bf16.md`)
  has a "tested shapes" row with Used by = `<model>` for every new
  (kernel, shape) this model exercises
- `<model>/docs/development_progress/` has the full per-kernel results
  (cosine, max_abs/max_rel, profile, status) backing those rows

If gaps, fix here before Phase 7.

## Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| `make verify` FAILS at token i ≥ 1 but `make diagnosis` prefill cosine is fine | KV cache update bug at decode time (diagnosis only probes prefill) | Print K/V cache values after token i-1 vs HF; usually a layout / write-offset bug in the decode kernel |
| `make verify` FAILS at token 0 | Prefill-side issue (LM Head precision, final norm) | Re-run `make diagnosis`; root cause is in prefill, not decode |
| `make verify` errors importing the NPU runner | `verify_adapter.py`'s `NpuRunner` not wired to THIS model's prefill/decode functions | Confirm the import points at `<model>_inference.py`, not the llama32_1b copy |
| TTFT regressed vs Phase 4 baseline | Integration introduced overhead (warmup hack creep, redundant setup in main flow) | Compare Phase 4 standalone profile to current `make profile` setup section |
| TPS regressed vs Phase 5 baseline | Same as above for decode | Compare Phase 5 standalone profile |
| `make profile` shows huge "Setup" time inside profiled scope | `setup()` called inside the timed region instead of once before | Refactor — `inference()` must call `setup()` BEFORE `t0 = time.time()` |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

On Phase 6 PASS:

- `<model>/docs/development_progress/phase6_finalize.md`: TTFT + TPS +
  profile breakdown + LESSONS summary
- `<model>/TODO.md`: mark Phase 6 PASSED
- `<model>/ARCHITECTURE.md`: write or update with final summary (model
  config, key file map, perf headline). NOTE: use `ARCHITECTURE.md`, not
  `CLAUDE.md` — the top-level `.gitignore` excludes `CLAUDE.md`, so it
  would not ship in the PR.
- **Hand off to Phase 7**: deploy-new-llm orchestrator now spawns
  `independent-evaluator` to re-derive every claim from scratch.
  Phase 6 is "deployment is internally complete"; Phase 7 is "deployment
  is independently audited".
