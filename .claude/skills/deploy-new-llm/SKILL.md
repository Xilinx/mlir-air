---
name: deploy-new-llm
description: Entry point for deploying a new decoder-only LLM on AMD NPU2. Invoked by the user as `/deploy-new-llm <hf_model_id> [--name <dirname>] [--target npu2|npu1] [--dtype bf16|fp16]`. Bootstraps the per-model workspace, validates architecture is in scope, and dispatches the 7 per-phase skills with the gate of each phase enforced by that phase's skill.
---

## Purpose

Single user-facing entry point that scaffolds a new model deployment
and orchestrates the per-phase skills. The orchestrator does NOT do
correctness work itself — every gate is enforced by the corresponding
per-phase skill. This skill's job is workflow coordination + workspace
bootstrap.

## Orchestrator success criteria

This skill is "successful" when:

1. **Workspace scaffolded** correctly (Steps 1-6 below complete)
2. **Phases 0-6 dispatched in order**, each phase's HARD gate (defined
   inside the per-phase SKILL.md) passes
3. **Phase 7 (`independent-evaluator`) verdict** = PASS or
   PASS-with-warnings
4. **Hand-off report written** to the human (Step 9)

If any phase's gate fails irrecoverably, the deployment is marked
`needs-human-review` in TODO.md and the orchestrator stops — the human
triages.

## Knowledge base references

- `programming_examples/llms/llama32_1b/` — the reference Tier-A deployment
  (everything in scope today inherits from this)
- `programming_examples/llms/verify/` — the **shared** verify subsystem
  (HF bf16 reference; `make verify` token-set check is the PASS/FAIL gate,
  `make diagnosis` per-layer cosine is the informational lens); every model
  hooks in via its own `verify_adapter.py`, not by copying this.
- `programming_examples/llms/llama_kernel_builder/` — the **shared** kernel
  builder (KernelCache, external kernels, stitching, the `ffn_swiglu/`
  harness); per-model scripts import from it.
- `programming_examples/kernel_registry/` — the model-agnostic kernel
  registry. Human half: `supported_kernels.md` (index) + `details/<Kernel>_bf16.md`
  per kernel (GEMM is split by output dtype: `GEMM_bf16_in_bf16_out.md` +
  `GEMM_bf16_in_fp32_out.md`). Machine half: `details/*.json` +
  `registry_lookup.py` (`gemm_config(...)` returns the best measured tile
  config; raises for unmeasured shapes). Phase 1 appends this model's
  verified (kernel, shape) rows (Used by = `<model>`); Phase 4 builders read
  tile configs back via the lookup instead of hardcoding them.
- `docs/superpowers/specs/2026-04-17-llm-mapping-skills-design.md`
  — original design spec for the skill chain

## Workflow

### Step 0: Preconditions check

This skill assumes the user already has:

- **mlir-air built and the environment sourced** (see the repo build
  docs; for manual shells, source the mlir-air env before invoking).
  Smoke test: `cd programming_examples/llms/llama32_1b && make help` prints
  the target list.
- **NPU2 hardware accessible via XRT** (no other process holding it).
- **HuggingFace login + model access**. For gated models like
  `meta-llama/Llama-3.2-3B`, run `huggingface-cli login` and accept the
  model card on huggingface.co before invoking this skill. ~6 GB disk
  per BF16 3B model in `~/.cache/huggingface/hub/`.
- **System DRAM ≥ 16 GB** for 1-3 B models; deeper deployments
  approach the limit.

If any are missing, halt and ask the user to address them. Do NOT try
to install MLIR-AIR / set up XRT / log into HF on the user's behalf.

### Step 1: Parse arguments

- Required: HF model ID (e.g., `meta-llama/Llama-3.2-3B`)
- Optional: `--name <dirname>` (default: derived from model ID,
  lowercased, slashes → underscores)
- Optional: `--target npu2|npu1` (default: `npu2`)
- Optional: `--dtype bf16|fp16` (default: `bf16`)

### Step 2: Architecture compatibility check

Fetch HF `config.json`. Reject if any of:

- Architecture is MoE (e.g., `MixtralForCausalLM`, gpt-oss class)
- Has sliding-window attention (`sliding_window` set in config AND
  `use_sliding_window=true`)
- Uses MLA (Multi-head Latent Attention)
- Uses encoder-decoder structure

**QKV bias is supported** (Qwen2-family models with `qkv_bias=true`):
the bias is added on the HOST after the bias-free kernels return,
exploiting RoPE's linearity (`RoPE(q + bq) = RoPE(q) + RoPE(bq)`). The
bias-on-host wrapper is re-derived per deployment (technique described
in `single-block-validation` Step 2). Per-deployment effort: ~1-2 hours;
surface in TODO.md as a Phase 2 prerequisite.

If rejected, print clear message and do NOT proceed.

### Step 3: Check for the shared infra + reference exemplar

```bash
test -d programming_examples/llms/llama_kernel_builder && \
test -d programming_examples/llms/verify && \
test -d programming_examples/llms/llama32_1b && echo OK || echo MISSING
```

The first two are **required**: every deployment composes kernels via the
shared `llama_kernel_builder` toolkit and gates on the shared `verify/`
subsystem. The third, `llama32_1b`, is the **reference exemplar** — read
to mirror its assembly, and imported directly on a bit-for-bit match. If
any is missing, halt and instruct the human.

### Step 4: Scaffold `<model>/` directory — kernel-first, minimal

The model lives at `programming_examples/llms/<dirname>/`, a sibling of
`llms/llama32_1b/` (the reference exemplar) and the shared `llms/verify/`
+ `llms/llama_kernel_builder/` (the toolkit every deployment builds on).

**Default mindset: build this model up from registry kernels** using the
shared `llama_kernel_builder` toolkit (KernelCache, stitching,
external_kernels). The per-phase skills write the model's own
`<model>_prefill.py` / `<model>_decode.py` / `multi_launch_builder/` by
composing the Phase-1-verified leaf kernels, reading `llama32_1b`'s
assembly as the worked exemplar. This generalizes to any in-scope
architecture — it does not assume the model resembles llama.

**Do NOT `cp -r llama32_1b <model>`.** Two reasons depending on path
(the kernel-first-vs-inheritance decision + the bit-for-bit match rule are
owned by `single-block-validation` Step 1 — Phase 2 makes the call):
- Kernel-first (default): you're writing model-specific assembly, not
  copying the reference's — bulk-copying just duplicates stale code.
- Inheritance shortcut (bit-for-bit llama variant only): the reference's
  `llama32_1b_*.py` resolve via sys.path to `../llama32_1b/`, so there's
  nothing to copy; a local copy would silently use outdated logic and miss
  upstream bug fixes.

The minimal Tier-A scaffold is:

```
llms/<dirname>/
├── .gitignore                       # copy from llama32_1b/.gitignore + add *.o, *kernel_cache/
├── Makefile                         # template-render with model name (run / verify / verify-full / diagnosis / profile + compile / clean)
├── README.md                        # placeholder; final version written by finalize-and-learn
├── ARCHITECTURE.md                  # model-specific guide (NOT CLAUDE.md — top-level .gitignore excludes it, so it would not ship)
├── TODO.md                          # phase status (template in Step 5)
├── verify_adapter.py                # written by finalize-and-learn; hooks this model into the shared llms/verify/
└── docs/development_progress/
    ├── progress.md                  # header-only; phases append as they pass
    ├── LESSONS.md                   # header-only; appended on novel failures
    └── debug_log.md                 # header-only; appended on debug-recipe firings
```

**Per-model scripts use this sys.path block** to resolve the shared
`llms/` packages (always) and the llama32_1b reference (as exemplar, or
to import directly on a bit-for-bit match):

```python
from pathlib import Path
import sys
_THIS_DIR = Path(__file__).resolve().parent
_LLMS_DIR = _THIS_DIR.parent              # programming_examples/llms/
for p in (_LLMS_DIR, _LLMS_DIR / "llama32_1b", _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ALWAYS — the shared kernel toolkit you compose this model FROM:
from llama_kernel_builder.external_kernels import compile_all_external_kernels
from llama_kernel_builder.cache import KernelCache

# DEFAULT (kernel-first) — write <model>_prefill.py / multi_launch_builder/
# that assemble the registry leaf kernels for THIS model's sequence,
# mirroring llama32_1b's builders as the worked example.

# SHORTCUT (bit-for-bit llama variant ONLY) — skip writing builders and
# reuse the reference's assembly directly:
#   from llama32_1b_prefill import run_transformer_block, ...
#   from llama32_1b_decode import run_decode_block, compile_decode_kernels
```

For models that fork one reference module (partial arch divergence), copy
ONLY the file being forked, rename it `<model>_prefill.py`, compose the
divergent part kernel-first, and import the unchanged rest from
`../llama32_1b/`. Don't bulk-copy.

**Per-phase skills produce these model-specific files**:

- Phase 0 (`build-cpu-reference`): `<model>_weights.py`, `<model>_cpu_helpers.py`
- Phases 1-3 (validation): `<model>_phaseN_test.py` per phase
- Phase 6 (`finalize-and-learn`): `<model>_inference.py` (clean
  end-to-end NPU runner: setup → prefill → decode_loop) + `<model>/verify_adapter.py`
  (hooks this model into the shared `llms/verify/`; mirrors
  `llama32_1b/verify_adapter.py`)

**Makefile template** (mirror `programming_examples/llms/llama32_1b/Makefile`):

- `make compile` — compile all kernels
- `make run` — `<model>_inference.py --n-tokens 100`
- `make verify` / `make verify-full` — `../verify/verify_runner.py --runner=<model>.verify_adapter` token-set gate
- `make diagnosis` — `../verify/verify_runner.py --runner=<model>.verify_adapter` per-layer cosine
- `make profile` — `<model>_inference.py --profile`
- Env vars: `PROMPT`, `N_TOKENS`, `MODEL` (base/instruct) plumbed through
- `make clean` — remove `*kernel_cache/`, `air_project/`, `build_*/`, `*.o`, `verify/reports/`

### Step 5: Initialize `<model>/TODO.md`

Template (filled with Step 2 config):

```markdown
# Deployment: <model_name>

## Phase status
- [ ] 0: Build CPU Reference
- [ ] 1: Kernel Validation
- [ ] 2: Single-Block Validation
- [ ] 3: Full-Model Validation
- [ ] 4: Prefill Optimization
- [ ] 5: Decode Optimization
- [ ] 6: Finalize & Learn
- [ ] 7: Independent Evaluation

## Active blockers
(none yet)

## Resolved config (pulled from HF)
n_layers: <N>, emb_dim: <D>, n_heads: <H>, n_kv_heads: <K>,
head_dim: <hd>, hidden_dim: <F>, vocab_size: <V>, rope_theta: <R>
```

### Step 6: Initialize per-model docs

Create `<model>/docs/development_progress/`:

- `progress.md` (header only)
- `LESSONS.md` (header only)
- `debug_log.md` (header only)
- `phase_timing.md` (REQUIRED — per-phase wall-clock log; schema below)

**`phase_timing.md` schema** (per-phase effort breakdown — useful for
understanding which architectural axes are genuinely hard). Capture the
deployment-session start timestamp (`date +"%s"`) at scaffold time.
Update at every phase boundary:

```markdown
# <Model> deployment — per-phase wall-clock log

## Baselines

- Deployment session start: <YYYY-MM-DD HH:MM:SS TZ> (epoch=<N>)
- Scaffold complete:        <YYYY-MM-DD HH:MM:SS TZ> (epoch=<N>)

## Phase log

### Phase N — <Name>  (PENDING / PASS / PASS-with-warnings / BLOCKED, YYYY-MM-DD)

- start_ts:           <epoch s>  (HH:MM:SS TZ)
- end_ts:             <epoch s>  (HH:MM:SS TZ)
- wall_min:           **<N>**
- npu_compile_min:    <N>   (sum of NPU kernel compile times in this phase)
- npu_runtime_s:      <N>   (sum of XRTRunner / inference NPU run time)
- dev_min:            **<N>**   ≈ wall - compile - runtime (agent thinking/code/debug)
- notable_events:     <bullets — record honestly even if "stuck on debug for K min">

## Summary table  (filled at deployment end)

| Phase | wall_min | npu_compile_min | npu_runtime_s | dev_min | notes |
|---|---:|---:|---:|---:|---|
| Scaffold + Step 0-3 | | | | | |
| 0: CPU Oracle | | | | | |
| ... | | | | | |
| **Total** | | | | | |
```

**Why this matters**: `dev_min` (vs `npu_compile_min` / `npu_runtime_s`)
is the real "agentic deployment cost". Even debug-stuck phases should be
honestly recorded — high `dev_min` on a phase reveals which
architectural axes are genuinely hard, which informs future deployments.

### Step 7: Dispatch the 7 phases

**Phase → skill mapping** (each gate enforced by the per-phase skill):

| Phase | Skill | Gate (in 1 line — see the skill itself for full criteria) |
|---|---|---|
| 0 | `build-cpu-reference` | `<model>_weights.py` + `<model>_cpu_helpers.py` produced; HF bf16 baseline loads & runs canonical prompt via `verify/` HfRunner (sane top-1, no NaN); config matches HF `config.json` |
| 1 | `kernel-validation` | Every leaf kernel × shape: harness atol/rtol element-wise check vs FP32 ref PASSES (GPU/vLLM standard), or `make diagnosis` cosine vs HF bf16 for no-harness kernels; each new shape recorded as a `kernel_registry` row (Used by = `<model>`) + full results in `<model>/docs/` |
| 2 | `single-block-validation` | Single transformer block on NPU: per-layer cosine vs HF bf16 (diagnosis lens) ≥ 0.99 (whole-tensor) + per-position min ≥ head_dim-scaled threshold |
| 3 | `full-model-validation` | Full N layers: `make diagnosis` per-layer cos ≥ 0.85 + no cliff; `make verify` token-set gate (top-5 inclusion vs HF bf16) PASSES |
| 4 | `prefill-optimization` | Apply optimization patterns; correctness preserved (`make verify` token-set still PASSES — diagnosis cosine is the localization lens, not the gate) AND prefill kernel time strictly < Phase 3 baseline |
| 5 | `decode-optimization` | Same shape: correctness preserved (`make verify` token-set still PASSES) AND decode time/token strictly < Phase 4 baseline |
| 6 | `finalize-and-learn` | Clean `<model>_inference.py` + `<model>/verify_adapter.py` (shared `llms/verify/`) + Makefile; `make verify` (top-5 token-set vs HF bf16) PASSES |
| 7 | `independent-evaluator` | Fresh subagent: audit `make verify` (anti-reward-hacking) + re-run as primary gate; produce structured `evaluation_report.md` |

Report current state to the human:

> "Workspace scaffolded at `programming_examples/llms/<dirname>/`. Resolved
> config: <summary>. Ready to start Phase 0 (Build CPU Reference). Invoke
> `build-cpu-reference` to begin, or say 'go' for me to invoke it now."

For each phase (0 → 6):

1. **Capture phase start_ts** via `date +"%s : %Y-%m-%d %H:%M:%S %Z"`,
   record in `phase_timing.md` under that phase's `start_ts:` line
2. Invoke the per-phase skill from the table
3. Wait for the skill to complete or escalate
4. **Capture phase end_ts** the same way; compute `wall_min`,
   `npu_compile_min` (from skill output), `npu_runtime_s` (from skill
   output), `dev_min` (residual). Record in `phase_timing.md` under
   that phase's section. Even if the phase was stuck on debug for most
   of the time, record honestly — it reveals which axes are hard.
5. Report PASS/FAIL/BLOCKED to the human
6. On PASS: ask permission to advance to next phase
5. On BLOCKED: surface the blocker, human resolves
6. Advance to the next phase

### Step 8: Phase 7 — Independent evaluation

After Phase 6 PASSES but BEFORE the final hand-off, spawn the
`independent-evaluator` skill as Phase 7. It re-derives every
correctness claim with a fresh subagent and produces
`<model>/docs/evaluation_report.md`.

Why: Phases 0-6 are autonomous and self-reporting. The deployment
agent has no incentive to cheat, but also no incentive to catch its
own silent regressions (preload errors, fallback gates, etc.). Phase 7
is the independence check.

> "Spawning Phase 7 — independent evaluation. The evaluator subagent
> will audit `make verify` (anti-reward-hacking), re-run it as the
> primary gate, and write a structured report. Expected runtime:
> 15-30 min."

Then call the `independent-evaluator` skill with `<model_dir>` as input.

If the evaluator reports:

- **PASS** → proceed to Step 9
- **PASS-with-warnings** → proceed to Step 9; warnings go in TODO.md
  as "follow-up"
- **FAIL** → mark deployment `needs-human-review` in TODO.md and STOP.
  Do NOT hand off. Surface specific failures.

If the deployment touched shared infra (any of `kernel_registry/`,
`matrix_vector_multiplication/`, `llms/llama_kernel_builder/`,
`llms/verify/`, `llms/llama32_1b/multi_launch_builder/`,
`llms/llama32_1b/llama32_1b_*.py`), the evaluator's Step 7 (Conditional
Cross-deployment regression) will also re-verify every OTHER deployment's
`make verify` to catch back-compat breaks. Budget ~5 min per deployment.

### Step 9: On all-PASS, hand off to the human

Once Phase 6 PASS AND Phase 7 PASS (or PASS-with-warnings), report:

> "Deployment complete. See:
> - `programming_examples/llms/<dirname>/docs/development_progress/progress.md` — phase summary
> - `programming_examples/llms/<dirname>/docs/development_progress/phase_timing.md` — per-phase wall+dev time
> - `programming_examples/llms/<dirname>/docs/evaluation_report.md` — independent audit
> - `programming_examples/kernel_registry/supported_kernels.md` — the kernel × shape rows (Used by = `<dirname>`) this deployment added"

Optional: tag the deployment if the project workflow uses git tags
(`git tag -a deployment-<dirname>-v1 -m "..."`). Most deployments
don't tag — git log + commit messages are the durable record.

## Failure modes

| Symptom | Likely cause | What to do |
|---|---|---|
| Architecture rejected at Step 2 | MoE / sliding-window / MLA / encoder-decoder model | Halt; tell the user this model is out of scope |
| `llama32_1b/` missing at Step 3 | Reference deployment not present | Halt; instruct human (per-model scripts resolve imports against it) |
| Per-phase gate fails | Per-phase skill should escalate via TODO.md "Active blockers" | Don't try to fix here; the per-phase skill's failure-mode table is the right place |
| Phase 7 = FAIL | Evaluator surfaced a real correctness or reward-hacking issue | Mark `needs-human-review` in TODO.md; STOP; do NOT hand off |
| Cross-deployment regression at Phase 7 | This deployment's shared-infra change broke another deployment | Revert or fix the shared-infra change before tagging |
| User skips a phase to "save time" | Skipped phases mean later phases verify against unverified upstream | Refuse to advance past the skipped gate; explain the dependency chain (Phase 1 → 2 → 3 → 4/5 → 6 each need the previous PASS) |

For any failure not in the table, escalate to the human (this skill is
orchestration; debugging belongs in the per-phase skills' failure-mode
tables or the cross-cutting `debug-*` skills).

## Update protocol

This skill primarily reads `TODO.md` and dispatches; it doesn't write
to progress files itself (per-phase skills do that). On all-PASS:

- `<model>/TODO.md` reflects all 7 phases checked
- `<model>/docs/development_progress/progress.md` has each phase's
  summary entry (written by per-phase skills)
- `<model>/docs/development_progress/phase_timing.md` complete with
  Summary table filled (this skill writes it via Step 7 phase-boundary
  timestamp captures)
- `<model>/docs/evaluation_report.md` exists (written by Phase 7)
- (optional) git tag created
