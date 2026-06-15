---
name: independent-evaluator
description: Phase 7 of LLM deployment — spawn a fresh subagent that treats the deployment as UNTRUSTED, audits the `make verify` implementation (anti-reward-hacking: confirms the token-set gate runs the production path vs HF bf16), then re-runs it as the primary gate. Produces a structured `evaluation_report.md` a human can read in 2 minutes to know the full deployment state. Invoke as `/independent-evaluator <model_dir>` or auto-spawn from deploy-new-llm after Phase 6 PASS.
---

## Purpose

The deploy-new-llm chain is autonomous. Phase 6 wires up a `make verify`
gate that's supposed to compare NPU vs the HF bf16 reference via the
`verify/` subsystem's top-k token-set check. But the deployment agent
wrote (or copied) both the gate AND the production code it exercises —
nothing structurally prevents the gate from being mocked, pointed at a
stale baseline, or wired to a verify-only code path that bypasses the
real kernels. Phase 7 closes this by:

1. **Spawning a FRESH subagent** with no context from the deployment
2. **Re-running the end-to-end `make verify` independently — THE PRIMARY
   JUDGMENT**: the shared `verify/` top-k token-set check (vLLM-aligned,
   GPU/industry standard) of NPU vs HF bf16 on the real production
   prefill/decode path. A clean independent PASS here is the core verdict.
3. **Auditing the `make verify` code path** as the supplementary
   safeguard — confirming the re-run in (2) actually exercised the
   production kernels vs HF bf16 and gated on the token-set check, not a
   mocked / stale / verify-only path (anti-reward-hacking). The audit
   exists to make the end-to-end re-run trustworthy, not to replace it.
4. Layering on adversarial + anti-fallback sanity checks
5. **Producing a structured evaluation report** humans can read in
   2 minutes to know the full deployment state

The deployment is NOT considered trustworthy until this report is
written and its overall verdict is PASS or PASS-with-warnings.

## Phase 7 PASS criteria (HARD GATES)

1. **`make verify` audited**: subagent has read the Makefile target AND
   the shared `llms/verify/verify_runner.py` + `llms/verify/comparators.py`
   + the model's `verify_adapter.py`,
   and confirmed the gate (a) drives the model's PRODUCTION
   `run_npu_prefill` / `run_npu_decode_step` via the `NpuRunner` in
   `<model>/verify_adapter.py` (not a mock), (b) compares against HF
   transformers loaded in bf16 (shared `llms/verify/runners/hf_runner.py`,
   `torch_dtype=torch.bfloat16`), (c) gates on
   `compute_topk_set_check` (top-5 set inclusion at first divergence),
   not a hardcoded `return True` or a `> 0` threshold. Reward-hacking
   smell test.
2. **`make verify` PASSES under fresh subagent run**: Phase 6's gate
   re-runs cleanly — token-set gate exits 0 (no `npu_vs_hf` FAIL).
3. **`make run` reproducible (twice)**: byte-identical generated text
   across two runs (greedy decode is deterministic). Variance per-
   trial recorded.
4. **Adversarial prompts pass**: 2-3 prompts NOT in the verify prompt
   set pass the same top-5 token-set check vs HF bf16 (catches
   over-tuning to the canonical prompts).
5. **Anti-fallback heuristics pass**: kernels really fired (per-kernel
   ms within expected range; kernel cache exists; cold/warm gap
   observed).
6. **Evaluation report written** in the structured template (see Step
   8 / reference example below). Verdict PASS or PASS-with-warnings.
7. (Optional, conditional) **Cross-deployment regression PASSES** if
   shared infra changed in this deployment.

If the subagent declares PASS without showing measured numbers OR
without auditing the verify code path, the report is REJECTED — re-spawn
with stricter instructions.

## Knowledge base references

- `programming_examples/llms/verify/README.md` — the verify
  methodology the audit checks against (HF bf16 reference, top-k
  token-set gate, cosine-as-diagnosis)
- `programming_examples/llms/verify/comparators.py` — the gate
  implementation (`compute_topk_set_check`) the audit confirms is real
- `<model>/docs/development_progress/` + the `kernel_registry` rows with
  Used by = `<model>`
  — Phase 1 catalog (subagent compares its measurements against this)
- `programming_examples/kernel_registry/supported_kernels.md`
  — kernel-by-kernel ground truth

## Workflow

### Step 1: Spawn fresh subagent with independence + audit constraints

Use the `general-purpose` Agent type. Critical instructions in the
spawn prompt:

- **Independence**: do NOT read `<model>/docs/development_progress/{LESSONS,progress,phaseN_*}.md`
  BEFORE forming your own measurements. You may CITE them AFTER
  measuring (compare your numbers vs claimed).
- **Re-derivation**: every PASS/FAIL verdict must be backed by a number
  YOU measured or a code path YOU read during this audit, not a number
  copied from a deployment doc.
- **Reward-hacking smell test**: if the deployment-claimed numbers look
  surprisingly good, audit the gate implementation FIRST — does
  `make verify` really drive the production prefill/decode vs HF bf16
  and gate on the token-set check, or did the agent shortcut to "if NPU
  output exists → PASS" / point the `verify_adapter` NpuRunner at a mock /
  compare against a stale cached baseline?

### Step 2: Audit `make verify` — anti-reward-hacking

BEFORE running anything, READ:

1. **The Makefile**: what command does `make verify` invoke?
   ```bash
   grep -A 5 "^verify:" <model_dir>/Makefile
   ```
   It should run the **shared** runner `../verify/verify_runner.py
   --runner=<model>.verify_adapter --prompts topk_token ...` — not a
   model-local copy of the runner.
2. **`<model_dir>/verify_adapter.py`**: this is the model's only verify
   code. Confirm its `NpuRunner` (the `.prefill()` / `.decode_step()`
   methods) imports and calls THIS model's production `run_npu_prefill` /
   `run_npu_decode_step` from `<model>_inference.py` — NOT a stub, NOT the
   llama32_1b copy. (The runner/comparators/report/HF runner live once in
   the shared `llms/verify/`; the adapter is the per-model hook.)
3. **`llms/verify/runners/hf_runner.py`** (shared): confirm the reference
   loads with `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)`.
4. **`llms/verify/comparators.py`** + **`llms/verify/report.py`** (shared):
   confirm the gate is `compute_topk_set_check` (top-5 set inclusion at
   first divergence) and `report.has_failure()` drives a real exit code —
   not a hardcoded `return True` / `> 0` threshold.
5. **If any of (2)-(4) fails**: report `[FAIL] verify gate is
   reward-hacked` and stop here. Tag the deployment as needs-remediation.

### Step 3: Run `make verify` — the PRIMARY gate

```bash
cd <model_dir>
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
```

Expected (per Phase 6 / verify subsystem design):

- Both NPU and HF bf16 greedy-decode the prompt set × 32 tokens.
- At the first divergence, NPU's chosen token ∈ HF top-5 AND HF's chosen
  token ∈ NPU top-5 → PASS; exit 0; no `npu_vs_hf` FAIL in the report.
- The report written under the model's build dir (`reports/`, by the
  shared `llms/verify/` runner) records the first divergence + top-5 sets.

Also run `make diagnosis` to capture the per-layer cosine table (each
layer ≥ 0.85, no cliff > 0.05) as a supporting numerical signal.

If the gate fails: record exact failure + cite the divergence position
and which token left the top-5. Verdict = FAIL.

### Step 4: `make run` × 2 reproducibility + perf capture

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=30
flock -x -w 1800 /tmp/mlir-air-npu.lock make run N_TOKENS=30
```

Verify:

- Both runs complete without traceback
- Generated text is byte-identical between runs (greedy → deterministic)
- Capture TTFT (prefill ms) + TPS (tokens/sec) from each run
- If 2nd run is significantly faster than 1st, that's expected (kernel
  cache hit). If they're the same speed AND fast (<5% gap), kernels may
  have been compiled before this audit; that's fine
- If they're the same speed AND surprisingly FAST (e.g., prefill
  < 100 ms for 16+ layers), see anti-fallback (Step 6)

### Step 5: Adversarial prompts (catch over-tuning)

Run NPU + HF bf16 greedy on 2-3 prompts NOT in the verify prompt set.
Examples:

- `"Light travels at"` (expected ` the` or ` approximately`)
- `"DNA stands for"` (expected ` deoxy`)
- `"The Pacific Ocean is the"` (expected ` largest`)

The simplest route: append these to a scratch prompt file and run
`verify_runner.py --prompts topk_token` against it. For each: NPU first
token must be in HF's top-5, AND HF first token in NPU's top-5. Catches
the case where the deployment passed the canonical prompts by happenstance
but doesn't generalize.

### Step 6: Anti-fallback heuristics

Verify kernels actually fire on NPU (not silent CPU fallback):

- Check `<model_dir>/build/{prefill,decode}_kernel_cache/` exists and
  contains `.elf` files
- Per-kernel ms sanity: per-layer prefill ms should be > 5 ms (a CPU
  forward at the same shape is much slower; "kernel didn't actually
  run" looks like 0.01 ms = cache load only)
- For NPU FA: compare `cpu_attn=False` (NPU FA) vs `--cpu-attn` flag
  (CPU). NPU should be ≥ 1.5× faster. If parity, kernel may not have
  fired
- LM-head GEMV: should be > 10 ms (typical 14–22 ms). If < 5 ms,
  check whether it actually ran on NPU vs CPU softmax shortcut

### Step 7: (Conditional) Cross-deployment regression

Trigger only if the recent diff touched shared infra. Quick check:

```bash
git diff main..HEAD --name-only | grep -E "^programming_examples/(kernel_registry/|matrix_vector_multiplication/|llms/(verify/|llama_kernel_builder/|llama32_1b/(multi_launch_builder/|llama32_1b_)))"
```

If matches: re-run `make verify` on EVERY OTHER deployment under
`programming_examples/llms/<model>/` (each has a `verify_adapter.py` into
the shared `llms/verify/`). NPU is a singleton — run sequentially with
`flock`. Budget ~5 min per deployment.
If ANY other deployment's `make verify` regresses, FAIL Category 7 and
require the shared-infra change to be reverted or fixed.

If no shared infra changed: mark N/A.

### Step 8: Write the evaluation report

Output: `<model_dir>/docs/evaluation_report.md`. Use the structure below
(human reviewers expect specific information in specific places).

Required sections (in this order):

```markdown
# Evaluation Report: <Model> on NPU2

**Reference deployment**: `<reference>/` — what was inherited; this
report covers what's different.

## 1. Current Status

### Verified ✓ (<date> — N of M protocol steps)

| Check | Result |
|---|---|
| Auditor agent (`Skill: independent-evaluator`) | <verdict>: <one-line summary> |
| `make run` smoke | First token `<token>` (id=...). N-trial mean prefill <X> s ± <Y> ms |
| `make verify` (NPU vs HF bf16, top-5 token-set) | PASS — first divergence at token <i>; NPU/HF chosen tokens both in the other's top-5 |
| `make diagnosis` (per-layer cosine vs HF bf16) | Per-layer cosine <X>→<Y> over N layers, no cliff |
| Code review | <one-line: clean / silent-fallback / etc.> |

### Performance (<N>-trial mean)

| Phase | Per-layer | Total |
|---|---:|---:|
| Prefill (<N> layers, ...) | X ms/layer | **<Y> s ± <Z> ms** |
| Decode steady-state | X ms/layer | **<Y> ms/token** (<Z> tok/s) |

### Manual Verify Commands

```bash
cd <model_dir>
flock ... make verify
flock ... make run N_TOKENS=30 PROMPT="..."
# Expected first token: ...
# Expected prefill:    ...
# Expected decode:     ...
```

## 2. Architectural Differences vs Reference Deployment

| Field | <Ref Model> | <This Model> | Why it matters |
|---|---:|---:|---|
| n_layers | ... | ... | ... |
| ... | ... | ... | ... |

**The single delta that matters**: <one-line summary>.

## 3. Implementation: Reused vs New

| What | Reused from | New (model-specific) |
|---|---|---|
| Per-layer prefill orchestration | <ref> | <if applicable> |
| ... | ... | ... |

## 4. End-to-End Inference Workflow

### Setup (one-time)
[code-block trace of compile + weight load + BO preload]

### Prefill — runs N times, then once at end
[per-layer XRT call breakdown with ascii boxes for each ELF]

### Decode — per token
[per-layer XRT call breakdown]

### What's on NPU vs CPU
[bulleted lists]

## Notes

- Why per-layer K/V cosine drift looks reasonable here
- Anything redundant vs reference deployment
- Recent fixes worth flagging

## File Map

| File | Role | Lines |
|---|---|---:|
| `<model>_inference.py` | ... | ... |
| ... | ... | ... |
```

The structure is rigid because human reviewers expect to find specific
information in specific places. Don't reorder sections.

## Failure modes

| Symptom | Likely cause | What to do |
|---|---|---|
| `make verify` script doesn't drive the production path vs HF bf16 (Step 2 audit fails) | Reward-hacked gate (mock NpuRunner in verify_adapter, stale baseline, hardcoded pass) | Report `[FAIL] verify gate is reward-hacked`; deployment needs remediation; do NOT mark PASS |
| `make verify` runs but returns PASS suspiciously fast (<10 s for full N-layer model) | Token generation count too low or NpuRunner not actually invoking kernels | Inspect the shared `verify_runner.py` GATE_N_TOKENS + the model's `verify_adapter.py` NpuRunner; confirm it generates 32 tokens through the real prefill/decode |
| `make run` non-deterministic across two runs (greedy) | Sampling enabled by mistake OR uninitialized BO state | `[FAIL]`; bisect to find which call introduces non-determinism |
| Adversarial prompt: NPU top-1 NOT in CPU top-5 | Real correctness issue OR over-tuning to canonical set | Report measurement; verdict depends on severity (one fail = WARN, multiple = FAIL) |
| Per-kernel ms surprisingly low (LM head < 5 ms) | Kernel didn't actually run (silent CPU fallback) | Check kernel cache files exist; flag if missing |
| Subagent reads LESSONS/progress before measuring | Skill prompt wasn't strict enough | Reject the report; re-spawn with stricter instructions |

For any failure not in the table, invoke `superpowers:systematic-debugging`.

## Update protocol

This is the terminal verification phase. On Phase 7 PASS or
PASS-with-warnings:

- `<model_dir>/docs/evaluation_report.md` is the durable artifact
- Append to `<model>/TODO.md`: "Independently evaluated YYYY-MM-DD: <verdict>"
- Reference the report from `<model>/docs/development_progress/progress.md`

If FAIL: deployment cannot be tagged. Issues must be remediated and
Phase 7 re-run.
