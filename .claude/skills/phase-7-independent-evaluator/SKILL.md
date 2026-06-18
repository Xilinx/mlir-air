---
name: phase-7-independent-evaluator
description: Phase 7 of LLM deployment — spawn a fresh subagent that treats the deployment as UNTRUSTED, audits the `make verify` implementation (anti-reward-hacking: confirms the token-set gate runs the production path vs HF bf16), then re-runs it as the primary gate. Produces a structured `evaluation_report.md` a human can read in 2 minutes to know the full deployment state. Invoke as `/phase-7-independent-evaluator <model_dir>` or auto-spawn from deploy-new-llm after Phase 6 PASS.
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
2. **Auditing the `make verify` code path** (anti-reward-hacking) —
   confirming the gate actually drives the production kernels vs HF bf16
   and gates on the token-set check, not a mocked / stale / verify-only
   path. This is what makes the independent re-run meaningful: without it,
   "the agent re-ran the gate it wrote itself" proves nothing.
3. **Re-running the end-to-end `make verify` independently** — the shared
   `verify/` top-k token-set check (vLLM-aligned) of NPU vs HF bf16 on the
   real production prefill/decode path. A clean independent PASS, on a gate
   the audit confirmed is real, is the verdict.
4. **Producing a short evaluation report** humans can read in 2 minutes.

The deployment is NOT considered trustworthy until this report is
written and its overall verdict is PASS or PASS-with-warnings.

## Phase 7 PASS criteria (HARD GATES)

1. **`make verify` audited** (anti-reward-hacking): subagent has read the
   Makefile target + the model's `verify_adapter.py` + the shared
   `llms/verify/{verify_runner,comparators,runners/hf_runner}.py`, and
   confirmed the gate (a) drives the model's PRODUCTION `run_npu_prefill` /
   `run_npu_decode_step` via the `NpuRunner` (not a mock), (b) compares
   against HF transformers in bf16 (`torch_dtype=torch.bfloat16`), (c)
   gates on `compute_topk_set_check` (top-5 set inclusion at first
   divergence), not a hardcoded `return True` / `> 0` threshold.
2. **`make verify` PASSES under the fresh subagent run**: token-set gate
   exits 0 (no `npu_vs_hf` FAIL), on the audited-real gate.
3. **Short evaluation report written** with the audit verdict + the
   re-run result + the measured numbers behind them. Verdict PASS or
   PASS-with-warnings.

If the subagent declares PASS without showing measured numbers OR without
auditing the verify code path, the report is REJECTED — re-spawn with
stricter instructions.

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

Also run `make diagnosis` to capture the per-layer cosine table as an
**informational** sanity signal — eyeball it for a gross cliff or a NaN
layer. It is NOT a gate: the verify subsystem retired threshold-based
diagnosis (`compare_pair` reports cosine with no pass/fail), so do not
fail the evaluation on a cosine number. `make verify` (the token-set gate)
is the sole binding numeric verdict; the cosine table just helps localize
*if* verify fails.

If the gate fails: record exact failure + cite the divergence position
and which token left the top-5. Verdict = FAIL.

### Step 4: Write the evaluation report

Output: `<model_dir>/docs/evaluation_report.md`. Keep it short — its job is
to let a human know, in 2 minutes, whether the deployment is trustworthy
and why. Use the structure below:

```markdown
# Evaluation Report: <Model> on NPU2

## Verdict: <PASS / PASS-with-warnings / FAIL>  (<date>)

## 1. Gate audit (anti-reward-hacking)
- `make verify` drives production `run_npu_prefill` / `run_npu_decode_step`
  via `<model>/verify_adapter.py` NpuRunner (not a mock): <yes/no + evidence>
- Reference is HF transformers bf16 (`torch_dtype=torch.bfloat16`): <yes/no>
- Gate is `compute_topk_set_check` (top-5 at first divergence), not a
  hardcoded pass / `> 0` threshold: <yes/no>
- Conclusion: gate is REAL / reward-hacked

## 2. Independent `make verify` re-run
- Result: PASS / FAIL — first divergence at token <i>; NPU & HF chosen
  tokens both in the other's top-5
- `make diagnosis` per-layer cosine (informational only): <X→Y; cliff/NaN?>

## 3. Manual reproduce
    cd <model_dir>
    flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
```

## Failure modes

| Symptom | Likely cause | What to do |
|---|---|---|
| Step 2 audit fails — `make verify` doesn't drive the production path vs HF bf16 | Reward-hacked gate (mock NpuRunner, stale baseline, hardcoded pass) | Report `[FAIL] verify gate is reward-hacked`; deployment needs remediation; do NOT mark PASS |
| `make verify` returns PASS suspiciously fast (<10 s for a full N-layer model) | Token count too low, or NpuRunner not actually invoking kernels | Check the shared `verify_runner.py` GATE_N_TOKENS + the model's NpuRunner generates 32 tokens through the real prefill/decode |
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
