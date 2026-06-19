# Evaluation Report: SmolLM2-1.7B on NPU2

## Verdict: PASS  (2026-06-18)

Independent audit + re-run by an external evaluator treating the deployment as
untrusted. The correctness gate was read end-to-end (no hidden short-circuit),
then re-run on the NPU in this session. `make verify MODEL=base` exited 0 with
2/2 top-k PASS.

## 1. Gate audit (anti-reward-hacking)

- **Shared runner, not a model-local copy.** `smollm2_1_7b/Makefile:89-92`
  `verify` runs `$(srcdir)/../verify/verify_runner.py --runner=smollm2_1_7b.verify_adapter
  --prompts topk_token --model $(MODEL) --max-prompts 2`. The runner lives in
  the shared `llms/verify/` tree (`VERIFY_RUNNER` at line 85). REAL.

- **Adapter runs real NPU kernels, patching is genuine (not a bypass).**
  `verify_adapter.py` `NpuRunner.prefill()` calls production `run_npu_prefill`
  (line 176) and `decode_step()` calls `run_npu_decode_step` (line 239), both
  imported from `llama32_1b_inference`. The inheritance pattern patches two
  functions — `_llama_inf.run_transformer_block = run_prefill_block` and
  `.preload_prefill_weights = _smol_preload` (lines 55-56) — with the MHA-safe
  fork in `smollm2_1_7b_prefill.py`. That fork is real NPU code: it issues
  `cache.load_and_run("rms_gemms_rope"/"o_ffn", ...)` against compiled ELFs
  (prefill.py lines 130, 210; preload lines 281, 313). The patch only makes
  the f32 C-scratch arg set registry-driven (Q for GQA → Q,K,V for MHA so
  K/V fused-cast GEMMs don't read unallocated scratch → NaN). No numpy
  substitution of the kernel path. REAL.

- **HF reference is bf16.** `llms/verify/runners/hf_runner.py:59-61`
  `AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)`.
  Adapter `hf_reference()` returns the same checkpoint (NPU bf16 vs HF bf16 on
  the same weights). REAL.

- **Gate is top-5 set inclusion with a real exit code.** `comparators.py
  compute_topk_set_check` (lines 175-249) finds the first chosen-token
  divergence and FAILs only if either side's chosen token is absent from the
  other's top-5. `report.has_failure()` (`report.py:54-61`) returns True iff a
  `npu_vs_hf` record has `status == "FAIL"`; `verify_runner.py:379-382` then
  `sys.exit(1)`. No hardcoded `return True`, no `>0` threshold cheat. REAL.

**Conclusion: the gate is REAL.** It exercises the same NPU code path as
`make run` and compares against a true bf16 HF reference with a discrete,
non-gameable inclusion criterion.

## 2. Independent `make verify` re-run

Command exited **0**, verdict **PASS**, summary
`{'topk_passed': 2, 'topk_failed': 0}`. The run was not suspiciously fast:
kernels compiled live (rms_gemms_rope 24.3s, o_ffn 51.3s, flash_attn 38.3s +
decode kernels), 24 transformer layers preloaded into per-layer BOs ("MHA-safe"),
3584 MB decode weights staged, runtime prepared in 14.6s — i.e. it genuinely
built and ran a 24-layer model on the NPU.

Report: `llms/verify/reports/verify_topk_token_20260618-174007.md`

| Prompt | Diverge step | NPU choice (rank in HF) | HF choice (rank in NPU) | Status |
|--------|-------------:|-------------------------|-------------------------|:-------|
| 0 "Introduce me what is GPU" | 14 | " electronic" (#2) | " processor" (#2) | OK |
| 1 "Briefly describe ... AI 1950-2020" | 1 | "-" (#2) | "\n" (#2) | OK |

Both divergences sit at rank #2 in the other side's top-5 — well within band.
Generated text is coherent; prompt 0 agreed prefix (steps 0-13):
`"?\n\nGPU stands for Graphics Processing Unit. It is a specialized"`.

**Diagnosis cosine (informational, not a gate)** —
`llms/verify/reports/diagnosis_20260618-174302.md`, 24 layer records:
per-layer ffn_out median cosine ≥ 0.9986 for all layers; cos_min ≥ 0.92
(layer 7 dips to 0.923 then monotonically recovers). No NaN, no gross cliff in
the transformer stack. The layer-23 cell shows cos_min 0.030 / p5 0.321 but
median 0.9989 — this row is the post-final-norm hidden-state probe, a known
NPU-vs-HF convention pairing (HF exposes `hidden_states[-1]` as post-norm; see
`hf_runner.py:94-113`), not a kernel regression. Informational only; the top-k
gate is the correctness signal.

## 3. Manual reproduce

```bash
cd /home/jiajli/apps/mlir-air/.claude/worktrees/skills-deploy-test/programming_examples/llms/smollm2_1_7b

# Primary gate (NPU bf16 vs HF bf16, 2 prompts x 32 tokens, k=5). Exit 0 = PASS.
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify MODEL=base

# Informational per-layer cosine table (not a gate).
flock -x -w 1800 /tmp/mlir-air-npu.lock make diagnosis MODEL=base

# Reports land in:
#   ../verify/reports/verify_topk_token_<stamp>.md
#   ../verify/reports/diagnosis_<stamp>.md
```
