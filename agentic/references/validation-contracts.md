# Kernel and full-model verification contracts

The package has two verification levels. Select the reference and criteria before a candidate run, preserve the exact implementation and coverage, and do not relax a failed threshold. Block/layer/logit/state observations support diagnosis or explicit task/interface requirements; they are not additional global acceptance levels. Reference preparation and final evaluation are responsibilities, not extra numerical gates.

## Kernel: full-output elementwise comparison

Compare every required output element against the applicable registry reference using `abs(actual - reference) <= atol + rtol * abs(reference)`. Record shape, input distribution, input/output dtype, actual arithmetic and accumulation, reference conversions, harness, `rtol`, `atol`, nonfinite handling, mismatch count and complete compared-output coverage. A BF16 interface can use BFP/Q4NX arithmetic internally.

The intended default permits zero out-of-tolerance elements. Use the actual registry/operator contract; do not infer tolerances from a mean-error statistic or lookup result. GEMM lookup returns implementation/tile/performance/error metadata, not a tolerance object or a verified PASS. Registered `NUMFAIL` rows remain failures. A nearby shape or successful build does not supply missing numerical evidence.

The inspected references follow GPU/framework numerical-testing practice: an appropriate high-precision/framework reference, output-representation conversion, and full-output relative/absolute tolerance. Their BF16 `rtol=0.016` agrees with the inspected PyTorch BF16 default; **their operator-specific absolute tolerances are not PyTorch defaults** (the inspected PyTorch BF16 `atol` is `1e-5`). Framework practice does not make one threshold valid for every kernel or input distribution.

### Representative inspected contracts

Paths below are relative to the supplied AIR root, at commit `6746658f74200feb957f99ad4163297f3185739c`. Reinspect the selected source/harness when its revision changes. This table records existing criteria, without changing any executable threshold.

| Kernel/path | Reference | rtol | atol | Source anchor |
|---|---|---|---|---|
| BF16-input/BF16-output GEMM, high fused-cast or drain path | FP32 matmul of BF16 inputs, cast reference output to BF16 | 0.016 | 0.0015 | `programming_examples/matrix_multiplication/bf16_in_bf16_out/run.py:1347-1366` |
| BF16-input/BF16-output GEMM, low direct path | Same reference/output conversion | 0.016 | 0.004 | Same harness and lines |
| BF16-input/FP32-output GEMM | FP32 matmul of BF16 inputs; output remains FP32 | 0.016 | 0.0015 | `programming_examples/matrix_multiplication/bf16_in_fp32_out/run.py:1111-1118,1131-1144` |
| Weighted RMSNorm | FP32 weighted RMSNorm, reference cast to BF16 | 0.016 | 0.05 | `programming_examples/weighted_rms_norm/weighted_rms_norm.py:109-113,259-264` |
| SiLU-and-mul | FP32 `SiLU(gate) * up`, reference cast to BF16 | 0.016 | 0.08 | `programming_examples/silu_and_mul/silu_and_mul.py:139-142,210-219,238-243` |
| Sequence-first attention | FP32 scaled dot-product attention, reference cast BF16 | 0.016 | 0.1 | `programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py:724,784-790` |

These listed paths compare the complete output with zero allowed mismatches. **Heads-first attention is a source discrepancy:** `programming_examples/flash_attention/kernel_fusion_based/attn_npu2.py:743-750` instead uses `rtol=0.04`, `atol=0.15`, allows 0.5% mismatches, and requires correlation at least 0.99. Its inputs are nonnegative uniform (`:680`); the sequence-first harness uses signed standard-normal inputs (`:724`). Do not claim the current harnesses enforce the same registry contract or convert a looser heads-first harness PASS into the intended strict full-output PASS. Record the discrepancy and establish an aligned comparison or leave that intended criterion unverified; do not silently alter production thresholds or the frozen task.

Only passing evidence enters kernel best-valid ranking and a verified local handoff. Include ABI/packing assumptions and scoped timing when claimed. A kernel-only task can finish here; model deployment separately applies the model gate. [Integration diagnosis](integration-diagnostics.md) can localize a failure, but layer cosine is not an isolated kernel PASS.

## Full model: AIR top-5 output-token inclusion

### Current affine-Q4 integration

Use `programming_examples/llms/shared/deployment_cli.py:compare_capture` for the shared affine-Q4 examples. It checks matching model/revision/input IDs, finite `(32, vocab)` logits and saved-choice/argmax consistency before invoking the shared comparator. `shared/deployment_ci.py` applies the gate before profiling. Keep these guards when adding an adapter; importing the raw comparator alone does not validate capture completeness. The source discussion below records the comparator's historical contract and explains why the wrapper is required.

The default uses the production AIR model adapter and shared verifier. At the same AIR revision, `programming_examples/llms/verify/verify_runner.py:71-72` sets `GATE_K=5` and `GATE_N_TOKENS=32`. `runners/hf_runner.py:54-94` under that verifier directory loads the HF BF16 CPU reference; an adapter's `build_hf_model` hook can override construction. Bind the actual checkpoint, tokenizer, hook and precision. The Q4NX/BFP NPU target need not share the reference's arithmetic.

The runner free-runs each implementation greedily from the same prompt. Generation runs the fixed number of steps and **does not stop on EOS** (`verify_runner.py:110-140`). Do not substitute a production generation command's EOS behavior for this verifier's capture semantics. Prompt selection, capture and report handling are at `:365-378` and `:410-498`; freeze and record the actual prompt cohort.

`compute_topk_set_check`, in `programming_examples/llms/verify/comparators.py:194-272`, behaves as follows:

1. Set the available comparison length to the minimum of the two chosen-token sequence lengths and the two top-k trace lengths.
2. Walk chosen tokens along that available prefix. Equal chosen tokens continue without checking per-step top-k contents.
3. At the first differing chosen token, require nonempty top-k lists and check mutual inclusion: the reference token is in the NPU top-5 and the NPU token is in the reference top-5. Return OK or FAIL and stop comparison at that position.
4. If no divergence occurs in the available prefix, return OK. This includes an empty available prefix; unmatched tails are ignored by this helper.

Top-k selection is at `comparators.py:144-161`. An empty per-step top-k list makes mutual inclusion fail at a divergence; an equal-token prefix does not inspect those lists. Missing outer-trace entries shorten the available minimum length. The returned `n_steps` is the **available minimum length**, not the number of positions actually visited before a divergent-token return. There is no numerical logit/logprob-value comparison. An accepted first divergence does not validate subsequent divergent generation.

### Capture integrity and reporting

Before claiming the intended model check completed, confirm the expected prompt coverage and intact, nonempty token/top-k traces at the verifier's intended lengths. For this runner that means the requested fixed-length generation, including tokens after EOS; shortened traces are incomplete evidence, not an implicit EOS exception. Record all four lengths, available comparison length, first divergence/actual visited prefix, inclusion result, errors and any missing capture. These are evidence prerequisites for interpreting the unchanged comparator, **not another numerical gate or a rewritten comparator**. Preserve a raw OK while marking intended coverage incomplete when traces are empty/truncated.

A model PASS covers the recorded top-5 token proxy on the observed prompts/prefix. It does not establish bitwise output, numerical logits/logprobs, full KV-state equality, all later generated tokens or general model quality. `make diagnosis` and block/layer/state probes are optional localization. Additional task-specific requirements remain explicit instead of becoming package-wide defaults.

### vLLM provenance and limits

AIR identifies vLLM's `check_logprobs_close` as the source idea but does not record an exact ancestor commit. Independently inspected [vLLM v0.10.2 source](https://github.com/vllm-project/vllm/blob/01efc7ef781391e744ed08c3292817a773d654e6/tests/models/utils.py#L83), commit `01efc7ef781391e744ed08c3292817a773d654e6`, corroborates the method: its first-divergence path checks token membership in logprob dictionaries and then breaks (`:207-241`). Despite the function name, that path does not compare numerical logprob values. Upstream configurable top-N/options and prompt checks differ from AIR's fixed-K simplified trace comparator; this is methodological provenance, not proof that AIR is byte-equivalent to that revision or inherits every vLLM check.

## Evaluation and reuse

The [evaluator](../skills/evaluate-deployment/SKILL.md) audits the actual candidate/runtime/reference/comparator path and applies these same two verification levels, then checks the declared delivery and performance boundary. It is an execution/review role, not a third numerical gate. Label fresh versus reused evidence and self-validation versus independent evaluation.

After a fix, rerun affected kernel/full-model checks with the same criteria. [Staged checks](staged-checks.md) permits justified reuse of an unchanged build; runtime changes still require the applicable gate. Use [optional integration diagnosis](integration-diagnostics.md) for boundary, layout, state or layer hypotheses. Report unknown required evidence as incomplete instead of inventing a PASS or weakening a threshold.
