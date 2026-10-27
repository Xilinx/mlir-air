---
name: verify-model
description: "Verify the production NPU model with AIR's vLLM-derived top-5 output-token method, recording exact comparison coverage and using intermediate checks only for diagnosis or explicit task requirements."
---

# Verify the production model

Use the actual production functions, complete weight mapping, declared arithmetic and fixed [verification contract](../../references/validation-contracts.md). Bind checkpoint/tokenizer/reference, candidate source/build/artifacts, prompts and runner/adapter identity. Inspect layer-indexed arguments, weights, BO/cache keys, shared aliases, final norm/head and token mapping when connecting the evaluator.

The default full-model gate is AIR's vLLM-derived **top-5 token inclusion** method. The inspected shared verifier generates 32 greedy tokens per prompt for both NPU and HF BF16 reference, compares chosen tokens along their common prefix, then checks mutual top-5 inclusion at the first divergence and stops comparison. It does not stop generation on EOS. Exact empty/length/top-k/reference behavior and the differences from upstream vLLM options and prompt checking are in [verification contracts](../../references/validation-contracts.md).

Use `make verify` only after inspecting the current Makefile, adapter, comparator and report/exit handling. Confirm the intended production artifact/checkpoint and prompt/token capture. Record the requested and captured lengths, available comparison length, first divergence, inclusion verdict and early stopping. The comparator can return OK on an empty or truncated shared prefix; report missing/incomplete evidence rather than claiming the intended run was covered. These capture checks interpret the existing comparator; they do not change it or impose an additional numerical gate.

A full-model PASS is scoped to this output-token proxy on the observed prompts/prefix. It does not establish bitwise output, logit/logprob equality, full KV-state equivalence, later divergent generation quality or a general quality benchmark. Do not turn layer cosine, block comparisons, logits or state probes into additional mandatory global acceptance levels. Use [integration diagnosis](../../references/integration-diagnostics.md) and [debug routing](../../references/debugging.md) when a failure needs localization; retain extra checks only when the user's specific task or kernel interface requires them.

After an integration or end-to-end optimization change, rerun the applicable model gate before adopting the candidate. Changed kernels also restore their affected kernel verification; [staged evidence reuse](../../references/staged-checks.md) can retain an unchanged build with a recorded dependency basis. Local kernel PASS does not replace the model gate. Supply the verified deployment baseline, actual coverage and limitations to [Optimization](../npu-optimization/SKILL.md), or [finalize the deployment](../finalize-deployment/SKILL.md).
