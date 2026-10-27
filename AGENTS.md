# Working in MLIR-AIR

Use the checkout's [README](README.md) and platform-specific
[build and test guides](docs/building.md) for the current toolchain flow.
Compiler code lives under `mlir/`, Python interfaces under `python/`, runtime
code under `runtime_lib/`, and runnable designs under `programming_examples/`.
Use the relevant source, tests and example Makefile when changing an interface.

## LLM deployment, adaptation and optimization

Use the [deploy-new-llm skill](agentic/skills/deploy-new-llm/SKILL.md)
for these tasks. Codex exposes it as `$deploy-new-llm`; Claude Code exposes it
as `/deploy-new-llm`. Start with the HF model ID or supplied task and load
supporting skills and references as needed. The
[harness guide](agentic/README.md) describes the entry point.

Record the model revision, requested workload, weight format, compute precision,
placement, correctness criteria and measurement boundaries in the task.
Distinguish explicit user requirements from defaults; resolve material conflicts
before dependent implementation. For the supported affine-Q4 path, prefill GEMM
defaults to BFP16, with model-specific attention and head precision. Reuse the
shared LLM infrastructure and compatible implementations.

## Validation and delivery

Run checks appropriate to the changed behavior and use the existing test flow.
For deployment work, reuse still-applicable evidence and complete the required
affected gates and profiling; investigate additional behavior when a failure or
changed dependency warrants it. Use the runtime's device-locking mechanism for
NPU work and serialize measurements that share a device.

Report reproducible commands, source/toolchain identity, verification outcomes,
measurement boundaries and remaining limitations. Keep measured results tied
to the implementation that produced them.

Project instructions are maintained in this file. Client discovery and loading
requirements are documented in the
[installation guide](agentic/INSTALLATION.md).
