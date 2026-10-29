# MLIR-AIR agentic harness v0.1 for AMD XDNA2 NPUs

This is v0.1 of an agentic harness for the AMD XDNA2 NPU compilation flow built on the MLIR-AIR compilation stack. It connects Codex or Claude Code to AIR implementations, measured kernel configurations, and 17 task-focused skills. The agent adapts kernels, composes the model, checks correctness, and profiles the resulting implementation. Skills guide decisions about arithmetic, layout, buffers, data movement, and synchronization; AIR supplies the compiler and runtime feedback.

The harness extends [From Human Guidance to Autonomy: Agent Skill System for End-to-End LLM Deployment on Spatial NPUs](https://arxiv.org/abs/2606.07586), presented at **MLArchSys'26, co-located with ISCA 2026**.

## Start a deployment

Use an MLIR-AIR checkout with its documented toolchain environment and shared [project instructions](../AGENTS.md). For the affine-Q4 example below, the shared six-model deployment interfaces are required. See [installation](INSTALLATION.md) for Codex/Claude loading requirements and [compatibility](profiles/README.md).

From the repository root, start the client and give it a Hugging Face model ID:

```text
# Claude Code
/deploy-new-llm HuggingFaceTB/SmolLM2-135M

# Codex
$deploy-new-llm HuggingFaceTB/SmolLM2-135M
```

Both registrations load the same canonical skill. The agent resolves the model
revision and task, inspects its architecture, reuses or adapts existing kernels,
implements the model runtime, then verifies and profiles it. An existing shared
model, such as `microsoft/phi-2`, starts in reproduction mode; a new model starts
in deployment mode. A single primary agent can complete the workflow.

Add requirements in the same request when needed:

```text
/deploy-new-llm microsoft/phi-2
Use affine Q4 group32 weights and BFP16 prefill GEMM.
Use P512/C1024 and P1024/C2048, batch one.
Run the standard BF16-reference model gate and complete-request profiling.
```

The [task input guide](references/task-input.md) lists the default precision,
native configurations, NPU/host placement, gate and timing contract. The agent
shows the resolved settings and records explicit choices separately. It asks
about material ambiguity and continues when the task is clear. Custom precision,
geometry or validation requests can require implementation work; the task marks
those differences rather than silently using an incompatible shared adapter.

The agent prepares a task with:

```bash
python agentic/tools/task.py microsoft/phi-2 --out /tmp/phi2-task/task.json
```

For shipped examples this reads the pinned local specification without network
access. For a new HF model ID, it resolves a commit and reads config.json; set
HF_TOKEN if model access requires it. An offline new-model task uses
`--config config.json --revision FULL_HF_COMMIT`. Explicit choices are supplied
with `--requirements requirements.json`. The agent then inspects reference code
and weights, declares architectural exceptions, and adds the model configuration/
adapter. Continue through the [new-model procedure](references/project-integration.md#implement-a-new-model)
to produce an executable deployment.

Follow the [runnable Phi-2 example](examples/q4nx-supported-model.md) for weights, compilation, full-logit verification and profiling. New-model and optimization requests use the same [workflow](WORKFLOW.md), entering at the affected implementation stage. Explicit user task requirements take precedence over these example defaults.

Completion includes the production source and build/runtime commands, the
resolved task, numerical gate evidence, raw timing samples and measured TTFT,
decode throughput and request latency. Failed or incomplete deployments retain
their artifacts and unmet requirements. Architecture support and achievable
performance depend on the model, target and the implementation produced.

## What ships

| Component | Responsibility |
|---|---|
| [17 skills](skills/deploy-new-llm/SKILL.md) | Deployment entry, reference preparation, two verification levels, optimization, diagnosis, finalization and review |
| [Megakernel guidance](modules/megakernel/README.md) | Trace changes through computation, ownership, lifetime, layout and transfer protocols |
| [Kernel registry integration](references/kernel-registry.md) | Exact measured configurations, opt-in evidence-bound selection, and reusable local proposals |
| [Current AIR bindings](references/project-integration.md) | Shared packing, model adapters, fixed capture checks and production profiling |
| [Host tools](references/tools-and-evidence.md) | Task setup, source-interface inspection, package consistency and evidence validation |

See [attribution and public source references](references/historical-attribution.md) for the preceding workflow and engineering examples. Model implementation results live with the [six-model examples](../programming_examples/llms/shared/AFFINE_Q4_RESULTS.md).

## Host checks

No compiler, model weights or NPU is needed:

```bash
python -m pip install numpy ml_dtypes pyyaml
python agentic/tools/sync_package.py --check
python -m unittest discover -s agentic/tests -v
```

After editing canonical skills or package files, run `python agentic/tools/sync_package.py`, then the checks. Client entries are generated portable Markdown files. See [maintenance](references/maintenance.md) for evidence proposals and CI.
