---
name: deploy-new-llm
description: "Coordinate NPU model deployment, adaptation, debugging or optimization through functional skills, kernel and full-model verification, and scoped performance evidence."
---

# Deploy or adapt an NPU model

Start from the HF model ID and any requirements following the skill invocation,
or from the supplied task. Translate explicit requirements into the supported
JSON fields described in [task input](../../references/task-input.md), then run
`python agentic/tools/task.py OWNER/MODEL --requirements /path/requirements.json --out /path/task.json`
from AIR's root. Omit `--requirements` for the documented defaults. Summarize
the resolved model/revision, task mode, precision, configurations, placement and
acceptance before substantial implementation or device work. Continue when
requirements or documented defaults resolve the task; ask only about material
ambiguity or conflicts. Preserve requirements outside the structured fields in
the task record and resolve their implementation explicitly.

Read `execution` in the prepared task. `SHARED_ADAPTER_AVAILABLE` permits reuse
of the existing production workflow for the selected configurations.
`ADAPTATION_REQUIRED` identifies architecture, precision, workload or protocol
work to complete before those commands can satisfy this task. Preserve supplied
quantization rather than silently switching formats, and never replace a
requested gate or timing protocol with the shared tool's fixed defaults.
Unknown-model preparation is followed by reference inspection and implementation;
continue through verification and profiling rather than stopping at `task.json`.
The [task preparation guide](../../references/project-integration.md) covers
offline inputs and adding a new model adapter. Explicit user requirements
override example defaults.

Read the [workflow](../../WORKFLOW.md), supplied source/seed, new requirement, allowed changes/resources, workload, reference, verification criteria, performance target and budget. Choose the needed work from the current evidence; a valid seed for an old requirement does not establish new-valid behavior.

- **Resolve architecture and dependencies.** Inspect config and reference computation order, tokenizer, weight names/shapes, precision and state. Use [reference preparation](../build-cpu-reference/SKILL.md) and the actual [source profile](../../references/project-integration.md). The workshop composition path is a starting asset for matching architectures, not a requirement to decompose a working megakernel. Unsupported operations need scoped implementation work.
- **Identify the implementation path.** Trace the production runtime and build entry for each requested mode. For current Q4NX fused decode, start with the matching existing fused-decode builder, C++ kernels, packing and runtime; use [megakernel change analysis](../megakernel-change-analysis/SKILL.md) for affected relationships. Prefill or other dispatches may separately consume registry-backed shared builders. Read [registry scope](../../references/kernel-registry.md): a registry miss does not force standalone composition or redevelopment of unchanged internal kernels.
- **Create only the needed workspace.** For a new AIR model, use a sibling under `programming_examples/llms` with model-specific configuration/weights, production helpers, build/runtime entry and task records. Reuse shared infrastructure and evaluator. Adapt a module when semantics diverge; avoid copying a whole model with stale imports.
- **Develop and optimize affected kernels.** Registry configurations, operator seeds and megakernel seeds enter the workflow's same adaptation/optimization loop. Reuse unchanged seed implementations and still-applicable evidence. [Kernel verification](../verify-kernel/SKILL.md) supports required development and local [Optimization](../npu-optimization/SKILL.md), applying the applicable registry reference with full-output elementwise tolerances. A standalone row does not verify an operator embedded in a changed fused mapping; preserve missing coverage and integration obligations explicitly. Use [staged checks](../../references/staged-checks.md). Carry the verified kernel's ABI, assumptions, coverage and timing into model integration.
- **Integrate and optimize complete execution.** Supported affine-Q4 prefill GEMMs use BFP16 compute by default; preserve model-specific attention/head precision. [Model verification](../verify-model/SKILL.md) applies AIR's vLLM-derived top-5 output-token gate to the actual production path after complete finite capture and identity checks. Reuse the shared wrapper and create a minimal production-linked adapter when needed. Use the same Optimization entry for host processing, BO/handoffs, synchronization and submission; kernel bottlenecks return to kernel work and affected verification. [Block/layer comparisons](../../references/integration-diagnostics.md) are optional diagnosis, not an extra acceptance stage.
- **Deliver and evaluate.** [Finalize deployment](../finalize-deployment/SKILL.md) retains source/artifacts, reproducible commands, verification/measurement evidence and lessons. [Evaluate deployment](../evaluate-deployment/SKILL.md) audits the production path and applies the same kernel/model gates, checking delivered performance and declaring whether evaluation was independent.

For failures use [debug routing](../../references/debugging.md); for spatial relationships use [megakernel analysis](../../modules/megakernel/README.md). Each entry returns evidence, remaining obligations and a next action to the same task record. Continue within the authorized scope. Acceptance is controlled by the applicable verification contract, not the number or order of skills read.

For a supported-model reproduction or optimization task, use the [runnable Phi-2 example](../../examples/q4nx-supported-model.md). It connects task preparation to the shared production gate and profiling commands.

Single-agent use is complete. Optional reviewers require the user's delegation authorization and use [focused handoffs](../../references/review-and-handoff.md). Keep shared skills/registry unchanged during frozen evaluations; collect [proposed lessons and rows](../../references/maintenance.md) locally. Stop after the required successful affected gates and timing batch; add checks for specific failures or changed dependencies. At the stopping condition, report the best verified result, unmet requirements, attempts and measured/unknown cost.

For the AIR Q4NX fused path, read [fused-decode compatibility](../../references/fused-decode-compatibility.md) when changing compiler flags, packed weights, instruction templates, or KV ownership. Its source-specific constraints and example complement this skill; they do not choose a new model's implementation.
