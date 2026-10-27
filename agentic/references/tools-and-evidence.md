# Tools and evidence

The coding agent performs the source reasoning described by the skills. AIR compiler diagnostics, artifacts, numerical comparisons and timing supply executable feedback. Keep those provenances distinct when reporting a result.

| Tool / interface | Executes | Does not establish |
|---|---|---|
| `agentic/tools/task.py` | Pinned local-spec or HF-config task preparation, explicit requirements and shared-adapter compatibility | Architecture support, model correctness or performance |
| `agentic/tools/air_source_profile.py` | AST symbol presence and current source hashes | Build compatibility or resource feasibility |
| `agentic/tools/sync_package.py --check` | Canonical payload/registration consistency | Actual skill use by a session |
| `programming_examples/kernel_registry/evidence.py` | File identity, record consistency, exact verified selection | Authenticity of an unreviewed contributor's experimental claims |
| `programming_examples/llms/shared/deployment_cli.py` | Model preparation/build/reference/capture/profile through production adapters | Exhaustive numerical logits or model-quality equivalence |
| `programming_examples/llms/shared/deployment_ci.py` | Fixed model gate before production profiling, raw evidence | Other models/configurations beyond the run |

Paths are relative to the repository root. Read [project integration](project-integration.md) before executing model commands. The host-only tools require no NPU; task preparation for a new HF ID accesses the Hub unless given local config and revision.

[Staged checks](staged-checks.md) use source inspection, compiler evidence and runtime results as each becomes available. A source relationship is not a compiler allocation proof. A compiled kernel is not numerically verified. A local kernel PASS is not a full-model gate. Record observed input/workload, implementation identity, coverage and the next unresolved condition in the existing task record.

[Verification contracts](validation-contracts.md) define the two numerical levels. [Maintenance](maintenance.md) describes local registry and lesson proposals. Research-only relation extractors and archived session runners are not dependencies of this public workflow.
