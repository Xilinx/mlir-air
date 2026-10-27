# Resolve an HF ID into a deployment task

Users can begin with `/deploy-new-llm OWNER/MODEL` in Claude Code or
`$deploy-new-llm OWNER/MODEL` in Codex. Text following the model ID can specify
precision, workload, placement, acceptance, performance goals or a budget.
The agent converts those choices into a task and carries it through implementation,
verification and profiling.

## Defaults and explicit choices

An ID-only request uses these defaults:

| Requirement | Default |
|---|---|
| Mode | Reproduce a pinned shared example when available; otherwise deploy a new model |
| Model identity | The example's pinned revision, or a full Hub commit resolved when preparing a new model |
| Target / batch | NPU2 / batch one |
| Weights | Affine Q4 group32, `w = scale*q + min`; preserve model-specific BF16 exceptions and supplied logical Q4 values |
| Prefill GEMM | BFP16 compute; attention/head precision remains model-specific |
| Configurations | P512/C1024, P1024/C2048, P2048/C4096, P4096/C8192, restricted by the documented native context/cohort |
| Placement | All model blocks and full head on NPU; embedding, orchestration/sampling, byte-preserving transfers and a declared final single-row prefill norm may run on host |
| Model gate | Original HF CPU BF16; 32 greedy outputs, complete finite logits, mutual top-5 at first divergence, ignoring EOS |
| Profiling | Fresh 128-output complete requests; warmup ≥5 s, ≥3 measurements and ≥10 measured seconds |
| Performance goal / budget | None supplied; report measured performance and stop after the required successful checks and timing batch |

Q4 weight representation, BFP16 GEMM compute, the BF16 reference and the model
acceptance rule are separate choices. The default token gate does not require
all 32 choices to agree. A user can request exact token agreement instead.

Prepare a default task from the repository root:

```bash
python agentic/tools/task.py OWNER/MODEL --out /tmp/deployment/task.json
```

For a new model, the tool retrieves only Hub metadata and config, pins the
revision and marks architecture work as required. It does not download weights
or execute model code. Offline preparation uses `--config config.json
--revision FULL_HF_COMMIT`. `--native-context N` records a context limit established
from the pinned reference when config alone is insufficient. A shared example's
largest tested capacity is a task limit, not proof of the model's full context.
Larger requested capacities without that reference review are marked
`CONTEXT_REVIEW_REQUIRED`.

## Requirements file

Use `--requirements requirements.json` to replace specific defaults. Nested
objects merge with their defaults; unknown fields and invalid geometry are
rejected. The output retains `explicit_requirements` alongside the resolved
task so reviewers can distinguish user choices from defaults.

For example, a user selecting the two native Phi-2 configurations:

```json
{
  "mode": "reproduce",
  "configurations": [
    {"prefill": 512, "capacity": 1024},
    {"prefill": 1024, "capacity": 2048}
  ]
}
```

An optimization task with different precision and acceptance:

```json
{
  "mode": "optimize",
  "target": "npu2",
  "batch_size": 1,
  "configurations": [{"prefill": 512, "capacity": 1024}],
  "arithmetic": {"logical_weights": "BF16", "prefill_gemm": "BF16"},
  "gate": {"comparison": "exact_tokens"},
  "performance_goal": "Minimize TTFT while preserving exact token agreement",
  "budget": "Stop after 90 minutes and report the best verified candidate"
}
```

Supported top-level fields are `mode`, `target`, `batch_size`, `configurations`,
`arithmetic`, `placement`, `gate`, `timing`, `performance_goal` and `budget`.
Their defaults and nested fields are defined in `DEFAULTS` in
[task.py](../tools/task.py). Weight/placement/reference descriptions can express
model-specific requirements. For example, an existing GPTQ checkpoint needs its
exact quantization semantics recorded; it cannot be treated as the default
affine Q4 merely because both use four bits.

For first-token-only profiling, set `timing.boundary` to `first_token` and
`timing.outputs` to `1`. The default full-model gate remains in place. Physical
KV handoff/reset completed after the first token is reported separately from
TTFT. The shared complete-request profiler needs adaptation for this request.

The agent presents a concise task summary before substantial work. Requirements
already supplied or resolved by documented defaults need no additional approval
step. Clarify conflicting precision/asset requirements or other material
ambiguity once, while continuing independent inspection.

## Bind the implementation to the task

`execution.status` separates two cases:

- `SHARED_ADAPTER_AVAILABLE`: the pinned shared implementation accepts these
  settings. Run its existing workflow for each required P/C in the task, using
  the bound model/input specifications. A subset of supported configurations
  stays a subset; do not silently run the full cohort.
- `ADAPTATION_REQUIRED`: implement the listed architecture, configuration,
  precision, placement or verification/profiling changes. The task records what
  is wanted; it does not certify support. Confirm the resulting build/runtime
  and evaluator actually consume those settings before running the final gate.

The current shared CLI fixes its model gate and 128-output profiling protocol.
Passing a custom task file does not change that CLI. Adapt or add the affected
execution interface when required, and record the mapping from task fields to
the actual build flags, adapter, inputs and verification/profiling commands.
Keep reference identity, finite captures and the requested criteria intact.

For a new model, inspect reference computation, tensors and state, reuse a
compatible seed, implement/register its model specification and production
adapter, then complete the [deployment workflow](../WORKFLOW.md). A family name
or successful config download does not establish architecture compatibility.
