# Validation and measurement

The tests exercise the production runtime and retain separate numerical scopes.
Run them in the normal AIR environment after `make weights` and `make compile`.
NPU commands in the Makefile take an exclusive `/tmp/mlir-air-npu.lock`.

## Standard checks

```sh
make verify-host   # 440 acceptance/EOS/clipping cases and streaming tests
make verify-state  # Repeats, visibility, context injection, NPU prefill and loop
make verify-model  # Separate NPU/HF captures, then the AIR top-5 comparison
make verify        # All three
```

Reports go to a timestamped `build/verification/` subdirectory. The model check
captures 32 tokens per prompt with EOS stopping disabled. AIR's comparator
walks equal chosen tokens, checks mutual top-5 inclusion at the first divergence,
and stops comparison there. The report gives the first divergence and number of
positions actually visited; all captures must contain the full 32 tokens.

The state check compares DFlash with a sequential oracle using the same B8
matrix arithmetic. It checks exact output tokens, repeat acceptance decisions,
and committed target KV. That oracle isolates speculative logic from B1 GEMV's
different accumulation order. Decoder tests additionally perturb future KV and
the last block row to check causal target and bidirectional drafter visibility.

NPU prefill reuses upstream's BFP attention, whose operand exponent is shared
across groups of eight key positions. Extending a partly filled group can change
rounding inside that group, even with a causal mask. The prefill check requires
exact repeats, exact completed groups, and exact visibility across a new
attention tile; it also records the partial-group differences. The B8 decoder's
future-KV and commit checks remain exact within their matching-arithmetic scope.

## Focused kernel checks

The attention harness links the production attention object into a small graph:

```sh
python tests/check_attention.py --build --source build/target --output build/attention-target
flock /tmp/mlir-air-npu.lock python tests/check_attention.py --run \
  --output build/attention-target --result build/attention-target.json
python tests/check_attention.py --build --no-causal --source build/draft --output build/attention-draft
flock /tmp/mlir-air-npu.lock python tests/check_attention.py --run \
  --output build/attention-draft --result build/attention-draft.json
make benchmark-data
flock /tmp/mlir-air-npu.lock python tests/check_prepass.py \
  --artifacts build/prepass --fixture build/benchmark_bundle/inputs/prefill/00.npz \
  --output build/prepass-check.json
```

Attention compares every output element with FP32 SDPA converted to BF16,
`rtol=0.016`, `atol=0.1`, zero outliers. Both visibility modes also require exact
repeats, row isolation and invisibility of stale future KV.

The prepass check covers 1–8 rows, prompt features, random values, unequal row
scales, identical rows, zeros, mixed positions and multi-chunk inputs. Projection
implementation checks use a matching BFP operand-conversion reference; full
BF16/FP32 reference differences are also reported as arithmetic-error metrics.
Full-output projection comparisons use `rtol=0.016`, `atol=0.0015`; feature norm
uses `atol=0.05`. Repeats and padding/chunk isolation must match exactly.

## Benchmark

```sh
make benchmark
# Short coverage check, using one frozen prompt from each dataset:
make benchmark PER_DATASET=1 STUDY_TOKENS=256 BENCHMARK_OUTPUT=build/pilot.jsonl
```

The default is 80 requests: 16 per dataset, greedy, thinking disabled, maximum
2048 output tokens, one pass. MT-Bench uses first turns. Frozen prefill states
make this a decode comparison, not a prefill/TTFT benchmark. Interactive `run`
and `chat` use NPU prefill. See [benchmark inputs](../benchmark/README.md).

Each JSONL record includes tokens, acceptance trace, component times and input
identity. The adjacent `.summary.json` updates after each request and reports
per-dataset and overall results. Pooled throughput is total emitted tokens
divided by total decode seconds. Equal-request throughput averages each request's
throughput. Emitted tokens/round is total emitted tokens divided by total rounds.

The main timer includes initial and later NPU context preparation, drafting,
verification, acceptance and host work. `official_tokens_per_second` uses the
paper-era boundary after the first complete draft, immediately before its
verification. Both exclude the prefilled first token from emitted-token counts.
No historical B1 time is silently used as a new-machine baseline.

## Recorded self-validation

On the upstream base and dependencies listed in [compatibility.md](compatibility.md),
the Ryzen AI 9 HX 370 / NPU2 validation completed the following checks:

| Check | Executed coverage |
|---|---|
| Compiler | 195 AIRToAIE / AIRDependencyScheduleOpt regressions passed |
| Host | 440 acceptance cases and four streaming tests passed |
| Target and drafter | Six KV prefixes each: 0, 7, 16, 64, 1024, 4088 |
| Attention | 24 causal and 24 bidirectional full-output cases |
| Context preparation | 40 input/row cases, plus mixed positions and chunking |
| Saved-prefill loop | 32 tokens, nine rounds; oracle tokens, repeated decisions and every-round committed KV exact |
| NPU-prefill loop | Repeats, visibility and exact oracle/commit checks; eight output tokens before EOS |
| HF model check | Two complete 32-token captures; top-5 checks pass at first divergences 5 and 18 (zero-based), visiting six and nineteen positions |
| User commands | `make run` and a resident two-prompt `make chat` completed with NPU prefill and streaming |

The `make benchmark` pilot used one request per dataset and a 256-token cap:
1003 emitted tokens, **48.960 pooled tokens/s**, **44.961 equal-request tokens/s**,
and **5.091 emitted tokens/round**. This short coverage check is separate from
the historical Full80 result in the README. Full80 and a fresh B1 comparison
have not been rerun on this toolchain. Tests were performed by the implementation
author; they are self-validation, not an independent review.
