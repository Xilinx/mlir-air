# Validation and measurement

The tests exercise the production runtime and retain separate numerical scopes.
Run them in the normal AIR environment after `make weights` and `make compile`.
NPU commands in the Makefile take an exclusive `/tmp/mlir-air-npu.lock`.

## Standard checks

```sh
make verify-host   # Acceptance/EOS/clipping and streaming unit tests
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
  --artifacts build/prepass --fixture "$HOME/.cache/mlir-air/dflash/benchmark-16x5-v1/inputs/prefill/00.npz" \
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
make full20       # Nightly paired B1 + DFlash, four fixed prompts per dataset
make full80       # Optional larger pair, sixteen fixed prompts per dataset
# Short coverage check, using one frozen prompt from each dataset:
make benchmark PER_DATASET=1 STUDY_TOKENS=256 BENCHMARK_OUTPUT=build/pilot.jsonl
```

The default `make benchmark` is 80 requests: 16 per dataset, greedy, thinking disabled, maximum
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

### Current upstream base, October 17

The submission is rebased on upstream `4bf1d78f`, with a fresh AIR compiler and
fresh target, drafter, prepass and prefill builds. Dependency wheel pins and
the three model revisions are unchanged. Local HX370/NPU2 checks:

| Check | Executed coverage |
|---|---|
| Compiler | 647 PASS; 7 expected failures; 7 unsupported |
| Compile LIT | PASS; target, drafter, prepass and C2048 prefill, 388.33 s |
| Verify LIT | PASS, 270.86 s; host gates, six-prefix state checks, NPU-prefill loop and HF model checks |
| Complete loop | Exact repeated decisions, 32-token same-B8 sequential oracle and committed KV after all 12 rounds |
| HF model proxy | Two full 32-token captures; mutual top-5 PASS at first divergences 2 and 13, visiting 3 and 14 positions |
| Attention | 24 causal and 24 bidirectional full-output cases; repeats/isolation/visibility pass |
| Context preparation | 40 input/row cases plus mixed positions and multi-chunk checks pass |
| Output preservation | Target/drafter case records, non-timing loop fields and both NPU/HF chosen-token/top-5 captures match the October 16 run |
| Profile LIT | PASS, 104.82 s including initialization; 18-token GPU prompt, cap256, NPU prefill, warm TTFT 5047.138 ms and decode 22.429 tokens/s |
| Full20 LIT | PASS, 827.82 s; 20 actual B1 requests and 20 DFlash requests, with paired identities and round accounting checked |

Initialization now shares resident MLP weights before LM-head setup, reads the
original BF16 embedding without an FP32 temporary, and clears mapped BOs without
a second KV-sized array. All 388,956,160 embedding elements match the previous
conversion path exactly. Verify completed under the original 20 GiB job cap and
memory-pressure guard, with a sampled peak of 17.4 GiB. Decoder arithmetic,
checkpoint revisions, test tolerances and benchmark timing are unchanged.

| Dataset (4 requests each) | B1 tok/s | DFlash tok/s | Pooled speedup | Emitted/round |
|---|---:|---:|---:|---:|
| GSM8K | 15.862 | 50.398 | 3.177× | 5.251 |
| MATH-500 | 15.726 | 55.685 | 3.541× | 6.069 |
| HumanEval | 15.719 | 48.276 | 3.071× | 5.195 |
| MBPP | 15.833 | 48.953 | 3.092× | 5.076 |
| MT-Bench, first turn | 15.822 | 19.853 | 1.255× | 2.084 |
| **Overall, 20 requests** | **15.775** | **42.240** | **2.678×** | **4.490** |

Proposed-token acceptance is 50.130%. Both engines' 20 output arrays and all
1,870 DFlash non-timing round traces match the previous Full20. The B1 and
DFlash processes take 9.75 and 3.88 minutes. Full80 was not rerun on this base;
its earlier measurements below retain their own source/toolchain identity.
The workflow/result-collector sources are unchanged from the validated Full20
integration. These are local implementation-author checks; GitHub Actions and
a second NPU2 machine have not executed this branch.

### Previous upstream base, October 16

On October 16, the branch was rebuilt against upstream `25e913ff` and its
MLIR-AIE `1.4.5.dev46+ge7048fa`, Peano `22.0.0.2026101201+83ba9ea9` and
LLVM/MLIR `24.0.0.2026100106+bdf04ebb` dependencies. Local HX370/NPU2 checks:

| Check | Executed coverage |
|---|---|
| Compiler | 620 PASS; 7 expected failures; 7 unsupported |
| Compile LIT | PASS: target, drafter, NPU prepass and C2048 prefill, 392.15 s |
| Verify LIT | PASS, 271.41 s; host gates, six-prefix target/drafter state checks, NPU-prefill loop and HF model checks |
| Complete loop | Fixed 32 tokens; exact repeat acceptance/oracle output and committed KV after each of 12 rounds |
| HF model proxy | Two complete 32-token captures; mutual top-5 PASS at zero-based first divergences 2 and 13, visiting 3 and 14 positions |
| Attention | 24 causal and 24 bidirectional full-output cases; repeats/isolation/visibility pass |
| Context preparation | 40 input/row cases plus mixed positions and multi-chunk checks pass |
| Profile LIT | PASS, 107.39 s including initialization; 18-token GPU prompt, cap256, NPU prefill, warm TTFT 4816.903 ms and decode 22.467 tokens/s |
| Full80 LIT | PASS, 3216.51 s; 80 actual B1 requests and 80 DFlash requests, with all paired input identities checked |
| Historical selected B8 comparison | All 80 output arrays and non-timing acceptance traces match, covering 7935 speculative rounds |
| CI result collection | Actual XUnit/profile/Full80 outputs pass the workflow's collection and job-summary code; partial request logs are retained |

The actual profile and verification tests ran with eight CPU threads. The
Full80 study used the same setting and the frozen protocol described above.
These are implementation-author checks, not an independent or GitHub Actions
evaluation.

| Dataset (16 requests each) | B1 tok/s | DFlash tok/s | Pooled speedup | Emitted/round |
|---|---:|---:|---:|---:|
| GSM8K | 15.866 | 46.191 | 2.911× | 4.815 |
| MATH-500 | 15.757 | 51.058 | 3.240× | 5.588 |
| HumanEval | 15.824 | 44.581 | 2.817× | 4.791 |
| MBPP | 15.857 | 43.655 | 2.753× | 4.509 |
| MT-Bench, first turn | 15.815 | 24.718 | 1.563× | 2.612 |
| **Overall, 80 requests** | **15.809** | **40.787** | **2.580×** | **4.347** |

DFlash emitted 34,493 timed tokens over 845.684 seconds; B1 emitted 35,376
over 2237.772 seconds. Proposed-token acceptance is 48.062%. DFlash's alternate
paper-era timer gives 41.026 tokens/s, and its equal-request average gives
43.303 tokens/s; the table uses complete-loop pooled throughput for both engines.
All recorded round positions, anchors, accepted prefixes and emitted-token
counts pass an accounting audit. The historical-output comparison establishes
preservation of the selected implementation on these 80 inputs; it is separate
from the HF BF16 proxy. B1 GEMV and B8 matrix arithmetic do not require identical
free-running output arrays (2 of 80 arrays match in this pair).

The historical selected DFlash run was 40.045 tokens/s with the same acceptance
traces; its B1 control was 19.310 tokens/s. The fresh B1 is slower, so the new
2.580× ratio must not be interpreted as an additional kernel optimization gain.
This compatibility study did not isolate the cause of the B1 performance gap.

### Full20 nightly integration, October 17

The default nightly paired study is now Full20; Full80 is optional. Full20
uses the first four fixed cases per dataset, with the same model revisions,
artifacts and timing as the October 16 run. The actual Full20 LIT passes in
830.53 seconds (13.8 minutes), including cached-build/resource checks and both
engine processes. B1 reaches **15.738 tokens/s**, DFlash **42.234 tokens/s**,
or **2.684×**, with **4.490 emitted tokens/round** and **50.130% acceptance**.
These values belong to the 20-request cohort, separate from Full80 above.

All 20 B1 outputs and all 20 DFlash outputs match their corresponding October
16 records. DFlash also matches all 1,870 non-timing round traces. Artifact
hashes, runtime sources, input identities, round accounting and aggregates pass
the audit. No kernel or model computation changed; the previous numerical
validation remains applicable. Workflow collection handles both cohort sizes
and partial request logs. Full80 opt-out and opt-in discovery are checked;
the optional full study was not rerun. This is local self-validation, with no
GitHub Actions execution yet.

### Preserved October 15 checkpoint

The October 15 validated checkpoint used upstream `cca4f090`, MLIR-AIE
`1.4.5.dev22+g35fa6b1` and Peano `22.0.0.2026100501+94b320c0`. On the
Ryzen AI 9 HX 370 / NPU2 it completed the following checks. This historical
evidence does not qualify later compiler or dependency changes:

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
the historical Full80 result in the README. No Full80 or fresh B1 comparison
was performed at that checkpoint. Tests were performed by the implementation
author; they are self-validation, not an independent review.
