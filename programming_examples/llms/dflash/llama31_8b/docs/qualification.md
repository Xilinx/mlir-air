# Local qualification

## FIFO refeed compiler qualification

The retained Full20 measurement uses FIFO BD-chain ordering from AIR PR #2152
(`51e0f801`), on upstream `9748ba54`, replacing the initial separate-channel
allocator workaround. Both #2152 and the Qwen3-8B prerequisite #2145 are now
merged upstream. MLIR-AIE, Peano, host MLIR and XRT versions remain unchanged.
Fresh target, drafter, NPU prepass and prefill builds pass the actual compile
LIT. The actual verify LIT passes: six-prefix state/repeatability/visibility
checks, exact DFlash same-B8 oracle and per-round committed KV, and two
32-token NPU/HF captures with the upstream mutual top-5 comparison.

All recorded logits/hidden/KV hashes in the six-prefix target/drafter tests
match the retained pre-migration results. This is scoped to those calls; the
HF comparator checks the common prefix and mutual top-5 at first divergence.
The replay stream is O^8/down^8 on MemTile5 MM2S0. Its even/odd BD pools use
22/24 and 5/24 descriptors at AIR-to-AIE output, and the complete programs
compile and execute successfully. Compiler tests pass (657 MLIR, three C++),
as do both NPU refeed-order test modes with dependent and independent producers.
These are local self-validation results on Ryzen AI 9 HX 370 NPU2.

The measured execution source is `a433c4fc`; subsequent integration changes
update documentation only. Actual profile and Full20 LIT also pass. The
Full20 table below uses the new FIFO-chain compiler. All 20 fresh NPU-prefill
states match the original compiler's files byte for byte; both engines'
20 complete output sequences and all DFlash semantic acceptance traces match.
DFlash throughput is 24.166 tok/s, versus 24.175 before migration (-0.038%
in these single-pass measurements). The previous B1 result was 10.149 tok/s;
the new native upstream B1 measures 10.160 tok/s.

## Initial numerical qualification

Qualified on October 21, 2026, on a Ryzen AI 9 HX 370 NPU2 with 96 GB unified
memory, using the upstream dependency versions in [compatibility](compatibility.md). These are
local self-validation results. The initial preparation was stacked on Qwen3-8B PR #2145;
no additional AIR compiler change is needed for Llama. The initial execution
source was `812edd4f`, using the former separate-channel allocator workaround.

### Numerical and integration checks

The actual compile, verify and profile LIT entries passed on this host.
Qualification also exercised the following checks:

- Target and drafter at prefixes 0, 7, 16, 64, 1024 and 4088: complete repeated
  logits/hidden outputs and KV hashes, finite values, untouched prefix/tail,
  target causality, drafter bidirectional visibility and KV injection.
- Production-linked attention: 24 cases for each role against FP32 SDPA cast
  to BF16 (rtol 0.016, atol 0.1, zero allowed outliers), with repeats and row
  isolation.
- Residual/normalized-activation cache: seven input cases, three repeats each
  and eight row-isolation cases. All residual and replay words matched exactly;
  complete weighted RMS outputs passed rtol 0.016, atol 0.05.
- Context prepass: 40 input/row-count cases, with full-output BFP projection,
  normalization and RoPE comparisons, mixed positions, partial/multiple
  chunks, repeatability, padding and injection checks.
- Prefill weights: all 225 tensors matched the reference conversion bit for bit.
- Complete NPU-prefill DFlash loop: 32 output tokens, repeated outputs and
  acceptance decisions, exact same-B8 sequential-oracle output and committed
  target KV checked after each of 14 rounds.
- HF BF16 model gate: two complete 32-token captures per engine. AIR's mutual
  top-5 comparator passed at first divergences 1 and 11 (zero-based), visiting
  2 and 12 positions. Both alternative tokens ranked second in the other
  trace. Later divergent trajectories are outside that comparator's coverage.
- Native Llama weight preparation: streamed NPZ arrays matched the old packer
  exactly for a complete layer and all LM-head waves on both the untied 8B
  and tied 1B checkpoints. Cold B1 loading checked every uploaded NPZ word
  against its mapped weight buffer, then reproduced two 32-step decode traces
  exactly. Peak cgroup memory was 12.03 GiB in that cold-load check.
- Shared-code regression: Qwen3-8B AIR/ABI and machine objects match the
  published implementation; attention LLVM differs only in source labels.
  Its complete loop passed. Qwen3-4B's unchanged kernels passed the complete
  loop with the new shared host code; the shared host LIT passed.

The native B1 comparator uses GEMV, its existing normalization epsilon 1e-6,
and fixed-C4096 xclbin templates. The B8 target uses epsilon 1e-5 and batched
BFP arithmetic. Exact DFlash correctness is checked against matching B8
arithmetic; B1 supplies the existing autoregressive throughput comparison.
B1's template KV readback covers allocated capacity, while B8 uses live-context
transfers. These benchmarks characterize the supplied runners. The newer
native B1 ELF/scratchpad mode requires a separate comparison.

## Benchmark protocol

Full20 uses the first four cases per dataset from the shared frozen seed-0
cohort: GSM8K, MATH500, HumanEval, MBPP and first-turn MT-Bench. Both engines
receive identical saved NPU-prefill states and run one greedy pass, with EOS
stopping and a 2048-token cap. Primary decode timing includes initial and
incremental NPU prepass, draft, verify, acceptance and host work. Loading,
prefill, reset, warmup and the first token produced by prefill are excluded.

| Dataset | B1 tok/s | DFlash tok/s | Speedup | Emitted/round |
|---|---:|---:|---:|---:|
| Overall | 10.160 | 24.166 | 2.379× | 3.796 |
| gsm8k | 10.158 | 27.320 | 2.690× | 4.274 |
| math500 | 10.161 | 21.492 | 2.115× | 3.365 |
| humaneval | 10.161 | 30.551 | 3.007× | 4.926 |
| mbpp | 10.159 | 29.667 | 2.920× | 4.586 |
| mt-bench | 10.159 | 17.075 | 1.681× | 2.667 |

B1 generated 8,081 timed tokens in 795.390 seconds;
DFlash generated 7,684 in 317.965 seconds. Pooled throughput
is total timed tokens / total decode time. Candidate acceptance was
39.95%. Different B1/B8 arithmetic can produce different
output trajectories and EOS lengths; speedup compares their throughputs.

The paired processes took 19.7 minutes, including loading, warmup and
reset. The twenty initial NPU-prefill computations took 5.8
minutes in total, excluding one-time model loading and compilation; later
runs can reuse those states. The complete Full20 LIT including native B1
compilation and preparation took 27.3 minutes. These are single-pass local
measurements.

All 40 request records passed the cohort, token-count, timing-total and
acceptance-trace audit. All 37 DFlash runtime-source hashes in the benchmark
match the model-verification capture; both engines match the final runtime
source files. The initial public `make run` and two-prompt `make chat` checks passed
with incremental stdout and NPU prefill, using 24-token budgets. Actual
Full20 LIT execution passed. Full80 remains optional and has not been run
for this preparation.

The initial separate fixed-prompt profile (`Introduce me what is GPU`, 16 input tokens,
cap 256) measured 16.486 decode tokens/s and 17.099 seconds warm TTFT. This
profile includes NPU prefill in TTFT and uses a different workload from Full20.

See [validation](validation.md) for the LIT targets and reproduction commands.
GitHub nightly execution is separate from these local qualification runs.
