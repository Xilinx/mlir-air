# Local qualification

## FIFO refeed compiler qualification

The current branch replaces the initial separate-channel allocator workaround
with FIFO BD-chain ordering from AIR PR #2152 (`51e0f801`), on upstream
`9748ba54`. MLIR-AIE, Peano, host MLIR and XRT versions remain unchanged.
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

The measured execution source is `4b6caef7`; subsequent changes only update
documentation. The full profile and Full20 LIT entries also pass with this
compiler.

## Full20 with the FIFO-chain compiler

First four cases from each dataset in the shared frozen seed-0 manifest,
greedy, thinking off, cap 2048/EOS, one pass per engine. MT-Bench uses its first
turn only. Both engines use identical freshly prepared NPU-prefill states.
Primary timing includes initial and incremental NPU prepass, draft, verify
and host work, excluding saved prefill, loading, warmup, reset and prefill's
first token.

| Dataset | B1 tok/s | DFlash tok/s | Speedup | Emitted/round |
|---|---:|---:|---:|---:|
| Overall | 9.711 | 26.036 | 2.681× | 4.514 |
| gsm8k | 9.735 | 31.797 | 3.266× | 5.411 |
| math500 | 9.692 | 34.862 | 3.597× | 6.183 |
| humaneval | 9.711 | 30.341 | 3.124× | 5.320 |
| mbpp | 9.730 | 28.489 | 2.928× | 4.812 |
| mt-bench | 9.712 | 12.560 | 1.293× | 2.158 |

B1 generated 9,599 timed tokens in 988.507 seconds; DFlash generated 9,321 in
357.999 seconds. Pooled throughput is total tokens / total decode time.
Candidate acceptance is 50.43%. All 40 request records passed cohort, source /
artifact identity, timing and acceptance-trace audits. Benchmark runtime hashes
match the NPU model-verification capture and the delivered source files.

All twenty freshly prepared prefill states match the original compiler's saved
states byte for byte. Both engines reproduce all twenty complete output
sequences, and all DFlash semantic acceptance traces match. The previous
Full20 result was 26.038 DFlash / 9.701 B1 tok/s (2.684×); DFlash throughput
changed by -0.006% in these single-pass measurements.

The B1 comparator uses the upstream C4096 xclbin/template runner, freshly
rebuilt after the KV-buffer helper update. Its KV readback covers allocated
capacity while RTP length bounds active attention; B8 uses live-context
transfers. Different B1 GEMV and B8 BFP arithmetic can produce different greedy
trajectories and lengths. These ratios characterize the supplied runners;
runtime-round xclbin and newer ELF/scratchpad B1 modes need separate measurements.

The paired processes took 23.7 minutes including loading. The twenty NPU-prefill
computations took 6.1 minutes; the complete Full20 LIT including preparation
and native B1 compilation took 31.6 minutes. The separate fixed-prompt profile
measured 13.377 decode tok/s; its workload differs from this cohort.

## Initial qualification

The initial qualification on October 21, 2026 used execution source
`04bc59309efd761171ebfe3d641f861499acc360`, based on upstream `7e9ec2b2`
plus the former separate-channel allocator workaround `1bd7e222`.
It ran on the same Ryzen AI 9 HX 370 NPU2 with 96 GB unified memory.
The historical Full80 and initial numerical checks below retain that source;
Full80 has not been rerun for the compiler migration.

## Full80 with the initial compiler

All sixteen cases per dataset from the same manifest, with the same settings
and timing boundary as Full20. The earlier twenty saved input states are
unchanged within the extended cohort.

| Dataset | B1 tok/s | DFlash tok/s | Speedup | Emitted/round |
|---|---:|---:|---:|---:|
| Overall | 9.704 | 25.251 | 2.602× | 4.365 |
| gsm8k | 9.725 | 28.309 | 2.911× | 4.817 |
| math500 | 9.687 | 31.988 | 3.302× | 5.615 |
| humaneval | 9.707 | 28.178 | 2.903× | 4.913 |
| mbpp | 9.722 | 27.169 | 2.795× | 4.576 |
| mt-bench | 9.701 | 15.541 | 1.602× | 2.684 |

B1 generated 35,034 timed tokens in 3610.449 seconds;
DFlash generated 34,610 in 1370.652 seconds.
Candidate acceptance was 48.32% overall.
The paired processes took 83.8 minutes
(60.6 B1 and 23.2 DFlash), excluding preparation.
The same fixed-C4096 B1 comparator qualification applies to this table.

All 200 request records across the two paired studies passed cohort, token-count,
timing-total and acceptance-trace checks. For each engine, all twenty overlapping
requests reproduced the complete output IDs and semantic acceptance traces
exactly. The median Full80/Full20 per-request throughput ratio was
0.9998 for B1 and 1.0000 for DFlash.
These replay checks complement the separate arithmetic and KV-state checks.

## Numerical and integration checks

- Complete compiler suite: 657 PASS, 7 XFAIL, 7 unsupported.
- Full target and drafter state checks at prefixes 0, 7, 16, 64, 1024, 4088;
  complete logits/hidden/KV repeatability, finite values, unchanged prefix/tail,
  target causality, drafter visibility and injected KV.
- Production-linked attention: 24 cases each for causal and bidirectional
  variants, with full-output SDPA tolerances, repeatability and row isolation.
- Context prepass: BFP arithmetic, row scales, mixed positions and partial /
  multi-chunk processing.
- Actual NPU-prefill DFlash loop: 32 tokens, repeatable output and acceptance,
  matching-B8 sequential oracle, committed target KV checked after every round.
- Upstream model gate: two 32-token captures, mutual top-5 PASS at first
  divergence positions 2 and 24, visiting 3 and 25 positions respectively.
  This does not compare the remaining divergent trajectories.
- Existing 4B: generated AIR/ABI equivalence, object/LLVM-IR comparison and
  fresh loop/prepass regression checks; five shared host tests passed.
- Fresh target/drafter packed weights match retained identities; all 253
  prefill tensors match. Incremental native B1 packing matches the old packer
  for a full layer and all vocabulary-head waves; B1 32-token repeats match.
- Public `make run` and two-prompt resident `make chat`: passed with NPU
  prefill, incremental stdout output and a 24-token budget per request.

See [validation](validation.md) for commands and numerical scope, including
prefill BFP exponent groups. All five new LIT entries are discovered; the
device tests were executed through their underlying commands on this host.

## CI alignment after measurement

The branch also includes upstream #2142's offline HF-reference correction for
4B and applies the same file-selection rule to 8B. A fresh 8B HF run with only
nightly-seeded files reproduced both 32-token captures exactly, including every
top-5 list. Comparison with the retained NPU captures still passes at the same
first divergences.

8B prefill now explicitly registers `air_examples` before its shared imports.
All prefill class/function definitions match the measured source, and a fresh
import check resolves the same native prefill and `KernelCache` classes.
The upstream `examples_sys_path.py` LIT and shared DFlash host LIT pass.
The existing `verify_loop.py` naming follows the correction in #2123.

These initial CI changes affect import setup and reference-file lookup; they
did not require an NPU or performance rerun. The later FIFO-chain qualification
and new Full20 measurements are recorded at the top of this document. GitHub nightly execution is separate from these local
checks and will run after the branch is submitted or explicitly dispatched.
