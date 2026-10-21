# Local qualification

Qualified on October 21, 2026, on a Ryzen AI 9 HX 370 NPU2 with 96 GB unified
memory. The execution source is `04bc59309efd761171ebfe3d641f861499acc360`, based
on upstream `7e9ec2b2` and the compiler change described in [compatibility](compatibility.md).
These are local self-validation results; no independent reviewer run is implied.

## Full20

First four cases from each dataset in the shared frozen seed-0 manifest,
greedy, thinking off, cap 2048/EOS, one pass per engine. Both engines use
identical saved NPU-prefill states. Primary timing includes initial and
incremental NPU prepass, draft, verify and host work, excluding saved prefill,
loading, warmup, reset and the first token from prefill.

| Dataset | B1 tok/s | DFlash tok/s | Speedup | Emitted/round |
|---|---:|---:|---:|---:|
| Overall | 9.701 | 26.038 | 2.684× | 4.514 |
| gsm8k | 9.721 | 31.731 | 3.264× | 5.411 |
| math500 | 9.684 | 34.886 | 3.602× | 6.183 |
| humaneval | 9.703 | 30.335 | 3.126× | 5.320 |
| mbpp | 9.718 | 28.487 | 2.931× | 4.812 |
| mt-bench | 9.703 | 12.567 | 1.295× | 2.158 |

B1 generated 9,599 timed tokens in 989.473 seconds; DFlash generated 9,321 in
357.977 seconds. Pooled throughput is total tokens / total decode time.
Different arithmetic in B1 GEMV and B8 BFP can produce different greedy
trajectories and lengths; speedup compares their measured throughputs.
Acceptance among proposed candidates is 50.43% overall.

The B1 comparator follows the existing 4B DFlash study's C4096 xclbin/template
convention. Its KV readback covers the allocated capacity while the RTP length
bounds active attention; B8 uses live-context transfers. The ratios therefore
characterize these supplied runners. Upstream's runtime-round xclbin and newer
ELF/scratchpad B1 modes require separate paired measurements.

The warm paired processes took 23.0 minutes, including loading. Preparing the
20 initial NPU-prefill states took 6.1 minutes; subsequent runs reused them.
The separate short GPU-explanation profile measured 13.386 tok/s and
17.868 seconds TTFT. It is a different prompt/workload from this table.

## Full80

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
