# Merlin — Round 2 Review

## Triage Validation

All three of my Critical findings (C1-C3) were upheld without rebuttal. I concur with the triage assessment.

### On Soren's clone-then-strip alternative (Finding 5)

I agree with the triage's deferred status. The clone-then-strip approach is architecturally cleaner and correct-by-construction. The custom cloning approach is only justified if profiling shows the `builder.clone()` call itself dominates the cost — not downstream pass processing.

However, I want to add nuance: `builder.clone()` for a large herd body creates all the SSA values, types, and attributes in memory even if they're immediately erased. For very large bodies (thousands of ops), this allocation/deallocation churn may matter. The ticket should profile both approaches before choosing.

### On the overall architecture

The decision to detect shim-level loops via `containsSegment` and route them to a different unrolling implementation creates a fork in the compilation path that must be maintained. Every future change to `loopUnrollFullWithAsyncTokenPreserved` must now consider both paths.

If the clone-then-strip approach achieves comparable performance, it would be preferable because it keeps a single unrolling path and adds body stripping as a clean post-processing step.

## Verdict

**BLOCK MERGE** — concur with triage. No fixes have been applied.
