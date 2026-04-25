# Soren — Round 2 Review

## Triage Validation

I concur with the triage. The deferred status on my clone-then-strip suggestion (Finding 5) is fair — the decision depends on where the bottleneck actually lies (clone time vs downstream pass time).

### Strengthening the case for clone-then-strip

Merlin raises a valid point that `builder.clone()` creates SSA values that are immediately erased. But consider:

1. **MLIR's `Operation::erase()` is O(1) per op** (unlinks from block, frees memory). Walking + erasing N ops is O(N), same as skipping them during custom clone.

2. **`builder.clone()` is battle-tested infrastructure**. It handles all the edge cases the custom implementation gets wrong: block arguments, terminators, region nesting, operand mapping. The current implementation has at least 3 correctness bugs in these areas.

3. **The original problem is O(N*body_size) for the ENTIRE PIPELINE, not just the clone step**. The 1.9s pass time and 2.9MB IR size indicate that downstream passes (BD folding, canonicalization) operate on the bloated IR. Even if the clone is fast, the downstream cost remains until the bodies are stripped.

So clone-then-strip actually addresses the root cause: downstream passes see small herd bodies, even though the initial clone was O(N*body_size).

If profiling shows the clone step itself is the bottleneck (unlikely given 1.9s for trip count 48), then the custom approach is justified — but it needs to be correct first, which means essentially reimplementing `builder.clone()` with selective filtering. That's exactly what `cloneRegionLightweight` tries and fails to do.

## Verdict

**BLOCK MERGE** — concur with triage. I recommend the author adopt clone-then-strip as the implementation strategy for the next iteration.
