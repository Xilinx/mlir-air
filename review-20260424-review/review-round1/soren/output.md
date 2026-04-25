# Soren — Round 1 Review

**Change**: Lightweight loop unrolling to reduce O(N*body_size) IR explosion
**File**: `mlir/lib/Util/Dependency.cpp` (+329/-12 lines)

## Summary

The change adds 329 lines of custom loop unrolling code to avoid cloning large herd bodies at the shim level. The motivating problem is real — O(N*body_size) IR explosion for large trip counts is a measurable bottleneck. The question is whether this is the right layer and the right approach.

## Findings

### S1 — Suggestion: Consider `builder.clone()` + selective erasure instead of custom cloning

**Location**: `Dependency.cpp:968-1108` (entire `cloneRegionLightweight` function)

The 140-line `cloneRegionLightweight` function manually reconstructs segment/herd ops, maps block arguments, handles control flow inside herds, etc. This reimplements a significant portion of MLIR's region cloning infrastructure, with multiple correctness issues identified by other reviewers.

A simpler alternative:
1. Use `builder.clone(segmentOp, mapper)` to get a correct deep clone
2. Walk the cloned herd body and erase all non-essential ops

```cpp
Operation *clonedOp = builder.clone(segmentOp, mapper);
auto clonedSegment = cast<air::SegmentOp>(clonedOp);
clonedSegment.walk([](air::HerdOp herd) {
    // Erase non-essential ops from herd body
    SmallVector<Operation *> toErase;
    for (auto &op : herd.getBody()->without_terminator()) {
        if (!isa<air::ChannelInterface, memref::AllocOp, memref::DeallocOp,
                 air::WaitAllOp, scf::ForOp, scf::YieldOp>(&op))
            toErase.push_back(&op);
    }
    for (auto *op : llvm::reverse(toErase))
        op->erase();
});
```

This approach:
- **Correct by construction**: `builder.clone()` handles all the region/block/argument/terminator machinery correctly
- **Simpler**: ~20 lines instead of ~140
- **Maintainable**: Adding new op types to preserve is a one-line change, not a structural modification
- **Still achieves the goal**: The deep clone is O(N*body_size) but the erase is also O(body_size). However, the point is that the erased ops don't persist in memory through subsequent passes. If the concern is pass time (not memory), the erasure could happen before BD folding.

The tradeoff: the initial clone is still O(N*body_size) in construction time, but the resulting IR is small. If the bottleneck is BD folding (not the cloning itself), this simpler approach should achieve similar speedup.

If the bottleneck truly IS the cloning step itself, then the custom approach is justified — but it needs to be correct first.

---

### S2 — Suggestion: The `loopUnrollFullLightweight` function reimplements standard loop unrolling

**Location**: `Dependency.cpp:1113-1266` (154 lines)

This function manually implements loop unrolling: creating constants for the induction variable, threading iter_args, cloning the body, handling scf.yield. MLIR's `loopUnrollByFactor` already does all of this correctly with the `annotateFn` callback for customization.

Could the lightweight cloning be integrated as a callback or pre/post-processing step around `loopUnrollByFactor`?

For example:
1. Standard `loopUnrollByFactor(forOp, tripCount, annotateFn)` — correct unrolling
2. Walk unrolled segments, strip herd body internals — achieves the same lightweight result

This keeps the unrolling infrastructure correct and tested, and only adds the body-stripping as a follow-up pass.

---

### S3 — Suggestion: The `containsSegment` detection is a coarse heuristic

**Location**: `Dependency.cpp:1293-1294`

```cpp
bool containsSegment = false;
forOp.walk([&](air::SegmentOp) { containsSegment = true; });
```

This walks the entire subtree and uses the presence of *any* segment as the trigger. A more precise heuristic would check whether the segment body is "large enough" to warrant lightweight cloning. For a segment with a small herd body (e.g., a single channel op), the overhead of custom cloning may exceed that of standard cloning.

That said, this is a minor point — the heuristic is safe (it's always correct to use lightweight cloning, assuming the implementation is correct).

---

### N1 — Nit: 329 lines of new code without any test

A change of this scope, especially one reimplementing core infrastructure, needs test coverage proportional to its complexity. The ratio of 329 lines of implementation to 0 lines of test is concerning.
