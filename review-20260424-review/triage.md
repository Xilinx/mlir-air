# Triage — Round 1 Findings

## Critical Findings (Blocks merge)

### Finding 1: Incorrect SegmentOp/HerdOp construction (Merlin-C1, Irene-C1)

**Status**: UPHELD — Must fix before merge

Both Merlin and Irene independently identified that `builder.create<air::SegmentOp>(loc, resultTypes, operands, attrs)` uses a signature that doesn't match any available `build` method. `SegmentOp` and `HerdOp` have `skipDefaultBuilders = 1`, and their custom builders expect separate `ValueRange` arguments for async deps, sizes, and kernel operands — not a single flat operand list.

The generated header (`AIR.h.inc`) confirms: no generic `(TypeRange, ValueRange, ArrayRef<NamedAttribute>)` builder exists for these ops.

**Fix required**: Rewrite segment/herd construction to use the custom builder with properly partitioned operands, or use `builder.clone()` followed by selective body stripping.

---

### Finding 2: OpBuilder insertion point corruption (Merlin-C2)

**Status**: UPHELD — Must fix before merge

`cloneRegionLightweight` modifies the shared `builder`'s insertion point via `builder.setInsertionPointToEnd(destBlock)`. After the function returns, the builder points inside the cloned segment's body. Subsequent unroll iterations create ops at the wrong scope, producing nested segments.

The function's own recursive calls use local builders (lines 1098-1099), demonstrating the correct pattern. The `loopUnrollFullLightweight` caller should do the same.

**Fix required**: Use a local `OpBuilder` for the `cloneRegionLightweight` call in `loopUnrollFullLightweight`.

---

### Finding 3: ForOp body cloning produces ops-after-terminator (Merlin-C3, Irene-C2)

**Status**: UPHELD — Must fix before merge

`scf::ForOp::build` creates a body with a `scf::YieldOp` terminator. `cloneRegionLightweight` uses `setInsertionPointToEnd`, placing cloned ops after this yield. The source's yield is also cloned, producing a duplicate.

**Fix required**: Erase the auto-generated yield before cloning, or insert before the terminator.

---

### Finding 4: Zero test coverage for the lightweight path (Vera-C1)

**Status**: UPHELD — Must fix before merge

The existing `opt_shim_dma_bds.mlir` test has no segments in its loop bodies, so `containsSegment` is always false. No new test was added. 329 lines of new code with 0 lines of test coverage.

**Fix required**: Add at least one MLIR LIT test with a `scf.for` loop containing `air.segment` > `air.herd` with channel ops, trip count >= 2.

---

## Suggestions (Should address)

### Finding 5: Consider clone-then-strip approach (Soren-S1, Soren-S2)

**Status**: DEFERRED to author — worth considering

Soren proposes using `builder.clone()` for correct deep cloning followed by walking the herd body to erase non-essential ops. This is correct-by-construction and ~20 lines vs ~300. The tradeoff is that the initial clone is still O(N*body_size), but the resulting IR is small.

**Rebuttal**: If the bottleneck is the clone step itself (memory allocation, SSA construction), the clone-then-strip approach doesn't help. However, if the bottleneck is downstream passes processing the large IR, then strip-after-clone is equally effective and much simpler. The ticket should clarify which phase dominates.

### Finding 6: scf.parallel/scf.if inside herds silently lose channel ops (Merlin-S1)

**Status**: Should address

Warnings are emitted but channel ops inside these constructs are lost. Consider returning `failure()` to fall back to standard unrolling.

### Finding 7: Code duplication (Irene-S1)

**Status**: Should address if the custom cloning approach is kept. Moot if clone-then-strip is adopted.

### Finding 8: annotateFn receives corrupted builder (Irene-S2)

**Status**: Subsumed by Finding 2 fix.

---

## Nits (Non-blocking)

- Irene-N1: Redundant bounds checking (tripCount + lb/ub/step constant checks)
- Irene-N2: Intermediate variables for bounds can be inlined
- Soren-N1: 329 lines without test (same as Vera-C1)

---

## Actions Taken

No fixes applied — all Critical findings require fundamental changes to the approach. The code likely does not compile in its current form (Finding 1). The overall design needs revision before individual fixes are meaningful.

## Recommendation

**BLOCK MERGE**. The change has 4 Critical findings, at least one of which (Finding 1) likely prevents compilation. The remaining Critical findings (builder corruption, terminator handling, test coverage) affect correctness for all trip counts > 1.

The author should consider Soren's clone-then-strip alternative, which avoids all three implementation bugs (Findings 1-3) by delegating to MLIR's existing correct cloning infrastructure.
