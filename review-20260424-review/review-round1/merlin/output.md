# Merlin — Round 1 Review

**Change**: Lightweight loop unrolling to reduce O(N*body_size) IR explosion
**File**: `mlir/lib/Util/Dependency.cpp` (+329/-12 lines)

## Summary

This change adds a `loopUnrollFullLightweight` function that avoids deep-cloning segment/herd body internals during loop unrolling. Instead, it clones only the segment/herd shell + channel ops (via `cloneRegionLightweight`), skipping compute ops. The optimization targets shim-level loops (detected by `containsSegment`) and falls back to standard unrolling otherwise.

The architectural intent is sound: the O(N*body_size) explosion comes from repeatedly cloning large herd bodies that are irrelevant at the shim level. Keeping only channel ops preserves the information needed by downstream passes (air-to-std channel matching, BD folding).

## Findings

### C1 — Critical: `builder.create<air::SegmentOp>(loc, resultTypes, operands, attrs)` likely does not compile

**Location**: `Dependency.cpp:1179-1181`, `1210-1212`

```cpp
auto clonedSegment = builder.create<air::SegmentOp>(
    segmentOp.getLoc(), segmentOp.getResultTypes(), mappedOperands,
    segmentOp->getAttrs());
```

`SegmentOp` and `HerdOp` have `skipDefaultBuilders = 1` in their tablegen definition. The only available `build` methods are:

1. `build(builder, state, ValueRange sizes, ValueRange segment_operands)`
2. `build(builder, state, ValueRange asyncDeps, ValueRange sizes, ValueRange segmentOperands, bool isAsync=false, ArrayRef<NamedAttribute> attrs={})`

The call passes `(TypeRange, ValueRange, ArrayRef<NamedAttribute>)`. `TypeRange` is not implicitly convertible to `ValueRange` (different element types: `Type` vs `Value`). No available build method matches this signature.

Even if some C++ overload trick makes this compile, the flat `mappedOperands` list conflates async dependencies, sizes, and segment/herd operands. The custom builder expects these as separate `ValueRange` arguments and uses them to construct the body block arguments (2 per size + 1 per operand). The generic call doesn't provide this partitioning.

**Impact**: The code either fails to compile, or the cloned segment/herd has incorrect body block arguments (wrong count, wrong types), causing SSA violations in downstream passes.

**Fix**: Use the custom builder with properly partitioned operands:
```cpp
auto clonedSegment = air::SegmentOp::create(builder,
    segmentOp.getLoc(),
    mapper.lookup(segmentOp.getAsyncDependencies()),
    mapper.lookup(segmentOp.getSizeOperands()),
    mapper.lookup(segmentOp.getKernelOperands()),
    segmentOp.isAsync(),
    segmentOp->getAttrs());
```
Or use `builder.clone(segmentOp, mapper)` for the shell and then selectively strip the body.

---

### C2 — Critical: OpBuilder insertion point corruption across unroll iterations

**Location**: `Dependency.cpp:1197-1199`, `1228-1229`

```cpp
// In loopUnrollFullLightweight:
cloneRegionLightweight(segmentOp.getRegion(),
                       clonedSegment.getRegion(), mapper, builder,
                       /*insideHerd=*/false);
```

`cloneRegionLightweight` takes `OpBuilder &builder` by reference and immediately calls `builder.setInsertionPointToEnd(destBlock)` (line 981), moving the insertion point into the cloned segment's body block. After `cloneRegionLightweight` returns, `builder` permanently points inside the cloned segment body.

On the **next unroll iteration** (i > 0):
- `builder.create<arith::ConstantIndexOp>(...)` (line 1150) places the IV constant **inside** the previous iteration's segment body
- `builder.create<air::SegmentOp>(...)` (line 1179) creates the next segment **nested inside** the previous one
- SSA dominance violations: the IV constant is scoped inside one segment but referenced by the operands of a sibling segment

For the flash attention use case (trip count 48), this would produce 47 levels of nested segments — completely wrong IR.

**Fix**: Use a local builder for region cloning, or save/restore the insertion point:
```cpp
OpBuilder segBuilder(&clonedSegment.getRegion().front(),
                     clonedSegment.getRegion().front().begin());
cloneRegionLightweight(segmentOp.getRegion(),
                       clonedSegment.getRegion(), mapper, segBuilder,
                       /*insideHerd=*/false);
```

This is exactly what `cloneRegionLightweight` does internally when recursing into nested segments (line 1098-1099). The outer `loopUnrollFullLightweight` call should follow the same pattern.

---

### C3 — Critical: `cloneRegionLightweight` may produce ops-after-terminator

**Location**: `Dependency.cpp:980-981`

```cpp
Block *destBlock = &destRegion.front();
builder.setInsertionPointToEnd(destBlock);
```

Both `SegmentOp` and `HerdOp` have `SingleBlockImplicitTerminator`. Their custom builders call `ensureTerminator()`, creating a `SegmentTerminatorOp`/`HerdTerminatorOp` in the body block. `setInsertionPointToEnd` places the insertion point **after** the terminator.

If the custom builder is somehow reached (resolving C1), cloned ops would be placed after the terminator — invalid MLIR. Additionally, the source's terminator (listed in the clone filter at line 987-989) would be cloned, creating a duplicate.

**Fix**: Check for an existing terminator and insert before it:
```cpp
if (destBlock->mightHaveTerminator())
  builder.setInsertionPoint(destBlock->getTerminator());
else
  builder.setInsertionPointToEnd(destBlock);
```

---

### S1 — Suggestion: The `cloneRegionLightweight` function should handle `scf.parallel` and `scf.if` inside herds

**Location**: `Dependency.cpp:1029-1037`

Currently these emit warnings and skip the ops entirely. If channel ops exist inside `scf.if` branches (conditional DMA transfers) or `scf.parallel` bodies, they'd be silently lost. The warning-only approach is fragile — it's easy to miss warnings in a large build log.

Consider either:
- Implementing lightweight cloning for these (similar to `scf::ForOp`)
- Returning `failure()` instead of emitting a warning, to make the pass fall back to standard unrolling for these cases

---

### S2 — Suggestion: Multi-block regions silently merged

**Location**: `Dependency.cpp:979-980`

```cpp
for (Block &srcBlock : srcRegion) {
    Block *destBlock = &destRegion.front();
```

All source blocks are merged into a single destination block. While MLIR structured ops typically use single-block regions, this should at minimum assert or warn if multiple blocks are found.
