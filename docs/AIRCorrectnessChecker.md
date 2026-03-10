# AIR Program Correctness Checker — Implementation Plan

This document specifies the design and implementation plan for a static
correctness checker for AIR programs. The checker verifies four properties
that together guarantee a program is free of resource oversubscription,
deadlock, and token constraint contradictions. It is intended to run as part
of the compiler pipeline, after the `-air-dependency` pass has materialised
async token edges and before lowering to a backend.

---

## 1. Properties verified

| ID | Property | Precondition |
|----|----------|-------------|
| P1 | **Channel balance** — every `put` is matched by a `get` on every execution path | None |
| P2 | **Deadlock freedom** — no circular blocking in the async dependency or channel graph | P1 |
| P3 | **Resource constraint satisfaction** — the program's concurrent resource footprint fits within hardware capacity | P2 |
| P4 | **Token constraint consistency** — affinity and concurrency constraints are mutually satisfiable | P3 |

P1 must be verified before P2 because an unbalanced channel is a special case
of deadlock; confirming balance first simplifies the deadlock analysis. P3
depends on a deadlock-free graph so that resource liveness intervals are
well-defined. P4 uses the resource bounds computed by P3.

---

## 2. Architecture

```
air-dependency pass          (existing — materialises !air.token edges)
        │
        ▼
AIRAsyncDependencyGraph      (new Analysis — the central data structure)
        │
   ┌────┴────┬──────────┬──────────┐
   ▼         ▼          ▼          ▼
  P1        P2         P3         P4
Channel   Deadlock   Resource   Token
Balance   Freedom    Bounds     Consistency
   │         │          │          │
   └────┬────┴──────────┴──────────┘
        ▼
   AIRVerifyProgram pass     (new umbrella pass — runs all checks,
                               reports diagnostics, fails on error)
```

All four checkers are implemented as MLIR analyses (not transformation passes)
so they can be composed, cached, and invalidated independently. The umbrella
pass `AIRVerifyProgram` drives them in dependency order and collects
diagnostics.

**New files:**

```
mlir/include/air/Analysis/
    AsyncDependencyGraph.h
    ChannelBalance.h
    DeadlockAnalysis.h
    ResourceBounds.h
    TokenConstraints.h

mlir/lib/Analysis/
    AsyncDependencyGraph.cpp
    ChannelBalance.cpp
    DeadlockAnalysis.cpp
    ResourceBounds.cpp
    TokenConstraints.cpp
    CMakeLists.txt

mlir/lib/Transform/
    AIRVerifyProgram.cpp      (umbrella pass, registered as -air-verify-program)

test/Analysis/
    channel-balance.mlir
    deadlock.mlir
    resource-bounds.mlir
    token-constraints.mlir
```

---

## 3. Core data structure: `AIRAsyncDependencyGraph`

### 3.1 Node taxonomy

Every async `Operation*` in the program is a node. The node set includes:

- Hierarchy ops: `air.launch`, `air.segment`, `air.herd`
- Data movement: `air.dma_memcpy_nd`, `air.channel.put`, `air.channel.get`
- Sync ops: `air.execute`, `air.wait_all`, `air.token.alloc`

Each node carries:
- `Operation *op` — the MLIR operation
- `ResourceFootprint footprint` — `{tiles, l2_bytes, dma_channels}` (populated by P3)
- `ScopeKind scope` — which loop/region nesting level the op lives in

### 3.2 Edge taxonomy

| Kind | Direction | Source | Sink | Meaning |
|------|-----------|--------|------|---------|
| `Dependency` | A → B | token producer | token consumer in `dependency` list | B does not begin until A signals |
| `Affinity` | undirected group | — | — | members share hardware partition, execute sequentially |
| `Concurrency` | undirected group | — | — | members must have overlapping execution lifetimes |
| `Channel` | put → get | `channel.put` | matching `channel.get` | put blocks until get consumes |
| `StructuralBack` | yield → iter_arg | `scf.yield` token | next-iteration consumer | loop-carried dep; excluded from cycle checks |
| `ParallelInit` | init → instance | `scf.parallel` init token | each instance op | broadcast dependency |
| `ParallelReduce` | instance → result | each instance token | `scf.reduce` result | tree-reduction; excluded from cycle checks |

Affinity and concurrency groups are stored as hyperedges (sets of node IDs)
keyed by the `!air.token` value used in the respective attribute list.

### 3.3 Building the graph

The builder walks the module in program order. For each `!air.token` SSA value:

1. Find the defining op → source node.
2. Walk all uses of the value:
   - If used in a `dependency` attribute list of op B → add `Dependency` edge A→B.
   - If used in an `affinity` attribute list of op B → add B to the affinity
     group keyed by this value.
   - If used in a `concurrency` attribute list of op B → add B to the
     concurrency group keyed by this value.
3. For `scf.for` iter_args: identify the `scf.yield` operand that corresponds
   to each iter_arg block argument. Add a `StructuralBack` edge from the yield
   operand's defining op to the first consumer of the iter_arg in the loop
   body.
4. For `scf.parallel`: add `ParallelInit` edges from the init token's defining
   op to all ops in the parallel body that consume it. Add `ParallelReduce`
   edges from the `scf.reduce` operand's defining op to the parallel result.
5. For channels: after all ops are collected, match each `channel.put` to its
   corresponding `channel.get` by `(channel_name, indices)` and add a `Channel`
   edge.

The builder reuses `DirectedAdjacencyMap` (already in
`mlir/lib/Util/DirectedAdjacencyMap.cpp`) for the directed dependency and
channel edges. Affinity and concurrency groups are stored as
`DenseMap<Value, SmallVector<NodeId>>`.

### 3.4 Scope decomposition

For resource analysis the graph must be queryable per scope (segment body, loop
body, parallel region). The builder tags each node with its enclosing scope
chain. A helper `nodesInScope(Region *)` returns all nodes whose immediate
enclosing region is the given one.

---

## 4. P1 — Channel balance

### 4.1 Goal

For each channel index, along every possible execution path, the number of
`put` operations equals the number of `get` operations.

### 4.2 Lattice

```
BalanceLattice = map< (StringAttr channel_name, SmallVector<Value> indices),
                      pair<int64_t put_count, int64_t get_count> >
```

The lattice is a `DenseMap`. A channel entry is **balanced** when
`put_count == get_count`. An **unbalanced** entry is an error.

### 4.3 Transfer functions

Implement as a forward dataflow analysis (`mlir::dataflow::DataFlowAnalysis`):

- `air.channel.put @C[i,j]` → increment `put_count` for key `(@C, [i,j])`
- `air.channel.get @C[i,j]` → increment `get_count` for key `(@C, [i,j])`
- `scf.for` body: assert per-iteration balance before exiting the loop. The
  loop contributes the per-iteration `(put_count, get_count)` to the enclosing
  scope — an unequal pair is a compile error regardless of trip count.
- `scf.parallel` body: assert per-instance balance. The instances are
  symmetric; the enclosing scope sees the per-instance count multiplied by the
  instance count (but imbalance is caught at instance level first).
- `scf.if` / conditional branches: run the analysis independently on each
  branch; assert balance holds on both. The merge point takes the (equal)
  counts from either branch.

### 4.4 Error reporting

```
error: channel '@C' has 2 puts but 1 get on some execution path
note: put at <loc>
note: put at <loc>
note: get at <loc>
```

---

## 5. P2 — Deadlock freedom

### 5.1 P2a — Dependency cycle detection

**Input**: the dependency-edge subgraph of the ADG, with `StructuralBack` and
`ParallelReduce` edges excluded.

**Algorithm**: Tarjan's SCC (available via `llvm::scc_iterator` on any graph
that exposes `GraphTraits`). Any SCC with more than one node, or a single node
with a self-loop, is a deadlock.

Implement `GraphTraits<AIRAsyncDependencyGraph *>` specialisation over
dependency edges (excluding structural back-edges) so that LLVM's existing SCC
infrastructure applies directly.

**Error reporting**:

```
error: dependency cycle detected — the following operations form a deadlock:
note: %t0 = air.herd ... (depends on %t1)
note: %t1 = air.segment ... (depends on %t0)
```

### 5.2 P2b — Channel blocking cycle detection

**Input**: the full ADG including `Channel` edges (put → get), but still
excluding `StructuralBack` edges.

**Algorithm**: Re-run Tarjan's SCC on this extended graph. A cycle that
traverses at least one `Channel` edge is a channel deadlock. A cycle composed
entirely of `Dependency` edges would have been caught by P2a.

The minimal deadlock case from the model (sequential put then get, depth 1)
creates a cycle: the `put` node has a `Channel` edge to the `get` node, and the
sequential control-flow ordering in the ADG provides an implicit
`Dependency`-type path from `get` back to `put` (since `get` follows `put` in
program order within the same synchronous scope).

**Error reporting**:

```
error: channel deadlock — circular blocking between:
note: air.channel.put @C (blocks waiting for get)
note: air.channel.get @C (unreachable while put is blocking)
```

### 5.3 Handling loop back-edges

`scf.for` iter_arg back-edges (`StructuralBack`) are excluded from both P2a
and P2b cycle checks. They are structural, not deadlock-forming: iteration N
signals its token, which unblocks iteration N+1. The within-iteration subgraph
(ignoring back-edges) must be acyclic, which P2a/P2b verify.

`scf.parallel` init and reduce edges are also excluded: init is a broadcast
(no cycle possible from init→instance), and reduce is always a convergent tree
(DAG by construction).

---

## 6. P3 — Resource constraint satisfaction

### 6.1 Resource footprint type

```cpp
struct ResourceFootprint {
  int64_t tiles;       // total tile-columns × tile-rows reserved
  int64_t l2_bytes;    // bytes in address space 1 simultaneously live
  int64_t dma_channels; // concurrent active DMA/channel ops
};

ResourceFootprint operator+(ResourceFootprint, ResourceFootprint); // sum
ResourceFootprint max(ResourceFootprint, ResourceFootprint);       // per-field max
```

### 6.2 Concurrent resource set computation

For a given scope (MLIR `Region *`), compute the maximum concurrent resource
footprint by finding the maximum antichain of async nodes in that scope's ADG
subgraph.

Using the transitive closure already computed by `DirectedAdjacencyMap::getClosure()`:
- Two nodes A and B are **concurrent** if neither `closure[A][B]` nor
  `closure[B][A]` is true.
- The maximum antichain is the maximum set of mutually concurrent nodes.
- The concurrent footprint for the scope is the sum of footprints of all nodes
  in the maximum antichain.

For practical performance, Dilworth's theorem equates the maximum antichain
size to the minimum path-cover size (computable via bipartite matching). For
typical AIR programs (tens to hundreds of async ops per segment body) this is
fast enough. A greedy approximation (traverse topologically; greedily extend
the current active set) is acceptable as a conservative over-approximation if
needed.

### 6.3 `scf.for` pipeline depth

For each `scf.for` loop in the scope, compute the pipeline depth D separately
per resource type:

1. Collect all `StructuralBack` edges entering this loop (the iter_args).
2. For each resource type (tiles, L2, DMA), identify the subset of iter_args
   whose token chain controls that resource:
   - A tile iter_arg is one whose token chain passes through an `air.herd` op.
   - An L2 iter_arg is one whose chain passes through `air.execute` containing
     `memref.alloc` in address space 1.
   - A DMA iter_arg is one whose chain passes through `air.dma_memcpy_nd` or
     `air.channel.put/get`.
3. D for each resource type = number of independent iter_args controlling that
   resource type.
4. Per-iteration resource bound × D = loop's contribution to the enclosing
   scope's concurrent footprint.

### 6.4 `scf.parallel` aggregation

All instances are concurrent. The parallel region's footprint is the
per-instance footprint × total instance count (product of all dimension
ranges).

### 6.5 Bottom-up recursive analysis

The analysis visits scopes innermost-first:

```
function analyseScope(Region *scope) -> ResourceFootprint:
  footprint = zero
  for each op in scope:
    if op is air.herd:
      herd_fp = {Nx * Ny tiles, 0 L2, 0 DMA}
      update concurrent-set with herd_fp
    elif op is air.segment:
      inner_fp = analyseScope(op.body)       // recurse
      inner_fp *= op.iterationSpaceSize()    // Stage 2
      update concurrent-set with inner_fp
    elif op is scf.for:
      loop_fp = analyseScope(op.body)
      loop_fp *= pipelineDepth(op)
      update concurrent-set with loop_fp
    elif op is scf.parallel:
      par_fp = analyseScope(op.body)
      par_fp *= instanceCount(op)
      update concurrent-set with par_fp
    elif op is memref.alloc (address space 1):
      l2_fp = {0, alloc.size, 0}
      update concurrent-set with l2_fp
    elif op is air.dma_memcpy_nd or air.channel.put/get:
      dma_fp = {0, 0, 1}
      update concurrent-set with dma_fp
  return maxAntichainFootprint(concurrent-set, scope-ADG)
```

### 6.6 Stage 2 and launch-level aggregation

At the launch level, sum the `analyseScope` results for all `air.segment` ops
directly in the launch body (they are co-resident by the co-residency
guarantee):

```
total = sum of analyseScope(seg.body) * seg.iterationSpaceSize()
        for each seg in launch.body
```

### 6.7 Device capacity model

The device capacity is read from a target attribute on the module or passed as
a pass option:

```
struct DeviceCapacity {
  int64_t total_tiles;
  int64_t total_l2_bytes;
  int64_t total_dma_channels;
};
```

Verification fails if `total.tiles > capacity.total_tiles`, etc.

**Error reporting**:

```
error: segment '@foo' requires 48 compute tiles across all iterations,
       but target has only 32
note: per-instance requirement: 6 tiles
note: iteration space: 8 instances
```

---

## 7. P4 — Token constraint consistency

### 7.1 Affinity satisfiability

For each affinity group (set of ops sharing an affinity token):

1. Verify no two members also share a `concurrency` token with each other.
   Affinity → sequential; concurrency → overlapping. The combination is
   contradictory.
2. Verify that the hardware partition implied by placement attributes
   (`x_loc`, `y_loc`, `x_size`, `y_size`) on all members is consistent —
   i.e., they can all be placed on the same physical region sequentially.
3. Verify that the resource footprint of the largest member fits within the
   shared partition.

### 7.2 Concurrency satisfiability

For each concurrency group (set of ops sharing a concurrency token):

1. Verify no two members also share a `dependency` edge with each other in the
   same direction. Concurrency → overlapping lifetimes; dependency → disjoint
   execution. The combination is only valid if the dependency edge connects an
   earlier phase of op A to a later phase of op B (e.g., A's prefetch depends
   on B's previous iteration), not if one op transitively depends on the other
   within the same iteration.
2. Sum the resource footprints of all members. Verify the sum fits within
   device capacity (cross-check with P3).

### 7.3 Contradiction detection

```
error: affinity token '%ta' and concurrency token '%tc' both constrain
       operations 'air.segment @A' and 'air.segment @B', which is contradictory:
       affinity requires sequential execution on the same resource;
       concurrency requires overlapping execution lifetimes
note: affinity token defined at <loc>
note: concurrency token defined at <loc>
```

---

## 8. Pass registration and options

The umbrella pass `AIRVerifyProgram` is registered as `-air-verify-program`
and accepts options to enable/disable individual checks:

```
mlir-opt --air-verify-program="checks=p1,p2,p3,p4 target-tiles=32
          target-l2-bytes=524288 target-dma-channels=16" input.mlir
```

Default: all checks enabled. If any check fails the pass emits diagnostics and
signals failure, causing the pipeline to abort.

The pass itself does not modify the IR. It runs the four analyses in order,
collects diagnostics via the MLIR `DiagnosticEngine`, and returns failure if
any error was emitted.

---

## 9. Implementation roadmap

### Phase 1 — ADG builder and P1 (channel balance)

Deliverables:
- `AsyncDependencyGraph.h/.cpp` — node/edge types, builder, `GraphTraits`
  specialisation, scope decomposition helper
- `ChannelBalance.h/.cpp` — forward dataflow analysis for P1
- `AIRVerifyProgram.cpp` — umbrella pass skeleton running P1 only
- Tests: `test/Analysis/channel-balance.mlir` covering balanced loops,
  unbalanced branches, scf.parallel per-instance balance

Acceptance: the pass correctly identifies unbalanced channels in the existing
test suite without false positives.

### Phase 2 — P2 (deadlock freedom)

Deliverables:
- `DeadlockAnalysis.h/.cpp` — P2a (dependency SCC) and P2b (channel-extended
  SCC), `GraphTraits` adapter that excludes structural back-edges
- Tests: `test/Analysis/deadlock.mlir` covering the minimal sequential
  put-then-get case, a token cycle, a legal pipelined loop (must not trigger)

Acceptance: the minimal deadlock example from §2.5 of `AIRComputeModel.md` is
flagged; all existing `-air-dependency`-produced programs pass without false
positives.

### Phase 3 — P3 (resource bounds)

Deliverables:
- `ResourceBounds.h/.cpp` — bottom-up recursive analysis, pipeline depth
  computation, device capacity model, `DeviceCapacity` struct populated from
  target attributes
- Tests: `test/Analysis/resource-bounds.mlir` covering a segment that fits,
  a segment that exceeds tile count, a double-buffered loop with D=2 correctly
  multiplied

Acceptance: the existing `AIRHerdPlacementPass` and `AIRHerdAssignPass` can be
informed by the `ResourceFootprint` computed here rather than recomputing
heuristically.

### Phase 4 — P4 (token constraint consistency)

Deliverables:
- `TokenConstraints.h/.cpp` — affinity/concurrency group extraction and
  contradiction checks
- Tests: `test/Analysis/token-constraints.mlir` covering affinity+concurrency
  contradiction, valid affinity-only group, valid concurrency group within
  resource bounds

Acceptance: the four checks compose cleanly under `-air-verify-program` and
the pass is added to the default NPU and GPU compilation pipelines after
`-air-dependency`.

---

## 10. Test strategy

Each phase produces both **positive tests** (correct programs that must pass)
and **negative tests** (deliberately invalid programs that must produce
specific diagnostics). The negative tests use the MLIR `// expected-error`
annotation scheme so that `mlir-opt --verify-diagnostics` validates both the
presence and the text of every error message.

Positive tests draw from the existing `test/airhost/` programs (tests 46–51)
which use the full scf.for iter_arg pipelining pattern verified by the model.
These must all pass P1–P4 without false positives.

The device capacity model for tests uses a small synthetic target
(`total_tiles=16, total_l2_bytes=65536, total_dma_channels=4`) to make
resource bound violations easy to construct without requiring real device
parameters.
