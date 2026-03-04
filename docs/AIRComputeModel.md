# AIR Compute Model

This document describes the compute and memory model of the AIR dialect, defines the
semantics of its operations, and explains how those operations map onto the two supported
hardware backends: AMD NPU (AIE) and AMD GPU (ROCDL/HIP).

The AIR dialect is designed to be architecture-independent at the programming-model
level. A program expressed in terms of `air.launch`, `air.segment`, `air.herd`, and the
associated data-movement operations carries a well-defined meaning that can be compiled
to either backend without changing the source IR.

A detailed description of the dialect and its use for mapping AI workloads onto AMD NPUs
can be found in:

> E. Wang et al. "[From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with
> MLIR-AIR](https://arxiv.org/abs/2510.14871)". arXiv:2510.14871, October 2025.

---

## 1. Abstract Compute Model

### 1.1 Hierarchy

The AIR model defines a three-level execution hierarchy:

```
air.launch          — outermost level; groups co-resident work
  air.segment       — intermediate level; groups cores with shared on-device memory
    air.herd        — innermost level; an array of processing elements (PEs)
```

Each level defines an **iteration space** (a multi-dimensional index range) and a **body
region** that executes across that iteration space. The bodies of `launch` and `segment`
execute at the _host/controller_ level, while the body of `herd` executes in _data-
parallel_ fashion across each point of its iteration space.

Execution at each level is **spatially parallel**: points in the iteration space may
execute concurrently on distinct hardware resources.

Operands from the enclosing scope are passed explicitly as **kernel operands** into each
level. Levels are `IsolatedFromAbove`, meaning they cannot implicitly capture values from
outer regions.

### 1.2 Memory Hierarchy

The AIR model defines three levels of memory, identified by the `air::MemorySpace` enum:

| Level | Enum value | Scope | Description |
|-------|-----------|-------|-------------|
| `L3`  | 0 | System | Off-chip or host-accessible memory (e.g. DDR, host RAM) |
| `L2`  | 1 | Segment | On-device memory shared across all PEs in a segment (e.g. on-chip SRAM, memory tiles) |
| `L1`  | 2 | Herd tile | Per-PE scratchpad memory local to each processing element |

Memory at each level must be accessed through the appropriate data-movement operations.
Direct reads or writes across levels (e.g., from L1 directly into L3) are not supported;
all cross-level data movement is explicit.

### 1.3 Data Movement

Data movement between memory levels is expressed with:

- **`air.dma_memcpy_nd`** — an N-dimensional DMA-style bulk copy between two memrefs.
  The source and destination may reside at any combination of memory levels.
- **`air.channel` / `air.channel.put` / `air.channel.get`** — point-to-point or
  broadcast communication channels that decouple producers from consumers.

### 1.4 Asynchrony

All data-movement and hierarchy operations may execute **asynchronously**. An op that
produces an `!air.token` result is dispatched immediately and executes concurrently
with subsequent operations. Token values are used to express data, control, affinity,
and concurrency constraints (see §1.5). `air.wait_all` merges multiple tokens into a
single synchronization point. `air.execute` wraps sequential computation in an
asynchronous region.

### 1.5 Token taxonomy and operation lifetimes

#### Operation lifetimes

Every AIR hierarchy operation has two distinct lifetimes:

- **Resource lifespan** — the interval during which the physical resources
  assigned to the operation (compute tiles, memory banks, DMA channels) are
  reserved and unavailable to other operations.
- **Execution lifespan** — the interval during which the nested operations
  within the body are actively executing.

The execution lifespan must be a **subset** of the resource lifespan: resources
must be allocated before execution begins and must remain allocated until
execution completes.

#### Token type

All token kinds share a single MLIR type: **`!air.token`**. The kind is
determined by which attribute list the token appears in at a consumer op, not
by the token's type. `!air.token` is the replacement for `air.async.token`;
existing uses of `air.async.token` are dependency tokens.

#### Token creation

A token is created in one of two ways:

**1. Explicit allocation:**
```
%t = air.token.alloc : !air.token
```
Used to introduce a token that groups a set of ops under a shared affinity or
concurrency constraint without one op "owning" the token.

**2. Op result:**
```
%t = air.segment @foo { … }
%t = air.herd tile (%x,%y) in (%sx=%Nx,%sy=%Ny) [dependency = [%dep]] { … }
```
The token becomes signaled when the op and all of its nested operations
complete. The result is optional; omitting it gives the synchronous (blocking)
form of the op.

#### Token consumption — the three attribute lists

Tokens are consumed by including them in named attribute lists on hierarchy and
data-movement ops. An op may carry any combination of the three lists:

```
air.segment @bar [dependency = [%t0, %t1]]
                 [affinity   = [%ta]]
                 [concurrency = [%tc]]
                 { … }
```

| Attribute list | Constraint imposed | Effect on resource lifespan | Effect on execution lifespan |
|----------------|-------------------|----------------------------|------------------------------|
| `dependency`   | Op does not begin until all listed tokens are signaled (happens-before) | Compiler/runtime may reuse resources freely across the edge | Disjoint from the producers of the listed tokens |
| `affinity`     | Op is placed on the same hardware resources as all other ops that list the same token | Disjoint — shared resources used sequentially | Disjoint — operations execute at different times |
| `concurrency`  | Op must have an overlapping execution lifetime with all other ops that list the same token | Overlapping — all resource sets must be live simultaneously | Overlapping |

- **`dependency` list**: The direct successor of `air.async.token` dependency
  chains. The op waits for all listed tokens to be signaled before acquiring
  resources or beginning execution. The compiler may reuse the releasing op's
  resources for the waiting op once the token fires.

- **`affinity` list**: All ops that share the same `!air.token` in their
  `affinity` list are bound to the same hardware partition (e.g. the same set
  of tile columns on NPU, or the same CU on GPU). Because they share resources
  they must execute sequentially; both their resource and execution lifespans
  are disjoint in time.

- **`concurrency` list**: All ops sharing the same token in their `concurrency`
  list are required to have overlapping execution lifetimes. The compiler must
  allocate distinct, simultaneously live resource sets for each such op. This
  is the mechanism that enforces co-residency (e.g. ensuring two segments are
  stamped out on hardware at the same time).

#### Scope rule

Tokens are SSA values and follow normal MLIR dominance. When a token must be
passed into a hierarchy body it is threaded through the `args(…)` operand list.
The sole exception to free token passing is `air.launch`: only tokens used in a
`dependency` list may be passed into the body of a launch. Affinity and
concurrency tokens may not cross the launch boundary, because imposing shared-
resource or forced-overlap constraints on the may-be-parallel instances of a
launch would contradict its execution model.

---

## 2. Operation Semantics

### 2.1 `air.launch`

#### Syntax

```
// Asynchronous form — produces !air.token, non-blocking
[%token =] air.launch (%x₀, …, %xₙ) in (%sx₀=%N₀, …, %sxₙ=%Nₙ)
           args(%a₀=%v₀, …) : <types>
           [dependency = [%t₀, …]]
           [affinity   = [%t₀, …]]
           {
  …
  air.launch_terminator
}

// Synchronous form — caller blocks until all launch instances complete
air.launch sync (%x₀, …, %xₙ) in (%sx₀=%N₀, …, %sxₙ=%Nₙ)
           args(%a₀=%v₀, …) : <types>
           [dependency = [%t₀, …]]
           [affinity   = [%t₀, …]]
           {
  …
  air.launch_terminator
}
```

#### Semantics

`air.launch` defines the outermost scope of an AIR program. It:

- Declares an optional **iteration space** of shape `N₀ × … × Nₙ`. Each point
  `(x₀, …, xₙ)` in this space constitutes an independent _launch instance_.
- Implements **may-be-parallel** semantics: the runtime is permitted to execute
  instances concurrently, but is not required to. The compiler must not assume a
  fixed ordering between instances. A `concurrency` list attribute is therefore
  **not permitted** on `air.launch`; expressing an explicit concurrency bound
  would impose ordering constraints that contradict may-be-parallel semantics.
- Ensures that all `air.segment` operations declared within the body (and their
  associated L2 allocations) are **co-resident on the device** when the launch
  body begins executing, guaranteeing a bounded resource footprint.
- Passes L3 memrefs and scalar values into the body through explicit kernel
  operands. The body is `IsolatedFromAbove` and cannot implicitly capture values
  from the enclosing scope.

#### When to use the launch iteration space

The launch iteration space expresses a pool of **independent work units** that
the runtime should schedule as hardware becomes available. It does not guarantee
simultaneous execution of all instances; it expresses that instances are
independent and may be run in any order or concurrently at the runtime's
discretion.

Use the launch iteration space when:
- Each instance is fully self-contained and produces no result that another
  instance depends on.
- The workload naturally decomposes into a variable number of tiles determined
  at runtime or by a host-side loop.
- Runtime flexibility in scheduling is acceptable or desirable (e.g. when
  multiple launch streams share device resources).

**Do not** use the launch iteration space when you need a **guarantee** that all
instances execute simultaneously — for that, use the `air.segment` iteration
space (§2.2), which provides always-concurrent semantics (all instances are
co-resident and active for the duration of the segment).

#### Expressing pipelining between output tiles

A common goal is to overlap computation on one output tile with data movement
for the next — "software pipelining" across tiles. This cannot be expressed
across separate launch instances, because dependency tokens are the only token
kind that may cross the launch boundary, and they only express ordering, not
overlap.

Instead, express inter-tile pipelining **within a single launch body**:

- **Segment-level pipelining**: place two `air.segment` ops inside the launch
  body — a compute segment and a prefetch segment — connected by `dependency`
  tokens. Both segments are co-resident (the co-residency guarantee applies to
  all segments in the same launch body), so their execution genuinely overlaps:
  the compute segment processes tile N while the prefetch segment loads tile N+1.

- **Loop-based pipelining**: use an `scf.for` loop inside the launch body (or
  inside a segment) with `iter_args`-carried tokens to overlap consecutive tile
  iterations, as described in the `scf.for` section of §2.2. This is the
  standard double-buffering pattern and is typically the most efficient approach.

#### Backend-specific concurrency behaviour

The degree of actual concurrency achieved by a launch iteration space depends
on the backend:

| Backend | Launch iteration space behaviour |
|---------|----------------------------------|
| **GPU (ROCDL)** | Maps directly to `gpu.launch gridDim`; all instances execute in parallel as GPU thread blocks. The GPU scheduler guarantees concurrent dispatch of all grid points subject to occupancy. |
| **NPU (AIE)** | The runtime schedules instances as hardware partitions become available. Instances may run sequentially if only one hardware partition is available for the launch, or concurrently if multiple are available. No concurrency is guaranteed. |

This asymmetry is intentional: GPU hardware is designed for SIMT-style uniform
parallel dispatch, while NPU execution is more task-queue-like with variable
partition availability. Programs that require a guaranteed minimum concurrency
should use `air.segment` iteration spaces or `concurrency` tokens on segments
within a launch body rather than relying on the launch iteration space.

#### Synchrony

The synchrony of `air.launch` is determined by whether a result value is bound,
not by a separate `async` keyword. The `sync` keyword is an explicit synonym for
the no-result form and is provided for readability.

| Form | Behaviour |
|------|-----------|
| `[%token =] air.launch (…) {…}` | **Asynchronous**: dispatches immediately; returns `!air.token` that becomes signaled when all instances complete. Caller may overlap subsequent work. |
| `air.launch (…) {…}` | **Synchronous** (implicit): blocks the calling thread until all launch instances complete. Equivalent to the `sync` form. |
| `air.launch sync (…) {…}` | **Synchronous** (explicit): identical to the no-result form; `sync` keyword makes the blocking intent unambiguous, particularly useful when the op appears alongside async launches. |

#### Allowed token lists

| List | Purpose |
|------|---------|
| `dependency` | `!air.token` values that must be signaled before any instance begins executing. Establishes a data/control dependency. |
| `affinity`   | `!air.token` values that bind this launch to the same hardware partition as other ops sharing the same token (e.g. a specific column range on NPU or a specific XCD on GPU). |

`air.launch` **must not** carry a `concurrency` list. Concurrency constraints
belong to operations with always-parallel semantics (`air.segment`, `air.herd`);
`air.launch` instead expresses may-be-parallel work whose degree of parallelism
is determined by available hardware resources at runtime. Similarly, affinity
and concurrency tokens from enclosing scopes may not be passed into the launch
body (see §1.5 scope rule).

**Nesting constraint**: `air.segment` ops may appear inside `air.launch`, but not the
reverse.

---

### 2.2 `air.segment`

#### Syntax

```
// Asynchronous form — produces !air.token, non-blocking
[%token =] air.segment [@name]
           (%x₀, …, %xₙ) in (%sx₀=%N₀, …)
           args(%a₀=%v₀, …) : <types>
           [x_loc=<col>] [y_loc=<row>] [x_size=<cols>] [y_size=<rows>]
           [dependency  = [%t₀, …]]
           [affinity    = [%t₀, …]]
           [concurrency = [%t₀, …]]
           {
  …
  air.segment_terminator
}

// Synchronous form — caller blocks until the segment (and all its herds) complete
air.segment [@name] sync
           (%x₀, …, %xₙ) in (%sx₀=%N₀, …)
           args(%a₀=%v₀, …) : <types>
           [x_loc=<col>] [y_loc=<row>] [x_size=<cols>] [y_size=<rows>]
           [dependency  = [%t₀, …]]
           [affinity    = [%t₀, …]]
           [concurrency = [%t₀, …]]
           {
  …
  air.segment_terminator
}
```

#### Semantics

`air.segment` represents a **physically contiguous grouping of processing
elements** together with their associated L2 (on-device) memory. It:

- Groups all `air.herd` operations, L2 allocations, and inter-level data
  movement required to implement a coherent kernel.
- Optionally defines an **iteration space** of shape `N₀ × … × Nₙ`. When
  present, every point in the space instantiates an independent copy of the
  segment body. All instances run with **overlapping lifetimes**: the runtime
  guarantees that every instance is active simultaneously, so the full set of
  instances is co-resident on the device for the duration of the segment. This
  is a "stamp-out" of the segment across independent hardware partitions. Each
  instance occupies a **contiguous** rectangular block of compute resources
  (columns × rows of tiles, L2 banks, DMA engines). The placement of distinct
  instances across the iteration space is **independent**: instances need not be
  adjacent or form a globally contiguous region; each may be placed anywhere in
  the device array subject only to non-overlap with other live instances.
- Has access to L2 memory (allocated within its body with `memref.alloc` at
  address space 1) and can access L3 memory through DMA or channel operations.

The body of a segment may contain operations that express both **temporal
constraints** (sequencing via `!air.token` dependency lists, `air.wait_all`,
`scf.for` iter_arg-carried ordering, `scf.parallel` init/reduce token
merging) and **spatial constraints** (placement of `air.herd` tiles via
`x_loc`/`y_loc`, channel routing that fixes communication topology).

#### Synchrony

As with `air.launch`, synchrony is determined by the presence or absence of a
result value; `sync` is an explicit synonym for the no-result form.

| Form | Behaviour |
|------|-----------|
| `[%token =] air.segment (…) {…}` | **Asynchronous**: returns `!air.token` immediately; the enclosing launch body may overlap subsequent work with this segment. Token signals when all herds and data-movement ops within the segment complete. |
| `air.segment (…) {…}` | **Synchronous** (implicit): blocks until all herds and data-movement operations within the segment complete. |
| `air.segment sync (…) {…}` | **Synchronous** (explicit): identical to the no-result form. |

#### Compile-time resource analysis

The compiler **must** determine the total resource requirements of a segment
statically, at compile time. The analysis is **recursive and bottom-up**:
inner segments are fully analysed before the segments that contain them, so
that each level can treat its nested segments as opaque resource blocks.

The analysis at each level has two stages.

**Stage 1 — per-instance worst-case analysis.**

The analysis covers four structural cases: leaf segments (direct `air.herd`
ops), containing segments (nested `air.segment` ops), `scf.for` loops (with
iter_arg-based pipelining), and `scf.parallel` regions (all instances
concurrent, tokens reduced).

*Leaf segment* (body contains `air.herd` ops, no nested `air.segment`):

- **L2 memory**: the maximum number of bytes in address space 1 that may be
  live simultaneously, taken over all possible execution paths (maximum across
  conditional branches; see below for loop-carried allocations).
- **Compute tiles**: the maximum number of `air.herd` tiles simultaneously
  active, derived from the herd iteration spaces and any `x_size`/`y_size`
  placement constraints.
- **DMA channels**: the maximum number of concurrently active
  `air.dma_memcpy_nd` or `air.channel` operations, determined from the async
  dependency graph.

*`scf.for` and `scf.parallel` loops within a segment body:*

**`scf.for` — loop-carried token dependencies.**

`scf.for` iterations are logically sequential in program order, but the async
token model allows controlled overlap between consecutive iterations. The
mechanism is **`iter_args`**: the token yielded by `scf.yield` at the end of
iteration N becomes the iter_arg received at the start of iteration N+1.

```mlir
%t_final = scf.for %k = %lb to %ub step %s
    iter_args(%t_prev = %t_init) -> !air.token {
  // Ops that depend on %t_prev wait for iteration N-1 to reach a known point.
  // Ops that do NOT depend on %t_prev may execute concurrently with N-1.
  %t_dma = air.dma_memcpy_nd async [%t_prev] …
  %t_compute = air.herd async [%t_dma] …
  %t_out = air.wait_all async [%t_compute]
  scf.yield %t_out : !air.token
}
```

An op in iteration N+1 that lists `%t_prev` in its `dependency` list does not
begin until iteration N has signaled that token. An op that does **not** depend
on `%t_prev` may begin before iteration N's `scf.yield` fires, creating genuine
concurrency between consecutive iterations.

The **pipeline depth** D for a given resource type is the number of independent
token chains maintained as separate `iter_args` for that resource:

- **D = 1** (single `iter_arg` chaining all activity): strictly sequential — only
  one iteration's worth of resources is active at a time.
- **D = 2** (double-buffering): one DMA and one compute op from different
  iterations are simultaneously active. The common pattern is to yield the compute
  token as one iter_arg and a separate prefetch token as another, allowing the
  next iteration's data fetch to overlap with the current iteration's compute.
- **D > 2**: multiple independent in-flight streams tracked via multiple
  `iter_args`. Each stream contributes independently to the concurrent resource
  count.

Resource bounds inside an `scf.for`:

- **L2 memory**: `memref.alloc` results that are live across `scf.yield`
  (loop-carried allocations) are permanently live for the entire loop and
  contribute to every iteration's baseline. Allocations freed before `scf.yield`
  require only one slot (reused each iteration).
- **Compute tiles**: the maximum number of simultaneously active herd instances
  across D iterations. Multiply the per-iteration herd footprint by the pipeline
  depth D for the compute resource chain.
- **DMA channels**: similarly, multiply the per-iteration DMA count by the
  pipeline depth D for the DMA resource chain.

**`scf.parallel` — broadcast and tree-reduction of tokens.**

`scf.parallel` expresses a set of independent instances that may execute
concurrently. The async token model for `scf.parallel` has two parts:

```mlir
%t_out = scf.parallel (%i, %j) = (%lb0, %lb1) to (%ub0, %ub1) step (%s0, %s1)
    init (%t_in) -> !air.token {
  // All instances may start as soon as %t_in is signaled.
  %t_instance = air.channel.put async [%t_in] @ch[%i, %j] …
  scf.reduce(%t_instance : !air.token) {
  ^bb0(%a: !air.token, %b: !air.token):
    %t_merge = air.wait_all async [%a, %b]
    scf.reduce.return %t_merge : !air.token
  }
}
// %t_out is signaled only when ALL instances have completed.
```

- **`init (%t_in)`**: a single token that all parallel instances must wait for
  before beginning. It acts as a broadcast dependency — every instance sees the
  same `%t_in` as a prerequisite.
- **`scf.reduce` block**: performs a pairwise tree-reduction over the tokens
  produced by each instance. Each call to the reduce block receives two instance
  tokens (`%a`, `%b`) and uses `air.wait_all` to merge them. The final result
  `%t_out` is signaled only when all instances' tokens have been reduced to one.

Resource bounds inside `scf.parallel`: all instances execute concurrently, so
their resource requirements are **summed** (not maxed), exactly as for a segment
iteration space (Stage 2 above). The aggregate resource demand is the per-instance
bound multiplied by the total number of parallel instances.

*Containing segment* (body contains one or more nested `air.segment` ops):

Each nested segment is first fully analysed through its own Stage 1 and Stage 2
to produce a **total resource footprint** — a tuple (tiles, L2 bytes, DMA
channels) representing the resources that nested segment holds across all of its
iterations simultaneously. This footprint is then treated as an indivisible unit
at the outer level. The containing segment's Stage 1 computes:

- **Compute tiles**: the maximum number of tile-rows × tile-columns
  simultaneously reserved, obtained by summing the footprints of all
  concurrently active nested segments (and any direct `air.herd` ops) in the
  outer body. "Concurrently active" is determined from the outer body's async
  dependency graph: nested segments linked only by `dependency` tokens are
  treated as sequential and their resources may be reused; nested segments with
  no ordering constraint between them, or linked by `concurrency` tokens, are
  treated as simultaneous and their resources are summed.
- **L2 memory**: maximum simultaneous L2 allocation across all concurrently
  active nested segments plus any direct allocations in the outer body.
- **DMA channels**: maximum concurrent DMA/channel ops, counting each nested
  segment's DMA footprint as a unit.

**Stage 2 — scaling by the iteration space.**
Because all instances of a segment's iteration space execute with overlapping
lifetimes, the per-instance bounds from Stage 1 are **multiplied by the total
iteration count** (`N₀ × … × Nₙ`) to obtain that segment's total resource
footprint. A segment with no iteration space has an implicit count of 1.

Stage 2 applies **independently at each nesting level**: the inner segment
scales by its own iteration count first; the resulting total footprint is what
the outer Stage 1 sees as a unit; the outer segment then scales its own
per-instance result by its iteration count. Resources are therefore multiplied
by the product of all iteration counts along the nesting path, reflecting the
fact that all instances at every level are simultaneously live.

The total footprint describes the *sum* of per-instance blocks, not a single
contiguous region: the compiler allocates a contiguous block per instance and
may place those blocks anywhere in the device array that fits.

These statically computed totals are used to:
1. Verify that the segment hierarchy (across all iterations at all levels) fits
   within the target hardware partition before lowering begins.
2. Assign physical resources (tile columns/rows, memory banks, DMA engines) to
   each iteration at each level without runtime negotiation.
3. Guarantee that all segment instances — and any other segments declared within
   the same `air.launch` — can be **co-resident** without resource conflicts.

Optional physical placement attributes (`x_loc`, `y_loc`, `x_size`, `y_size`)
annotate the column/row offset and extent of **one** instance on devices with
2D tile arrays. For a segment with an iteration space, the compiler assigns a
separate `(x_loc, y_loc)` origin to each iteration point independently; those
per-instance origins need not be adjacent. When these attributes are present
they are taken as authoritative for a single instance's contiguous footprint;
when absent the compiler derives a per-instance contiguous footprint from the
worst-case analysis above and places each instance freely.

The optional `backend-granularity` attribute communicates the hardware
granularity that this segment instance should map to on the target backend.
Currently defined values:

| Value | Meaning |
|-------|---------|
| `XCD` | The segment maps to one XCD (Accelerator Complex Die) on AMD MI3xx GPUs. Used with a multi-XCD `air.launch` iteration space to achieve full-device occupancy (see §4.2). |

When `backend-granularity` is absent the backend applies its default mapping
(e.g., one workgroup / thread block on GPU).

**Nesting constraints**:
- `air.segment` must be nested inside `air.launch` or inside another
  `air.segment` (directly or indirectly — the outermost segment must be inside
  a launch).
- `air.herd` must be nested inside `air.segment`. It is the innermost hierarchy
  level and must not contain `air.segment` or another `air.herd`.
- The nesting depth of `air.segment` is unbounded by the model; backends may
  impose device-specific limits (e.g. two levels on MI3xx — XCD and CU).

---

### 2.3 `air.herd`

#### Syntax

```
// Synchronous form — acquires resources immediately, blocks until all PEs complete
air.herd [@name] tile (%x, %y) in (%sx=%Nx, %sy=%Ny)
         args(%a₀=%v₀, …) : <types>
         [x_loc=<col>] [y_loc=<row>]
         [link_with="<object>"]
         [dependency  = [%t₀, …]]
         [affinity    = [%t₀, …]]
         [concurrency = [%t₀, …]]
         {
  …
  air.herd_terminator
}

// Asynchronous form — produces !air.token; acquires resources when deps met, signals on completion
[%token =] air.herd [@name]
           tile (%x, %y) in (%sx=%Nx, %sy=%Ny)
           args(%a₀=%v₀, …) : <types>
           [x_loc=<col>] [y_loc=<row>]
           [link_with="<object>"]
           [dependency  = [%t₀, …]]
           [affinity    = [%t₀, …]]
           [concurrency = [%t₀, …]]
           {
  …
  air.herd_terminator
}
```

#### Semantics

`air.herd` defines a **1D or 2D array of processing elements** (PEs) that all
execute the same body code in data-parallel fashion. It is the innermost level
of the hierarchy and maps directly to physical compute tiles (NPU) or GPU
threads:

- The iteration space `%Nx × %Ny` determines how many PE instances execute the
  body and defines a **contiguous** block of compute resources. All instances
  run concurrently on distinct hardware resources; this is **always-parallel**
  (not may-be-parallel).
- Block arguments `%x` and `%y` give each instance its own coordinates,
  enabling each PE to independently address its portion of L1 or L2 memory.
- Within the herd body, `L1` memory (address space 2) is per-PE local
  scratchpad.
- Data needed from L2 or L3 must be explicitly fetched via `air.dma_memcpy_nd`
  or channel operations before use.

Operations within the herd body may impose both **temporal constraints**
(explicit token dependencies ordering DMA fetches before compute) and
**spatial constraints** (channel indices or tile-coordinate arithmetic that
binds each PE to a specific region of a shared buffer).

#### Platform-specific iteration space semantics

The meaning of "contiguous" and the constraints imposed on iteration space
dimensions depend on the target backend:

- **AIE (NPU)**: A herd defines a logical rectangular array of compute units.
  The compiler may **reshape** the iteration space (e.g., collapse a 2D herd
  into a 1D arrangement) via the `AIRCollapseHerdPass`. Reshaping is inhibited
  automatically when the herd body uses cascade channels (`channel_type =
  "cascade"`), because cascade connections are topology-dependent and cannot
  survive reindexing. Explicit placement attributes (`x_loc`, `y_loc`,
  `x_size`, `y_size`) on the enclosing segment also constrain the legal shapes
  by fixing the tile footprint. The pass accepts a `max-col-size` option to
  bound the width of the collapsed arrangement. A dedicated per-herd attribute
  to disable reshaping independently of these conditions is not yet
  implemented. The resulting contiguous rectangular region of physical tiles is
  determined after any reshaping has been applied.

- **GPU (AMD MI3xx family)**: A herd executes entirely within a single Compute
  Unit (CU), with PE instances mapped to individual warps. The combined
  resource requirements (register file, LDS, wavefront slots) of all PE
  instances must fit within one CU. On MI3xx devices a CU provides fewer than
  32 wavefront slots, so the total number of PE instances (`%Nx × %Ny`) is
  correspondingly limited.

#### Synchrony

| Form | Behaviour |
|------|-----------|
| No result (synchronous) | Acquires resources immediately; blocks the enclosing body until all PE instances complete; releases resources at the herd terminator. |
| `[%token =]` (asynchronous) | Acquires resources when all `dependency` tokens are signaled; returns `!air.token` immediately; releases resources when the token fires (all PE instances complete). |

Optional attributes `x_loc`/`y_loc` specify the base placement on 2D tile
arrays; each PE at tile `(%x, %y)` occupies physical column/row
`(x_loc + %x, y_loc + %y)`. The `link_with` attribute names an external kernel
object to link into the herd at compile time.

---

### 2.4 `air.dma_memcpy_nd`

```
air.dma_memcpy_nd (dst[dst_offsets][dst_sizes][dst_strides],
                   src[src_offsets][src_sizes][src_strides]) : (type_dst, type_src)
```

Describes an **asynchronous N-dimensional strided bulk copy** between two memrefs. The
`[offsets][sizes][strides]` triples follow the same convention as `memref.subview`:
`offsets` is the starting index in each dimension, `sizes` is the number of elements to
transfer in each dimension, and `strides` is the stride in units of elements.

The operation is direction-agnostic: either the source or the destination (or both) may
be at any memory level. The direction (L3→L2, L2→L1, L1→L2, etc.) is inferred from the
address spaces of the operand memrefs and mapped to the appropriate hardware mechanism.

An empty `[offsets]`, `[sizes]`, or `[strides]` list for a side means the entire memref
is addressed with unit strides.

---

### 2.5 `air.channel`, `air.channel.put`, `air.channel.get`

```
// Channel declaration — at module scope
air.channel @name [dim₀, dim₁, …] {channel_type = "dma_stream", depth = <N>}

// Synchronous put/get — block until the transfer completes
air.channel.put @name[indices] (src[offsets][sizes][strides]) : (type_src)
air.channel.get @name[indices] (dst[offsets][sizes][strides]) : (type_dst)

// Asynchronous put/get — return !air.token; transfer completes when token signals
[%token =] air.channel.put @name[indices] [dependency = [%t₀, …]]
           (src[offsets][sizes][strides]) : (type_src)
[%token =] air.channel.get @name[indices] [dependency = [%t₀, …]]
           (dst[offsets][sizes][strides]) : (type_dst)
```

Channels are **declared** once (at module scope) and **used** at distinct producer and
consumer sites. They decouple the put and get sides, allowing the compiler to schedule
them independently and to introduce double-buffering.

A channel may be an array (e.g., `[4, 4]` for a 4×4 array). The `indices` operand on
`put`/`get` selects the specific channel within the array.

The `channel_type` attribute controls the underlying mechanism:

| Value | Mechanism |
|-------|-----------|
| `"dma_stream"` (default) | DMA engines with streaming (circuit-switched) interconnect |
| `"dma_packet"` | DMA engines with packet-switched interconnect |
| `"cascade"` | Core-to-core cascade connections between adjacent tiles |

The `broadcast_shape` attribute enables one-to-many communication following NumPy
broadcasting rules.

#### Capacity and depth

A channel has a finite buffer capacity set by the `depth` attribute (default **1**).
The depth specifies how many transfers may be in-flight simultaneously:

| Depth | Effect |
|-------|--------|
| 1 (default) | Rendezvous: each `put` must be consumed by a `get` before the next `put` can proceed. |
| 2 | Double-buffering: producer may issue one transfer ahead of the consumer. |
| N | Producer may have up to N transfers outstanding before it must wait. |

Increasing depth relaxes the coupling between producer and consumer at the cost of
additional buffer memory (each slot requires storage for one transfer's worth of data).

#### Flow control semantics

`put` and `get` are **blocking** at the channel boundary:

- A `put` issued when the channel already holds `depth` unread transfers stalls until
  the consumer issues a `get` and frees a slot.
- A `get` issued when the channel holds no data stalls until the producer issues a `put`.

In the asynchronous form the operation is dispatched immediately and the returned
`!air.token` does not signal until the blocking condition is resolved and the transfer
is complete. This allows other work to proceed while the channel handshake is in
progress, but does not remove the fundamental flow-control constraint.

#### Balance requirement

For a channel to make progress, every `put` must be matched by a `get` on the same
channel index, and vice versa. The compiler enforces the **static balance condition**:

- Along every possible execution path through the program the number of `put` operations
  and the number of `get` operations at each channel index must be equal.
- For channels inside loop bodies, balance must hold per iteration (equal puts and gets
  in the loop body).
- For channels connecting herds with different iteration spaces, the total transfer count
  must balance: `(puts per producer instance) × (producer instances)` must equal
  `(gets per consumer instance) × (consumer instances)`.
- For channels inside conditional branches, balance must hold independently on each
  branch.

A violation of the balance condition is a compile-time error.

#### Deadlock conditions

A program deadlocks when a `put` is waiting for a `get` that can never execute, or a
`get` is waiting for a `put` that can never execute, and no other operation can break
the wait.

**Minimal deadlock** — `put` and `get` in the same sequential scope:

```mlir
air.channel @C [] {depth = 1}

air.segment @deadlock {
  // put blocks: channel is at capacity, waiting for a get to free space
  air.channel.put @C[] (%src[][][]) : (memref<…, 1>)
  // never reached — sequential control flow cannot reach get while put blocks
  air.channel.get @C[] (%dst[][][]) : (memref<…, 1>)
  air.segment_terminator
}
```

The necessary condition for deadlock freedom is that for every channel there exists a
**concurrent execution context** in which the matching `put` and `get` can both
make progress simultaneously. In practice:

- `put` and `get` on the same channel must appear in different herds, different async
  branches, or different segment instances — any context that allows them to execute
  concurrently.
- The communication graph (nodes = ops, edges = channel put→get dependencies) must
  be **acyclic**: a cycle means at least one op in the cycle is waiting on another in
  the same cycle, which can never be resolved.

---

### 2.6 `air.execute` and `air.wait_all`

`air.execute` wraps a sequential block of host-visible computation (e.g., `memref.alloc`,
arithmetic on scalars) in an asynchronous region. The region produces an
`!air.token` (dependency kind) that becomes ready when all operations inside have completed.

`air.wait_all` takes a variadic list of `!air.token` values and produces a single
token that becomes ready only when all of its inputs are ready. It acts as a join node
in the async dependency graph.

---

## 3. NPU (AIE) Backend Mapping

On AMD Versal AI Engine (AIE) and Ryzen AI NPU targets the three-level hierarchy maps
directly to physical hardware structures:

| AIR concept | AIE hardware |
|-------------|-------------|
| `air.launch` | Device-level kernel dispatch; L3 = DDR via NOC |
| `air.segment` | A contiguous rectangle of AIE tiles + associated memory tiles or URAMs (L2) |
| `air.herd tile (x, y)` | A single AIE compute tile at physical column/row offset `(x_loc+x, y_loc+y)` |
| L1 memory (space 2) | Per-tile local data memory (32 KB on AIE2) |
| L2 memory (space 1) | Shared memory tiles or URAMs accessible via the tile interconnect |
| L3 memory (space 0) | DDR accessible via AXI4-MM NOC |

### Data movement on NPU

| AIR operation | AIE mechanism |
|---------------|--------------|
| `air.dma_memcpy_nd` (L3→L2) | Shim DMA + AXI-S routing + memory tile DMA |
| `air.dma_memcpy_nd` (L2→L1) | Tile DMA + local AXI-S stream |
| `air.channel` (`dma_stream`) | DMA + circuit-switched streaming interconnect |
| `air.channel` (`dma_packet`) | DMA + packet-switched overlay network |
| `air.channel` (`cascade`) | Core cascade interface (direct register-to-register) |

The `air-to-aie` pass translates each `air.herd` into AIE tile operations (`aie.core`,
`aie.buffer`, `aie.lock`), routes channels through the switch boxes, and generates the
SHIM DMA configuration for L3 transfers.

### Placement

Placement of a segment on physical tiles is guided by the optional `x_loc`/`y_loc`
(column/row offset) and `x_size`/`y_size` (column/row count) attributes on
`air.segment`. The herd's tile indices are added to the segment offset to obtain the
absolute physical tile location.

---

## 4. GPU (ROCDL/HIP) Backend Mapping

On AMD GPU targets the AIR hierarchy maps onto the GPU execution model. The
`air-to-rocdl` and `air-gpu-outlining` passes perform this translation. The
basic mapping uses three levels (launch → segment → herd); on MI3xx devices a
four-level mapping with nested segments is used to achieve full-device occupancy
(see §4.2).

### 4.1 Hierarchy mapping

| AIR concept | GPU concept |
|-------------|------------|
| `air.launch (%bx,%by) in (%gbx,%gby)` | `gpu.launch` grid: `gridDim = (gbx, gby, 1)` |
| `air.segment` | Workgroup (thread block); the segment body runs within a single `gpu.launch` |
| `air.herd tile (%x,%y) in (%bx,%by)` | Thread block dimensions: `blockDim = (bx, by, 1)` |
| Herd tile index `(%x, %y)` | `(threadIdx.x, threadIdx.y)` |
| Launch index `(%bx, %by)` | `(blockIdx.x, blockIdx.y)` |

The `air.launch` iteration space becomes the **grid** (number of thread blocks), and
the `air.herd` iteration space becomes the **block** (number of threads per block).
The `air.segment` body is the thread-block body: code that runs once per workgroup before
and after the per-thread `air.herd` body.

After translation the hierarchy is flattened: `air.segment` and `air.herd` are erased
and their bodies are moved into the enclosing `gpu.launch` region. The
`air-gpu-outlining` pass then extracts the `gpu.launch` body into a `gpu.func` kernel
and injects the appropriate `gpu.BlockIdOp`, `gpu.ThreadIdOp`, `gpu.GridDimOp`, and
`gpu.BlockDimOp` intrinsics.

### 4.2 Device-scale mapping via nested segments (MI3xx)

On AMD MI3xx devices the full device can be occupied using a four-level
hierarchy built from a unit-iteration-space launch, two nested segments, and
an innermost herd:

| Level | AIR construct | MI3xx granularity |
|-------|--------------|-------------------|
| 1 | `air.launch` (no iteration space — implicit single instance) | Whole-device dispatch |
| 2 | Outer `air.segment` (`backend-granularity=XCD`, iteration space = 8) | All 8 XCDs simultaneously — all instances stamp out concurrently |
| 3 | Inner `air.segment` (iteration space = CUs per XCD) | One CU within the XCD — all instances stamp out concurrently |
| 4 | `air.herd` (iteration space ≤ 32) | Concurrent warps on that CU |

**`air.launch`** carries no iteration space (an implicit single-point space).
Its sole purpose at this level is to anchor the co-residency guarantee for
everything nested inside.

**Outer `air.segment`** (`backend-granularity=XCD`) has an iteration space of
cardinality 8, one instance per XCD on the MI3xx device. Because segment
instances always execute with overlapping lifetimes, all 8 XCDs are occupied
simultaneously.

**Inner `air.segment`** is nested inside the outer and iterates over the CUs
within its XCD. Again all instances run concurrently, so every CU in the XCD
is active at the same time.

**`air.herd`** is nested inside the inner segment and targets the warps running
on that single CU. The iteration space must not exceed 32 PE instances (the
wavefront-slot limit of a MI3xx CU, as described in §2.3). Iterations within
the herd may communicate through:

- **LDS memory** (`L2` address space) — per-CU local data store, shared among
  all warps in the herd.
- **Global memory** (`L3` address space) — device HBM, accessible to all
  levels of the hierarchy.

The `backend-granularity=XCD` attribute on the outer segment is the only
MI3xx-specific annotation required; all other tiling falls out of the standard
segment and herd iteration space mechanisms.

### 4.3 Memory space mapping

| AIR memory space | Enum | GPU address space | GPU scope |
|-----------------|------|------------------|-----------|
| L3 (space 0) | `MemorySpace::L3` | 0 (generic/global) | Device global memory (HBM) |
| L2 (space 1) | `MemorySpace::L2` | 3 (local) | Workgroup (LDS/shared memory) |
| L1 (space 2) | `MemorySpace::L1` | 5 (private) | Per-thread private memory (VGPRs/scratch) |

`memref.alloc` ops in L2 space are hoisted to `gpu.launch` **workgroup attributions**
(shared among all threads in the block). `memref.alloc` ops in L1 space become
**private attributions** (one copy per thread). Explicit `memref.dealloc` ops are
removed because GPU attributions have implicit kernel-scoped lifetimes.

### 4.4 Data movement on GPU

`air.dma_memcpy_nd` operations are lowered to explicit SCF loops containing
`memref.load` / `memref.store` pairs. The offsets, sizes, and strides from the DMA
descriptor drive the loop bounds and index arithmetic. Subsequent lowering passes
(ROCDL, LLVM CodeGen) optimise these into coalesced global loads, LDS stores, or
register-to-register moves depending on the inferred address spaces.

`gpu.barrier` instructions (already present in well-formed AIR GPU programs or inserted
by the compiler) synchronize threads at workgroup boundaries between DMA-equivalent
loads into shared memory and subsequent compute.

### 4.5 Example: 4k×4k matrix multiplication

The GPU test in `test/gpu/4k_4k_mul/air_sync.mlir` illustrates the model:

```mlir
// 32×32 grid of segments, one per 128×128 output tile
air.launch (%bx, %by) in (%nbx=%c32, %nby=%c32)
    args(%A=%arg0, %B=%arg1, %C=%arg2) : … {

  air.segment @forward_0 args(%bx=%bx, %by=%by, %A=%A, %B=%B, %C=%C) : … {
    // Segment body: runs once per workgroup
    // L2 allocations — shared memory tiles
    %As = memref.alloc() : memref<128x8xf32, 1>   // L2 → LDS
    %Bs = memref.alloc() : memref<8x128xf32, 1>   // L2 → LDS

    scf.for %k = 0 to 4096 step 8 {
      // Load A and B tiles into shared memory (DMA equivalent: global → LDS)
      // Uses gpu.thread_id to distribute loads across threads
      …
      gpu.barrier

      // 256 herd tiles (threads) compute the outer product
      air.herd @herd_0 tile (%tx, %ty) in (%ntx=%c256, %nty=%c1)
               args(%As=%As, %Bs=%Bs, …) : … {
        // L1 accumulation registers (VGPRs)
        %acc = memref.alloc() : memref<64xf32, 2>  // L1 → private/VGPRs
        …
        gpu.barrier
      }
    }
    // Write accumulated results back to L3 (global memory)
  }
}
```

The mapping:
- `air.launch` → `gpu.launch` with `gridDim = (32, 32, 1)`
- `air.segment` → workgroup body (executed once per thread block)
- `air.herd tile (%tx, .) in (%c256, %c1)` → `blockDim = (256, 1, 1)`, `threadIdx.x` = `%tx`
- L2 memrefs (space 1) → LDS (shared memory)
- L1 memrefs (space 2) → VGPRs / private scratch

### 4.6 Compilation pipeline

The full GPU lowering pipeline is:

1. **`air-to-rocdl`** — Flattens `air.launch`/`air.segment`/`air.herd` into a
   `gpu.launch`, converts L1/L2 allocations to workgroup/private attributions, and
   lowers `air.dma_memcpy_nd` to SCF loops.
2. **`air-gpu-outlining`** — Outlines the `gpu.launch` body into a `gpu.func` inside
   a `gpu.module`, injects `gpu.BlockIdOp`, `gpu.ThreadIdOp`, etc., and converts
   the launch site to `gpu.launch_func`.
3. **LLVM lowering** — Lowers affine, SCF, and CF dialects.
4. **ROCDL binary generation** — Attaches a ROCDL target descriptor (`rocdl-attach-
   target`), converts `gpu.module` to ROCDL IR (`convert-gpu-to-rocdl`), and
   serialises the GPU binary with `gpu-module-to-binary`.

See [buildingGPU.md](buildingGPU.md) for build instructions and the complete
`aircc.py` compilation pipeline.

---

## 5. Summary Comparison

| Concept | NPU (AIE) | GPU (ROCDL) |
|---------|-----------|-------------|
| `air.launch` iteration point | Device-level work unit | One GPU thread block |
| `air.segment` | Rectangle of AIE tiles + memory tiles | Thread block workgroup body |
| `air.herd` tile | Single AIE compute tile | Single GPU thread |
| L1 (space 2) | 32 KB tile-local data memory | Thread-private VGPRs / scratch |
| L2 (space 1) | Memory tiles / URAMs | LDS (shared memory, ~64 KB / CU) |
| L3 (space 0) | DDR via NOC | HBM via global memory |
| `dma_memcpy_nd` | AIE Shim/Tile DMA engines | SCF load/store loops |
| `channel` (`dma_stream`) | Streaming AXI-S switch | — (not yet mapped to GPU) |
| Synchronization | AIE locks | `gpu.barrier` |
| `!air.token` (dependency) | AIE runtime completion signals | GPU stream/event dependencies |
