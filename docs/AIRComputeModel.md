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
carries an `air.async.token` result is dispatched immediately and executes concurrently
with subsequent operations. Token values are used to express data and control
dependencies. `air.wait_all` merges multiple tokens into a single synchronization point.
`air.execute` wraps sequential computation in an asynchronous region.

---

## 2. Operation Semantics

### 2.1 `air.launch`

```
air.launch (%x₀, …, %xₙ) in (%sx₀=%N₀, …, %sxₙ=%Nₙ)
           args(%a₀=%v₀, …) : <types> {
  …
  air.launch_terminator
}
```

`air.launch` defines the outermost scope of an AIR program. It:

- Declares an optional **iteration space** of shape `N₀ × … × Nₙ`. Each point
  `(x₀, …, xₙ)` in this space executes the body concurrently as an independent
  _launch instance_.
- Ensures that all the initial `air.segment` operations (and their associated L2 allocations) are
  **co-resident on the device** when the launch body begins executing.
- Ensures that there is a bounded resource

- Passes L3 memrefs and scalar values into the body through explicit kernel operands.

The iteration space of a launch is typically used to divide the output problem into
independent tiles, one per segment.

**Nesting constraint**: `air.segment` ops may appear inside `air.launch`, but not the
reverse.

---

### 2.2 `air.segment`

```
air.segment @name (%x₀, …, %xₙ) in (%sx₀=%N₀, …) args(%a₀=%v₀, …) : <types> {
  …
  air.segment_terminator
}
```

`air.segment` represents a physically contiguous grouping of processing elements together
with their associated L2 (on-device) memory. It:

- Groups all `air.herd` operations, L2 allocations, and inter-level data movement
  required to implement a coherent kernel.
- Optionally defines its own **iteration space**, which spatially replicates the segment
  across independent hardware resources (a "stamp-out" of the segment).
- Has access to L2 memory (allocated within its body with `memref.alloc` at address
  space 1) and can access L3 memory through DMA or channel operations.

Optional physical placement attributes (`x_loc`, `y_loc`, `x_size`, `y_size`) annotate
the column/row offset and extent on devices with 2D tile arrays.

**Nesting constraint**: `air.herd` ops may appear inside `air.segment`, and `air.segment`
must appear inside `air.launch` (directly or indirectly).

---

### 2.3 `air.herd`

```
air.herd @name tile (%x, %y) in (%sx=%Nx, %sy=%Ny)
         args(%a₀=%v₀, …) : <types> {
  …
  air.herd_terminator
}
```

`air.herd` defines a **1D or 2D array of processing elements** that all execute the same
body code. It is the innermost level of the hierarchy and maps directly to physical
compute tiles or GPU threads:

- The iteration space `%Nx × %Ny` determines how many PE instances execute the body.
- Block arguments `%x` and `%y` give each instance its own coordinates, enabling each
  PE to independently compute its tile of the output.
- Within the herd body, `L1` memory (address space 2) is per-PE local scratchpad.
- Data needed from L2 or L3 must be explicitly fetched via `air.dma_memcpy_nd` or
  channel operations before use.

Optional attributes `x_loc`/`y_loc` specify placement on 2D tile arrays. The
`link_with` attribute names an external kernel object to link into the herd.

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
air.channel @name [dim₀, dim₁, …] {channel_type = "dma_stream"}

air.channel.put @name[indices] (src[offsets][sizes][strides]) : (type_src)
air.channel.get @name[indices] (dst[offsets][sizes][strides]) : (type_dst)
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

---

### 2.6 `air.execute` and `air.wait_all`

`air.execute` wraps a sequential block of host-visible computation (e.g., `memref.alloc`,
arithmetic on scalars) in an asynchronous region. The region produces an
`air.async.token` that becomes ready when all operations inside have completed.

`air.wait_all` takes a variadic list of `air.async.token` values and produces a single
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

On AMD GPU targets the same three-level hierarchy maps onto the GPU execution model. The
`air-to-rocdl` and `air-gpu-outlining` passes perform this translation.

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

### 4.2 Memory space mapping

| AIR memory space | Enum | GPU address space | GPU scope |
|-----------------|------|------------------|-----------|
| L3 (space 0) | `MemorySpace::L3` | 0 (generic/global) | Device global memory (HBM) |
| L2 (space 1) | `MemorySpace::L2` | 3 (local) | Workgroup (LDS/shared memory) |
| L1 (space 2) | `MemorySpace::L1` | 5 (private) | Per-thread private memory (VGPRs/scratch) |

`memref.alloc` ops in L2 space are hoisted to `gpu.launch` **workgroup attributions**
(shared among all threads in the block). `memref.alloc` ops in L1 space become
**private attributions** (one copy per thread). Explicit `memref.dealloc` ops are
removed because GPU attributions have implicit kernel-scoped lifetimes.

### 4.3 Data movement on GPU

`air.dma_memcpy_nd` operations are lowered to explicit SCF loops containing
`memref.load` / `memref.store` pairs. The offsets, sizes, and strides from the DMA
descriptor drive the loop bounds and index arithmetic. Subsequent lowering passes
(ROCDL, LLVM CodeGen) optimise these into coalesced global loads, LDS stores, or
register-to-register moves depending on the inferred address spaces.

`gpu.barrier` instructions (already present in well-formed AIR GPU programs or inserted
by the compiler) synchronize threads at workgroup boundaries between DMA-equivalent
loads into shared memory and subsequent compute.

### 4.4 Example: 4k×4k matrix multiplication

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

### 4.5 Compilation pipeline

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
| Async tokens | AIE runtime | GPU stream/event dependencies |
