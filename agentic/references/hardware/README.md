# NPU2 / AIE2P resources for mapping decisions

Use this reference when assigning an owner, changing buffering or parallelism, diagnosing a resource failure, or checking a proposed transfer path. It supplies **versioned source declarations**, not a measured hardware inventory or a proof that a mapping will route, complete, or meet a performance target.

This profile is MLIR-AIE `npu2`, target architecture `AIE2p`, at source revision `7e00b57955e108fe9d8e9419f5828a0c7e650858`, with AIE runtime source revision `8849e208bdcc533b20a0ed3f95c1ce961dee9c3a`. These identify the source model. Installed-wheel, compiler-binary and candidate-artifact provenance must be bound separately. The [source catalog](sources.md) records the inspected source/install relationships and every file/symbol used below. No compiler or device was executed for this reference.

## Select the actual partition first

The full `npu2` model declares **8 columns**, with shim at row **0**, MemTile at row **1**, and **4 compute rows**, rows **2–5**. Thus the full model contains **32 compute tiles, 8 MemTiles and 8 shim NOC tiles**. These are declared full-array counts, not currently free resources. `npu2_1col` through `npu2_7col` select smaller column partitions; an N-column partition has **4N compute tiles, N MemTiles and N shim tiles**. [S1](sources.md#s1-target-model)

Read the candidate's `aie.device(...)`, final placement and runtime partition binding. A logical herd coordinate need not be a final physical coordinate. Edge neighbors may be absent, reserved resources may be unavailable, and another partition's unused capacity cannot be assumed accessible. The operating system's current allocation and physical device availability were not queried here.

## Quick resource table

All counts below are **per tile of the named type**. KiB denotes 1024 bytes. RX is S2MM, stream to memory; TX is MM2S, memory to stream. Counts are per direction, and RX0 and TX0 are different channels. A stream route, a DMA channel and a buffer descriptor (BD) are different resources.

| Resource | Compute tile | MemTile | Shim NOC tile | What the mapping can actually use |
|---|---|---|---|---|
| Owned data SRAM | **64 KiB** | **512 KiB** | No compute/MemTile payload SRAM budget is exposed for shim | Count owned allocations, live aliases, alignment, staging and reservations. External BO memory reached through shim is a different address space. [S1](sources.md#s1-target-model), [S3](sources.md#s3-aie2p-driver-declarations) |
| Compiler allocation banks | **4 × 16 KiB** | **8 × 64 KiB** | Not applicable to payload SRAM | These are the target model's bank partitions used by the allocator; sizes are capacity/count. Check actual bank/address assignment. This table does not assert the physical microbank or port topology. [S1](sources.md#s1-target-model), [S4](sources.md#s4-allocation-and-stack) |
| Program memory | **16 KiB**, separate from data SRAM | No compute core | No compute core | Inspect the linked core's program footprint separately. Program space cannot make up a data-memory shortage. [S3](sources.md#s3-aie2p-driver-declarations) |
| Core stack reservation | Default **1024 bytes**, overridden by `aie.core` `stack_size` | No core stack | No core stack | Compiler policy, not a hardware capacity. The reserved stack consumes local data SRAM; the default does not establish actual frame depth. [S4](sources.md#s4-allocation-and-stack) |
| DMA channels | **2 RX + 2 TX** | **6 RX + 6 TX** | **2 RX + 2 TX** | Subtract channels already assigned in the same tile and direction. Count genuinely independent scheduled paths; more buffers or BDs do not add channels. [S2](sources.md#s2-connectivity-and-address-views), [S3](sources.md#s3-aie2p-driver-declarations) |
| BD slots | **16 total**, IDs **0–15** | **48 total**, IDs **0–47** | **16 total**, IDs **0–15** | Shared per-tile descriptor pools, not a fresh pool per channel or direction. Account for all resident chains and any controlled reuse. [S1](sources.md#s1-target-model), [S5](sources.md#s5-descriptor-allocation-and-validation) |
| BD accessibility | No additional channel partition in this model | Even channels **0/2/4** use IDs **0–23**; odd channels **1/3/5** use **24–47** | No additional channel partition in this model | MemTile has two shared pools of **24** slots. A chain can exhaust its parity pool while slots remain in the other pool. The parity rule applies in either direction. [S1](sources.md#s1-target-model), [S3](sources.md#s3-aie2p-driver-declarations) |
| Local lock objects | **16**, IDs **0–15** | **64**, IDs **0–63** | **16**, IDs **0–15** | These count local synchronization objects, not descriptor count or encoded remote-lock indices. Count compiler-added ownership/credit locks too. [S1](sources.md#s1-target-model), [S3](sources.md#s3-aie2p-driver-declarations) |

The shim column is intentionally explicit even where its counts equal compute's. A shim connects DMA to external memory through the memory-mapped NOC; it does not inherit compute SRAM, banks, stack or core load/store addressing. The target's generic non-MemTile branch alone is not evidence of such shim storage. [S1](sources.md#s1-target-model), [S3](sources.md#s3-aie2p-driver-declarations)

**Usable storage is candidate-specific.** For a compute tile with only the default stack reservation, subtracting 1024 from 65536 leaves **64512 bytes (63 KiB)** before all other data allocations, padding and staging. This arithmetic is an upper budget, not an allocatable-buffer guarantee. The bank allocator reserves the stack at the start of the local bank map. Linked globals, constants, spill/frame requirements and compiler-generated buffers must be reconciled with actual addresses; do not count aliased objects twice or assume unimplemented lifetime reuse. [S4](sources.md#s4-allocation-and-stack)

## Access and connectivity change the budget

| Question | Supported source fact | Conditional design implication |
|---|---|---|
| Can a core borrow nearby storage? | Core memory affinity includes its own memory (called the **east** view in this model), west, north and south where legal. The bottom compute row has no southward core-memory access into MemTile. These are distinct address windows, not additional local SRAM. [S2](sources.md#s2-connectivity-and-address-views) | A remote owner needs a legal neighbor, allocation ownership and completion protocol. Do not turn an address aperture into free capacity or conflate core load/store reach with DMA reach. |
| Can a MemTile reach another MemTile's storage? | The model provides west/local/east memory and lock views. The objectFIFO channel allocator restricts paths requiring adjacent-MemTile access to channels **0–3**. [S2](sources.md#s2-connectivity-and-address-views), [S5](sources.md#s5-descriptor-allocation-and-validation) | Check neighbor existence and those channels' availability, parity pools and ownership. Channels **4/5** are not spare substitutes for this lowering. This is a versioned compiler-supported constraint, not a claim that every other backend is impossible. |
| Is horizontal memory access a horizontal stream link? | MemTile switchboxes expose north/south stream connections and no East/West stream bundle in this target model. Core and shim switchboxes have horizontal stream connections, subject to boundaries. [S2](sources.md#s2-connectivity-and-address-views) | A MemTile-to-MemTile stream may need another row's switches. Neighbor-memory access and a routed stream are different mechanisms. |
| Does passing through shim imply a host/DDR transfer? | Shim stream-switch connections and shim DMA/NOC endpoints are separate resources. [S2](sources.md#s2-connectivity-and-address-views) | Inspect the route endpoints. A route through a shim switch alone does not establish a DDR transfer. |
| Are the local lock counts inconsistent with MemTile DMA's larger count? | The driver declares **64 local locks** in `Aie2PMemTileLockMod` and **192 DMA-addressable lock IDs** in `Aie2PMemTileDmaMod`. The model translates west/local/east lock views with base indices **0/64/128**. [S2](sources.md#s2-connectivity-and-address-views), [S3](sources.md#s3-aie2p-driver-declarations) | Count **64 owned locks** at each MemTile. Decode a remote index against its owning tile; **192** is an addressable domain spanning views, not a larger local pool. Partition-edge availability still applies. |

See [DMA addressing and field limits](dma-addressing.md) for address units, payload extent, descriptor dimensions and the difference between an address aperture and owned SRAM. Stream port counts do not establish route availability: the selected physical path still needs compatible switch/arbiter assignments, framing, ordering, receiver capacity and progress.

## Two small decision cases

**Capacity case — reject one placement.** Assume two distinct payloads must be live concurrently, each with **8 rows × 2560 elements × 2 bytes = 40960 bytes (40 KiB)**. These are hypothetical case inputs, not measurements. Co-locating both requires **80 KiB**, which already exceeds one compute tile's **64 KiB** before any stack or staging. That particular placement fails the necessary capacity budget. [S1](sources.md#s1-target-model)

Possible alternatives are splitting the owner, retaining smaller chunks, shortening a proven lifetime, or recomputing. A MemTile's larger SRAM does not automatically make relocation feasible: it adds producer/consumer transfer, channel, BD, lock and route obligations. If the lifetimes do not overlap and the implementation legally aliases the buffers, the capacity conclusion changes; proving that reuse requires following DMA and the final reader too.

**Supply case — free SRAM does not provide another receive engine.** Assume one MemTile has two independently scheduled input paths and four independently scheduled result paths, all requiring distinct RX channels. This consumes **2 + 4 = 6 RX**, the tile's full RX count. Adding another independent receive path would require **7**, regardless of free SRAM or spare TX channels. These are hypothetical requirements evaluated against the declared channel count. [S3](sources.md#s3-aie2p-driver-declarations)

Consider a different owner/receiver, an upstream gather, or phase-multiplexing a channel after establishing framing and producer order. Multiplexing independent suppliers onto a receive chain requires an actual ordering/handoff mechanism; names or packet IDs alone do not establish that mechanism. Extra ping-pong buffers add storage and BDs on an existing channel. After changing the channel allocation, check BD parity pools and lock/credit balance as separate budgets.

## Check the candidate in this order

1. Record the exact device/partition, source and installed toolchain identity. Read the real `aie.device`, tile types and final coordinates. Rebind these sources if versions differ.
2. Count owned buffers, overlapping lifetimes and actual allocation ranges by tile/bank, including stack and generated data. Inspect linked program and data footprints separately. A live-set lower bound can reject a placement; passing it does not establish allocation.
3. For each transfer, record producer, consumer, units, offsets/strides/tail, direction/channel, every BD in the chain, local or remote lock owner and first/last-use events. Check [addressing](dma-addressing.md) and all applicable per-pool limits.
4. Inspect retained output from the relevant compiler stage: allocated buffers, assigned BDs/locks and final routes are different stages. Validate generation/reset state and supply order when the task permits execution; source totals cannot establish runtime progress.
5. State the unresolved obligation and next discriminating evidence. Preserve task gates; a resource total, route success or compiler success cannot replace numerical/state checks or complete-call performance measurement.

A resource inspector covers the objects and rules it implements. Check whether
it distinguishes compute, MemTile and shim resources and accounts for the
relevant buffers, descriptors, banks, stack, routes and synchronization. Use a
PASS for its stated checks; capacity and identifier checks alone do not
establish complete routing or forward progress.

No peak bandwidth, clock-derived throughput, exhaustive routability or current free-device claim is supplied. The compiler's bank groups do not document all physical ports or contention. The source declarations, compiler policy, candidate allocation and actual measurement remain distinct evidence levels.
