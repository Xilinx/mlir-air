# Change residency and supply together

Use this method when shape, parallelism, owner, layout, buffering, or execution boundaries change; when compilation reports resource/routing problems; or when a candidate stalls, receives wrong data, or corrupts state despite apparently sufficient capacity. Its purpose is to decide whether data can remain valid where it is stored and reach each consumer correctly, with forward progress and useful throughput.

Keep the task's fixed acceptance criteria and authorized scope. Reuse unaffected evidence only when its source/configuration, toolchain, artifact, and assumptions still match. Add the relevant findings to the shared [task record](task-record.md). If the change concerns storing a repeated intermediate, also use [reuse or recompute](reuse-or-recompute.md).

Here, “valid” means valid for the current requirement. A seed valid only for the old requirement is a rollback/reference asset; if no candidate meets the new requirement, preserve the seed and report the unmet requirement.

## Trace the affected value from allocation to last use

Start at the changed object or earliest observed mismatch; expand to the producers, consumers, and shared resources that can explain it. A small explicit relation is more useful than a nominal whole-graph inventory.

| Relation | Establish from source/configuration | Confirm in the actual compiled/runtime mapping when available |
|---|---|---|
| Semantic identity | Layer/phase, row/token, position, chunk, parameter/state generation; producer and each consumer's expected value | Packing/data provenance and actual bound buffers; correct value at the first affected boundary |
| Storage | Shape, dtype, layout, allocation owner, live interval, aliasing, fill and final reader | Physical tile/memory, buffer extent/address/bank, alignment, compiler-added staging, stack and reserved/runtime storage |
| Transfer | Units, source/destination offsets, strides, chunk/tail lengths, count and order | DMA channel, BD address/length/stride/next descriptor, segment-relative addressing, packet/circuit route and endpoint |
| Handoff | Readiness, consumption completion, free/ready credits, initial state, repeat/reset policy | Lock/credit actions and initial values, receive-chain order, buffer-slot rotation, shared arbiters and backpressure |

Keep source expectation and artifact observation separate. A logical herd hint is not a physical coordinate. A tensor extent is not a transfer schedule. Equal total bytes do not establish that the consumer receives the correct elements, segment, or generation.

For each relevant allocation, identify the fill/read/last-use/overwrite events, including outstanding DMA. Storage may be shared only when access intervals are compatible and the implementation actually reuses the memory. Conversely, summing all conceptual tensors can overstate demand when valid reuse is implemented. Derive a necessary live-set lower bound, then inspect actual allocated extents, alignment gaps, bank placement, stack/staging and reserved regions. Do not double-count aliased storage, and do not assume the compiler aliases it merely because lifetimes could permit it. Static stack declarations are not measured dynamic stack depth.

If the necessary live set already exceeds an owner's usable storage, reject that particular placement. If it fits, capacity is still only one condition: actual placement, addressing, supply, and synchronization remain to be checked.

## State what kind of limit the evidence establishes

| Limit | Evidence and implication |
|---|---|
| Device/architecture limit | Versioned device facts: memory per relevant owner/bank, channel/BD/lock/address resources, connectivity or access rules. Excludes mappings that require more than the supported resource. |
| Implementation limit | The selected buffers, retained values, address representation, framing, or schedule. A different implementation may remove the limit. |
| Compiler/backend restriction | A supported lowering, allocation/routing behavior, or known workaround in a specific version. Inspect emitted objects or diagnostics before generalizing it to hardware. |
| Measured performance limit | A particular path and demand pattern under recorded conditions. It is a reference for matching mappings, not a universal hardware ceiling. |

Read [versioned hardware facts](hardware/README.md) and select the actual target/partition before using any number. Record the constraining object and version. For example, an encoded offset width can limit a transfer even when total system memory is sufficient. A compile failure can be a placement or backend limitation; a successful compile does not establish correct runtime supply or deadlock freedom.

## Verify supply as an ordered protocol

Follow a unit of data through producer framing, host packing if any, intermediate storage, receive descriptors, and the consumer view. Check element identity and address units as well as totals. Include the partial final chunk and crossings of layer, phase, segment, or repeated-call boundaries.

For rotating buffers, compare producer/DMA slot state with the core's slot state at entry and exit. Determine which side resets and which carries phase. Only then derive the compatible frame counts or reset policy. A two-buffer implementation may impose a parity condition; the operator itself does not imply even counts.

When independent suppliers feed a shared receive chain, find the mechanism establishing their arrival order relative to the active receive descriptor. Source channel names, source order in Python, or packet IDs alone do not prove that order. Check what the target's routing/receive semantics actually guarantee. If order is unestablished, mark the dependency `UNKNOWN`/`INCONCLUSIVE`; the topology alone need not prove failure.

Trace waits and releases across shared resources. A producer blocked on output may prevent a downstream consumer from receiving another required input. Include routes/arbiters, receive capacity, buffer credits and terminal/drain behavior; available bytes cannot rule out this cycle. A finite checker can establish only the relationships it recognizes. For an unsupported topology, inspect the relevant compiled chains/locks and source endpoints or propose a bounded trace/probe that distinguishes an ordering failure from simple backpressure.

## Compare changes that preserve the required relations

| Observation | Candidate changes | Objects to change or recheck together |
|---|---|---|
| Owner's necessary live set is too large | Shorten lifetimes, legally reuse storage, chunk/shard, move an owner, use another storage level, or produce values again | Reader completion, allocation/aliasing, chunk layout, fill/evict schedule, consumer order, transport and additional staging |
| Capacity fits but placement/routing fails | Change placement, banks, buffering, channel use or partition; reduce competing traffic | Physical endpoints, buffer addresses, locks/BDs, route/arbiter assignment, producer and consumer service rates |
| Sizes agree but values are misgrouped | Align packing, framing, consumer view, address/stride, tails, or BO segment binding | Actual packed data regeneration, builder/kernel constants, runtime arguments, lowered transfers and numerical/state gate |
| Boundary phase disagrees | Derive compatible framing, carry phase consistently, reset it explicitly if supported, or choose another buffer policy | Both core and DMA rotation, credit totals, initial/repeat state, and throughput effects |
| Independent supply has no proven order | Establish an ordering handoff, serialize the required supply, or allocate independent receive paths where supported | Supplier schedule, receiver chains, channels/BDs/locks, convergence at the consumer, and resource pressure |
| More compute increases waiting or imbalance | Adjust partition, broadcast/relay/gather or buffering; retain the existing partition when supply cannot sustain more | Input and output paths, partial state/merge arithmetic, stage overlap, and complete-call time |

No row mandates adding a tile, doubling buffers, pinning channels, or applying a historical stride. Select a change supported by this mapping's resources and semantics. Complete the related source/configuration edits before treating intermediate inconsistencies as a candidate failure.

## Select the next evidence and acceptance check

Use the smallest check that can distinguish the current hypotheses, then validate the affected scope of the final candidate:

- **Storage hypothesis:** inspect actual allocation addresses/banks and generated staging/stack; expect the identified live objects to coexist without forbidden overlap. A lower-bound failure can reject a placement before compilation, but a lower-bound pass cannot accept it.
- **Wrong-value hypothesis:** trace the first mismatching semantic element through packing, offset/stride and consumer interpretation; expect agreement at tails and boundary transitions as well as the interior. Rebuilding code does not automatically regenerate stale packed weights or host caches.
- **Order/progress hypothesis:** inspect actual supplier endpoints, receive chains, lock initialization and routes; where execution is authorized, use a bounded completion/order observation with distinctive payloads and relevant repeat/reset state. Record timeout or incomplete evidence without guessing which supplier is wrong.
- **Performance hypothesis:** compare matching workload and timing boundaries after the applicable correctness/state gate. Include added copies, routing contention, waits, changed overlap, and host work. Match path/topology/demand for a probe; do not sum isolated stage times into a fused critical path or infer a hardware ceiling from one mapping's throughput.

Apply the relevant full-output kernel and full-model token gates. For layout, residency or transport changes, exact output/state and repeat/reset comparisons are useful diagnosis or task/interface checks. Changed arithmetic retains its agreed reference and tolerance; do not add a universal extra state suite. A fixed task gate cannot be weakened to accommodate the candidate. Component PASS does not establish full-model quality or every input's correctness.

Bind observations to the candidate's actual source, configuration, packed inputs, toolchain and artifacts. Report partial evidence at its own level: source consistency, compiled mapping, saved/new execution, and measured performance. An unresolved backend or mapping property blocks acceptance only when the frozen task contract or an execution prerequisite requires it; UNKNOWN alone cannot satisfy that requirement. Seek targeted inspection, a bounded probe, or alternate evidence that satisfies the same obligation, and retain any candidate valid for the current requirement while resolving it. An unsupported finite checker or unknown detail outside those obligations is a diagnostic limit, not an additional gate: report it separately and assess acceptance against the frozen contract.

Return observation, hypotheses/evidence, affected objects, candidate/coupled edits, the next check and expected distinguishing result, and unknowns/acceptance conditions.

## Public source and illustrative protocol examples

**Framing and phase.** The Qwen3-4B adaptation in [#1916](https://github.com/Xilinx/mlir-air/pull/1916)
used 512 values per GLU frame: 256 up plus 256 gate. Its development notes
describe a copied consumer header expecting 1024 while the builder supplied 512.
[#1921](https://github.com/Xilinx/mlir-air/pull/1921) added a builder/header
assertion; inherit it as an existing check. The conditional even-frame rule
comes from a core that restarts slot phase while DMA carries it. A core that
also carries phase, a valid reset, or another receive organization changes that
obligation.

Even when dimensions agree, packing can put an up value where the consumer
expects a gate value. Trace element identity through producer, packed stream
and consumer indices in addition to checking total lengths. Related source
references are in [attribution](historical-attribution.md).

**Independent suppliers.** Consider a receive chain that expects input A before
input B while separate channels supply them independently. Buffer capacity
alone cannot establish that order. Trace readiness and consumption through the
lowered descriptors and locks; establish an ordering handoff, serialize supply,
or use independent receive paths when the target permits them. Check forward
progress and the consumer's actual data order on the selected implementation.

This is a protocol inspection example. The correct remedy depends on the
mapping's supported channels, ordering and resource use.
