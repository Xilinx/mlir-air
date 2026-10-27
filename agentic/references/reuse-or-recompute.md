# Decide whether to reuse an intermediate or produce it again

Use this method when a new shape, batch, consumer schedule, or ownership change makes an intermediate expensive to produce or difficult to retain. It also applies when a valid megakernel repeatedly normalizes, dequantizes, or converts the same data. Those operations share a decision pattern; evidence for one does not validate another.

The outcome is a conditional mapping choice, or a concrete inspection/measurement that can resolve the choice. Keep the task's reference, fixed acceptance criteria, permitted changes, and budget. Record the result in the existing [task record](task-record.md); do not create a second investigation report.

Here, “valid” means valid for the current requirement. A seed valid only for the old requirement is a rollback/reference asset; if no candidate meets the new requirement, preserve the seed and report the unmet requirement.

## Establish what can actually be reused

Start with the producer loop and one consumer-visible value. Identify its input generation, parameters, layer/phase, row or token, position, chunk, dtype, and representation. Include mask or mutable state when it affects the result. Equal shapes, addresses, or variable names do not establish equal values. For a proposed reuse interval, find the last write to every dependency and the first write that would invalidate the saved result.

Inspect the actual arithmetic before describing repeated work. Separate reductions or parameter preparation outside a consumer loop from elementwise production inside it. Check whether moving production changes rounding, conversion, reduction order, random/state updates, or which values are observed. A reuse plan must preserve required side effects as well as the saved value. If equivalence is unestablished, retain the original valid computation while locating the missing dependency or reference evidence.

Make a small work ledger for the same semantic interval in both candidates:

| Work or object | Evidence to seek |
|---|---|
| Intermediate production | Executed iteration bounds, conditions, tails, and work per value; distinguish source call sites from known dynamic execution |
| Reuse demand | Which consumers need the identical value, their order, and the number of uses before invalidation |
| Fill and replay | Additional stores, copies, conversions, DMA transactions, and synchronization introduced by saving the value |
| Downstream transport and compute | Work that remains, disappears, or changes route; replay is still supply work |
| Storage | Raw input and output that must coexist, saved result, staging, and other simultaneously needed allocations |

Label source-derived work and logical operand bytes as such. Compiler elimination, physical traffic, executed instructions, and critical-path time remain unknown until supported by the corresponding artifact or measurement. A large producer count alone does not establish a bottleneck.

## Compare feasible choices

Keep the current mapping in the comparison, even when caching looks attractive.

| Choice | Conditions favoring it | Costs and coupled changes |
|---|---|---|
| Produce again near each use | Few uses, cheap production, changing input, scarce storage, or communication more expensive than production | Retains producer work; may preserve useful overlap and simple local ownership |
| Retain locally | Value remains valid and the owner can retain it without displacing needed data or blocking another stage | Extend lifetime; update allocation, aliasing, readiness, last-use, and consumer access |
| Retain a chunk or shard | Full retention is infeasible but a subset can serve several consumers before eviction | Derive chunk layout and tail rules; coordinate consumer order, refill/eviction, and any gather or redistribution |
| Retain at another owner or memory level | A reachable owner has capacity and can supply consumers at the required rate | Add fill, replay, staging, channels, route, synchronization, and possibly extra latency or contention |
| Change the production/consumer schedule | Moving consumers closer together exposes reuse or releases storage sooner | Revisit dependencies, overlap, live state, and numerical order; this is a joint schedule change |

A compute tile, memory tile, or external buffer is a candidate only when the target backend can implement the required access and supply pattern. Do not assume spare tiles, multicast, arbitrary replay, or concurrent reads are available. Use [residency and supply](residency-and-supply.md) for the actual resource and protocol obligations.

## Specify the joint change before editing

For the selected candidate, name the affected producer, allocation, packing/layout, consumer loop, transport, and state/reset objects. Describe:

- **Stored identity and layout:** semantic value generation, dimensions, element/byte strides, conversion, chunk order, tails, and the consumer view. Reusing an existing allocation does not preserve its former layout automatically.
- **Ownership and lifetime:** who fills each unit, how readiness is observed, which readers still need it, and when overwrite or alias reuse becomes legal. DMA reads can outlive the core's enqueue operation. A partially filled cache can be legal if readiness and last use are tracked at the same granularity.
- **Supply and invalidation:** expected replay counts and order, fill/replay channels and buffers, completion/credit handoff, and the event that invalidates the value. Cover phase transitions, input/parameter updates, independent rows, repeat, and reset where relevant.

Raw input may remain live for residual or state updates after a transformed copy is ready. Do not overwrite it merely because the producer has finished. Preserve the original value version until every relevant reader has completed.

## Check validity, then compare net time

Use existing supported checks first, and retain their stated scope. Inspect the compiled allocation and supply objects relevant to the edited candidate; source agreement or a capacity lower bound cannot establish that the compiler implemented the intended owner, addresses, locks, or route. If an unknown organization leaves a property required by the frozen acceptance contract or an execution prerequisite unresolved, obtain targeted artifact inspection, a bounded probe, or alternate evidence satisfying that obligation; UNKNOWN alone cannot satisfy it. An unsupported finite checker or unknown detail outside those obligations is a diagnostic limit to report separately, not an extra gate. Do not infer PASS from resemblance to an old case.

Choose the correctness gate before comparison. Use the applicable registry full-output kernel comparison and the model token gate at their respective scopes. Exact output/state, repeat/reset or boundary comparisons can diagnose an arithmetic-preserving transformation or satisfy an explicit task/interface requirement; they are not extra global gates. If arithmetic or reduction changes, retain the agreed reference and numerical criteria. Do not relax a fixed gate after observing the candidate's error. Model/task quality remains a separate obligation when required.

Compare saved production with added fill/copy, waits, contention, and changed overlap on the actual critical path. A serialized model such as “production saved exceeds fill plus replay” is usable only if serialization and matched service costs are supported. Concurrent stages require an overlap/service-rate argument or a complete candidate measurement. Do not sum unrelated probe latencies into a fused breakdown, substitute nominal bandwidth for an actual path, or turn an operation-count ratio into a speedup.

Use a probe when it distinguishes competing explanations: for example, a matched producer/supply path can test whether replay is faster than production while preserving consumer demand, buffering, topology, and completion boundaries. Report any unmatched drain/dispatch work. Otherwise compare valid complete candidates with the same timing boundary and workload, recording artifact/toolchain identity, order, warmup, noise, and relevant device state. If evidence cannot rank them, record `ABSTAIN` and the next discriminating observation; retain the best valid measured candidate.

## Boundaries that change the advice

- **The value changes between uses:** invalidate and refill per generation, narrow the reuse interval, or retain production. Sharing across those uses is not a valid optimization.
- **There is one use:** saving the result removes no repeated production. Prefer the existing valid mapping unless a separate supply, placement, or scheduling benefit is supported.
- **The full saved result does not fit with required live storage:** reject that full-retention configuration. Compare chunking, sharding, another owner/level, or production again; do not conclude that the workload itself is infeasible.
- **Copies or waits consume the anticipated saving:** preserve the current design or revise supply/scheduling. Fewer producer operations do not override a slower valid complete call.
- **Compiler or protocol effects are unknown:** identify whether the missing allocation, address, readiness, or timing evidence affects a required acceptance/execution property or only the current diagnosis. For a required property, seek an in-budget inspection, probe, or alternate evidence; otherwise report the diagnostic limit separately. An unimplemented candidate remains a proposal.

Return the observation, hypotheses with evidence, affected objects, candidate and coupled edits, next check/measurement with its expected distinguishing result, and remaining unknowns/acceptance conditions.

## Illustrative work and storage accounting

Suppose a projection consumes N normalized chunks on each of R passes. A design
that produces each chunk again performs N×R productions. Retaining them reduces
production to N, but still supplies N×R chunks and introduces fill/replay work.
An RMS factor computed once per row is already shared; count the repeated
elementwise scaling separately. A residual update changes the input and
invalidates the saved normalization.

For a BF16 slab with B rows of width D, full retention requires 2×B×D bytes,
plus other simultaneously live buffers, staging and stack. Compare that sum
with the selected owner's usable capacity from the [hardware scope](hardware/README.md).
If it cannot fit, inspect chunking, sharding, a different owner or recomputation.

This accounting identifies candidate changes. Choose among them using the
applicable correctness gate and measured complete-call time.
