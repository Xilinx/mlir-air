# B16 with a 4096-wide hidden state

The 8B mapping uses four residual owners and a DMA-managed normalized-activation
cache in DDR. The target and drafter compile as separate programs from the same
AIR graph. Model configuration selects their layer count, FFN width, RoPE,
normalization and attention visibility.

## Storage and dataflow

Sixteen BF16 hidden rows occupy 128 KiB. Each of four compute cores holds four
residual rows (32 KiB). A small L2 relay broadcasts input activations and gamma;
each owner retains its own rows. Residual updates also use one broadcast, with
every owner draining the stream and accumulating only its rows. RMSNorm remains
independent per row and is computed once per normalization phase.

The B8 normalized-cache cores are occupied by the additional residual owners.
B16 therefore captures normalized chunks into a disjoint 128-KiB scratch span
after the logits in the output BO, then replays them through NPU DMA. The host
does not read, normalize or copy this scratch. The physical ABI includes it.

Owners 0/2 occupy column 2 rows 2/3; owners 1/3 occupy column 5 rows 2/3. Owners
2/3 send chunks by vertical cascade to owners 0/1, which serialize their own and
their peer's chunks into two output streams. A separate 512-word peer receive
buffer prevents overwriting an outstanding DMA read. Final hidden output uses
the same `[chunk,row,512]` framing; DDR scatter restores `[row,4096]`.

```text
four residual owners --normalize--> paired output streams
                                          |
                                    capture into DDR
                                          |
                                     replay by DMA -----+
attention output from MemTile --------------------------+--> xnorm receiver
GLU output from compute core ---------------------------+       |
                                                        projection cores
```

The three `xnorm` producers operate in decoder phase order. Capture, replay and
the consuming weight feeds belong to the same AIR region. Weight-bank selection
surrounds the complete layer, keeping the shared owner-output FIFO receives in
order: attention normalization, FFN normalization, then final hidden state.

Projections retain the packed Q4NX weight path. A 2×2 schedule reuses each
unpacked weight across two eight-row groups with four FP32 accumulators.
Attention runs two eight-row waves with native K layout, early K release and
P/V operand reuse. Per-wave query staging bounds storage. Independent receivers
collect a complete sixteen-row output before publishing it to projection;
the build uses AIR's automatic lock selection. C4096's last full-block prefix
is 4080. GLU uses a single 48-KiB input/output pair and derives its gate offset
from the block size; B8 retains its two-slot pipeline.

## AIR source and compiler responsibilities

Earlier experimental builds continued from generated MLIR-AIE checkpoints:
they edited DMA wait placement and merged phase-exclusive projection inputs,
and some variants repaired physical packet routes. Those repairs depended on
specific emitted task, channel and routing identities. This implementation
expresses its ownership, placement, packet endpoints and phase dependencies in
AIR source, then runs the normal AIR/MLIR-AIE/Peano flow.

The source mapping and the compiler fixes solve different parts of the problem:

| Part | Responsibility |
|---|---|
| `b8_graph.py`, `b16_owners.py` | Common projection receiver, scratch capture/replay, complete-layer weight selection, FIFO order and owner placement |
| `AIRRtToNpuPass.cpp` | Coordinate synthesized tail waits for ordinary and coalesced shim feeds in the same phase |
| `AIRToAIESchedulingUtils.cpp` | Preserve on-device packet producer allocations and allocate an L3 shim endpoint when the flow also has a DDR producer |
| `AIRToAIEPass.cpp` cleanup | Canonicalize unused index-switch results before AIR wait legalization |

For mixed DMA feeds, waiting for activation replay before starting weights can
deadlock: the projection core needs those weights to drain the activation
stream. Both feeds must start before their synthesized completion waits. The
fix extends the existing block/wave and phase-major scheduling rules, preserving
same-channel fragment boundaries. It does not infer arbitrary cross-region
dependencies; the source keeps capture/replay and consuming feeds together.

For mixed packet producers, a single aggregate memory-space field cannot
describe L3, L2 and L1 sources. Allocation examines actual source memory spaces,
retains existing L1/L2 allocations, and appends a shim endpoint for L3 sources.
These compiler rules depend on IR structure, not model names or block size.

The switch cleanup uses existing SCF canonicalization and is independently
reviewable. It fixed an intermediate source organization; the final Qwen target
produced identical AIRToAIE IR with and without that cleanup. The tested compiler
included it, so this stage comparison does not qualify a different installation.

Compiler regressions are `mixed_coalesced_paced_siblings.mlir` under
`mlir/test/Conversion/AIRRtToNpu`, and `packet_producers_with_shim.mlir` /
`packet_switch_dead_token.mlir` under `mlir/test/Conversion/AIRToAIE`.
They run without model checkpoints or NPU hardware. Other programs matching
these structures also use the fixes; the changes are not isolated behind B16.

Build and install AIR with these fixes before compiling this mapping. External
MLIR-AIE/Peano requirements follow the repository's normal setup. The build
fingerprints the package-resolved `aircc`, which runs passes in-process;
replacing only `air-opt` on PATH does not replace that compilation driver.

## Code map and checks

- `b8_config.py`: block-specific mapping and logical/physical buffer ABI.
- `b8_graph.py`: shared decoder phases, channels and host DMA supply.
- `b16_owners.py`: four owners, cascade pairing and output framing.
- `kernels/rms_b16.cc`: bit-preserving input/output framing around existing RMS
  and residual arithmetic; `kernels/glu_batch.cc`: block-aware gate/up layout.
- `build_target.py`: normal compilation and manifests. Model entry scripts
  provide the role-specific configuration and kernel defines.
- `check_decoder.py`, `verify_loop.py`, `check_model.py`: shared state,
  matching-arithmetic sequential-loop, and HF mutual-top-5 checks.

State checks include full-output repeatability, position and visibility,
the row-7/8 wave boundary, injected draft KV, and unchanged cache regions.
Loop checks compare output, acceptance and committed target KV with a sequential
oracle using the same block arithmetic. Native B1 GEMV is the performance
comparator; HF mutual-top-5 checks compare through the first divergence and do
not establish equality of subsequent divergent continuations.
