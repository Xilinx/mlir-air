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

The mapping is written entirely in AIR source and built by the normal
AIR/MLIR-AIE/Peano flow; no generated MLIR-AIE is edited afterwards.

| Part | Responsibility |
|---|---|
| `b8_graph.py`, `b16_owners.py` | Common projection receiver, scratch capture/replay, complete-layer weight selection, FIFO order and owner placement |
| AIRRtToNpu shim-feed pacing | Starts a paced feed and the coalesced phase it belongs to before awaiting either |
| AIRToAIE DMA allocation | Gives the DDR replay a shim allocation on a packet channel that also has MemTile and core producers |

The replay is a paced feed (strided, so never coalesced), issued in the same
region just before the coalesced weights of the projection it feeds. The
compiler takes a paced feed to belong to the first coalesced phase that ends
after it, so keep that order: a replay issued after its weights, in any phase
but the last, would see the weights awaited before it starts, and the
projection core would stall waiting for it.

The `xnorm` receiver has three producers: the DDR replay, the MemTile attention
output and the compute-core GLU output. It must stay a packet channel; the
compiler rejects a stream channel that mixes host and on-device producers.

The compiler regressions for both are `paced_coalesced_sibling_drains.mlir`
under `mlir/test/Conversion/AIRRtToNpu` and
`packet_producers_mixed_memspace_shim.mlir` under
`mlir/test/Conversion/AIRToAIE`. They need no checkpoints or NPU hardware.

Build and install AIR before compiling this mapping. External MLIR-AIE/Peano
requirements follow the repository's normal setup. The build fingerprints the
package-resolved `aircc`, which runs passes in-process; replacing only
`air-opt` on PATH does not replace that compilation driver.

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
