# Mapping and code map

The decoder retains the same Transformer operations as the native Qwen3-8B
B1 megakernel. B8 adds a token dimension, per-row RMS reductions and positions,
causal target attention (bidirectional draft-block attention), eight logits
rows, and the target hidden boundaries used by DFlash. Packed projection
weights are reused across all eight token rows in the `aie::mmul<8,8,8>` path.

| Property | Target | Drafter |
|---|---|---|
| Layers | 36 | 5 |
| Hidden / FFN width | 4096 / 12288 | 4096 / 12288 |
| Query / KV heads | 32 / 8 | 32 / 8 |
| Head width | 128 | 128 |
| Attention inside the block | Causal | Bidirectional |
| Vocabulary | 151936 | Same target head |
| Target hidden boundary slots | 2, 10, 18, 26, 34 | Used by context prepass |

## Why the 4B physical graph is not reused verbatim

Eight BF16 rows of width 4096 occupy 64 KiB, the full compute-tile L1 capacity.
A residual owner also needs gamma vectors, staging buffers and stack. The 8B
mapping therefore uses **two owners of four rows each** and two corresponding
normalized-activation caches. The 4B B8 graph needs one owner and one cache.

Coordinates below use the device's compute rows 2–5; MemTiles occupy row 1.

| Function | Placement / data ownership |
|---|---|
| Residual and RMSNorm owners | (2,2), (5,2); four token rows each |
| Normalized-activation caches | (2,3), (5,5); replay the corresponding four rows |
| RoPE | (2,5) |
| Attention Q/K relays | (2,4), (5,4) |
| Projection | The existing 16-core cascade array |
| Attention compute | Eight shared query/head shards |
| GLU | The remaining compute core |
| Residual update demultiplex | MemTile2 distributes four-row halves |
| Normalized-input join | MemTile3 reconstructs the projection input stream |
| Projection activation fanout | MemTile4 |
| O/down resident replay | MemTile5, independently allocated drain rings |

All 32 compute tiles are occupied. The source also changes projection packet
framing (`PAIR_ROWS=1`), GLU stream order, and output gathering to match these
owners. The separate O/down replay rings are essential: down cannot be produced
until the preceding O replays have completed. The compiler must not alternate
these independent buffers on one lock-driven drain ring.

Both examples compile shared arithmetic sources. The preserved 8B profile uses
four-query shards and fused P/V rescaling. The later 4B native-K, early K-release
and P/V operand-reuse options remain enabled only in the selected 4B profile.
Sharing source does not silently enable new arithmetic or mapping optimizations.

## Follow the implementation

| File | Role |
|---|---|
| [selected_config.py](../selected_config.py), [target_config.py](../target_config.py) | Model-role configuration, selected C4096/B8 mapping, upstream geometry adapter and logical/physical buffer ABI |
| [batch_graph.py](../batch_graph.py) | Complete AIR graph: weight feed, activation fanout, projection packets, RoPE, KV append, attention and GLU |
| [owner_graph.py](../owner_graph.py) | Two residual owners, two caches, four-row update routing and normalized-input join |
| [common/fused_attention_graph.py](../../common/fused_attention_graph.py) | Shared query/KV distribution, query/head shards and output gather |
| [common/kernels/](../../common/kernels/) | Shared projection, attention, RMS/residual, cache, GLU and context-prepass arithmetic |
| [build_target.py](../build_target.py), [build_bfp_prepass.py](../build_bfp_prepass.py) | Peano objects and standard AIR compilation; source/artifact/tool identities and optional compatible object reuse |
| [target_runtime.py](../target_runtime.py), [common/decoder.py](../../common/decoder.py) | Model-specific BO/weight binding; shared dispatch, logits/hidden access and KV operations |
| [common/buffers.py](../../common/buffers.py) | Disjoint logical tensor views in bounded physical BOs; offsets originate in AIR subviews |
| [runtime.py](../runtime.py), [run.py](../run.py) | Model initialization, NPU prefill, single-prompt streaming and resident chat |
| [common/speculative.py](../../common/speculative.py) | Shared draft/verify/accept/commit loop, EOS/budget handling and timers |
| [npu_prefill.py](../npu_prefill.py), [prefill_weights.py](../prefill_weights.py) | Existing upstream Q4NX prefill with tapped outputs and mapped, streamed BF16 weights |
| [common/prepass.py](../../common/prepass.py), [common/build_prepass.py](../../common/build_prepass.py) | Standalone NPU context preparation; width is 4096 here, 2560 for 4B |
| [prepare_target.py](../prepare_target.py), [prepare_draft.py](../prepare_draft.py) | Raw target block reordering and drafter quantization; inverse packing checks |
| [model_paths.py](../model_paths.py), [resources.py](../resources.py), [prepare_weights.py](../prepare_weights.py) | Pinned resources, persistent cache preparation and identity checks |
| [benchmark_data.py](../benchmark_data.py), [common/benchmark_runner.py](../../common/benchmark_runner.py), [common/paired_benchmark.py](../../common/paired_benchmark.py) | Model-specific prompt states, common timing/aggregation and paired studies |
| [baseline.py](../baseline.py), [build_baseline.py](../build_baseline.py) | Native upstream B1 xclbin builder and benchmark adapter |
| [verify_loop.py](../verify_loop.py), [tests/](../tests/) | State, arithmetic, loop and model-token verification |

The target and drafter are separate programs built from this mapping with their
own role/layer/visibility settings and weights. The context prepass is a third
program. It projects `[8,20480]` target features into `[8,4096]` drafter features,
normalizes them, and projects/rotates the five layers' context K/V. The host
injects only committed rows. A long initial prompt is processed in eight-row
prepass chunks, with a padded final chunk whose unused rows are not committed.
