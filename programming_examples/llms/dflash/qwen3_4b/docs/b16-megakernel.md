# Qwen3-4B B16 decode megakernel

`make compile BLOCK_SIZE=16` builds separate target and drafter programs.
It uses the same pinned models, Q4NX packing, NPU prefill, eight-row BF16/BFP
context-preparation kernel and host loop as B8. Context preparation handles a
sixteen-row update in two dispatches. No new checkpoint download is needed.

## Mapping and ownership

| Path | B8 | B16 |
|---|---|---|
| Projections | One token group × four output subtiles | Two eight-token groups × two output subtiles |
| Matrix primitive | BF16/BFP `mmul<8,8,8>` | Same primitive; each unpacked weight serves both groups |
| Full hidden activation | 40 KiB | 80 KiB, partitioned between two owners |
| Residual and normalization | One eight-row owner plus one cache | Two eight-row owners plus two caches |
| Attention | One eight-row wave | Two waves through the same eight shards |
| Final valid C4096 prefix | 4088 | 4080 |

`batch_graph.py` selects `b16_owners.py` only for B16. Each owner keeps eight
residual rows (40 KiB), two gamma vectors (10 KiB), two 2-KiB staging buffers,
a 1-KiB source buffer, 32-byte normalization factors and a 6080-byte stack. It computes each row's
RMSNorm once per phase and sends 128-wide chunks to its cache. The cache's
`b16_normalized_cache_chunk` regroups pairs of chunks into 256-wide rows
without another normalization. The second cache receives the first cache's
eight rows into a separate 4-KiB buffer and combines them with its own eight
rows in an 8-KiB output buffer. Together with its 40-KiB slab and 6080-byte
stack, this fits L1. Separate input/output buffers prevent a following peer
DMA from overwriting a packet still being transmitted. B8 retains its
512-wide feed. The B16 X-memory staging and DMA strides use 256 consistently.

The initial residual/gamma input enters one shim endpoint, then a small L2
relay broadcasts it to both owners. Each owner retains only its eight rows.
The normalized output goes directly from the second cache to the projection
input memory tile, so it does not share the FFN-down DMA transmitter.

The shared projection kernel retains B8's 1×4 schedule and adds B16's 2×2
schedule. Four FP32 accumulators cover two token groups and two output subtiles;
the quantized weight stream and affine dequantization rounding are unchanged.

The shared attention graph derives causal bounds separately for each wave.
The target sees only the prefix and preceding rows; the drafter sees the full
sixteen-row candidate block. Each shard sends two eight-row outputs into one
sixteen-row receive transaction. AIR's automatic lock strategy keeps the four
receivers independent and makes the projection reader wait for all four.
This matters for both correctness and progress: publishing each half as a full
slab exposes unwritten rows, while serializing full-slab receivers can stall
shared KV supply. B16 therefore uses `use_lock_race_condition_fix_auto`, already
provided by upstream AIR. B8 retains its existing lock selection. RoPE
parameters still differ for every row; a contiguous LUT transfer avoids
queuing sixteen shim requests before projection weights can run.

The larger mapping changes both memory and DMA occupancy. QKV, query and attention-output staging share MemTile 5; FFN/down staging
moves to MemTile 3, while residual updates stay on MemTile 2. The query buffer
holds eight rows per wave (64 KiB), rather than a full sixteen-row slab. Weight memory tiles already use their six receive channels for
weights and projection-result gather and cannot simply host another circuit
receiver. These placements are expressed in AIR source, not patched into a
compiled transaction or routed program.

## Build, tests and timing

Use `BLOCK_SIZE=16` consistently with `compile`, `run`, `chat`, `verify`,
`profile`, `benchmark`, `full20` and `full80`. B8 remains the default.
`run_npu2_*_b16.lit` entries use their own `dflash-b16-build` directory and
`qwen3_4b_dflash_b16` performance identity. Existing nightly suite filters
include them. Full20 selects four cases per dataset; Full80 remains opt-in.
Both use complete-loop decode timing including initial NPU preparation.
Acceptance rate divides accepted drafts by the actual proposed drafts.

State checks cover every output word, repeatability, future masking, the
row-7/row-8 attention-wave transition, KV injection and unmodified cache
regions. The loop check uses a same-B16 sequential oracle and compares
committed target KV after every round. The standard two-prompt HF BF16 top-5
check is a separate model gate; it does not assert equality with B1 GEMV.
