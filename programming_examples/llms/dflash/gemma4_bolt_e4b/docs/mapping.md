# Mapping

## Target

One dispatch verifies a block: 42 decode waves, one per layer, then 26 waves
of the vocabulary head. The graph (`b8_graph.py`) keeps `dflash/qwen3_4b`'s B8
structure: sixteen projection cores fed by a dual-channel weight stream, an X
memory tile that re-reads each projection's input, an egress gather, a
normalized-activation cache core and a residual core. Gemma4 adds:

| Gemma4 feature | Where |
|---|---|
| Post norms on the attention, MLP and per-layer-embedding outputs | The cache core normalizes each sublayer output as it arrives; the residual core applies the weight and adds it |
| Per-layer embedding | Two more projection phases (gate on the unnormalized residual, then projection); GELU(gate) times the token's per-layer input on the GLU core; the per-layer inputs come from the host |
| Layer output scale | Folded away: the device keeps layer L's residual divided by the product of the earlier scales (`weights_b8.residual_scales`), the host rescales the boundaries it reads |
| KV sharing | Shared layers read their source's slab; their own appends land in a scratch slab |
| Sliding window | Per-wave window; a sliding layer's K/V stream starts at the first block its first token sees |
| Head dims 256 and 512 | The graph is built at 512; see below |

Every layer is computed at the full-attention width, so all waves share one
graph. A sliding layer's 256-wide heads sit in 512-wide slots, q and k as
`[lo | 0 | hi | 0]` (the rope kernel pairs `i` with `i + 256`), v as `[v | 0]`.
Each layer class streams only its real weights (`b8_config.class_geometry`):

- a sliding layer's q/k/v rows arrive compact; the rope core spreads them
  (`kernels/rope_b8.cc`);
- the attention gather writes its output half-major, `[half][head][256]`, so a
  sliding layer's o projection takes weights for the first half of its x
  blocks only; a full layer's o weights are packed in the same column order;
- a KV-shared layer has no k/v rows;
- the projection cores keep the build's rounds and x blocks and take no
  weights past their class's, and no x for a round without weights.

Attention (`attention_graph.py`, `kernels/attn_b8.cc`) runs four shards per KV
head; each shard computes two tokens' four query heads, eight rows of an
`mmul<8,8,8>`. K and V stream from DDR in eight-key blocks through ring memory
tiles. The softmax's running max moves only when a score passes it by a fixed
margin, so the output's rescale is rare.

## Two independent sequences

`Config(batch=2, sequences=2)` keeps the graph and gives each of the two
tokens its own sequence:

- the projection cores take two rows with a bf16 GEMV that dequantizes each
  weight block once (`common/kernels/proj_batch.cc`, `PROJ_MM_BATCH=2`), in
  place of the eight-row mmul;
- each sequence has its own KV cache (`kv_sequence` elements apart), its own
  K and V streams and its own attention shard per KV head; a shard's frame
  carries its token's four query heads;
- the dispatch takes each sequence's length and append row, and a mask of the
  sequences with a token. An idle sequence attends one key, appends to its
  cache's last row, which no context reaches, and the projections skip its
  row.

## Drafter

The drafter is `dflash/qwen3_4b`'s B8 drafter graph at the draft's geometry
(`fused_decode`'s `gemma4-e4b-dflash-draft`): five Qwen3-style layers with
hidden 2560 and FFN 5120, over the target's vocabulary head. Its context K/V
is projected on the host from the target's five tap boundaries
(`context_prepass.py`). The device head's top eight candidates per row are
scored again in float32 against the target's int8 embedding rows, to which
the head is tied (`draft_weights.GemmaDrafter.propose_tokens`).

## Host

`npu_prefill.py` wraps `gemma4_e4b_q4nx`'s fused prefill, which also returns
the tap boundaries the drafter starts from. The target's per verify host work
is the block's embeddings, its per-layer inputs and its rope rows.

## Files

| File | Role |
|---|---|
| `b8_config.py`, `b8_graph.py`, `attention_graph.py`, `kernels/` | Target geometry, graph and kernels |
| `build_target.py`, `build_draft.py`, `build_baseline.py`, `npu_prefill.py` | Builds |
| `weights_b8.py` | Target weight packing and host inputs |
| `target_runtime.py` | Target runtime and the shared loop's decoder constructor |
| `draft_weights.py`, `draft_cpu.py`, `context_prepass.py` | Drafter weights, runtime and context K/V |
| `dflash_model.py` | What the shared DFlash code needs from this model |
| `reference_b8.py` | Float32 reference forward of the target |
| `tests/` | Checks run by `make verify` |
