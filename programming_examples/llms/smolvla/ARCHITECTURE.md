# SmolVLA SigLIP Vision Encoder on NPU2 — Architecture

Companion to [README.md](README.md) (overview, quick start, results). This doc
covers how the per-layer kernel chain and the runtime are organized.

**Scope.** SmolVLA is three stages; only the SigLIP vision encoder and the
connector run on the NPU here. The SmolLM2-360M backbone (seq 241) and the
action expert (seq 50, ×10 denoise steps) were both ported and verified, and
both measured slower than the CPU at their shapes, so they run as unmodified
lerobot CPU code; see [docs/profile.md](docs/profile.md) for the numbers. Those
two ports are not part of this example.

## Vision Config

12 layers, seq_len=1024 (32×32 patches of a 512×512 image), emb_dim=768,
n_heads=12, head_dim=64, hidden_dim=3072, attn_scale=1/8, LayerNorm eps=1e-6,
BF16. Attention is **bidirectional with no mask**. Connector: pixel-shuffle
(space-to-depth ×4) → 64 tokens × 12288 → linear → 960.

## Per-Layer Kernel Sequence

Pre-norm, 18 operations per layer:

```
LayerNorm(ln1)                              1024×768
Q, K, V GEMM                                1024×768×768    ×3
  + bias                                    1024×768        ×3
FlashAttention, non-causal, 12 heads        1024², 12/12
O GEMM + bias + residual                    1024×768×768
LayerNorm(ln2)                              1024×768
fc1 GEMM + bias                             1024×768×3072
GELU-tanh                                   1024×3072
fc2 GEMM + bias + residual                  1024×3072×768
```

Then once per image: `post_layernorm` (1024×768), host pixel-shuffle, and the
connector projection (64×12288×960).

Every kernel is looked up by shape in `../kernel_registry/registry_lookup.py`,
so tile sizes and the GEMM method are never hardcoded here. Affine LayerNorm
and GELU-tanh were promoted into the registry by this work.

## Fused ELFs: 18 launches → 3 dispatches

`smolvla_vision_builders.py` stitches the per-layer launches into three
multi-launch ELFs via `shared/infra/stitching.stitch_elf`:

| ELF | launches | contents |
|---|---|---|
| `vit_ln_qkv` | 7 | ln1 + Q/K/V GEMM + three on-device bias-adds |
| `flash_attn` | 1 | the registry FlashAttention ELF, unchanged |
| `vit_o_ffn` | 10 | O + bias + residual + ln2 + fc1 + bias + GELU + fc2 + bias + residual |

12 layers × 3, plus post-LayerNorm and the connector, is **38 dispatches per
image** (121 unfused). Measured 368 ms → 141.6 ms.

The gain is mostly not driver overhead. It comes from moving the per-Linear
bias-adds and the two residual adds on-device, which removed a
bf16 → f32 → bf16 host round-trip per operation: host-side gap per image fell
212 ms → 4 ms, and encoder-output cosine *improved* 0.945 → 0.9906, because
that round-trip had been re-quantizing every intermediate.

## Runtime Flow

```
warmup_npu()                     once per process, outside any timing
  VisionRuntime.__init__
    load_vision_weights()        safetensors → transpose → bf16     ~355 ms
    ensure_kernels()             load the 5 cached ELFs              ~34 ms
  .warmup()
    encode([zeros])              XRT context + BO alloc + static
                                 weight upload; result discarded    ~345 ms

encode(images)                   per inference
  im2col_patch_embed ×N          host, before the thread clamp
  for each image:
    12 × [vit_ln_qkv → flash_attn → vit_o_ffn]
    post_layernorm
    pixel_shuffle (host) → connector GEMM
  → (N, 64, 960) RAW connector output         ~166 ms/image (earlier session; see docs/profile.md)
```

`encode` returns what `vlm_with_expert.embed_image` returns: the raw connector
output. lerobot applies the sqrt(960) scale afterwards.

`get_vision_runtime()` is a process-wide singleton. That is forced by the
integration point rather than chosen: lerobot calls `embed_image(image)` with
one argument, so there is nowhere to thread a runtime handle through. The
siblings call `prepare_runtime()` from their own `main()` and pass the cache
down explicitly.

## Key Design Patterns

**The splice.** `smolvla_inference.py` swaps `vlm_with_expert.embed_image` for
the duration of one `embed_prefix` call and restores it in a `finally`.
`embed_prefix` itself stays lerobot's — it is an *assembly* function and SigLIP
is one line inside it, so the sqrt(960) scale, the language embedding, the state
projection and the mask construction all remain upstream code. That is what
makes the comparison against the pure-CPU baseline meaningful.

**All images encoded in one call.** The wrapper encodes every camera up front
and hands the results out through the swapped `embed_image`. Encoding lazily,
one image per call, is simpler — one swap, no counter — and passes the gate,
but measured 818 → 920 ms end to end against a 913 ms CPU baseline in the same
session (absolute numbers predate the current gate input; the comparison holds).
The cause is not established; the ruled-out suspects are recorded in
`run_hybrid_forward`.

**Static weight BOs.** Per-layer weights pass `static_input_indices`, so they
upload once per `bo_key` and are reused across all 12 layers and every later
inference. Kernel outputs and scratch pass `intermediate_indices`, so the host
does not write them before the call.

**tile_n-keyed object names.** `compile_gemm_mm` bakes `DIM_M`/`DIM_N`/`DIM_K`
into the external `mm.o` at compile time, but the shared helper names that
object from `tile_n` alone. At seq=1024 the vision GEMMs resolve to two distinct
`tile_n` (96 for q/k/v/o and fc2, 128 for fc1) under one "drain" method, so
`_force_tile_n_suffix` forces the tile_n-keyed name. Without it a GEMM links a
stale generic `mm_m32.o` with the wrong baked `DIM_N` — the first attempt scored
cosine 0.07. `DIM_K` is still unkeyed; see `docs/explain.md` §4.

**Two host steps, deliberately.** The im2col patch embed is a one-time reshape
before the layer loop, not hot-loop work. The connector's pixel-shuffle is pure
space-to-depth with zero arithmetic, verified bit-exact against HF. The
connector's actual math, the 64×12288×960 projection, runs on the NPU.

**FlashAttention is why this stage wins.** SigLIP attention is bidirectional
with no mask and an even 12 heads, which is exactly what the registry kernel
expresses: two heads per compute unit, filling the 8×4 array. The other two
stages need real masks and this kernel applies none, which is the single largest
reason they stay on the CPU.
