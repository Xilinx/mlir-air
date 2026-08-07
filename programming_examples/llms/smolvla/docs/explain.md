# Implementation guide: SmolVLA vision encoder on MLIR-AIR

How the SigLIP vision encoder is mapped onto NPU2, how its kernels are compiled
and stitched, and how the result is spliced back into the unmodified LeRobot
pipeline.

For the measured numbers see [`profile.md`](profile.md); for what is verified
and how, [`correctness.md`](correctness.md).

---

## 1. What is on the NPU

Per camera image, the encoder is 12 identical layers over **1024 patch tokens**
of width 768, then a post-LayerNorm and a connector projection:

```
image (3,512,512)
  → im2col patch embed (host, one-time)      → (1024, 768)
  → 12 × [ LayerNorm → q/k/v → MHA → out-proj → +residual
           → LayerNorm → fc1 → GELU → fc2 → +residual ]
  → post-LayerNorm                            → (1024, 768)
  → pixel-shuffle (host reshape)              → (64, 12288)
  → connector projection                      → (64, 960)
```

Everything in that list runs on the NPU except two host steps, both deliberate:

- **im2col patch embedding** — a one-time reshape before the layer loop, not
  hot-loop work. Moving it on-device would be a Conv/im2col kernel for no
  throughput gain.
- **pixel-shuffle** — a pure space-to-depth reshape with zero arithmetic,
  verified bit-exact against HuggingFace. The connector's only actual math, the
  64×12288×960 projection, does run on the NPU.

## 2. Kernels, and where they come from

Every kernel is looked up in the shared registry
(`../../kernel_registry/registry_lookup.py`) by shape, so tile sizes and the
GEMM method are never hardcoded here:

| Operation | Kernel | Shape |
|---|---|---|
| q/k/v/out projections, patch embed | GEMM bf16→bf16, drain | 1024×768×768 |
| MLP fc1 | GEMM | 1024×768×3072 |
| MLP fc2 | GEMM | 1024×3072×768 |
| attention | FlashAttention, non-causal | 1024², 12/12 MHA |
| LayerNorm (affine) | `layer_norm` | 1024×768 |
| GELU-tanh | `gelu` | 1024×3072 |
| bias-adds, residuals | `eltwise add` | 1024×768 |
| connector projection | GEMM | 64×12288×960 |

Two of these — affine LayerNorm and GELU-tanh — were promoted into the registry
by this work and have their own detail pages there.

**FlashAttention is why the vision stage wins.** SigLIP attention is
bidirectional with no mask and an even 12 heads, which is exactly what the
registry kernel expresses: it packs two heads per compute unit and fills the
whole 8×4 array. The other two SmolVLA stages need real masks, and the kernel
applies none — that is the single largest reason they stay on the CPU.

## 3. Fusion: 121 dispatches → 38

A layer is 18 kernel launches. Issuing 18 separate programs would make the host
pay its per-dispatch cost 18 times, and would round-trip every intermediate
through the host in fp32. Instead `smolvla_vision_builders.py` stitches them
into **three multi-launch ELFs** with `shared/infra/stitching.stitch_elf`:

| Program | Launches | Contents |
|---|---|---|
| `vit_ln_qkv` | 7 | affine LayerNorm + Q/K/V GEMM + three on-device bias-adds |
| `flash_attn` | 1 | the registry FlashAttention ELF, unchanged |
| `vit_o_ffn` | 10 | O GEMM + bias + residual + LayerNorm + fc1 + bias + GELU + fc2 + bias + residual |

12 layers × 3, plus the post-LayerNorm and the connector, is **38 dispatches per
image**. Measured effect: 368 ms → 141.6 ms, a 2.6× improvement on identical
kernels.

The gain is *not* mainly from removing driver overhead. It is from moving the
per-Linear bias-adds and the two residual adds on-device, which eliminated a
bf16 → f32 → bf16 host round-trip per operation. Host-side gap per image fell
from 212 ms to 4 ms. Accuracy also *improved* as a side effect (encoder-output
cosine 0.945 → 0.9906), because that round-trip had been re-quantizing every
intermediate.

## 4. A fused-ELF trap worth knowing

`compile_gemm_mm` bakes `DIM_M`, `DIM_N` and `DIM_K` into the external `mm.o`
microkernel at compile time, but the shared helper `disambiguate_by_tile_n`
names that object from `tile_n` alone. At seq=1024 the vision GEMMs resolve to
two distinct `tile_n` (96 for q/k/v/o and fc2, 128 for fc1) under the same
"drain" method, so a GEMM built in isolation would link a stale generic
`mm_m32.o` with the wrong baked `DIM_N` and silently produce garbage — the first
attempt scored cosine 0.07. `_force_tile_n_suffix` in
`smolvla_vision_builders.py` forces the tile_n-keyed name so every ELF links the
object it was compiled against.

## 5. Splicing into LeRobot

`smolvla_inference.py` wraps `policy.model.embed_prefix` and, only for the
duration of that call, swaps `vlm_with_expert.embed_image` for one that serves
results the NPU already computed. All camera images are encoded in one call into
the runtime, then handed out one at a time as LeRobot's own code asks for them.

This keeps the sqrt(960) scale, the pad and attention masks, prefix assembly,
the backbone and the action expert as untouched LeRobot code — which is what
makes the comparison against the pure-CPU baseline meaningful. The wrapper is
always restored in a `finally`.

## 6. One process

`smolvla_runtime.py` holds a process-wide `VisionRuntime`: weights, compiled
ELFs, the XRT context and the device buffer objects are created once and reused
by every inference. An earlier two-process design paid ~585 ms per inference in
process spawn, weight reload and ELF load — none of it NPU cost.
