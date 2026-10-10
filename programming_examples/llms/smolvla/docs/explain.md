# Implementation guide: SmolVLA on MLIR-AIR

How the SigLIP vision encoder is mapped onto NPU2, how its kernels are compiled
and stitched, how the result is spliced back into the unmodified LeRobot
pipeline, and how the opt-in all-NPU path runs the backbone and the action
expert.

For how to measure see [`profile.md`](profile.md); for what is verified and
how, [`correctness.md`](correctness.md).

---

## 1. What is on the NPU

Per camera image, the encoder is 12 identical layers over 1024 patch tokens of
width 768, then a post-LayerNorm and a connector projection:

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

- im2col patch embedding: a one-time reshape before the layer loop, not
  hot-loop work. Moving it on-device would take a conv or im2col kernel for no
  throughput gain.
- pixel-shuffle: a space-to-depth reshape with no arithmetic, bit-exact against
  HuggingFace. The connector's only math, the 64×12288×960 projection, runs on
  the NPU.

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

Affine LayerNorm and GELU-tanh were added to the registry for this example and
have their own pages there.

SigLIP attention is bidirectional with no mask and an even 12 heads, which is
what the registry FlashAttention kernel computes: it packs two heads per compute
unit and fills the 8×4 array. The backbone and the action expert need masks;
section 7 covers how the all-NPU path handles them.

## 3. Fusion: 38 dispatches per image instead of 121

A layer is 18 kernel launches. Issuing 18 separate programs would make the host
pay its per-dispatch cost 18 times and round-trip every intermediate through the
host in fp32. `smolvla_vision_builders.py` instead stitches them into three
multi-launch ELFs with `shared/infra/stitching.stitch_elf`:

| Program | Launches | Contents |
|---|---|---|
| `vit_ln_qkv` | 7 | affine LayerNorm + Q/K/V GEMM + three on-device bias-adds |
| `flash_attn` | 1 | the registry FlashAttention ELF, unchanged |
| `vit_o_ffn` | 10 | O GEMM + bias + residual + LayerNorm + fc1 + bias + GELU + fc2 + bias + residual |

12 layers × 3, plus the post-LayerNorm and the connector, is 38 dispatches per
image.

Most of the gain is not driver overhead. Moving the per-Linear bias adds and the
two residual adds on-device removed a bf16 → f32 → bf16 host round trip per
operation, which had been most of the host time. Accuracy improved too, because
that round trip re-quantized every intermediate.

## 4. A fused-ELF trap worth knowing

`compile_gemm_mm` bakes `DIM_M`, `DIM_N` and `DIM_K` into the external `mm.o`
microkernel at compile time, but the shared helper `disambiguate_by_tile_n`
names that object from `tile_n` alone. At seq=1024 the vision GEMMs resolve to
two distinct `tile_n` (96 for q/k/v/o and fc2, 128 for fc1) under the same
"drain" method, so a GEMM built in isolation would link a stale generic
`mm_m32.o` with the wrong baked `DIM_N` and silently produce garbage.
`_force_tile_n_suffix` in
`smolvla_vision_builders.py` forces the tile_n-keyed name so every ELF links the
object it was compiled against.

## 5. Splicing into LeRobot

`smolvla_inference.py` wraps `policy.model.embed_prefix` and, only for the
duration of that call, swaps `vlm_with_expert.embed_image` for one that serves
results the NPU already computed. All camera images are encoded in one call into
the runtime, then handed out one at a time as LeRobot's own code asks for them.

This keeps the sqrt(960) scale, the pad and attention masks and prefix assembly
as LeRobot's own code, and the backbone and action expert too unless
`--npu-all` is set. That is what makes the comparison against the pure-CPU
baseline meaningful. The wrapper is always restored in a `finally`.

## 6. One process

`smolvla_runtime.py` holds a process-wide `VisionRuntime`: weights, compiled
ELFs, the XRT context and the device buffer objects are created once and reused
by every inference. Spawning a process per inference would pay process start,
weight reload and ELF load every time, none of it NPU work.

## 7. Backbone and action expert (experimental)

`--npu-all` runs the two remaining stages on the NPU too. Both are small for the
array (seq 241, and 50 action tokens padded to 64) and need masked attention, so
one launch per operation would spend most of the time on launch overhead. Both
paths therefore put a whole layer, or many layers, in one launch.

Backbone (`experimental/backbone_npu.py`, `experimental/backbone_runtime.py`):
one fused ELF per layer covering RMSNorm, the Q/K/V projections and RoPE, a
masked FlashAttention variant, the O projection and the FFN. Weights are bfp16.
The ELF is compiled on the first `--npu-all` run and cached.

Action expert (`experimental/gemm_engine.py`, `experimental/expert_runtime_v2.py`):
a GEMM engine is one launch, one segment and one herd that runs a list of GEMM
jobs. RMSNorm, RoPE, residual adds, SwiGLU, `exp` and the softmax divide run in
the matmul drain, so attention becomes GEMM jobs: scores with an `exp` drain,
P·V with a divide drain. Activations live in an arena in host memory, and a job
reads the outputs earlier jobs drained there. The step engine runs all 16
layers in one launch per denoise step.

The K and V the expert attends over change once per action chunk, not per step.
A separate prefix engine turns the backbone's K/V rows into the expert's K and V
tiles once per chunk: for cross-attention layers it applies the expert's k/v
projections, for self-attention layers it rotates K by −p0 and copies V. The
step engine reads those tiles from the buffer the two engines share, so the host
does no K/V packing.

Because jobs read back what earlier jobs drained, the compiler must keep each
of those reads behind the transfers that write the region it reads. Current
mlir-air derives that from the async dependencies; an older compiler can
produce an engine that reads stale data or hangs.

`make compile-expert` builds the two engine ELFs
(`experimental/expert_v2_probe.py --layers 16 --compile-only`). The layer
pattern is fixed at build time: self-attention layers are the even ones, and an
engine built with the other pattern produces garbage.
