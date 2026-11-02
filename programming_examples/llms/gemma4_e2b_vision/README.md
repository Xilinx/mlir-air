# Gemma4-E2B image + text on NPU2

Image turns for Gemma4-E2B-IT: the vision tower runs on the NPU, and its soft
tokens go into the text model of [`gemma4_e2b_q4nx`](../gemma4_e2b_q4nx). That
model's fused prefill runs the prompt, and its fused decode generates the
answer.

```bash
export PEANO_INSTALL_DIR=/path/to/llvm-aie   # >= 22.0.0, as ../gemma4_e2b_q4nx needs
pip install -r requirements.txt
make compile          # the text model's fused prefill + decode
make vision           # NPU tower vs HF, cosine >= 0.99
make run IMAGE=photo.jpg PROMPT="What is in this picture?"
make verify           # image-turn top-k gate vs HF bf16 + the shapes keyword gate
make profile          # TTFT and decode tok/s at a 70-soft-token image
make help             # everything else
```

Weights: the vision tower and the multimodal projection come from the bf16
`unsloth/gemma-4-E2B-it` checkpoint (`G4_VISION_MODEL_SOURCE` overrides it).
The text model is the `FastFlowLM/Gemma4-E2B-IT-NPU2` q4nx bundle
(`Q4NX_MODEL_SOURCE`).

## Pipeline

| Stage | Where | Code |
|---|---|---|
| Resize, patchify | host | `Gemma4ImageProcessorPil` (`gemma4_vision_weights.preprocess`) |
| Patch embedding GEMM, + 2-D position table | NPU, host | `gemma4_vision_encoder` |
| 16 layers: q\|k\|v, o, gate (GELU), up, down GEMMs; attention | NPU | `NpuOps` |
| Norms, 2-D rope, ClippableLinear clamps, gated product | host | `hostops.c` |
| 3x3 pooling, embedder norm; projection to 1536 GEMM | host, NPU | `VisionEncoder.encode` |
| Prompt with soft tokens in the image slots | NPU | `../gemma4_e2b_q4nx/fused_prefill` |
| Decode | NPU | `../gemma4_e2b_q4nx` full-ELF decode |

The image's soft tokens replace the embeddings at the image placeholder's
positions, which carry the pad id. As in `Gemma4Model.forward`, the per-layer
token table is looked up with those pad ids, and the per-layer projection reads
the embeddings after the soft tokens are in place.

## Two things specific to this tower

**ClippableLinear.** Every projection clamps its input and its output to
calibrated per-tensor bounds, and on real images the clamps bind. The GEMMs
stay plain. The host op that writes a GEMM's input applies the input
clamp, and the host op that reads its output applies the output clamp. The gate
projection's GELU runs in the GEMM's drain, so its output clamp is applied past
the activation as `min(GELU(y), GELU(hi))`. This matches `GELU(clip(y))` to
within 2e-5 because every layer's lower bound is below -4.4, where GELU is 0 to
that precision.

**Padding patches as keys.** The processor pads an image's patches to a fixed
count per budget, and HF masks the padding out of attention. The real patch
count depends on the aspect ratio and is rarely a multiple of the kernel's
64-key block, and the FlashAttention kernel has no key mask. So each head is
widened from 64 to 128 columns: column 64 is 1 in Q, 0 in a real key and -1e4
in a padding key. Every padding score becomes -1e4 and drops out of the softmax.
Zero-padded keys would not: they score 0, which is not small next to some
queries' real scores.

## Gates

- `make vision`: soft tokens vs HF's `Gemma4VisionModel` + `Gemma4MultimodalEmbedder`
  in f32, cosine >= 0.99. `make diagnosis` adds per-layer cosine against the
  NumPy tower, which `make reference` checks against HF.
- `make verify`: `verify_image.py` generates 32 greedy tokens per image turn on
  both the NPU and HF bf16, and checks them with the shared
  `verify.comparators.compute_topk_set_check`. The HF side builds its prompt
  with HF's own processor, and that prompt is asserted equal to this example's.
  A keyword check then requires the shapes image's description to name both
  shapes, their colors and a position.
