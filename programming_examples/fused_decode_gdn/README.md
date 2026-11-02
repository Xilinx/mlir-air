# Fused superkernel decode with a Gated DeltaNet mixer (Qwen3.5-0.8B) on NPU2

This implementation reimplements the corresponding AMD NPU LLM design,
originally developed by the [FastFlowLM](https://github.com/ROCm/FastFlowLM)
team, using the higher-level abstractions of the MLIR-AIR dialect: FastFlowLM's
Qwen3.5 decoding layer (`qwen3_5_decoding`), on top of `fused_decode`'s
reimplementation of its generic decoding layer.

A fork of [`fused_decode`](../fused_decode) for Qwen3.5-0.8B:
- **one dispatch = 24 layers + LM head**: 18 Gated DeltaNet layers and 6 gated full-attention layers
  (layer % 4 == 3);
- the KV cache and the Gated DeltaNet recurrent state stay on the device, in DDR, across tokens.

Consumed by [`llms/qwen35_0_8b_q4nx`](../llms/qwen35_0_8b_q4nx) (`q35_fdg.py`), whose prefill hands over
the prompt's KV rows, conv history and DeltaNet states.

## Why a fork

The Gated DeltaNet's recurrent state is 16 heads x 128 x 128 FP32 per layer: 1 MiB, read and written
every token. It needs its own cores and DMA channels.
- **LFM2's short conv** fitted into `fused_decode` in place (the hybrid arm).
- **Gemma4's per-layer embeddings** did not, and became [`fused_decode_ple`](../fused_decode_ple).
- **FastFlowLM** likewise forked its generic decoding layer for Qwen3.5 (`qwen3_5_decoding`, with new
  Gated_Delta / Conv / DeltaNetNorm / AB / gate modules).

The fork keeps `fused_decode`'s one-dispatch property; FLM's own Qwen3.5 decode is a per-layer runlist.

## Design

| Tiles | Role | Origin |
|---|---|---|
| cols 0,1,6,7 x rows 2-5 | projection engine (every projection and the LM head), dual-MM2S weight feed | `fused_decode`, unchanged dataflow; kernel extended (below) |
| (2,2) | rms / residual / final norm / logit relay | `fused_decode` |
| (5,3) | SiLU(gate) * up | `fused_decode`; SiLU on bf16 MACs |
| (4,2)-(4,5) | 2 attention CUs (1 KV head and 4 query heads each, head dim 256); the kv core applies the sigmoid output gate, or takes the GDN output | `fused_decode`, `conv_o_pass` extended |
| (5,4) stage, (5,5) conv | the three mixer phases land on the stage tile, which runs q/k-norm + partial RoPE (64 of 256) on attention waves; the conv tile takes the 16 GDN heads' rows, computes their gates (beta, decay) in one 16-lane pass, and the attention gate's sigmoid | the LFM2 hybrid arm, new kernels (`gdn_mix.cc`) |
| (3,2)-(3,5) | Gated DeltaNet herd: 4 heads per core; causal conv over the history, q/k l2-norm, the delta rule on the FP32 state (bf16 hi + lo), gated RMSNorm | new (`gdn_step.cc`), FLM's column mapping |
| memtile 3_1 | scatters the conv tile's rows to the 4 cores; gathers their outputs | new |
| (5,2) | mixer-output tile: GDN output or attention gate, by the arm, to the kv cores (`@mixToCU`) | new |

### Per decode wave (one per layer)

Uniform for both layer kinds, as in LFM2's hybrid:
- **ph0-ph2:** three mixer phases of 3072 rows.
  - Gated DeltaNet: 16 heads x `[q k v (384) | z (128) | beta / alpha hi / lo rows (4) | pad]`.
  - Attention: `[q | k | v]`, then `[gate | pad]`, then pad.
- **ph3:** o / out_proj, K 2048.
- **ph4:** gate|up, 7168 rows.
- **ph5:** down, K 3584.

18 LM-head waves follow.

On a Gated DeltaNet wave:
- The herd streams per core 38 chunks of 4096 words from DDR and back:
  `[reset | (X, 8 state units) x 4 heads | NOP]`. X carries the conv history, taps and norm
  weight; a state unit is `[hi | lo][64 k pairs][16 v][2]`.
- One buffer is this token's input and the other its output; the host swaps them every token.
- The heads' rows and outputs move on-chip.
- The attention CUs run no KV blocks, and the shim reads no KV back.

On an attention wave the herd's chunks pass through untouched.

### Weights

- **Q4K projections:** the bundle's 4736-byte blocks run exactly in the engine's 5120-byte slot.
  - The slot holds a tagged header (uint8 sc, m; bf16 d, dmin), then the nibbles in the
    column-pair layout.
  - The kernel computes `d * sum(sc * S) + dmin * sum(m * X)` with the products and group sums split
    bf16 hi + lo.
- **Q8 tensors** (the LM head, beta / alpha) are carried as two affine-Q4 rows per row,
  `s q = 16 s hi - 128 s + s lo`, which is exact. Their logits are row pairs, which the host adds
  when it reads the logits.

## Build

```bash
# llvm-link from LLVM < 23 on PATH (see ../fused_decode/README.md), Peano >= 22
make compile-decode-elf BUILD=<dir> LBUILD=<capacity>   # kernels + one full ELF
make -C ../llms/qwen35_0_8b_q4nx compile-decode-fdg BUILD=<dir>   # every capacity
```

The context length reaches the device as scratchpad parameters (`../fused_decode/decode_elf.py`).
These are the KV append offset, the read-back block count and the CUs' block bound, so one ELF
serves every context up to its capacity.

The compile gate is `llms/qwen35_0_8b_q4nx/run_npu2_compile.lit`.

## Files

| File | Purpose |
|---|---|
| `fused_decode_gdn.py` | The builder: `fused_decode.py` with the `qwen3.5-0.8b` entry, the GDN herd, the mixer tiles, the hand-offs |
| `models/qwen3.5-0.8b.h` | Model constants for the kernels |
| `kernels/gdn_step.cc` | Gated DeltaNet herd (chunk interpreter) |
| `kernels/gdn_mix.cc` | Stage, conv and mixer-output tiles |
| `kernels/q4_k.h`, `kernels/proj_qmm.cc` | Projection blocks: Q4K and affine, column-pair layout |
| `kernels/attn_kv.cc`, `kernels/glu.cc`, `kernels/model_spec.h` | Changed copies of `fused_decode`'s |
| remaining `kernels/*` | Unchanged copies of `fused_decode`'s |

Why copies rather than including the parent's kernels:
- A quoted `#include` resolves next to the including file, so a fork header cannot shadow the
  parent's.
- The fork therefore carries its own kernel tree.
- It takes `proj_qmm_pack.py`, `decode_kernels.mk` and `decode_elf.py` from `../fused_decode`.
