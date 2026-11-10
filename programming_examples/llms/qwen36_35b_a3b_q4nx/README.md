# Qwen3.6-35B-A3B Q4NX prefill + decode on AMD NPU2

This implementation reimplements the corresponding AMD NPU LLM design,
originally developed by the [FastFlowLM](https://github.com/ROCm/FastFlowLM)
team, using the higher-level abstractions of the MLIR-AIR dialect: FastFlowLM's
Qwen3.6 MoE decoding layer and LM head, and its prefill, on the shared
[fused_prefill](../shared/fused_prefill) device.

Text generation (prefill and greedy autoregressive decode, batch 1) of
Qwen3.6-35B-A3B with the weights FastFlowLM itself ships:
[`FastFlowLM/Qwen3.6-35B-A3B-NPU2`](https://huggingface.co/FastFlowLM/Qwen3.6-35B-A3B-NPU2),
branch `flm_q4k_high_precision` at revision
`18adb6adc1338ba29711e58b787df609fc55cd3a`, file `model.q4nx` (21,044,871,704
bytes, sha256 `a763f82ad34f09af85615c600265fa26be5723176e483ef22cd1ec735f12d175`).
The logical quantized values are used as supplied; only their device layout
changes. The vision tower is out of scope.

- **prefill**: the prompt in 128-row chunks on the shared fused prefill
  device. Every projection and expert GEMM and the full-attention layers'
  attention are ops of one xclbin; the router, GLU, norms, RoPE and the Gated
  DeltaNet recurrence run on the host, as in FastFlowLM.
- **decode**: **one NPU dispatch per token for the 40 layers**, the router
  choosing and fetching each layer's experts on device, then a second dispatch
  for the Q8 LM head. The KV cache, the conv caches and the Gated DeltaNet
  states stay on the device across tokens.

## Architecture

| | |
|---|---|
| layers | 40: 30 Gated DeltaNet and 10 gated full attention, in groups of 3 + 1 (layer % 4 == 3 is attention) |
| Gated DeltaNet | 16 query / key heads and 32 value heads (each key head serves 2), head dim 128, causal conv1d (kernel 4), FP32 recurrent state 32 x 128 x 128 per layer, gated RMSNorm |
| full attention | 16 query heads, 2 KV heads, head dim 256, q/k RMSNorm, partial RoPE over 64 of 256 dims (theta 1e7), sigmoid output gate |
| MoE | 256 routed experts, top 8 (softmax over all 256, the top 8 renormalized), intermediate 512; one shared expert (intermediate 512) scaled by sigmoid(w · x) |
| hidden / vocab | 2048 / 248320, untied Q8 LM head |
| norms | zero-centred (stored as `1 + w`), eps 1e-6 |

## Weights

`qwen36_35b_a3b_q4nx_weights.py` reads the bundle from `QWEN36_FLM_DIR` or
`Q4NX_MODEL_SOURCE` (a `model.q4nx` or a directory holding it), else from the
Hub at the pinned revision, checking its sha256. Every projection and expert is
**Q4_K** in blocks of 32 rows x 256 columns (4736 bytes,
`w = d * sc * q + dmin * m`); the LM head is **Q8** (8704 bytes per block);
the norms, router, alpha / beta and the shared-expert gate are bf16.

How each side takes them:

- **Decode**: the Q4_K blocks as stored, in FastFlowLM's layout per layer. The
  experts of all 40 layers sit in one buffer (`WE`, about 18.6 GB) that the
  router core reads at the chosen experts' offsets.
- **Prefill**: the same blocks, read in place by the fused prefill device's
  Q4_K GEMM (`Config.q4k`): the routed experts from the decode's `WE`, the
  dense projections from the prefill's own copy.

## Placement

Decode (`decode.py`), FastFlowLM's tile map, one launch repeated once per four
layers:

| Columns | Role |
| --- | --- |
| 0, 1, 6, 7 | Q4_K GEMV array: the dense projections and the experts, two shim weight channels per column |
| 2 | RMS norm and residual, shared-expert gate, RoPE with q/k norm, output gate (attention) or SiLU(z) and the DeltaNet norm (GDN) |
| 3 | attention: 2 units, one KV head and 8 query heads each, the KV cache streamed per token |
| 4 | Gated DeltaNet recurrence, 4 cores of 32 value columns, the state streamed per token |
| 5 | conv1d + SiLU + l2 norm, alpha / beta, router (top 8 of 256), GLU and the experts' weighted sum |

The router core starts each layer's dense weights and its experts' transfers
itself. The LM head (`lm_head.py`) is FastFlowLM's Q8 program on 8 x 4 cores.

Prefill (`prefill.py`, device spec `fused_spec.py`):

| Computation | Where | Implementation |
| --- | --- | --- |
| projection GEMMs (DeltaNet q \| k \| v \| z and out; attention q \| gate \| k \| v and o), routed and shared experts' up \| gate and down | NPU | fused_prefill GEMM ops on Q4_K blocks (`wq=3`), dequantized by the cores to BFP16 |
| softmax attention of the 10 attention layers (causal, head dim 256) | NPU | `attn_bfp16`: query heads 0-7 with KV head 0 on column 0, 8-15 with KV head 1 on column 7 |
| norms, residual adds, q/k norm, partial RoPE, K/V records, output gate | host | `hostops` (shared), `kernels/q36_host.c` |
| alpha / beta, router, expert GLU, the experts' weighted sum | host | torch, `kernels/q36_host.c` |
| Gated DeltaNet conv1d, gates, delta rule on an FP32 state, gated norm | host | `kernels/q36_host.c` (Qwen3.5's `gdn_chunk` at 16 / 32 heads) |

Each expert, the shared one included, is an up | gate dispatch, the GLU on
the host and a down dispatch; the busiest experts go first, and the next
expert's up | gate is queued while the host works on the current one. After
prefill the host writes the prompt's K / V rows, conv caches and DeltaNet
states into the decode's buffers, and the LM head takes the last row for the
first token.

## Correctness

- **CPU reference** (`reference.py`): float32 torch on the bundle's
  dequantized weights. `check_vs_hf.py` checks it layer by layer against HF
  transformers' `Qwen3_5Moe` decoder layers fed the same weights (cosine
  1.0000000 on a GDN and an attention layer).
- **`make verify`** (`run_npu2_verify.lit`): the MoE block, a GDN layer, an
  attention layer and the LM head each against the reference; the fused
  prefill's last-row logits against it (300 tokens); the prompt prefilled into
  the decode and every greedy token compared with the reference's; the
  reference against HF.

## Results

NPU2, NPU otherwise idle. FastFlowLM measured on the same machine with its own
binary, timed with an XRT run hook.

| | this example | FastFlowLM |
| --- | --- | --- |
| decode, short context | 50.9 ms/token | 60.5 ms/token |
| decode, 1024-1300 tokens | 54.6-55.0 ms/token | 60.6 ms/token |
| prefill, ~1.16k-token prompt | 3.87 s | 9.67 s |
| prefill, ~2.04k-token prompt | 5.37 s | 11.3 s |
| prefill, ~4.07k-token prompt | 9.5 s | 16.8 s |

The prefill figures use the same prompt text for both (FastFlowLM's token
count includes its chat template), warm, with the experts resident.

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README), `HF_HOME` for the Hub
  cache, or `QWEN36_FLM_DIR` for a local copy of the bundle
- a locked-memory limit above the expert buffer's size (`ulimit -l`)
- `pip install -r requirements.txt`

## Build and run

```bash
make compile                  # the decode ELF (MAXL = 64) and the LM head
make compile-fused-prefill    # the prefill device
make run PROMPT="The capital of France"   # prefill + greedy decode, vs the CPU reference
make verify                   # the gate
make profile CTX=1024         # decode time per token at a 1024-token context
make profile-decode           # decode tok/s at each of CONTEXTS
make profile-prefill CTX=2048 # TTFT of a 2048-token prompt
make kernels                  # every kernel test
```

## Tests

| lit | What it checks |
| --- | --- |
| `run_npu2_compile.lit` | the decode and LM head ELFs compile; no NPU, no weights |
| `run_npu2_verify.lit` | `make verify` |
| `run_npu2_profile.lit` | decode time per token at a 1024-token context |
| `run_npu2_sweep.lit` | decode tok/s at contexts 1024, 4096 and 8192 (`make profile-decode` through `bench/sweep_decode_runtime.py`) |
| `run_npu2_prefill_sweep.lit` | TTFT at 128, 512 and 2048 tokens (`bench/sweep_prefill.py`) |

## Files

| File | Purpose |
| --- | --- |
| `decode.py` | The 40-layer decode ELF, the LM head dispatch, the prefill hand-off (`Q36Decoder`); `make run` / `profile` |
| `gdn_layer.py`, `attn_layer.py`, `moe_block.py`, `lm_head.py` | One GDN layer, one attention layer, the MoE block and the LM head alone, each against the reference |
| `prefill.py` | The prefill on the fused prefill device (`Qwen36Prefill`) |
| `fused_spec.py`, `fused_build.py` | The prefill device: GEMM shapes, attention groups, build |
| `kernels/` | The decode's core kernels and the prefill's host C (`q36_host.c`) |
| `kernel_tests/` | Each decode kernel alone against a CPU computation |
| `qwen36_35b_a3b_q4nx_weights.py` | The bundle reader |
| `reference.py`, `check_vs_hf.py` | CPU reference, and its check against HF transformers |

## Known limitations

- `make verify` does not use the shared `llms/verify` token-set gate: that
  gate needs the bf16 HF checkpoint (about 70 GB). The reference is checked
  against transformers' layers on the bundle's weights instead.
- The decode's KV cache length (`MAXL`) is compiled in, and every attention
  layer reads it back whole each token, so a larger build is slower at every
  context.
- Prompts up to 4096 tokens (the prefill device's KV records).
- The Gated DeltaNet recurrence of the prefill runs on the host.
