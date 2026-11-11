# Qwen3-Coder-30B-A3B Q4_K prefill + decode on AMD NPU2

Text generation (prefill and greedy autoregressive decode, batch 1) of
[`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
(revision `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`) with the MLIR-AIR
dialect. FastFlowLM has no port of this model; the design is the
[Qwen3.6-35B-A3B](../qwen36_35b_a3b_q4nx) example's (FastFlowLM's Qwen3.6 MoE
decoding layer and LM head, and the shared
[fused_prefill](../shared/fused_prefill) device) at this model's shapes, every
layer an attention layer.

- **prefill**: the prompt in 128-row chunks on the shared fused prefill
  device. Every projection and expert GEMM and the attention are ops of one
  xclbin; the router, GLU, norms and RoPE run on the host.
- **decode**: **one NPU dispatch per token for the 48 layers**, the router
  choosing and fetching each layer's experts on device, then a second dispatch
  for the Q8 LM head. The KV cache stays on the device across tokens.

## Architecture

| | |
|---|---|
| layers | 48, each GQA attention then MoE |
| attention | 32 query heads, 4 KV heads, head dim 128, q/k RMSNorm, RoPE over the whole head (theta 1e7) |
| MoE | 128 routed experts, top 8 (softmax over all 128, the top 8 renormalized), intermediate 768; no shared expert |
| hidden / vocab | 2048 / 151936, untied LM head |
| norms | RMSNorm, eps 1e-6 |

## Weights

There is no pre-quantized bundle: `make prepare` (`python3
qwen3_coder_30b_a3b_q4k_weights.py --prepare`) quantizes the bf16 checkpoint
once into a bundle of about 18 GB at `QWEN3C_BUNDLE` (default
`$HF_HOME/qwen3_coder_30b_a3b.q4k`), in the block formats of the Qwen3.6
example's bundle:

- **Q4_K** for every projection and expert: blocks of 32 rows x 256 columns
  (4736 bytes), `w = d * sc * q + dmin * m` with bf16 `d`, `dmin` per row and
  256 columns and 8-bit `sc`, `m` per 32 columns; each 32-column group's scale
  and min are fit by llama.cpp's weighted search (`make_qkx2_quants`).
- **Q8** for the LM head (8704 bytes per block).
- bf16 for the embedding, norms, q/k norms and router.

How each side takes them:

- **Decode**: the Q4_K blocks as stored. The experts of all 48 layers sit in
  one buffer (`WE`, about 16.8 GB) that the router core reads at the chosen
  experts' offsets.
- **Prefill**: the same blocks, read in place by the fused prefill device's
  Q4_K GEMM (`Config.q4k`): the experts from the decode's `WE`, the dense
  projections from the prefill's own copy.

## Placement

Decode (`decode.py`), one launch repeated once per four layers:

| Columns | Role |
| --- | --- |
| 0, 1, 6, 7 | Q4_K GEMV array: the dense projections and the experts, two shim weight channels per column |
| 2 | RMS norm and residual (2, 2); q/k norm and RoPE (2, 3); the array's assemble memtile |
| 3, 4 | four attention units, one KV head and 8 query heads each; memtiles 3 and 4 stream KV heads 0-1 and 2-3, memtile 3 gathers the outputs |
| 5 | router GEMV, router (top 8 of 128) and the expert transfers, GLU and the experts' weighted sum |

The router core starts each layer's dense weights and its experts' transfers
itself. The LM head (`lm_head.py`) is Qwen3.6's Q8 program on 8 x 4 cores.

Prefill (`prefill.py`, device spec `fused_spec.py`):

| Computation | Where | Implementation |
| --- | --- | --- |
| q \| k \| v and o, experts' up \| gate and down | NPU | fused_prefill GEMM ops on Q4_K blocks (`wq=3`) |
| causal attention, head dim 128 | NPU | `attn_bfp16`: query heads 0-15 with KV heads 0-1 on column 0, 16-31 with 2-3 on column 7 |
| norms, residual adds, q/k norm, RoPE, K/V records | host | `hostops` (shared) |
| router, expert GLU, the experts' weighted sum | host | torch, the Qwen3.6 example's `exp_glu_tile` |

Each expert is an up | gate dispatch, the GLU on the host and a down dispatch;
the busiest experts go first, and the next expert's up | gate is queued while
the host works on the current one. After prefill the host writes the prompt's
K / V rows into the decode's cache, and the LM head takes the last row for the
first token.

## Correctness

- **CPU reference** (`reference.py`): float32 torch on the bundle's
  dequantized weights. `check_vs_hf.py` checks it layer by layer against HF
  transformers' `Qwen3MoeDecoderLayer` fed the same weights.
- **`make verify`** (`run_npu2_verify.lit`): a layer and the LM head each
  against the reference; the fused prefill's last-row logits against it (300
  tokens); the prompt prefilled into the decode and its greedy tokens compared
  with the reference's; the reference against HF; then the shared
  [verify](../verify) top-k token-set gate of the NPU against the bf16
  checkpoint.

## Results

NPU2, NPU otherwise idle. llama.cpp b11302 on the same machine's Radeon 890M
(Vulkan), with the `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` Q4_K_M weights,
measured with `llama-bench`.

| | this example | llama.cpp (Vulkan) |
| --- | --- | --- |
| decode, 1024-token context | 45.5 ms/token (22.0 tok/s) | 36.3 ms/token (27.5 tok/s) |
| decode, 4096-token context | 56.2 ms/token (17.8 tok/s) | 35.7 ms/token (28.0 tok/s) |
| prefill, 1024-token prompt | 2.45 s | 1.92 s |
| prefill, 2048-token prompt | 4.66 s | 4.11 s |
| prefill, 4096-token prompt | 11.7 s | 10.2 s |

Decode streams about 1.9 GB of weights a token; the NPU's DDR reads top out
near 58 GB/s, so its floor is above llama.cpp's on the integrated GPU.

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, a Peano (llvm-aie) install
- `source utils/env_setup.sh` (see the repo README), `HF_HOME` for the Hub
  cache
- a locked-memory limit above the expert buffer's size (`ulimit -l`)
- `pip install -r requirements.txt`

## Build and run

```bash
make prepare                  # the Q4_K bundle, once
make compile                  # the decode ELF (MAXL = 64) and the LM head
make compile-fused-prefill    # the prefill device
make run PROMPT="The capital of France"   # prefill + greedy decode, vs the CPU reference
make verify                   # the gate
make profile CTX=1024         # decode time per token at a 1024-token context
make profile-decode           # decode tok/s at each of CONTEXTS
make profile-prefill CTX=2048 # TTFT of a 2048-token prompt
```

## Tests

| lit | What it checks |
| --- | --- |
| `run_npu2_compile.lit` | the decode and LM head ELFs compile; no NPU, no weights |
| `run_npu2_verify.lit` | `make verify` |
| `run_npu2_profile.lit` | decode time per token at a 1024-token context |
| `run_npu2_sweep.lit` | decode tok/s at contexts 1024 and 4096 (`make profile-decode` through `bench/sweep_decode_runtime.py`) |
| `run_npu2_prefill_sweep.lit` | TTFT at 128, 512 and 2048 tokens (`bench/sweep_prefill.py`) |

## Files

| File | Purpose |
| --- | --- |
| `decode.py` | The 48-layer decode ELF, the LM head dispatch, the prefill hand-off (`Q3CDecoder`); `make run` / `profile` |
| `attn_layer.py`, `lm_head.py` | One layer (attention and MoE) and the LM head alone, each against the reference |
| `moe_block.py` | The MoE block's shapes and weight layout |
| `prefill.py` | The prefill on the fused prefill device (`Q3CPrefill`) |
| `fused_spec.py`, `fused_build.py` | The prefill device: GEMM shapes, attention groups, build |
| `kernel_tests/common.py` | The Qwen3.6 example's kernels compiled at this model's sizes |
| `qwen3_coder_30b_a3b_q4k_weights.py` | The quantizer and the bundle reader |
| `reference.py`, `check_vs_hf.py` | CPU reference, and its check against HF transformers |
| `verify_adapter.py` | The decode as a runner of the shared token-set gate |

## Known limitations

- The weights are 4-bit: greedy text follows the bf16 model's closely but can
  diverge from it where two tokens are near-tied (the token-set gate allows
  for this).
- The decode's KV cache length (`MAXL`) is compiled in, and every layer reads
  it back whole each token, so a larger build is slower at every context.
- Prompts up to 4096 tokens (the prefill device's KV records).
- The kernels and the host GLU are the Qwen3.6 example's, so that example's
  directory is needed alongside this one.
