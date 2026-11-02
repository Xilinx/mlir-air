# d3-edge decision model on AMD NPU2

This implementation reimplements the corresponding AMD NPU LLM design,
originally developed by the [FastFlowLM](https://github.com/ROCm/FastFlowLM)
team, using the higher-level abstractions of the MLIR-AIR dialect. It runs on
the shared [fused_prefill](../shared/fused_prefill) device as the
[Qwen3.5-0.8B prefill](../qwen35_0_8b_q4nx) does, by subclassing its
`Qwen35Prefill`.

*System One* decisions of the vLLM Semantic Router decision model
[`vllm-sr/d3-edge`](https://huggingface.co/vllm-sr/d3-edge) at revision
`c454ee39b862248d70abd7f84195d2f54ffe6d43`, with its BF16 checkpoint: one
forward pass per question over the package's own prompt format, the option
logits read by a 255-way FP32 readout at the last prompt position, the option
softmax on the host. Batch 1, text only, no decode. Its sibling
[`d3_lite_q4nx`](../d3_lite_q4nx) runs vllm-sr/d3-lite on FastFlowLM's Q4K
weights with the same engine; the runtime, gate, reference, timing and
request set in this directory serve both.

## Architecture

| | |
| --- | --- |
| Layers | 12: `L L A L L A L L A A A A` (`L` = Gated DeltaNet, `A` = gated full attention) |
| d_model / head_dim / MLP | 1024 / 256 / 3584 |
| Attention | 8 query / 2 KV heads, q/k RMSNorm, partial RoPE 64/256, output gate; **noncausal** |
| Gated DeltaNet | 16 heads, conv1d 4 taps, delta rule on an FP32 state; causal |
| Output | final RMSNorm of the last row, FP32 readout [255, 1024], softmax over the question's options |

`decision_config.json` sets `attention_mode: noncausal_full_attention`: every
query of a full-attention layer reads every prompt key, while the Gated
DeltaNet layers stay causal. The tokenizer, `decision_config.json`,
`readout.safetensors` and `d3_runtime.py` are the upstream files.

## Placement

One fused_prefill xclbin, one hardware context per forward pass:

| Op | Where |
|---|---|
| every projection GEMM (attention q\|gate\|k\|v and o; DeltaNet in_proj and out_proj; MLP gate\|up, down) | **NPU** (fused_prefill GEMM ops, BFP16 compute, 128-row chunks, chunk pairs) |
| full-attention softmax attention, noncausal with the key-length mask | **NPU** (`attn_bfp16`, `Attn.noncausal` mode) |
| tokenization, d3 prompt rendering, embedding, orchestration | host |
| RMSNorms, q/k norm, partial RoPE, attention output gate, residual adds, SwiGLU | host (written straight into the next GEMM's bfp16 A, `kernels/d3_host.c`) |
| DeltaNet conv1d + SiLU, gates, gated norm, the delta-rule scan (FP32 state) | host (`qwen35_0_8b_q4nx/kernels/q35_host.c`), on a second host thread from 2048 tokens |
| final norm, FP32 readout of the last row, option softmax | host |

**Noncausal attention.** Every chunk's queries read the keys of the whole
prompt, so an attention layer first writes every chunk's K / V records, then
queues each chunk's attention with its queries (key count = prompt length).
The last layer (full attention) feeds only the readout: its K / V cover every
row, its queries, output projection and MLP only the last chunk / row.

**Weights.** The device's BF16-weight GEMM reads each weight as BFP16 (8
values per shared exponent). `WFORM` chooses how a BF16 weight W is held:

- `hilo:gu,down` (default): the MLP's gate|up and down as `[bfp16(W) | W −
  bfp16(W)]` at depth 2k against `[x | x]`, the cores' FP32 accumulator summing
  both; the attention and DeltaNet projections as one BFP16 copy at depth k.
- `hilo`: every projection as `[bfp16(W) | W − bfp16(W)]`. It is kept as an
  option because it passes the gate's deciding question (`b1024_r3/escalate`,
  reference 0.547 / 0.453) with the same margin as the default (0.5246 against
  0.5252), at 10-13% more time.

One BFP16 copy everywhere fails that question (0.492 / 0.508) and is not
offered.

## Correctness

d3 has no LM head and no causal-LM reference, so the repository's
`llms/verify` top-5 next-token check does not apply. **`make verify` runs the
d3 decision gate** (`gate.py`, `run_npu2_verify.lit`) instead, the model's
correctness gate: the production path on the frozen request set
`requests.json` (17 requests, 67 questions of the `choice`, `noul` and `score`
types at 512 to 4096 tokens; content sha256 `21c8e0ba…7bf7`) against the upstream d3 runtime on CPU in BF16
(`make reference`, `reference.py`). A question passes when its option logits
are all finite, each answer is in the other's top 2, and the answers are equal
when the reference's top-2 margin is at least 0.05; every question must pass
and every input id sequence must equal the reference's.

| Weight form | Gate | Answers agree | max \|Δp\| |
|---|---|---|---|
| `hilo:gu,down` | PASS 67/67 | 66/67 | 0.106 |
| `hilo` | PASS 67/67 | 66/67 | 0.136 |

The one disagreement is a question whose reference margin is 0.015.

## Results

NPU2 (Ryzen AI MAX+ 395), MLIR-AIR built from this tree's base, Peano
(llvm-aie) 22.0.0.2026102001, mlir-aie 1.4.5.dev83, XRT 2.21.0, 2026-11-02.
`make bench`: one `route` question grown to exactly P tokens; the time from
the token ids on the host to the option probabilities on the host; at least
5 s of warm-up, then at least 3 runs and 10 s measured; medians. All rows were
measured back to back in one session; the earlier port, on its own build and
with the same prompts, is shown for comparison.

| ms | P = 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| `hilo:gu,down` | 52.9 | 96.0 | 207.7 | 513.4 |
| `hilo` | 59.8 | 106.3 | 227.9 | 570.8 |
| earlier d3-edge port (42 ELFs, every row op on the NPU) | 329.3 | 520.0 | 933.6 | 1855.3 |

At 4096 tokens the NPU spends about 300 ms in the GEMMs and 208 ms in the
attention per forward pass, each dispatch waited for in turn; the host work
overlaps them. With the cores' compute skipped, the GEMMs keep 98% of their
time and the attention 61%: both are bound by data movement.

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, a Peano (llvm-aie >= 22) install
- `source utils/env_setup.sh` (see the repo README), `HF_HOME` for the Hub
  cache
- `pip install -r requirements.txt`

## Build and run

```bash
make compile                  # the prefill device (17 ops)
make run                      # a System One example
make verify                   # the CI gate: make reference if needed, then make gate
make reference                # CPU BF16 reference of the gate's request set (about a minute)
make gate WFORM=hilo          # the gate for one weight form
make bench                    # single-question latency at P = 512 ... 4096
```

Builds go to `BUILD` (default `./build`), outputs to `OUT` (default `./out`).
`MODEL_SOURCE` may name a local copy of the package (sha256 checked). NPU runs
should be serialized, e.g. `flock /tmp/mlir-air-npu.lock make ...`.

## Tests

| lit | What it checks |
| --- | --- |
| `run_npu2_compile_fused_prefill.lit` | the prefill device compiles; no NPU, no weights |
| `run_npu2_verify.lit` | `make verify` (PASS 67/67) |

There is no profile lit: the shared profile and prefill-sweep scripts read
TTFT and decode lines that a decision model does not produce; `make bench` is
its timing.

## Files

| File | Purpose |
| --- | --- |
| `d3_npu.py` | System One on the NPU (`D3Npu`): prompts, forward, readout, answers; `make run` |
| `../shared/fused_prefill/d3.py` | The forward pass on the fused_prefill device (`D3Prefill`), the device specs, the host library build |
| `d3_spec.py`, `device_build.py` | The edge device: GEMM shapes, attention groups, build |
| `kernels/d3_host.c` | Host row ops writing the GEMMs' bfp16 A |
| `d3_weights.py` | Pinned weights of both models: the edge safetensors, the d3-lite bundle |
| `reference.py`, `gate.py`, `bench.py`, `requests.json` | CPU reference, decision gate, timing, the frozen gate request set |

## Known limitations

- The attention re-reads each KV head's K / V for each of its 4 query heads
  on a column (the shared `attn_bfp16` op); this is most of the attention's
  data movement at long prompts.
- The GEMMs are bound by data movement on the shared GEMM op (A re-read per
  384-column round); BF16 weights stream as two BFP16 packets per K step.
- Prompts up to 4096 tokens (the device's KV records).
