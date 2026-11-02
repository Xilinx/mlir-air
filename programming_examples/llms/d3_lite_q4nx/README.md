# d3-lite Q4NX decision model on AMD NPU2

This implementation reimplements the corresponding AMD NPU LLM design,
originally developed by the [FastFlowLM](https://github.com/ROCm/FastFlowLM)
team, using the higher-level abstractions of the MLIR-AIR dialect. It runs on
the shared [fused_prefill](../shared/fused_prefill) device as the
[Qwen3.5-0.8B prefill](../qwen35_0_8b_q4nx) does, by subclassing its
`Qwen35Prefill`.

*System One* decisions of the vLLM Semantic Router decision model
[`vllm-sr/d3-lite`](https://huggingface.co/vllm-sr/d3-lite) at revision
`03560c26f5199c3c04129a2ccb38960c913635e0`, with the weights FastFlowLM ships
for it: [`FastFlowLM/D3-Lite-NPU2`](https://huggingface.co/FastFlowLM/D3-Lite-NPU2)
at revision `51563157b724a1bf69c5c177e000dd7038acd63c`, file `model.q4nx`
(sha256 `04a20706f44dcc8f142a1774034fc2a01110bef14031fc8ccf26295db00bef42`,
the file FastFlowLM 1.0.11 serves as `d3:lite`). The logical quantized values
are used as supplied; only their device layout changes. One forward pass per
question over the package's own prompt format, the option logits read by a
255-way FP32 readout at the last prompt position, the option softmax on the
host. Batch 1, text only, no decode.

The engine, runtime, gate, reference, timing and request set are shared with
[`d3_edge`](../d3_edge), whose README describes the mapping in full; this
directory holds the d3-lite device, its Makefile and tests.

## Architecture

| | |
| --- | --- |
| Layers | 24: Gated DeltaNet, with gated full attention every 4th layer |
| d_model / head_dim / MLP | 1024 / 256 / 3584 |
| Attention | 8 query / 2 KV heads, q/k RMSNorm, partial RoPE 64/256, output gate; **noncausal** |
| Gated DeltaNet | 16 heads, conv1d 4 taps, delta rule on an FP32 state; causal |
| Output | final RMSNorm of the last row, FP32 readout [255, 1024], softmax over the question's options |

## Weights

The bundle has the layout of FastFlowLM's Qwen3.5-0.8B `flm_q4k_high_precision`
bundle, so it is read with [`q35_weights.py`](../qwen35_0_8b_q4nx/q35_weights.py):
Q4K projections on the GEMM's affine int4 grid (scale bf16(d·sc), min
bf16(dmin·m)), the Q8 beta / alpha projections as two Q4 halves along K
against `[x | x]`, the embedding from the tied Q8 lm_head. The tokenizer,
`decision_config.json`, `readout.safetensors` and `d3_runtime.py` come from
`vllm-sr/d3-lite` (the same files as in FastFlowLM's package).

## Placement

As [d3-edge](../d3_edge/README.md#placement): every projection GEMM (BFP16
compute on the Q4 weights) and the noncausal softmax attention on one
fused_prefill device, one hardware context per forward pass; tokenization,
embedding, the row ops, the Gated DeltaNet and the FP32 readout on the host.
d3-lite also runs the beta / alpha GEMM on the device, which d3-edge folds
into its DeltaNet input projection.

## Correctness

d3 has no LM head and no causal-LM reference, so the repository's
`llms/verify` top-5 next-token check does not apply. **`make verify` runs the
d3 decision gate** (`d3_edge/gate.py`, `run_npu2_verify.lit`) instead, the
model's correctness gate, on the 67 questions of `d3_edge/requests.json`. Its
rule is d3-edge's (finite logits, each answer in the other's top 2, equal
answers where the reference's margin is at least 0.05, input ids equal).

The reference (`make reference`, `d3_edge/reference.py --model lite`) is the
upstream d3 runtime on CPU in BF16 with every weight the bundle quantizes
replaced by its exact dequantized value (Q4K d·sc·q + dmin·m, Q8 scale·q,
rounded to the bf16 parameters) and the embedding taken from the bundle's Q8
lm_head; norms, conv, dt_bias, A_log and the readout are the upstream ones.
It therefore checks the NPU path against the weights it runs.

| | Gate | Answers agree | max \|Δp\| |
|---|---|---|---|
| NPU vs the Q4K reference | PASS 67/67 | 67/67 | 0.075 |

For information: the Q4K reference itself agrees with upstream BF16 d3-lite on
59 of the 67 answers, the effect of the quantization (FastFlowLM reports 90.6%
top-1 agreement with BF16 over 7,226 questions).

## Results

NPU2 (Ryzen AI MAX+ 395), MLIR-AIR built from this tree's base, Peano
(llvm-aie) 22.0.0.2026102001, mlir-aie 1.4.5.dev83, XRT 2.21.0, 2026-11-02.
`make bench`: one `route` question grown to exactly P tokens; the time from
the token ids on the host to the option probabilities on the host; at least
5 s of warm-up, then at least 3 runs and 10 s measured; medians.

| ms | P = 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| d3-lite | 89.4 | 165.4 | 340.2 | 775.8 |
| FastFlowLM 1.0.11 `d3:lite` | 182.5 | 376.9 | 750.0 | 1596.8 |

FastFlowLM 1.0.11 ran on the same weights and prompts (`flm serve --decision
d3:lite`, client wall time per `/v1/systemone` request over one persistent
connection; it reported P input tokens for every P), in the same session.

## Prerequisites

- NPU2 (Strix / AIE2P) + XRT, a Peano (llvm-aie >= 22) install
- `source utils/env_setup.sh` (see the repo README), `HF_HOME` for the Hub
  cache
- `pip install -r requirements.txt`

## Build and run

```bash
make compile                  # the prefill device (13 ops)
make run                      # a System One example
make verify                   # the CI gate: make reference if needed, then make gate
make reference                # CPU reference of the gate's request set (a few minutes)
make bench                    # single-question latency at P = 512 ... 4096
```

Builds go to `BUILD` (default `./build`), outputs to `OUT` (default `./out`).
`MODEL_SOURCE` may name a local `model.q4nx` or a directory holding it (sha256
checked). NPU runs should be serialized, e.g.
`flock /tmp/mlir-air-npu.lock make ...`.

## Tests

| lit | What it checks |
| --- | --- |
| `run_npu2_compile_fused_prefill.lit` | the prefill device compiles; no NPU, no weights |
| `run_npu2_verify.lit` | `make verify` (PASS 67/67) |
| `run_npu2_prefill_sweep.lit` | decision latency at P = 512 ... 4096 for the nightly prefill table (`make profile-fused-prefill`, through `bench/sweep_prefill.py --fused`); latency only |

## Files

| File | Purpose |
| --- | --- |
| `d3_spec.py`, `device_build.py` | The lite device: GEMM shapes, attention groups, build |
| `../shared/fused_prefill/d3.py` | The forward pass on the fused_prefill device (`D3Prefill`) |
| `../d3_edge/` | Runtime (`d3_npu.py`), weights, host C, reference, gate, timing, request set |

## Known limitations

- As d3-edge's: the attention re-reads each KV head's K / V for each of its 4
  query heads; the GEMMs are bound by data movement; prompts up to 4096 tokens.
- Of the host work, the 18 Gated DeltaNet layers are not fully hidden behind
  the device at long prompts.
