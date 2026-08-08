# Qwen2.5-3B Q4_0 Prefill on AMD NPU2 (MLIR-AIR)

Structural port of [`../llama32_1b_q4nx/`](../llama32_1b_q4nx/) to Qwen2.5-3B:
4-bit weights, host dequant at load, then the whole transformer block —
RMSNorm, Q/K/V GEMM, QKV bias, RoPE, causal GQA flash attention, O projection,
SwiGLU, Down — on the NPU, with resident per-layer weight BOs and an on-device
LM-head GEMV.

The block builders are reused from [`../qwen25_3b/`](../qwen25_3b/) (same
relationship `llama32_1b_q4nx` has to `llama32_1b`); this example supplies the
4-bit weight path and the prefill driver.

## Status

| Gate | Result |
|---|---|
| First token, `"The capital of France is"`, 36 layers on NPU2 | **argmax 12095 = `" Paris"`** — PASS |
| Warm TTFT @ seq_len=2048 | **4.17 s** (491 tok/s prefill), NPU-dispatch 4.17 s, host 5 ms |
| End-to-end NPU prefill -> NPU fused decode, 36 layers, no reference model | coherent text, **9.54 tok/s** device-only decode |

The 36 transformer layers run on the NPU for both prefill and decode. The final
RMSNorm + LM-head projection and the argmax still run on the host each decode
token; moving them on-device is a performance item, not a correctness one.

Measured 2026-08-06 on a quiet NPU2 box (load < 0.1). Verify the box is idle
before trusting any timing here — a busy host makes this DDR-bandwidth-bound
workload ~30% slower.

## Quant codec: Q4_0, not Q4NX

The Llama example reads an already-4-bit `model.q4nx` bundle whose codec is
per-block **affine** quant (`w = scale*(q - min)`, unsigned nibbles). Qwen2.5-3B
has no such bundle, and its NPU kernels are built with `-DQ4_0` — a
**symmetric signed-int4** codec (`w = q*scale`, `q ∈ [-8,7]`, `scale = amax/8`,
per 32-element group along the reduction dim). So this example quantizes the
full-precision HF checkpoint directly, which is also strictly more accurate
than re-quantizing someone else's 4-bit weights.

The quantizer is `fused_decode/qwen25_3b_requant.requant_q4_0` — the same
function that builds the fused-decode Qwen weight cache — so **the prefill and
the fused NPU decode see bit-identical weight values**, and the prefill's KV
cache is a valid decode input.

Layer-0 quant error (rel L2 vs bf16), `make weights`:

| q | k | v | o | gate | up | down |
|---|---|---|---|---|---|---|
| 0.098 | 0.095 | 0.098 | 0.095 | 0.096 | 0.097 | 0.097 |

QKV **bias is not quantized** (it is not quantized in the reference design
either) — carried as bf16 and fused into the `rms_qkv_bias_rope` ELF.

## Model config

36 layers, emb 2048, 16 heads × head_dim 128, 2 KV heads (GQA group 8),
hidden 11008, vocab 151936, rope_theta 1e6, eps 1e-6, tied embeddings,
QKV bias, no QK-norm.

`head_dim=128` → head-first flash attention with host seq↔head transposes; see
[`../qwen25_3b/ARCHITECTURE.md`](../qwen25_3b/ARCHITECTURE.md).

## Quick start

```bash
make compile          # build/cache the prefill ELFs (no weights, no NPU)
make run              # first-token gate -> " Paris"
make bench BENCH_L=2048   # warm TTFT + per-ELF profile
make weights          # Q4_0 quant-error report, layer 0
```

Weights are downloaded from HuggingFace on first use and cached under
`~/.cache/huggingface/hub/`. Override with `--model` or
`QWEN_Q4_MODEL_SOURCE=<repo-id-or-local-dir>`.

`seq_len` is fixed at 2048: the GEMM registry shapes exist at M=2048, so the
prompt is padded and the last real token's row supplies the logit — the same
constraint the Llama Q4NX example has.

## Where the 4.17 s goes

Per-ELF, averaged over 36 layers (`make bench`):

| ELF | avg/layer | ×36 | note |
|---|---|---|---|
| `up` | 23.92 ms | 861 ms | GEMM 2048×2048×11008 |
| `gate` | 24.10 ms | 868 ms | GEMM 2048×2048×11008 |
| `flash_attn` | 22.48 ms | 809 ms | ~5% of the FLOPs, 21% of the time |
| `down_add` | 13.06 ms | 470 ms | GEMM 2048×11008×2048 + residual |
| `swiglu` | 10.52 ms | 379 ms | elementwise; 86 MB host→DDR per layer |
| `rms_qkv_bias_rope` | 7.30 ms | 263 ms | fused RMSNorm+QKV+bias+RoPE |
| `o_res_norm` | 5.48 ms | 197 ms | GEMM 2048×2048×2048 + add + norm |
| `lm_head_gemv` | 15.92 ms | 16 ms | once |

The GEMMs run at ~4–7 TFLOP/s; `flash_attn` runs at ~0.76 TFLOP/s, and
`swiglu` is pure data movement. Those two, plus the ~11% of wall spent in BO
writes of intermediates, are the open optimization targets.

## Hand-off to the fused NPU decode

`--dump-kv` writes the per-layer roped-K / biased-V cache, which
[`../../fused_decode/qwen_prefill_to_decode.py`](../../fused_decode/qwen_prefill_to_decode.py)
feeds straight into the fused single-dispatch Qwen decode. Both halves quantize
with the same Q4_0 codec, so the prefill's K/V are exactly what the decode's own
rope core would have appended at those positions, and the hand-off is a slice
(both sides are `[tokens, 256]` head-major) rather than a shuffle.

```bash
python3 qwen25_3b_q4_prefill.py --dump-kv /tmp/kv.npz --prompt "<32+ tokens>"
cd ../../fused_decode
QWEN_NLAYERS=36 python3 fused_decode_qwen.py           # 36-layer xclbin
QWEN_NLAYERS=36 python3 qwen_prefill_to_decode.py --kv /tmp/kv.npz --n-gen 16
```

Measured: prompt *"The capital city of France is called Paris, ... the very tall
iron tower that stands right"* -> *" in the middle. The Eiffel Tower. The tower
is 33"*, 104.8 ms/token device. The decode's first token equals the prefill's own
argmax, which is the cross-check that the KV hand-off is correct.

## Files

| File | Purpose |
|---|---|
| `qwen25_3b_q4_weights.py` | Q4_0 requant/dequant loader → the `qwen25_3b` `LlamaWeights` container |
| `qwen25_3b_q4_prefill.py` | `Qwen25Q4Prefill` — compile/preload/prefill/KV-cache/bench |
| `Makefile` | compile / run / bench / weights / clean |
| `../../fused_decode/qwen_prefill_to_decode.py` | KV hand-off + generation loop on the fused decode |
