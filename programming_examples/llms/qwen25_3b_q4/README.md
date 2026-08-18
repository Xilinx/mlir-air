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
| End-to-end NPU prefill -> NPU fused decode, 36 layers, no reference model | coherent text, **30.3 tok/s** device-only decode (`LBUILD=256`; 28.2 at the 2048 default) |
| Top-k token-set inclusion vs HF bf16 (`make verify-full`) | **8/8 prompts PASS** (shared prompt file) — the CI gate |

The 36 transformer layers run on the NPU for both prefill and decode. The final
RMSNorm + LM-head projection and the argmax still run on the host each decode
token; moving them on-device is a performance item, not a correctness one.

Prefill measured 2026-08-06, decode re-measured 2026-08-18, both on a quiet
(load < 0.1) **AMD Ryzen AI 7 350 (Krackan Point)**, XRT 2.21.75. Verify the box
is idle before trusting any timing here — a busy host makes this
DDR-bandwidth-bound workload ~30% slower. The part matters as much as the load:
"NPU2" spans more than one array configuration, so a number from another NPU2
box is not comparable to these.

## Quant codec: Q4_0, not Q4NX

What differs for Qwen is the **on-device** format, not the weight bundle.
FastFlowLM ships Qwen2.5-3B as a `model.q4nx` bundle in the same per-block
affine encoding as Llama and Gemma (`w = scale*q + min`, unsigned nibbles,
32x256 blocks). But their Qwen decode design sets `#define Q4_0` (their
`Qwen2_5/decoding_3b/models/qwen2_3b.h`), which switches the kernel to a
**symmetric signed-int4**
form (`w = q*scale`, `q ∈ [-8,7]`, `scale = amax/8`, per 32-element group along
the reduction dim). The AIR port mirrors that, so its kernels are built
`-DQ4_0` too.

That mismatch is why this example ignores the bundle and quantizes the
full-precision HF checkpoint directly. For Llama and Gemma the bundle and the
device agree — affine to affine, same groups, same 4 bits — so the host
dequant/requant round-trip lands back on the same grid. Symmetric cannot
represent an offset range, so feeding it affine 4-bit weights would quantize
twice and lose more than starting from fp does.

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
make compile-decode   # build the fused decode kernels + decode_L<N> templates (no weights)
make run              # first-token gate -> " Paris"
make gen              # end-to-end: NPU prefill -> KV hand-off -> NPU fused decode
make bench BENCH_L=2048   # warm TTFT + per-ELF profile
make weights          # Q4_0 quant-error report, layer 0
```

Correctness gates (CI runs these via `run_npu2_*.lit`):

```bash
make verify           # top-k token-set inclusion vs HF bf16 (2 prompts x 32 tokens, k=5)
make verify-full      # the same over every prompt in the shared prompt file
make verify-paris     # fast prefill-only weight-integrity smoke (no decode build, no reference)
make diagnosis        # single-prompt lens through the shared runner (informational)
```

This is the same gate, on the same shared prompt file, as the other three Q4NX
examples: at the first token where NPU and HF bf16 disagree, each side's pick
must be in the other's top-5. [`verify_adapter.py`](verify_adapter.py) binds the
Q4_0 prefill and the fused decode to the shared runner contract.

This example used to carry its own 5-prompt file and score 4/5, which is why it
used to gate on something else. The prompt file is the whole difference —
measured on one build, changing only that file: 4/5 on the old one, 8/8 on the
shared one. Note that is not the same five prompts passing: the old lines were
mid-sentence continuations padded to 32-34 tokens to clear the long-gone fixed
decode window, the shared ones are complete instructions, and a next-token
distribution after a complete instruction is sharper. The decode cap (`LBUILD`)
is not a factor either way -- these prompts reach L~47, and the decode attends
over the real context, so a 256 and a 2048 build score identically.

The gate is not just a numeric check: it runs 32 fused decode dispatches per
prompt, so it is also the guard against an under-primed fill lock (a decode
deadlock), the failure mode the refeed mechanism exists to prevent.

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

`make compile-decode && make gen` drives the whole thing. By hand:

```bash
python3 qwen25_3b_q4_prefill.py --dump-kv /tmp/kv.npz --prompt "<32+ tokens>"
cd ../../fused_decode
QWEN_NLAYERS=36 python3 fused_decode_qwen.py           # 36-layer xclbin
QWEN_NLAYERS=36 python3 qwen_prefill_to_decode.py --kv /tmp/kv.npz --n-gen 16
```

`QWEN_NLAYERS` must be exported for **both** commands: `fused_decode_qwen` reads
its geometry from the environment at import and defaults to 1 layer (a fast
lowering check), so omitting it packs a 1-layer weight stream for a 36-layer
xclbin.

**Context.** The decode grows a KV cache inside a build-time cap. The attention
herd takes the context length `L` as a scalar operand, which AIR gives an RTP slot
the runtime sequence writes, so `DecodeInstsGen` rewrites the RTP-L words and the
KV-append offset per token from one template pair -- the same per-token
specialization the Llama Q4NX decodes use. The device therefore appends at the
real position and attends over the real context: any prompt up to `LBUILD` works,
with no minimum, and the decode's first token always equals the prefill's argmax
(the cross-check the adapter warns on).

This replaced a fixed 32-slot sliding window that could not run a prompt shorter
than 32 tokens and silently dropped the head of a longer one. `make compile-decode
LBUILD=<N>` sets the cap; it costs two builds, one L apart, to calibrate the
slope. The default is **2048**, matching the three Llama/Gemma Q4NX decodes.

Measured: prompt *"The capital city of France is called Paris, ... that stands
right in the middle of"* -> *" the city. It is called the Eiffel Tower, and it
was built"*. Same box and power mode, 16-token `make gen`, output byte-identical
in all four cells:

| `LBUILD` | device | `NONATTN_EXTRA=-DQ4_SFIX_MODE=0` |
|---|---|---|
| 256 | **33.0 ms/token** (30.3 tok/s) | 81.8 ms/token (12.2 tok/s) |
| 2048 | **35.5 ms/token** (28.2 tok/s) | 84.2 ms/token (11.9 tok/s) |

The 2.5 ms between the caps is the padded readback: the KV nd-DMA streams
`ATTN_MAXL` positions whatever the real context is. `make
compile-decode-dynseq` takes the block count off the runtime scalar instead and
removes it.

The right-hand column is the same design built against the pre-xor-bias int4
sign fix (see [`kernels/q4_k.h`](../../fused_decode/kernels/q4_k.h)), measured in
the same session on the same box so the 2.4x is not a cross-box comparison. It
is also what this table used to report.

## Files

| File | Purpose |
|---|---|
| `qwen25_3b_q4_weights.py` | Q4_0 requant/dequant loader → the `qwen25_3b` `LlamaWeights` container |
| `qwen25_3b_q4_prefill.py` | `Qwen25Q4Prefill` — compile/preload/prefill/KV-cache/bench |
| `Makefile` | compile / compile-decode / run / gen / verify / verify-full / bench / weights / clean |
| `verify_adapter.py` | Binds the prefill + fused decode to the shared `verify/` runner contract |
| `run_npu2_*.lit` | CI gates: prefill compile, decode compile, verify, profile |
| `../../fused_decode/qwen_prefill_to_decode.py` | `QwenFusedDecoder` (KV hand-off + per-token dispatch) and its CLI |
