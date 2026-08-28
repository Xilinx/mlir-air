# Qwen2.5-3B Q4_0 Prefill + Fused Decode on AMD NPU2 (MLIR-AIR)

This implementation reimplements the corresponding AMD NPU LLM design,
originally developed by the [FastFlowLM](https://github.com/ROCm/FastFlowLM)
team, using the higher-level abstractions of the MLIR-AIR dialect.

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
| End-to-end NPU prefill -> NPU fused decode, 36 layers, no reference model | coherent text, **23.2 tok/s** logits-out (`LBUILD=2048`) |
| Top-k token-set inclusion vs HF bf16 (`make verify`) | **2/2 prompts PASS** — the CI gate |

The 36 transformer layers **and the LM head** run on the NPU, for both prefill
and decode: one dispatch per token produces vocab logits, and only the argmax is
left on the host. The retired Qwen-only builder stopped at the final hidden
state and projected the 152k-row head on the CPU, which is why its published
25.3 tok/s was a *device-only* number against 4.5 tok/s end to end.

Prefill measured 2026-08-06, decode re-measured 2026-08-27, both on a quiet
(load < 0.1) **AMD Ryzen AI 7 350 (Krackan Point)**, XRT 2.21.75. Verify the box
is idle before trusting any timing here — a busy host makes this
DDR-bandwidth-bound workload ~30% slower. The part matters as much as the load:
"NPU2" spans more than one array configuration, so a number from another NPU2
box is not comparable to these.

## Quant codec: Q4_0, not Q4NX

FastFlowLM ships Qwen2.5-3B as a `model.q4nx` bundle, but the bundle carries the
**Q4_0** variant of that container, not the affine Q4NX one Llama and Gemma use:
the `mins` field is exactly zero in every tensor of all 36 layers plus the
lm_head, and the nibbles are signed (`w = q*scale`, `q ∈ [-8,7]`, per 32-element
group along the reduction dim; the scale may be negative, the llama.cpp
`d = max/-8` convention). Reading it as affine gives rel_l2 ≈ 2.8 against the
reference — noise. Their decode design sets `#define Q4_0`
(`Qwen2_5/decoding_3b/models/qwen2_3b.h`) to match, and so does this port.

The bundle also folds an AWQ-style per-input-channel smoothing into the weights:
the RMSNorm weight absorbs `s` while q/k/v are divided by it, and the GLU's `up`
rows are scaled by `t` with `down`'s columns divided by it. Both are exact
rewrites, so the bundle is self-consistent and a tensor-by-tensor comparison
against the HF checkpoint disagrees by design. Nothing in the loader needs to
know `s` or `t`.

Because the shipped codes are already in the block geometry the decode cascade
wants, they are carried through **untouched** — only re-ordered into the
device's stream order. There is no dequant/requant round trip to lose accuracy
to, and the prefill and the fused decode see bit-identical weight values, so the
prefill's KV cache is a valid decode input. Pointing `MODEL=` at a
full-precision HF checkpoint instead still works; it is quantized once, on load,
through the same `q4_0_codec.requant_q4_0`.

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
make gen              # end-to-end: NPU prefill -> NPU fused decode
make bench BENCH_L=2048   # warm TTFT + per-ELF profile
make weights          # Q4_0 weight-load smoke, layer 0
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
prefill and the fused decode to the shared runner contract.

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

The prefill's per-layer roped-K / raw-V cache seeds the decode's device-resident
KV cache directly. Both halves read the same Q4_0 weights, so the prefill's K/V
are exactly what the decode's own rope core would have appended at those
positions, and the hand-off is a slice (both sides are `[tokens, 256]`
head-major) rather than a shuffle.

`make compile-decode && make gen` drives the whole thing;
[`qwen25_3b_q4_inference.py`](qwen25_3b_q4_inference.py) is the driver, and it
sets the engine's geometry itself before importing `fused_decode.py`, so the
runner cannot be handed a different model than the xclbin was built for.

**Attention placement.** Qwen2.5-3B is the only model in the engine's table with
2 KV heads, so its attention herd is 2 compute units in a single column rather
than the usual 4 across two. Which column is load-bearing: swept on NPU2, cols
0/1/2/6/7 do not build, col 3 reaches `COMPLETED` about 1 run in 10 and the rest
time out at a random position, and cols 4 and 5 are both 20/20 over 29
dispatches. `ATTN_PCOL` (default 4) is the knob, so the sweep can be repeated.

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

Measured on the **retired Qwen-only builder** (device-only, LM head on the
host), 16-token `make gen`, same box and power mode, output byte-identical in
all four cells. Kept for the `Q4_SFIX_MODE` ratio, which is a property of the
kernel and still holds; the absolute numbers are not comparable to the
logits-out rate in Status above and have not been re-measured:

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
| `qwen25_3b_q4_weights.py` | Q4_0 bundle loader/dequant → the `qwen25_3b` `LlamaWeights` container |
| `qwen25_3b_q4_prefill.py` | `Qwen25Q4Prefill` — compile/preload/prefill/KV-cache/bench |
| `Makefile` | compile / compile-decode / run / gen / verify / verify-full / bench / weights / clean |
| `verify_adapter.py` | Binds the prefill + fused decode to the shared `verify/` runner contract |
| `run_npu2_*.lit` | CI gates: prefill compile, decode compile, verify, profile |
| `qwen25_3b_q4_inference.py` | `FusedDecoder` (KV seed + per-token dispatch) and the generate / REPL CLI |
| `qwen25_3b_q4_requant.py` | Cascade-packs the bundle's Q4_0 blocks into the decode weight cache |
