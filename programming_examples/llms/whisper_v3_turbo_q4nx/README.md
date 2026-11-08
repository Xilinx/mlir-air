# Whisper large-v3-turbo Q4NX speech-to-text on AMD NPU2

This example reimplements the corresponding AMD NPU design, originally
developed by the [FastFlowLM](https://github.com/ROCm/FastFlowLM) team, using
the higher-level abstractions of the MLIR-AIR dialect. It runs greedy
transcription (batch 1, one 30 s window) of
[openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
with the weights FastFlowLM itself ships:
[`FastFlowLM/Whisper-V3-Turbo-NPU2`](https://huggingface.co/FastFlowLM/Whisper-V3-Turbo-NPU2)
at revision `594eecd2d80b20cbb04ef0099162335d1dd1899a`, file `model.q4nx`
(650,175,128 bytes, sha256
`8fb97604bf5762ee26efa696cfc9eb70724110358c7b4ca628a7973bf8a16291`). The
logical quantized values are used as supplied; only their device layout
changes.

Every projection GEMM, the conv front end's two convs and the LM head run on
one [fused_prefill](../shared/fused_prefill) device (`whisper_spec.py`). The
encoder's self-attention is one
[FlashAttention](../../flash_attention/kernel_fusion_based) ELF call per
layer over all 1536 rows (`whisper_attn.py`), on a second hardware context,
following FastFlowLM's split of its encoder into a matmul and an attention
xclbin.

## Architecture

| | |
|---|---|
| encoder | 2 conv layers (128 -> 1280, then stride 2), 32 pre-LayerNorm layers over 1500 frames: 20 heads x 64, no mask, GELU MLP 5120, biases on q, v, o, fc1, fc2 (none on k) |
| decoder | 4 layers: causal self-attention, cross-attention over the 1500 encoder frames, GELU MLP; learned positions, at most 448 tokens |
| head / vocab | 51866, tied to the embedding (the bundle carries a Q4 copy as `lm_head`) |
| norms | LayerNorm with bias, eps 1e-5; GELU in its erf form |

## Weights

`whisper_weights.py` reads the bundle, a safetensors file, from the Hub at the
pinned revision (cached under `HF_HOME`) or from `MODEL_SOURCE=<model.q4nx |
directory holding it>`, and checks its sha256 either way. The conv front end,
the embeddings, every LayerNorm and every bias are BF16. Each linear weight is
affine Q4 in blocks of 32 rows x 128 columns (2560 bytes): a bf16 scale and min
per row and group of 32 columns, then the 4-bit codes,
`w = scale * q + min`. That is the canonical AIR affine Q4 the device's int4
GEMM dequantizes, so the codes, scales and mins load unchanged (`q4()`).

## Placement

| | |
|---|---|
| NPU | the conv front end (both convs as GEMMs, BF16 weights), every encoder and decoder projection GEMM (int4 weights, BFP16 compute), the encoder's self-attention (FlashAttention, keys past the 1500th masked), the decoder's self- and cross-attention (attn_bfp16 at head dim 64, noncausal with a run-time key count), the cross-attention K/V projection, the LM head |
| host | log-mel spectrogram (`WhisperFeatureExtractor`), the token and position embedding lookup, LayerNorms, biases, GELU and residual adds between GEMMs (`shared/fused_prefill/hostops`), greedy sampling |

As in FastFlowLM, each conv is a GEMM whose rows of A are 3 consecutive
frames of its input (one frame apart for conv1, two for conv2), read as
overlapping rows of the time-major input rather than copied out.
The encoder's GEMMs run its 1500 frames as 12 chunks of 128 rows, two chunks
per dispatch; the attention reads keys [0, 1500), so the 36 pad rows never
enter a softmax. As in FastFlowLM, V's bias is folded into the output
projection's (each row of attention weights sums to 1), the qkv GEMMs write
into the buffer the attention reads, and the host's work on one chunk pair
overlaps the device's GEMMs on the others. The decoder steps one token at a
time, the prompt's tokens too. Its self- and cross-attention are the fused
prefill device's noncausal attention op; the self-attention reads keys
[0, pos + 1), which for a single query is the causal mask.

Decoding follows `WhisperForConditionalGeneration.generate` without timestamps:
the prompt `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>`,
`begin_suppress_tokens` at the first generated token and `suppress_tokens` at
every one (from the original checkpoint's `generation_config.json`), greedy
until EOS or `MAX_NEW` tokens (`whisper_npu.py`).

## Usage

```bash
make compile                      # the device and the attention ELF, into build/device
make run AUDIO=speech.flac        # 16 kHz, at most 30 s
make verify                       # the gate on the first N_VERIFY clips
make verify-full                  # the gate on every clip
make profile                      # complete-request timing on the first N_VERIFY clips
```

`make verify` writes the CPU BF16 reference once (`make reference`, into
`out/ref`). See `requirements.txt` for the Python packages.

## Tests

- `run_npu2_compile.lit`: builds the device and the attention ELF.
- `run_npu2_verify.lit`: `make verify`, the token gate.
- `run_npu2_profile.lit`: `make profile`, captured by `../bench/extract_perf.py`
  into `whisper_v3_turbo_q4nx.perf.json`: `ttft_ms` from the audio to the
  first token, `decode_tokens_per_sec` after it, and `real_time_factor`. The
  nightly benchmark publishes it in the speech-to-text table of the
  [LLM page](https://xilinx.github.io/mlir-air/dev/llms/).

The verify and profile tests need the three repos this example pins in
`../hf_models.txt` in the Hugging Face cache, the LibriSpeech clips among them.

## Verification

`gate.py` runs the production path (`whisper_npu.transcribe`) on the 73
LibriSpeech dev-clean utterances of
[`hf-internal-testing/librispeech_asr_dummy`](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
(revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`) and compares it with
`reference.py`, the original checkpoint in BF16 on the CPU decoded the same
way. Per clip both decode until EOS with every step's logits finite; at the
first step where the chosen tokens differ, each side's token must be in the
other side's top-5.

Measured 2026-11-08: all 73 clips pass. 48 of the 73 transcripts are token-for-token those of the
BF16 reference; the other 25 differ in casing, punctuation or a near-homophone
(the first divergence is often whether a transcript starts in sentence case).
The Q4 weights account for nearly all of it: the same weights computed in
float32 on the CPU match 50 of the 73, the original weights in float32 match
66. Word error rate against the dataset's text (lower-cased, punctuation
removed):

| | WER |
|---|---|
| original weights, BF16 (CPU, the reference) | 4.52% |
| original weights, float32 (CPU) | 4.43% |
| Q4NX weights, float32 (CPU) | 4.00% |
| Q4NX weights on the NPU | 3.83% |

## Performance

`make profile` on a Ryzen AI 7 PRO 350 (Krackan Point, NPU2), measured
2026-11-08, on the gate's first 8 clips (5 to 23 s of audio, 33 tokens on
average), after 5 s of warm-up:

| | |
|---|---|
| front end (log-mel, convs) | 12 ms |
| encoder (32 layers + cross K/V) | 457 ms |
| decode after the first token | 13.6 ms per token (73.4 tok/s) |
| request | 961 ms; real-time factor 0.089 (11.2x real time) |

The NPU was shared with other jobs while measuring; those runs were up to 15%
slower.

Per encoder layer the attention ELF takes about 3.6 ms and the two hardware
context switches around it about 2.5 ms. The decoder runs each token as about
40 one-row dispatches on the prefill device's 128-row ops.

FastFlowLM 1.0.13, built from
[ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) source (commit
`2de4084`) with the same bundle and measured 2026-11-07, served through
`flm serve --asr whisper-v3:turbo` (`/v1/audio/transcriptions`, 16 kHz WAV)
on the same clips and protocol:

| | FastFlowLM | this example |
|---|---|---|
| encoder with the convs (and here the log-mel) | 576 ms | 469 ms |
| decoder | 18.2 ms per step | 13.6 ms per step |
| request | 1280 ms; 8.4x real time | 961 ms; 11.2x real time |

FastFlowLM always decodes with timestamps (about 37 steps per request here
against 36), and its request time includes the HTTP round trip and WAV
decoding. Its WER on the 73 clips, normalized as above, is 27.91%: on
`1272-135031-0024` it repeats "the head-" until its token limit. Without that
clip it is 5.04% (this example: 3.87%).

## Limitations

- One 30 s window per request; no long-form chunking, timestamps or language
  detection (the language is given, English by default).
- Greedy decoding only.
