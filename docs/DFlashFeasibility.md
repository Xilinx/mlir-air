# DFlash on NPU2: what is verified, what is modelled, what is unknown

Speculative decoding with [DFlash](https://github.com/z-lab/dflash)
(`z-lab/Qwen3-4B-DFlash-b16`) on top of this repo's `fused_decode` superkernel,
targeting Qwen3-4B on NPU2. Excluded from the published site (`exclude_docs` in
`mkdocs.yml`).

**This document was rewritten from a 4810-line running log after an audit found
several load-bearing claims to be false — most importantly a measurement taken
on llama-3.2-1b and written up as though it were qwen3-4b.** The rewrite keeps
only claims traceable to a specific run on a specific model. Section 9 records
what was removed and why, so the same errors are not re-derived.

**Every claim below carries a tag:**

- `[hw]` — measured on real NPU2 hardware. The model is always named.
- `[static]` — read off the builder, a bundle count, an ELF, or source. Real,
  but not a run.
- `[cpu]` — measured, but on CPU (the drafter acceptance work).
- `[model]` — arithmetic on top of measured inputs. Not a measurement.
- `[unknown]` — open.

## 1. Status

| | state |
|---|---|
| qwen3-4b batch 1 decode, end to end | **works** `[hw]` |
| DFlash acceptance rate, real drafter vs real target | **measured** `[cpu]` |
| **qwen3-4b (target) batch 8 decode** | **passes the equivalence gate** `[hw]` |
| **qwen3-4b-draft (drafter) batch 8 decode** | **passes the equivalence gate** `[hw]` |
| **RMS_BAND_STREAM level 3** (the batch-16 route) | **passes the equivalence gate at 36 layers, same output as levels 0 and 2** `[hw]` (§5.4 x) |
| qwen3-4b batch 16 (what the checkpoint needs) | **builds** — every L1/L2/BD ceiling cleared (§5.4 xii); hangs in wave 0 (§5.4 xiii) `[hw]` |
| batch-8 dispatch cost, both models | **measured on device** `[hw]` (§6) |
| accepted-length DISTRIBUTION, blocks 16 / 8 / 4 | **measured** `[cpu]` (§3.1) — **block 16 is the wrong block size** |
| bidirectional draft mask, selected per dispatch on ONE program | **verified** `[hw]` (§3.2) |
| the drafter's non-decode half (`fc`, context K/V, `k_norm`, RoPE) | **one 24-launch dispatch, int4, gates on device** `[hw]` (§3.3, §3.4) |
| **the verify pass on the shipping driver** | **batch 8 reproduces eight batch-1 steps, argmax 8/8** `[hw]` (§3.5) |
| the pre-pass vs the **real** drafter's own KV rows | **cos 0.991–0.997, engine 1e-02** `[hw]` (§3.6) |
| **the whole draft pass** (pre-pass + bidirectional 5-layer decode) | **runs on the array; 4/7 draft tokens match the bf16 drafter** `[hw]` (§3.7) |
| what quantizing the drafter costs in ACCEPTANCE | **measured: ~16%** — 3.981 tok/verify against the bf16 drafter's 4.73 (§3.12) |
| **on-device draft + verify loop** | **runs end to end; speculative == non-speculative** `[hw]` (§3.8) |
| **the batch-8 verify pass beyond a 16-token context** | **fixed** — a shared-L1 lock was taken one loop level too high (§3.10, §3.11) |
| **the pre-pass (the third PDI)** | **weight-bandwidth-bound at 0.46 GB/s, 118.5 ms/block** (§3.13); layout + format gated on CPU, engine hook inert, arm wiring open (§3.14) |
| **the verify pass being exact** | **fixed** — a BFP16 shared exponent spanned the context boundary (§3.9) |
| DFlash speedup with the DEVICE drafter | **measured: 0.69×** at 3.981 tok/verify against a 328.3 ms step; 0.92× with the pre-pass hidden (§3.12) |

The short version: **the loop runs end to end on the array and accepts 3.98
tokens per verify dispatch on gsm8k, and it is still 0.69× — the pre-pass, not
acceptance, is what it costs** (§3.12). Getting there took two numerics fixes in
the attention path: a BFP16 shared exponent spanning the context boundary (§3.9)
and a shared-L1 lock taken one loop level too high, which silently gave every
attention block of a token the *last* block's scores at batch > 1 (§3.11).

§6's 217.9 ms step against a 56.9 ms baseline token — break-even 3.83 — prices
draft + verify and **omits the pre-pass**, which measures 82 ms, a quarter of a
real 328.3 ms step. Break-even is really 5.76. What is still missing is batch
**16** (the checkpoint's block size), and a way to overlap or amortize the
pre-pass.

Batch 16 needs `RMS_BAND_STREAM` level 3, and **level 3 now works**: at batch 8
and 36 layers it passes the equivalence gate at 5.56e-03 and produces the SAME
output as levels 0 and 2, for **4.4%** more dispatch time (§5.4 x). Ten defects
between here and there — BD-block exhaustion, a lock-count mismatch, the rms
core needing a third S2MM port, two classes of descriptor that are correct AIR
but do not survive lowering, the launch-side POSITION of the banded ph0 feed,
arming every banded feed one weight phase ahead of the await that depends on it,
giving `h` its own DDR region instead of sharing X, and two compiler fixes in
`AIRRtToNpuPass`.

The last one was the interesting one, and it had nothing to do with the rms
core. `@rmsX` and `@outY` both reach that core from the SOUTH and, packet-
switched, share one physical stream. Level 3 is the first shape that leaves a
band in flight while the core waits on `@outY` — so the band parks at the head
of the link and the projection output queues behind it, forever. Giving `@rmsX`
its own circuit-switched channel at level 3 fixes it in one line; the reason it
was a packet (converging with the o-proj/down id on one port) stopped applying
when `RMS_W_ON_X` split the ports (§5.4 ix).

Batch **16 now BUILDS**, five walls further on. After the rms core came the
attention core's BD-block limit (`@attnO` was one put per token; the
un-interleave moved to the memtile, which has a fourth BD dimension where a
compute tile does not), then three allocation ceilings in a row — 512 KB of L2
on one memtile, 24 BD ids shared per memtile channel, and 64 KB of L1 on the rms
core where the budget model had never counted the per-herd RTP word (§5.4
xi-xii). It still hangs in wave 0 with nothing written; batch 8 with the
identical settings passes, so it is the batch and not the fixes (§5.4 xiii).

**And that comparison has now been made, and it says to stop.** §3.1 measures
the accepted-length distribution the block-size question needs — 2058 blocks at
block 16, then the same 60 prompts re-run natively at blocks 8 and 4. An extra
verify slot costs **0.368** of a baseline token step (§6 said 0.30 by counting
only the verify half; the draft pass batches too), so slot *k* pays iff
`P(produced ≥ k) > 0.368`, and that runs out at k=5. Priced against §6's
measured dispatch times:

| block | math | code | chat |
|---|---|---|---|
| 8 | **1.24×** | **1.24×** | 0.88× |
| 16 | 0.90× | 0.89× | 0.57× |

**Block 16 is slower than not speculating at all, in every category.** Batch 16
is the wrong thing to finish: the wave-0 hang is now an unclaimed bug rather
than a blocker, and batch 8 — which already works — is the configuration to
build the loop on.

**The loop is now built (§3.8), and building it found — and fixed — an engine
defect that had to go before any acceptance number could mean anything.** A
decode step at context length L depended on KV rows `L..ceil(L/8)*8-1`. The
cause was not attention: `S·V` is a **BFP16** mmul contracting over keys, 8 keys
share one exponent, and when `L % 8 != 0` that shared exponent spans past the
context boundary — so out-of-context rows cost the in-context rows *mantissa
bits* (§3.9). Invisible in ordinary decode, where those rows are zero and a zero
exponent never wins a max; not invisible in a verify pass, where they hold the
block's own later tokens. Four lines in `attn_kv_blk` zero the V tile's tail
before the mmul. Ordinary decode is bit-identical across the fix, and the
speculative loop now reproduces the non-speculative stream exactly.

## 2. What DFlash is

Read from the checkpoint's own source and the current upstream package, not
inferred `[static]`:

- **Block size 16, fixed by the checkpoint** (`-b16`). This is why batch 16,
  not batch 8, is the number that matters for a real deployment.
- The drafter is **5 Qwen3-4B-shaped layers**, not a truncated target. Its
  per-layer geometry is identical to the target's; only the layer count differs
  (`_MODELS["qwen3-4b-draft"]`, `UNI_DEC=5`).
- Draft attention is **non-causal cross-attention** (`is_causal=False`): Q comes
  from embedding a 16-token block (mostly `mask_token_id=151669`), K/V are
  `concat(k_proj(target_hidden), k_proj(hidden_states))`.
- **Context fusion** is one linear plus one norm: `fc` (12800→2560, no bias)
  then `hidden_norm`, over the target's hidden states tapped at
  `target_layer_ids=[1,9,17,25,33]` (HF `hidden_states` indices `[2,10,18,26,34]`,
  `offset=1`).
- **The LM head is tied to the target's embedding.** The checkpoint's
  safetensors header carries only `fc.weight [2560,12800]`,
  `hidden_norm.weight [2560]`, `norm.weight [2560]` beyond the layer weights —
  no `embed_tokens`, no `lm_head`. So the draft pass pays the target's LM-head
  cost.
- **The draft-side KV cache persists and grows**, one entry per accepted
  position. An earlier reading of this document concluded it was stateless;
  that was wrong, and traced to `transformers.Cache.crop()` changing meaning
  between the version the checkpoint was written against (4.57.3, "keep first
  N") and the installed one (5.15, "remove N from end").

Coupling between the two models is **one handoff per block** — target runs its
36 layers, 5 taps are fused to one vector, the drafter's 5 layers consume it.
There is no per-layer exchange.

## 3. Acceptance rate: the number the whole idea rests on

Measured with the **real, unmodified upstream `dflash_generate`** against the
real target, greedy (`temperature=0.0`, which is upstream `cli.py`'s own
default), block 16 `[cpu]`.

The headline result, on real dataset samples drawn exactly the way upstream's
own `benchmark.py` draws them (seeded `random.Random(42)`):

| dataset | blocks | mean accepted / 16 |
|---|---|---|
| gsm8k | 610 | **6.09** (38.0%) |
| humaneval | 660 | **6.03** (37.7%) |
| **overall** | **1270** | **6.06** (37.9%) |

Close to the paper's own Table 1 for Qwen3-4B (math 6.53, code 7.84).

**Thinking mode is the whole story on the gap that used to be here.** Earlier
passes of this work measured τ≈2.2/16 and concluded the drafter was weak. That
was Qwen3's chat template defaulting to thinking mode enabled; the paper's
Table 1 caption says "with thinking mode disabled". Same GSM8K prompt, same
code, `enable_thinking=False`: **2.79 → 6.16** `[cpu]`.

Ruled out as explanations, each by direct experiment:

- **Quantization** — clean bf16 target gives 2.34 where Q4NX gives 3.00 on the
  same prompt `[cpu]`. Not the cause.
- **A bug in this repo's harness** — the real upstream `dflash_generate` on a
  raw prompt gives 2.32 against this harness's 2.22 over 10 prompts `[cpu]`.
  The harness was never the problem.

**It is strongly task-dependent, and that matters for deployment** `[cpu]`:

| category | blocks | mean / 16 |
|---|---|---|
| math | 74 | 6.30 (39.4%) |
| code | 79 | 7.56 (47.2%) |
| open-ended chat | 200 | **2.62** (16.4%) |

Chat sits near the original low numbers. A chat-oriented deployment gets a
different answer from a math/code one.

**What this measurement is not:** the drafter ran on CPU against a recorded
target continuation. Nothing here is an on-device batched draft/verify loop.
The replay is valid because the target's greedy decode is a deterministic
function of the token prefix, so replaying accept/reject against a recorded
continuation is equivalent to interleaving — but it measures *acceptance*, not
*speed*.

Harness detail worth keeping: driving the checkpoint's own model code required
a `SimpleCropCache` reimplementing the old `crop()` semantics, verified
bit-exact against the model's own `past_key_values=None` forward for block 0
before being trusted. A first attempt that hand-transcribed the attention was
**not** bit-exact (max abs diff 1.25) and was discarded rather than patched.

### 3.1 The distribution, and what it says about the block size

§3's means were never enough: `E[min(a, 8)]` cannot be recovered from a mean
over 16, and the tail is the entire thing a larger block buys. So the same run
was repeated keeping the raw per-block lengths
(`dflash_acceptance_hist.py`), 20 prompts per dataset, greedy, thinking off,
real upstream `dflash_generate` `[cpu]`.

**First, what the quantity is**, because the name misleads and §3's column
header was loose about it. Upstream appends `produced = accepted + 1` — the
drafted tokens that matched, plus the bonus token the verify pass emits for
free. It is tokens per speculative step, range 1..16 at block 16, never 0. That
is the same scale §6's break-even is stated in, so the two compare directly.

The harness reproduces §3 exactly — same seed, same selection, same block
counts (gsm8k 610 blocks / 6.09, humaneval 660 / 6.03), which is the cross-check
that says the distribution below and the mean above come from the same thing.
The chat row is now mt-bench, the dataset upstream's own benchmark uses, rather
than §3's three hand-written prompts; it scores 3.86 where those scored 2.62.

| category | dataset | prompts | blocks | mean / 16 |
|---|---|---|---|---|
| math | gsm8k | 20 | 610 | 6.09 |
| code | humaneval | 20 | 660 | 6.03 |
| chat | mt-bench | 20 | 788 | 3.86 |
| **all** | | 60 | **2058** | **5.22** |

`P(produced ≥ k)` — the chance verify slot *k* emits a token at all, which is
the value side of the per-slot trade `[cpu]`:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| math | 1.00 | 0.88 | 0.73 | 0.61 | 0.53 | 0.45 | 0.38 | 0.33 | 0.26 | 0.22 | 0.16 | 0.09 | 0.05 |
| code | 1.00 | 0.84 | 0.70 | 0.58 | 0.46 | 0.39 | 0.34 | 0.31 | 0.27 | 0.23 | 0.18 | 0.13 | 0.10 |
| chat | 1.00 | 0.73 | 0.51 | 0.38 | 0.27 | 0.20 | 0.15 | 0.13 | 0.10 | 0.09 | 0.06 | 0.05 | 0.04 |
| **all** | 1.00 | 0.81 | 0.64 | 0.51 | 0.41 | 0.33 | 0.28 | 0.25 | 0.20 | 0.17 | 0.13 | 0.09 | 0.06 |

**A slot has to clear 0.368, and the curve crosses it between k=5 and k=6.**
Slots 6 through 16 all cost more than they return; slots 9-16, which is what
finishing batch 16 buys, return 0.20 down to 0.06 against a bar of 0.368.

**The truncation step was removed rather than assumed.** Predicting block 8 from
a block-16 run means assuming `produced_B ≈ min(produced_16, B)`, and that is not
obviously safe — the drafter's attention is non-causal across the whole block, so
a native block-8 draft is not a block-16 draft cut in half, and the checkpoint is
trained at 16. So the whole 60-prompt sweep was re-run natively at block 8 and
block 4. The drafter works fine at both, and the approximation holds `[cpu]`:

| mean produced | math | code | chat | all |
|---|---|---|---|---|
| block 8, native | 4.73 | 4.75 | 3.36 | 4.25 |
| block 8, predicted from the block-16 run | 4.92 | 4.63 | 3.36 | 4.23 |
| block 4, native | 3.21 | 3.23 | 2.63 | 3.03 |
| block 4, predicted from the block-16 run | 3.23 | 3.13 | 2.62 | 2.96 |

Within 4%. Two useful things fall out: the block-16 histogram is a valid
predictor for smaller blocks, and **the b16 checkpoint is not actually pinned to
block 16** — it runs at 8 and 4 without retraining, and its per-step yield goes
*up* as a fraction (4.73/8 = 59% against 6.09/16 = 38%).

**The block size is numerically free, not bit-free** `[cpu]`. Greedy speculative
decoding keeps a drafted token only where it equals the target's own `argmax`
and emits the target's `argmax` as the bonus, so in exact arithmetic every block
size produces the identical stream. It does not.
`dflash_block_equiv.py` ran blocks 1/4/8/16 on the same prompts: gsm8k identical
everywhere, humaneval diverged at token 141 (block 8 only — block 16 matched),
mt-bench at 175 and 103. Non-monotone in the block size, and late.

`dflash_tie_probe.py` found the cause and it is arithmetic, not algorithm. A
batch-B verify pass and a batch-1 pass take different reduction orders through
sdpa and the projections; in bf16 that moves a logit by a fraction of an ulp,
which flips the `argmax` wherever the top two are nearly tied, and one flip
changes every token after it. The divergences land exactly where that predicts:

| | top-2 gap at the divergence | percentile among that run's 200 decisions |
|---|---|---|
| humaneval, token 141 | 0.125 | **0.0** (the smallest gap in the whole generation) |
| mt-bench, token 175 | 0.250 | 1.5 (p1 = 0.125) |

**This is the right gate to set, and it is not an exact token match** — no block
size can hold that, on CPU or on device, and a device implementation adds Q4NX
quantization on top. The repo's existing gates are the correct ones: `make
verify`'s top-5 token-set inclusion, and `batch_equiv.py`'s 5e-2 at the layer
level. Divergence at a near-tie is expected behaviour; divergence at a wide
logit margin is a bug.

Priced against §6's measured dispatch times, using the native block-8 and
block-4 acceptance and the interpolated block-4 dispatch cost `[hw]` + `[cpu]`:

| block | step ms | math | code | chat | all |
|---|---|---|---|---|---|
| 4 | 134.0 `[model]` | 1.36× | 1.37× | 1.12× | 1.29× |
| 8 | 217.9 `[hw]` | **1.24×** | **1.24×** | 0.88× | 1.11× |
| 16 | 385.8 `[model]` | 0.90× | 0.89× | 0.57× | 0.77× |

**Block 16 is below 1.0× everywhere** — slower than plain autoregressive decode
on math, on code, and on chat. Block 8 is the best configuration that exists
today, and it is the one that already works.

**Block 4 looks better still, and is not buildable.** `proj_qmm.cc`'s
`proj_qmm_mm_flush_row` de-tiles for `aie::mmul<8,8,8>` and asserts
`PROJ_MM_BATCH % 8 == 0`; at batch 4 `q4k_mmul_any` picks `mmul<4,8,8>`, `size_C`
is 32 rather than 64, and `RA` integer-divides to zero. The kernel comment
already diagnoses this precisely and notes `q4k_mm.h` itself is bit-exact at
batch 4 — it is one de-tiling variant, not a redesign. Until it exists the
buildable batch set is {1, 8, 16, 24, 32} and there is nothing between 1 and 8.
Note also that block 4's 1.36× rests on a step time interpolated between the
measured b1 and b8 points, which is exactly where linear scaling is least
trustworthy (the mmul intrinsic only engages at batch ≥ 8). If a batch-4
dispatch in fact costs what batch 8 costs, block 4 is 0.84×, not 1.36×. The
honest range is **[0.84×, 1.36×] and it cannot be narrowed without the kernel
variant**.

### 3.2 The draft pass's bidirectional mask, on device

The verify pass needs a causal mask over the block; the DFlash **draft** pass
needs a bidirectional one — every query attends to the whole block
(`_dflash_upstream/model.py:388`, K/V is `concat(ctx, block)` with no mask).
`batch_attn_mask.py` argued in 2026 that this costs no kernel change, because a
per-query mask is nothing but a per-query VALUE of L. That was an argument from
three lines of `attn_qk.cc`. It is now measured `[hw]`.

`_tok_L` in `fused_decode.py` gave token *t* of a block `L+t` keys. It now takes
a step: `L + t·S`, with `S = 1` the causal staircase and `S = 0` giving every
token `L + B - 1` — the whole block. Two ways to set it:

- `DECODE_MASK_BIDIR=1` bakes it at build time.
- `DECODE_MASK_MODE_RTP=1` decodes it per dispatch from **bit 30 of the RTP-L
  scalar the host already writes**, so one device program serves both passes.
  Real context lengths are under 2^30 by six orders of magnitude. L keeps one
  meaning in both modes (token 0's context length), so the four other consumers
  of L — shim readback count, memtile dequeue count, core trip count, KV append
  slot — already size off `L + B - 1`, which is what every token sees
  bidirectionally. They need the bit stripped and nothing else.

**Measured, qwen3-4b, batch 8, L 128, one layer** `[hw]`, via `batch_equiv.py
--bidir`, which compares every token against a batch-1 dispatch at L+B-1 = 135:

| token | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| rms rel vs batch-1 at L 135 | 1.37e-03 | 1.37e-03 | 1.37e-03 | 1.37e-03 | 1.37e-03 | 1.37e-03 | 1.37e-03 | 1.37e-03 |

All eight agree with L=135 **and with each other**, which is the discriminating
part: under the causal mask each token sees a different context, so only token 7
would match L=135 and the other seven would not match anything. The causal build
still gates at **3.86e-03**, unchanged.

**And the RTP form is verified too, on one device program** `[hw]`. Same
xclbin (`dyn_b8_L144`, `DECODE_MASK_MODE_RTP=1`), two dispatches differing only
in bit 30 of the RTP-L scalar:

| dispatch | bit 30 | compared against | result |
|---|---|---|---|
| verify | clear | batch-1 at L 128, token 0 | **1.83e-03** |
| draft | set | batch-1 at L 135, all 8 tokens | **1.37e-03** each |

The bidirectional number is identical to the baked-in build's, so the two routes
agree. **This is stronger than it was scoped to be.** The mask was the one thing
that looked like it forced draft and verify onto separate device programs; it no
longer does, so the draft/verify boundary needs no PDI reload at all — which is
the cost §7 item 4 was worried about. Everything else that differs between the
two passes is host-side: the wave range (5 layers against 36, and the layer loop
is a rolled `scf.for` whose device is wave-invariant) and the weight BO.

**Two things had to be fixed to get there, both outside `fused_decode.py`:**

- **DYNSEQ did not build at batch > 1.** `airrt.dma_memcpy_nd` on `@inKV_K`
  failed to legalize. `DmaToNpuPattern` requires every dimension above a
  runtime-valued size to have length 1; a batched KV readback is
  `[B, ceil(L/16), 16, 512]` with strides `[0, 8192, 512, 1]`, so dim 0 is 8.
  But its stride is **zero** — a pure repeat, which contributes nothing to
  addressing and which `repeat_count` already carries, and the dynamic-length BD
  is emitted with no dimensions at all. Excluding zero-stride dimensions from
  that contiguity test is a two-line fix in `AIRRtToNpuPass.cpp` and it is what
  unblocked batch-8 DYNSEQ. This matters well beyond the mask: §7 item 3's KV
  rollback needs a runtime L.
- **The TXN builder could not run on Windows.** A DYNSEQ build assembles its
  stream from a compiled shim at DISPATCH time, and `txn_builder.py` hardcoded
  `g++`, which this host does not have — surfacing as a bare
  `FileNotFoundError: [WinError 2]` naming nothing. It now takes `AIR_TXN_CXX`
  and drives MSVC with MSVC flags, and the generated shim carries
  `__declspec(dllexport)`: `extern "C"` gives the symbol a C name but does not
  put it in a DLL's export table, so ctypes reported "function not found" on a
  library that had compiled and loaded cleanly.

### 3.3 The rest of the drafter, and why it fits the engine

§3.2 put the drafter's mask on the array. Three things still separate a drafter
layer from a target layer (`_dflash_upstream/model.py:340-397`):

```
q      = q_proj(hidden_states)                        B rows
k, v   = k/v_proj(cat[target_hidden, hidden_states])  ctx + B rows
target_hidden = hidden_norm(fc(taps))   12800 -> 2560, ONCE per block
```

Two structural claims make that fit, and `dflash_draft_decomp.py` checks both
against the real checkpoint rather than asserting them `[cpu]`:

| claim | check | result |
|---|---|---|
| `fc(cat[h1..h5])` = Σ per-tap 2560→2560 projections | vs `draft.fc` in fp32 | 3.4e-05 on max │fc│ 246 — summation-order noise |
| the context input is layer-invariant and equals `hidden_norm(fc(taps))` | `k_proj` forward hook, all 5 layers | **0.0**, exactly |

The second is the load-bearing one: `target_hidden` never flows through the
stack and never sees `input_layernorm` (model.py:446-453 hands it to
`self_attn` raw), so every layer re-projects the SAME vector with its own k/v
weights. All five layers' context K/V are therefore computable **before the
layer loop**, from the taps alone. The check hooks `k_proj` — each layer calls
it twice, first on the context then on its own hidden states — and confirms the
first call's input is byte-identical across layers while the second changes at
every one of the 4 layer steps, so it is not passing vacuously.

**The position convention falls out consistent with §3.2**, which is worth
stating because it was chosen for a different reason. RoPE gives `k` the full
`ctx+B` positions and `q` only the last B (`model.py:334`), so the drafter is a
plain rotation over `ctx+B` contiguous positions with only the last B as
queries. Setting the RTP-L to `ctx+1` with the mode bit then does both jobs at
once: every token sees `L+B-1 = ctx+B` keys, and the append slot `(L-1)+t`
lands the block's own K/V at `ctx+t`, immediately after the context.

**`fc` needs a phase, not a kernel.** The engine's phase dimensions are
`J2P = K/(2·COL_BLOCK)` and `I2P = out/(ROW_BLOCK·NCX·NCY·PAIR_ROWS)`; qwen3-4b
has `2·COL_BLOCK = 512`, giving `J2P=[5,8,5,19]` for inputs `[2560, 4096, 2560,
INTER]`. So `fc` at 12800 → 2560 is `I2P=5, J2P=25` — a well-formed entry in the
existing scheme, since accumulating across input column-blocks is exactly what
`J` already does. The 5-way split above is the reason the tap buffer can be fed
as one 12800-wide input; it does not need to appear in the IR.

**The drafter's weight bundle is built** `[static]`. `qwen3_4b_draft_weights.py`
reads the checkpoint's bf16 `model.safetensors` behind the same accessor surface
the target's packer already consumes, and `qwen3_4b_draft_requant.py` packs it:
5 layers × 31,539,200 elements plus 10 tied vocab slabs = 280,576,000, fc, and
the norms. Two things the target has no analogue for — `fc` (2560×12800) and
`hidden_norm` — and one it cannot supply: the drafter carries **no embedding
table at all**, its head being tied to the target's, so the target bundle is a
required argument rather than a default. Its per-layer shapes are the target's
exactly (q 4096×2560, k/v 1024×2560, o 2560×4096, gate/up 9728×2560, down
2560×9728), which is why `qwen3-4b-draft` is the target's geometry with
`UNI_DEC=5` and nothing else changed. The fc requant round-trips at 4.5e-02
relative against a quantization step of 9.2e-02.

**`fc` cannot be a 5th decode phase.** Everything else about the phase machinery
is table-driven off `I2P`/`J2P`/`DEST`, so a 5th entry looked like a config
change. It is not: `FULL4` (`fused_decode.py:1180`) is
`NPH == 4 and DOWN_PHASE == 3 and DEST[1] == DEST[3] and NDEST == 3`, and it
gates the whole fused four-phase structure — including the `RMS_BAND_STREAM`
level 3 path §3.2 depends on. Setting `NPH = 5` silently switches the design to
a different, far less exercised shape. So fc takes **its own launch**, which is
also what the multi-launch route wants, and a standalone launch is served by the
bf16 GEMM builder in `llms/shared/builders` rather than by the q4k cascade. At
ctx ≤ 8 the shape is a thin `[8, 12800] × [12800, 2560]`; 65 MB of bf16 against
the drafter's ~300 MB of Q4 layers is worth not introducing a quantized shape on
a path nothing else uses.

**The fc launch runs on NPU2** `[hw]` — the first piece of the DRAFTER, as
opposed to the target, to run on the array. Two `air.launch` ops in one
`func.func` (a 32×12800×2560 GEMM then the RMSNorm), built by
`dflash_fc_builder.py`, gated by `dflash_fc_gate.py` against the **real**
`fc.weight` and `hidden_norm.weight` — not random fill, because a transposed or
mis-strided weight still correlates well on random data whose rows are
exchangeable:

| | rows 0-7 |
|---|---|
| rms rel vs f32 numpy | 1.13e-02 – 1.24e-02 |
| correlation | 0.99993 – 0.99995 |
| padded rows 8-31 | 0 non-zero elements |

1.2e-02 is bf16 through a K=12800 reduction; the registry quotes 9.3e-3 for its
own high-precision tier at smaller K. The padded-row check is there because
nothing else would catch a GEMM writing outside the rows it was given.

Four things had to be right, and each failed first in a way that named nothing:

- **ELF, not xclbin.** Multi-launch is the ELF path — that is what emits
  `load_pdi` between launches. On the xclbin path the two launches' instruction
  streams collide, reported as `edge 'air.insts.bin' produced duplicate output
  path` and then a bare `pipeline failed` several stages later, with every
  intermediate `.ll` compiling cleanly by hand.
- **GEMM, not GEMV.** fc applies one 2560×12800 weight to every context row. A
  GEMV re-streams it per row: 8 × 65 MB per draft call, more traffic than the
  drafter's whole 5 layers. The M=32 padding costs arithmetic that is free next
  to the weight stream.
- **The herd and tiles are not free.** `tile_m` is forced to 32 by the drain
  method, `mm_aie2p.cc` additionally static-asserts `DIM_M % (2·r) == 0`, and
  `M % (tile_m·herd_m) == 0` then forces a 1×4 herd with M a multiple of 32.
  The registry's own 320×128 staging is 80 KB and does not fit a 64 KB tile.
- **A stale `mm_m32.o`** built at different tile parameters sat in
  `build_peano/` and was picked up ahead of the fresh one.

**The context K/V runs on NPU2 too** `[hw]`. 15 `air.launch` ops in one func —
per drafter layer a K GEMM, a V GEMM and `k_norm` — checked against the real
per-layer `k_proj`/`v_proj`/`k_norm`:

| layer | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| k, pre-norm | 9.51e-03 | 9.71e-03 | 9.59e-03 | 9.76e-03 | 9.61e-03 |
| v | 9.62e-03 | 9.71e-03 | 9.73e-03 | 9.68e-03 | 9.73e-03 |
| k, post-`k_norm` | 1.21e-02 | 1.20e-02 | 1.18e-02 | 1.20e-02 | 1.18e-02 |

A fused `[k|v]` form (5 launches instead of 15) also passes at 9.6e-03 and is
kept, but it cannot feed `k_norm`: the norm is over `head_dim`, i.e. over K
viewed as `[ctx·8, 128]`, and in a fused `[ctx, 2048]` output the K half is
strided rather than contiguous.

Three things this cost, each worth keeping:

- **`k_norm` is NOT shared between layers.** The first version passed layer 0's
  weight to all five, and **every layer still gated clean** — because the
  reference made the same assumption. What caught it was comparing the five
  checkpoint tensors to each other directly. The gate now does that, and each
  launch gets its own weight.
- **The pre-norm K is checked as well as the post-norm K.** RMSNorm is
  scale-invariant in its input, so a K wrong by a per-row scale — exactly what a
  mis-strided reshape gives — comes out of the norm looking right.
- **`memref.reinterpret_cast`, not `collapse_shape` + `expand_shape`.**
  `aie.dma_bd` accepts a buffer rooted at a subview/view/cast/reinterpret_cast
  chain and rejects anything else, so the reshape pair lowers cleanly until DMA
  lowering and then fails.

The cross-layer check on the fused form is the other guard worth naming: every
launch takes the same input and differs only in its weight, so a mis-wired arg
map yields a plausible result belonging to another layer. The distance matrix is
diagonal by two orders of magnitude (9.7e-03 against 1.2–1.8).

### 3.4 int4 for the two non-decode projections, and a GEMM bug it found

bf16 costs 65 MB for `fc` and 52 MB for the context k/v against a draft pass
whose 5 Q4 decode layers are ~315 MB — a **37% surcharge**, and the k/v half is
worse than it looks because those are the same tensors the decode already
streams in Q4 in the same call. `fc` cannot join the superkernel's own q4k
cascade (`FULL4` is `NPH == 4 and ...`, and `GLU_PHASE = 2 if NPH == 4 else -1`,
so a fifth phase disables both the residual and the GLU paths), so the int4-AWQ
GEMM is the available quantized route. Quantized, **fc is 65.5 MB → 17.2 MB**,
round-tripping at 5.5e-02 against a 1.1e-01 step `[cpu]`.

Getting it to run turned up three defects. The last is a real constraint on
`matmul_int4_packed` that its own tests cannot see, and it is what decides the
shape of everything below.

**`compile_mv_int4_bf16` builds the GEMV, not the GEMM** `[static]`. It passes
`-DDIM_K` and **no `-DDIM_N` at all**, where the int4 GEMM's own Makefile passes
`-DDIM_N` and `-DDIM_K_CHUNK`. The GEMV object links, loads and runs inside the
GEMM — and returns NaNs and uncorrelated rows, because the kernel's tile
constants disagree with the IR's.

**The default stack silently computes the wrong answer** `[hw]`. The reference
passes `stack_size=16384`; without it the same module runs to completion and
returns rel ≈ 0.85 instead of 7.4e-03. No crash, no diagnostic.

**And the int4 GEMM is only correct when `tile_k_l2 == K`** `[hw]` — i.e. one
K-outer iteration. Bisected on device, M=64 N=128 herd 2×4:

| K | tile_k_l2 | K-outer | rel |
|---|---|---|---|
| 128 | 128 | 1 | 7.4e-03 |
| 256 | 256 | 1 | 7.2e-03 |
| 1280 | 1280 | 1 | 7.3e-03 |
| 256 | 128 | **2** | **1.12** |
| 1280 | 128 | **10** | **1.03** |

M and N are innocent: N=2560 and M=32 both pass at K=128. The accumulation
across L2 K-tiles is what breaks, and every lit test in
`matrix_multiplication/int4_awq` uses K=128 with `tile_k_l2=128`, so K-outer is
never exercised. (Their Makefile also cannot build the kernel in this
environment — `PEANOWRAP2P_FLAGS` is missing the `aie_api` include path — which
is why the bug had to be found through `_compile_kernel`.)

Reading the builder says why: `matmul_int4_packed.build_module` puts the whole
herd **inside** the K-outer loop, and the herd body allocates its L1 f32
accumulator, zeroes it, converts and drains to L2 on every iteration. The
accumulator cannot survive a K-outer step, so the L2 C tile ends up holding the
last chunk's partial product rather than the sum.

That blocks int4 `fc` at K=12800 directly: L2 needs `tile_k_l2 < K` to fit
(at `tile_k_l2 = K` the A and B stages are 400 KB + 429 KB against 512 KB), and
`tile_k_l2 < K` is the broken case. **The way out is the decomposition §3.3
already verified** — `fc` over a concatenation is a sum of per-chunk GEMMs, so K
can be split across launches and the partials added. Any partition works, not
just the tap boundary; **K=6400** stages 200 KB + 209 KB and needs only one add.

**A herd tile takes at most two incoming L3 streams** `[hw]`, and this shapes
every launch here. Bisected on the sum/norm herd, counting L3 operands as
(inputs + weight):

| L3 operands | outcome |
|---|---|
| 2 in, 1 out | correct — rel 0.0 for a copy, 2.6e-03 for a sum |
| 3 in, 1 out | **silently wrong** — rel 1.12–1.17, padded rows fill with other rows' data, and the value **varies run to run**, so it is a race rather than a miscompile |
| 5 in, 1 out | `aircc` dies with `0xC0000005` and no diagnostic |

A single AIE tile has two S2MM channels, which is exactly where the first
boundary sits. The same class of hazard shows up **within** a launch: draining
an L1 accumulator straight to L3 lets the next loop iteration's input DMA
overwrite it mid-flight — a WAR the dependency pass does not order. The symptom
is a run-to-run-varying output whose rows are permutations of *other* rows'
data (measured 0.85 / 0.88 / 0.94 / 1.04 on repeated runs of the same pure-copy
module). Draining a separate L1 buffer fixes it.

**Both projections now run in int4 on device** `[hw]`, against the real
z-lab/Qwen3-4B-DFlash-b16 weights, with the dequantized weight as the reference
so the number measures the engine and not the quantizer:

| | launches | engine rel | end to end |
|---|---|---|---|
| `fc` + `hidden_norm` | 2 GEMM (K=6400) + add + norm = **4** | 7.09e-03, 7.29e-03 | **1.10e-02** |
| context K/V + `k_norm` + RoPE | 5 × (2 GEMM + norm + RoPE) = **20** | k 7.0–7.3e-03, v 7.2e-03 | k_norm 9.8e-03–1.01e-02, RoPE **1.15–1.18e-02** |

The bf16 `fc` gated at 1.13–1.24e-02, so quantizing costs nothing measurable
downstream of the norm. Cross-pair and cross-layer distance matrices are
diagonal by two orders of magnitude in both, and the RoPE rows are also compared
against the **unrotated** `k_norm` output (0.65–0.76 away), so a LUT stuck at
position 0 or one that ignores the 8 KV heads per position cannot read as a
pass. Positions are absolute — the drafter is called with
`position_ids[start - ctx_len : start + block]` — and the gate defaults to
`start=137` rather than 0 for that reason.

**The draft pass's non-decode traffic drops from 117.9 MB to 30.9 MB**, i.e.
from a 37% surcharge on the drafter's ~315 MB of Q4 decode layers to **9.8%**.

**And the two halves now run as one dispatch** `[hw]` —
`dflash_draft_prepass.py`, **24 launches in one `func.func`**, taps in and the
five layers' RoPE'd context K/V out:

```
  target_hidden: 1.10e-02, padded spill 0
  layer 0: k 5.32e-03, v 5.36e-03, k_ctx 1.07e-02 (end to end 1.44e-02), spill 0
  ...      k 4.1-5.3e-03  v 4.7-6.1e-03  k_ctx 1.00-1.07e-02  e2e 1.32-1.44e-02
```

`target_hidden` is an intermediate no host code sees, so the K/V numbers are
referenced against the **device's own** `target_hidden` — that isolates the
wiring — with the fully-host-computed end-to-end number printed beside it. The
cross-layer matrix stays diagonal by two orders of magnitude after the merge.

**What is left:**

1. ~~The fc launch.~~ **Done**, bf16 and int4.
2. ~~The context-K/V projection, `k_norm` and RoPE.~~ **Done.** None of it can
   reuse a plain decode pass, because the engine RMS-norms X before projecting
   and the context must reach `k/v_proj` raw. Pre-compensating the input to
   cancel the norm does not work — RMSNorm is not invertible that way, and the
   residual scale would land in `v_ctx`, where `k_norm`'s scale-invariance
   cannot absorb it.
3. ~~Stitching those into one dispatch.~~ **Done** — 24 launches, one func.
4. Seeding the drafter's KV cache from this pass's output, and the host loop.

### 3.5 The verify pass, on the shipping driver

§5.3b established that the batched *template* is correct: `batch_equiv.py` runs
it on synthetic q4k weights and compares layer outputs (3.86e-03 at one layer,
5.56e-03 at 36). That is not the same as a verify pass. The verify pass is the
shipping **driver** handing the target B draft tokens at B consecutive positions
and taking B next-token distributions back, and it exercises everything
`batch_equiv.py` cannot see: the batched X buffer, the B rope slabs, the batched
logit readback, and the ATTN_MAXL window the batch moves.

`FusedDecoder` now takes `batch=`, and `dflash_verify_gate.py` checks it against
the only thing it has to equal — **eight sequential batch-1 dispatches of the
same tokens from the same KV seed, on the real model** `[hw]`:

```
  batch-1 : [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13]
  batch-8 :  one dispatch, argmax = at all 8 positions
             margins 1.75 - 12.75, corr 0.977 - 0.993, top5 3-5/5
```

**One defect, and it was invisible in the sizes.** The rope region is
LAYER-major then token — layer L's block at `rms_lut_off + L*B*ROPE_W_LEN`,
token t at `+ t*ROPE_W_LEN` inside it — which is the *transpose* of B copies of
the batch-1 slab. Writing it the other way gives the right region size and the
right element count, and returns `[13, 1096, 279, 315, 279, 30, 279, 30]`: token
0 correct, because its slab lands first either way, and every later token
rotated by another layer's angles.

**The logits agree loosely and that is the projection kernel, not the batching.**
RMS-relative logit error is 6.5e-02–2.0e-01 with the argmax unchanged. The
batch-1 template runs the v1 GEMV and the batched one the q4k mmul — different
kernels, different accumulation orders, which `proj_qmm_gate.py` already
measures at 1.4× the GEMV's error. **The toolchain was ruled out by measurement,
not by argument**: the same batch-1 design built by `build_template.sh` (which
skips the Peano pin preflight — and this sandbox's nightly index no longer
carries the pinned build) is **bit-identical** to the shipping Makefile-built
pair, all 8 tokens, rel 0.0. So the gate is argmax agreement outside a near-tie,
a correlation floor, and `rel` reported rather than targeted.

Two host-side traps, both of which present as a device fault:

- **Dropping a `FusedDecoder` segfaults the process.** Its BOs and the XRT
  device go down in whatever order the collector picks, and the symptom is a
  bare SIGSEGV after the last flushed line — which reads exactly like a fault in
  the dispatch that just succeeded. Keep both decoders alive.
- **`DECODE_STACK` must match the build.** At batch 8 it is not optional: the
  default stack leaves the rms core 55280 B of L1 against the 59424 B a batch-8
  residual plus staging plus norm weights need, and the builder refuses to
  import rather than build something that fits by truncation. `FusedDecoder`
  takes `env_extra` for exactly this.

Batched templates now build from the model's own Makefile
(`make compile-decode-batch DBATCH=8 LBUILD=16`), which carries the
`-DPROJ_MM_BATCH` the batched projection needs and restores the batch-1 objects
afterwards — a batched `proj_qmm.o` left behind is linked into the next batch-1
build.

### 3.6 The pre-pass against the real drafter, not against itself

§3.4's gates check the pre-pass against a numpy chain built from the same
weights. That proves the *engine* and it cannot prove the chain is the
drafter's: the tap concatenation order, that `fc` sees the taps raw, that
`hidden_norm` sits between `fc` and `k/v_proj`, that `k_norm` precedes RoPE, and
that the positions are absolute are all assumptions the reference shares.

`dflash_draft_oracle.py` runs **z-lab/Qwen3-4B-DFlash-b16 itself** — the model
code, via `dflash_phase2_replay.py`'s crop cache, not a reimplementation — over
the recorded NPU target state, and dumps one block: taps, `target_hidden`, and
the per-layer K/V rows **as the model's own cache holds them**. It has to be a
dumper rather than a call, because torch and XRT segfault in one process.
`dflash_prepass_oracle_gate.py` then replays that npz through the device
`[hw]`, block 0 of the Paris state (ctx = the whole 5-token prompt, positions
0–4):

```
  target_hidden: int4 vs oracle 1.22e-01, engine vs dq 1.12e-02, cos 0.9925
  layer 0: k int4 9.2e-02 dq 9.9e-03 cos 0.9957 | v int4 1.28e-01 dq 6.3e-03 cos 0.9919
  ...      k 8.4-9.6e-02  dq 9.8-10.9e-03  cos 0.9953-0.9965
           v 1.20-1.30e-01 dq  5.6-6.3e-03  cos 0.9915-0.9929
```

**The structure is right and the quantization is the whole gap.** `dq` — the
device against the dequantized weights fed its *own* `target_hidden` — is
5.6e-03 to 1.1e-02, the same scale §3.4 measured in isolation. The `int4`
column is the AWQ round-trip, which `dflash_int4.self_check` puts at 5.5e-02
against a 1.1e-01 step.

**And that cost is now a named risk on the thing the whole idea rests on.**
Cosine 0.991–0.997 against the real drafter is not free: §3.1's 1.24× priced a
**bf16** drafter's acceptance distribution, and a quantized drafter proposes
slightly different tokens. Nothing here measures how much acceptance that costs
— it needs the acceptance sweep re-run with the device drafter in the loop, and
until then the 1.24× is an upper bound rather than a prediction.

One methodological note, because it cost a rerun: referencing the layer K/V
against the *oracle's* `target_hidden` folds `fc`'s quantization into every
layer a second time and turns a 1e-02 engine number into 7e-02. Reference the
engine against what the engine was actually given.

### 3.7 The whole draft pass, on the array

`dflash_draft_gate.py` runs a block end to end on NPU2 `[hw]` — taps through
the 24-launch pre-pass, its output seeded into the drafter's KV cache, then a
**bidirectional 5-layer decode at batch 8** — against the same oracle block.
This is the first point at which the drafter half of DFlash exists on the array
rather than as pieces.

The drafter reuses the target's own `FusedDecoder`: `qwen3-4b-draft` is
qwen3-4b's per-layer geometry with `UNI_DEC=5`, so once `decode_model`,
`weights`, `npz` and `artifact_dir` are arguments nothing in that driver is
target-specific. Its templates carry their own `draft_b8_L<N>` prefix — a
different xclbin for a different model at the same context length would
otherwise be picked up by whichever scan ran last. RTP-L is **ctx+1**, which is
one value doing both jobs: with the bidirectional bit every token sees ctx+B
keys, and the append slot `(L-1)+t` puts token *t*'s K/V at `ctx+t`.

```
  seed layers 0-4      k cos 0.9953-0.9965   v cos 0.9915-0.9929
  block K/V layer 0    k cos 0.999555        v cos 0.996433     [structural]
  block K/V layers 1-4 k 0.9886 0.9834 0.9609 0.9738  v 0.9774 0.9595 0.9288 0.9284
  draft tokens matching the bf16 drafter: 4/7
```

**Layer 0's block K/V is the structural gate and it is clean.** It is a function
of the mask-token embedding, `input_layernorm`, the batch-8 k/v projection,
`k_norm`, the RoPE angles at positions ctx..ctx+B-1 and the append slot — and of
nothing that has been through a quantized layer yet. A wrong slot or a wrong
block position shows up there and nowhere else, because five layers of attention
have not mixed it yet. Reading it needs the KV BO back off the device;
`read_block_kv` inverts `seed_kv`'s region-major layout.

**Layers 1-4 fall monotonically, and `--seed-oracle` says whose fault that is.**
Re-seeding with the oracle's bf16 context rows instead of the int4 pre-pass
lifts depth-4 K from 0.974 to 0.985 and V from 0.928 to 0.961 — so roughly half
the drift is the pre-pass and half is the decode layers' own q4k. **The draft
tokens are identical either way**, all seven of them. Quantizing the pre-pass to
int4 is therefore not what moves the draft; the decode layers are.

**4 of 7 draft tokens match the bf16 drafter, and that is the open number.**
§3.1's 1.24× priced a bf16 drafter's acceptance distribution, and a quantized
drafter proposes different tokens. One block of one prompt says nothing about
the distribution — block 0 here is also the degenerate case, where the context
is the whole prompt rather than the previous round's `produced`, and the bf16
drafter itself only produced 1 of 8. What this needs is the §3.1 sweep re-run
with the device drafter in the loop. Until then 1.24× is an upper bound.

### 3.8 The loop, and the engine defect it exposed

`dflash_loop.py` closes it: per block, one batch-8 **taps** verify dispatch
(`taps_b8_L<N>`, `DECODE_HIDDEN_TAPS=1`, so the same dispatch returns both the
distributions that decide acceptance and the five tap slots the drafter needs),
the 24-launch pre-pass over the positions that verify just committed, a KV seed
or append into the drafter, one bidirectional draft dispatch, and greedy
accept. Neither cache needs an explicit rollback: both engines append in place
and the next dispatch's own writes cover every rejected slot before anything
reads it. Three device programs do not fit on the array at once, so the
pre-pass ELF is loaded and unloaded around each block — a correctness vehicle,
not a shipping shape, and `dflash_loop.py` reports that reload separately for
exactly that reason.

It runs `[hw]`. With the drafter switched off (`--no-spec`) it reproduces
`PARIS_GREEDY` **10/10**, which is the mechanical gate on the taps template,
the accept path and the tap indexing (slot 0 of the X buffer comes back
bit-identical to the embeddings the host wrote, which is the one check that
pins both indices of `(slot, token)`). With the drafter on it committed 32
tokens over 21 blocks at a **mean of 1.476 tokens per verify dispatch**.

**That number is not yet a measurement of DFlash, because the verify pass it
rests on is not causal.** Chasing why the speculative stream diverged from the
non-speculative one at a wide margin found a defect in the decode engine
itself, present in the shipping batch-1 path:

> **A decode step at context length L depends on KV rows `L..ceil(L/8)*8-1`.**

`dflash_causal_probe.py` is the minimal repro: poison KV rows `L..L+7`, which
the mask must exclude, and re-dispatch the same token from the same state. At
batch 1 on the shipping template the logits move at every L except `L % 8 == 0`
(max|Δ| 0.48–1.25, corr 0.994–0.999), and at `L % 8 == 0` they are
bit-identical. It is a **V-side** defect, not a masking one: poisoning only K
changes nothing at all — `attn_qk`'s `aie::le(idx, rem)` mask is exact per key —
poisoning only V changes the output, and the reach stops exactly at
`ceil(L/8)*8`, even though the whole 16-key block containing those rows is
streamed to the core.

It has never mattered because in ordinary decode every row past `L-1` is zero,
so the leak is a small fixed pull that no gate resolves — `batch_equiv.py`'s
own regression point is L=128, which is `L % 8 == 0` and therefore exactly in
the clean class. It matters here because in a verify pass those rows are **the
block's own later tokens**, written by the same dispatch. Measured at batch 8
with the taps build, holding slot 0's token fixed and changing only the tail:
slot 0's hidden state diverges from **layer 1** onward at every `L % 8 != 0`
and is bit-identical at every `L % 8 == 0` — 51 consecutive L, no exceptions —
and by the 36th layer max|Δ| reaches 208. Over 29 consecutive positions the
tail moved slot 0's **argmax** at 2 of them (7%), which is the rate at which an
accept/reject decision is currently made on a corrupted logit.

**It is not extra attention, and that took a second instrument to establish.**
`dflash_attn_leak_probe.py` measures the WEIGHT those rows carry. Seed every
position with the same key `k0` so the softmax is uniform whatever *q* is —
the attention output is then the plain mean of the attended V — build ONE layer
with `DECODE_ACC_STOP=2` so the layer output is `x + o_proj(attn)` and the
readback is linear in it, and vary only the phantom rows' V. (`DECODE_PROBE=2`,
the o-gather memtile tap, does not route in this configuration — the dispatch
times out. `ACC_STOP` changes only what a buffer holds, so there is nothing new
to route, which is what the engine's own notes say it is for.) Then:

| | |
|---|---|
| `f`, the leaked fraction of the attention output | **−0.0063**, ~0.6% |
| at L = 16, 24, 32 | **+0.000000** exactly — three controls |
| across L = 9…34 | constant; independent of L, of R, and of how many phantom rows there are |
| vs the phantom V value | linear (same `f` at V=1 and V=16) |
| one phantom row set vs all five | **the identical difference vector**, cosine 1.0000 |
| rows at or past `ceil(L/8)*8` | exactly 0 |

The last two lines refute the two mechanisms that survived the poison test —
"the softmax includes the phantom keys" and "`attn_fv` pairs valid scores with
the wrong V rows" — because both predict a per-row additive weight, and the
measured contribution does not depend on the row count at all. (The write was
verified by reading the KV cache back: only the intended row is set.) Combined
with the K/V asymmetry, what is left is a **descriptor** asymmetry in the KV
readback for a partial last group, not kernel arithmetic. The next instrument
is `shim_volume.py` / `shim_schedule.py`, which print what each channel really
moves after lowering.

The severity scales accordingly: 0.6% of the attention weight, not the ~25% a
genuine extra key would carry — which is why decode looks fine (corr 0.997) and
why it still compounds to an argmax flip at 7% of positions across 36 layers.

Nothing on the host can work around it. Token *t* of a block has `L_t = L_0+t`,
so exactly one of any eight consecutive tokens lands in the clean class, and
the rows in question are written by the dispatch that reads them.

### 3.9 The cause: a BFP16 shared exponent that spans the context boundary

**It is not extra attention at all. It is precision.**

`S·V` runs as a **BFP16** mmul contracting over KEYS, and BFP16 shares ONE
exponent across 8 elements of that dimension. Those groups are 8-key aligned,
so whenever `L % 8 != 0` the group straddling the context boundary also covers
KV rows past `L-1` — and their exponents still enter the shared-exponent max,
right-shifting the *valid* rows' mantissas. The masked keys contribute no
**weight**; what they cost is **bits**.

Every measurement falls out of that, and each was made before the mechanism was
named (`dflash_attn_leak_probe.py`, batch 1, one layer, `DECODE_ACC_STOP=2` so
the readback is linear in the attention output, every key seeded identical so
the softmax is uniform whatever *q* is):

| observation | what it means |
|---|---|
| sign ignored — `v` and `−v` byte-identical, even for a random per-dim vector | not an arithmetic contribution |
| mantissa ignored — 1.00 / 1.25 / 1.50 / 1.75 byte-identical, likewise 2.0…3.5 | **only the exponent is read** |
| `‖d‖` doubles per power-of-two step | it *is* the exponent |
| any one row = all five rows, identical vector, cos 1.0000 | a **max**, not a sum |
| valid V at 256 vs phantom 1 / 16 / 256 → **exactly 0**; only 4096 shows | the max is **shared**, and only an exponent that *exceeds* the valid rows' does anything |
| K poison does nothing, ever | QK contracts over DH (all valid), and masked scores are exactly `0.0`, whose exponent never raises a max |
| bit-identical at `L % 8 == 0` | the group boundary coincides with the context boundary |

**The fix is four lines in `attn_kv_blk`**: on the last block, zero the V tile's
keys from `rem` to the end before the mmul. Their scores are already zero, so it
changes no arithmetic — it only keeps their exponents out of the shared max.
Only `attn_kv_blk` / `attn_kv_fin` / `attn_kv_fin_row` are reachable from AIR;
the in-kernel-lock `attn_kv` is the reference form and is left alone.

Verified `[hw]`:

- `dflash_attn_leak_probe.py`: `f = +0.000000`, off-direction residual 0.000, at
  **every** L from 9 to 26 (was −0.0063 at every `L % 8 != 0`).
- `dflash_causal_probe.py` on the real 36-layer batch-1 template: **PASS**, K and
  V both, every L from 6 to 24 — max|Δ| exactly 0.00000, corr 1.000000.
- **Strictly no-op in ordinary decode**, which is the claim the mechanism makes
  and it was checked rather than assumed: pre-fix and post-fix 36-layer batch-1
  builds are bit-identical over 20 decode steps, max|Δlogit| **0.0**, same
  tokens, PARIS_GREEDY intact. Rows past `L-1` are zero there, and a zero
  exponent never wins a max.
- **The loop is now exact.** With a causal verify pass the speculative stream
  reproduces the non-speculative one on all 32 tokens; before the fix they
  diverged at position 19 at a top-2 margin of 2.19, which is the divergence
  that started this. Acceptance is **1.409** tokens per verify dispatch — now
  measured on a sound pass, but on the Paris prompt, which degenerates into
  *"of the of the"* by token 12 under plain greedy and gives a drafter nothing
  real to predict. The sweep that prices DFlash has to run on §3.1's math/code
  prompts.

### 3.10 The batch-8 verify pass only works below a 16-token context

Running the acceptance sweep on real prompts (`dflash_acceptance_device.py`, 12
gsm8k prompts drawn through the upstream's own loader and formatted exactly as
§3.1 formats them) returned **1.095** tokens per verify dispatch — 92% of blocks
committing nothing but the bonus token, which prices at **0.29×**. Against
§3.1's bf16 measurement on the same data (**4.73**, P(slot 1 accepted) 0.87
against 0.079) that gap is far too large to be quantization, and it is not the
drafter.

**The target's own batch-8 output is garbage at that context length.** On gsm8k
prompt 0 (96 tokens) batch 1 decodes coherently — *"We are given the following
… measurements: - First measurement: **47 kg** …"* — while batch 8 emits
`[1654, 686, 387, 2952, 311, 387, 220, 198, 198, 198, …]`, eleven newlines.
They agree on the first token and diverge immediately.

`dflash_verify_ctx_sweep.py` puts a boundary on it — the same comparison
§3.5 makes, swept over context length instead of run once at P=5:

| P | attention blocks | agree/8 | corr(slot 0) | worst-slot corr |
|---|---|---|---|---|
| 5 | 1 | **8/8** | 0.99559 | 0.97126 |
| 8 | 1 | 7/8 | 0.99478 | 0.99061 |
| 12 | 2 | 4/8 | 0.99504 | 0.22384 |
| 16 | 2 | 5/8 | 0.30692 | 0.30692 |
| 20 | 2 | 1/8 | 0.48705 | 0.38601 |
| 96 | 7 | 1/8 | 0.32474 | 0.07635 |

The break is exactly where the block stops fitting in **one 16-key attention
block**. Batch 1 at the same P is fine — it decodes coherently at P=96 across
seven blocks — so this is batch>1 **and** rounds>1, not either alone.

**§3.5's verify gate ran at P=5, which is the only regime that works.** That is
the lesson worth keeping: it compared batch 8 against eight batch-1 steps, on
the real model, and passed 8/8 — and it was a single-point test sitting inside
the one window where the pass is correct. The context sweep is now the gate.

`--per-token` narrows it further, and the boundary turns out to be per token,
not per dispatch. Every token whose own `ceil(L_t/16)` is 1 is correct; every
token that spans two blocks is wrong, in the same dispatch:

| P | token | L_t | own blocks | blocks pushed | corr |
|---|---|---|---|---|---|
| 9 | 6 | 16 | 1 | 2 | 0.98933 |
| 9 | 7 | 17 | **2** | 2 | 0.25554 |
| 15 | 0 | 16 | 1 | 2 | 0.98774 |
| 15 | 1 | 17 | **2** | 2 | 0.08701 |

Two things fall out. The tokens that **skip** a block — own count 1 against a
pushed count of 2, the `rem <= 0` early return — are the correct ones, so the
skip path is not implicated. And a single block is correct even when a later
block is pushed behind it, so neither the push count nor the KV data is wrong.
What fails is only the accumulation from one block to the next, and only at
batch > 1.

### 3.11 The cause: a shared-L1 lock taken one loop level too high

`AIRToAIEPass.cpp`'s `allocateSharedL1BufferLocks` chose the **outermost**
`scf.for` containing every access to a shared L1 buffer:

```cpp
// Find the OUTERMOST scf.for that contains ALL accessing ops
while (candidate && candidate != coreOp.getOperation()) {
  if (isa<scf::ForOp>(candidate)) { ... if (containsAll) info.lockScope = candidate; }
  candidate = candidate->getParentOp();          // keeps overwriting
}
```

The locks bracket that loop's body, so the buffer is handed over **once per
iteration of whatever loop is chosen**. The attention score buffer is written by
`attn_qk_blk_row` on one core and read by `attn_kv_blk` on its neighbour, and at
`DECODE_BATCH=8` both sit inside `for token { for block { … } }`. Both loops
contain every access, so the scope came out as the **token** loop: one handover
per token, while the cores exchange one buffer per **block**. The producer wrote
the buffer `ceil(L_t/16)` times before the consumer read it once, and every
block of a token was multiplied against the last block's scores.

At batch 1 the same code has a single `for block` loop, so outermost and
innermost coincide and the handover is per block — which is why this survived
every batch-1 gate, including `make verify` at real prompt lengths. At batch 8
with one block per token there is exactly one write per acquire, which is why it
also survived `dflash_verify_gate.py` at P=5. It is a lost update, not a stall,
so nothing hangs and nothing warns.

The fix picks the **deepest loop every participating core can reach, counted
from its outermost**. Counting from the outermost rather than simply taking each
core's innermost is what keeps the cadence equal across cores: a core nested one
level deeper must not cycle the lock more often, or the two deadlock. When the
cores disagree completely the choice is unchanged, so this only tightens designs
where the nesting already matched.

`decode_b1_L16.insts.bin` is **bit-identical** across the change `[hw]`, which
is the property that matters — the shipping batch-1 path is untouched.

Batch 8 after the fix, same sweep, same prompts:

| P | blocks | agree/8 | corr(slot 0) | worst-slot corr | was (worst) |
|---|---|---|---|---|---|
| 8 | 1 | 6/8 | 0.98693 | 0.94432 | 0.97027 |
| 16 | 2 | **8/8** | 0.98565 | 0.95742 | 0.30692 |
| 20 | 2 | **8/8** | 0.98802 | 0.95238 | 0.38601 |
| 32 | 3 | 4/8 | 0.97896 | 0.93574 | — |
| 64 | 5 | 7/8 | 0.98995 | 0.97514 | — |
| 96 | 7 | **8/8** | 0.98857 | 0.96504 | 0.07635 |

The collapse is gone: every token at every context now correlates 0.93–0.99,
where before the multi-block tokens fell to 0.08–0.5. What is left is the
ordinary batch-vs-batch-1 disagreement — a batch-8 pass and eight batch-1 passes
take different reduction orders, and §3.1's tie analysis already says argmax
flips wherever the top two logits are close. The single-block P=8 row sits at the
same 0.944, so it is not a residue of this bug. The 0.95 threshold in the sweep
is tuned for the failure mode it was written to catch and is now too tight to be
a pass/fail line on its own.

### 3.12 Acceptance with the device drafter, on a sound verify pass

This is the number the whole document has been trying to get to, and it is now
measured rather than bounded. `dflash_acceptance_device.py`, the same 12 gsm8k
prompts §3.1 used, 32 tokens each, **106 verify dispatches** `[hw]`:

| accepted per dispatch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| blocks | 17 | 23 | 14 | 12 | 11 | 5 | 8 | 16 |
| share | 16.0% | 21.7% | 13.2% | 11.3% | 10.4% | 4.7% | 7.5% | **15.1%** |

**Mean 3.981 tokens per verify dispatch**, against **1.095** through the broken
pass. The full block commits 15% of the time. §3.1 measured **4.73** on the same
prompts with a *bf16* drafter, so **quantizing the drafter to q4k/int4 costs
about 16% of the acceptance rate** — the line the status table has carried as
"not measured" since §3.6.

The token stream is coherent now, which is the qualitative half of the same
result. On gsm8k prompt 0 the loop emits the same structured arithmetic the
batch-1 decode does, where before the fix it emitted eleven newlines.

**It still does not pay for itself at the measured step cost.** Per block on
that prompt, over 8 blocks `[hw]`:

| | ms/block |
|---|---|
| verify (36 layers, batch 8) | 198.8 |
| draft (5 layers, batch 8) | 47.5 |
| pre-pass (24 launches) | 82.0 |
| **step** | **328.3** |
| pre-pass ELF load/unload | 45 — *not a shipping cost* (§3.8) |

**§6's 217.92 ms step prices draft + verify and omits the pre-pass entirely**,
which is 25% of a real step. Against §6's 56.95 ms baseline token:

| priced with | break-even | at 3.981 accepted |
|---|---|---|
| §6's 217.92 ms (no pre-pass) | 3.83 | 1.04× |
| the measured 328.3 ms | 5.76 | **0.69×** |
| 246.3 ms — pre-pass fully hidden behind the target | 4.33 | 0.92× |

So the ordering of the work is now clear: acceptance is no longer the problem,
the pre-pass is. Even hiding it completely leaves the step short of break-even
at this acceptance rate; hiding it *and* recovering §3.1's bf16 acceptance
(4.73) would give 1.09×. The 1.24× in §3.1 and §6 assumed a bf16 drafter and no
pre-pass, and is now clearly an upper bound that neither half of the pair
reaches.

### 3.13 The third PDI: 118 ms/block, and it should be deleted rather than tuned

The pre-pass is the largest line in a speculative step after the verify pass —
**82.0 ms of dispatch plus 36.5 ms of ELF load/unload per block**, 36% of the
328.3 ms step. `dflash_prepass_cost.py` tests the three candidate explanations
`[hw]`; two of them are wrong, which is why it is worth writing down.

**Not the way it is called.** `air/backend/xrt.py`'s invoker allocates a fresh
BO per argument on *every* call, copies each host array into it twice, and syncs
all of them back afterwards — and the pre-pass passes all its weights as
arguments: 45 args, 33.7 MB per dispatch, **92% of it constant weights**, moved
in both directions every block. That is the obvious answer and it is not the
answer. Of the 81 ms, **68 ms is device time**; the whole host side is 13 ms.

**Not the 24 launches.** Varying the drafter layer count moves both the launch
count and the weight bytes, and the result is a straight line through the
origin against **bytes**:

| drafter layers | launches | weight MB | device ms | GB/s |
|---|---|---|---|---|
| 1 | 8 | 19.90 | 42.7 | 0.47 |
| 3 | 16 | 25.38 | 55.1 | 0.46 |
| 5 | 24 | 30.87 | 68.9 | 0.45 |

Same rate at 8, 16 and 24 launches; the intercept is within noise of zero. The
launches cost nothing measurable and fusing them would gain nothing.

**It is weight bandwidth, at 0.46 GB/s.** The batch-8 verify pass streams the
model's ~2.26 GB in 198.8 ms — about **11 GB/s** — and batch-1 decode, which is
bandwidth-bound by definition, does 56.95 ms/token on the same weights, near
**39 GB/s**. The pre-pass runs 24–85× below what this silicon does a few
milliseconds earlier in the same loop, because it streams **30.9 MB of constant
weights through a generic int4 GEMM on four cores, once per block, to produce at
most eight rows.**

Widening that GEMM is not a parameter change. `HERD_N` is the **row** extent and
NPU2 has four compute rows, so `HERD_N=4` already saturates the single column it
uses; 8 and 16 fail placement outright. Going wider means more *columns*
(`HERD_M > 1`), which `dflash_int4`'s L2 A-stage assert blocks at fc's K — the
stage is `herd_m · tile_m · K · 2` bytes and already spends 400 KB of a 512 KB
memtile at `herd_m=1`. Even a perfect four-column rewrite is ~4×, i.e. 68 → 17
ms, which is still six times the cost of not having a third PDI at all.

**So: fold `fc` and the context K/V into the target's program as trailing
waves.** Three reasons it belongs there rather than in the drafter's:

- The taps it consumes are produced by the verify pass that just ran, so it
  keeps them on device and drops an 819 KB host round trip as well.
- Rows are free — the pass is weight-bound, not row-bound — so computing all
  eight slots and discarding the rejected ones costs nothing, and the host no
  longer has to know which were accepted before the pre-pass can start.
- Three device programs is what forces the load/unload; two do not.

The shapes already fit the engine. §3.3 established that
`fc(concat(h₁..h₅)) = Σᵢ Wᵢ·hᵢ` is **five accumulating 2560→2560 projections** —
the shape the proj cores already run, accumulated across phases — and
`k_proj`/`v_proj` at 2560→1024 is the drafter's own projection shape. What it
costs is requantizing those eleven matrices from AWQ int4 to the cascade's q4k
(§3.4 keeps the two conventions deliberately separate). §3.7's result says there
is slack for that: re-seeding with the oracle's bf16 context rows instead of the
int4 pre-pass left **all seven draft tokens identical**.

What it is worth: 30.9 MB is **1.4%** of what the verify pass already streams,
so ~3 ms rather than 118.5.

| | step | break-even | at 3.981 accepted |
|---|---|---|---|
| today | 328.3 ms | 5.76 | 0.69× |
| pre-pass folded in | ~250 ms | 4.39 | **0.91×** |

**Which is still short of 1.0, and that is the point of measuring it.** A free
pre-pass does not make DFlash pay here; it moves the question to the verify pass
at 198.8 ms — 3.49 baseline tokens for eight — and to the 0.75 tokens of
acceptance that quantizing the drafter costs (§3.12). Neither is reachable by
tuning the third PDI, which is the argument for deleting it in one step rather
than optimizing it in several.

### 3.14 Folding it in: the layout, and what the engine will and will not allow

`dflash_prepass_waves.py` is the new builder. It owns every DFlash-specific
decision so `fused_decode.py` — the conventional-LLM engine — takes only a
generic hook and carries no DFlash code.

**Six waves, not ten**, which corrects §3.13's own sketch. `fc` is **one**
2560×12800 projection at I2=5 J2=25, not five accumulating 2560×2560 ones:
accumulating across input column-blocks is what the proj cores already do, and
`qwen3_4b_draft_requant.py` already packs it that way as `W_fc`. The context K/V
needs nothing packed at all — it is the drafter's own `k_proj`/`v_proj`, already
inside each layer's phase-0 `concat([Wq;Wk;Wv])` slab.

| wave | M | K | I2 | J2 | iter_lo | blocks | MB | X | dest |
|---|---|---|---|---|---|---|---|---|---|
| fc | 2560 | 12800 | 5 | 25 | 0 | 4000 | 20.5 | taps | rms |
| ctxkv0–4 | 2048 | 2560 | 4 | 5 | **8** | 640 | 3.3 ea | xnorm | rope |

36.9 MB, **0.58 of a drafter layer slab** — which is where §3.13's ~3.4 ms came
from, now derived rather than estimated.

**Both CPU gates pass** `[cpu]`, on the shipped cache rather than a second copy
made by the same code:

- **Layout.** The K/V window is *derived*: under `iter_major` a row-iteration
  owns a contiguous `NCX*NCY*ROW_BLOCK`-row span, so K and V are iterations
  8–11. fc reads back at **4.29e-02** and the window at **5.75e-02** against a
  6.67e-02 quantization step. The negative control is the point — the same
  window against `q`'s rows reads **1.218e+00**. A layout error looks like that,
  not like the line above it.
- **Format.** q4k instead of AWQ int4, through the numpy chain §3.6 already
  validated the *order* of. q4k is **not worse — it is better at every stage but
  one** (`target_hidden` 9.16e-02 against int4's 1.06e-01, cos ≥ 0.995), which
  is what a group of 32 against AWQ's 128 should do.

**The hook is inert.** `DECODE_EXTRA_WAVES` is a validated JSON list; unset, the
emitted AIR is byte-identical at batch 1 and batch 8 `[static]`.

**What the engine will not allow, and it decides the last open question.** An
extra wave is a third arm, and an arm may select **a count and never a program**.
That is not style — `_rms_batched`'s header records the measurement: AIR derives
a buffer's lock credit from how many channel ops *name* it, counting across
`scf.if` arms because it cannot know they are exclusive, so a second arm doubled
the credit on `@layerOut`'s MM2S and `@outY`'s S2MM and decode hung in wave 0
with the KV written and the layer output never landing. The memtile's X feed
already obeys the same discipline: both its arms call `_feed_inX("xnorm", n)`
and differ only in `n`.

So §3.13's two candidates for `fc`'s raw 12800-wide X both fail. Routing it
through the rms core with `rms_copy_aie` instead of `rms_norm_aie` is a
different *program* on the tightest tile in the design. A dedicated `@tapsX`
channel into the X memtile is a different program on the memtile, which is the
same doubled-credit hazard one hop out.

**What is left is to put the taps on `@xnorm` from the launch side.** `@xnorm`
is already a convergent packet channel with four producers (the rms core twice,
the o memtile, the GLU down buffer), consumed by the X memtile in phase-time
order; a fifth, time-disjoint producer is that same pattern. The memtile then
changes by a count only, and the rms core is not touched at all for the X
source. It is legal in the direction that matters — a channel with several
producers is refused only when the *consumer* is the shim (§5.4's `@layerOut`
finding), and here it is a memtile.

Remaining, and it needs device iteration rather than analysis: the arm-2 wiring
across the launch feed, the two memtile counts, the proj core's scalars, the rms
core's `hidden_norm` on `fc`'s output, and the rope core's `k_norm`/RoPE into
`appendK`/`appendV` at the drafter's KV base.

## 4. qwen3-4b at batch 1: verified

The target model works end to end on NPU2 `[hw]`:

- Prefill: 36 layers + tied LM-head GEMV, real Q4NX weights, first token
  **12095** (" Paris"), matching HF bf16.
- Decode: `[12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576]` →
  *" Paris. The capital of Germany is Berlin. The"*.
- Production `ATTN_MAXL=2048` templates, two prompts, all facts correct,
  **14.99 tok/s**.

**The bug that blocked this was `GLU_SLICE`** `[static]`: `models/qwen3-4b.h`
carried 1024, inherited from qwen3-8b.h when the header was copied. The Python
builder computes its own value from this model's egress round-count parity
(`ROUNDS_PER_DEST[GLU_DEST]=38`, `38//2=19`, odd → `GLU_PKTS=1` → 512). With
1024 in the header the kernel consumed slices at double the width the IR fed
it, and the FFN/down contribution came out ~zero for every layer — verified by
`DECODE_ACC_STOP`, where the real decode's output was byte-identical to an
FFN-skipped debug build. Fixed to 512; shipped as PR #1916. qwen2.5-7b hits the
identical odd-parity case and already had 512.

**Per-layer hidden-state taps work** (`DECODE_HIDDEN_TAPS`), which DFlash needs
to feed the drafter `[hw]`. Read back on device at P=5 against HF bf16, cosine
per tap slot: 0.999843 / 0.995922 / 0.989015 / 0.980171 / 0.981922 / 0.985569,
**mean 0.988740**. Before this it had only ever been an IR-level no-op check.

**Where a batch-1 dispatch's time goes** `[hw]`, qwen3-4b, median of 25,
ctx 8→1800, three template families (`_decode_cost/run_family.py`):

| term | ms |
|---|---|
| fixed | 1.78 |
| per layer | 1.593 |
| × 36 layers | 57.34 |
| attention | 7.82 |
| lm head | 7.00 |
| **total (ctx 1800)** | **73.94** (13.5 tok/s) |

The 73.94 ms here and the 14.99 tok/s above are different measurement
conditions (synthetic-KV sweep vs a real generation run), not a contradiction.

## 5. Batching: the actual blocker

### 5.1 The L1 ceiling

The batched rms core holds, at peak `[static]`:

```
_rms_l1_bytes(b) = 2*(b*K + b*STG_W + 2*K) + 4*b        # levels 0-2
```

against `_L1_ROWTILE_BUDGET = 65536 - STACK_SIZE`. Both norm weights are
resident (the body allocs `w` and `w2` together); reusing one buffer for both
saves K bf16 and **hangs the batched decode in wave 0**, so it is not available.

Ceilings this gives `[static]`, independently re-derived:

| model | K | ceiling |
|---|---|---|
| llama-3.2-1b | 2048 | 9 |
| **qwen3-4b** | **2560** | **7** |
| gemma3-4b | 2560 | 7 (also refused: sandwich norm) |
| llama-3.2-3b, phi4-mini | 3072 | 5 |
| qwen2.5-7b | 3584 | 4 |
| qwen3-8b, llama-3.1-8b | 4096 | 4 |

**`DECODE_BATCH` must be a multiple of 8** — `aie::mmul<8,8,8>`'s A tile is 8
rows and `q4k_mm.h` asserts `rowA == 1`, so 8/16/32 compile and 2/4 fail at
`static_assert` `[static]`. So qwen3-4b's ceiling of 7 means **it could not
batch at all**, and llama's 9 means 8 is the only batch it can do.

This is the single fact that shaped everything else: **all batched work in this
repo's history was done on llama-3.2-1b, because qwen3-4b could not build.**

### 5.2 What was cleared, and how

`STACK_SIZE` (env `DECODE_STACK`, default 10240) is a knob, and the budget is
`65536 - STACK_SIZE`. qwen3-4b at batch 8 needs 59,424 B by the formula and
**59,428 B** in the emitted IR (the formula omits the 4-byte herd RTP buffer);
tile_2_2's buffers are 40960 residual + 8192 staging + 5120 + 5120 norms + 32
scales + 4 RTP. At `DECODE_STACK=6080` the budget is 59,456 B and it fits.

**qwen3-4b now builds and dispatches at batch 8** `[hw]` — no hang, 20480
elements returned, 1500 distinct values. This is the first time any qwen3-4b
build above batch 1 has run.

**The stack reduction is not a fudge, and that was checked rather than
asserted** `[hw]`: llama-3.2-1b at batch 8 produces **byte-identical** output at
`DECODE_STACK=10240` and `DECODE_STACK=6080` (rms rel 1.81e-03, the same 226
differing elements). Reducing the stack changes nothing about the result.

### 5.3 The second bug: `build_template.sh` compiled the wrong model's kernels

With the ceiling cleared, batch 8 ran but **failed** the equivalence gate at
rms rel 5.02e-01 against a 5e-2 tolerance. That turned out not to be a batching
fault at all.

The trail, in the order it actually went — each step ruling out the obvious
suspect:

1. A per-512-chunk profile (`_b8_chunk_profile.py`) showed chunks 0-3 agreeing
   at ~1.8% and chunk 4 of the **batch-1 reference** reading all zeros `[hw]`.
   So most of the 50% was the reference, not the batched output.
2. An X-buffer probe (`_x_hole.py`), which distinguishes *zeroed* from
   *untouched* from *written*, showed the qwen3-4b batch-1 reference was
   **bit-identical to the host fill** in chunks 0-3 — the device never wrote
   them — with chunk 4 zeroed `[hw]`. llama-3.2-1b wrote all four of its
   chunks normally `[hw]`.
3. Not the wave count (`UNI_WAVE_HI` 1, 2 and 36 all identical `[hw]`), not the
   context length (same at L=2048 `[hw]`), not `W_DUAL_CHAN` (defaults to 1),
   not the RTP arity (`dispatch_args` returns `[]` for non-dynseq builds).
4. The **production** qwen3-4b template — the one that generates correct text —
   passed the same probe with all five chunks written `[hw]`. So two qwen3-4b
   batch-1 templates disagreed, and the difference had to be in how they were
   built.

**Root cause:** `build_template.sh` takes its kernel flags from *this*
directory's Makefile via `check_kernels_inert.makefile_kbase()`, and that
`PEANO_KBASE` hardcodes **`-DMODEL_TYPE=LLAMA_3_2_1B`** — the fused_decode
Makefile only ever builds llama-3.2-1b, and each `llms/<model>_q4nx/Makefile`
carries its own `-DMODEL_TYPE`. So every `DECODE_MODEL != llama-3.2-1b`
template this script produced had **llama kernels** (`MODEL_DIM` 2048, `DH` 64)
inside a design built for another model's dimensions. Its own usage text
documents `DECODE_MODEL=qwen3-4b ... ./build_template.sh 8 2048`, so this was
reachable exactly as intended and silently wrong.

**The failure mode is the dangerous kind**: it links, it dispatches, it returns
`COMPLETED`, and the layer output is simply never written. Nothing reports an
error.

Fixed by deriving `MODEL_TYPE` from `DECODE_MODEL` in `build_template.sh` (a
`case` covering all ten models, with an explicit failure for an unknown one),
substituted into the flags before any kernel is compiled — the same
"the object and the design cannot disagree" rule the script already applied to
`PROJ_MM_BATCH`.

### 5.3b Batch 8, with correct kernels: both models pass

`batch_equiv.py --tokens 0`, L=128, `DECODE_STACK=6080` `[hw]`:

| design | layers | rms rel vs batch 1 | verdict |
|---|---|---|---|
| **qwen3-4b** (target) | 36 | **2.79e-03** | pass (tol 5e-2) |
| **qwen3-4b-draft** (drafter) | 5 | **3.75e-03** | pass |
| llama-3.2-1b (control) | 16 | 1.81e-03 | pass, unchanged |

Both DFlash halves now run correctly at batch 8 on qwen3-4b, and the residual
error sits where llama's does — this is the GEMV-vs-mmul kernel difference the
gate is designed to tolerate, not a wiring fault.

The llama control is byte-for-byte what it was before the fix (same 226
differing elements), so making `MODEL_TYPE` explicit changed nothing for the
model that was accidentally correct.

**Scope of what this invalidates:** any qwen3-4b (or gemma3-4b, phi4-mini,
qwen2.5-*, qwen3-8b, llama-3.1-8b, lfm2) template ever built through
`build_template.sh` was built with llama kernels. Measurements taken on those
templates mean nothing and must be re-run — including this document's own
level-3 results (§5.4), which went through the same script.

**Still not measured:** what a batch-8 dispatch *costs* on either model. It runs
correctly; it has not been timed.

### 5.4 RMS_BAND_STREAM, and what is actually known about it

The band-streamed residual is the intended route past the ceiling: move the
residual out of L1 and round-trip one band at a time through DDR on
`@rmsX`/`@layerOut`, which would raise qwen3-4b's ceiling to 21 `[model]`.

| level | what it does | state |
|---|---|---|
| 0 | off | shipping |
| 1 | two-pass scale kernel | **llama-3.2-1b, batch 8, on device: 8/8 accepted** `[hw]` |
| 2 | bands the initial `rmsX` read; `xb` still full-size | **verified: llama-3.2-1b 8/8 accepted; qwen3-4b batch 8 byte-identical to level 0** `[hw]` |
| 3 | `xb` shrinks to one band; six-phase refetch | builds clean, hangs in wave 0 — see below |

**Levels 1 and 2 were verified on llama-3.2-1b, not qwen3-4b.** They cannot
have been verified on qwen3-4b: neither level changes `_xb_w`, so neither moves
qwen3-4b's ceiling of 7, so neither can build at batch 8 there. An earlier
version of this document claimed level 2 was "already verified on real qwen3-4b
hardware at `DECODE_BATCH=8`" — see §9.

**Level 3 now builds, and hangs on a wall one layer further down.** Three
separate defects were found and two of them fixed; the third is structural.
Everything below is post-`MODEL_TYPE`-fix, qwen3-4b, `DECODE_BATCH=8`,
`DECODE_STACK=6080`.

#### (i) BD-block exhaustion — FIXED

`aircc` used to fail with `'aie.mem' op has more than 16 blocks`: the rms core
wanted 22 channel ops against a core tile's 16 BD blocks.

| op | before | after |
|---|---|---|
| `get @rmsX` | 14 | **6** |
| `put @xnorm` | 2 | 2 |
| `get @outY` | 2 | 2 |
| `put @layerOut` | 2 | 2 |
| `get @rmsW` | 1 | 1 |
| `get @rmsW2` | 1 | 1 |
| **total** | **22** | **14** |

The 14 came from the scale pre-pass being a PYTHON-unrolled loop over bands:
`_nband` textually identical gets, one BD block each. Rolling it into an
`scf.for` costs one block for the whole pre-pass — the loop becomes a repeat
count on a single BD — and the only per-band value it carried, the `first`
flag, is just `band == 0` and comes from the loop index via the same
`extui(cmpi)` shape `PROJ_RC_CACHE`'s own first-visit flag uses. The count is
then `2 pre-pass + 2 regen + 2 residual = 6`, **independent of K**, and the
whole tile lands at 14 of 16 (11 BD blocks after AIR folds equivalent ops).

This retires the previous section's whole band-width analysis: with the op
count no longer proportional to `nband`, there is no BD-vs-L1 squeeze and no
need to move off `w = STG_W = 512`.

#### (ii) A lock-count mismatch on `@layerOut` — FIXED

With the blocks under the limit, the first level-3 build ever to reach the
device timed out with **nothing written at all**. Cause, read straight out of
`aie.air.mlir` `[static]`:

```
DMA  (MM2S, layerOut):  use_lock(lock_2_2_128, AcquireGreaterEqual, 3) ... Release 3
core (the band put):    use_lock(lock_2_2_128, Release, 1)
```

A tile-DMA BD's acquire/release COUNT comes from
`air::getLockValuePair(targetModel, buffer)` — the two-argument overload in
`AIRToAIEPass.cpp` — which is `ceil(reads/writes)` over the memcpy ops naming
that buffer. The core side is hardwired to 1. Sharing one band buffer between
the `@rmsX` gets (6, writes) and the `@layerOut` puts (2, reads) therefore asks
the DMA for 3 credits per firing against a core that issues 1: the DMA never
fires, the core blocks on its second band, and nothing lands.

Fix: `residual_acc_row_banded_out_aie`, an out-of-place `out = acc + x` instead
of `acc += x`, so the fetch buffer is written-only and the drain buffer is
read-only and both ratios are 1:1. Costs one more `BATCH*STG_W` band buffer
(level 3 is now `2*b*STG_W` of band where levels 0-2 are `b*K` of resident
row). It also keeps `@rmsX`'s BD ring uniform, which (iii) turns out to
require.

#### (iii) The rms core has two S2MM ports and level 3 needs three — FIXED

`air-to-aie` turns a tile's per-channel memcpy list into BD **tasks**, one per
run of equal static trip counts (`getRepeatCounts`), and emits a count-free,
lock-driven **infinite BD ring** only when the whole physical channel collapses
to ONE task (`generateDmaBdProgram`: `infiniteBDLoopMode = repeat_counts.size()
== 1`). A terminating task fires once per dispatch, which is wrong for anything
consumed per layer. So every S2MM port here must be a single ring — and a ring
fires each of its slots exactly once per rotation.

That gives the rule the whole problem reduces to: **all channels sharing a port
must have equal per-wave transfer counts, consumed in rotation order.** The
shipping level-0 build satisfies it exactly `[static]`:

```
S2MM 0:  ring [ rmsX, rmsW, rmsW2 ]   counts 1, 1, 1     -- one rotation per layer
S2MM 1:  ring [ outY ]                count 10
```

Level 3's counts are `rmsX 270, rmsW 1, rmsW2 1, outY 10` per layer. `rmsX` must
be alone (270 is not a ratio any 16-slot ring expresses) and `outY` must be alone
(its BD is deliberately unstamped — the channel pins several packet ids and the
core writes the routing id into the payload header, so the DMA must not filter),
and that is both ports. Forcing the weights onto `outY`'s port instead was
measured and hangs the **known-good level-0 batch-8 design** too `[hw]`.

**The fix is `RMS_W_ON_X`: the norm weights ride the `@rmsX` stream, so they
stop needing a port at all.** A norm weight is K bf16 and a band fetch already
moves `BATCH*STG_W >= K` of them, so the weight fits in ONE band-shaped transfer
on `@rmsX`: the launch puts it from the rms weight buffer with a band-shaped
descriptor, the core takes it with the same `_rms_band_get` every other visit
uses, and `band_to_weight_aie` copies the first K elements out. The descriptor is
identical to every other rmsX get, so AIR folds it into the same BD and the ring
stays one count-free slot. Cost: `BATCH*STG_W - K` wasted elements per weight per
layer (0.1% of a layer's weight traffic) and one K-element on-core copy.

**Verified on device, in isolation** `[hw]`: carrying both norm weights on
`@rmsX` at **level 2** — where no banding is involved and the weight goes
straight into its own buffer with a 2-D descriptor — passes the batch-8
equivalence gate at **1.83e-03**, identical to level 2 without it
(`_RMS_W_ON_X=1 RMS_BAND_STREAM=2`). The rms core's inbound set is then
`{rmsX, outY}`, exactly two, and the level-3 tile emits **one self-looping BD
per port**, every lock count 1.

That test also established a rule worth keeping: **the launch must feed a
channel in the same order the core's BD ring rotates, and the ring is emitted in
the CORE's program order.** At level <3 the row get sits at the top of
`_rms_batched` with the allocs while the weight gets come later, so the ring is
`[row, w, w2]`; feeding the weights first does not hang, it silently swaps the
row and the weights into each other's buffers (measured: rms rel **1.56e+00**).
Level 3 is immune only because all its rmsX gets share one descriptor and one
buffer.

#### (iv) Still hanging, and what is now ruled out

With (i)-(iii) fixed, level 3 builds and emits a structurally ideal tile
program — **one self-looping BD per port**, every lock count 1, packet ids
matching end to end — and still times out in wave 0 with **nothing written at
all, not even the KV cache** `[hw]`.

Two more real defects were found and fixed on the way, both of the same family
(a descriptor that is correct AIR and does not survive lowering, with no
diagnostic):

- **A 5-D shim descriptor loses its outermost dimension.** `_rms_x_feed_ph`
  folded a whole phase into one op via a stride-0 refeed dimension. A shim BD
  holds three dimensions plus a repeat, and the fifth was dropped:
  `[13,5,8,8,64]` came out as `[5,8,8,64]` — five bands where the core waits
  for sixty-five. **Fixed by going TOKEN-MAJOR**: a band walked token-major
  makes each token's STG_W elements contiguous, so the shim describes a band in
  two dimensions instead of three and the whole phase fits in a legal 4-D op
  (`[1+nrefeed, nband, BATCH, STG_W]`). The compute tile keeps its
  `_RMS_DMA_CHUNK` sub-dicing — its wrap field is 8 bits, the shim's is 10.
- **`offsets` are multiplied by their dimension's stride.** An `air.channel`
  offset list is the strided-view convention: the linear offset is
  `sum(offsets[i]*strides[i])`, not `offsets[0]`. Every banded helper put the
  buffer offset on dimension 0, which is only correct because that base is
  0 (`X_SLOTS == 1`). It bit as soon as a real base appeared: the vocab arm's
  norm-weight read lowered to `offset = 18874368`, i.e. `_final_norm_off * 64`.
  The same bug was live in the **ropeLUT fold** (`_rope_off * ROPE_W_LEN`),
  where the element COUNT stayed right — so `shim_volume.py` and
  `check_channel_balance.py` both read balanced — and rope silently got
  garbage. That fold has been removed outright: it was added for the
  rmsX-vs-rmsW port contention that `RMS_W_ON_X` retires, and the comment
  beside it already warned that a single B-wide put against B per-token gets is
  not something the channel promises.

What is established about the remaining hang `[hw]`, all measured:

- **A one-layer repro exists.** `UNI_DEC_OVERRIDE=1` hangs identically, and the
  same one-layer build at level 0 and at level 2 both pass at 1.83e-03.
- **Level 2 is verified on qwen3-4b at batch 8**, at one layer and at the full
  36 — where it is **byte-identical to level 0** (both 5.56e-03). The banded
  `@rmsX` get is not the problem, and level 2 is a usable fallback.
- **The fault is in ph0 or earlier.** `_RMS_PH0_ONLY=1` feeds ph0 and nothing
  else; it still writes no KV, and the KV append is downstream of ph0 through
  the QKV projection and rope.
- **Not the launch-side folding, in either direction.** The folded feed (6 shim
  tasks/wave) and `_RMS_NO_FOLD=1` (272 puts against 272 gets, strict 1:1
  packet pairing) hang identically. So it is neither the repeat-count BD nor a
  packet spanning several core BD firings.
- **Not the shim task ORDER.** The final order is the intended one: QKV weights
  fed and awaited, then ph0's feed, then the KV append and readback, then the
  remaining weight phases, then residual1/ph2/residual2. Both neighbouring
  placements were measured and both deadlock for understood reasons — too early
  starves the projections of weights, too late puts `await appendK` in front of
  ph0.
- **Not the X memtile.** Its BD program is structurally identical to level 2's;
  only lock and packet-id numbering shift.
- **Not the `air.await_appends` barrier any more** (see (v)).

**The bisection ladder found it, and level 3 now reaches the KV append.** Each
rung is a separate device run at `UNI_DEC_OVERRIDE=1`, `DECODE_BATCH=8` `[hw]`:

| what changed | result |
|---|---|
| weights only on `@rmsX` (`_RMS_FETCH_OFF=1`) — 2 transfers | **KV written** |
| a THIRD straight-line weight-shaped transfer (`_RMS_W3=1`) | **KV written** |
| that third transfer moved after the arm-selecting `scf.index_switch` | nothing written |
| the same transfer, but FED from the top of `_uni_dec` instead | **KV written** |
| **the whole ph0 feed, fed from the top of `_uni_dec`** | **KV written** |

**The cause is the LAUNCH-side position of the banded ph0 feed, and nothing on
the compute tile.** Fed from the QKV weight boundary — after `_feed_wcols(p=0)`,
just before the KV append, which is where a dependency argument puts it — the
dispatch hangs with nothing written at all. Fed from the top of `_uni_dec`,
right after the norm weights and before the rope LUT and the projection weights,
it runs through ph0, the QKV projection and rope, and **the KV cache comes back
complete** (16,384 of 294,912 elements = 8 tokens x KVSZ_TOK, exactly one
block). Same descriptor, same count, same core code; only the position differs.
That is where levels 0 and 2 put their single whole-row `@rmsX` read, so the
banded feed wants the same slot rather than a "smarter" one.

The launch order must also still match the order the core takes the transfers
in, since they all share one BD: the feed goes AFTER the two norm-weight puts.
Getting that backwards does not hang, it swaps the buffers (§5.4 iii).

This retires a lot of theory that was in this section, each killed by its own
rung: it is not the loop form (unrolled hangs identically), not the transfer
size (one band hangs as readily as five), not the source buffer
(`_RMS_SRC_RMS=1` sources the same shapes from the rms buffer), not the
descriptor (the weight's own descriptor hangs in the wrong position), and not a
transfer-count limit (three work when all three are fed early).

**What is left, after ph0.** With ph0 fixed the layer got as far as
residual1 and stopped there; three more defects were found and fixed, and the
remaining fault is now isolated to a single line of the design. In order:

#### (vi) The weight-feed loop awaits each phase, so every banded feed has to be armed a phase early — FIXED

`_feed_wcols` emits a coalesced put per column and then an await barrier across
all of them, and that await only returns once the projection cores have
CONSUMED the phase — 1.9 M bf16 for gate-up, far more than the memtile holds, so
consumption means the phase actually ran. Every level-3 banded feed sat AFTER
that whole loop, which is a deadlock:

```
host: await(gate-up weights)   needs ph2 to run
      ph2                      needs residual1
      residual1                needs an @rmsX band
      @rmsX band               not armed yet
core: residual1's _rms_band_get, forever
```

Levels 0-2 are immune because their residual never leaves L1: nothing the core
needs mid-layer comes from the host at all. The fix (`_rms_interleave_after_phase`)
arms each feed one phase EARLIER than the phase whose await depends on it —
residual1's round-trip after p=0, ph2's feed and residual2's drain after p=1,
residual2's put after p=2. p=0 and not p=1 for residual1: the o-proj await needs
the o-proj to drain five `@outY` rounds into the rms core, and the rms core takes
round r+1 only after round r's band round-trip, so the round-trip has to be armed
before the o-proj feed rather than after it.

`_RMS_LATE_RT=1` restores the old placement. Two other placements were measured
and are worse, both for reasons the above explains: per-band put/drain pairs
(`_RMS_RT_PER_BAND=1`) order band b's drain before band b+1's put, so the host
blocks on the first drain before it has issued the weight feed that drain's data
depends on — nothing written at all; and hoisting only residual1's put while
leaving its drain late changes nothing, because the core cannot put band 0 back
until the drain is armed.

#### (vii) `h` cannot live in X — FIXED

residual1's output is a whole intermediate hidden state that ph2 re-reads 39
times and residual2 reads once more. Writing it back over X — the obvious
in-place choice, and what levels 0-2 do with their single whole-row write — puts
every level-3 feed and drain on one memref, and AIR chains same-memref channel
ops in program order. In `placed.air.mlir`:

```
%138 = air.channel.put  @rmsX     (X[0,0,0] [5,8,512])            <- reads X
%139 = air.channel.get  @layerOut (X[0,0,0] [5,8,512]) [.. %138]  <- writes X
```

so the drain lowers to `await(put); start(get)`, which the core cannot satisfy:
the put does not retire until the core has taken all five bands, and the core
will not take band b+1 until band b's drain is armed.

`h` now gets its own region — `RMS_SCRATCH`, one block of hidden states appended
to the END of the Y buffer, so it needs no fifth DDR argument and no ABI change
(`llms/bench/decode_geometry.py` adds the same term from the same symbol). X is
then touched twice per layer, at the two ends:

```
read X   ph0's sweep, residual1's put
write H  residual1's drain
read H   ph2's sweep, residual2's put
write X  residual2's drain   <- the layer output
```

Without the host-side term the drain runs off the end of the Y BO and into
whatever is mapped next — measured: it landed in the KV cache, which is how the
missing term was found.

**With (vi) and (vii), residual1 completes.** `h` comes back whole: 20,480 of
20,480 elements in the scratch region, one contiguous run `[hw]`. (In-place in X
the same data reads as "10,564 of 20,480 written" — the other 9,916 are elements
where `x + o_proj == x` in bf16, not a partial write. That is worth knowing
before reading a partial-write count as a partial write.)

#### (viii) The paced tail-drain anchor skipped coalesced feeds — FIXED (compiler)

`synthesizeDoubleBufferedAwaits` (AIRRtToNpuPass.cpp) drains a paced MM2S
channel's in-flight tail after "the last start among ALL paced channels in the
segment", and excluded `air.coalesced_shim_feed` tasks from that anchor. The
anchor is only asking where the segment stops issuing, and a coalesced weight
feed issued after the last paced start is still part of the segment — excluding
it drains the paced channels while the segment is still going. Measured here:
the last `@rmsX` task is residual2's band feed, and the core takes those bands
only after the DOWN projection has produced its output, so the drain awaited it
BEFORE the down weight feed was issued and the projection never ran. Coalesced
feeds now count towards the anchor. The anchor only ever moves LATER within the
same segment, so the fence it provides is unchanged.

This does move IR for every design, so it was re-gated rather than argued:
**qwen3-4b batch 8 at 36 layers still passes at 5.56e-03 at level 0 AND at level
2, byte-for-byte the same output (1,491 of 2,560 bytes differ, 1,623 distinct
values, both levels)** `[hw]`.

#### (ix) residual2, and the head-of-line deadlock — FIXED

The last stall was **head-of-line blocking on a shared stream link**, and the
bisection that found it is worth keeping because five plausible explanations
died on the way.

Cutting the round-trip into its two halves separated them. Both halves keep
their launch-side op, so the channels stay balanced either way `[hw]`:

```
_RMS_BANDS_GET=1  _RMS_BANDS_PUT=12   residual2 puts back, never gets  -> COMPLETES
_RMS_BANDS_GET=12 _RMS_BANDS_PUT=1    residual2 gets, never puts back  -> hangs
```

So the fault was residual2's `@rmsX` GET. Then a **progress witness** placed
that get precisely. `_RMS_RES2_PUT_FIRST` moves residual2's band put-back to the
front of the loop body, ahead of both gets: its contents are stale garbage but
the drain is host-visible, so reaching the loop lands exactly one band in X.
`_RMS_RES2_PUT_MID` puts it between the two gets instead. Nothing else can tell
these apart — everything the core does after ph2 is invisible from the host.

| witness position | X | reading |
|---|---|---|
| before both gets (`_PUT_FIRST`) | one band, 8 runs of 512 at stride K | core reaches residual2 |
| between the gets (`_PUT_MID`) | empty | it blocks on the `@outY` get |

**On the `@outY` get** — not the band, even though dropping the band get is what
makes it complete. Two flows reach the rms core and they both come from the
SOUTH:

```
aie.packet_flow(0)   shim_noc(2,0) DMA0 -> tile(2,2) DMA0    @rmsX
aie.packet_flow(30)  mem_tile(2,1) DMA1 -> tile(2,2) DMA1    @outY
```

Packet-switched, they are multiplexed onto one south→north stream. The rms core
is single-buffered on `@rmsX`, so at the end of residual1 it takes one band of
residual2's ahead and then has NO credit until residual2's first iteration —
which is waiting on `@outY`. The remaining bands park at the head of the shared
link, the DOWN projection's `@outY` round queues behind them, and neither moves.

Levels 0-2 never hit it because their `@rmsX` carries ONE whole-row transfer at
the top of the layer, consumed immediately: there is never a surplus band in the
network while the core is waiting on `@outY`. That is the invariant level 3
breaks, and it is why the fault looked like it belonged to residual2 — residual1
runs while the shim is still streaming, so no queue has built up yet.

**The fix is one line.** `@rmsX` is packet-switched because of `FULL4`, so it can
converge with the o-proj/down id on the rms core's S2MM0 — and under `RMS_W_ON_X`
that reason is gone: `@rmsX` is pinned to S2MM0 and `@outY` to S2MM1, they no
longer share a port, and `@rmsX` has no id to demux against. Declaring it
CIRCUIT-switched at level 3 gives it a dedicated physical stream channel on the
shared link, and the head-of-line disappears. `_RMS_X_PACKET=1` restores the
packet flow, which is the A side of the measurement.

Killed by measurement on the way, each its own device run:

| hypothesis | test | result |
|---|---|---|
| residual2 is second | `_RMS_RES_ONLY=2` — it is the only round-trip | still hangs |
| the destination buffer | `_RMS_H_SWAP=1` — `h` in X, output in the scratch | the LATE one still hangs |
| the arming position | `_RMS_RES2_{PUT,DRAIN}_PHASE=0 _RMS_PH2_PHASE=0` | still hangs |
| ph0/ph2's band sweeps | `_RMS_FETCH_OFF=1` (@rmsX down to 3 tasks) | still hangs |
| residual2's descriptor or source | `_RMS_RES2_AS_W=1` — the norm weight's own descriptor | still hangs |
| a second shim task on `@rmsX` | `_RMS_RT_COMBINED=1` — both residuals in ONE task | still hangs |
| holding the `@outY` buffer across the band | `_RMS_STG_COPY=1` — copy the round out, release early | still hangs |
| `@outY` taken before the band | `_RMS_BAND_FIRST=1` | still hangs |

The last two are the near misses: both are about back-pressure, and both are
powerless because the blockage is one hop upstream, in the switch, not in the
tile. `_RMS_BAND_FIRST` in particular cannot help — the core takes the band it
already has buffered and then waits for `@outY`, which is still behind the NEXT
band on the link.

#### (x) Level 3 is verified

`[hw]`, qwen3-4b, DECODE_BATCH=8, L=128, `batch_equiv.py`:

| | one layer | 36 layers |
|---|---|---|
| level 0 | 1.83e-03 | 5.56e-03 |
| level 2 | 1.83e-03 | 5.56e-03 |
| **level 3** | **3.86e-03** | **5.56e-03** |

At 36 layers all three produce the SAME output — 1,491 of 2,560 bytes differ
from the batch-1 reference, 1,623 distinct values, on every level. Gate is
5e-2.

**What it costs**, same templates, `dispatch_time.py`, median of 25 `[hw]`:

| | ms | per token |
|---|---|---|
| level 2 | 166.302 | 20.788 |
| level 3 | 173.624 | 21.703 |

**4.4%** for streaming the whole residual through DDR twice a layer — 1.11 M
bf16 per layer of extra shim traffic. Cheap enough that level 3 is not a
compromise made for batch 16; it is a viable shape in its own right.

#### (xi) The attention core's BD blocks — FIXED

With level 3 working, `BATCH_MAX_RMS` is 16 and the rms tile lands at exactly
59,456 bytes of a 59,456-byte budget. Batch 16 then failed somewhere else:

```
'aie.mem' op has more than 16 blocks     <- air.herd @attn_blk, the [2,4] attention block
```

The rms core is at 9 BD blocks there; the attention cores were over. The cause
was `_attn_o_put`: `@attnO` was **one put per token, Python-unrolled**, so one BD
block per token. Its own comment said why it could not fold — the descriptor
already spends all three dimensions a compute tile has on the un-interleave from
the kernel's `[q_head, dc, de]` layout to natural `(q_head, dh)`, so the token
had to ride the stride-1 offset.

**Fixed by moving the un-interleave to the memtile**, which has FOUR BD
dimensions where a compute tile has three. The CU now sends its whole block
flat — one op, ONE BD block at any batch — and the o-gather does the transpose
with the token as its fourth dimension:

```
src  j = t*DQ_PER_CU + dc*(QH*8) + qh*8 + de     (a_o, sent contiguously)
dst  p = t*DQ + c*DQ_PER_CU + qh*DH + dc*8 + de
     sizes [BATCH, DH/8, QH, 8]  strides [DQ, 8, DH, 1]
```

Same permutation, same data, same order; only which end of the channel describes
it changed. No kernel edit, so batch 1 is untouched and `check_kernels_inert.py`
stays satisfied. `_ATTN_O_PERTOKEN=1` restores the old form.

Verified: qwen3-4b batch 8, 36 layers, **5.56e-03 at levels 0, 2 AND 3** — the
same output as before the change, on all three. The attention change is not
gated on `RMS_BAND_STREAM`, so all three had to be re-gated, and were.

#### (xii) Three more ceilings between level 3 and a batch-16 BUILD — all cleared

Batch 16 gets past both BD-block limits and then hits three allocation ceilings
in a row, none of them visible in AIR and each reported by mlir-aie as a bare
error naming at most one buffer. `tile_budget.py` prints the map instead.

**MEMTILE L2, 512 KB.** `mem_tile(3,1)` wanted 573,440:

```
buf127  311,296   the DOWN phase's X refeed buffer (BATCH x 9728, refeed_count 5)
buf138  196,608   the QKV transpose staging (BATCH x 6144)
buf133..136  4 x 16,384   the KV block buffers
```

That column was always the fullest — 319,488 (60%) at batch 8 — so this is
plain doubling, not a placement regression. Nothing else is close: at batch 8,
5,1 is at 37%, 4,1 at 12%, 2,1 at 6% and the four weight-fan columns at 8%.

Neither big buffer is free to move. `DOWN_PCOL` is pinned by a routing
constraint the code already documents: with the weight hub on the X column
(`MAIN_PCOL == XMT_PCOL == 2`, which is qwen3-4b's floorplan) a down→X route
from col 4 crosses switches already carrying hub traffic, and the pathfinder
cannot complete it — measured, it fails to build. So the QKV staging moves
instead, and the search for a column was entirely empirical `[hw]`:

| `QKV_RELAY_COL` | result |
|---|---|
| 3 (default) | works, but is the column that overflows |
| 4 | builds, HANGS with nothing written — at batch 8 too, so it is the column and not the batch |
| 1, 6 | `'aie.dma_start' op repeat_count 384 is out of range [0, 255]` — the weight-fan columns are excluded for a real reason |
| **2** | **works**: batch 8 at 3.86e-03, identical to the default |

Column 2 is the X-memtile column, and a col-2 QKV relay was removed once before
for stalling in vocab mode — the arm guard that replaced it is what makes it
viable now. It leaves col 2 at 262,144 (50%) and col 3 at 376,832 (72%).

**MEMTILE BD IDS, 24 per channel — SHARED between MM2S and S2MM.** With the
staging moved, `mem_tile(4,1)` channel 1 held 24 BDs of MM2S and 2 of S2MM. The
24 were `@toRope`: one put per token again, Python-unrolled. The rows are
consecutive and M-wide, so B puts at `t*M` ARE one contiguous `BATCH*M` run —
folded, it is one BD at any batch, and the rope core still gets its M-wide row
per token (one producer BD feeding B consumer firings, the same shape the shim's
banded `@rmsX` task already has). `_QKV_ROPE_PERTOKEN=1` restores the old form.

**COMPUTE L1, and the RTP word.** `_L1_ROWTILE_BUDGET` was `65536 - STACK_SIZE`
and did not account for the per-herd RTP word, which the allocator places AFTER
every buffer. At batch 16 the rms core lands on 65,536 EXACTLY —

```
(stack)  6,080     buf122   5,120   w
buf126  16,384 xb  buf121   5,120   w2
buf125  16,384 xr  buf123      64   scl
buf124  16,384 stg ------------------
                   __air_herd_rtp_2_2 : 0x10000-0x10003   <- past the end
```

— so it is over by four bytes and nothing says which buffer to blame. The budget
now reserves `_L1_RTP_RESERVE = 16`, and batch 16 takes the bytes back from
`DECODE_STACK=6016`.

**16, not 64.** The first attempt reserved 64 and refused levels 0 and 2 at
batch 8, which put the rms core at 59,424 — 32 under the un-reserved ceiling and
verified on hardware all day. A ceiling model that rejects a working
configuration is a worse bug than the one it fixes; the map shows the RTP packed
with no alignment padding, so 4 is the true cost and 16 is the guard.

With all three cleared, **batch 16 BUILDS**. Levels 0, 2 and 3 all still pass at
batch 8 / 36 layers at 5.56e-03 `[hw]`.

#### (xiii) Batch 16 builds and hangs

`[hw]`, one layer, `QKV_RELAY_COL=2 DECODE_STACK=6016`: the dispatch times out
with **nothing written at all** — not X, not Y, not the KV cache — so it stalls
in wave 0, before the KV append.

What is already excluded:

- **Not the column, and not any of the three fixes above.** Batch 8 with the
  IDENTICAL settings passes at 3.86e-03.
- **Not the shim.** Every channel's element count is exactly 2x its batch-8
  value, or 1x for the weights, with the same task counts (`shim_volume.py`).
- **Not a descriptor field.** No multi-dimensional BD has an extent over its
  tile's wrap limit (255 for a core, 1023 for a memtile or shim) and no
  `repeat_count` is over 255.
- **Not tile capacity.** Every compute tile has headroom at batch 16 except the
  rms core, which now fits; no memtile is over L2 or over 24 BDs per channel.

So it is on-chip and early. The next step is the same one that worked for
residual2: a progress witness that is host-visible, placed to bisect wave 0.

**Level 3 at batch 1 is not a supported configuration** and should not be used
to judge it: the decode arm's band feed is gated on `RMS_BAND_STREAM >= 3`
alone while the vocab arm gates on `RMS_BAND_STREAM >= 3 and BATCH > 1`, and
the banded body is a batched-path construct. A batch-1 level-3 build hangs with
nothing written, with and without the launch-side folding `[hw]` — that is the
mismatch, not evidence about level 3's design.

**Superseded, for the record.** Before the `MODEL_TYPE` fix this section
reported that level 3 "loses the design's context-length dependence" (L=2047
and L=2048 insts byte-identical). That was measured on llama-kernel templates
and is withdrawn pending a re-run; it is not currently known to be true.

Older findings, re-read against (iii):

- The `operand #0 does not dominate this use` failure on the second
  `air.channel.put` to `@rmsX` came with the pass's own warning that **the tile
  has no spare S2MM channel to move `@rmsW` onto** `[hw]`. That warning was
  right and is the same wall as (iii); the launch-side op folding that made it
  compile addressed the symptom. Whether the folding is still needed now that
  the compute side is fixed has not been re-tested — `_RMS_NO_FOLD=1` restores
  the pre-fold shape.
- **"Level 3 loses the design's context-length dependence" (L=2047 and L=2048
  insts byte-identical) is still withdrawn** — measured on llama-kernel
  templates, never re-run.

### 5.5 Known-hard mechanisms (kept because they cost real time to find)

`[static]`, all from source or ELF reads:

- **A core tile's DMA BD wrap field is 8 bits per dimension**
  (`getDmaBdWrapBits`: 8 for core tiles, 10 for mem/shim). `STG_W`=512 does not
  fit and is silently truncated — no verifier error, deterministic wrong data.
  This was the root cause of three failed level-2 variants; `_RMS_DMA_CHUNK=64`
  is the fix.
- **`air.preserve_shim_dma_order` is a GLOBAL order**, not per-channel. A drain
  placed early forces every later independent feed to queue behind it.
- **Two packet flows into one tile from the same side share a physical stream**,
  and a packet the destination has no credit for blocks everything behind it.
  `aie.packet_flow` says source and destination, not which stream channel the
  router picks; when both flows come from the south (a shim below a memtile
  below the core, as here) they are multiplexed. A CIRCUIT-switched channel gets
  its own. This is invisible in every AIR-level tool -- both sides balance, the
  volumes match, the descriptors are right, the locks are right, and the
  dispatch still hangs. It cost most of a day (§5.4 ix).
- **`getLockValuePair` counts users across `scf.if` arms** for memtile buffers;
  the L1 overload counts buffer *instances* instead. A prior theory blaming the
  latter for a level-2 failure was disproven by reading the source.
- **A launch-scope `scf.for` deadlocks the shim**, so launch-side feeds are
  Python-unrolled.
- **32- vs 64-byte L1 alignment**: a misaligned 512-bit access masks the
  address rather than faulting.
- **torch/transformers cannot share a process with an open XRT session** —
  immediate segfault (0xC0000005) after HF weight loading. All NPU+HF work uses
  separate processes handing off `.npz` files.
- **`build_template.sh` skips the Peano version-pin preflight**, and says so:
  its templates are good for dataflow work and **not for numerics claims**.
  Every `batch_equiv` / `spec_accept` number in this document rides on them.

## 6. Cost model — now measured on qwen3-4b

This section used to be arithmetic on qwen3-4b's batch-1 terms scaled by
**llama-3.2-1b's** batch-1→8 curve. It no longer is. Every row below is a
qwen3-4b dispatch timed on NPU2 `[hw]` — median of 25 after 5 warmup, p10/p90
within 0.5% of the median, `dispatch_time.py`, L=128, ctx 128.

| template | layers | LM head | batch | median ms |
|---|---|---|---|---|
| `nolm`      | 36 | no  | 1 | **50.010** |
| `decode`    | 36 | yes | 1 | **56.946** |
| `decode`    | 36 | no  | 8 | **159.572** |
| `withlm`    | 36 | yes | 8 | **177.511** |
| `nolmdraft` | 5  | no  | 1 | **7.383** |
| `draft`     | 5  | yes | 1 | **14.107** |
| `draft`     | 5  | no  | 8 | **22.472** |

`DECODE_NO_LM_WAVES` is gated on `BATCH > 1`, so a batch-1 template built with
it still runs the vocab waves; the head-free batch-1 rows use `UNI_WAVE_HI`
instead. Comparing a batch-1 template built the default way against a batch-8
one measures the head into the batch ratio and reads 2.80× where the body is
3.19×.

Solving the two layer counts against each other:

| term | batch 1 | batch 8 | scaling |
|---|---|---|---|
| per decoder layer | **1.375 ms** | **4.423 ms** | **3.22×** |
| fixed | 0.508 ms | 0.359 ms | — |
| LM head | **6.94 ms** | **17.94 ms** | 2.59× |
| decode body, 36L | 50.01 ms | 159.57 ms | 3.19× |
| decode body, 5L | 7.38 ms | 22.47 ms | 3.04× |

The fit is not over-determined by two points: the fixed term comes out at
0.36-0.51 ms at *both* batches, and the head is measured twice independently —
6.94 ms off the 36-layer pair and 6.72 ms off the 5-layer pair, the same tied
head, agreeing to 3%.

**qwen3-4b batches better than llama-3.2-1b**: 3.22× per layer against llama's
measured 3.74×, for 8× the tokens. The earlier §6 borrowed the llama number and
was therefore pessimistic by ~16% on the dominant term.

### What that means for DFlash

Baseline, plain autoregressive decode: **56.95 ms/token = 17.56 tok/s** `[hw]`.

One block-8 speculative step:

```
draft   5 layers @ batch 8 + tied head   22.47 + 17.94  =  40.41 ms
verify 36 layers @ batch 8 + tied head                  = 177.51 ms
                                                   step = 217.92 ms
```

- **Break-even is 3.83 accepted tokens per step** (217.92 / 56.95). Below that
  DFlash is slower than not doing it.
- **The marginal cost of one more slot is 20.98 ms = 0.368 of a baseline token
  step.** So slot *k* pays for itself iff `P(produced ≥ k) > 0.368`. That single
  number is the whole block-size question, and §3.1 now measures the other side
  of it.

  *This corrects an earlier 17.22 ms / 0.302 in this section.* That figure was
  `(177.51 − 56.95) / 7`, the marginal **verify** slot alone. Growing the block
  grows the *draft* pass too — it drafts the whole block — and that adds
  `(40.41 − 14.11) / 7` = 3.76 ms. The full per-slot cost is
  `17.22 + 3.76 = 20.98 ms`, and the step time is
  `step(B) = 71.05 + 20.98·(B−1)` ms, which reproduces the measured 217.9 at
  B=8. Understating the slot cost by 18% made every block size look better than
  it is; §3.1 uses the corrected number.

| accepted + bonus per step | ms/token | tok/s | vs baseline |
|---|---|---|---|
| 3.5 | 62.3 | 16.1 | 0.91× |
| 4.0 | 54.5 | 18.4 | 1.05× |
| 5.0 | 43.6 | 22.9 | 1.31× |
| 6.0 | 36.3 | 27.5 | 1.57× |
| 7.0 | 31.1 | 32.1 | 1.83× |

§3.1 supplies the distribution this table needs, so the band above collapses to
a number: a block-8 step is **1.24× on math and on code, 0.88× on chat** — the
opposite-verdict problem §7 item 5 names, now measured on both sides.

**This also reprices batch 16, downward, past 1.0×.** At 0.368 baseline steps
per slot, slots 9-16 pay only where `P(produced ≥ k) > 0.368`, and §3.1 measures
that curve at 0.20 falling to 0.06 across exactly that range. Block 16 comes out
at **0.90× / 0.89× / 0.57×** — slower than not speculating at all. §5.4 (iii)
shows what batch 16 costs to build; this says the return is negative, and it is
now measured rather than estimated.

**Batching helps the projection and does not help attention** `[hw]`, llama:
per-layer cost scales 3.74× for 8× the tokens (the `aie::mmul<8,8,8>` intrinsic
engages at batch ≥ 8; batch 1 runs a degenerate `rowA==1` path), while
attention scales **8.33×** — worse than linear, since each token still walks
its own keys. On a batch-8 llama dispatch the layer body is 65%, attention 16%,
LM head 16%. The equivalent qwen3-4b split has not been measured (it needs the
four-family decomposition `decode_cost.py` does, and a batched qwen driver);
what is measured here is the total, which is the term the cost model uses.

## 7. What would have to be true for this to be worth building

In order, cheapest first:

1. ~~Time qwen3-4b batch 8, target and drafter, on device.~~ **Done** — §6.
   `dispatch_time.py`, both models, head-free and with-head at both batches.
2. ~~Re-run the acceptance measurement keeping the per-block HISTOGRAM.~~
   **Done** — §3.1. Blocks 16, 8 and 4, 60 prompts each, distribution kept.
   It settled the block-size question outright, and against item 3.
3. ~~**Batch 16, if item 2 says so**~~ — **item 2 says no.** Block 16 is
   0.90×/0.89×/0.57× on math/code/chat: slower than not speculating. Slots 9-16
   return `P(produced ≥ k)` of 0.20 down to 0.06 against a 0.368 bar. Batch 8
   already works and is the best configuration that exists (1.24× on math and
   code), so **build the loop on batch 8**.

   The batch-16 wave-0 hang (§5.4 xiii) is therefore an unclaimed bug, not a
   blocker — worth recording, not worth chasing. Level 3 itself is kept: it is
   verified, byte-identical to levels 0 and 2 at 36 layers, and it costs 4.4%,
   but nothing needed today depends on it. Level 2 remains the default.

   What item 2 *did* open is smaller and better-defined: **block 4 is worth
   between 0.84× and 1.36×**, and both ends are pinned by things that can be
   settled. It needs a `size_C=32` de-tiling variant in
   `proj_qmm_mm_flush_row` (the batch set is currently {1, 8, 16, 24, 32} —
   `q4k_mm.h` is already bit-exact at batch 4, only the de-tiling is not), and
   then a `dispatch_time.py` run at batch 4 to replace the interpolated step
   time. One bounded kernel change and one timing run, against a possible +10%
   over batch 8 — or a clean negative.
4. **The loop itself.** Draft and verify are separate xclbins with different
   `UNI_DEC`; alternating them per block has a switching cost that has never
   been measured, and §6's 217.9 ms step assumes it is zero. §3.1 makes the
   stake sharp: at block 8, math produces 4.73 tokens per step, so the step can
   absorb `4.73 × 56.95 − 217.92` = **51 ms of swap before block 8 stops paying
   at all** (code: 53 ms).

   **§3.2 may have removed this term rather than measured it.** The mask is now
   selectable per dispatch from one device program, and the layer loop is a
   rolled `scf.for` whose device is wave-invariant — so draft and verify plausibly
   differ only in the wave range and the weight BO, both host-side, with no
   xclbin swap and no PDI reload to pay for. That is not yet demonstrated end to
   end (the drafter still needs §3.2's missing pieces before there are two
   passes to alternate), but the thing that looked like it forced two device
   programs no longer does.
5. **The workload question**, which no measurement can answer: at the block
   size that actually wins, math and code are 1.24× and chat is 0.88×. The two
   give opposite verdicts, and no amount of further measurement decides which
   one the deployment is.

## 8. Reproducing what is here

```bash
# qwen3-4b batch 1, end to end (the verified path)
cd programming_examples/llms/qwen3_4b_q4nx && make run

# qwen3-4b batch 8, target then drafter (both pass -- see 5.3b)
cd programming_examples/fused_decode
for M in qwen3-4b qwen3-4b-draft; do
  DECODE_MODEL=$M VOCAB_CHUNK_I2=30 DECODE_STACK=6080 ./build_template.sh 8 128
  DECODE_MODEL=$M VOCAB_CHUNK_I2=30 DECODE_STACK=6080 ./build_template.sh 1 128
  DECODE_MODEL=$M DECODE_STACK=6080 python3 batch_equiv.py \
      --model $M --vocab-chunk-i2 30 --batch 8 --L 128 --tokens 0
done

# is a template's layer output actually being written? (the 5.3 probe)
DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 python3 _x_hole.py

# RMS_BAND_STREAM level 2 -- verified, and byte-identical to level 0 at batch 8
DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 DECODE_STACK=6080 RMS_BAND_STREAM=2     ./build_template.sh 8 128
RMS_BAND_STREAM=2 DECODE_STACK=6080 python3 batch_equiv.py --model qwen3-4b     --vocab-chunk-i2 30 --batch 8 --L 128 --tokens 0

# level 3, and the knobs that bisect it (5.4). UNI_DEC_OVERRIDE=1 is the fast
# repro -- one layer, hangs identically, and both level 0 and level 2 pass there.
UNI_DEC_OVERRIDE=1 RMS_BAND_STREAM=3 ...      # one-layer repro
#
# THE HEAD-OF-LINE A/B (5.4 ix). One line apart, and the whole difference is
# whether @rmsX gets its own physical stream channel:
_RMS_X_PACKET=1                               # @rmsX back on the packet flow: HANGS
#                                             # (default at level 3 is circuit: PASSES)
#
# The bisection that got there. GET/PUT split the round-trip's two halves;
# _RES2_PUT_{FIRST,MID} is the progress witness that placed the stall:
_RMS_BANDS_GET=1  _RMS_BANDS_PUT=12           # residual2 puts back, never gets
_RMS_BANDS_GET=12 _RMS_BANDS_PUT=1            # residual2 gets, never puts back
_RMS_RES_ONLY=N                               # shorthand for setting both to N
_RMS_RES2_PUT_FIRST=1                         # witness ahead of BOTH gets
_RMS_RES2_PUT_MID=1                           # witness between the two gets
_RMS_RT_COMBINED=1                            # both residuals' bands in ONE shim task
_RMS_RES2_AS_W=1                              # residual2's bands, weight descriptor
_RMS_STG_COPY=1                               # release the @outY buffer before the band
_RMS_BAND_FIRST=1                             # take the @rmsX band before the @outY round
_RMS_H_SWAP=1                                 # h in X, layer output in the scratch
_RMS_{PH2,RES2_PUT,RES2_DRAIN}_PHASE=<p>      # which weight boundary arms each feed
_RMS_LATE_RT=1                                # every banded feed after the weight loop
_RMS_RT_PER_BAND=1                            # five put/drain pairs, not one folded pair
_RMS_FETCH_OFF=1                              # drop ph0/ph2's band sweeps entirely
_RMS_NO_FOLD=1                                # one shim put per band (1:1 packets)
_RMS_PH0_ONLY=1                               # feed ph0 and nothing else
_RMS_W_ON_X=1 RMS_BAND_STREAM=2               # weights on @rmsX, no banding
_ATTN_O_PERTOKEN=1                            # @attnO one put per token (5.4 xi)
_QKV_ROPE_PERTOKEN=1                          # @toRope one put per token (5.4 xii)
QKV_RELAY_COL=<c>                             # which column the QKV staging lands on
DOWN_PCOL=<c>                                 # ... and the down memtile (routing-pinned)

# every knob above changes the DESIGN, so check the emitted IR before believing
# a device result -- three runs in this work produced confident-looking wrong
# conclusions from a patch that silently did not apply or that changed a second
# thing (5.4 iv).
# what each shim channel really moves after lowering (catches a dropped
# descriptor dimension, which check_channel_balance.py cannot see):
python3 shim_volume.py --per-wave 36

# the emitted start/await ORDER. An await whose consumer has not been started
# yet is a deadlock, and the await positions are synthesized during lowering --
# they are in no Python file. This is how (vi) and (viii) were found:
python3 shim_schedule.py --channels rmsX layerOut inW0c0

# what is on each tile against the three ceilings that bite at high batch --
# 512 KB of L2, 24 BD ids per memtile channel (MM2S and S2MM SHARE the pool),
# and 64 KB of L1 including the stack and the per-herd RTP word. mlir-aie
# reports each as a bare error naming at most one buffer:
python3 tile_budget.py --stack 6080

# batch 16 (needs level 3 and the two knobs its ceilings forced):
QKV_RELAY_COL=2 DECODE_STACK=6016 RMS_BAND_STREAM=3 ./build_template.sh 16 128

# the section 6 timings. Seven templates; RENAME each as it is built --
# build_template.sh always writes decode_b<B>_L<N>. UNI_WAVE_HI, not
# DECODE_NO_LM_WAVES, is what drops the head at batch 1 (see 6).
cd programming_examples/fused_decode
E="VOCAB_CHUNK_I2=30 DECODE_STACK=6080 RMS_BAND_STREAM=0"
env $E DECODE_MODEL=qwen3-4b       UNI_WAVE_HI=36        ./build_template.sh 1 128  # nolm
env $E DECODE_MODEL=qwen3-4b                             ./build_template.sh 1 128  # decode b1 (+head)
env $E DECODE_MODEL=qwen3-4b                             ./build_template.sh 8 128  # decode b8
env $E DECODE_MODEL=qwen3-4b       DECODE_NO_LM_WAVES=0  ./build_template.sh 8 128  # withlm
env $E DECODE_MODEL=qwen3-4b-draft UNI_WAVE_HI=5         ./build_template.sh 1 128  # nolmdraft
env $E DECODE_MODEL=qwen3-4b-draft                       ./build_template.sh 1 128  # draft b1 (+head)
env $E DECODE_MODEL=qwen3-4b-draft                       ./build_template.sh 8 128  # draft b8
DECODE_STACK=6080 python3 dispatch_time.py --model qwen3-4b --vocab-chunk-i2 30 \
    --batches 1 8 --L 128 --prefix decode

# acceptance rate, real upstream code, real datasets (CPU, no NPU)
cd programming_examples/llms/qwen3_4b_q4nx
python3 dflash_phase2_upstream_sweep_large.py           # the means (section 3)

# the DISTRIBUTION (section 3.1), which is what decides the block size. ~20 min
# per block size on CPU. Writes the raw per-block lengths after every prompt, so
# it can be killed and the partial result is still analyzable:
python3 dflash_acceptance_hist.py --n 20 --block 16 --out dflash_acceptance_hist.json
python3 dflash_acceptance_hist.py --n 20 --block  8 --out dflash_acceptance_hist_b8.json
python3 dflash_acceptance_hist.py --n 20 --block  4 --out dflash_acceptance_hist_b4.json
python3 dflash_acceptance_hist.py --analyze dflash_acceptance_hist.json   # no models loaded

# the drafter's two non-decode projections, on device (section 3.3, 3.4).
# All of these need the DRAFTER checkpoint, and the target's for the tied head.
python3 dflash_draft_decomp.py        # the two structural claims, CPU
python3 qwen3_4b_draft_weights.py     # weight-reader self-check
python3 dflash_fc_gate.py             # fc, bf16, 2 launches
python3 dflash_ctxkv_gate.py --split  # context k/v + k_norm, bf16, 15 launches
python3 dflash_int4.py                # AWQ round-trip on the real fc, CPU
python3 dflash_int4_fc_gate.py        # fc, int4, 4 launches   (--synthetic: no ckpt)
python3 dflash_ctxkv_int4_gate.py     # context k/v + k_norm + RoPE, int4, 20 launches
python3 dflash_draft_prepass_gate.py  # BOTH halves in one func, 24 launches
# against the REAL drafter (section 3.6). Two processes: torch and XRT segfault
# together, so the oracle is a dumper.
python3 dflash_draft_oracle.py --block 0 --block-size 8
python3 dflash_prepass_oracle_gate.py

# the WHOLE draft pass (section 3.7). The drafter's templates are a different
# model at the same context length, so they carry their own prefix:
cd ../../fused_decode
DECODE_MODEL=qwen3-4b-draft VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 DECODE_STACK=6080 \
  DECODE_NO_LM_WAVES=0 DECODE_MASK_BIDIR=1 ./build_template.sh 8 16   # and 15
for L in 15 16; do for e in xclbin insts.bin; do \
  mv decode_b8_L$L.$e ../llms/qwen3_4b_q4nx/draft_b8_L$L.$e; done; done
cd -
python3 dflash_draft_gate.py                 # pre-pass -> KV seed -> draft
python3 dflash_draft_gate.py --seed-oracle   # attribution, not a gate

# the VERIFY pass on the shipping driver (section 3.5). Needs batch-8 templates
# in this directory first. The model's own Makefile builds them with the
# -DPROJ_MM_BATCH the batched projection needs and restores the batch-1 objects
# afterwards -- but it also runs the Peano pin preflight, which this sandbox
# cannot satisfy (the pinned nightly is gone from the index):
make compile-decode-batch DBATCH=8 LBUILD=16
# ... so the numbers in 3.5 came from build_template.sh, which skips that
# preflight deliberately. That is only defensible because the skip was CHECKED:
# a batch-1 pair built this way is bit-identical to the shipping Makefile one.
cd ../../fused_decode
DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 DECODE_STACK=6080 \
  DECODE_NO_LM_WAVES=0 ./build_template.sh 8 16     # and again at 15, for the slope
cp decode_b8_L1{5,6}.{xclbin,insts.bin} ../llms/qwen3_4b_q4nx/
cd -
python3 dflash_verify_gate.py         # batch 8 vs eight batch-1 steps

# THE LOOP (section 3.8). It needs a TAPS target -- one dispatch has to return
# both the distributions and the drafter's input -- and both families at a
# context length a real generation reaches. build_template.sh names a
# DECODE_HIDDEN_TAPS build `taps_b<B>_L<N>` on its own, and the drafter's
# decode_b8_* have to be RENAMED on the way over or the target's scan finds
# them:
cd ../../fused_decode
export DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 W_DUAL_CHAN=1 DECODE_STACK=6080 \
       DECODE_NO_LM_WAVES=0
DECODE_HIDDEN_TAPS=1 ./build_template.sh 8 128    # and 127 -> taps_b8_L*
DECODE_MODEL=qwen3-4b-draft DECODE_MASK_BIDIR=1 ./build_template.sh 8 128  # +127
cd - && make _decode_kernels_only DBATCH=1        # ALWAYS, after any batched build
python3 dflash_loop.py --no-spec --n-tokens 10    # must emit PARIS_GREEDY 10/10
python3 dflash_loop.py --n-tokens 32              # the loop, with acceptance

# IS THE ENGINE CAUSAL? (section 3.8). One dispatch pair per L, no drafter, and
# it fails on the SHIPPING batch-1 template:
python3 dflash_causal_probe.py --prefix decode_b1_L --split
```

Three traps in that last group, each of which cost real time:
`shared.infra.external_kernels.compile_mv_int4_bf16` builds the **GEMV** and the
object links and runs inside the GEMM; `XRTBackend` needs `stack_size=16384` or
the same module returns rel ≈ 0.85 with no diagnostic; and multi-launch needs
`output_format="elf"` — on xclbin the launches' instruction streams collide with
`edge 'air.insts.bin' produced duplicate output path`, then a bare
`pipeline failed` many stages later while every intermediate `.ll` compiles by
hand.

The `--block 8` and `--block 4` runs are the reason the block-8 number is a
measurement and not an extrapolation: they say the b16 checkpoint runs at
smaller blocks without retraining, and that truncating a block-16 histogram
predicts them to within 4%.

`build_template.sh` **always** writes `decode_b<B>_L<N>` and the gates read
whatever is in the directory. Rename or rebuild deliberately: a llama template
left in place will be dispatched against qwen3-4b geometry and produce
plausible, meaningless numbers. This happened during the work above.

## 9. Corrections — claims removed in this rewrite

Recorded so they are not re-derived from the old text.

- **"Level 2 verified on real qwen3-4b hardware at `DECODE_BATCH=8`, 8/8
  accepted."** False, three ways. The source passage names no model and is
  llama-3.2-1b (it quotes K=2048 BD shapes; `spec_accept.py` is llama-only);
  it mis-quoted the *level 3* heading as level 2; and qwen3-4b at batch 8 was
  refused by the builder at the time, a fact the same document stated
  elsewhere. This claim was used to argue the next step was low-risk.
- **"Level 3 built and IR-correct, but does not compile."** The "IR-correct"
  half is disproven — level 3 loses L-dependence (§5.4).
- **"Level 3 is batch-16-only."** It is not; level 3 exists to give qwen3-4b
  block 8, and the `air-to-aie` failure reproduces with regen counts forced
  to 0.
- **"At batch 8 on qwen3-4b the emitted IR contains no batch-1 projection call
  at all [measured]."** Cannot be true: qwen3-4b at batch 8 exited before
  emitting IR until the change in §5.2. Same for the family of batch-8
  qwen3-4b sizing claims around it — those are legitimate *sizing rules*
  `[static]`, not evidence of a build.
- **"54304 B of 55296 at batch 8, so `BATCH_MAX_RMS` is exactly 8"** for
  qwen3-4b. Stale: that count charged one resident norm weight where the body
  holds two. Corrected count is 59,424 B and the ceiling is 7.
- **Attention is 69-71% of a batched dispatch `[measured]`.** It was a
  roofline, mis-tagged as measured, and it is wrong: on device it is **16%** of
  a batch-8 llama dispatch. The conclusion it produced — "making the
  projections faster cannot move this much" — is backwards.
- **Four mutually inconsistent break-even τ values** (1.23 traffic-only; 4.31/
  4.65 with the draft not paying its LM head; 5.28/9.59 llama; 4.81/8.90
  qwen3-4b). Only the last is qwen3-4b's, and earlier text applied llama's 5.28
  to qwen3-4b's baseline.
- **tok/s estimates of ~30.5 and "~2.0x"**. They divided by *llama's* 198.6 ms
  iteration while comparing against *qwen3-4b's* 14.99 tok/s baseline. §6
  supersedes them, with the opposite conclusion.
- **"No network access" / "no inference driver in `llms/qwen3_4b_q4nx/`" /
  "HIDDEN_TAPS not verified" / "no numeric gate for the batched kernel".** All
  overtaken by later work in the same document.

The pre-rewrite text is in git history; it is a log of how the work went, not a
record of what is true.

## References

[arXiv:2602.06036](https://arxiv.org/abs/2602.06036) ·
[z-lab/dflash](https://github.com/z-lab/dflash) ·
[Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)

In tree: [`fused_decode/`](../programming_examples/fused_decode/),
[`llms/qwen3_4b_q4nx/`](../programming_examples/llms/qwen3_4b_q4nx/),
[`llms/llama32_1b_q4nx/`](../programming_examples/llms/llama32_1b_q4nx/)
(where every batched measurement before this rewrite was taken).
