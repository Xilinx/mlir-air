# SmolVLA NPU vision encoder — complete performance breakdown

**Question.** Where does the shipping NPU vision encoder's ~465 ms (3 images)
actually go, and what is worth optimising next?

**Headline.** The encoder is **NPU-bound, not host-bound**: **408 ms of the
471 ms wall (86.6%) is the NPU executing.** The A3-6b fusion did its job — host
glue inside the dispatch loop is down to **10.4 ms (2.2%)**. Consequently:

- **H1 (fused ELFs cost extra device time) is REFUTED for vision.** The two
  fused ViT ELFs cost **1.01–1.03×** the sum of their launches measured
  individually *in the same regime* — not the **1.39–1.79×** the backbone's
  fused ELFs show. Vision fusion is ~free.
- **H2 (compromise tiles inside the fused ELFs) is largely REFUTED.** In the ELF
  regime the deployment actually runs in, `tile_n` barely matters: **0%** on
  q/k/v/o, **2.5%** on fc2, **5.6%** on fc1 — ~4 ms total, nothing like the
  connector's 1.33×.
- **The real gap is somewhere neither hypothesis looked.** The identical GEMM
  module lowered to a multi-launch **ELF** (what the deployment must use) runs
  **1.35–1.94× slower** than lowered to **xclbin** (what every kernel-registry
  `drain` throughput number was measured on). That is compiler/runtime work, not
  kernel tuning, and §8 ranks it accordingly.

Every number below was measured on real NPU2 hardware in one session
(2026-07-27/28). Nothing is carried over from a previous run or from the kernel
registry, except where a registry row is explicitly quoted for contrast.

> **Where these numbers come from.** `make profile` and `make run` reproduce the
> end-to-end and per-image figures. The per-ELF and per-kernel decompositions in
> §1–§8 were produced by one-off harnesses cited by name throughout; those are
> development tools and are not part of the shipping example.

---

## 0. Machine state

Verified before measuring, and unchanged throughout:

| knob | value |
|---|---|
| CPU scaling governor (all 24 threads) | `performance` |
| CPU energy_performance_preference (all 24) | `performance` |
| NPU power mode (`xrt-smi examine -r platform`) | **Turbo** |
| CPU | AMD Ryzen AI 9 HX 370, 12C/24T |
| NPU | NPU2 / Strix (AIE2P), device `0000:c6:00.1` |

Configuration profiled: the shipping one — `smolvla_runtime.VisionRuntime`,
single process, NPU vision + CPU backbone
(`run_hybrid_forward(npu_vision=True)`),
3 camera images per inference, 12-layer SigLIP ViT at seq=1024 + connector.

**Measurement noise.** Process-to-process spread on this machine is **±10–15%**
for a fixed kernel. Every A/B below is therefore run **interleaved round-robin
inside one process**; where that was impossible the spread is quoted.

---

## 1. Per-ELF device breakdown

`scripts/vision_profile_run.py` — the deployed `VisionRuntime.encode` path with
`shared.infra.cache.Profiler` enabled: 5 timed `encode([img,img,img])` calls
after 2 warmups = **570 real dispatches**. Averages per invocation.

| ELF | inv/image | BO write | **NPU run** | BO read | driver total | device/image | share of device |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vit_o_ffn` (10 launches) | 12 | 0.210 ms | **5.504 ms** | 0.032 ms | 5.816 ms | **66.05 ms** | **48.6%** |
| `vit_ln_qkv` (7 launches) | 12 | 0.111 ms | **2.901 ms** | 0.070 ms | 3.158 ms | **34.81 ms** | **25.6%** |
| `flash_attn` (1 launch) | 12 | 0.328 ms | **2.798 ms** | 0.028 ms | 3.219 ms | **33.58 ms** | **24.7%** |
| `gemm_connector` | 1 | 0.090 ms | 0.951 ms | 0.009 ms | 1.128 ms | 0.95 ms | 0.7% |
| `layer_norm` (post_ln) | 1 | 0.109 ms | 0.628 ms | 0.021 ms | 0.814 ms | 0.63 ms | 0.5% |
| **TOTAL** | **38** | | | | | **136.02 ms** | 100% |

Per 3-image encode: **BO write 23.9 ms (5%) · NPU run 408.1 ms (93%) · BO read
4.8 ms (1%)**.

Three ELFs carry 98.9% of the device time; the post-`layer_norm` and the
connector are rounding errors.

`make profile` reports the same `device/image` column and can be re-run at any
time. On the current toolchain (LLVM 23.0.0.2026071405, mlir-aie db06374d) it
reads 65.40 / 36.15 / 36.77 / 1.12 / 0.73, total **140.18 ms** against the
136.02 above — the shape of the breakdown is unchanged, but individual ELFs
moved by up to 10% across the rebuild, and `flash_attn` and `vit_ln_qkv` swapped
places. Treat the absolute figures in §2–§7 as of their measurement date.

---

## 2. Within-ELF breakdown

### 2.1 Why the obvious method does not work

Rebuilding each sub-kernel as its own **single-launch ELF** — the natural
"standalone at the same shape" reference — gives numbers that are both slow and
*wrong*. A single-launch, multi-M-iteration drain GEMM returns correct output on
the first invocation and garbage in the first half of its output rows on every
one after:

```
M=1024 (4 M-launch iterations), 1024x768x768 drain, single-launch ELF:
  call 0: mean_rel_L1=9.601e-03   bad 32-row blocks  0/32
  call 1: mean_rel_L1=2.087e-01   bad 32-row blocks 16/32   [0,1,2,3,4,5...]
M=256  (1 M-launch iteration): correct on every call.
```

Not a buffer-reuse artifact (reproduced with every buffer re-uploaded and with
the `xrt.run` object reused) and not `stack_size`. Worth filing against the
AIRRt→NPU multi-iteration launch lowering.

**The shipping ELFs are unaffected**: they are multi-launch (7 and 10 launches),
and `make verify` passes. Only the single-launch form corrupts — which is why
§2.2 measures with replicated multi-launch ELFs instead. Truncated prefixes of
the real fused ELF were the other candidate and hang (`ERT_CMD_STATE_TIMEOUT`)
at some prefix lengths.

### 2.2 The method that does work: replicated-launch ELFs

`scripts/vision_replicate_bench.py` builds, for each sub-kernel, two ELFs
containing **2 and 4 identical copies** of that launch (each writing its own
output arg), then

```
cost_per_launch = (t4 - t2) / 2      # fixed ELF overhead cancels
```

Both ELFs are multi-launch and even-length — the exact regime the shipping ELFs
run in. Confirmed correct (`mean_rel_L1` 9.56e-3 / 9.58e-3 / 9.60e-3). The three
GEMM numbers below come from the tighter interleaved A/B of §6 (25 rounds,
p25–p75 within 1%); the rest from the x2/x4 slope.

| sub-kernel | shape | **in-situ device cost** | notes |
|---|---|---:|---|
| affine LayerNorm | 1024×768 | **387.6 µs** | ×2 per layer (LN1, LN2) |
| GEMM q/k/v/o | 1024×768×768, tn96 | **578.5 µs** | ×4 per layer |
| GEMM fc1 | 1024×768×3072, tn128 | **1461.2 µs** | |
| GEMM fc2 | 1024×3072×768, tn96 | **1142.6 µs** | |
| bias-add (broadcast) | 1024×768 | **147.3 µs** | ×4 per layer |
| bias-add (broadcast) | 1024×3072 | **363.3 µs** | fc1 bias |
| residual add | 1024×768 | **174.9 µs** | ×2 per layer |
| GELU-tanh | 1024×3072 | **576.4 µs** | |
| FlashAttention | 1024/1024, 12 heads, hd 64 | **2798 µs** | own ELF, §1 |

### 2.3 Σ parts vs the fused ELF — the H1 test

| group | Σ of its launches | fused, measured **isolated** | fused, measured **in the encoder** | Σ→isolated | Σ→deployed |
|---|---:|---:|---:|---:|---:|
| `vit_ln_qkv` = LN + 3×GEMM + 3×bias | **2565 µs** | 2603 µs | 2901 µs | **1.015×** | 1.131× |
| `vit_o_ffn` = 10 launches | **5154 µs** | 5281 µs | 5504 µs | **1.025×** | 1.068× |

*(the "isolated" column is the full 7-/10-launch prefix ELF measured alone,
`scripts/vision_prefix_bench.py`.)*

> **H1 VERDICT: REFUTED for the vision ELFs.** Fusing 7 and 10 launches into one
> ELF costs **1.5–2.5%** of device time, not the 39–79% the backbone's
> `rms_gemms_rope` / `o_ffn` pay. Vision fusion bought its 368→141 ms *and* is
> essentially free in device time. The remaining 4–11% (isolated → deployed)
> is **hw-context interleaving**, not fusion: in the encoder the three ELFs
> alternate every layer, and each alternation costs device time
> (independently visible: alternating an ELF context with an xclbin context in
> one process inflated *both* by 1.4–2.5× — §6).

### 2.4 Per-layer device time by op class

Per layer (12 per image, 36 per inference), from §2.2:

| bucket | µs/layer | share |
|---|---:|---:|
| GEMM (4× qkvo + fc1 + fc2) | **4917.8** | 43.9% |
| FlashAttention | **2798.0** | 25.0% |
| bias-adds (5× 768 + 1× 3072) | 1099.8 | 9.8% |
| LayerNorm ×2 | 775.2 | 6.9% |
| GELU | 576.4 | 5.1% |
| residual adds ×2 | 349.8 | 3.1% |
| fusion + context-interleave overhead (residual) | 686.0 | 6.1% |
| **total (= 2901 + 2798 + 5504)** | **11203** | 100% |

**Only 44% of the encoder's device time is matmul.** A quarter is attention;
another quarter is elementwise/norm launches moving 3–12 MB each.

---

## 3. Host-side accounting

`scripts/vision_host_bench.py`, every op at the exact shape/dtype the deployed
path uses, median of 30 (15 for im2col), machine idle.

| host op | each | ×/encode | total |
|---|---:|---:|---:|
| `im2col_patch_embed` (all threads) | 4.220 ms | 3 | **12.66 ms** |
| patch_embed f32→bf16 (1024×768) | 0.211 ms | 3 | 0.63 ms |
| connector A f32→bf16 (64×12288) | 0.210 ms | 3 | 0.63 ms |
| `np.zeros` (1024,768) bf16 — FA out | 0.010 ms | 36 | 0.37 ms |
| `pixel_shuffle` (1024,768)→(64,12288) | 0.101 ms | 3 | 0.30 ms |
| post_ln bf16→f32 (1024×768) | 0.038 ms | 3 | 0.11 ms |
| `ascontiguousarray` on bf16 (1024,768) | <0.001 ms | 216 | 0.08 ms |
| `np.zeros` (1024,768) bf16 — LN out | 0.010 ms | 3 | 0.03 ms |
| `_to_chw_f32` (torch 1×3×512×512) | 0.003 ms | 3 | 0.01 ms |
| LN param `np.concatenate` (2×768) | 0.001 ms | 3 | <0.01 ms |
| connector B `ascontiguousarray` (12288,960) | <0.001 ms | 3 | <0.01 ms |
| **TOTAL itemised host** | | | **14.84 ms** |

For reference: `im2col_patch_embed` at **1 thread** cost 9.455 ms instead of
4.220 ms, which is why it ran outside the host BLAS clamp this session measured.
That clamp has since been removed — ten later runs found it made no difference
either way (442.5 vs 441.1 ms).

Note: the connector weight (12288×960 bf16, 23.6 MB) is **not** copied per call
— it is already contiguous bf16, so `ascontiguousarray` is a no-op. Same for
every per-layer weight (they are `static_input_indices` + cached in
`run_vit_block_fused._arg_cache`).

---

## 4. Full reconciliation

Median of 5 timed `encode([img,img,img])` calls; individual walls
**464.2 / 455.9 / 470.9 / 485.0 / 472.4 ms** (median **470.9**, spread ±3%).
This reproduces the reported ~465 ms.

```
encode() wall                                470.92 ms   (3 images, 157.0 ms/image)
├─ im2col + _to_chw_f32  (host, pre-loop)     15.79 ms    3.4%
└─ dispatch loop                             455.20 ms   96.7%
   ├─ driver calls (114 = 38 x 3)            444.78 ms   94.4%
   │  ├─ NPU run (device)                    408.05 ms   86.6%   <-- the workload
   │  ├─ BO write (host->DDR)                 23.93 ms    5.1%
   │  ├─ BO read (zero-copy view)              4.78 ms    1.0%
   │  └─ filelock + xrt.run + set_arg          8.01 ms    1.7%   (70 us/dispatch)
   └─ host glue inside the loop                10.42 ms    2.2%
      ├─ itemised numpy (§3, in-loop rows)      2.15 ms
      └─ UNATTRIBUTED                           8.27 ms    1.8%  <-- residual
```

**Residual: 8.27 ms of 470.92 ms = 1.8%.** It is Python interpreter overhead
spread over 114 `load_and_run` calls (≈73 µs/call: argument list slicing, the
`_arg_cache` dict lookups, tuple/`np.frombuffer` construction per output, and —
in this instrumented run — the profiler's own bookkeeping). It is **not**
further attributed, and it is not claimed to be anything else.

The top-level split (`wall = im2col + loop`, `loop = driver + glue`,
`driver = write + run + read + lock`) is exact by construction; the only
genuinely unexplained term is the 8.27 ms above.

Cross-check: §3's itemised host total (14.84 ms) vs the runtime's own
`t_im2col_ms` (15.79 ms) — the 0.95 ms difference is the list comprehension and
`_to_chw_f32` dispatch inside `encode`. Consistent.

---

## 5. Efficiency vs the kernels' own capability

The kernel-registry `drain` throughput numbers are measured on the **xclbin**
path (`matrix_multiplication/bf16_in_bf16_out/run.py:1156` —
`use_elf = ... and method == "fused-cast"`). The vision deployment must use
**ELF** (the multi-launch fused modules cannot be lowered to xclbin). Both
measured this session:

| GEMM | deployed, in-situ ELF | best standalone (xclbin harness, this session) | ratio | registry row (for reference) |
|---|---|---|---:|---|
| q/k/v/o 1024×768×768 | 578.5 µs / **2088 GFLOP/s** | 297.5 µs / **4060 GFLOP/s** | **1.94×** | 3798 |
| fc1 1024×768×3072 | 1461.2 µs / **3307 GFLOP/s** | 1085.1 µs / **4453 GFLOP/s** | **1.35×** | 4195 |
| fc2 1024×3072×768 | 1142.6 µs / **4229 GFLOP/s** | 848.3 µs / **5696 GFLOP/s** | **1.35×** | 5790 |

All three xclbin runs PASS at `mean_rel_L1` 9.42–9.47e-3.

**The biggest gap is the q/k/v/o projection — 1.94×** — and it is also the
biggest absolute term (4 launches/layer). Closing all three would take the
per-layer GEMM cost from 4917.8 µs to 3123.4 µs.

### Is it the *format* or the *fusion*?

Same module, same driver, same tiles, **separate processes** (interleaving the
two formats penalises both — measured):

| shape | ELF | xclbin | ELF penalty |
|---|---:|---:|---:|
| 1024×768×768 | 620.2 µs | 435.3 µs | **1.42×** |
| 1024×768×3072 | 1962.2 µs | 1555.0 µs | **1.26×** |
| 1024×3072×768 | 1501.1 µs | 1140.1 µs | **1.32×** |

So a large part of the gap is the **ELF lowering path itself**, independent of
fusion, of tiling, and of the driver. (The absolute xclbin numbers here are
worse than the harness's because `KernelCache.load_and_run` builds a fresh run
handle per call and several hw contexts are live; the *ratio* is the point.)

### The non-GEMM launches

| op | bytes moved | in-situ | effective bandwidth |
|---|---:|---:|---:|
| bias-add 1024×3072 | 12.6 MB | 363.3 µs | **34.7 GB/s** |
| GELU 1024×3072 | 12.6 MB | 576.4 µs | 21.8 GB/s |
| residual add 1024×768 | 4.7 MB | 174.9 µs | 27.0 GB/s |
| bias-add 1024×768 | 3.1 MB | 147.3 µs | 21.4 GB/s |
| **affine LayerNorm 1024×768** | 3.1 MB | 387.6 µs | **8.1 GB/s** |

**The affine LayerNorm is 4.3× off the best bandwidth this fabric shows on the
same-sized traffic** and runs twice per layer. That is the clearest
single-kernel inefficiency in the encoder after the GEMM format gap.

### FlashAttention

Deployed: **2798 µs/layer = 1151 GFLOP/s** (3.22 GFLOP of useful work), i.e.
**3.7× below** what the same fabric gets on fc2. The identical configuration
built standalone through the FA harness measured **2919 µs** — so the
deployment is already at (slightly below) the kernel's standalone speed and
**there is no pipeline penalty to recover**. Any gain here needs kernel work.
*(Caveat: that standalone FA run's correctness gate failed — the harness builds
its own `attn_npu2.o` and its reference did not match; the number is quoted only
as an instruction-stream timing. The deployed FA is validated end to end by
`make verify`.)*

---

## 6. H2 — are the tiles inside the fused ELFs registry-optimal?

`scripts/vision_tile_ab.py`: the `_x4` replicate ELF at each candidate `tile_n`,
all variants driven **round-robin in one process**, 25 rounds, median.
Interleaving is essential — measured sequentially, the same configs disagreed by
up to 15% run-to-run and produced spurious 15% "wins".

**q/k/v/o 1024×768×768** (deployed `tile_n=96`):

| tile_n | per-launch | GFLOP/s | vs deployed | mean_rel_L1 |
|---:|---:|---:|---:|---|
| **96** (deployed) | 578.5 µs | 2088 | 1.000× | 9.560e-3 |
| 64 | 575.9 µs | 2098 | 1.005× | 9.560e-3 |
| 48 | 578.3 µs | 2089 | 1.000× | 9.560e-3 |
| 192 | 578.2 µs | 2089 | 1.001× | 9.560e-3 |

**fc1 1024×768×3072** (deployed `tile_n=128`):

| tile_n | per-launch | GFLOP/s | vs deployed |
|---:|---:|---:|---:|
| **128** (deployed) | 1461.2 µs | 3307 | 1.000× |
| 192 | 1458.3 µs | 3313 | 1.002× |
| **256** | **1383.6 µs** | 3492 | **1.056×** |
| **64** | **1383.2 µs** | 3493 | **1.056×** |

**fc2 1024×3072×768** (deployed `tile_n=96`):

| tile_n | per-launch | GFLOP/s | vs deployed |
|---:|---:|---:|---:|
| **96** (deployed) | 1142.6 µs | 4229 | 1.000× |
| **192** | **1114.7 µs** | 4335 | **1.025×** |
| 64 | 1118.5 µs | 4320 | 1.021× |
| 48 | 1165.0 µs | 4148 | 0.981× |

All variants produce **bit-identical accuracy** (`mean_rel_L1` 9.560e-3 /
9.577e-3 / 9.601e-3 by shape, independent of `tile_n`).

> **H2 VERDICT: largely REFUTED.** The fused ELFs' tiles are not a compromise.
> `tile_n` is **flat within 0.5%** on the q/k/v/o GEMM (the one that runs 4×
> per layer), and the best retune available is **5.6% on fc1** and **2.5% on
> fc2** — together **≈4 ms of the 465 ms**. This is not the connector case: the
> connector's 1.33× came from a weight-DMA-bound 64×12288×960 shape where N-tile
> count dominates; the encoder's GEMMs are M=1024 and are not in that regime.
>
> A related caveat worth recording: the **ELF-regime optimum is not the
> xclbin-regime optimum**. `qkvo tile_n=64` measures 384.1 µs on the xclbin
> harness (worse than tn96's 297.5) but is a dead heat in ELF; `fc2 tile_n=48`
> is 1273.5 µs on xclbin (much worse than tn96's 848.3) but only 2% worse in
> ELF. Tiles tuned on the harness do not transfer rank-for-rank to the
> deployment.

---

## 7. Where the 465 ms goes — one table

Per inference (3 images), from §1–§4:

| bucket | ms | share of wall |
|---|---:|---:|
| GEMM on device (4× qkvo + fc1 + fc2, ×36 layer-instances) | 177.0 | 37.6% |
| FlashAttention on device (×36) | 100.7 | 21.4% |
| bias-adds on device (×36) | 39.6 | 8.4% |
| LayerNorm on device (2×36 + 3 post_ln) | 29.8 | 6.3% |
| GELU on device (×36) | 20.8 | 4.4% |
| residual adds on device (×36) | 12.6 | 2.7% |
| connector GEMM on device (×3) | 2.9 | 0.6% |
| fusion + hw-context interleave overhead | 24.7 | 5.2% |
| BO write (host→DDR) | 23.9 | 5.1% |
| host im2col patch-embed + to_chw | 15.8 | 3.4% |
| filelock + `xrt.run` + `set_arg` | 8.0 | 1.7% |
| **unattributed Python in the loop** | **8.3** | **1.8%** |
| BO read (zero-copy view) | 4.8 | 1.0% |
| other itemised host numpy | 2.2 | 0.5% |
| **TOTAL** | **471.0** | (vs 470.9 measured) |

This closes to **0.01%** because the §2.4 "fusion + interleave overhead" row is
*derived* as the difference between the measured fused-ELF times and the sum of
the in-situ launch costs — it is a residual, not an independent measurement, and
it absorbs the difference by construction. The genuinely unexplained term in
this study remains the **8.3 ms of Python** in §4.

---

## 8. Where the remaining time could come from

Directions, not budgets. Each one was sized during this study, but those figures
came from harnesses run on the previous toolchain (LLVM 23.0.0.2026060107,
mlir-aie 92e9475c) and the absolute numbers no longer hold — a rebuild moved the
per-ELF device times by up to 10% on its own, enough to reorder the smaller
items. Re-measure before committing to any of them. Ordered by confidence and
tractability rather than by size.

| # | opportunity | why it should pay | effort |
|---|---|---|---|
| **1** | **Fold the bias-adds and residual adds into the GEMM drain epilogue.** Five bias-adds and two residual adds per layer are pure launch plus DDR traffic, and the drain GEMM already carries an epilogue cast. | the largest block of device time that does no arithmetic worth a launch | medium — builder work; the primitives exist |
| **2** | **Close the ELF-vs-xclbin GEMM gap** (§5). The same module at the same tiles runs 1.35–1.94× slower purely from the output-format lowering. | the single biggest gap between what the kernels can do and what the deployment gets | compiler/runtime (AIRRt→NPU ELF path); unscoped |
| **3** | **Merge `flash_attn` into the fused ELFs** → one ELF per layer instead of three. H1 (§2.3) found fusion is ~free for vision, and the isolated-vs-deployed delta is hw-context interleaving. | removes an interleave boundary per layer | medium; H1 supports it |
| **4** | **Speed up the affine LayerNorm** — 8.1 GB/s against 34.7 GB/s for a same-sized bias-add (§5), and it runs twice per layer. | a bandwidth gap that large is usually a kernel problem, not a shape problem | kernel tuning, bounded |
| **5** | **Re-tile fc1 and fc2** (§6). Accuracy is unchanged by it. | small: H2 found `tile_n` barely matters inside a fused ELF | trivial — two constants |
| — | ~~Re-tile the q/k/v/o GEMM~~ | flat within 0.5% across `tile_n` 48/64/96/192 | — |
| — | ~~Cut host glue / dispatch overhead~~ | host is already ~7% of the vision stage and most of it is irreducible; the fusion work in §3 spent this lever | — |
| (6) | **FlashAttention** is the largest single consumer after the GEMMs and already runs at its standalone speed, so no *pipeline* gain exists — only a faster kernel at this shape would help. | the biggest remaining target, and the only one needing a kernel rewrite | large |

Opportunities 1 and 2 partially overlap: 1 removes launches that 2 would also
speed up. 3, 4 and 5 are independent.

The one direction this study ranked highly and that has since been **retired**:
clamping the host BLAS pool around `im2col`. It measured well at the time
(−32.6 ms) but against a code path that had a clamp to extend; that clamp was
later removed after ten runs found no difference either way (442.5 vs 441.1 ms).

---

## 9. Reproduce

```bash
cd programming_examples/llms/smolvla

make profile   # CPU and NPU arms interleaved in one process (wrap in the NPU lock)
make run       # one forward; prints t_im2col / t_encode / per-image timings
make verify    # the correctness gate (cosine >= 0.99, nMSE <= 0.04)
```

`make profile` warms both arms with a discarded forward, then runs `REPS`
(default 5) interleaved CPU/NPU pairs and reports the median; `REPS=10` tightens
the estimate. The per-run spread quoted above applies to any single reading.

The §1–§8 decompositions are not reproducible from this directory — see the note
at the top.

## Appendix — traps found while measuring

1. **A single-launch multi-M-iteration drain GEMM lowered to ELF silently
   corrupts after the first invocation** (§2.1). It compiles, runs, returns no
   error, and half its output rows are stale. Reproducer committed. Any future
   "measure this kernel standalone" work through `KernelCache` must use a
   multi-launch ELF or the xclbin path.
2. **The registry's `drain` GFLOP/s are xclbin-path numbers** and overstate what
   the ELF-based deployments actually get by 1.26–1.42× (§5). They are still the
   right numbers for *choosing* a method/tile, but not for predicting deployed
   latency.
3. **The ELF-regime tile optimum ≠ the xclbin-regime tile optimum** (§6). Two
   tilings that are 1.3–1.5× apart on the harness are within 2% in ELF.
4. **Never A/B two kernels sequentially on this machine** — process-to-process
   spread is ±10–15%, larger than every effect in §6. Interleave round-robin in
   one process. Sequential measurement produced a spurious "tile_n=48 is 15%
   faster" that vanished under interleaving.
5. **Do not interleave an ELF hw_context with an xclbin hw_context**: it
   inflated *both* formats by 1.4–2.5× (qkvo ELF 620→912 µs, xclbin 435→2941 µs).
   Different-format A/Bs need separate processes.
6. **`xrt.hw_context` is a limited resource** — a process that loads ~10+
   distinct ELFs starts failing with
   `DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)` and, before that,
   spurious `ERT_CMD_STATE_TIMEOUT`s. Sweeps must spawn one process per ELF.
7. **`KernelCache.__init__` does not load the manifest** — call
   `cache.load_manifest()` explicitly or every "cached" ELF is silently
   recompiled.
