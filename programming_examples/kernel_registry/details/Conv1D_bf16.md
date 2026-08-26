<!---//===- Conv1D_bf16.md ------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Conv1D (BF16, causal depthwise, k=3) — NPU2

Causal depthwise 1-D convolution — the convolution inside LFM2's
`Lfm2ShortConv` operator, and a prerequisite for the other conv-hybrid
families (Gemma4, Qwen3.5) whose FLM designs also carry a `conv1d` kernel.

```
y[t, c] = w[0, c]·x[t+0, c] + w[1, c]·x[t+1, c] + w[2, c]·x[t+2, c]
```

**Depthwise** = each channel has its own 3 taps, no cross-channel mixing.
The channel axis is therefore the vectorization axis and is contiguous in
both input and output.

- Harness: `programming_examples/conv1d_depthwise`
- Microkernel: `conv1d_depthwise.cc` → `conv1d_depthwise.o` (Peano, aie2p)
- Builder: `conv1d_depthwise.py` (raw AIR bindings, `silu_and_mul` pattern)

## Interface and the pre-padding convention

| operand | shape | notes |
|---|---|---|
| `x` | `(seq + 2, C)` bf16 | **pre-padded**; row `t` is original position `t − 2` |
| `w` | `(3, C)` bf16 | **tap-major**; oldest tap first |
| `y` | `(seq, C)` bf16 | |

Causality is expressed by **pre-padding, not masking**. Because `x` carries
two extra leading rows, input row `t` is the oldest sample feeding `y[t]` and
pairs with tap 0. This matches PyTorch `nn.Conv1d`, which computes
*cross-correlation* over a left-padded input:

```python
# HF Lfm2ShortConv: nn.Conv1d(..., padding=L_cache-1) then [..., :seqlen]
out[t] = Σ_j w[j] · x_padded[t + j]      # j=0 -> x[t-2] (oldest)
```

Independently corroborated by HF's own decode path, which does
`sum(conv_state * weight[:, 0, :], dim=-1)` with `conv_state` ordered
oldest→newest.

**The two leading rows are the conv state.** Zeros at the start of a sequence
(prefill); the carried tail of the previous chunk at decode. So **prefill and
decode are the same kernel with a different pad** — there is no separate
decode variant, and no `seq=1` special case.

`w` is passed **tap-major** `(3, C)` rather than HF's channel-major
`(C, 1, 3)` so that `w + j·C` is a contiguous channel slice per tap and the
inner loop can use unit-stride vector loads. The host transposes once at
weight-load time.

## Numerical datapath

Per output element, in the microkernel:

```cpp
aie::accum<accfloat, 16> acc = aie::mul(v0, c0);   // bf16 x bf16 -> f32 accum
acc = aie::mac(acc, v1, c1);
acc = aie::mac(acc, v2, c2);
aie::store_v(pY, acc.to_vector<bfloat16>());       // single bf16 rounding
```

Three bf16 products accumulated in **FP32**, rounded to bf16 exactly once on
store — the GPU/HF standard for a bf16 depthwise conv. Written as
`mul` + 2× `mac` so it lowers to the vector FMA unit; deliberately **not** a
`mulf`→`addf` chain, which aievec rejects.

### Tolerances

| | value | rationale |
|---|---|---|
| `rtol` | `1.6e-2` | canonical bf16 (PyTorch/vLLM) |
| `atol` | `5e-2` | same as RoPE / Element-wise Add — an f32-accumulate op with one output rounding needs no transcendental slack |

Measured `mean_rel_L1 = 2.816e-3`, `rel_err max = 7.8e-3`,
`abs_err max = 6.25e-2`. That places Conv1D in the registry's **cleanest
tier**, tied with RoPE (2.8e-3) and just above Element-wise Add (1.9e-3) —
a 3-term reduction is far too short to accumulate meaningful error, and
there is no transcendental. Accuracy is bit-identical across every placeable
config, as expected (tile/herd are pure performance knobs).

### ⚠️ Gate on a NONZERO pad — a zero pad does not test tap order

The harness defaults to a **nonzero random** conv-state pad (the decode case)
and takes `--zero-pad` for the prefill / sequence-start case. That default
is deliberate, and it is the single most important thing to preserve about
this kernel's verification.

With a **zero** pad, the tap-0 and tap-1 contributions to the first two
outputs vanish, so a wrong tap **order** — newest-first instead of
oldest-first — still passes on essentially every row. Tap order is exactly
the convention most likely to be implemented backwards (PyTorch `nn.Conv1d`
is cross-correlation, not convolution), so a zero-pad-only gate is close to
no gate at all for the one thing worth checking.

| pad | seq×C | mean_rel_L1 | verdict |
|---|---|---|---|
| zero (prefill) | 2048×2048 | 2.816e-3 | ✅ |
| **nonzero (decode state)** | 2048×2048 | 2.813e-3 | ✅ |
| zero (prefill) | 8×2048 | 2.775e-3 | ✅ |
| **nonzero (decode state)** | 8×2048 | 2.836e-3 | ✅ |

The nonzero-pad rows are what certify the **decode** path, since prefill and
decode are the same kernel with a different pad. Generalizes: when a kernel's
edge case is expressed as padding, make the *general* (nonzero) case the
default and the degenerate case opt-in — zeros silently satisfy a whole class
of wrong implementations.

## Tunable parameters

| knob | meaning | constraint |
|---|---|---|
| `herd_x` | AIE columns; splits the **channel** axis | `C % herd_x == 0`; `tile_c = C/herd_x` must be a multiple of the 16-lane vector |
| `herd_y` | AIE rows; splits the **sequence** axis | all verified configs use **1**; `>1` is UNTESTED — see below |
| `tile_s` | sequence rows produced per tile iteration | `seq % (tile_s · herd_y) == 0`, and the L1 budget below |

**L1 budget (hard):** three live buffers per tile —

```
(tile_s + 2)·tile_c   halo input window
+        3·tile_c     weights
+   tile_s·tile_c     output
```

at 2 bytes/element, must be ≤ **65536**. Equivalently
`(2·tile_s + 5)·tile_c·2 ≤ 65536`.

**Halo cost:** a tile producing `tile_s` rows reads `tile_s + 2`, so small
`tile_s` pays a `2/tile_s` read amplification (12.5% at `tile_s=16`).

## Placement sweep (NPU2, seq=2048, C=2048)

| hx×hy | tile_s | L1 | verdict | latency |
|---|---|---|---|---|
| 8×1 | 4 | 7 KB | **PASS** | ⏳ |
| 8×1 | 8 | 11 KB | **PASS** | ⏳ |
| 8×1 | 16 | 19 KB | **PASS** | ⏳ |
| 8×1 | 32 | 35 KB | **PASS** | ⏳ |
| 4×1 | 16 | 37 KB | **PASS** | ⏳ |
| 1×1 | 4 | 53 KB | **PASS** | ⏳ |
| 4×1 | 32 | 71 KB | ⚠️ **silently wrong** (compiles) | — |
| 4×2 | 32 | 71 KB | ⚠️ **silently wrong** (compiles) | — |
| 2×1 | 32 | 141 KB | aircc compile failure | — |
| 1×1 | 32 | 282 KB | aircc compile failure | — |
| 8×4, 2×4 | 32 | — | aircc compile failure | — |
| 8×2 | 32 | 35 KB | TIMEOUT — **cause unresolved** (see below) | — |
| 2×1 | 8 | 42 KB | TIMEOUT — **cause unresolved** (see below) | — |

⏳ **No latency is recorded.** Every timing run was contaminated by an
unrelated NPU holder; see "Measurement hazard" below. Correctness is
unaffected — all PASS verdicts are full-output element-wise checks.

### Shipped config: `herd_x=8, herd_y=1` — 8 tiles

All four `tile_s ∈ {4, 8, 16, 32}` PASS at `herd_x=8`, as do `4×1` and `1×1`
within the L1 budget. Accuracy is bit-identical across every one of them,
matching the registry rule that accuracy is set by the datapath, not the
tiling.

**Which config is *fastest* is not yet known** — see "Measurement hazard".

#### ⚠️ RETRACTED: "`herd_y > 1` hangs, so Conv1D is capped at 8 tiles"

An earlier revision of this page asserted, as a **measured** result, that
`herd_y=2` hangs and that Conv1D therefore sits under the same 3-shim-DMA /
one-8-column-row ceiling as Element-wise Add and SiLU-and-Mul — explicitly
framed as "swept rather than inherited", per the registry methodology note.

**That is withdrawn.** The `ERT_CMD_STATE_TIMEOUT` behind it occurred while
an unrelated **LLM server held the NPU**, discovered only after a reboot. Contention explains a
submission timeout at least as well as the herd shape does, and the same
contention wedged the device moments later. The two hypotheses cannot be
separated without an idle device.

Conv1D *does* issue 3 shim DMAs per tile, so the ceiling remains plausible —
but plausible is not measured, and this page previously presented it as
measured. **Sweep `herd_y` on a quiet box before recording any ceiling.**

The irony is worth preserving: the registry's own methodology warns against
inheriting a sibling kernel's herd cap, and this page inherited exactly that
cap while claiming to have measured it.

### ⚠️ Two traps

**1. L1 over-allocation is silent at `herd_x=4`.** `4×1, tile_s=32` needs
70.6 KB against 64 KB and *compiles cleanly, then returns wrong results*,
while the larger `herd_x ∈ {1,2}` overflows fail inside aircc. The failure
mode is **non-monotonic in overflow size**, so "it built, therefore it fits"
is false. `conv1d_depthwise.py` carries an explicit `L1_BYTES` assert
rejecting these configs up front — **do not remove it**. Same class as the
GEMM `N % (tile_n × herd_n)` silent-corruption trap.

**2. A hanging config can wedge the entire device.** After a config hit
`ERT_CMD_STATE_TIMEOUT`, every subsequent NPU submission failed with
`DRM_IOCTL_AMDXDNA_EXEC_CMD IOCTL failed (err=-5)` — including configs of
this kernel that had passed minutes earlier, untouched upstream examples
(`eltwise_add`), and `xrt-smi validate --run latency` itself. A
`modprobe -r amdxdna && modprobe amdxdna` reload did **not** recover it: the
kernel log showed the real failure one layer lower —

```
aie2_smu_exec: SMU cmd 4 failed, 0xff
aie2_smu_start: Access power failed, ret -22
amdxdna_probe: Hardware init failed, ret -22
```

— i.e. the SMU would not power the NPU rail back up. Only a reboot cleared
it. **When sweeping a new kernel, re-run a known-good control after any
TIMEOUT before trusting later results**, otherwise a wedged device reads as
a long run of genuine placement failures. That is precisely what happened
here: three configs recorded as failures were re-run after the reboot and
all **PASS**.

## Measurement hazard — read before filling any latency column

Every timing run for this kernel shared the NPU with an unrelated **LLM
server** (autostarted at boot, holding `/dev/accel/accel0`). The contention
signature:

- a 4 M-element `eltwise_add` — registry row **437 µs** — measuring **16.9 ms**
- a tile × herd sweep in which **more tiles came out slower**, backwards for
  a memory-bound kernel
- an apparent runtime hang that, after the box was cleared, did not reproduce

Before recording any number:

```bash
for p in $(ls /proc | grep -E '^[0-9]+$'); do \
  ls -l /proc/$p/fd 2>/dev/null | grep -q accel && echo "$p $(cat /proc/$p/comm)"; done
uptime    # a stray 99%-CPU process alone skews a memory-bound kernel
```

then re-run a known-good sibling as a control (`eltwise_add` at
`N = 4194304` should reproduce ≈437 µs).

## Reproduce

```bash
source ~/new_session/toolchain/air_env.sh   # mlir-aie pin 7e00b57 + Peano
cd programming_examples/conv1d_depthwise

# correctness (element-wise vs FP32 reference)
flock -x -w 2400 /tmp/mlir-air-npu.lock \
  make run HERD_X=8 HERD_Y=1 TILE_S=16

# correctness + latency
flock -x -w 2400 /tmp/mlir-air-npu.lock \
  make profile HERD_X=8 HERD_Y=1 TILE_S=16 ITERATIONS=30

# MLIR only, no device
make print HERD_X=8 HERD_Y=1 TILE_S=16
```

Defaults are the LFM2-1.2B ShortConv prefill scale (`SEQ=2048`,
`CHANNELS=2048`, `TILE_S=32`, `HERD_X=8`, `HERD_Y=1`).

The harness generates N(0,1) activations with the two leading pad rows left
**zero** — deliberately the start-of-sequence case, so the first two outputs
exercise the zero-padding path — and taps at N(0, 0.5), matching the O(0.1–1)
scale of trained LFM2 conv weights.
