<!---//===- EltwiseMul_bf16.md --------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//-->

# Element-wise Multiply (BF16)

`c[i] = a[i] · b[i]`, per-element, BF16 in / BF16 out.

Added for **LFM2-1.2B**, whose `Lfm2ShortConv` operator applies two
element-wise gates per conv layer:

```
h = B · x                       # gate 1
h = causal_depthwise_conv1d(h)  # Conv1D_bf16
y = C · h                       # gate 2
```

With 10 of LFM2-1.2B's 16 layers being conv layers, this kernel runs 20 times
per prefill.

## Why it is its own kernel

The registry already had two multiply-adjacent kernels; neither covers this:

| kernel | computes | why it does not substitute |
|---|---|---|
| Element-wise Add | `a + b` | different op; a sum and a product have different error |
| SiLU-and-Mul | `SiLU(a) · b` | applies a transcendental to one operand first — the `aie::tanh` LUT dominates its error budget (`1.0e-2` vs `2.7e-3` here) |

Reusing SiLU-and-Mul's multiply half would mean paying for, and being gated
at the tolerance of, a `tanh` this datapath never performs.

## Datapath and precision

Two bf16 loads, one vector multiply, one bf16 store. **A single rounding**,
no accumulation, no transcendental — the cleanest tier in the registry.

| kernel | mean_rel_L1 | why |
|---|---|---|
| Element-wise Add | 1.9e-3 | one rounding, sum |
| **Element-wise Multiply** | **2.7e-3** | one rounding, product |
| Conv1D (k=3) | 2.8e-3 | 3 products, FP32 accumulate, one rounding |
| SiLU-and-Mul | 1.0e-2 | `aie::tanh<bf16>` LUT |

Multiply sits just above add, as expected: the relative errors of both
operands compound in a product, whereas in a sum the larger operand's
rounding dominates. Both remain far inside the canonical bf16
`rtol = 1.6e-2`.

## Tolerances

`rtol = 1.6e-2` (canonical bf16, PyTorch/vLLM standard), `atol = 5e-2` —
the same pairing as Element-wise Add and RoPE. `atol` covers a few
large-magnitude bf16 *output*-rounding ULPs; it is not a relaxation of the
datapath. Gated on a **full-output element-wise `np.isclose`** against an
FP32 reference (bf16 inputs upcast to f32, multiplied in f32, cast back) —
never on cosine.

## Per-shape data

| N | (as 2-D) | latency | bandwidth | mean_rel_L1 | Status |
|---|---|---|---|---|---|
| 1048576 | 1024×1024 | ⏳ | ⏳ | 2.728e-3 | ✅ correctness |
| 4194304 | 2048×2048 | ⏳ | ⏳ | 2.736e-3 | ✅ correctness |

⏳ **Perf not yet measured on an idle device.** See "Measurement hazard".

## Measurement hazard — read before filling the perf columns

Every timing attempt for this kernel was contaminated by an unrelated **LLM
server holding `/dev/accel/accel0`**, which was not discovered until
afterwards. Symptoms, all of
which should be treated as a contention signature rather than a kernel
result:

- 17 ms at `N = 4194304` — a kernel that should stream at tens of GB/s
  measuring ~0.7 GB/s.
- Element-wise Add, whose registry row records **437 µs** at the same `N`,
  measuring **16.9 ms** in the same session. *This control is what
  identified the problem.*
- A tile × herd sweep in which **more tiles came out slower**, which is
  backwards for a memory-bound streaming kernel.

Before recording any number here:

```bash
for p in $(ls /proc | grep -E '^[0-9]+$'); do \
  ls -l /proc/$p/fd 2>/dev/null | grep -q accel && echo "$p $(cat /proc/$p/comm)"; done
uptime    # a stray 99%-CPU process alone is enough to skew a memory-bound kernel
```

and re-run a **known-good sibling** (Element-wise Add at `N = 4194304`
should reproduce ≈437 µs) as a control.

## Provenance

This kernel was derived from `eltwise_add`, so the emitted MLIR was checked
to contain **1 `arith.mulf` and 0 `arith.addf`** — confirming it is a real
multiply and not an add that survived the copy:

```bash
make print SHAPE="2048 2048" | grep -c "arith.mulf"   # 1
make print SHAPE="2048 2048" | grep -c "arith.addf"   # 0
```

Worth repeating for any kernel cloned from a sibling. A decoy that shares the
original's shape signature passes every shape test and every "does it run"
check — the same trap the registry documents for RoPE (half-split vs
interleaved) and FlashAttention (heads-first vs seq-first).

## Reproduce

```bash
source ~/new_session/toolchain/air_env.sh
cd programming_examples/eltwise_mul

# correctness at the LFM2 ShortConv gate shape
flock -x -w 1800 /tmp/mlir-air-npu.lock make run SHAPE="2048 2048"

# perf — ONLY on an idle device (see Measurement hazard)
flock -x -w 1800 /tmp/mlir-air-npu.lock make run SHAPE="4194304" PERF_ITERS=30
```

## Tunable parameters

| knob | meaning | constraint |
|---|---|---|
| `--tile` | elements per tiled dimension | defaults to the largest power of two whose ping-ponged buffers fit L1 |
| `--herd-shape` | physical herd | chosen per target by default |
| `--vector-size` | compute vector lanes | 16 for bf16 |

Both the tile and herd defaults are shared with `eltwise_add` (same DSL
body), so a config that places for one places for the other — which is what
makes Element-wise Add a valid control for this kernel.
