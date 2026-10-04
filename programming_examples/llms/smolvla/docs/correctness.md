# Correctness: what is verified, and how

## The gate

`make verify` runs the pipeline twice in one process, from the same policy
object and the same batch, changing exactly one thing: where `embed_image` gets
its answer. It compares the final **action chunk** — the (1,50,6) tensor the
robot would execute — not an intermediate activation, and not a
reimplementation.

| | |
|---|---|
| Reference | the unmodified LeRobot CPU model, computed live in the same run |
| Metric | median per-position cosine (primary) + normalized MSE |
| Thresholds | cosine ≥ 0.99, nMSE ≤ 0.04 |
| Determinism | flow-matching noise pinned to zero, images from a fixed seed |

SmolVLA emits a continuous action chunk rather than tokens, so this uses
`regression_gate` from `../../verify/comparators.py` rather than the top-k
token-set gate the autoregressive siblings use.

The reference is recomputed every run rather than loaded from a fixture. A saved
fixture goes stale against a checkpoint or LeRobot upgrade, and being gitignored
it would make the verify lit test fail on a clean checkout. It costs one extra
CPU forward (~0.9 s).

Measured (2026-10-04, AMD Ryzen AI MAX+ 395):

```
3 cameras   cosine 0.999328   nMSE 0.006984   PASS
2 cameras   cosine 0.999148   nMSE 0.002331   PASS
1 camera    cosine 0.998187   nMSE 0.003715   PASS
```

nMSE is normalized rather than absolute so the threshold does not drift with
action magnitude; 0.04 is ~5× the observed value.

## The synthetic input

The gate's images are generated, not recorded, so it stays deterministic and CI
needs no dataset. They are **not** all-zero and **not** white noise; both fail
for measurable reasons.

`_synthetic_image` draws one uniform value per SigLIP patch on the encoder's
32×32 grid, upsamples bilinearly, and adds 5% per-pixel grain. Seed 0, via an
explicit `torch.Generator` so the batch does not depend on global RNG state.
Only the images are randomized — state feeds the CPU-only `state_proj`.

**Why not all-zero.** After LeRobot's normalizer a zero image is a constant −1
frame, which makes all 1024 patches identical: the patch-embedding GEMM is then
probed by a **rank-1** operand, so only its per-channel weight sums matter. A
sum-preserving shuffle of the weights within a channel left the action chunk
bit-identical — a mutation the gate could not see. With the patch-grid input the
same mutation moves the chunk by 0.40.

**Why not white noise.** Per-pixel noise has no spatial structure, and the model
answers it with a near-zero action chunk (rms 0.13 against 0.5–1.6 on real
frames). Cosine on a near-zero reference is hypersensitive: the gate read 0.9869
and FAILED on arithmetic that was correct.

The grain earns its place separately — it lifts the im2col rank from 27/768 to
297 without moving the model out of its normal output range.

## Real frames

`make verify INPUT=real` runs the same CPU-vs-NPU comparison frame by frame
against the same thresholds, and always exits 0: it reports headroom, it is not
the gate. 100 frames of `lerobot/droid_100`, one from the middle of each of its
100 episodes.

| Cameras | Within threshold | cosine median | P10 | cosine worst | nMSE worst |
|---|---|---|---|---|---|
| **3 (shipping)** | **100/100** | 0.999752 | 0.998841 | 0.997604 | 0.009617 |
| 2 | 98/100 | 0.999616 | 0.997230 | 0.983156 | 0.036285 |
| 1 | 97/100 | 0.999142 | 0.996696 | 0.893272 | 0.086942 |

(2026-10-04, with the host-side vision fixes — a vectorised patch-embed unfold and
a capped BLAS thread count, both bit-identical in output; the gate's cosine is
unchanged to every digit by them. The earlier table, from 2026-09, had the same
shape: 100/100, 98/100, 97/100.) The report lists at most the five worst frames
below threshold.

Agreement on real images is better than on the synthetic gate input, so the gate
is not flattered by its generated input.

Cosine is scale-blind on a physical actuator command, so the absolute figure is
worth carrying too: the largest per-dimension error was 1.36 on action dim 5 at
1 camera, 0.64 at 3 (0.95 at 2).

## All three stages on the NPU (experimental)

`make verify-all` (and `make verify-all INPUT=real`) run the same comparison with
the vision encoder, the language backbone and the action expert all on the NPU
(`--npu-all`; see the README, "Experimental: backbone and expert on the NPU"). Same
reference, same thresholds, same 100 frames, 2026-10-04:

| Gate (synthetic) | Cameras | Result | cosine | nMSE |
|---|---|---|---|---|
| vision-only | 3 / 2 / 1 | PASS / PASS / PASS | 0.99933 / 0.99915 / 0.99819 | 0.00698 / 0.00233 / 0.00372 |
| **all NPU** | 3 / 2 / 1 | **PASS / PASS / PASS** | 0.99660 / 0.99857 / 0.99778 | 0.01188 / 0.00716 / 0.01131 |

| `droid_100`, 100 frames | Cameras | Within threshold | cosine median | P10 | cosine worst | nMSE median | nMSE worst |
|---|---|---|---|---|---|---|---|
| vision-only | **3** | 100/100 | 0.999752 | 0.998841 | 0.997604 | 0.00078 | 0.00962 |
| **all NPU** | **3** | **100/100** | 0.999628 | 0.998693 | 0.996188 | 0.00130 | 0.00956 |
| vision-only | 2 | 98/100 | 0.999616 | 0.997230 | 0.983156 | 0.00142 | 0.03629 |
| all NPU | 2 | 98/100 | 0.999321 | 0.996595 | 0.970506 | 0.00213 | 0.05671 |
| vision-only | 1 | 97/100 | 0.999142 | 0.996696 | 0.893272 | 0.00276 | 0.08694 |
| all NPU | 1 | 94/100 | 0.998355 | 0.993409 | 0.762683 | 0.00564 | 0.20952 |

The all-NPU path passes the gate at every camera count and agrees with the CPU model on
every real frame at the shipping 3 cameras, but it is **measurably less accurate than
vision-only**: at 3 cameras the median cosine is 0.00012 lower and the worst frame
0.0014 lower (nMSE median 1.7× higher, worst case about the same); on the gate's
synthetic input the cosine is 0.0027 lower. The gap grows as cameras are removed: at 1
camera 6 frames fall below threshold instead of 3 (the report lists five), and the
worst frame (2861, also the worst for vision-only) goes from cosine 0.89 to 0.76.
Fewer cameras is the case to keep on the default path.

Where the extra error comes from: the backbone and expert use bfp16 weights (8-bit
mantissas, a shared exponent per 8 values) and bf16 activations. Each stage is very
close on its own — the expert is at cosine ≥ 0.99997 per layer against the fp32
reference and 0.99989 for the final output against LeRobot on a captured step, the
backbone 0.99997 chained over its 16 layers — and the error accumulates through the
10 flow-matching steps. This is agreement with the CPU model's action chunk, not
task success.

## Why a few frames fall below threshold

(Analysis of the vision-only path, measured on the 2026-09 build; the shape is
unchanged.) Not because the NPU is less accurate on those images. Measured per frame across
the same 100:

| Quantity | Behaviour |
|---|---|
| Vision-stage error (NPU vs CPU connector) | cosine 0.9959 ± 0.004 — near constant |
| Its correlation with the final chunk error | **r = +0.06** |
| Final ‖error‖ | 0.16 → 3.63, a **22× spread** |

The perturbation entering the CPU backbone is the same size on every frame; what
varies is how much the backbone and the 10 flow-matching Euler steps amplify it.
Injecting a *random* perturbation of equal magnitude instead of the NPU's
reproduces the spread — frame 20540 lands at 0.987 either way, while the best
frames stay above 0.9997 no matter what is injected. The sensitivity belongs to
those frames, not to the port.

One secondary effect: 24% of the connector error energy sits in a fixed,
input-independent direction (pairwise cosine 0.133 between per-frame error
vectors, against a 0.004 white-noise floor) — the systematic residue of bf16
weight rounding. On an already-sensitive frame it can therefore hurt more than
equal-norm white noise would.

Fewer cameras is consistently worse because the cameras' independent error
components average out; three feeds leave less of the vision error to propagate.

## What this does and does not claim

It claims the port did not change the model's behaviour. It does not claim the
model is good at the task — that needs evaluation against recorded robot
trajectories, which is out of scope for a kernel port.
