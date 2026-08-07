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

Measured, 3 cameras:

```
cosine 0.998427   nMSE 0.007423   PASS
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
| **3 (shipping)** | **100/100** | 0.999679 | 0.998782 | 0.997360 | 0.008050 |
| 2 | 98/100 | 0.999435 | 0.997137 | 0.987832 | 0.036063 |
| 1 | 97/100 | 0.998840 | 0.995269 | 0.957068 | 0.086860 |

Agreement on real images is better than on the synthetic gate input, so the gate
is not flattered by its generated input.

Cosine is scale-blind on a physical actuator command, so the absolute figure is
worth carrying too: the largest per-dimension error was 1.05 on action dim 5 at
1 camera, 0.36 at 3.

## Why a few frames fall below threshold

Not because the NPU is less accurate on those images. Measured per frame across
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
