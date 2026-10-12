# Correctness: what is verified, and how

## The gate

`make verify` runs the pipeline twice in one process, from the same policy
object and the same batch, changing one thing: whether the NPU stages are used.
It compares the final action chunk, the (1,50,6) tensor the robot would execute,
not an intermediate activation and not a reimplementation.

| | |
|---|---|
| Reference | the unmodified LeRobot CPU model, computed live in the same run |
| Metric | median per-position cosine (primary) and normalized MSE |
| Thresholds | cosine ≥ 0.99, nMSE ≤ 0.04 |
| Determinism | flow-matching noise pinned to zero, images from a fixed seed |

SmolVLA emits a continuous action chunk rather than tokens, so this uses
`regression_gate` from `../../verify/comparators.py` instead of the top-k
token-set gate the autoregressive examples use.

The reference is recomputed every run rather than loaded from a fixture. A saved
fixture goes stale with a checkpoint or LeRobot upgrade, and being gitignored it
would fail the verify lit test on a clean checkout. The cost is one extra CPU
forward.

nMSE is normalized so that the threshold does not move with the action
magnitude. The shipping configuration clears both thresholds by a wide margin,
so a pass with little margin is itself worth a look.

`make verify-all` is the same gate with every stage on the NPU. `CAMERAS=2` and
`CAMERAS=1` run either gate with fewer feeds.

## The synthetic input

The gate's images are generated, not recorded, so it stays deterministic and CI
needs no dataset. They are neither all-zero nor white noise, because both hide
errors.

`_synthetic_image` draws one uniform value per SigLIP patch on the encoder's
32×32 grid, upsamples bilinearly and adds 5% per-pixel grain. The seed is fixed
and drawn from an explicit `torch.Generator`, so the batch does not depend on
global RNG state. Only the images are randomized; the state feeds the CPU-only
`state_proj`.

Why not all-zero: after LeRobot's normalizer a zero image is a constant −1
frame, so all 1024 patches are identical and the patch-embedding GEMM sees a
rank-1 operand. Only the per-channel weight sums then matter: shuffling the
weights within a channel, sums preserved, left the action chunk bit-identical. A
gate on that input cannot see such a bug. The patch-grid input moves the chunk
well past the threshold under the same mutation.

Why not white noise: per-pixel noise has no spatial structure, and the model
answers it with a near-zero action chunk. Cosine against a near-zero reference
is hypersensitive, and correct arithmetic failed the gate on it.

The grain is there for the patch embedding: it raises the rank of the im2col
matrix without moving the model out of its normal output range.

## Real frames

`make verify INPUT=real` runs the same comparison frame by frame against the
same thresholds over `FRAMES` frames of `DATASET` (by default 100 frames of
`lerobot/droid_100`, one from the middle of each episode). It reports the
fraction within threshold, the cosine and nMSE distribution, the worst action
dimension, and the five worst frames below threshold. It always exits 0: it
measures headroom, it is not the gate.

With three cameras, the shipping configuration, nearly all frames are within
threshold on the vision-only path; the all-NPU path, which is less accurate,
can leave more below it. Fewer cameras leaves more frames below threshold on
both paths, and more on the all-NPU path than on vision only. Agreement on real frames is better than on the
synthetic input, so the gate is not flattered by its generated images.

Cosine ignores scale, which matters for an actuator command, so the report also
gives the largest absolute error per action dimension.

## The all-NPU path

The all-NPU path is less accurate than vision only. The backbone and the expert
use bfp16 weights (8-bit mantissas, one exponent shared by 8 values) and bf16
activations. Each stage on its own agrees closely with its fp32 reference; the
error accumulates through the 10 flow-matching steps that each re-run the
expert.

## Why a few frames fall below threshold

On the vision-only path, the frames below threshold are not frames the NPU
computes less accurately. Measured per frame:

- The vision-stage error (NPU against CPU connector output) is nearly the same
  on every frame.
- It is uncorrelated with the final action-chunk error.
- The final error varies by more than an order of magnitude between frames.

The perturbation entering the backbone is the same size on every frame; what
varies is how much the backbone and the 10 Euler steps amplify it. Injecting a
random perturbation of the same size instead of the NPU's reproduces the
spread: the same frames fall below threshold either way, while the best frames
stay well above it whatever is injected. The sensitivity belongs to those
frames, not to the port.

A secondary effect: part of the connector error points in a fixed,
input-independent direction, the systematic residue of bf16 weight rounding. On
an already sensitive frame it can hurt more than white noise of the same size.

Fewer cameras is consistently worse because the cameras' independent errors
average out; with three feeds less of the vision error propagates.

## What this does and does not claim

It claims the port did not change the model's behaviour. It does not claim the
model is good at the task; that needs evaluation against recorded robot
trajectories, which is out of scope for a kernel port.
