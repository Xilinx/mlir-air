# Profiling SmolVLA on the NPU

How to measure the example, how to read what it prints, and where the time
goes. For the latest numbers, run the commands below on the machine you care
about, or look at the `smolvla` and `smolvla_all` rows of the nightly
benchmark.

## Measuring

```bash
make profile REPS=15       # pure CPU against NPU vision
make profile-all REPS=15   # adds NPU vision + backbone, and every stage on the NPU
```

Both run in one process. Every arm is warmed with a discarded forward, then the
arms run interleaved, one rep of each in turn, so drift and thermal changes hit
them alike. Medians are reported.

Before comparing two runs:

- Set the CPU governor and energy-performance preference to `performance` and
  the NPU power mode to Turbo. A `balanced` machine slows the CPU arms more than
  the NPU ones, so it changes the ratios, not only the absolute times.
- Make sure nothing else is using the CPU or the NPU (`uptime`,
  `fuser /dev/accel/accel0`), and wrap the run in the NPU lock described in
  [`usage.md`](usage.md).
- Compare runs from the same session. Process-to-process spread on one machine
  is larger than many of the effects worth measuring.

## Reading the output

`make profile` prints:

- End to end: the median, min and max action-chunk latency of the pure-CPU and
  NPU-vision arms, and their ratio.
- Per stage: vision (SigLIP and connector, all cameras), backbone and action
  expert for each arm. In `make profile` the backbone and expert are the same
  CPU code in both arms, so those rows match.
- NPU device time per image, per ELF: how many dispatches of each ELF an image
  takes and their device time, and the total against the wall time of the
  vision stage. The difference is host work.
- An `Action chunk (...)` line that `../bench/extract_perf.py` reads for the
  nightly benchmark.

`make profile-all` adds a table of the experimental arms (NPU vision + backbone,
and all NPU) with the same per-stage rows, their speedup over pure CPU, each
arm's action chunk compared with pure CPU, and the action expert's time per
call. The CPU stages in the NPU arms run faster than in the pure-CPU arm, because
those arms pin the CPU threads; compare each stage against its own column.

## Where the vision time goes

The vision stage is bound by the NPU, not the host: most of its wall time is
device time, and the per-ELF table shows it. Three ELFs per layer, `vit_ln_qkv`,
`flash_attn` and `vit_o_ffn`, carry nearly all of it; the post-LayerNorm and the
connector projection are small.

Within a layer, the GEMMs are under half of the device time. FlashAttention is a
large share, and the rest is elementwise and normalization launches (bias adds,
residual adds, LayerNorm, GELU) that move a few MB each and do little
arithmetic.

Fusing a layer's launches into one ELF costs little device time compared with
running the same launches on their own. Most of the gap between a fused ELF
measured alone and the same ELF inside the encoder comes from the three ELFs'
hardware contexts alternating every layer.

Two things to keep in mind when comparing with the kernel registry:

- The registry's GEMM throughput is measured on the xclbin path. The fused vision
  ELFs run through the ELF path, which has been measured slower for the same
  module, so registry numbers are for choosing a method and tile, not for
  predicting deployed latency.
- The best tile in the ELF regime is not always the best one on the xclbin
  harness; tiles tuned on the harness do not transfer rank for rank. Inside the
  fused ELFs, `tile_n` matters little.

## Where the time could come from

Directions, not budgets; measure before committing to one.

- Fold the bias adds and residual adds into the GEMM drain epilogue. They are
  the largest block of device time that does no matmul work.
- Close the gap between the ELF and xclbin lowering of the same GEMM. That is
  compiler and runtime work.
- Merge `flash_attn` into the fused ELFs, so a layer is one ELF instead of three
  and the hardware contexts stop alternating.
- Speed up the affine LayerNorm, whose effective bandwidth is well below that of
  a bias add moving the same amount of data.
- A faster FlashAttention at this shape. The deployed kernel already runs at its
  standalone speed, so only kernel work helps here.

For the action expert: each call streams all of its weights from DDR, and the
engine stalls where a group of drains has to complete before the next job may
read their output. The `NPU expert per call` line of `make profile-all` shows
how much of a call is device time.

## Measurement traps

- Never A/B two kernels in two separate runs; interleave them in one process.
  Sequential runs have produced "wins" that vanished under interleaving.
- Do not run an ELF hardware context and an xclbin hardware context in the same
  process for an A/B: both slow down. Compare formats in separate processes.
- `xrt.hw_context` is a limited resource. A process that loads many distinct
  ELFs starts failing to create contexts, and before that sees spurious
  `ERT_CMD_STATE_TIMEOUT`s. Sweeps should use one process per ELF.
- `KernelCache.__init__` does not load the manifest; call
  `cache.load_manifest()`, or every cached ELF is recompiled.
