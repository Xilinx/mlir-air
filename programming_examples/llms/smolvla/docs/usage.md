# SmolVLA on NPU2 — usage guide

Every command this example provides, and what each one does.

## Prerequisites

**Hardware and toolchain**

- AMD NPU2 (Strix, AIE2P)
- MLIR-AIR with the Peano compiler (`PEANO_INSTALL_DIR` set)
- The project environment: `source utils/env_setup.sh ...`

**Python.** One interpreter needs both sides: `torch` + `lerobot` to run the
policy, `air` + `pyxrt` to drive the NPU. LeRobot *is* the CPU baseline this
example verifies against, so it is a dependency, not an optional extra.

```bash
pip install -r requirements.txt      # into the env that has air + pyxrt
```

LeRobot requires `numpy>=2.0,<2.3`. An older MLIR-AIR environment may still be
on 1.x, and installing will upgrade it for every example sharing that
interpreter. To keep the shared environment untouched, use a venv — sourcing the
MLIR-AIR environment puts `air` and `pyxrt` on `PYTHONPATH` / `LD_LIBRARY_PATH`,
so they import from anywhere:

```bash
python3 -m venv ~/smolvla-venv
~/smolvla-venv/bin/pip install -r requirements.txt
make verify LEROBOT_PYTHON=~/smolvla-venv/bin/python
```

**Model access.** `HF_TOKEN` must be set; `lerobot/smolvla_base` (450M
parameters) downloads on first use.

---

## Targets

| Target | What it does | Touches the NPU |
|---|---|---|
| `make help` | list the targets | no |
| `make compile` | build every vision ELF — no dispatch, no download | no |
| `make cpu-baseline` | run the unmodified CPU model alone, for inspection | no |
| `make run` | one end-to-end forward; prints the chunk shape and magnitude | yes |
| `make verify` | **the gate** — action chunk vs the pure-CPU model, PASS/FAIL | yes |
| `make profile` | CPU vs NPU interleaved, with the per-ELF breakdown | yes |
| `make clean` | remove the kernel cache and build artifacts | no |

## Variables

| Variable | Default | Applies to | Meaning |
|---|---|---|---|
| `INPUT` | `synthetic` | run, verify | `synthetic` = seeded-random images, nothing downloaded. `real` = frames from a LeRobot dataset |
| `CAMERAS` | `3` | run, verify, profile | 1, 2 or 3 feeds. No recompile needed — the count is only how many times the host encode loop runs |
| `DATASET` | `lerobot/droid_100` | `INPUT=real` | any LeRobot dataset; camera keys are read from its metadata |
| `FRAMES` | `100` | `verify INPUT=real` | one frame per episode, from the middle of each |
| `REPS` | `5` | profile | interleaved CPU/NPU pairs; the median is reported |
| `LEROBOT_PYTHON` | `python3` | all | interpreter with torch + lerobot + air + pyxrt |
| `SMOLVLA_FORCE_COMPILE` | unset | all | `=1` rebuilds every ELF instead of reusing the cache |

```bash
make verify                                  # the gate: synthetic, 3 cameras
make verify CAMERAS=1                        # synthetic, one camera
make verify INPUT=real                       # 100 real frames, reports a distribution
make verify INPUT=real FRAMES=20 CAMERAS=2
make run INPUT=real                          # one forward on one real frame
make profile CAMERAS=1 REPS=10               # timing; always synthetic
```

**Only plain `make verify` is a gate.** It is deterministic, exits non-zero on
failure, and is what CI and the lit test run. `INPUT=real` reports a
distribution and always exits 0 — see
[`correctness.md`](correctness.md).

---

## First run

```bash
make compile        # a few minutes; produces build/vision_kernel_cache/
make verify         # no fixture needed — the CPU reference is computed live
```

```
============================================================
SmolVLA verify: end-to-end action-chunk regression gate
  NPU stages     : vision
  execution model: single-process (air/pyxrt in the lerobot venv)
============================================================
  cosine   = 0.9984266757965088
  cos_min  = 0.99
  mse      = 0.00026965420693159103
  nmse     = 0.007422617636620998
  nmse_max = 0.04
  max_abs  = 0.05484716594219208
  passed   = True
============================================================
[verify] PASS
```

To sanity-check the harness itself, run the adapter directly with
`--cpu-vision`: it compares the unmodified model against its own baseline and
should score exactly 1.0. The Makefile has no pass-through for that flag.

```bash
$(LEROBOT_PYTHON) verify_adapter.py --cpu-vision
```

---

## `make profile`

Both arms are warmed with a discarded forward, then `REPS` CPU/NPU pairs run
interleaved so drift hits both equally, and the median is reported.

```
  end to end                              median       min       max
  ------------------------------------ --------- --------- ---------
  pure CPU (unmodified lerobot)            919.9     902.1     980.4
  NPU vision + CPU backbone/expert         831.6     808.9     866.0

  speedup (median)  1.106x

  per stage                                  CPU   NPU run   speedup
  ------------------------------------ --------- --------- ---------
  vision: SigLIP + connector (x3)          544.8     453.6     1.20x
  backbone: SmolLM2-360M (x1)               78.6      78.6  CPU both
  action expert (x10 denoise steps)        281.9     281.9  CPU both

  NPU device time, per image (of 3)        calls  ms/image
  ------------------------------------ --------- ---------
  vit_o_ffn                                   12     65.40
  flash_attn                                  12     36.77
  vit_ln_qkv                                  12     36.15
  gemm_connector                               1      1.12
  layer_norm                                   1      0.73
  TOTAL device / image                              140.18

  x3 images = 420.5 ms device, of the 453.6 ms vision stage (93% device, 33.0 ms host)
```

The backbone and expert rows carry the same number in both arms on purpose:
they are the same unmodified CPU code either way. For what the breakdown means
and where the remaining time could go, see [`profile.md`](profile.md).

Two conditions change the numbers materially:

- **Power state.** Everything here was measured with the CPU governor and EPP at
  `performance` and the NPU at `pmode=Turbo`. A `balanced` machine moves the CPU
  baseline more than the NPU stage, so the *ratio* changes, not just the
  absolutes.
- **Other NPU users.** Check with `fuser /dev/accel/accel0`.

---

## The NPU lock

On a machine where several sessions share one NPU, take the project lock around
anything that touches the device:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
```

The recipes do not take it themselves — if they did, the command above would
deadlock against itself. Correctness does not depend on it either:
`shared/infra/cache.py` holds `/tmp/npu.lock` around every dispatch, so
concurrent runs interleave safely. The outer lock is for timing.

---

## Rebuilding kernels after an edit

The ELF cache is reused whenever its manifest resolves and contains every
expected kernel. The manifest does not track source hashes, so after editing a
kernel builder, force a rebuild:

```bash
SMOLVLA_FORCE_COMPILE=1 make verify
```

---

## `INPUT=real`

First use downloads the whole dataset — video-backed LeRobot datasets store one
MP4 per camera per chunk, so a frame subset is not possible. `droid_100` is
464 MB; other datasets range into the GBs.

Datasets need not match the checkpoint's dimensions. `droid_100` is 180×320 with
a 7-wide state against the checkpoint's 256×256 and 6; `resize_with_pad` and
`pad_vector` absorb both, and the NPU sees seq 1024 either way.
