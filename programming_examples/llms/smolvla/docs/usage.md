# SmolVLA on NPU2: usage guide

Every command this example provides, and what each one does.

## Prerequisites

Hardware and toolchain:

- AMD NPU2 (AIE2P)
- MLIR-AIR with the Peano compiler (`PEANO_INSTALL_DIR` set)
- The project environment: `source utils/env_setup.sh ...`

Python: one interpreter needs both sides: `torch` + `lerobot` to run the
policy, `air` + `pyxrt` to drive the NPU. LeRobot *is* the CPU baseline this
example verifies against, so it is a dependency, not an optional extra.

```bash
pip install -r requirements.txt      # into the env that has air + pyxrt
```

LeRobot pins its own numpy range, and installing it may change numpy for every
example sharing that interpreter. To keep the shared environment untouched, use
a venv; sourcing the MLIR-AIR environment puts `air` and `pyxrt` on
`PYTHONPATH` and `LD_LIBRARY_PATH`, so they import from anywhere:

```bash
python3 -m venv ~/smolvla-venv
~/smolvla-venv/bin/pip install -r requirements.txt
make verify LEROBOT_PYTHON=~/smolvla-venv/bin/python
```

Model access: `lerobot/smolvla_base` and the
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` backbone it loads download on first
use. Both are public, so no `HF_TOKEN` is needed, though setting one raises the
Hub rate limit.

---

## Targets

| Target | What it does | Touches the NPU |
|---|---|---|
| `make help` | list the targets | no |
| `make compile` | build every vision ELF; no dispatch, no download | no |
| `make cpu-baseline` | run the unmodified CPU model alone, for inspection | no |
| `make run` | one end-to-end forward; prints the chunk shape and magnitude | yes |
| `make verify` | the gate: action chunk against the pure-CPU model, PASS/FAIL | yes |
| `make profile` | CPU vs NPU interleaved, with the per-ELF breakdown | yes |
| `make run-all`, `make verify-all`, `make profile-all` | experimental: the same three with the vision encoder, the language backbone and the action expert all on the NPU (`--npu-all`); `verify-all` is the same gate on that path | yes |
| `make compile-expert` | build the action expert's two engine ELFs, needed by the `-all` targets | no |
| `make clean` | remove the kernel cache and build artifacts | no |

## Variables

| Variable | Default | Applies to | Meaning |
|---|---|---|---|
| `INPUT` | `synthetic` | run, verify | `synthetic` = seeded-random images, nothing downloaded. `real` = frames from a LeRobot dataset |
| `CAMERAS` | `3` | run, verify, profile | 1, 2 or 3 feeds. No recompile needed: the count is only how many times the host encode loop runs |
| `DATASET` | `lerobot/droid_100` | `INPUT=real` | any LeRobot dataset; camera keys are read from its metadata |
| `FRAMES` | `100` | `verify INPUT=real` | one frame per episode, from the middle of each |
| `REPS` | `5` | profile | interleaved CPU/NPU pairs; the median is reported |
| `LEROBOT_PYTHON` | `python3` | all | interpreter with torch + lerobot + air + pyxrt |
| `SMOLVLA_FORCE_COMPILE` | unset | all | `=1` rebuilds every ELF instead of reusing the cache |
| `SMOLVLA_CPU_THREADS` | `8` | NPU path | threads for the CPU stages: torch's, and numpy's BLAS (capped with `threadpoolctl` inside the NPU forward only; the pure-CPU arm keeps its defaults). `0` keeps the defaults |
| `SMOLVLA_NPU_ALL` | unset | all | `=1` is `--npu-all`: backbone and expert on the NPU too (what `make *-all` set) |

```bash
make verify                                  # the gate: synthetic, 3 cameras
make verify CAMERAS=1                        # synthetic, one camera
make verify INPUT=real                       # 100 real frames, reports a distribution
make verify INPUT=real FRAMES=20 CAMERAS=2
make run INPUT=real                          # one forward on one real frame
make profile CAMERAS=1 REPS=10               # timing; always synthetic
```

Only `make verify` and `make verify-all` on synthetic input are gates. They are
deterministic, exit non-zero on failure, and are what the lit tests run.
`INPUT=real` reports a distribution and always exits 0; see
[`correctness.md`](correctness.md).

---

## First run

```bash
make compile        # a few minutes; produces build/vision_kernel_cache*/
make verify         # no fixture needed: the CPU reference is computed live
```

`make verify` prints the NPU stages it used, the cosine, MSE, nMSE and largest
absolute difference against the CPU model with the thresholds next to them, and
ends with `[verify] PASS` or `[verify] FAIL`.

To sanity-check the harness itself, run the adapter directly with
`--cpu-vision`: it compares the unmodified model against its own baseline and
should score exactly 1.0. The Makefile has no pass-through for that flag.

```bash
$(LEROBOT_PYTHON) verify_adapter.py --cpu-vision
```

---

## `make profile`

Every arm is warmed with a discarded forward, then `REPS` reps of the arms run
interleaved so drift hits them alike, and the median is reported. The output has
an end-to-end table, a per-stage table and the NPU device time per ELF;
`make profile-all` adds the NPU vision + backbone and all-NPU arms.
[`profile.md`](profile.md) explains each table and the conditions that change
the numbers (power settings, other users of the CPU and NPU).

---

## The NPU lock

On a machine where several sessions share one NPU, take the project lock around
anything that touches the device:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
```

The recipes do not take it themselves; if they did, the command above would
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

First use downloads the whole dataset: video-backed LeRobot datasets store one
MP4 per camera per chunk, so a subset of frames cannot be fetched on its own.
`droid_100` is a few hundred MB; other datasets run into GBs.

Datasets need not match the checkpoint's dimensions. `droid_100` is 180×320 with
a 7-wide state against the checkpoint's 256×256 and 6; `resize_with_pad` and
`pad_vector` absorb both, and the NPU sees seq 1024 either way.
