# SmolVLA on NPU2 — usage guide

## Prerequisites

### Hardware and toolchain
- AMD NPU2 hardware (Strix, AIE2P)
- MLIR-AIR installed with the Peano compiler (`PEANO_INSTALL_DIR` set)
- The project's standard environment: `source utils/env_setup.sh ...`

### Python environment
This example needs **one** interpreter that has both sides:

- `torch` + `lerobot` — to run the real SmolVLA policy
- `air` + `pyxrt` — to drive the NPU

That combination is what makes the single-process path work. `lerobot` is in
`requirements.txt` — it IS the CPU baseline this example verifies against, so
nothing needs to be fetched by hand.

Install into the environment that already has `air` + `pyxrt`:

```bash
pip install -r requirements.txt
make verify                      # default LEROBOT_PYTHON=python3
```

**One caveat.** lerobot requires `numpy>=2.0,<2.3`, and `utils/requirements.txt`
leaves numpy unpinned, so an older mlir-air environment may still be on 1.x —
installing will upgrade it across a major version in the interpreter every
other example shares. That upgrade was tested here (1.26.4 → 2.2.6, with
`qwen25_0_5b` and `qwen3_1_7b` `make verify` PASSing identically before and
after), but check your own models rather than taking that on faith.

To leave the shared environment untouched, use a separate venv instead.
Sourcing the mlir-air environment puts `air` and `pyxrt` on `PYTHONPATH` /
`LD_LIBRARY_PATH`, so they import from any venv:

```bash
python3 -m venv ~/smolvla-venv
~/smolvla-venv/bin/pip install -r requirements.txt
make verify LEROBOT_PYTHON=~/smolvla-venv/bin/python
```

### Model access
`HF_TOKEN` must be set; the example downloads `lerobot/smolvla_base` (450M
parameters) on first use.

---

## Targets

```bash
make help      # this list
make compile   # build every vision ELF — no NPU dispatch, no download
make cpu-baseline  # run the unmodified CPU model on its own, for inspection (no NPU)
make run       # one end-to-end forward; prints the action chunk
make verify    # THE GATE — action chunk vs the pure-CPU model (PASS/FAIL)
make profile   # CPU vs NPU, interleaved and warmed, with the per-ELF breakdown
make clean     # remove the kernel cache and build artifacts
```

### The NPU lock
On a machine where several sessions share one NPU, take the project's lock
around anything that touches the device — the same convention the sibling
examples use:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
flock -x -w 1800 /tmp/mlir-air-npu.lock make profile
```

The recipes deliberately do **not** take that lock themselves; if they did,
the command above would deadlock against itself.

Correctness does not depend on it: `shared/infra/cache.py` holds
`/tmp/npu.lock` around every dispatch, so concurrent runs interleave safely.
The outer lock matters for *timing* — it stops someone else's dispatches from
landing in the middle of your 38.

---

## First run

```bash
make compile        # a few minutes; produces vision_kernel_cache/
make verify         # no fixture needed — the gate computes its CPU reference live
```

Expected:

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

Running the adapter directly with `--cpu-vision` compares the unmodified model
against its own baseline and should score exactly 1.0 — a sanity check of the
harness itself. The Makefile has no pass-through for it:

```bash
$(LEROBOT_PYTHON) verify_adapter.py --cpu-vision
```

---

## Rebuilding kernels after an edit

The ELF cache is reused whenever its manifest resolves and contains every
expected kernel. The manifest does **not** track source hashes, so after editing
any kernel builder you must force a rebuild:

```bash
SMOLVLA_FORCE_COMPILE=1 make verify
```

---

## Environment variables

| Variable | Effect |
|---|---|
| `LEROBOT_PYTHON` | interpreter with torch + lerobot + air + pyxrt |
| `HF_TOKEN` | required for the checkpoint download |
| `SMOLVLA_FORCE_COMPILE=1` | rebuild every ELF instead of reusing the cache |

---

## Measuring honestly

Two things will corrupt a timing run on a shared machine:

1. **Other processes on the NPU.** `flock` is advisory — it only protects
   against processes that also take the same lock. Check with
   `fuser /dev/accel/accel0`.
2. **CPU/NPU power state.** Every number in this example was measured with the
   CPU governor and EPP at `performance` and the NPU at `pmode=Turbo`. A
   `balanced` machine reads very differently, and the CPU baseline moves more
   than the NPU stage does — which changes the *ratio*, not just the absolutes.

### What `make profile` reports

One process, both arms warmed with a discarded forward, then `REPS`
CPU/NPU pairs interleaved so drift hits both equally. `make profile REPS=10`
for a tighter estimate:

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

Two things worth reading off it:

- **Vision is 55% of the NPU run** (453.6 / 831.6). Even an infinitely fast
  vision stage would only reach 1.84x end to end; the CPU backbone and expert
  are the ceiling. That is why a 1.20x stage win shows up as 1.11x overall.
- **Vision is 93% device time** (420.5 / 453.6). Fusion has squeezed host glue
  down to 33 ms, so further gains have to come from the kernels — and
  `vit_o_ffn` alone is 47% of device time.

The backbone and expert columns carry the same number across both arms on
purpose: they are the same unmodified CPU code either way, and showing them
makes it visible that one stage moved and two did not.

---

## Choosing the input — `INPUT` and `CAMERAS`

`run` and `verify` take the same two knobs; `profile` takes `CAMERAS` only,
since pixel values do not change how much the NPU computes.

| Variable | Default | Applies to | Notes |
|---|---|---|---|
| `INPUT` | `synthetic` | run, verify | `synthetic` = seeded-random images, nothing to download. `real` = frames from a LeRobot dataset |
| `CAMERAS` | 3 | run, verify, profile | 1, 2 or 3. The model takes any non-empty subset and the NPU needs no recompile |
| `DATASET` | `lerobot/droid_100` | `INPUT=real` | any LeRobot dataset; camera keys are read from its metadata |
| `FRAMES` | 100 | `verify INPUT=real` | one per episode, from the middle of each |

```bash
make verify                          # the gate: synthetic, 3 cameras, PASS/FAIL
make verify CAMERAS=1                # synthetic, 1 camera
make verify INPUT=real               # 100 real frames, reports a distribution
make verify INPUT=real FRAMES=20 CAMERAS=2
make run INPUT=real                  # one forward on one real frame
make profile CAMERAS=1               # timing, always synthetic
```

**Only the default is a gate.** `make verify` with no arguments is
deterministic, exits non-zero on failure, and is what CI and the lit test run.
`INPUT=real` answers a different question -- how much headroom the precision
has on inputs the model will actually see -- and always exits 0. A few frames
below threshold at 1 or 2 cameras is a finding, not a broken build.

### Notes on `INPUT=real`

First use downloads the dataset. **There is no way to fetch a few frames**: one
MP4 per camera per chunk, so it is the whole file — 464 MB for `droid_100`,
1.3 GB for `aloha_mobile_wipe_wine`, 3.0 GB for `abc_130k_v3_smoke`.

`smolvla_dataset.load()` passes `video_backend="pyav"`, and that is not a
preference. LeRobot defaults to `torchcodec`, which fails to load here — it
cannot find a compatible FFmpeg/libtorchcodec for torch 2.10.0+cpu. `pyav`
comes with `lerobot[dataset]`. Every multi-camera LeRobot dataset is
video-backed, so this applies to all of them.

Dimensions need not match the checkpoint. `droid_100` is 180x320 with a 7-wide
state against the checkpoint's 256x256 and 6; `resize_with_pad` and
`pad_vector` absorb both, and the NPU sees seq 1024 either way. The state is
then semantically meaningless, which is fine for a CPU-vs-NPU numerical
comparison and would not be for a task-quality claim.
