# SmolVLA on AMD NPU2 (MLIR-AIR) — vision encoder on the NPU

End-to-end [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — a
Vision-Language-Action robot policy — running with its **SigLIP vision encoder
and connector on AMD NPU2 (AIE2P)** via MLIR-AIR, spliced into the unmodified
LeRobot pipeline. The first non-LLM model in `programming_examples/llms/`, built
on the same shared infrastructure (`../shared/`, `../verify/`) and the same
kernel registry as the Llama/Qwen siblings.

| Doc | |
|---|---|
| [`docs/usage.md`](docs/usage.md) | every command and what it does |
| [`docs/explain.md`](docs/explain.md) | how the implementation works |
| [`docs/correctness.md`](docs/correctness.md) | the gate, the thresholds, the measurements behind them |
| [`docs/profile.md`](docs/profile.md) | where the time goes, kernel by kernel |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | design notes and traps |

## What runs where, and why

SmolVLA is three stages. **All three were ported to the NPU and verified; only
the vision encoder runs there by default**, because it is the only one
measurably faster. The language backbone and the action expert have
**experimental, opt-in NPU paths** (see
[Experimental: backbone and expert on the NPU](#experimental-backbone-and-expert-on-the-npu));
on this machine they match the CPU, they do not beat it. The decision was made
stage by stage, by measurement.

| Stage | Shape · how often | CPU | NPU | Ships on |
|---|---|---|---|---|
| **① SigLIP vision + connector** | seq **1024**, hidden 768 · **×3 cameras** | ~234 ms | **~160 ms** | **NPU — ~1.47×** |
| ② Language backbone (SmolLM2-360M) | seq 241, hidden 960 · ×1 | **~47 ms** | ~42 ms (experimental) | CPU by default — NPU path ties (see below) |
| ③ Action expert (flow matching) | seq 50, hidden 720 · **×10 denoise steps** | **~192 ms** | ~156 ms (experimental) | CPU by default — NPU path at parity (see below) |

(CPU and vision numbers as of 2026-09-27, AMD Ryzen AI MAX+ 395 / NPU2, `make profile` REPS=15, averaged over 3 alternating pairs;
the experimental backbone/expert NPU numbers are from 2026-10-04 — see [Performance](#performance) below. Earlier revisions of this table were measured on a Ryzen AI 9 HX 370 and are
superseded.)

The NPU wins when shapes are large enough to fill its 8×4 compute array and
loses when they are not. Vision has 1024 tokens and 768/3072-wide matmuls; the
backbone has 256 tokens; the action expert has 50, which pads to 64 and leaves
half the array idle before a single instruction runs. Two further reasons the
small stages lose: every launch costs ~85 µs regardless of the work in it, and
the registry FlashAttention kernel applies no mask, so those two stages fall
back to attention decomposed into 11 dispatches per layer instead of 1.

The backbone and action-expert NPU paths live under `benchmarking/` and are
**off by default** (`--npu-backbone`, `--npu-expert`). The reasons above are
why the *early, unfused* ports of those two stages lost; the experimental paths
get around them (a masked FlashAttention kernel, and GEMM engines that run many
jobs or a whole layer in one launch) and reach parity, not a win — numbers
and prerequisites in the section below.

## Performance

One action chunk, end to end. **AMD Ryzen AI MAX+ 395 / NPU2, CPU governor and
EPP `performance`, NPU `pmode=Turbo`, machine idle** (an earlier revision of
this section was measured on a Ryzen AI 9 HX 370 — different machine, not
comparable). Reproduce with `make profile REPS=15` — one process, both arms
warmed, CPU/NPU pairs interleaved, median reported. Numbers below are the
one run of `make profile REPS=15`, 2026-10-04, on top of the accumulated GEMM/FA
kernel work (#2006, #2020–#2023), the zero-copy/CPU-thread/K-V-memo/
FlashAttention changes in this PR, and the host-side vision fixes below.

| Configuration | Action chunk | Vision stage | Speedup |
|---|---|---|---|
| Pure CPU (unmodified LeRobot) | ~483 ms | ~234 ms | 1.00× |
| **NPU vision + connector** | **~378 ms** | **~160 ms** | **~1.28×** |

Two host-side fixes (2026-10-04), both bit-identical in output (`make verify`
cosine unchanged to every digit): the patch embed's 1024-iteration Python slice
loop is one vectorised transpose (1.4 → 0.2 ms per image), and numpy's BLAS
threads are capped to `SMOLVLA_CPU_THREADS` (default 8) inside the NPU forward
only (`threadpoolctl`, see `requirements.txt`; the pure-CPU arm keeps its default
threads). With all 32 BLAS threads the host patch embed took 9 or 19 ms (tail
54 ms) and the vision stage ~185 ms; capped it is a steady ~160 ms. Without
`threadpoolctl` the cap is skipped with a warning.

Vision itself is ~1.47× (~53 vs ~78 ms per image) and is ~42% of the run; the
CPU backbone and expert (unchanged, still CPU-only) are the rest. NPU device
time is ~50 ms/image across the 5 vision ELFs, and the host side is ~10 ms of
the 160 ms stage (device was 136 ms/image before this line of optimization work — see
[`docs/profile.md`](docs/profile.md) for the historical per-ELF study and
`Vu_exp/smolvla_perf/PR_smolvla_perf.md` in the mlir-air repo for the current
per-commit breakdown).

### Experimental: backbone and expert on the NPU

Opt-in: the default `make run` / `make verify` / `make profile` stay vision-only
(the shipped gate covers that path). The all-NPU path has its own switch,
**`--npu-all`**, and matching targets that mirror the default ones:

| Command | What it does |
|---|---|
| `make run-all` | one forward, vision + backbone + action expert on the NPU (`smolvla_inference.py --npu-all`) |
| `make verify-all` | the same regression gate (cosine ≥ 0.99, nMSE ≤ 0.04) on the all-NPU path — PASS, cosine 0.9966, nMSE 0.0119 |
| `make profile-all REPS=15` | CPU vs NPU vision vs NPU vision+backbone vs all NPU, interleaved |
| `make compile-expert` | build the expert's two engine ELFs once (see the prerequisites below) |

`--npu-all` is `--npu-backbone --npu-expert` plus the expert without host K/V
packing; `SMOLVLA_NPU_ALL=1` is the environment form. If the expert ELFs are
missing it stops with a message that says how to build them. Numbers below:
one process, 15 interleaved reps, median ms, **2026-10-04**, same machine and
settings as above, from `make profile-all REPS=15` (average of two runs, with the
host-side vision fixes above). **Bold** = the stage runs on the NPU in that
configuration.

| Configuration | Vision (3 cam) | Language backbone | Action expert (10 steps) | End to end | Speedup | Chunk cosine vs CPU |
|---|---:|---:|---:|---:|---:|---:|
| Pure CPU (unmodified LeRobot) | 231.1 | 45.5 | 187.9 | 475.3 | 1.00× | reference |
| NPU vision; backbone, expert on CPU | **160.9** | 42.9 | 153.9 | 373.3 | 1.27× | 0.9988 |
| NPU vision + backbone; expert on CPU | **160.8** | **41.8** | 153.6 | 372.1 | 1.28× | 0.9953 |
| **All NPU** | **160.4** | **41.8** | **154.7** | 378.6 | 1.26× | 0.9963 |

End to end also contains 10–21 ms of host work (prompt/prefix embedding, glue).
The CPU stages are faster in the NPU rows than in the pure-CPU row because
those arms bind the CPU threads; read each stage against its own row. Latency
of one chunk (30 alternating runs, new observation each time): full NPU median
**380 ms** (p90 385, min 374, max 410, std 7.6) vs pure CPU 485 ms, 1.28×. Full-NPU
split: vision 160, expert 155, backbone 42, host 21 ms. Numbers drift ±15 ms between
sessions; compare only within a table.

Reading it honestly: the **~1.27× comes from vision**. The backbone ties the
CPU (41.8 vs 42.9 ms) and the expert is at parity (154.7 vs 153.9 ms; 189 ms
before the K/V work moved to the device). All-NPU therefore costs ~5 ms more
than NPU-vision-only (host layout, the first launch of each chunk) and leaves
the CPU free. Accuracy is lower than the
vision-only path (chunk cosine 0.9963 vs 0.9988, nMSE 0.0119; the expert alone
is 0.9999 per layer against the fp32 reference), still above the 0.99 gate.

How it works, briefly (details in `benchmarking/`):

* **Backbone**: one fused ELF per layer (RMSNorm, Q/K/V + RoPE, masked
  FlashAttention, O-proj, FFN as one GEMM engine), bfp16 weights, ~2.45 ms/layer
  on the device (`backbone_npu.py`, `backbone_runtime.py`).
* **Expert**: a GEMM *engine* runs all 16 layers as one launch; attention is
  expressed as GEMM jobs (scores with an `exp` drain, P·V with a divide drain).
  A separate *prefix engine* turns the backbone's K/V rows into K|V tiles once
  per chunk (cross layers: the expert's k/v projections; self layers: K rotated
  by −p0, V copied), and the step engine reads them as bf16 through a third
  argument that shares the prefix engine's buffer, so the host does no K/V
  packing (`expert_runtime_v2.py`, `expert_engine_probe.py`,
  `gemm_engine.py`).

Prerequisites beyond the vision path (why this stays experimental):

* The engines need a compiler that honours `air.order_drains` (drain ordering
  for launches whose later jobs read earlier jobs' outputs). That support is
  not in a released wheel or upstream `main`; it lives on the fork branch
  `smolvla-ship-compiler-fixes`.
* The expert engine must be built once, with a no-unroll Peano `opt` wrapper
  (the 16 KB core program does not fit the default unrolling):
  `make compile-expert` (= `python benchmarking/expert_v2_probe.py --layers 16
  --compile-only`), run with that compiler first on `PATH` (about
  2 minutes with the async-dependency speed-up, about an hour without; the patch
  is not upstream either). Self-attention layers are the even layers; an ELF
  built with the other pattern produces garbage (cosine −0.39).
* The backbone layer ELF is compiled lazily on the first `--npu-all` run and needs the same
  compiler first on `PATH` (with the stock wheel compiler it fails with `Basic sequential
  allocation also failed`); it is cached afterwards.
* The expert's per-call cost is almost all device time (13 ms of 14.6 ms): it
  streams ~243 MB of weights per call. Remaining levers are the drain-group
  stalls (~20 ms/chunk) and attention per-job cost.

## Correctness

`make verify` is the gate: it compares the final **action chunk** — the (1,50,6)
tensor the robot would execute — against the unmodified LeRobot CPU model, with
the flow-matching noise pinned so the comparison is deterministic. Thresholds
are cosine ≥ 0.99 and nMSE ≤ 0.04.

| Input | Cameras | Within threshold | cosine median | cosine worst |
|---|---|---|---|---|
| synthetic — **the gate** | 3 | **PASS** | 0.999328 | — |
| `droid_100`, 100 frames | **3 (shipping)** | **100/100** | 0.999752 | 0.997604 |
| `droid_100`, 100 frames | 2 | 98/100 | 0.999616 | 0.983156 |
| `droid_100`, 100 frames | 1 | 97/100 | 0.999142 | 0.893272 |

The experimental all-NPU path (`make verify-all`) passes the same gate at 3, 2 and 1
cameras and agrees on all 100 real frames at 3 cameras (median cosine 0.999628, worst
0.996188), but it is slightly less accurate than vision-only and degrades faster with
fewer cameras (94/100 at 1 camera); see
[`docs/correctness.md`](docs/correctness.md).

```bash
make verify                    # the gate — synthetic, deterministic, PASS/FAIL
make verify INPUT=real         # 100 recorded frames; reports, always exits 0
```

The shipping configuration agrees with the CPU model on every real frame. The
few frames below threshold at 1–2 cameras are downstream flow-matching
sensitivity rather than NPU error; [`docs/correctness.md`](docs/correctness.md)
has the evidence, along with the gate's design and where the thresholds come
from.

This says the port did not change the model's behaviour. It does not say the
model is good at the task.

## Model config

**Vision (on NPU):** SigLIP ViT, 12 layers, seq 1024 (512×512 image, patch 16),
hidden 768, MLP 3072, 12 heads × 64, **MHA, no mask**, affine LayerNorm
(eps 1e-6), GELU-tanh, every Linear has a bias. Connector: pixel-shuffle
(space-to-depth ×4, a pure reshape) then a 64×12288×960 projection.

**Downstream (on CPU, unmodified lerobot):** SmolLM2-360M backbone, 16 layers,
seq 241, hidden 960, GQA 15/5; action expert, 16 layers, hidden 720, 50 action
tokens, 10 denoise steps.

## Running it

Needs AMD NPU2 hardware, the MLIR-AIR environment, and one interpreter carrying
both `torch`+`lerobot` and `air`+`pyxrt` (`pip install -r requirements.txt`).
The checkpoint is public, so no `HF_TOKEN` is required.

```bash
make compile       # build every vision ELF — no NPU dispatch, no download
make verify        # THE GATE — action chunk vs the pure-CPU model
make run           # one end-to-end forward
make profile       # CPU vs NPU, interleaved and warmed
make cpu-baseline  # the unmodified CPU model on its own, for inspection
```

`run` and `verify` take `INPUT=synthetic|real` and `CAMERAS=1|2|3`; `profile`
takes `CAMERAS` only. Defaults are the shipping configuration. Setup, every
variable, and the NPU lock convention are in
[`docs/usage.md`](docs/usage.md).

## Files

The `smolvla_vision_*` trio is the whole NPU-mapped stage. Adding another stage
later means `smolvla_backbone_*` beside it, so the file names say which stages
are on the NPU.

| File | Role |
|---|---|
| `smolvla_vision_weights.py` | SigLIP config + weight loading from the checkpoint |
| `smolvla_vision_builders.py` | the two fused multi-launch ELF builders (`vit_ln_qkv`, `vit_o_ffn`) |
| `smolvla_fuse.py` | build-time choices read from the environment: the LayerNorm implementation (`SMOLVLA_LN_EXT`, C++ kernel by default) and the loop tilings of the two fused ELFs (`SMOLVLA_OFFN_TILING`, `SMOLVLA_LNQKV_TILING`, `12,6` by default) |
| `smolvla_vision_encoder.py` | the NPU driver: compiles the kernels, runs the 12 layers |
| `smolvla_dataset.py` | real observations from a LeRobot dataset, for `INPUT=real` |
| `smolvla_cpu_helpers.py` | fp32 numpy reference for every vision operation |
| `smolvla_runtime.py` | process-wide `VisionRuntime` singleton |
| `smolvla_inference.py` | splices the NPU vision result into lerobot's own `embed_prefix`; the single CLI entry point, including `--compile-only` |
| `smolvla_cpu_baseline.py` | runs the unmodified CPU model on its own and dumps the action chunk, for inspection (`make cpu-baseline`). `make verify` does not read it — it computes its reference live |
| `verify_adapter.py` | the regression gate |
| `ARCHITECTURE.md` | design notes: kernel sequence, fused ELFs, runtime flow, the traps |
| `docs/` | `usage.md`, `explain.md`, `correctness.md`, `profile.md` — see the table at the top |

## Kernels

Every kernel comes from the shared registry (`../../kernel_registry/`), and this
port added rows to it:

| Kernel | Shape | Note |
|---|---|---|
| GEMM (bf16→bf16, drain) | 1024×768×768 | q/k/v/out projections and the patch embedding |
| GEMM | 1024×768×3072 · 1024×3072×768 | MLP fc1 / fc2 |
| GEMM | 64×12288×960 | connector projection — M=64 forces `tile_m=16`, `herd_m=4` |
| FlashAttention (non-causal) | 1024², 12/12 MHA | fills the whole 8×4 array; the source of the vision win |
| **LayerNorm (affine)** | 1024×768 | **new registry page** |
| **GELU-tanh** | 1024×3072 | **new registry page** |
| EltwiseAdd | 1024×768 | bias-adds and residuals, on-device |
