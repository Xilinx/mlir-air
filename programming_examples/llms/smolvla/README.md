# SmolVLA on AMD NPU2 (MLIR-AIR) — vision encoder on the NPU

End-to-end [SmolVLA](https://huggingface.co/lerobot/smolvla_base) — a
Vision-Language-Action robot policy — running with its **SigLIP vision encoder
and connector on AMD NPU2 (AIE2P)** via MLIR-AIR, spliced into the unmodified
LeRobot pipeline. The first non-LLM model in `programming_examples/llms/`, built
on the same shared infrastructure (`../shared/`, `../verify/`) and the same
kernel registry as the Llama/Qwen siblings.

## What runs where, and why

SmolVLA is three stages. **All three were ported to the NPU and verified; only
the vision encoder ships there**, because it is the only one measurably faster.
The decision was made stage by stage, by measurement.

| Stage | Shape · how often | CPU | NPU | Ships on |
|---|---|---|---|---|
| **① SigLIP vision + connector** | seq **1024**, hidden 768 · **×3 cameras** | 546 ms | **465 ms** | **NPU — 1.19×/image** |
| ② Language backbone (SmolLM2-360M) | seq 256, hidden 960 · ×1 | **77 ms** | 229 ms | CPU — NPU ~3× slower |
| ③ Action expert (flow matching) | seq 50, hidden 720 · **×10 denoise steps** | **285 ms** | ~4× the CPU | CPU — NPU ~4× slower |

The NPU wins when shapes are large enough to fill its 8×4 compute array and
loses when they are not. Vision has 1024 tokens and 768/3072-wide matmuls; the
backbone has 256 tokens; the action expert has 50, which pads to 64 and leaves
half the array idle before a single instruction runs. Two further reasons the
small stages lose: every launch costs ~85 µs regardless of the work in it, and
the registry FlashAttention kernel applies no mask, so those two stages fall
back to attention decomposed into 11 dispatches per layer instead of 1.

The backbone and action-expert NPU paths are **not in this example**. They are
kept out to keep the contribution reviewable; they can land separately.

## Performance

End to end, one action chunk. AMD Ryzen AI 9 HX 370 / NPU2 (Strix, AIE2P), CPU
governor `performance` + EPP `performance`, NPU `pmode=Turbo`, machine idle,
configurations interleaved.

| Configuration | Action chunk | Speedup |
|---|---|---|
| Pure CPU (unmodified lerobot) | 913 ms | 1.00× |
| **NPU vision + connector** | **851 ms** | **1.07×** |

Reproduce with `make profile` (one process, both arms warmed, interleaved,
median of 10): 925.9 ms CPU against 856.5 ms, **1.081×**, with the vision stage
itself at 550.7 → 462.5 ms (**1.19×**) — the same numbers the table reports,
measured a second time through a different harness. See
[`docs/usage.md`](docs/usage.md) for the full output and what to read off it.

Vision alone is 1.19× per image (155 ms vs 184 ms). The end-to-end figure is
smaller because vision is 60% of the chunk: the ~87 ms saved is partly offset by
~6 ms of thread-pool re-entry on the CPU work that follows the NPU stage and
~19 ms of host-side tensor marshalling.

Getting there took fusion, not just kernels: the first working version issued
121 dispatches per image at 368 ms. Stitching each layer's operations into
**3 multi-launch ELFs** and moving the per-Linear bias-adds and both residual
adds on-device cut that to **38 dispatches and 141.6 ms** — 2.6× on identical
kernels, because it eliminated a bf16→f32→bf16 host round-trip per operation.

## Correctness

`make verify` is the gate. It compares the final **action chunk** — the (1,50,6)
tensor the robot would execute — against the **unmodified lerobot CPU model**,
with the flow-matching noise pinned to zero so the comparison is deterministic.

```
cosine 0.998915   (gate >= 0.99)
nMSE   0.003764   (gate <= 0.04)
PASS
```

SmolVLA emits a continuous action chunk rather than tokens, so this uses the
`regression_gate` in `../verify/comparators.py` rather than the top-k token-set
gate the autoregressive siblings use.

Note what this does and does not say: it says the port did not change the
model's behaviour. It does not say the model is good at the task — that would
need evaluation against recorded robot trajectories.

### On real frames

The gate runs on all-zero images, which is what makes it deterministic and
fast, and also means it is one reading from an input the model will never see.
`make verify INPUT=real` runs the same CPU-vs-NPU comparison on real recorded
observations, frame by frame, against the same thresholds.

100 frames of `lerobot/droid_100`, one from each of its 100 episodes:

| Cameras | **pass** | cosine median | P10 | **cosine worst** | nMSE worst |
|---|---|---|---|---|---|
| 1 | 97/100 | 0.998840 | 0.995269 | **0.957068** | 0.086860 |
| 2 | 98/100 | 0.999435 | 0.997137 | **0.987832** | 0.036063 |
| **3 (shipping)** | **100/100** | 0.999679 | 0.998782 | **0.997360** | 0.008050 |

**The shipping configuration passes every frame.** Agreement on real images is
in fact slightly better than on zeros, so the gate is not flattered by its
degenerate input.

The tail is worth knowing about. At 1 and 2 cameras a few frames fall below
cosine 0.99 (the run names them). And cosine is scale-blind on a physical
actuator command: the largest absolute per-dimension error was 1.05 on action
dim 5 at 1 camera, 0.36 at 3 — not something a cosine of 0.9974 conveys.

Fewer cameras is consistently worse, and it is a supported configuration rather
than a hypothetical: `smolvla_base` accepts any non-empty subset and the NPU
needs no recompile for it (`build_oracle_batch(n_cameras=...)`, since the count
is only how many times the host loop runs). A shorter prefix leaves the CPU
backbone and expert less to average the same vision error against.

```bash
make verify INPUT=real                  # 100 frames, 3 cameras
make verify INPUT=real CAMERAS=1        # same, one camera
```

`INPUT=real` always exits 0 — it reports headroom, it is not the gate. Plain
`make verify` (synthetic, 3 cameras) stays the pass/fail check, and is what CI
and the lit tests run so they need no dataset.

## Model config

**Vision (on NPU):** SigLIP ViT, 12 layers, seq 1024 (512×512 image, patch 16),
hidden 768, MLP 3072, 12 heads × 64, **MHA, no mask**, affine LayerNorm
(eps 1e-6), GELU-tanh, every Linear has a bias. Connector: pixel-shuffle
(space-to-depth ×4, a pure reshape) then a 64×12288×960 projection.

**Downstream (on CPU, unmodified lerobot):** SmolLM2-360M backbone, 16 layers,
seq 241, hidden 960, GQA 15/5; action expert, 16 layers, hidden 720, 50 action
tokens, 10 denoise steps.

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2 hardware, Peano, the project's
   standard env (`source utils/env_setup.sh ...`).
2. **A Python environment with both `torch`+`lerobot` and `air`+`pyxrt`.**
   This is what makes the single-process path work. Point `LEROBOT_PYTHON` at
   it; with the mlir-air environment sourced, a lerobot venv qualifies.
   ```
   pip install -r requirements.txt
   ```
3. `HF_TOKEN` for the `lerobot/smolvla_base` checkpoint download.

## Usage

```bash
make compile       # build every vision ELF — no NPU dispatch, no download
make verify        # THE GATE — action chunk vs the pure-CPU model
make run           # one end-to-end forward, prints the action chunk
make profile       # CPU vs NPU, interleaved and warmed, with the per-ELF breakdown
make cpu-baseline  # run the unmodified CPU model on its own, for inspection
```

`run` and `verify` take `INPUT=synthetic|real` and `CAMERAS=1|2|3`; `profile`
takes `CAMERAS` only. Defaults are the shipping configuration. See
[`docs/usage.md`](docs/usage.md).

On a machine where several sessions share one NPU, wrap the device-touching
targets the way the siblings are wrapped:

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
```

## Files

The `smolvla_vision_*` trio is the whole NPU-mapped stage. Adding another stage
later means `smolvla_backbone_*` beside it, so the file names say which stages
are on the NPU.

| File | Role |
|---|---|
| `smolvla_vision_weights.py` | SigLIP config + weight loading from the checkpoint |
| `smolvla_vision_builders.py` | the two fused multi-launch ELF builders (`vit_ln_qkv`, `vit_o_ffn`) |
| `smolvla_vision_encoder.py` | the NPU driver: compiles the kernels, runs the 12 layers |
| `smolvla_dataset.py` | real observations from a LeRobot dataset, for `INPUT=real` |
| `smolvla_cpu_helpers.py` | fp32 numpy reference for every vision operation |
| `smolvla_runtime.py` | process-wide `VisionRuntime` singleton; scoped BLAS-thread clamp |
| `smolvla_inference.py` | splices the NPU vision result into lerobot's own `embed_prefix`; the single CLI entry point, including `--compile-only` |
| `smolvla_cpu_baseline.py` | runs the unmodified CPU model on its own and dumps the action chunk, for inspection (`make cpu-baseline`). `make verify` does not read it — it computes its reference live |
| `verify_adapter.py` | the regression gate |
| `ARCHITECTURE.md` | design notes: kernel sequence, fused ELFs, runtime flow, the traps |
| `docs/` | `explain.md`, `profile.md`, `usage.md` — implementation, measurements, commands |

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
