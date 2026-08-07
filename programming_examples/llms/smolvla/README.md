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
the vision encoder ships there**, because it is the only one measurably faster.
The decision was made stage by stage, by measurement.

| Stage | Shape · how often | CPU | NPU | Ships on |
|---|---|---|---|---|
| **① SigLIP vision + connector** | seq **1024**, hidden 768 · **×3 cameras** | 544.8 ms | **453.6 ms** | **NPU — 1.20×** |
| ② Language backbone (SmolLM2-360M) | seq 241, hidden 960 · ×1 | **78.6 ms** | 229 ms | CPU — NPU ~3× slower |
| ③ Action expert (flow matching) | seq 50, hidden 720 · **×10 denoise steps** | **281.9 ms** | ~4× the CPU | CPU — NPU ~4× slower |

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

One action chunk, end to end. AMD Ryzen AI 9 HX 370 / NPU2 (Strix, AIE2P), CPU
governor and EPP `performance`, NPU `pmode=Turbo`, machine idle. Reproduce with
`make profile REPS=10` — one process, both arms warmed, CPU/NPU pairs
interleaved, median reported.

| Configuration | Action chunk | Vision stage | Speedup |
|---|---|---|---|
| Pure CPU (unmodified LeRobot) | 919.9 ms | 544.8 ms | 1.00× |
| **NPU vision + connector** | **831.6 ms** | **453.6 ms** | **1.106×** |

Vision itself is 1.20× (151 vs 182 ms per image) but only 55% of the run, so the
CPU backbone and expert cap the end-to-end gain at 1.84×. Fusion did most of the
work: 121 dispatches per image at 368 ms became **38 at 141.6 ms**, mostly by
moving bias-adds and residuals on-device and eliminating a bf16→f32→bf16 host
round-trip per operation. Full breakdown in
[`docs/profile.md`](docs/profile.md).

## Correctness

`make verify` is the gate: it compares the final **action chunk** — the (1,50,6)
tensor the robot would execute — against the unmodified LeRobot CPU model, with
the flow-matching noise pinned so the comparison is deterministic. Thresholds
are cosine ≥ 0.99 and nMSE ≤ 0.04.

| Input | Cameras | Within threshold | cosine median | cosine worst |
|---|---|---|---|---|
| synthetic — **the gate** | 3 | **PASS** | 0.998427 | — |
| `droid_100`, 100 frames | **3 (shipping)** | **100/100** | 0.999679 | 0.997360 |
| `droid_100`, 100 frames | 2 | 98/100 | 0.999435 | 0.987832 |
| `droid_100`, 100 frames | 1 | 97/100 | 0.998840 | 0.957068 |

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

Needs AMD NPU2 hardware, the MLIR-AIR environment, `HF_TOKEN`, and one
interpreter carrying both `torch`+`lerobot` and `air`+`pyxrt`
(`pip install -r requirements.txt`).

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
