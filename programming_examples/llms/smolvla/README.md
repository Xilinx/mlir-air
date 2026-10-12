# SmolVLA on AMD NPU2 (MLIR-AIR)

[SmolVLA](https://huggingface.co/lerobot/smolvla_base) is a
vision-language-action robot policy. This example runs it inside the unmodified
LeRobot pipeline with its SigLIP vision encoder and connector on AMD NPU2
(AIE2P), and optionally its language backbone and action expert too. It uses the
same shared infrastructure (`../shared/`, `../verify/`) and kernel registry as
the LLM examples beside it.

| Doc | |
|---|---|
| [`docs/usage.md`](docs/usage.md) | every command and variable |
| [`docs/explain.md`](docs/explain.md) | how the implementation works |
| [`docs/correctness.md`](docs/correctness.md) | the gate, its input and its thresholds |
| [`docs/profile.md`](docs/profile.md) | how to measure, and where the time goes |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | design notes and traps |

## What runs where

SmolVLA has three stages. By default only the vision encoder runs on the NPU;
`--npu-all` moves the other two there as well.

| Stage | Shape, how often | Default | With `--npu-all` |
|---|---|---|---|
| SigLIP vision + connector | seq 1024, hidden 768, once per camera | NPU | NPU |
| Language backbone (SmolLM2-360M) | seq 241, hidden 960, once | CPU | NPU |
| Action expert (flow matching) | seq 50, hidden 720, 10 denoise steps | CPU | NPU |

Vision is the stage the NPU is built for: 1024 tokens and 768/3072-wide matmuls
fill the 8×4 compute array, and its attention needs no mask, which is what the
registry FlashAttention kernel computes. The backbone and the expert are small
(the expert's 50 tokens pad to 64) and need masked attention, so per-launch
overhead dominates if they run as one launch per operation. The `--npu-all`
path avoids that with a masked FlashAttention variant and GEMM engines that run
many jobs, or a whole layer, in one launch.

Whether `--npu-all` beats the CPU backbone and expert depends on the host CPU:
the NPU stages change little from one NPU2 machine to another, the CPU stages
change a lot. Measure on the target machine.

## Performance

```bash
make profile          # pure CPU against NPU vision, interleaved in one process
make profile-all      # adds NPU vision + backbone, and every stage on the NPU
```

Both warm every arm, interleave the arms rep by rep and report medians, a
per-stage table and the NPU device time per ELF. Pin the CPU governor and the
NPU power mode before comparing runs; see [`docs/profile.md`](docs/profile.md).

The nightly benchmark records both configurations: `smolvla` is the action-chunk
latency with NPU vision, `smolvla_all` with every stage on the NPU.

## The all-NPU path

Opt-in. The default `make run`, `make verify` and `make profile` run vision only.

| Command | What it does |
|---|---|
| `make compile-expert` | build the action expert's two engine ELFs (needed once) |
| `make run-all` | one forward with every stage on the NPU (`smolvla_inference.py --npu-all`) |
| `make verify-all` | the same regression gate as `make verify`, on this path |
| `make profile-all` | `make profile` plus the two experimental arms |

`--npu-all` is `--npu-backbone --npu-expert`; `SMOLVLA_NPU_ALL=1` is the
environment form. The code is under `experimental/`.

- Backbone: one fused ELF per layer (RMSNorm, Q/K/V and RoPE, masked
  FlashAttention, O projection, FFN), bfp16 weights. It is compiled on the first
  `--npu-all` run and cached (`experimental/backbone_npu.py`,
  `experimental/backbone_runtime.py`).
- Action expert: a GEMM engine (`experimental/gemm_engine.py`) runs all 16
  layers in one launch, with attention as GEMM jobs (scores with an `exp`
  drain, P·V with a divide drain). A second, prefix engine turns the backbone's
  K/V rows into the expert's K and V tiles once per action chunk, and the step
  engine reads them from the buffer the two share, so the host does no K/V
  packing (`experimental/expert_runtime_v2.py`).

The expert engine's jobs read each other's outputs back from host memory, which
needs a compiler that orders those reads behind the transfers that write them;
current mlir-air does. Self-attention layers are the even ones: an engine built
with the other pattern produces garbage.

The all-NPU path is less accurate than vision only. The backbone and expert use
bfp16 weights and bf16 activations, and the error accumulates over the 10
denoise steps. It still passes the gate; see
[`docs/correctness.md`](docs/correctness.md).

## Correctness

`make verify` is the gate. It compares the final action chunk, the (1,50,6)
tensor the robot would execute, against the unmodified LeRobot CPU model, with
the flow-matching noise pinned so the comparison is deterministic. It passes at
cosine ≥ 0.99 and nMSE ≤ 0.04.

```bash
make verify                    # the gate: synthetic input, PASS/FAIL
make verify INPUT=real         # a survey over recorded frames; always exits 0
make verify-all                # the gate on the all-NPU path
```

This checks that the port did not change the model's behaviour, not that the
model is good at the task.

## Model config

Vision: SigLIP ViT, 12 layers, seq 1024 (512×512 image, patch 16), hidden 768,
MLP 3072, 12 heads × 64, MHA with no mask, affine LayerNorm (eps 1e-6),
GELU-tanh, a bias on every Linear. Connector: pixel-shuffle (space-to-depth ×4,
a pure reshape), then a 64×12288×960 projection.

Backbone: SmolLM2-360M, 16 layers, seq 241, hidden 960, GQA 15/5. Action expert:
16 layers, hidden 720, 50 action tokens, 10 denoise steps.

## Running it

Needs AMD NPU2 hardware, the MLIR-AIR environment, and one interpreter with both
`torch` + `lerobot` and `air` + `pyxrt` (`pip install -r requirements.txt`). The
checkpoint is public, so no `HF_TOKEN` is needed.

```bash
make compile       # build every vision ELF; no NPU dispatch, no download
make verify        # the gate
make run           # one end-to-end forward
make profile       # CPU against NPU, interleaved
make cpu-baseline  # the unmodified CPU model on its own
```

`run` and `verify` take `INPUT=synthetic|real` and `CAMERAS=1|2|3`; `profile`
takes `CAMERAS`. The defaults are the shipping configuration. Setup, every
variable and the NPU lock are in [`docs/usage.md`](docs/usage.md).

## Files

| File | Role |
|---|---|
| `smolvla_vision_weights.py` | SigLIP config and weight loading |
| `smolvla_vision_builders.py` | the two fused multi-launch ELF builders (`vit_ln_qkv`, `vit_o_ffn`) |
| `smolvla_fuse.py` | build-time choices read from the environment: the LayerNorm implementation (`SMOLVLA_LN_EXT`) and the loop tiling of the two fused ELFs (`SMOLVLA_OFFN_TILING`, `SMOLVLA_LNQKV_TILING`) |
| `smolvla_vision_encoder.py` | the NPU driver: compiles the kernels, runs the 12 layers |
| `smolvla_runtime.py` | the process-wide `VisionRuntime` |
| `smolvla_inference.py` | splices the NPU stages into LeRobot's own `embed_prefix`; the CLI, including `--compile-only` and `--npu-all` |
| `smolvla_dataset.py` | real observations from a LeRobot dataset, for `INPUT=real` |
| `smolvla_cpu_helpers.py` | fp32 numpy reference for every vision operation |
| `smolvla_cpu_baseline.py` | runs the unmodified CPU model alone and dumps its action chunk (`make cpu-baseline`); `make verify` does not read it |
| `verify_adapter.py` | the regression gate |
| `experimental/` | the backbone and action-expert NPU paths, their kernels and build probes |
| `run_npu2_*.lit` | the lit tests: compile, verify and profile; the `_all` ones verify and profile the all-NPU path |

## Kernels

The vision stage uses kernels from the shared registry
(`../../kernel_registry/`); this example added the LayerNorm and GELU-tanh pages.

| Kernel | Shape | Use |
|---|---|---|
| GEMM (bf16 in, bf16 out, drain) | 1024×768×768 | q/k/v/out projections, patch embedding |
| GEMM | 1024×768×3072, 1024×3072×768 | MLP fc1, fc2 |
| GEMM | 64×12288×960 | connector projection (M=64 needs `tile_m=16`, `herd_m=4`) |
| FlashAttention, non-causal | 1024², 12/12 MHA | attention; fills the 8×4 array |
| LayerNorm (affine) | 1024×768 | |
| GELU-tanh | 1024×3072 | |
| EltwiseAdd | 1024×768 | bias adds and residuals |
