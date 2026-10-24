# Qwen3-8B DFlash on NPU2

This example runs Qwen3-8B with the official `z-lab/Qwen3-8B-DFlash-b16`
drafter. Target prefill, both multi-token decoder forwards, and target-feature
preparation run on the NPU. The CPU tokenizes prompts, selects tokens, applies
greedy acceptance, and manages the committed KV positions.

B8 is the default: one known token and seven draft candidates. Use
`BLOCK_SIZE=16` for one known token and fifteen candidates, matching the
drafter checkpoint's training block size. See [the B16 mapping](docs/b16-megakernel.md)
for its storage design and compiler requirements.

## Get started

Use the normal MLIR-AIR environment, including XRT, MLIR-AIE, Peano and LLVM.
Build AIR from this branch, then source `utils/env_setup.sh` as for the other
`llms` examples. The existing `fused_decode` preflight also requires a pre-23
`llvm-link` for inline attention. No replacement compiler is downloaded by Make.
See [compatibility](docs/compatibility.md) for the compiler change in this branch.

```sh
cd programming_examples/llms/dflash/qwen3_8b
python -m pip install -r requirements.txt
make download       # Once: pinned target, drafter, tokenizer and HF reference
make weights        # Once: decode packing and mapped BF16 prefill cache
make compile        # Target, drafter, NPU prepass and C2048 NPU prefill
make run PROMPT="Explain speculative decoding." N_TOKENS=128
make chat           # Load once, then stream responses to independent prompts
```

The NPU needs its full eight-column NPU2 partition. The development machine has
96 GB unified memory. Keep disk space for the original checkpoints, packed
weights, mapped BF16 prefill tensors and build intermediates. HF reference
verification runs in a separate process from the NPU models.

The NPU decode weights occupy approximately 4.74 GB for the target and 1.00 GB
for the drafter, in addition to embeddings, KV, feature-preparation weights and
prefill buffers. Every physical BO stays below the 4 GiB DMA address limit.
Prefill streams one layer's weights through reusable device buffers; it does
not allocate a resident device weight set for all 36 layers.

Pass `BLOCK_SIZE=16` to `compile`, `run`, `chat`, `verify`, `profile`,
`compile-baseline`, `full20`, `full80` and `benchmark` to use B16. The packed
weight and prompt caches are shared between configurations.

## Models and caches

| Resource | Revision |
|---|---|
| `FastFlowLM/Qwen3-8B-NPU2` target Q4NX | `941c148c228496682e95c413cb90025f73200af2` |
| `Qwen/Qwen3-8B` tokenizer and HF BF16 reference | `b968826d9c46dd6066d109eabc6255188de91218` |
| `z-lab/Qwen3-8B-DFlash-b16` official drafter | `9b41424b7109f9c5413454f481b09a82b85333f4` |

Target projections preserve the original Q4NX blocks, including their scales
and offsets; preparation reorders these bytes for the projection streams. Drafter
projection weights are quantized to group-32 Q4NX for the decoder. Feature
preparation uses the official BF16 FC and K/V weights with the standard BFP
matrix primitive. Both decoders use the target's **untied** vocabulary head;
input embedding sharing never substitutes that head.

| Cache | Default | Override |
|---|---|---|
| Original snapshots | Standard Hugging Face cache | `HF_HOME` / `HF_HUB_CACHE` |
| Packed target | `~/.cache/mlir-air/dflash/qwen3_8b/target_q4nx` | `DFLASH_TARGET_CACHE` |
| Packed drafter | Same root, `draft_q4nx` | `DFLASH_DRAFT_CACHE` |
| Mapped prefill weights | Same root, `prefill_bf16` | `DFLASH_PREFILL_WEIGHTS` |
| Native B1 weights | Same root, `baseline_<identity>.npz` | Root: `DFLASH_CACHE_DIR` |
| Model-specific prompt states | Same root, `benchmark-npu-v1` | Make: `BENCHMARK_BUNDLE` |
| Compiled programs | `build/` (B8), `build_b16/` (B16) | Make: `BUILD_DIR` |

`DFLASH_CACHE_DIR` changes the weight-cache root. `DFLASH_DRAFT_SNAPSHOT` can
point to a local copy of the pinned drafter. After seeding, use
`HF_HUB_OFFLINE=1` and `OFFLINE=1`. Preparation checks file identities; an
incomplete or incompatible cache fails with a message instead of being reused.
Choose a fresh cache/build directory after changing its inputs.

## Verification and benchmarks

```sh
make verify             # Device state/loop checks and HF BF16 top-5 comparison
make profile            # Short fixed prompt; actual NPU prefill + decode
make compile-baseline   # Existing upstream Qwen3-8B B1 xclbin decoder
make full20             # Nightly cohort: four prompts from each of five datasets
make full80             # Optional: sixteen prompts from each dataset
make benchmark          # DFlash-only Full80
```

The first study prepares and caches Qwen3-8B NPU-prefill states for the requested
cohort. A cached Full80 cohort also serves Full20. Both B1 and DFlash consume
identical saved KV and first-token states, in separate processes. The B1 weight
cache uses the original B1 layout rather than the B8 projection layout.

Read the actual messages in [the shared prompt manifest](../common/benchmark/prompts.json).
The cohort is fixed: seed 0, GSM8K, MATH-500, HumanEval, MBPP and **first-turn**
MT-Bench. Full20 selects the first four of the frozen sixteen cases in each
dataset. Both studies use greedy, thinking-disabled decoding, a 2048-token cap,
EOS stopping and one pass per engine. This is a fixed subset study, not the
full paper evaluation or two-turn MT-Bench.

Primary decode timing excludes saved prefill, model loading, reset, warmup and
the first token produced by prefill. It includes **initial and subsequent NPU
prepass**, draft/verify dispatches, acceptance and CPU work. Pooled throughput
is total emitted decode tokens divided by total decode seconds. Speedup divides
fresh DFlash throughput by fresh B1 throughput on the same cohort. Per-dataset
throughput, emitted tokens/round, acceptance, individual outputs and round traces
are retained. A paper-era timer that excludes the first preparation/draft is
reported separately; it is not the primary speedup metric.

Outputs default to `build/qwen3_8b_dflash_b8.full20.json` (or `.full80.json`).
For a repeat, choose a fresh output name:

```sh
make full20 FULL20_OUTPUT=build/repeat2.full20.json
```

Read [the mapping and code map](docs/mapping.md),
[compatibility](docs/compatibility.md), and [CI/validation](docs/validation.md)
for the source paths, numerical gates and suite integration.
