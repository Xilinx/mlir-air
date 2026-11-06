# Nemotron-3.5-Bolt-E4B DFlash on NPU2

This example runs NVIDIA's Nemotron-3.5-Bolt-E4B, a Gemma4-E4B-architecture
model, with NVIDIA's DFlash drafter. Prompt prefill, the eight-token target
verification and the drafter run on the NPU. The CPU tokenizes prompts,
projects the drafter's context K/V from the target's features, applies greedy
acceptance and manages the committed KV positions.

Each round is B8: the last committed token and seven draft candidates, checked
by the target in one pass.

## Get started

Use the normal MLIR-AIR environment, including XRT, MLIR-AIE, Peano and LLVM,
and source `utils/env_setup.sh` as for the other `llms` examples. The drafter
reuses `dflash/qwen3_4b`'s graph, whose `fused_decode` preflight requires a
pre-23 `llvm-link`.

```sh
cd programming_examples/llms/dflash/gemma4_bolt_e4b
python -m pip install -r requirements.txt
make download       # Once: pinned target package and drafter
make weights        # Once: the drafter's packed device weights
make compile        # Target, drafter, prepass record and NPU prefill
make run PROMPT="Explain speculative decoding." N_TOKENS=128
make chat           # Load once, then stream responses to independent prompts
```

The NPU needs its full eight-column NPU2 partition. The target packs its
weights from `model.q4nx` when it loads; the drafter's packed weights are
cached. Host matrix products run with four threads by default
(`OMP_NUM_THREADS`): more threads compete with the NPU's weight stream for
memory bandwidth.

## Models and caches

| Resource | Revision |
|---|---|
| `FastFlowLM/Nemotron-3.5-Bolt-E4B-NPU2` target Q4NX, tokenizer and chat template | `f9ee21728399b0d570dd40273abe01f7662cbab8` |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash` drafter (BF16 GGUF) | `0949a2eb7152ec6fda1a91ff9302aecbcd464fbf` |

Both checkpoints are distributed under NVIDIA's OpenMDW-1.1 license.

The target's projections use the package's Q4NX blocks; its per-layer
embedding gate and projection, which ship in BF16, are packed as two Q4 terms
each. The drafter's layers are requantized from BF16 to group-32 Q4. The
drafter has no embedding or vocabulary head of its own and uses the target's.

| Cache | Default | Override |
|---|---|---|
| Original snapshots | Standard Hugging Face cache | `HF_HOME` / `HF_HUB_CACHE` |
| Packed drafter | `~/.cache/mlir-air/dflash/gemma4_bolt_e4b/draft_b8` | `DFLASH_DRAFT_CACHE` |
| B1 baseline weights | Same root, `baseline_wcache` | Root: `DFLASH_CACHE_DIR` |
| Model-specific prompt states | Same root, `benchmark-npu-v1` | Make: `BENCHMARK_BUNDLE` |
| Compiled programs | `build/` | Make: `BUILD_DIR` |

After seeding, use `HF_HUB_OFFLINE=1` and `OFFLINE=1`. The packed drafter's
file name carries a hash of its packing code and checkpoint revisions, so a
change packs a new one.

## Verification and benchmarks

```sh
make verify             # Target, long-context, attention, loop and top-5 model checks
make profile            # Short fixed prompt; NPU prefill + DFlash decode
make compile-baseline   # llms/gemma4_e4b_q4nx's single-token decoder, 4096 positions
make full20             # Nightly cohort: four prompts from each of five datasets
make full80             # Optional: sixteen prompts from each dataset
make benchmark          # DFlash-only Full80
```

## Two independent sequences

`build_target.py --batch 2 --sequences 2` builds the same target for two
independent sequences: each dispatch takes one token of each, and each
sequence has its own KV cache and position. A step can leave a sequence out;
its row then costs no projection work and its cache is not touched.

```sh
make compile-sequences  # PREFILL_CTX positions per sequence
make verify-sequences   # each step's logits against each sequence's float32 forward
```

`GemmaTarget.step(tokens, positions, active)` drives it. See
[the mapping](docs/mapping.md#two-independent-sequences).

There is no BF16 checkpoint of this model, so the model check compares the
NPU's tokens with a float32 forward of the target package's own weights
(`reference_b8.py`) rather than with Hugging Face. See
[validation](docs/validation.md) for each check and its gate.

The benchmark studies use the shared fixed cohort and timing of the other
DFlash examples (see [qwen3_8b](../qwen3_8b/README.md)): greedy,
thinking-disabled decoding, a 2048-token cap, EOS stopping and one pass per
engine, both engines on identical saved prefill states. The B1 baseline is
`llms/gemma4_e4b_q4nx`'s single-token decoder with this model's weights.

Read [the mapping](docs/mapping.md) for the target graph, the drafter and the
host side.
