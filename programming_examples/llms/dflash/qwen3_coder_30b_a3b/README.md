# Qwen3-Coder-30B-A3B DFlash on NPU2

This example runs Qwen3-Coder-30B-A3B-Instruct, a mixture-of-experts model
(48 layers, 128 experts, 8 per token), with the official
`z-lab/Qwen3-Coder-30B-A3B-DFlash` drafter. Prompt prefill, the eight-token
target verification and the drafter run on the NPU. The CPU tokenizes prompts,
projects the drafter's context K/V from the target's features, applies greedy
acceptance and manages the committed KV positions.

Each round is B8: the last committed token and seven draft candidates, checked
by the target in one pass.

## Get started

Use the normal MLIR-AIR environment, including XRT, MLIR-AIE, Peano and LLVM,
and source `utils/env_setup.sh` as for the other `llms` examples. The drafter
runs on the shared B8 decoder, whose `fused_decode` preflight requires a
pre-23 `llvm-link`.

```sh
cd programming_examples/llms/dflash/qwen3_coder_30b_a3b
python -m pip install -r requirements.txt
make weights        # Once: the target's Q4_K bundle and the packed drafter
make compile        # Target (one build per cache read), LM head and drafter
make run PROMPT="Write a Python function that merges two sorted lists." N_TOKENS=128
make chat           # Load once, then stream responses to independent prompts
```

The NPU needs its full eight-column NPU2 partition. The target's weights take
about 17 GB of device memory and stay resident; the first load lays them out
for the verify pass and caches the result. Prefill runs on the same verify
pass, eight prompt tokens per dispatch, so it needs no build or weights of its
own.

## Models and caches

| Resource | Revision |
|---|---|
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` BF16 checkpoint and tokenizer | `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` |
| `z-lab/Qwen3-Coder-30B-A3B-DFlash` drafter | `98ca0e3e2e6a372f2789d3a5e146566194084317` |

`make weights` quantizes the target once into the Q4_K bundle of
[qwen3_coder_30b_a3b_q4k](../../qwen3_coder_30b_a3b_q4k), whose B1 decode is
also the benchmark baseline. The drafter's layers are requantized from BF16 to
group-32 Q4NX; it has no embedding or vocabulary head of its own and uses the
target's.

| Cache | Default | Override |
|---|---|---|
| Original snapshots | Standard Hugging Face cache | `HF_HOME` / `HF_HUB_CACHE` |
| Target Q4_K bundle | `$HF_HOME/qwen3_coder_30b_a3b.q4k` | `QWEN3C_BUNDLE` |
| Target in the verify pass's layout | `~/.cache/mlir-air/dflash/qwen3_coder_30b_a3b/target_tiled` | `DFLASH_TARGET_CACHE` |
| Packed drafter | Same root, `draft_q4nx` | `DFLASH_DRAFT_CACHE` |
| Model-specific prompt states | Same root, `benchmark-npu-v1` | Make: `BENCHMARK_BUNDLE` |
| Compiled programs | `build/` | Make: `BUILD_DIR` |

`DFLASH_CACHE_DIR` changes the cache root. Each cache records what it was made
from and is rebuilt when that changes.

## Verification and benchmarks

```sh
make verify             # The DFlash loop against its oracle; the target against the CPU reference
make profile            # Fixed coding prompt; NPU prefill + DFlash decode
make compile-baseline   # llms/qwen3_coder_30b_a3b_q4k's B1 decode, 4096 positions
make full20             # Nightly cohort: four prompts from each of five datasets
make full80             # Optional: sixteen prompts from each dataset
make benchmark          # DFlash-only Full80
```

`make verify` runs two checks. The loop check compares DFlash with the same
target run one token at a time. The verify pass routes all eight rows through
one union of experts and sums each row's experts in the order that union
lists them, so a row can round differently in a block than alone. DFlash must
match the oracle up to the oracle's first tie, two logits one BF16 step apart,
pick one of the tied tokens there, and keep the committed KV within 1/8 of
each layer's largest value until then. The model check runs all 48 layers on
eight tokens and compares their argmax with the float64 CPU reference.

The benchmark studies use the shared fixed cohort and timing of the other
DFlash examples (see [qwen3_8b](../qwen3_8b/README.md)): greedy decoding, a
2048-token cap, EOS stopping and one pass per engine, both engines on
identical saved prefill states.

Full20 on one Strix NPU2 machine (decode tokens per second):

| Dataset | B1 | DFlash | Speedup | Tokens per round |
|---|---|---|---|---|
| GSM8K | 21.8 | 31.1 | 1.43x | 4.08 |
| MATH-500 | 21.1 | 36.0 | 1.71x | 4.58 |
| HumanEval | 20.8 | 41.1 | 1.97x | 5.86 |
| MBPP | 20.7 | 39.3 | 1.90x | 5.81 |
| MT-Bench | 21.3 | 11.9 | 0.56x | 1.44 |
| All | 21.0 | 33.4 | 1.59x | 4.53 |

The drafter is trained on code: on general chat it accepts few tokens and is
slower than B1 decode. `make profile` decodes a fixed coding prompt at about
41 tokens per second.

## Mapping

The verify pass is one dispatch of 48 waves, one per layer. A wave runs the
layer's attention, then its MoE, over the eight rows:

- Columns 0, 1, 6 and 7 hold the Q4_K projection array. It runs the dense
  projections, then the MoE in turns of four experts, one per column.
- A planner core routes on the device: router logits, each row's top 8, and
  the union of the block's experts. A router core starts each expert's weight
  transfer itself, two turns ahead.
- Columns 3 and 4 hold four attention units, one KV head each, reading the
  cache back in 16-key blocks.
- Residual cores keep each row's hidden state between layers.

A second dispatch runs the 8-row Q8 LM head. Each cache length the pass reads
(256, 512, 1024, 2048 or 4096 positions) is its own build, and a block uses
the shortest that covers it. While a verify dispatch runs, the host reads each
tap layer's output as soon as it is written and projects it into the
drafter's context K/V, so that work is off the critical path.
