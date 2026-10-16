# Qwen3-4B DFlash B8

Run Qwen3-4B with greedy, thinking-disabled speculative decoding on AMD NPU2.
The selected mapping uses cached normalized activations, parallel attention
shards, native K layout, and reuse of the attention P/V operands.

**Historical reference:** the selected implementation completed the fixed 80-prompt study at 40.045 decode tokens/s
versus 19.310 tokens/s for its historical AIR B1 control (2.074×). Those numbers
belong to the earlier toolchain; they are not measurements of this port.

## Environment

Use the same MLIR-AIR build environment as the other `llms` examples. Source
`utils/env_setup.sh` with your AIR, MLIR-AIE, Peano, and LLVM installations and
source XRT's setup script. Build AIR with the accompanying compiler fixes
before compiling this example. Install its additional Python packages:

```sh
cd programming_examples/llms/dflash/qwen3_4b
python -m pip install -r requirements.txt
make help
```

The checked upstream dependency versions and any required compiler changes are
listed in [compatibility.md](docs/compatibility.md). There is no DFlash-specific
replacement toolchain or compiler downloaded by `make`.

The implementation targets NPU2's eight-column partition, uses Peano and XRT,
and shares `fused_decode` primitives and the upstream Qwen3-4B NPU prefill.
Use a host with enough unified memory for the target, drafter, and prefill
weights; the reference machine has 96 GB. Initialization prepares prefill first,
then loads the resident decoders. MLP weights share their uploaded buffers, and
prefill/decode share the identical embedding. Verification loads its CPU
reference in a separate process. Disk space is also needed for the Hugging Face
checkpoints, packed caches, and compiler intermediates.

## Commands

```sh
make download          # Seed pinned models/reference and Full80 dataset once
make weights           # Pack persistent target/drafter caches
make compile           # Target + drafter + NPU prepass + NPU prefill
make run               # Stream one response
make run PROMPT="Explain speculative decoding." N_TOKENS=128
make chat              # Prepare once, then accept and stream prompts
make verify-host       # Acceptance, EOS and streaming checks without the NPU
make verify            # DFlash state checks and HF BF16 top-5 comparison
make benchmark         # Fixed 80-prompt DFlash decode benchmark
make compile-baseline  # Current upstream B1, for the paired study
make full80            # Paired B1 + DFlash, identical 80 inputs
make profile           # Short fixed-prompt NPU prefill + decode profile
```

NPU execution is serialized with `/tmp/mlir-air-npu.lock`. `chat` treats each prompt as an
independent request. Output appears as each verified token block is committed.

## Models and arithmetic

| Component | Source and computation |
|---|---|
| Target | `FastFlowLM/Qwen3-4B-NPU2` Q4NX bundle for `Qwen/Qwen3-4B`; 36 decoder layers |
| Drafter | `z-lab/Qwen3-4B-DFlash-b16`; five layers, executed with block size 8 |
| Decoder projections | Group32 affine Q4 weights; BF16 activations; BFP16 matrix products with FP32 accumulators |
| Context preparation | Official BF16 projection weights; BF16/BFP16 matrix products, normalization and RoPE |
| Host loop | Greedy acceptance; no sampling or thinking mode |

The checkpoint is trained at block size 16; this example uses its first eight
positions. Each round processes an anchor and seven proposed positions. The
number of newly emitted tokens is between one and eight.

`model_paths.py` records exact checkpoint revisions. Packed caches default to
`~/.cache/mlir-air/dflash/qwen3_4b/` and can be overridden with
`DFLASH_TARGET_CACHE` and `DFLASH_DRAFT_CACHE`. The target packing intentionally
reproduces the historical DFlash study's BF16-dequantize/requantize arithmetic;
current upstream B1 can instead pack the bundle's raw Q4 values. These caches
are kept separate. The drafter decoder is quantized; BF16 describes its source
checkpoint and context-preparation weights, not all decoder projections.

## What to read

- [ARCHITECTURE.md](ARCHITECTURE.md): model flow, state and tile responsibilities.
- `speculative.py`: complete greedy draft/verify/commit loop.
- `runtime.py` and `target_runtime.py`: model setup and device invocation.
- `batch_graph.py`, `fused_attention_graph.py`, `kernels/`: selected megakernel.
- `build_bfp_prepass.py` and `bfp_context_prepass.py`: context preparation.
- [Benchmark inputs](benchmark/README.md): readable prompts and measurement.
- [Compatibility](docs/compatibility.md): current upstream integration evidence.
- [Validation](docs/validation.md): checks, reports and timing boundaries.
- [Nightly CI](docs/ci.md): suite entrypoints, pinned resources, caching and budgets.

The same-arithmetic sequential oracle checks speculative logic independently
of B1's GEMV arithmetic. The upstream model comparator additionally checks
output tokens against a BF16 reference. These checks have different scopes;
passing a token comparator alone does not establish repeatability or correct
KV commit behavior.
