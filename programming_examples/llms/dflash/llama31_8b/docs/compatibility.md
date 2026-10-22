# Toolchain and checkpoint compatibility

This example uses the standard AIR Python builders and `XRTBackend`, including
the Qwen3-8B DFlash infrastructure and FIFO MemTile-refeed ordering already
merged upstream in PRs #2145 and #2152. There is no additional AIR compiler
change for Llama.

The qualification environment uses these upstream dependency versions:

| Dependency | Version |
|---|---|
| MLIR-AIE | `1.4.5.dev51+ge040370` |
| Peano | `22.0.0.2026101201+83ba9ea9` |
| Host LLVM/MLIR | `24.0.0.2026100106+bdf04ebb` |
| Inline-attention linker | LLVM 21; upstream fused decode requires `llvm-link` <23 |
| XRT Python binding | 2.21; the supplied B1 comparator uses the xclbin path |

Use the same AIR environment as the other LLM examples. Install
[`requirements.txt`](../requirements.txt) for model preparation and the HF
reference; these are the same additional packages used by the Qwen examples.
No historical research checkout, generated-IR patch, private binary, or custom
XRT binding is needed. Decoder manifests record compiler/tool hashes, source
snapshots, geometry, ABI and binary hashes. Use a fresh `BUILD_DIR` after a
build-input change.

## Native interfaces reused

- `fused_decode.py` has explicit target/drafter Llama DFlash geometries. They
  select the role's FFN width, Q/K normalization and padded vocabulary geometry.
  The native autoregressive model entry is unchanged.
- `LlamaQ4nxPrefill` accepts an injected kernel cache and a host-scratch policy.
  DFlash streams one layer's BF16 weights and discards transient host scratch
  after its last reader. Native callers retain the existing defaults.
- `FusedDecode8B` accepts a weight-cache directory and exposes the cache path for
  the benchmark's provenance receipt. It streams the existing NPZ payload
  into the native weight buffers and checks every uploaded word, avoiding a
  full-model host copy. Its B1 dispatch arithmetic is unchanged.
  The shared native Llama packer writes NPZ members incrementally to avoid
  holding two packed-model copies during cold preparation. The packed words
  and file format are unchanged.

The native B1 xclbin path remains usable with XRT 2.21. Upstream's newer ELF
path requires a scratchpad Python binding absent from this XRT installation;
this example does not emulate that binding. See the shared
[8B compatibility notes](../../qwen3_8b/docs/compatibility.md) for the compiler
refeed fix and B1 timing mode.

## Download once, reuse offline

| Role | Repository | Fixed revision |
|---|---|---|
| Q4NX target | `FastFlowLM/Llama-3.1-8B-NPU2` | `527006483dca6c9d75474ff92b8a6bdd8f257914` |
| Tokenizer and HF BF16 reference | `NousResearch/Meta-Llama-3.1-8B-Instruct` | `d10aef7999a2b5ba950ab3974312feeedbfe0b77` |
| Official drafter | `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` | `d3af30def9601abdd10810aba220d692f0e803f0` |

`make download` fetches the pinned files, including reference-model shards.
`make weights` prepares checked target/draft packed caches and mapped BF16
prefill weights. Set `HF_HOME` or `HF_HUB_CACHE` for the download cache and
`DFLASH_CACHE_DIR` for the prepared weights. Run subsequent commands with
`HF_HUB_OFFLINE=1` and `OFFLINE=1`.

The fixed benchmark messages are versioned in
[`common/benchmark/prompts.json`](../../common/benchmark/prompts.json). Runtime
dataset downloads are unnecessary. `make full20` prepares model-specific NPU
prefill states once; a later `make full80` adds the remaining selected requests.
`BENCHMARK_BUNDLE` selects this persistent cache. Source, checkpoint, prompt and
prefill-binary identities select its generation, and saved state hashes are
checked before use.
