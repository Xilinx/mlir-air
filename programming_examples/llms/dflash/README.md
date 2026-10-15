# DFlash speculative decoding on NPU2

DFlash uses a small block-diffusion drafter to propose several tokens in one
forward pass. A target model verifies the block and commits its accepted prefix.
This example implements both model forwards and target-feature preparation on
AMD NPU2 with MLIR-AIR. The host manages tokenization, acceptance, and KV state.

| Example | Target | Drafter | Block size |
|---|---|---|---|
| [Qwen3-4B](qwen3_4b/README.md) | Qwen/Qwen3-4B, Q4NX | z-lab/Qwen3-4B-DFlash-b16, quantized decoder | 8 |

Start with the model's README for installation and commands. The first example
ships one selected mapping; experimental mappings and previous performance
studies are not required to build it. Additional models will get separate
subdirectories with their own checkpoint and geometry declarations.

- [Architecture](qwen3_4b/ARCHITECTURE.md): execution, data layout, and NPU mapping.
- [Compatibility](qwen3_4b/docs/compatibility.md): toolchain and compiler changes.
- [Benchmark](qwen3_4b/benchmark/README.md): fixed inputs and measurement boundary.

References: [DFlash project](https://z-lab.ai/projects/dflash/),
[reference implementation](https://github.com/z-lab/dflash), and the
[released drafter](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16).
