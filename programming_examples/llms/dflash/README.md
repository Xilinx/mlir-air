# DFlash speculative decoding on NPU2

DFlash uses a small block-diffusion drafter to propose several tokens in one
forward pass. A target model verifies the block and commits its accepted prefix.
This example implements both model forwards and target-feature preparation on
AMD NPU2 with MLIR-AIR. The host manages tokenization, acceptance, and KV state.

| Example | Target | Drafter | Block size |
|---|---|---|---|
| [Qwen3-4B](qwen3_4b/README.md) | Qwen/Qwen3-4B, Q4NX | z-lab/Qwen3-4B-DFlash-b16, quantized decoder | 8 |
| [Qwen3-8B](qwen3_8b/README.md) | Qwen/Qwen3-8B, Q4NX | z-lab/Qwen3-8B-DFlash-b16, quantized decoder | 8 |
| [Llama-3.1-8B-Instruct](llama31_8b/README.md) | Llama-3.1-8B-Instruct, Q4NX | z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat, quantized decoder | 8 |

Start with the model's README for installation and commands. Each example ships one selected mapping and a pinned model pair.
The [shared package](common/README.md) contains the DFlash loop, decoder state
operations, arithmetic kernels, context prepass and benchmark harness. Model
geometry, resource ownership and packing remain explicit in each example.

- [Architecture](qwen3_4b/ARCHITECTURE.md): execution, data layout, and NPU mapping.
- [Code map](qwen3_4b/docs/code-map.md): file roles and build/runtime entry points.
- [B1 to B8 megakernel](qwen3_4b/docs/b8-megakernel.md): operator changes and the
  storage, layout and data-supply choices behind the selected mapping.
- [Compatibility](qwen3_4b/docs/compatibility.md): toolchain and compiler changes.
- [Benchmark](qwen3_4b/benchmark/README.md): fixed inputs and measurement boundary.

- [8B mapping and code map](qwen3_8b/docs/mapping.md): four-row owners, grouped BOs and shared code.
- [8B compatibility](qwen3_8b/docs/compatibility.md): independent MemTile replay fix and standard upstream interfaces.

References: [DFlash project](https://z-lab.ai/projects/dflash/),
[reference implementation](https://github.com/z-lab/dflash), and the
[released drafter](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16).
