# AIRCC - The AIR Compiler Driver

## Overview
`aircc` is a compiler driver processing AIR programs. It takes the `AIR` MLIR
dialect as input and produces NPU binaries (xclbin + instruction stream, PDI, or
full ELF) via `aiecc`. It can also lower AIR to ROCDL for AMD GPUs
(`--target=gpu`).

## Usage

```
aircc --device=<npu1|npu2|...> [-o OUTPUT] [-i INSTS] [--tmpdir DIR] [-v]
      [--output-format=xclbin|txn|pdi|elf|none] [other options] air_mlir_file
```

`--device` is required for the AIE target and must name an NPU device. Run
`aircc --help` for the full option list.

## Compilation pipeline

1. **Placement pipeline**: AIR optimization passes (`air-dependency`,
   `air-dma-to-channel`, `air-fuse-channels`, ping-pong transform, ...) followed
   by herd placement (`air-place-herds`). Output: `placed.<input>`.
2. **AIR to AIE**: `air-to-aie` and `air-merge-unrolled-devices` generate the
   `aie.device` module. Output: `aie.<input>`.
3. **Runtime sequence**: `air-opt-shim-dma-bds`, `air-to-std` and
   `airrt-to-npu` lower the host-side control code to an NPU runtime sequence.
   Output: `npu.<input>`.
4. **Backend**: `aiecc` compiles cores and emits the requested binary format.

With `--debug-ir`, every intermediate pass result is written to
`<tmpdir>/debug_ir/`.
