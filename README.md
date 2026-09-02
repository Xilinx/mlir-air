# MLIR-AIR

**An MLIR-based compiler for spatial accelerators. One architecture-independent
representation of hierarchical compute and asynchronous data movement, lowered
to AMD™ NPUs and GPUs.**

[![Build](https://img.shields.io/github/actions/workflow/status/Xilinx/mlir-air/buildAndTest.yml?branch=main&label=build&cacheSeconds=86400)](https://github.com/Xilinx/mlir-air/actions/workflows/buildAndTest.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Contributors](https://img.shields.io/github/contributors/Xilinx/mlir-air?cacheSeconds=86400)](https://github.com/Xilinx/mlir-air/graphs/contributors)

<img src="https://mlir.llvm.org/mlir-logo.png" width="180">

📖 **[Documentation](https://xilinx.github.io/mlir-air/)** &nbsp;·&nbsp; 🚀 **[Programming Guide](https://xilinx.github.io/mlir-air/dev/programming_guide/)** &nbsp;·&nbsp; 🧩 **[Compute Model](https://xilinx.github.io/mlir-air/dev/AIRComputeModel/)** &nbsp;·&nbsp; 💡 **[Examples](https://xilinx.github.io/mlir-air/dev/programming_examples/)**

The AIR dialect represents a design as a hierarchy of compute regions
(`air.launch`, `air.segment`, `air.herd`) over an explicit memory hierarchy
(L3, L2, L1). Data movement between levels is an explicit operation rather than
a side effect. Designs are written as imperative behavioral programs; the
compiler infers the dependencies between operations, represents them as
asynchronous tokens, and uses that graph to distribute computation across
hardware regions and overlap data movement with compute. The representation is
architecture-independent: `air-to-aie` lowers it to MLIR-AIE for Ryzen™ AI
NPUs, and `air-to-rocdl` lowers it to `gpu.launch` for AMD™ GPUs.

```python
from air import api as air
from air.api.types import dtype_of
import numpy as np

N, TILE, CORES = 4096, 512, 2
dt = dtype_of(np.float32)

A, B, C = air.tensor([N], dt), air.tensor([N], dt), air.tensor([N], dt)

with air.launch(name="eltwise_add") as launch:

    @launch.body
    def _():
        with air.segment(name="segment_0") as seg:

            @seg.body
            def _():
                l2_a = air.alloc([N], dt, scope=seg.private())     # L2
                l2_b = air.alloc([N], dt, scope=seg.private())
                l2_c = air.alloc([N], dt, scope=seg.private())

                air.ops.load(l2_a, A)                              # L3 -> L2
                air.ops.load(l2_b, B)

                with air.herd([range(0, N, TILE)], name="herd_0", shape=(CORES,)) as h:

                    @h.body
                    def _(tx):
                        i0 = tx * TILE
                        a = air.alloc([TILE], dt, scope=h.private())   # L1
                        b = air.alloc([TILE], dt, scope=h.private())
                        c = air.alloc([TILE], dt, scope=h.private())

                        air.ops.load(a, l2_a[i0 : i0 + TILE])          # L2 -> L1
                        air.ops.load(b, l2_b[i0 : i0 + TILE])
                        c[:] = a[:] + b[:]
                        air.ops.store(c, l2_c[i0 : i0 + TILE])

                air.ops.store(l2_c, C)                             # L2 -> L3

module = launch.build(target="npu2")
```

Two cores each add four 512-element tiles. `aircc` lowers the design through
MLIR-AIE and the [Peano](https://github.com/Xilinx/llvm-aie) compiler to an
`xclbin` and instruction stream that XRT runs on the NPU.

<details markdown="1">
<summary><b>More about the compiler</b></summary>

`air-opt` drives the pass pipeline and is the main tool for inspecting a design
between stages. `aircc` is the end-to-end compiler driver.
[`air-runner`](docs/AIRRunner.md) simulates a design and emits a Chrome trace,
so a schedule can be evaluated before hardware is involved.

MLIR-AIR is described in the following paper:

> E. Wang, S. Bayliss, A. Bisca, Z. Blair, S. Chowdhary, K. Denolf, J. Fifield,
> B. Freiberger, E. Hunhoff, P. James-Roxby, J. Lo, J. Melber, S. Neuendorffer,
> E. Richter, A. Rosti, J. Setoain, G. Singh, E. Taka, P. Vasireddy, Z. Yu,
> N. Zhang, J. Zhuang. "[From Loop Nests to Silicon: Mapping AI Workloads onto
> AMD NPUs with MLIR-AIR](https://arxiv.org/abs/2510.14871)". arXiv:2510.14871,
> October 2025.

</details>

## Install

Prebuilt wheels are the recommended path. Each guide also covers a source build.

| Host | Start here |
| --- | --- |
| Ryzen™ AI on Linux | [Ryzen AI (Linux)](https://xilinx.github.io/mlir-air/dev/buildingRyzenLin/) |
| Ryzen™ AI on Windows 11 | [Ryzen AI (Windows)](https://xilinx.github.io/mlir-air/dev/buildingRyzenWin/) |
| GPU on Linux | [GPU (Linux)](https://xilinx.github.io/mlir-air/dev/buildingGPU/) |

## Learn more

- [Programming Examples](https://xilinx.github.io/mlir-air/dev/programming_examples/) — operators and designs, with per-target test status
- [LLMs on NPU](https://xilinx.github.io/mlir-air/dev/llms/) — decoder-only models running end to end, with a nightly benchmark
- [AIR dialect and pass reference](https://xilinx.github.io/mlir-air/dev/mlir_reference/)
- [Used in / cited in](https://xilinx.github.io/mlir-air/dev/used_in/)

-----

<p align="center">Copyright&copy; 2018-2022 Xilinx, Inc.<br>Copyright&copy; 2022-2026 Advanced Micro Devices, Inc.</p>
