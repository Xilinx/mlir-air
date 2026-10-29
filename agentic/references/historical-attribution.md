# Attribution and public source references

The MLIR-AIR agentic harness v0.1 extends the workflow introduced in:

> Jiajie Li, Erwei Wang, Zhiru Zhang, and Samuel Bayliss. **From Human Guidance
> to Autonomy: Agent Skill System for End-to-End LLM Deployment on Spatial
> NPUs.** MLArchSys'26, co-located with ISCA 2026.
> [arXiv:2606.07586](https://arxiv.org/abs/2606.07586).

The preceding work organizes deployment around reference preparation, component
and model checks, optimization, debugging, and final evaluation. This release
connects that workflow to the current MLIR-AIR compilation stack with a shared
Codex/Claude skill payload, an HF-ID task entry, explicit deployment requirements,
fused-decode guidance, and host-tested evidence tools. Citation details are in
[Used in / Cited in](../../docs/used_in.md#how-to-cite).

## Public engineering examples

The following upstream changes illustrate relationships that matter when
adapting kernels. Inspect the current source and emitted artifacts before
applying a version-specific rule.

| Topic | Upstream source | What to inspect |
|---|---|---|
| FFN shape and framing | [#1854](https://github.com/Xilinx/mlir-air/pull/1854), [#1916](https://github.com/Xilinx/mlir-air/pull/1916), and the builder/header assertion in [#1921](https://github.com/Xilinx/mlir-air/pull/1921) | Producer framing, packed up/gate order, consumer width, and buffer phase must agree. |
| Large split weights | [#1836](https://github.com/Xilinx/mlir-air/pull/1836) | Segment-local offsets, packed sources, host bindings, and the selected backend's address representation. |
| Shared affine-Q4 deployments | [#2198](https://github.com/Xilinx/mlir-air/pull/2198) | Model-specific arithmetic and adapters, BFP16 prefill defaults, full-capture model gates, and complete-request profiling. |

The [reuse-versus-recompute](reuse-or-recompute.md) and
[residency-and-supply](residency-and-supply.md) guides generalize these inspection
questions. Their illustrative examples explain source work, storage and protocol
obligations; actual performance belongs to the measured implementation.
