# MLIR-AIR agentic harness v0.1

The v0.1 agentic harness supports the AMD XDNA2 NPU compilation flow built on
the MLIR-AIR compilation stack. It connects Codex or Claude Code to AIR
implementations, the measured kernel registry and 17 NPU skills. The agent iterates through kernel development
and model composition, using source inspection, compiler feedback, fixed
correctness checks and production profiling.

Start the coding agent in the AIR repository and use
`$deploy-new-llm OWNER/MODEL` in Codex or `/deploy-new-llm OWNER/MODEL` in Claude
Code. Add precision, workload or acceptance requirements to the same request.
The agent resolves a pinned task, implements the required model adaptation, then
verifies and profiles it. Existing supported models can use reproduction mode.

The [package quick start](https://github.com/Xilinx/mlir-air/blob/main/agentic/README.md) includes
task preparation, a runnable example, prerequisites and host tests. The
[workflow](https://github.com/Xilinx/mlir-air/blob/main/agentic/WORKFLOW.md) explains the two loops.
[Client installation](https://github.com/Xilinx/mlir-air/blob/main/agentic/INSTALLATION.md) uses
portable repository registrations pointing to one canonical skill payload.
Both clients use the shared [project instructions](https://github.com/Xilinx/mlir-air/blob/main/AGENTS.md); the installation
guide explains the Claude Code version and loading settings for this file.

Kernel reuse follows the declared arithmetic, layout and validation contract.
The [registry evidence interface](https://github.com/Xilinx/mlir-air/blob/main/agentic/references/maintenance.md)
adds opt-in verified selection while preserving legacy measured configurations.
Model checks use complete finite captures before applying mutual top-5 token
inclusion; measured performance retains the request boundaries and raw samples.

The harness extends [From Human Guidance to Autonomy: Agent Skill System for
End-to-End LLM Deployment on Spatial NPUs](https://arxiv.org/abs/2606.07586),
presented at **MLArchSys'26, co-located with ISCA 2026**. See
[Used in / Cited in](used_in.md#how-to-cite) for the citation.
