# LLAMA-3.2-1B int4-AWQ (NPU2)

Intended home for the int4-AWQ end-to-end Llama-3.2-1B example on NPU2:
loads a pre-AWQ-quantized HF checkpoint (e.g.
`amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead`), runs
prefill and decode on NPU with the int4 micro-kernels from PR #1639.

## Contents

| File                              | Purpose                                       |
| --------------------------------- | --------------------------------------------- |
| `awq_pack.py`                     | AWQ weight quantizer + packer for the int4 GEMM BO layout that `matmul_int4_packed.build_module` consumes. |
| `multi_launch_builder/rms_gemms_rope_int4_multi.py` | Prefill stitcher: RMSNorm + int4 Q/K/V GEMMs + RoPE Q/K (6-launch ELF). |
| `multi_launch_builder/o_ffn_int4_multi.py`          | Prefill stitcher: int4 O + ResAdd + FFN-RMS + int4 Gate/Up + SwiGLU + int4 Down + FFN-Add (8-launch ELF). |

## Status

* Prefill stitchers are present and compile-only verified at small shapes.
* Decode stitchers (`rms_qkv_int4_rope_multi.py`,
  `o_gemv_ffn_int4_multi.py`) currently live in
  `../llama32_1b/multi_launch_builder/`; they will migrate here in a
  follow-up PR.
* No end-to-end inference driver yet. The expected entry point
  (`llama32_1b_int4_inference.py`) is the natural home for the HF
  AutoAWQ checkpoint loader and the int4 `--quant=awq` flag that
  PR #1638 prototypes.

## Shared scaffolding

Imports stitching helpers, the SwiGLU/RMSNorm builders, and the RoPE
LUT generator from `../llama32_1b/` (cross-example). These should be
elevated to a shared location once the int4 example matures.
