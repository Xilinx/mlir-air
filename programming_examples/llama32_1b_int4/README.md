# LLAMA-3.2-1B int4-AWQ Prefill on AMD NPU2

End-to-end prefill of an AWQ-uint4 quantized LLAMA-3.2-1B (e.g.
`amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead`) on AMD
NPU2 (AIE2P) hardware via MLIR-AIR, with a top-K correctness gate
against a CPU bf16 reference built from the same dequantized weights.

The example ships two prefill backends behind a `--prefill-dtype` flag.
The bf16 path is the recommended one for prefill today; the int4 path
is preserved so the int4 decode driver (a separate follow-up PR) can
share this directory and the AWQ packer.

## Performance

NPU2 (AMD Strix), seq=2048, 16 layers, NPU flash attention, prompt
"The capital of France is":

| Backend (`--prefill-dtype`) | per-layer | end-to-end | top-10 vs HF | argmax |
|---|---|---|---|---|
| `bf16` (default) | 84 ms | **1.38 s** | 8/10 | "Paris" ✓ |
| `int4`           | 698 ms | 11.2 s | 8/10 | "Paris" ✓ |

Both backends consume the same AWQ checkpoint and produce identical
AWQ-quality output. The 8× gap is structural to the current int4 GEMM
kernel:

- Down GEMM hits the memtile L2 budget at K=8192, capping `herd_m=2`
  (8 PEs vs 32) — `matmul_int4_packed.py` can't tile `K_L2 < K`.
- Peano AIE2P `VLD_x_pstm_nrm_imm` 9-bit immediate range forces
  `tile_n=16` (16× more launch iterations than bf16's `tile_n=128`).

Decode is DMA-bandwidth-bound and benefits from int4's halved weight
footprint; the int4 decode driver lives in a follow-up PR.

## Prerequisites

1. **MLIR-AIR base environment** — AMD NPU2 hardware, Peano compiler,
   the project's standard env: `source utils/env_setup.sh ...`

2. **Extra Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
   Installs `safetensors`, `huggingface_hub`, `transformers`, `torch`.

3. **AWQ checkpoint** — the default
   (`amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead`) is
   **not gated** and downloads without a token. The HF reference path
   does pull the upstream tokenizer behind the AWQ checkpoint, which
   may be gated — in that case `huggingface-cli login` first.

## Quick Start

```bash
# Compile both prefill backends (int4 + bf16). ~1-2 min the first time
# (BF16 stitchers compile fast; int4 stitchers take longer on Down GEMM).
make compile

# Run NPU prefill end-to-end with the bf16 backend (default, ~1.4 s).
# Prints HF top-K, NPU top-K, overlap and argmax match.
make run

# Same but with the int4 backend (~11 s; same AWQ-quality output).
make run PREFILL_DTYPE=int4

# Run with per-kernel + per-layer profiling breakdown.
make profile

# Top-K correctness gate (used by run_npu2_verify.lit). PASS iff overlap
# >= MIN_OVERLAP (default 6) AND argmax matches HF.
make verify
make verify-int4    # same gate against the int4 backend
```

## Key Files

| Path | Purpose |
|---|---|
| `llama32_1b_int4_prefill.py` | Driver: loads AWQ, runs either backend, prints top-K vs HF |
| `awq_pack.py` | AWQ qweight/qzeros/scales → int4 GEMM packed BO + bf16 dense |
| `multi_launch_builder/rms_gemms_rope_int4_multi.py` | int4 RMSNorm + Q/K/V + RoPE Q/K (6-launch ELF) |
| `multi_launch_builder/o_ffn_int4_multi.py` | int4 O + ResAdd + FFN-RMS + Gate/Up + SwiGLU + Down + FFN-Add (8-launch ELF) |
| `Makefile` | Canonical `compile / run / verify / profile / clean` targets |
| `run_npu2_compile.lit` | Compile-only smoke test (no HF_TOKEN needed) |
| `run_npu2_verify.lit` | End-to-end top-K gate (HF_TOKEN required) |

Shared scaffolding (the bf16 prefill stitchers, the `kernel_builder/`
package, the `LlamaWeights`/`LlamaConfig` dataclasses, the bf16 CPU
helpers) is imported from `../llama32_1b/` via `sys.path`. The int4
GEMM module is imported from
`../matrix_multiplication/int4_awq/matmul_int4_packed.py`.

The verify subsystem under `../llama32_1b/verify/` (top-K gate,
per-layer diagnosis, prompt fixtures, HF + NPU runners) will move to a
shared location in a follow-up so this example can plug into it
instead of carrying its own monolithic verifier.
