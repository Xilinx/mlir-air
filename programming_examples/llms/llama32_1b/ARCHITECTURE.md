# LLAMA-3.2-1B BF16 Inference — Architecture

Companion to [README.md](README.md) (overview, quick start, performance,
file map). This doc focuses on how the per-layer kernel chain and the
runtime are organized.

## Model Config

16 layers, emb_dim=2048, n_heads=32, head_dim=64, n_kv_heads=8,
hidden_dim=8192, vocab_size=128256, BF16, rope_base=500000.

## Per-Layer Kernel Sequence

```
Prefill (per layer, 3 XRT calls):
  rms_gemms_rope.elf (6 launches) → flash_attn.elf (1 launch) → o_ffn.elf (8 launches)

Decode (per token per layer, 3 XRT calls):
  rms_gemv_rope.elf (6 launches) → CPU attention → o_gemv_ffn.elf (8 launches)
```

8 unique kernel configs compiled once via `KernelCache` and cached to disk:
- Prefill: `rms_gemms_rope.elf`, `flash_attn.elf`, `o_ffn.elf`
- Decode: `rms_gemv_rope.elf`, `o_gemv_ffn.elf`, `lm_head_gemv.elf` (also
  reused by prefill for the last-token LM Head projection)

## Runtime Flow

```
prepare_runtime()          ← one-time: load weights, pre-load into per-layer BOs
  ↓
run_npu_prefill()          ← 16 layers × 3 kernel calls + Final RMSNorm + LM Head
  ↓
generate() decode loop     ← per token: 16 layers × 3 kernel calls + LM Head GEMV
```

## Key Design Patterns

- **Multi-launch ELF**: multiple `air.launch` ops in one MLIR module → single
  `xrt.run()`. Intermediates flow through DDR without CPU round-trip.
- **Text-based MLIR stitching**: extract func body, rename SSA values with a
  prefix, remap func args, assemble combined module. See
  `llama_kernel_builder/stitching.py`.
- **Per-layer BOs** (`bo_key=f"kernel_L{layer_idx}"`): each layer gets its own
  Buffer Objects. Weights are written once during pre-load and reused on
  every inference call.
- **`static_input_indices`**: skip BO write for weights on non-first calls.
- **`intermediate_indices`**: skip BO write for buffers the kernel overwrites.
- **External kernel rename**: the K=8192 Down GEMV uses `mv_k8192.o`
  (compiled with `-D` renamed symbols) so it can coexist with K=2048 GEMVs
  in one ELF.
- **Seq-first layout**: RoPE and FlashAttention accept `(seq, heads*dim)`
  natively, eliminating host-side transposes. The seq-first FA variant lives
  at `programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py`.
- **Half-split RoPE kernel**: custom `rope_halfsplit.cc` matches
  HuggingFace Llama's rotation convention `(d[i], d[i+head_dim/2])`. LUT
  layout is `[cos..., sin...]` (concatenated, not interleaved). Replaces
  the upstream interleaved `rope.cc`. See `docs/explain.md`.

For deeper material — kernel timing breakdown, the compilation pipeline,
known limitations — see the [`docs/`](docs/) directory.
