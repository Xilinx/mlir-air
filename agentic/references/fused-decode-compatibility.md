# Fused-decode compatibility and state contracts

Use this reference when adapting the AIR Q4NX fused-decode path or diagnosing a changed build, packed-weight cache, instruction template, or KV organization. It supplements the generic [change analysis](../skills/megakernel-change-analysis/SKILL.md); it does not prescribe a new model's mapping or an optimization.

Historical source scope: AIR `6746658f74200feb957f99ad4163297f3185739c`. Paths below are relative to the current AIR repository root; line-number anchors identify that revision. The current integration uses shared deployment interfaces documented in [project integration](project-integration.md). Reinspect the selected builder/Makefile before applying a version-specific compiler workaround.

## Build and compiler compatibility

The selected `fused_decode/Makefile` compiles `proj_qmm`, `rms_residual`, `glu`, and `rope` to objects at **-O2**, and `attn_qk`/`attn_kv` to inline LLVM IR at **-O1** with `-DDECODE_INLINE_ATTN`. Preserve that per-kernel distinction when reusing the seed. Source comments associate other settings with prior compiler failures; these are version-specific implementation constraints, not hardware laws.

The inline-attention merge resolves an external **llvm-link < 23** from PATH. The Makefile also checks the pinned `llvm-aie` package version; satisfying the linker major alone does not establish compiler compatibility. A linker-only directory avoids shadowing Peano's clang/opt/llc with a system LLVM installation. Inspect the selected build's actual resolved tools and flags. Do not install or replace tools merely because this reference names them.

`ATTN_HOT` currently inlines by default. `ATTN_PEANO_NOINLINE` is a C/C++ preprocessor define whose presence controls the alternate branch; setting a similarly named Make variable does not automatically pass `-D` to the compiler. An old workaround is not a universal recommendation.

## Logical quantization, bundle, and device packing

Q4NX uses unsigned nibbles with `w = scale*q + min`; the `Q4_0` branch uses signed nibbles with `w = scale*q`. The device codec comes from model/header/kernel configuration, not the filename extension or a family nickname. The Peano Q4_0 path explicitly sign-corrects unsigned loads; a diagnostic mode without sign correction is not numerically valid.

A logical group32 affine Q4 input is not an NPU-ready packed stream. Existing Llama Q4NX prefill and fused decode use different consumer layouts. Trace quant values, scale/min rounding, row/column order, projection concatenation, up/gate interleave, cascade partitioning, and padding into the actual kernel indices. Reuse a packer only where its numerical and layout contract matches the supplied input; do not silently re-quantize a fixed task asset.

`q4nx_requant.build_requant_cache` derives `dual_chan` from the selected `fd.W_DUAL_CHAN`. The shared Makefile's `.decode_flags` stamp covers this switch because it changes the DDR stream layout. It is not a complete source/compiler/model/quantization cache key. Bind packed weights, compiled geometry and selected flags together; a matching shape or filename is insufficient.

## Active lengths, templates, and cached state

Keep the actual prompt length P, current context L, prefill build length, and allocated decode `ATTN_MAXL` distinct. `DecodeInstsGen` groups builds by `16*ceil(L/16)`, requires a same-window pair with a usable per-word slope, and selects the smallest calibrated window covering the requested reach. `.decode_windows`, when present, rejects a mismatch between declared and calibrated window sets. File presence, compatible sizes and a slope do not prove source or artifact identity.

In the affine-template path, changing runtime L patches selected words while the compiled KV-readback extent may still cover `ATTN_MAXL`. This can move padded KV data even when the attention kernel consumes only active blocks. Optional staircase dispatch changes windows; its region-major cache requires re-spacing live prefixes when the window stride changes. Source also includes a distinct dynamic-runtime-sequence path (`decode_dynseq.py` / `DECODE_DYNSEQ`), whose non-affine `ceil(L/16)` readback count is generated exactly. Do not generalize affine-template limitations to that separate path or enable either mode without binding it to the task's candidate and budget.

## One KV ownership and ordering example

For the selected attention path with L > 1, the RoPE producer supplies this token's roped K/raw V. Append S2MM writes slot L-1 in the shared L3 KV memref; the later readback consumes that same storage. `air-annotate-append-barrier` derives the append-to-readback read-after-write ordering from the shared memref. L=1 takes a different on-chip path.

The region-major representation holds each group's K and V in separate contiguous spans with stride `ATTN_MAXL*REGION_W`; position p starts at `p*REGION_W` within its region. K and V readback channels stream concurrently into MemTile rings. The builder describes this as fire-and-free MM2S, paced by ring locks rather than an await on each transfer. Adding a serial wait, changing an alias, or extending a ring buffer's live interval changes the protocol. Follow producer → append completion → readback → consumer release before changing ownership or reuse. Check the generated runtime sequence and relevant transition/reset cases when source alone cannot settle the relationship; this example is not a proof for every model or lowering.

## Historical source anchors

All anchors use the `programming_examples/` prefix under AIR. Line numbers identify this frozen revision; symbols remain useful when code moves.

| Topic | Source anchors |
|---|---|
| Per-kernel flags and linker preflight | `fused_decode/Makefile:23-29,102-131,150-190,232-240` |
| Inlining switch | `fused_decode/kernels/attn_qk.cc:42-54`; `kernels/attn_kv.cc:8-21` |
| Codec branches and bundle caveat | `fused_decode/kernels/q4_k.h`, `q4_0_codec.py`; `fused_decode/README.md:35-84` |
| Decode repacking and dual channels | `llms/llama32_1b_q4nx/q4nx_requant.py:25-34,53-75,105-115` |
| Calibration and window declaration | `fused_decode/decode_insts_gen.py:26-129,148-164` |
| Runtime window/state ownership | `fused_decode/decode_staircase.py:20-40,43-77,81-104`; `decode_dynseq.py` |
| Append/readback and region layout | `fused_decode/fused_decode.py:998-1008,1054-1063,2587-2604` |
| Existing seed placement | `llms/llama32_1b_q4nx/README.md:188-205` |

The existing Llama Q4NX seed uses NPU block compute/attention in prefill and fused decode, with host work such as embedding and token processing; prefill final RMSNorm is on the host. This differs from the BF16 examples' decode placement. Derive a new task's required placement from its supplied contract and actual production calls, not a broad README sentence. Apply the package's [two verification levels](validation-contracts.md) with the task's fixed coverage. No third mandatory numerical gate is introduced here.
