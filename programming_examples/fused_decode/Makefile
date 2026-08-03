# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Fused superkernel decode for Llama-3.2-1B on NPU2 (MLIR-AIR).
#
# ONE dispatch = 16 decoder layers + LM head, reading and appending a shared per-layer
# KV cache. A single xclbin built at ATTN_MAXL=2048 serves every context length L in
# [1, 2048]: the attention loop skips fully-masked blocks and the per-token instruction
# stream is patched on the host (decode_insts_gen.py) -- recompile-free per-token decode.
#
# Quick start:
#   make compile-decode   Build the decode kernels + two templates (decode_L2048 + L2047)
#
# This example provides the decode path consumed by the llama32_1b_q4nx LLM end-to-end
# (programming_examples/llms/llama32_1b_q4nx). It needs Q4NX weights for the GEMM shapes.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PARIS_WEIGHTS  ?= $(HOME)/q4nx_data/weights
DECODE_DRIVER  := $(srcdir)/fused_decode.py
AIEOPT_DIR     := $(shell realpath $(dir $(shell which aie-opt))/..)

# Per-kernel -O is LOAD-BEARING (do not use one -O for all):
#   proj_qmm/rms_residual/glu/rope MUST be -O2; attn_qk/attn_kv MUST be -O1.
#   rope.cc at -O1 MISCOMPILES (Peano loop-metadata) -> the decode HANGS on device
#   (6143 ms ERT_CMD_STATE_TIMEOUT). attn_qk/attn_kv at -O2 hit a separate do-while
#   deadlock, so they must stay -O1.
NONATTN_KERNELS := proj_qmm rms_residual glu rope   # object .o, -O2
ATTN_LL_KERNELS := attn_qk attn_kv                  # inline .ll only, -O1 (merge-linked)
PEANO_KBASE    := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=LLAMA_3_2_1B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(srcdir)/kernels -I $(srcdir)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
DECODE_ENV     := UNI_DEC=16 UNI_LM=9 VOCAB_CHUNK_I2=14 LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=$(PARIS_WEIGHTS)

.PHONY: help compile-decode clean

help:
	@echo "============================================================"
	@echo " Fused superkernel decode (Llama-3.2-1B) on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "  make compile-decode   Build decode kernels + templates (decode_L2048 + L2047, ~15 min)"
	@echo "  make clean            Remove kernels + templates"
	@echo ""
	@echo "Consumed by the llama32_1b_q4nx LLM e2e (make chat / make gen there)."
	@echo "Needs PARIS_WEIGHTS (Q4NX weights) for the GEMM shapes and a pre-change"
	@echo "llvm-link (< LLVM 23) on PATH -- see README.md Reproducibility."

## Build the fused decode: Peano kernels + two templates (decode_L2048 + decode_L2047
## slope ref). ~15 min. Needs PARIS_WEIGHTS for the GEMM shapes.
##
## REPRODUCIBILITY (toolchain): the inline-attn merge path shells out to an EXTERNAL
## `llvm-link` (resolved from PATH). It MUST be a pre-lifetime-change LLVM (e.g. the system
## /usr/bin/llvm-link, LLVM 20). A newer LLVM (>=23) llvm-link upgrades the llvm.lifetime
## intrinsic to the no-size form and Peano opt (LLVM 21) then rejects the merged IR
## ("Broken module found"). Keep any such LLVM's bin OFF PATH. The preflight aborts early.
compile-decode:
	@echo ">>> preflight: llvm-link must be a pre-change LLVM (<23) for the inline-attn merge"
	@LLVM_LINK=$$(command -v llvm-link || true); \
	  if [ -z "$$LLVM_LINK" ]; then echo "ERROR: no llvm-link on PATH (need system LLVM ~20)"; exit 1; fi; \
	  VER=$$($$LLVM_LINK --version | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | head -1); \
	  echo "    llvm-link = $$LLVM_LINK (LLVM $$VER)"; \
	  if [ "$$VER" -ge 23 ]; then echo "ERROR: llvm-link is LLVM $$VER (>=23); keep it OFF PATH (see README Reproducibility)"; exit 1; fi
	@echo ">>> non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(srcdir)/kernels/$$k.cc -o $(srcdir)/$$k.o || exit 1; done
	@echo ">>> attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@# attn_qk/attn_kv are built ONLY as inline .ll (merge-linked into the core; mlir-aie #3399).
	@# They MUST be -O1 (a -O2 attn do-while deadlock); non-attn MUST be -O2 (rope -O1 miscompiles).
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(srcdir)/kernels/$$k.cc -o $(srcdir)/$$k.ll || exit 1; done
	@echo ">>> template decode_L2048 (+ L2047 slope ref) [region-major KV + fire-and-free readback + inline attn]"
	@# The two same-ATTN_MAXL builds (L2048 + L2047) calibrate the L-slope so DecodeInstsGen can
	@# patch any context length in [1, 2048] into one template at runtime (recompile-free decode).
	@# The block loop is always single-buffered (air.disable_ping_pong, unconditional in
	@# fused_decode.py); without it the KV ring misaligns -> garbage continuation. Turbo for ~50 tok/s.
	@cd $(srcdir) && $(DECODE_ENV) DECODE_GOLDEN_L=2048 python3 $(DECODE_DRIVER) >/tmp/decode_L2048.log 2>&1; \
	  if [ -f decode.xclbin ]; then mv -f decode.xclbin decode_L2048.xclbin; mv -f decode.insts.bin decode_L2048.insts.bin; \
	  else echo "decode_L2048 FAILED (see /tmp/decode_L2048.log)"; exit 1; fi
	@cd $(srcdir) && $(DECODE_ENV) DECODE_GOLDEN_L=2047 python3 $(DECODE_DRIVER) >/tmp/decode_L2047.log 2>&1; \
	  if [ -f decode.insts.bin ]; then mv -f decode.insts.bin decode_L2047.insts.bin; mv -f decode.xclbin decode_L2047.xclbin; \
	  else echo "decode_L2047 FAILED (see /tmp/decode_L2047.log)"; exit 1; fi
	@echo "Decode build complete: decode_L2048 + decode_L2047 templates."

## Remove compiled kernels + decode templates.
clean:
	rm -f $(srcdir)/*.o $(srcdir)/*.ll $(srcdir)/decode.xclbin $(srcdir)/decode.insts.bin \
	      $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin 2>/dev/null || true
	@echo "Kernels and decode templates removed. Rebuild with 'make compile-decode'."
