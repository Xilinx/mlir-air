# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Gemma3-4B Q4NX end-to-end decode on NPU2 (MLIR-AIR), FLM-faithful.
#
# ONE dispatch = 34 decoder layers + tied LM head, reading and appending a shared per-layer
# KV cache. Drives the shared fused_decode engine (DECODE_MODEL=gemma3-4b): 4-norm sandwich,
# dual-theta RoPE + qk-norm, GELU-tanh GLU, DH=256 GQA2 attention.
#
# The KV seed + greedy first-token golden come from a numpy reference forward
# (gemma3_4b_q4nx_weights.forward_prompt); the fused per-token decode then generates.
#
# Quick start:
#   make compile-decode LBUILD=16   Build the gemma decode templates (small, for the Paris gate)
#   make gen                        Run the Paris gate (first token 9079 " Paris")
#   make compile-decode             Build production templates (ATTN_MAXL=2048, ~15 min)

srcdir     := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DEC        := $(srcdir)/../../fused_decode
DECODE_DRIVER := $(DEC)/fused_decode.py
AIEOPT_DIR := $(shell realpath $(dir $(shell which aie-opt))/..)

# Per-kernel -O is LOAD-BEARING (see fused_decode/Makefile): proj_qmm/rms_residual/glu/rope
# -O2; attn_qk/attn_kv inline .ll -O1. Gemma differs only by -DMODEL_TYPE=GEMMA3_4B (the
# header auto-sets A_FUNC=A_GELU, HAS_QK_NORM, ATTN_IMPL, dims/vocab).
NONATTN_KERNELS := proj_qmm rms_residual glu rope
ATTN_LL_KERNELS := attn_qk attn_kv
PEANO_KBASE := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=GEMMA3_4B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(DEC)/kernels -I $(DEC)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

# Gemma decode env (module-level constants read these): 34 layers, vocab-262208 chunking.
DECODE_ENV := DECODE_MODEL=gemma3-4b VOCAB_CHUNK_I2=5 UNIFIED=1 LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1

# ATTN_MAXL to build for. 16 = the Paris gate (P + a few tokens <= 16); 2048 = production.
LBUILD ?= 2048
LREF   := $(shell echo $$(($(LBUILD) - 1)))

.PHONY: help compile-decode gen clean

help:
	@echo "  make compile-decode LBUILD=16   Gemma decode templates (small, Paris gate)"
	@echo "  make gen                        Paris gate (first token 9079)"
	@echo "  make compile-decode             Production templates (ATTN_MAXL=2048, ~15 min)"
	@echo "  make clean                      Remove gemma decode templates"

## Build the gemma decode: kernels (GEMMA3_4B) + two same-ATTN_MAXL templates
## (decode_L$(LBUILD) + decode_L$(LREF)) into this dir. Weight-free build.
compile-decode:
	@echo ">>> preflight: llvm-link must be a pre-change LLVM (<23) for the inline-attn merge"
	@LLVM_LINK=$$(command -v llvm-link || true); \
	  if [ -z "$$LLVM_LINK" ]; then echo "ERROR: no llvm-link on PATH (need system LLVM ~20)"; exit 1; fi; \
	  VER=$$($$LLVM_LINK --version | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | head -1); \
	  echo "    llvm-link = $$LLVM_LINK (LLVM $$VER)"; \
	  if [ "$$VER" -ge 23 ]; then echo "ERROR: llvm-link is LLVM $$VER (>=23); keep it OFF PATH"; exit 1; fi
	@echo ">>> gemma non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.o || exit 1; done
	@echo ">>> gemma attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.ll || exit 1; done
	@echo ">>> gemma template decode_L$(LBUILD) (+ L$(LREF) slope ref)"
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LBUILD) python3 $(DECODE_DRIVER) >/tmp/gemma_decode_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ]; then mv -f decode.xclbin $(srcdir)/decode_L$(LBUILD).xclbin; mv -f decode.insts.bin $(srcdir)/decode_L$(LBUILD).insts.bin; \
	  else echo "decode_L$(LBUILD) FAILED (see /tmp/gemma_decode_L$(LBUILD).log)"; tail -30 /tmp/gemma_decode_L$(LBUILD).log; exit 1; fi
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LREF) python3 $(DECODE_DRIVER) >/tmp/gemma_decode_L$(LREF).log 2>&1; \
	  if [ -f decode.insts.bin ]; then mv -f decode.insts.bin $(srcdir)/decode_L$(LREF).insts.bin; mv -f decode.xclbin $(srcdir)/decode_L$(LREF).xclbin; \
	  else echo "decode_L$(LREF) FAILED (see /tmp/gemma_decode_L$(LREF).log)"; tail -30 /tmp/gemma_decode_L$(LREF).log; exit 1; fi
	@echo "Gemma decode build complete: decode_L$(LBUILD) + decode_L$(LREF) templates."

## Run the Paris gate (greedy; first token must be 9079 " Paris").
gen:
	@cd $(srcdir) && python3 gemma3_4b_q4nx_inference.py --n-tokens 9

clean:
	rm -f $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin 2>/dev/null || true
	@echo "Gemma decode templates removed."
