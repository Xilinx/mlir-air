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
# DECODE_GOLDEN=1 enables the post-attention-RMS decode path (a boolean flag; no
# weight files are read at build time -- the compile is weight-free).
DECODE_ENV     := VOCAB_CHUNK_I2=14 LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1

.PHONY: help compile-decode _compile_decode_build preflight-llvm-link clean

help:
	@echo "============================================================"
	@echo " Fused superkernel decode (Llama-3.2-1B) on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "  make compile-decode   Build decode kernels + templates (decode_L2048 + L2047, ~15 min)"
	@echo "  make clean            Remove kernels + templates"
	@echo ""
	@echo "Consumed by the llama32_1b_q4nx LLM e2e (make chat / make gen there)."
	@echo "Weight-free build; needs an llvm-link on PATH from LLVM <23"
	@echo "-- see README.md Reproducibility."

## Build the fused decode: Peano kernels + two templates (decode_L2048 + decode_L2047
## slope ref). ~15 min. Weight-free (runtime BOs), like the prefill compile.
##
## Idempotent: if both templates are already present, skip (the ~15 min build only
## runs once). `make clean` forces a rebuild. Consumers that call this repeatedly
## (e.g. the llama32_1b_q4nx verify + profile lits) thus pay the cost at most once.
##
## REPRODUCIBILITY (toolchain): the inline-attn merge path shells out to an EXTERNAL
## `llvm-link` (resolved from PATH), which must be PRE-LIFETIME-CHANGE, i.e. LLVM < 23.
## A >=23 llvm-link upgrades the llvm.lifetime intrinsic to the no-size form and Peano
## opt then rejects the merged IR ("Broken module found"). The preflight aborts on it.
## This is the same rule the lit harness gates on (lit.cfg.py, feature llvm_link_pre23);
## the two MUST agree -- see the preflight comment below.
compile-decode:
	@if [ -f $(srcdir)/decode_L2048.xclbin ] && [ -f $(srcdir)/decode_L2048.insts.bin ] \
	    && [ -f $(srcdir)/decode_L2047.xclbin ] && [ -f $(srcdir)/decode_L2047.insts.bin ]; then \
	  echo "Decode templates already built (decode_L2048 + decode_L2047); skipping. Run 'make clean' to force a rebuild."; \
	else \
	  $(MAKE) -f $(srcdir)/Makefile _compile_decode_build; \
	fi

## Shared preflight for every inline-attn decode build. Any model driving this
## engine (llms/*_q4nx) must run it rather than copy it, so the rule lives in
## exactly one place.
##
## The rule is LLVM < 23, and it must stay identical to the `llvm_link_pre23`
## feature in programming_examples/lit.cfg.py. Every decode e2e lit test
## (llama32_1b_q4nx, llama32_3b_q4nx, gemma3_4b_q4nx) is REQUIRES-gated on that
## feature, and lit will shim ANY <23 llvm-link it finds on PATH to the front --
## with no regard for Peano's major. If this preflight is stricter than lit, lit
## marks such a test supported, runs it, and the test hard-fails right here,
## which is the exact outcome the feature exists to avoid.
preflight-llvm-link:
	@echo ">>> preflight: llvm-link must be LLVM <23 (pre-lifetime-change) for the inline-attn merge"
	@# A >=23 llvm-link rewrites llvm.lifetime.* to the no-size form, which Peano
	@# opt then rejects outright ("Broken module found") -- loud, at build time.
	@# Peano's own major is reported below for context only; it is NOT the gate.
	@LLVM_LINK=$$(command -v llvm-link || true); \
	  PEANO_CLANG="$(PEANO_INSTALL_DIR)/bin/clang"; \
	  if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$$PEANO_CLANG" ]; then \
	    echo "ERROR: PEANO_INSTALL_DIR does not point at a Peano install (no $$PEANO_CLANG)"; exit 1; fi; \
	  PEANO_VER=$$($$PEANO_CLANG --version | grep -oE 'clang version [0-9]+' | grep -oE '[0-9]+' | head -1); \
	  if [ -z "$$LLVM_LINK" ]; then echo "ERROR: no llvm-link on PATH (need LLVM <23)"; exit 1; fi; \
	  VER=$$($$LLVM_LINK --version | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | head -1); \
	  if [ -z "$$VER" ]; then \
	    echo "ERROR: could not parse an LLVM major from '$$LLVM_LINK --version' (need LLVM <23)"; exit 1; fi; \
	  echo "    llvm-link = $$LLVM_LINK (LLVM $$VER); Peano = LLVM $$PEANO_VER"; \
	  if [ "$$VER" -ge 23 ]; then \
	    echo "ERROR: llvm-link is LLVM $$VER; the inline-attn merge needs LLVM <23"; \
	    echo "       (see README Reproducibility for how to fetch one)"; exit 1; fi

_compile_decode_build: preflight-llvm-link
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
