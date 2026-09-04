# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Fused superkernel decode WITH PER-LAYER EMBEDDINGS (PLE) on NPU2 (MLIR-AIR).
#
# A fork of ../fused_decode that adds the branch Gemma4-E2B needs and the base
# engine has no room for: a per-layer 1536->256 projection, a gelu gate against
# the per-layer token embedding, and a 256->1536 projection back into the
# residual -- three extra herds fed by their own weight channel.
#
# The fork is kept byte-faithful to its parent for every OTHER model. Re-check
# that after any edit; it is the only thing keeping the two from drifting:
#   cd ../fused_decode  && FUSED_DECODE_EMIT_ONLY=1 python3 fused_decode.py     > /tmp/a.mlir
#   cd ../fused_decode_ple && FUSED_DECODE_EMIT_ONLY=1 python3 fused_decode_ple.py > /tmp/b.mlir
#   diff /tmp/a.mlir /tmp/b.mlir
#
# Quick start:
#   make compile-decode            Gemma4 kernels + one decode template (ATTN_MAXL=16)
#   make compile-decode LBUILD=2048   Production-sized template
#
# Consumed by the gemma4_e2b_q4nx LLM example (llms/gemma4_e2b_q4nx).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

DECODE_DRIVER := $(srcdir)/fused_decode_ple.py
DEC           := $(srcdir)/../fused_decode
AIEOPT_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))

# Per-kernel -O is LOAD-BEARING and is inherited verbatim from ../fused_decode:
# proj_qmm/rms_residual/glu/rope MUST be -O2 (rope.cc at -O1 miscompiles and the
# decode hangs), attn_qk/attn_kv MUST be -O1 (they hit a do-while deadlock at
# -O2). ple.cc is this fork's own kernel and follows the -O2 group.
NONATTN_KERNELS := proj_qmm rms_residual glu rope   # object .o, -O2, from ../fused_decode
ATTN_LL_KERNELS := attn_qk attn_kv                  # inline .ll only, -O1 (merge-linked)
PLE_KERNELS     := ple                              # object .o, -O2, this dir's kernels/

# Gemma4-E2B is the only model this fork exists for; the others build here only
# because the fork stays faithful to its parent. Set BEFORE PEANO_KBASE, which is
# `:=` and would otherwise bake in an empty -DMODEL_TYPE= and fail deep inside
# model_spec.h with "use of undeclared identifier 'DH'".
DECODE_MODEL   ?= gemma4-e2b
MODEL_TYPE     ?= GEMMA4_E2B
VOCAB_CHUNK_I2 ?= 27

PEANO_KBASE := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=$(MODEL_TYPE) -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(DEC)/kernels -I $(DEC)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

# ATTN_MAXL to build for. 16 is the layer gate's size (llms/gemma4_e2b_q4nx
# scores one token at position 0); production is 2048.
LBUILD ?= 16
# Decode waves in the unified sequence, and the per-wave KV source map. Both are
# what the layer gate varies: UNI_DEC=2 with KV_SRC=0,0 puts a KV-SHARED layer
# (>=15, which has no k/v projection of its own) in slab 1 reading slab 0.
UNI_DEC ?= 1
KV_SRC  ?=

DECODE_ENV := DECODE_MODEL=$(DECODE_MODEL) VOCAB_CHUNK_I2=$(VOCAB_CHUNK_I2) \
              UNIFIED=1 LM_HEAD=0 NLAYERS=$(UNI_DEC) DECODE_GOLDEN=1 \
              DECODE_GOLDEN_L=$(LBUILD) DECODE_UNI_DEC=$(UNI_DEC) \
              $(if $(KV_SRC),DECODE_KV_SRC=$(KV_SRC),)

# GLU_SLICE is derived from this model's own egress-round parity, not looked up,
# so a header copied from another model can disagree with the IR -- silently, at
# runtime. Hand the builder's value to the kernel compile; glu.cc asserts on it.
# Empty (and the assert inert) if the query fails, so this cannot break a build
# that works today.
GLU_SLICE_EXPECTED := $(shell $(DECODE_ENV) FUSED_DECODE_PRINT_CONST=GLU_SLICE $(PYTHON) $(DECODE_DRIVER) 2>/dev/null | tail -1)
PEANO_KBASE += $(if $(GLU_SLICE_EXPECTED),-DGLU_SLICE_EXPECTED=$(GLU_SLICE_EXPECTED))

.PHONY: help compile-decode kernels preflight-peano clean

help:
	@echo "============================================================"
	@echo " Fused superkernel decode + per-layer embeddings (Gemma4-E2B)"
	@echo "============================================================"
	@echo ""
	@echo "  make compile-decode   Build the Peano kernels + one decode template"
	@echo "  make kernels          Kernels only"
	@echo "  make clean            Remove kernels + template"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  LBUILD=$(LBUILD)              ATTN_MAXL to build for (2048 = production)"
	@echo "  UNI_DEC=$(UNI_DEC)             decode waves in the unified sequence"
	@echo "  KV_SRC=0,0            per-wave KV source slab (KV-shared layers)"
	@echo ""
	@echo "Weight-free build. Consumed by llms/gemma4_e2b_q4nx."

## Peano needs to be at least 22.x for THIS model. The repo pin (21.0.0) is a
## silent miscompile here, not a build failure: every layer comes back all-NaN,
## which reads exactly like a numerics bug in the design. Checked loudly rather
## than left to the layer gate to discover.
##
## Deliberately NOT ../fused_decode's preflight-llvm-link. That one aborts unless
## PATH carries an llvm-link older than LLVM 23, and this build does not use one:
## instrumenting PATH with a logging llvm-link shim across a full build recorded
## ZERO calls, and the build succeeded with an LLVM 23 llvm-link first on PATH.
## The inline-attn merge here goes through mlir-aie's own path.
preflight-peano:
	@if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
	  echo "ERROR: set PEANO_INSTALL_DIR to an llvm-aie install (>= 22.0.0)."; exit 1; fi
	@v=`$(PEANO_INSTALL_DIR)/bin/llc --version 2>/dev/null | sed -n 's/.*LLVM version \([0-9]*\).*/\1/p' | head -1`; \
	  if [ -n "$$v" ] && [ "$$v" -lt 22 ]; then \
	    echo "ERROR: Peano is LLVM $$v; this model needs >= 22."; \
	    echo "       21.x miscompiles it to all-NaN -- a silent wrong answer, not a build error."; \
	    exit 1; \
	  fi

## Peano kernels. The non-attn and attn sets come from ../fused_decode (shared
## with every other model); ple.cc is this fork's own.
kernels: preflight-peano
	@echo ">>> non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(DEC)/kernels/$$k.cc -o $(srcdir)/$$k.o || exit 1; done
	@echo ">>> attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(srcdir)/$$k.ll || exit 1; done
	@echo ">>> PLE kernels (Peano -O2): $(PLE_KERNELS)"
	@for k in $(PLE_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(srcdir)/kernels/$$k.cc -o $(srcdir)/$$k.o || exit 1; done
	@echo "Kernels built (GLU_SLICE_EXPECTED=$(GLU_SLICE_EXPECTED))."

## Kernels + one decode template into this dir (decode.xclbin + decode.insts.bin).
## Weight-free: weights are runtime BOs, so no bundle is read at build time.
##
## Not idempotent, unlike ../fused_decode's compile-decode. The template here is
## keyed on THREE things the consumer varies per run (LBUILD, UNI_DEC, KV_SRC)
## rather than one, and a stale-but-present decode.xclbin is the single most
## expensive mistake available in this example: the layer gate does not build,
## it dispatches whatever template is on disk, so a skipped rebuild scores the
## wrong design and reads as a numerics regression.
compile-decode: kernels
	@echo ">>> decode template (ATTN_MAXL=$(LBUILD), UNI_DEC=$(UNI_DEC)$(if $(KV_SRC), KV_SRC=$(KV_SRC),))"
	@cd $(srcdir) && $(DECODE_ENV) $(PYTHON) $(DECODE_DRIVER) >/tmp/ple_decode_L$(LBUILD).log 2>&1; \
	  if [ -f $(srcdir)/decode.xclbin ] && [ -f $(srcdir)/decode.insts.bin ]; then \
	    echo "PLE decode build complete: decode.xclbin + decode.insts.bin"; \
	  else \
	    echo "decode build FAILED (see /tmp/ple_decode_L$(LBUILD).log)"; \
	    tail -30 /tmp/ple_decode_L$(LBUILD).log; exit 1; \
	  fi

clean:
	rm -f $(srcdir)/*.o $(srcdir)/*.ll $(srcdir)/decode.xclbin $(srcdir)/decode.insts.bin 2>/dev/null || true
	rm -rf $(srcdir)/air_project $(srcdir)/__pycache__ 2>/dev/null || true
	rm -f $(srcdir)/air.mlir 2>/dev/null || true
	@echo "PLE decode kernels, template and build dir removed."
