# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Fused superkernel decode WITH A GATED DELTANET MIXER on NPU2 (MLIR-AIR):
# Qwen3.5-0.8B, one dispatch = 24 layers (18 Gated DeltaNet, 6 gated full
# attention) + the LM head.
#
# A fork of ../fused_decode (as ../fused_decode_ple is for Gemma4): the Gated
# DeltaNet's recurrent state (1 MiB per layer, read and written every token)
# needs its own cores and DMA channels. The fork keeps the parent's projection
# engine, rms, GLU, LM-head waves and full-ELF machinery, uses the LFM2 hybrid
# arm for the mixer tiles, and adds the GDN herd (column 3). See README.md.
#
#   make compile-decode-elf BUILD=<dir> [LBUILD=<capacity>]
#       kernels + one full ELF for every context length up to LBUILD
#
# Consumed by llms/qwen35_0_8b_q4nx (q35_fdg.py), whose Makefile builds one ELF
# per decode capacity.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
BUILD ?= $(srcdir)/build
LBUILD ?= 2048

DECODE_DRIVER := $(srcdir)/fused_decode_gdn.py
DEC := $(srcdir)/../fused_decode
AIEOPT_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))

# Per-kernel -O as in ../fused_decode (load-bearing there: rope at -O1
# miscompiles, attention at -O2 deadlocks). The fork's own kernels (gdn_mix,
# gdn_step) follow the -O2 group.
NONATTN_KERNELS := proj_qmm rms_residual glu rope gdn_mix gdn_step
ATTN_LL_KERNELS := attn_qk attn_kv
ATTN_OPT ?= O1

DECODE_ENV := DECODE_MODEL=qwen3.5-0.8b UNIFIED=1 LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1 \
              PROJ_RC_CACHE=1

# The fork's kernels and model header only: the kernels include their headers
# by quoted path, which resolves next to the including file, so the parent's
# headers cannot be shadowed and the fork carries its own copies.
PEANO_KBASE := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=QWEN3_5_08B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(srcdir)/kernels -I $(srcdir)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DDECODE_ROUND_NEAREST

GLU_SLICE_EXPECTED := $(shell $(DECODE_ENV) FUSED_DECODE_PRINT_CONST=GLU_SLICE $(PYTHON) $(DECODE_DRIVER) 2>/dev/null | tail -1)
PEANO_KBASE += $(if $(GLU_SLICE_EXPECTED),-DGLU_SLICE_EXPECTED=$(GLU_SLICE_EXPECTED))

KERNEL_DIR      := $(BUILD)
KERNEL_SRC_DIRS := $(srcdir)/kernels
KERNEL_FLAGS     = $(PEANO_KBASE)
O2_KERNELS      := $(NONATTN_KERNELS)
LL_KERNELS      := $(ATTN_LL_KERNELS)
LL_OPT           = $(ATTN_OPT)
include $(DEC)/decode_kernels.mk

.PHONY: help compile-decode-elf kernels preflight clean

help:
	@echo "make compile-decode-elf BUILD=<dir> LBUILD=<capacity>   kernels + one full ELF"
	@echo "make kernels BUILD=<dir>                                kernels only"

## ../fused_decode's preflight: an llvm-link older than LLVM 23 on PATH (the
## inline-attention merge), and a Peano install.
preflight:
	@LLVM_LINK=$$(command -v llvm-link || true); \
	  if [ -z "$(PEANO_INSTALL_DIR)" ] || [ ! -x "$(PEANO_INSTALL_DIR)/bin/clang++" ]; then \
	    echo "ERROR: set PEANO_INSTALL_DIR to an llvm-aie install"; exit 1; fi; \
	  if [ -z "$$LLVM_LINK" ]; then echo "ERROR: no llvm-link on PATH (need LLVM <23)"; exit 1; fi; \
	  VER=$$($$LLVM_LINK --version | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | head -1); \
	  echo "    llvm-link = $$LLVM_LINK (LLVM $$VER)"; \
	  if [ "$$VER" -ge 23 ]; then \
	    echo "ERROR: llvm-link is LLVM $$VER; the inline-attn merge needs LLVM <23 (see ../fused_decode/README.md)"; exit 1; fi

kernels: preflight
	@mkdir -p $(BUILD)
	@# a failed kernel build must not leave the previous ELF behind
	@rm -f $(BUILD)/decode_scratchpad.*
	@$(MAKE) --no-print-directory -f $(srcdir)/Makefile decode-kernels BUILD=$(BUILD)

## One full ELF for every context length up to LBUILD (the KV cache capacity):
## the KV append offset, the readback block count and the cores' block bound
## are scratchpad parameters written per token (../fused_decode/decode_elf.py).
## The outputs are removed first, so a failed build cannot leave a stale ELF
## behind.
compile-decode-elf: kernels
	@echo ">>> fused_decode_gdn full ELF (ATTN_MAXL=$(LBUILD)) -> $(BUILD)"
	@cd $(BUILD) && rm -f decode.elf params.txt decode.args decode_scratchpad.*; \
	  $(DECODE_ENV) DECODE_OUTPUT_FORMAT=elf DECODE_GOLDEN_L=$(LBUILD) DECODE_DYNSEQ=1 DECODE_RT_ROUNDS=1 \
	  $(PYTHON) $(DECODE_DRIVER) > $(BUILD)/build.log 2>&1; rc=$$?; \
	  if [ $$rc -eq 0 ] && [ -f decode.elf ] && [ -s params.txt ] && [ -s decode.args ]; then \
	    mv -f decode.elf decode_scratchpad.elf; cp -f params.txt decode_scratchpad.params.txt; \
	    mv -f decode.args decode_scratchpad.args; \
	    echo $$(( (($(LBUILD)+15)/16)*16 )) > decode_scratchpad.maxl; \
	    echo "fused_decode_gdn full-ELF build complete: $(BUILD)/decode_scratchpad.elf"; \
	  else echo "fused_decode_gdn ELF build FAILED (see $(BUILD)/build.log)"; tail -30 $(BUILD)/build.log; exit 1; fi

clean:
	rm -rf $(BUILD)
