# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-1B int4-AWQ Prefill on NPU2 (MLIR-AIR)
#
# Quick start:
#   make compile         — Compile both prefill backends (int4 + bf16)
#   make run             — Run prefill end-to-end (default: --prefill-dtype=bf16)
#   make verify          — Top-K token-level inclusion gate vs HF bf16
#   make profile         — Run with per-kernel profiling breakdown
#
# Performance (NPU2 AMD Strix, seq=2048, 16 layers, NPU flash_attn):
#   --prefill-dtype=int4 :  ~700 ms/layer  (kernel-bound; see PR for bisection)
#   --prefill-dtype=bf16 :  ~80 ms/layer   (~9x faster, same AWQ weights)
#
# Decode is not yet wired in this directory — see ../llama32_1b/ for the
# bf16 decode driver; the int4 decode driver will land in a follow-up PR.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Configurable parameters
MODEL         ?= amd/Llama-3.2-1B-Instruct-awq-uint4-asym-g128-bf16-lmhead
PROMPT        ?= The capital of France is
N_LAYERS      ?= 16
PREFILL_DTYPE ?= bf16
TOPK          ?= 10
MIN_OVERLAP   ?= 6

.PHONY: help compile run profile verify verify-int4 clean

# ============================================================
# Help
# ============================================================

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-1B int4-AWQ Prefill on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile both prefill backends"
	@echo "  make run              Run prefill end-to-end + topK vs HF"
	@echo "  make verify           Same as run but gates on PASS / FAIL"
	@echo "  make profile          Run with per-kernel profiling"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove build artifacts and kernel cache"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  PREFILL_DTYPE=bf16    Backend: bf16 (fast, default) or int4"
	@echo "  MODEL=<HF id>         AWQ checkpoint (default: AMD Llama-3.2-1B AWQ)"
	@echo "  PROMPT=\"...\"          Input prompt"
	@echo "  N_LAYERS=16           Transformer layers to run"
	@echo "  TOPK=10               Top-K window for HF comparison"
	@echo "  MIN_OVERLAP=6         PASS threshold for 'make verify'"
	@echo ""
	@echo "Examples:"
	@echo "  make run PREFILL_DTYPE=int4"
	@echo "  make verify PREFILL_DTYPE=bf16 N_LAYERS=4"
	@echo "  make profile PREFILL_DTYPE=bf16 PROMPT=\"How does photosynthesis work?\""

# ============================================================
# Targets
# ============================================================

## Compile prefill stitchers + flash_attn for both backends. Caches ELFs
## under verify_kernel_cache/. Re-uses cached ELFs across both backends.
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_prefill.py \
		--prefill-dtype int4 --compile-only --model $(MODEL) \
		--cache-dir $(srcdir)/verify_kernel_cache
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_prefill.py \
		--prefill-dtype bf16 --compile-only --model $(MODEL) \
		--cache-dir $(srcdir)/verify_kernel_cache
	@echo "Compilation passed."

## Run NPU prefill end-to-end + top-K comparison vs HF bf16 reference.
run:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_prefill.py \
		--prefill-dtype $(PREFILL_DTYPE) --model $(MODEL) \
		--prompt "$(PROMPT)" --n-layers $(N_LAYERS) --topk $(TOPK) \
		--cache-dir $(srcdir)/verify_kernel_cache

## Run NPU prefill + top-K gate. Prints '[verify] PASS' when overlap
## >= MIN_OVERLAP AND argmax matches HF (matched by run_npu2_verify.lit).
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_prefill.py \
		--prefill-dtype $(PREFILL_DTYPE) --model $(MODEL) \
		--prompt "$(PROMPT)" --n-layers $(N_LAYERS) --topk $(TOPK) \
		--min-overlap $(MIN_OVERLAP) \
		--cache-dir $(srcdir)/verify_kernel_cache

## Run with the int4 backend (slower compute, low-memory).
verify-int4:
	$(MAKE) -f $(srcdir)/Makefile verify PREFILL_DTYPE=int4

## Run with per-kernel + per-layer profiling instrumentation.
profile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_prefill.py \
		--prefill-dtype $(PREFILL_DTYPE) --model $(MODEL) \
		--prompt "$(PROMPT)" --n-layers $(N_LAYERS) --topk $(TOPK) \
		--profile \
		--cache-dir $(srcdir)/verify_kernel_cache

## Remove build dir and the on-disk ELF cache.
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	rm -rf $(srcdir)/verify_kernel_cache
	@echo "Build directory and verify_kernel_cache/ removed. Run 'make compile' to rebuild."
