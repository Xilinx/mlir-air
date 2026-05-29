# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-1B BF16 Inference on NPU2 (MLIR-AIR)
#
# Quick start:
#   make compile        — Compile all kernels (~3 min, cached to disk)
#   make run            — Run inference: NPU prefill + NPU decode
#   make profile        — Run with per-kernel profiling breakdown
#
# Performance (NPU2, AMD Strix):
#   Prefill:  1.27s (seq_len=2048, 16 layers)
#   Decode:   92ms/token (10.8 tokens/sec)

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Build directory
ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Configurable parameters
N_TOKENS ?= 1000
PROMPT   ?= What is the capital of France?
MODEL    ?= instruct

.PHONY: help compile run profile chat verify diagnosis clean

# ============================================================
# Help
# ============================================================

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-1B Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (~4 min, one-time)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           Top-k token-level inclusion gate vs HF bf16 (8 prompts × 32 tokens, k=5)"
	@echo "  make diagnosis        Per-layer ffn_out cosine + max_abs vs HF bf16 (single prompt, informational)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build artifacts and verify reports"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens for run/profile/chat (instruct stops early on EOT)"
	@echo "  PROMPT=\"...\"          Input prompt text (run/profile/diagnosis)"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo ""
	@echo "Examples:"
	@echo "  make run N_TOKENS=50"
	@echo "  make run MODEL=base PROMPT=\"The capital of France is\" N_TOKENS=200"
	@echo "  make profile PROMPT=\"How does photosynthesis work?\""
	@echo "  make verify MODEL=base"
	@echo "  make diagnosis PROMPT=\"The capital of France is\""

# ============================================================
# Unified Pipeline (NPU prefill + NPU decode)
# ============================================================

## Compile all kernels (prefill + decode, ~4 min)
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py --compile-only

## Run unified inference
run:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with detailed profiling breakdown
profile:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL)

## Interactive chat: prepare runtime once, then loop on prompts
chat:
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

## Compile and run in one step
all: compile profile

## Run the top-k token-level inclusion gate (NPU vs HF bf16, 8 prompts × 32 tokens, k=5)
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/verify/verify_runner.py \
		--prompts topk_token --model $(MODEL)

## Run the diagnosis lens (per-layer ffn_out cosine vs HF bf16, single prompt, informational)
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/verify/verify_runner.py \
		--prompts single --prompt "$(PROMPT)" --model $(MODEL)

# ============================================================
# Clean
# ============================================================

## Remove all build artifacts and verify reports
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	rm -rf $(srcdir)/verify/reports
	@echo "Build directory and verify/reports/ removed. Run 'make compile' to rebuild."
