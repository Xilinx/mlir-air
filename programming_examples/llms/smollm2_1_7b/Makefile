# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# SmolLM2-1.7B BF16 Inference on NPU2 (MLIR-AIR)
#
# Quick start:
#   make compile        — Compile all kernels (cached to disk)
#   make run            — Run inference: NPU prefill + NPU decode
#   make profile        — Run with per-kernel profiling breakdown

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

.PHONY: help compile run profile chat verify verify-full diagnosis all clean

# ============================================================
# Help
# ============================================================

help:
	@echo "============================================================"
	@echo " SmolLM2-1.7B Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (one-time, cached)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           Top-k token-level inclusion gate vs HF bf16 (2 prompts × 32 tokens, k=5) — fast CI gate"
	@echo "  make verify-full      Same as above but runs the full prompt set (longer, exhaustive)"
	@echo "  make diagnosis        Per-layer ffn_out cosine + max_abs vs HF bf16 (single prompt, informational)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build artifacts and verify reports"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens for run/profile/chat"
	@echo "  PROMPT=\"...\"          Input prompt text (run/profile/diagnosis)"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"

# ============================================================
# Unified Pipeline (NPU prefill + NPU decode)
# ============================================================

## Compile all kernels
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/smollm2_1_7b_inference.py --compile-only

## Run unified inference
run:
	cd $(BUILD_DIR) && python3 $(srcdir)/smollm2_1_7b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with detailed profiling breakdown
profile:
	cd $(BUILD_DIR) && python3 $(srcdir)/smollm2_1_7b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL)

## Interactive chat: prepare runtime once, then loop on prompts
chat:
	cd $(BUILD_DIR) && python3 $(srcdir)/smollm2_1_7b_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

## Compile and run in one step
all: compile profile

# The shared verify framework lives under programming_examples/llms/verify/.
# Each model contributes a `verify_adapter.py`; we point the framework at
# ours via `--runner=smollm2_1_7b.verify_adapter`.
VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = smollm2_1_7b.verify_adapter

## Run the top-k token-level inclusion gate (NPU vs HF bf16, 2 prompts × 32 tokens, k=5).
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2

## Full-sweep variant of `make verify`.
verify-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL)

## Run the diagnosis lens (per-layer ffn_out cosine vs HF bf16, single prompt, informational)
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)" --model $(MODEL)

# ============================================================
# Clean
# ============================================================

## Remove all build artifacts and verify reports
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports
	@echo "Build directory and verify/reports/ removed. Run 'make compile' to rebuild."
