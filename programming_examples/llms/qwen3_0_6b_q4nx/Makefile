# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-0.6B Q4NX Inference on NPU2 (MLIR-AIR)
#
# Same on-device pipeline as the bf16 qwen3_0_6b example, but weights come from
# FastFlowLM's 4-bit model.q4nx bundle (I8/GGUF-Q4_1-derived codec, dequant->bf16
# on the host at load).
#
# Quick start:
#   make compile        — Compile all kernels (cached; no weights needed)
#   make run            — Run inference: NPU prefill + NPU decode
#   make verify         — Top-k token-set inclusion gate: NPU q4nx vs HF bf16

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

N_TOKENS ?= 1000
PROMPT   ?= What is the capital of France?
MODEL    ?= instruct
# Q4NX weight source (HF repo id with a model.q4nx, or a local dir/file).
MODEL_SOURCE ?= FastFlowLM/Qwen3-0.6B-NPU2
export Q4NX_MODEL_SOURCE := $(MODEL_SOURCE)

DRIVER := $(srcdir)/qwen3_0_6b_q4nx_inference.py

.PHONY: help compile run profile chat verify verify-full diagnosis clean

help:
	@echo "============================================================"
	@echo " Qwen3-0.6B Q4NX Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (one-time, cached; no weights)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           Top-k token-set inclusion gate: NPU q4nx vs HF bf16 (2 prompts x 32 tokens, k=5)"
	@echo "  make verify-full      Same gate over the full prompt set"
	@echo "  make diagnosis        Per-layer ffn_out cosine vs HF bf16 (single prompt, informational)"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens"
	@echo "  PROMPT=\"...\"          Input prompt text"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo "  MODEL_SOURCE=...       Q4NX weight source (default: $(MODEL_SOURCE))"

## Compile all kernels (prefill + decode; no weights needed)
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(DRIVER) --compile-only

## Run unified inference
run:
	cd $(BUILD_DIR) && python3 $(DRIVER) \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with detailed profiling breakdown
profile:
	cd $(BUILD_DIR) && python3 $(DRIVER) \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL)

## Interactive chat REPL
chat:
	cd $(BUILD_DIR) && python3 $(DRIVER) \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = qwen3_0_6b_q4nx.verify_adapter

## Top-k token-set inclusion gate (NPU q4nx vs HF bf16, 2 prompts x 32 tokens, k=5).
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2

## Full-sweep variant of `make verify`.
verify-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL)

## Diagnosis lens (per-layer ffn_out cosine vs HF bf16, single prompt, informational).
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)" --model $(MODEL)

## Remove all build artifacts and verify reports.
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports
	@echo "Build directory and verify/reports/ removed. Run 'make compile' to rebuild."
