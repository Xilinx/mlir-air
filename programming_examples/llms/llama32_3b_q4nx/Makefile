# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-3B Q4NX Inference on NPU2 (MLIR-AIR)
#
# Same on-device pipeline as the bf16 llama32_3b example, but weights come from
# FastFlowLM's 4-bit model.q4nx bundle (dequant->bf16 on the host at load).
#
# Quick start:
#   make compile        — Compile all kernels (~4 min, cached; no weights needed)
#   make run            — Run inference: NPU prefill + NPU decode
#   make verify         — Top-k token-set inclusion gate: NPU q4nx vs HF bf16
#
# Config: 28 layers, emb=3072, 24 heads / 8 KV heads, head_dim=128,
#         hidden=8192, vocab=128256 (NPU2, AMD Strix).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Configurable parameters
N_TOKENS ?= 1000
PROMPT   ?= What is the capital of France?
MODEL    ?= instruct
# Q4NX weight source (HF repo id with a model.q4nx, or a local dir/file).
MODEL_SOURCE ?= FastFlowLM/Llama-3.2-3B-NPU2
export Q4NX_MODEL_SOURCE := $(MODEL_SOURCE)

DRIVER := $(srcdir)/llama32_3b_q4nx_inference.py

.PHONY: help compile run profile chat verify verify-full diagnosis clean

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-3B Q4NX Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile all kernels (~4 min, one-time; no weights)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           Top-k token-set inclusion gate: NPU q4nx vs HF bf16 (2 prompts x 32 tokens, k=5)"
	@echo "  make verify-full      Same gate over the full 8-prompt set"
	@echo "  make diagnosis        Per-layer ffn_out cosine vs HF bf16 (single prompt, informational)"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens for run/profile/chat"
	@echo "  PROMPT=\"...\"          Input prompt text (run/profile/diagnosis)"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo "  MODEL_SOURCE=...       Q4NX weight source (default: $(MODEL_SOURCE))"

## Compile all kernels (prefill + decode, ~4 min; no weights needed)
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

# The shared verify framework lives under programming_examples/llms/verify/.
VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = llama32_3b_q4nx.verify_adapter

## Top-k token-set inclusion gate (NPU q4nx vs HF bf16, 2 prompts x 32 tokens, k=5).
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2

## Full-sweep variant of `make verify` (all prompts in the prompt file).
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
