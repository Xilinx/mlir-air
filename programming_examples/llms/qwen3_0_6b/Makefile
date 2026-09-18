# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-0.6B BF16 Inference on NPU2 (MLIR-AIR)
#
# Quick start:
#   make compile        — Compile all kernels (cached to disk)
#   make run            — Run inference: NPU prefill + NPU decode
#   make profile        — Run with per-kernel profiling breakdown

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

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

# Padded prefill length (see `make profile-prefill`). Bounded below by the
# fused-cast GEMM (multiple of 64) and the seq-first FlashAttention (multiple of
# 256), and above by the GEMM registry, which has no measured shape past
# M=4096. Which lengths this model actually has shapes for is recorded in its
# run_npu2_prefill_sweep.lit.
CTX ?= 2048
export LLM_SEQ_LEN   := $(CTX)
export LLM_CACHE_DIR := prefill_kernel_cache_$(CTX)

# KV depths `make profile-decode` measures decode tok/s at. These need no
# rebuild -- the decode kernels are per-token GEMVs with no context
# dependence and attention runs on the host -- so one run walks the whole
# list. The list is bounded by the host KV cache, which is numpy:
# n_layers * n_kv_heads * L * head_dim * 4 bytes.
CONTEXTS ?= 1024,2048,4096,8192,16384,32768

.PHONY: help compile run profile profile-prefill profile-decode chat \
        verify verify-full \
        diagnosis all clean

help:
	@echo "============================================================"
	@echo " Qwen3-0.6B Inference on NPU2 (MLIR-AIR)"
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

## Compile all kernels
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py --compile-only

## Run unified inference
run:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with detailed profiling breakdown
profile:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL)

## Warm prefill-only TTFT at CTX. Builds the engines at that length first --
## the kernel cache is keyed on kernel name, so reusing another CTX's directory
## would time the wrong engines and report a flat curve.
profile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py --compile-only
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py \
		--run-only --bench-prefill --model $(MODEL)

## Decode tok/s at each CONTEXTS depth, in ONE session. The context is set by
## sizing the KV cache, not by prefilling a prompt of that length, so a deep
## point costs seconds. Latency only -- `make verify` is the correctness gate.
profile-decode:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py --compile-only
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py \
		--run-only --bench-decode $(CONTEXTS) --model $(MODEL)

## Interactive chat
chat:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/qwen3_0_6b_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

all: compile profile

VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = qwen3_0_6b.verify_adapter

## Top-k token-level inclusion gate (NPU vs HF bf16, 2 prompts × 32 tokens, k=5).
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2

## Full-sweep variant of `make verify`.
verify-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL)

## Diagnosis lens (per-layer ffn_out cosine vs HF bf16, single prompt, informational)
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)" --model $(MODEL)

## Remove all build artifacts and verify reports
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports
	@echo "Build directory and verify/reports/ removed. Run 'make compile' to rebuild."
