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

# Padded prefill length (see `make profile-prefill`): 512 | 1024 | 2048 | 4096.
# Bounded below by the fused-cast GEMM (multiple of 64) and the seq-first
# FlashAttention (multiple of 256), and above by the GEMM registry, which has
# no measured shape past M=4096.
CTX ?= 2048
export LLM_SEQ_LEN   := $(CTX)
export LLM_CACHE_DIR := prefill_kernel_cache_$(CTX)

# KV depths `make profile-decode` measures decode tok/s at. Unlike CTX these
# need no rebuild -- the decode kernels are per-token GEMVs with no context
# dependence -- so one run walks the whole list. Bounded by the host KV cache:
# n_layers * n_kv_heads * L * head_dim * 4 bytes, 4.3GB at 128k here.
CONTEXTS ?= 1024,2048,4096,8192,16384,32768,65536,131072

.PHONY: help compile run profile profile-prefill profile-decode chat verify \
        verify-full diagnosis clean

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
	@echo "  make verify           Top-k token-level inclusion gate vs HF bf16 (2 prompts × 32 tokens, k=5) — fast CI gate"
	@echo "  make verify-full      Same as above but runs the full 8-prompt set (longer, exhaustive)"
	@echo "  make diagnosis        Per-layer ffn_out cosine + max_abs vs HF bf16 (single prompt, informational)"
	@echo ""
	@echo "Prefill latency sweep:"
	@echo "  make profile-prefill  Warm prefill TTFT at CTX=$(CTX) (builds the engines at that length)"
	@echo ""
	@echo "Decode throughput sweep:"
	@echo "  make profile-decode   Decode tok/s at each CONTEXTS depth (one run, no rebuild per point)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Remove all build artifacts and verify reports"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens for run/profile/chat (instruct stops early on EOT)"
	@echo "  PROMPT=\"...\"          Input prompt text (run/profile/diagnosis)"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo "  CTX=2048              Padded prefill length (512|1024|2048|4096)"
	@echo "  CONTEXTS=1024,...     KV depths for profile-decode"
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
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/llama32_1b_inference.py --compile-only

## Run unified inference
run:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with detailed profiling breakdown
profile:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/llama32_1b_inference.py \
		--run-only --n-tokens $(N_TOKENS) --profile --prompt "$(PROMPT)" --model $(MODEL)

## Warm prefill-only TTFT at CTX. Builds the engines at that length first --
## the kernel cache is keyed on kernel name, so reusing another CTX's directory
## would time the wrong engines and report a flat curve.
profile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/llama32_1b_inference.py --compile-only
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/llama32_1b_inference.py \
		--run-only --bench-prefill --model $(MODEL)

## Decode tok/s at each CONTEXTS depth, in ONE session. The context is set by
## sizing the KV cache, not by prefilling a prompt of that length, so a 128k
## point costs seconds. Latency only -- `make verify` is the correctness gate.
profile-decode:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/llama32_1b_inference.py --compile-only
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/llama32_1b_inference.py \
		--run-only --bench-decode $(CONTEXTS) --model $(MODEL)

## Interactive chat: prepare runtime once, then loop on prompts
chat:
	cd $(BUILD_DIR) && $(PYTHON) $(srcdir)/llama32_1b_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

## Compile and run in one step
all: compile profile

# The shared verify framework lives under programming_examples/llms/verify/.
# Each model contributes a `verify_adapter.py`; we point the framework at
# ours via `--runner=llama32_1b.verify_adapter`.
VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = llama32_1b.verify_adapter

## Run the top-k token-level inclusion gate (NPU vs HF bf16, 2 prompts × 32 tokens, k=5).
## This is the fast CI gate. For the full 8-prompt sweep, use `make verify-full`.
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2

## Full-sweep variant of `make verify`: runs all prompts in the prompt file
## (currently 8). Use locally for exhaustive validation; CI uses `make verify`.
verify-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL)

## Run the diagnosis lens (per-layer ffn_out cosine vs HF bf16, single prompt, informational)
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)" --model $(MODEL)

# ============================================================
# Clean
# ============================================================

## Remove all build artifacts and verify reports
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports
	@echo "Build directory and verify/reports/ removed. Run 'make compile' to rebuild."
