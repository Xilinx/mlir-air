# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-1B int4-AWQ end-to-end on NPU2 (MLIR-AIR)
#
# Quick start (prefill smoke / token-overlap verify):
#   make compile         — Compile both prefill backends (int4 + bf16)
#   make run             — Run prefill end-to-end (default: --prefill-dtype=bf16)
#   make verify          — Top-K token-level inclusion gate vs HF bf16
#
# Full e2e inference (bf16 NPU prefill + int4 NPU decode):
#   make compile-inference — Compile prefill ELFs + int4 decode ELFs
#   make run-inference     — Tokenize prompt, prefill, decode N tokens
#   make chat              — Interactive REPL (each turn independent)
#
# Performance (NPU2 AMD Strix, seq=2048, 16 layers, NPU flash_attn):
#   --prefill-dtype=int4 :  ~700 ms/layer  (kernel-bound; see PR for bisection)
#   --prefill-dtype=bf16 :  ~80 ms/layer   (~9x faster, same AWQ weights)
#   int4 NPU decode      :  ~56 ms/token   (17.8 tok/s on instruct model)

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
# Multi-prompt sweep. Default uses the bf16 sibling's instruct prompts file
# so both Llama examples gate against the same prompt set.
PROMPTS_FILE  ?= $(srcdir)/../verify/prompts/instruct.txt

N_TOKENS      ?= 30
PREFILL_INF_CACHE ?= $(srcdir)/inference_prefill_cache
DECODE_INF_CACHE  ?= $(srcdir)/inference_decode_cache

# The shared verify framework lives under programming_examples/llms/verify/.
# Each model contributes a `verify_adapter.py`; we point the framework at
# ours via `--runner=llama32_1b_int4.verify_adapter`.
VERIFY_RUNNER  = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = llama32_1b_int4.verify_adapter

.PHONY: help compile run profile verify-prefill verify-prefill-full \
        diagnosis-prefill verify verify-full diagnosis clean \
        compile-inference run-inference chat compile-decode

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
	@echo "End-to-end inference (bf16 NPU prefill + int4 NPU decode):"
	@echo "  make compile-inference  Compile prefill + decode kernels"
	@echo "  make run-inference      Decode N_TOKENS tokens after prefill"
	@echo "  make chat               Interactive REPL"
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

## Run NPU prefill end-to-end + top-K comparison vs HF bf16 reference
## (prefill-only argmax/overlap; for the full prefill+decode gate, see
## `make verify` which goes through the shared verify/ framework).
run:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_prefill.py \
		--prefill-dtype $(PREFILL_DTYPE) --model $(MODEL) \
		--prompt "$(PROMPT)" --n-layers $(N_LAYERS) --topk $(TOPK) \
		--cache-dir $(srcdir)/verify_kernel_cache

## Prefill-only top-K gate (single prompt, argmax + overlap). Faster but
## doesn't exercise decode. Use `make verify` for the full prefill+decode
## gate matching the bf16 baseline's CI gate.
verify-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_prefill.py \
		--prefill-dtype $(PREFILL_DTYPE) --model $(MODEL) \
		--prompt "$(PROMPT)" --n-layers $(N_LAYERS) --topk $(TOPK) \
		--min-overlap $(MIN_OVERLAP) \
		--cache-dir $(srcdir)/verify_kernel_cache

## Multi-prompt prefill-only sweep. Use `make verify-full` for the full
## prefill+decode equivalent through the shared framework.
verify-prefill-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_prefill.py \
		--prefill-dtype $(PREFILL_DTYPE) --model $(MODEL) \
		--n-layers $(N_LAYERS) --topk $(TOPK) \
		--min-overlap $(MIN_OVERLAP) \
		--prompts-file $(PROMPTS_FILE) \
		--cache-dir $(srcdir)/verify_kernel_cache

## Full prefill + decode top-K inclusion gate via the shared verify/
## framework (matches the bf16 baseline's `make verify`). 2 prompts × 32
## greedy tokens, k=5. NPU runs bf16 prefill on dequantized AWQ + int4
## decode; HF reference loads the same AWQ checkpoint in bf16.
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --max-prompts 2

## Same as `make verify` but runs the full prompt set (slower; local
## exhaustive validation). Mirrors the bf16 sibling's `make verify-full`.
verify-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token

## Diagnosis lens (per-layer ffn_out cosine). Note: int4 decode adapter
## does not expose per-layer intermediates yet — diagnosis works only for
## the prefill RMSNorm/QKV/RoPE stages. For full per-layer NPU-vs-HF
## probe, use the bf16 sibling's `make diagnosis`.
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)"

## Run with per-kernel + per-layer profiling instrumentation.
profile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_prefill.py \
		--prefill-dtype $(PREFILL_DTYPE) --model $(MODEL) \
		--prompt "$(PROMPT)" --n-layers $(N_LAYERS) --topk $(TOPK) \
		--profile \
		--cache-dir $(srcdir)/verify_kernel_cache

## ============================================================
## End-to-end inference (bf16 NPU prefill + int4 NPU decode)
## ============================================================

## Compile both prefill (bf16) ELFs and decode (int4) ELFs into separate
## kernel caches. Idempotent; re-run after editing stitchers.
compile-inference:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_inference.py \
		--compile-only \
		--model-path $(MODEL) \
		--prefill-cache-dir $(PREFILL_INF_CACHE) \
		--decode-cache-dir $(DECODE_INF_CACHE)
	@echo "Compilation passed."

## Decode-only kernel compile (no prefill). Useful when iterating on
## decode stitchers in isolation.
compile-decode:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(srcdir)/llama32_1b_int4_decode.py \
		--compile-only --cache-dir $(DECODE_INF_CACHE)

## End-to-end generation: prefill the prompt, then decode N_TOKENS tokens
## on the NPU. Override PROMPT, N_TOKENS, MODEL via the command line.
run-inference:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && flock /tmp/mlir-air-npu.lock python3 \
		$(srcdir)/llama32_1b_int4_inference.py \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" \
		--model-path $(MODEL) \
		--prefill-cache-dir $(PREFILL_INF_CACHE) \
		--decode-cache-dir $(DECODE_INF_CACHE)

## Interactive chat REPL — each prompt is independent (no chat history).
## Ctrl-D / /quit exits.
chat:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && flock /tmp/mlir-air-npu.lock python3 \
		$(srcdir)/llama32_1b_int4_inference.py \
		--run-only --interactive --n-tokens $(N_TOKENS) \
		--model-path $(MODEL) \
		--prefill-cache-dir $(PREFILL_INF_CACHE) \
		--decode-cache-dir $(DECODE_INF_CACHE)

## Remove build dir and all on-disk ELF caches (prefill verifier,
## inference prefill, inference decode).
clean:
	rm -r $(BUILD_DIR) 2>/dev/null || true
	rm -rf $(srcdir)/verify_kernel_cache $(PREFILL_INF_CACHE) $(DECODE_INF_CACHE)
	@echo "Build directory and kernel caches removed. Rebuild with:"
	@echo "  make compile             # prefill stitcher ELFs (verify/run/profile)"
	@echo "  make compile-inference   # bf16 prefill + int4 decode ELFs (run-inference/chat)"
