# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-1B Q4NX prefill on NPU2 (MLIR-AIR)
#
# Quick start:
#   make compile        — Build/cache the 4 prefill ELFs (~3 min; no weights needed)
#   make run            — Prefill the "Paris" prompt; gate on first-token argmax 12366
#   make profile        — Warm TTFT benchmark at CTX
#
# Performance (NPU2, AMD Strix), warm TTFT:
#   ctx 256 -> 243 ms | 512 -> 305 ms | 1024 -> 560 ms | 2048 -> ~933 ms
#
# Unlike the HF-reference examples, this one gates on the greedy first token
# (" Paris", 12366), not `make verify` vs HF bf16. It needs external weight data
# (see README.md):
#   PARIS_WEIGHTS : per-layer Q4NX bins  (L{k}_proj_w.bin / L{k}_rms_w.bin)
#   PARIS_GOLDEN  : golden embed/final_norm/lm_head (gen_paris_golden.py)

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Configurable parameters
CTX           ?= 2048
DEVICE_DEQUANT ?= 0
# Weight source: HF repo id (model.q4nx bundle) or a local dir/file. Default is the
# self-contained model.q4nx on the Hub; the prefill downloads + dequantizes it.
MODEL         ?= FastFlowLM/Llama-3.2-1B-NPU2
export Q4NX_MODEL         := $(MODEL)
export Q4NX_SEQ_LEN       := $(CTX)
export Q4NX_DEVICE_DEQUANT := $(DEVICE_DEQUANT)
export Q4NX_CACHE_DIR     := $(srcdir)/$(BUILD_DIR)/q4nx_kernel_cache_$(CTX)
# Use the tuned temporal+causal-skip prefill FlashAttention (auto-falls back to
# seqfirst when CTX % 512 != 0). This is the headline ~0.93s TTFT path.
export TEMPORAL_CAUSAL_SKIP := 1

DRIVER := $(srcdir)/llama32_1b_q4nx_prefill.py

# ---- fused per-token decode (the chatbot's second path) ----
# The decode is a SEPARATE example: programming_examples/fused_decode/ (one dispatch =
# 16 layers + LM head). `make compile-decode` here delegates to it; the chatbot driver
# loads its templates + decode_insts_gen at run time.
INFER          := $(srcdir)/llama32_1b_q4nx_inference.py
FUSED_DECODE   := $(srcdir)/../../fused_decode
PARIS_WEIGHTS  ?= $(HOME)/q4nx_data/weights

.PHONY: help compile run verify profile clean compile-decode chat ask gen

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-1B Q4NX prefill on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Prefill (single-path):"
	@echo "  make compile          Build/cache the 4 prefill ELFs (~3 min, no weights)"
	@echo "  make run              Prefill 'The capital of France is' -> Paris gate"
	@echo "  make verify           Same Paris argmax gate (12366) as the CI correctness check"
	@echo "  make profile          Warm TTFT benchmark at CTX"
	@echo ""
	@echo "Chatbot (prefill + fused decode):"
	@echo "  make compile-decode   Build the fused_decode templates (delegates; ~15 min)"
	@echo "  make chat             Interactive multi-turn chatbot (streams tokens)"
	@echo "  make ask PROMPT=...   Single Q&A turn"
	@echo "  make gen              Full prefill+decode Paris gate -> *** PARIS ***"
	@echo "  (needs PARIS_REQUANT_CACHE ~28min once + PARIS_GOLDEN; see README.md Build)"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  CTX=2048              Prefill context length: 256 | 512 | 1024 | 2048"
	@echo "  DEVICE_DEQUANT=1      Dequant Q4NX->bf16 ON-DEVICE (default 0 = host dequant)"
	@echo ""
	@echo "Data (external; see README.md):"
	@echo "  PARIS_WEIGHTS, PARIS_GOLDEN env vars point at the Q4NX weights + HF golden."
	@echo ""
	@echo "Examples:"
	@echo "  make run CTX=1024"
	@echo "  make profile CTX=2048"
	@echo "  make run DEVICE_DEQUANT=1"

## Build/cache the 4 prefill ELFs (rms_gemms_rope + flash_attn + o_ffn + lm_head_gemv).
## No weight data or NPU dispatch required.
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(DRIVER) --compile-only

## Prefill the canonical prompt; PASS iff first-token argmax == 12366 (" Paris").
run:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(DRIVER)

## Correctness gate: the HF greedy first-token argmax (" Paris", 12366). This is the
## Q4NX analogue of the other examples' `make verify` vs HF bf16 (which does not apply
## here — no HF checkpoint, external Q4NX weights). `run` and `verify` are the same
## gate; `verify` is the name CI expects.
verify: run

## Warm TTFT benchmark at CTX (per-ELF BO-write / NPU-run / BO-read breakdown).
profile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && Q4NX_BENCH_L=$(CTX) python3 $(DRIVER)

## Build the fused decode (delegates to the standalone programming_examples/fused_decode
## example): the Peano kernels + two templates (decode_L2048 + decode_L2047). ~15 min.
## Needs PARIS_WEIGHTS for the GEMM shapes and a pre-change llvm-link (see that example's
## README Reproducibility). The chatbot driver loads its templates at run time.
compile-decode:
	$(MAKE) -C $(FUSED_DECODE) compile-decode PARIS_WEIGHTS=$(PARIS_WEIGHTS)

## Interactive multi-turn chatbot (streams the reply token-by-token; /clear resets, /exit quits).
chat:
	python3 $(INFER) --interactive $(CHAT_ARGS)

## Single Q&A turn:  make ask PROMPT="What is the capital of France?"
ask:
	python3 $(INFER) --prompt "$(PROMPT)" $(CHAT_ARGS)

## Full prefill+decode Paris gate: first generated token 12366 -> *** PARIS ***.
gen:
	python3 $(INFER) --greedy --n-tokens 12

## Remove prefill build artifacts + kernel/weight caches. (Decode templates live in the
## fused_decode example; run `make -C ../../fused_decode clean` to remove those.)
clean:
	rm -rf $(BUILD_DIR) $(srcdir)/_q4nx_* $(srcdir)/.wcache 2>/dev/null || true
	@echo "Prefill build dirs and caches removed. Rebuild with 'make compile'."
