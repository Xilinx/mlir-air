# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-1B Q4NX Inference on NPU2 (MLIR-AIR)
#
# Same driver contract as the other llms/ examples: run/profile/chat drive the
# unified prefill+decode inference driver, and verify plugs into the shared
# verify/ subsystem (top-k token-set inclusion, NPU q4nx vs HF bf16). Weights
# come from FastFlowLM's 4-bit model.q4nx bundle (dequant->bf16 on the host).
#
# Quick start:
#   make compile        — Build/cache the prefill ELFs (~3 min; no weights needed)
#   make compile-decode — Build the fused decode templates (~15 min; no weights)
#   make run            — Run inference: NPU prefill + fused NPU decode
#   make verify         — Top-k token-set inclusion gate: NPU q4nx vs HF bf16
#
# Config: 16 layers, emb=2048, 32 heads / 8 KV heads, head_dim=64,
#         hidden=8192, vocab=128256 (NPU2, AMD Strix).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Configurable parameters (shared driver contract)
N_TOKENS ?= 1000
PROMPT   ?= What is the capital of France?
MODEL    ?= instruct
# Q4NX weight source (HF repo id with a model.q4nx, or a local dir/file).
MODEL_SOURCE ?= FastFlowLM/Llama-3.2-1B-NPU2
export Q4NX_MODEL_SOURCE := $(MODEL_SOURCE)

# Prefill-only latency sweep (see `make profile-prefill`): 256 | 512 | 1024 | 2048.
CTX ?= 2048
export Q4NX_SEQ_LEN   := $(CTX)
export Q4NX_CACHE_DIR := $(srcdir)/$(BUILD_DIR)/q4nx_kernel_cache_$(CTX)
# Tuned temporal+causal-skip prefill FlashAttention (falls back to seqfirst when
# CTX % 512 != 0). This is the headline ~0.93s TTFT prefill path.
export TEMPORAL_CAUSAL_SKIP := 1

# Dual-MM2S weight feed for the fused decode (see fused_decode/fused_decode.py).
# The BUILD and the RUN must agree -- the flag selects both the shim channel split
# and the weight cascade order -- so export it here and let it reach both the
# delegated `compile-decode` and the inference driver. The requant cache is keyed
# on it, so flipping it does not need a manual cache purge.
W_DUAL_CHAN ?= 1
export W_DUAL_CHAN

# Unified prefill+decode inference driver (run/profile/chat/gen/ask/compile).
DRIVER := $(srcdir)/llama32_1b_q4nx_inference.py
# Prefill engine (weight-integrity Paris gate + the CTX prefill latency sweep).
PREFILL := $(srcdir)/llama32_1b_q4nx_prefill.py
# The fused per-token decode is a SEPARATE reusable example: one dispatch =
# 16 layers + LM head. `make compile-decode` delegates to it; the inference
# driver loads its templates + decode_insts_gen at run time.
FUSED_DECODE := $(srcdir)/../../fused_decode

# Shared verify framework (programming_examples/llms/verify/).
VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = llama32_1b_q4nx.verify_adapter

.PHONY: help compile run profile chat verify verify-full diagnosis clean \
        compile-decode compile-decode-windows compile-decode-dynseq profile-prefill verify-paris ask gen

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-1B Q4NX Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Build/cache the prefill ELFs (~3 min, no weights)"
	@echo "  make compile-decode   Build the fused decode templates (~15 min, no weights)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens): NPU prefill + decode"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "Correctness:"
	@echo "  make verify           Top-k token-set inclusion gate: NPU q4nx vs HF bf16 (2 prompts x 32 tokens, k=5)"
	@echo "  make verify-full      Same gate over the full prompt set"
	@echo "  make diagnosis        Per-layer ffn_out cosine vs HF bf16 (single prompt, informational)"
	@echo "  make verify-paris     Fast weight-integrity smoke: prefill first-token argmax gate (12366 ' Paris')"
	@echo ""
	@echo "Prefill latency sweep:"
	@echo "  make profile-prefill  Warm prefill TTFT benchmark at CTX (per-ELF breakdown)"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens for run/profile/chat"
	@echo "  PROMPT=\"...\"          Input prompt text (run/profile/diagnosis)"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo "  CTX=2048              Prefill context length for profile-prefill: 256 | 512 | 1024 | 2048"
	@echo "  MODEL_SOURCE=...       Q4NX weight source (default: $(MODEL_SOURCE))"
	@echo ""
	@echo "Data (external overrides; see README.md):"
	@echo "  Q4NX_GOLDEN_DIR / Q4NX_DECODE_WEIGHTS_NPZ pre-supply the derived decode caches."

## Build/cache the prefill ELFs (rms_gemms_rope + flash_attn + o_ffn + lm_head_gemv).
## No weight data or NPU dispatch required.
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) --compile-only

## Build the fused decode (delegates to the standalone programming_examples/fused_decode
## example): the Peano kernels + two templates (decode_L2048 + decode_L2047). ~15 min.
## Weight-free build; needs a pre-change llvm-link (see that example's README
## Reproducibility). The inference driver loads its templates at run time.
compile-decode:
	$(MAKE) -C $(FUSED_DECODE) compile-decode

## Staircase templates (one pair per ATTN_MAXL window); run with DECODE_STAIRCASE=1.
compile-decode-windows:
	$(MAKE) -C $(FUSED_DECODE) compile-decode-windows

## Dynamic-runtime-sequence template: ONE build for every context length, taking
## the context length as a dispatch-time scalar. Run with DECODE_DYNSEQ=1. The
## 1B is fused_decode.py's default DECODE_MODEL, so this delegates unqualified,
## exactly as `compile-decode` does.
compile-decode-dynseq:
	$(MAKE) -C $(FUSED_DECODE) compile-decode-dynseq

## Run unified inference (NPU prefill + fused NPU decode).
run:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with the decode-throughput / TTFT profiling summary. --no-eos-stop so
## throughput is averaged over the full N_TOKENS budget: the default prompt
## answers in ~6 tokens, and a 6-sample average is dominated by the first
## (cold) dispatch and understates steady-state tok/s by ~20%.
profile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) \
		--run-only --n-tokens $(N_TOKENS) --profile --no-eos-stop \
		--prompt "$(PROMPT)" --model $(MODEL)

## Interactive chat REPL (streams the reply token-by-token; /clear resets, /exit quits).
chat:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

# ---- shared verify subsystem (top-k token-set inclusion, NPU q4nx vs HF bf16) ----

## Top-k token-set inclusion gate (NPU q4nx vs HF bf16, 2 prompts x 32 tokens, k=5).
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2

## Full-sweep variant of `make verify` (all prompts in the prompt file).
verify-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL)

## Diagnosis lens (per-layer ffn_out cosine vs HF bf16, single prompt, informational).
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)" --model $(MODEL)

## Fast weight-integrity smoke: prefill "The capital of France is"; PASS iff the
## first-token argmax == 12366 (" Paris"). Cheap (prefill only, no decode build).
verify-paris:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(PREFILL) --model $(MODEL_SOURCE)

## Warm prefill TTFT benchmark at CTX (per-ELF BO-write / NPU-run / BO-read breakdown).
profile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && Q4NX_BENCH_L=$(CTX) $(PYTHON) $(PREFILL)

## Single Q&A turn:  make ask PROMPT="What is the capital of France?"
ask:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) --run-only --prompt "$(PROMPT)" --model $(MODEL) --n-tokens $(N_TOKENS)

## Full prefill+decode Paris gate: first generated token 12366 -> *** PARIS ***.
gen:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) --run-only --greedy --n-tokens 12

## Remove all build artifacts, kernel/weight caches, and verify reports. Also cleans
## the fused_decode example, because this one's decode templates live THERE: leaving
## them makes the next `compile-decode` a no-op (it skips when both are present), so
## `clean && compile-decode` would verify whatever binaries ran last -- including
## another model's, or ones built by a different compiler.
clean:
	rm -rf $(BUILD_DIR) $(srcdir)/_q4nx_* $(srcdir)/.wcache 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports 2>/dev/null || true
	@$(MAKE) --no-print-directory -C $(FUSED_DECODE) clean
	@echo "Build dirs, caches, verify/reports/ and decode templates removed. Rebuild with 'make compile'."
