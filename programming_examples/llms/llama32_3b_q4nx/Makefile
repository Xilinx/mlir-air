# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LLAMA-3.2-3B Q4NX Inference on NPU2 (MLIR-AIR) — FLM-faithful decode
#
# The whole decoder layer runs on the AIE array (proj -> RoPE -> flash attention
# with an on-device KV cache -> o-proj -> FFN, x28, + LM head) in ONE dispatch per
# token via the fused_decode superkernel. No CPU attention, no host KV cache.
# Weights come from FastFlowLM's 4-bit model.q4nx bundle, re-quantized once into
# the decode's q4k-cascade layout and cached under .decode_wcache/.
#
# There is no separate prefill kernel: the prompt is consumed token-by-token
# through the same on-device decode, so each prompt token costs one decode step.
#
# Quick start:
#   make compile        — Build the fused-decode templates (~15 min; no weights needed)
#   make run            — Run inference: 28 layers + LM head on-device, per token
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
# The fused per-token decode engine is a SEPARATE reusable example (one dispatch =
# 28 layers + LM head); this Makefile drives it with the 3B model selection and
# builds the templates INTO THIS DIRECTORY, the same way gemma3_4b_q4nx does. Every
# model emits the same decode_L<N>.* names, so per-example output is what keeps a
# 1B build from silently satisfying a 3B run.
DEC           := $(srcdir)/../../fused_decode
DECODE_DRIVER := $(DEC)/fused_decode.py
AIEOPT_DIR    := $(shell realpath $(dir $(shell which aie-opt))/..)

# Per-kernel -O is LOAD-BEARING (see fused_decode/Makefile): proj_qmm/rms_residual/
# glu/rope -O2; attn_qk/attn_kv inline .ll -O1. 3B differs only by
# -DMODEL_TYPE=LLAMA_3_2_3B (the header sets the dims, DH=128, ATTN_IMPL, vocab).
NONATTN_KERNELS := proj_qmm rms_residual glu rope
ATTN_LL_KERNELS := attn_qk attn_kv
PEANO_KBASE := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=LLAMA_3_2_3B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(DEC)/kernels -I $(DEC)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

# 3B decode env: 28 layers, vocab chunked at VOCAB_CHUNK_I2=9. That value is NOT
# free -- the vocab relay drains whole-K blocks, so (K/PAYLOAD)=6 must divide
# VOCAB_I2*PAIR_ROWS=18, and it must match the model entry's UNI_LM=14. The 1B
# default (14) deadlocks the 3B vocab wave.
DECODE_ENV := DECODE_MODEL=llama-3.2-3b VOCAB_CHUNK_I2=9 UNIFIED=1 LM_HEAD=0 \
              NLAYERS=1 DECODE_GOLDEN=1

# ATTN_MAXL to build for. 2048 = production; a small value builds fast for a gate.
LBUILD ?= 2048
LREF   := $(shell echo $$(($(LBUILD) - 1)))

.PHONY: help compile compile-decode _compile_decode_build run profile chat verify verify-full ask gen clean

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-3B Q4NX Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Build fused-decode templates (~15 min, one-time; no weights)"
	@echo "  make compile-decode   Same build (templates land in this directory)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           Top-k token-set inclusion gate: NPU q4nx vs HF bf16 (2 prompts x 32 tokens, k=5)"
	@echo "  make verify-full      Same gate over the full 8-prompt set"
	@echo "  make gen              Paris gate: greedy decode of \"The capital of France is\""
	@echo "  make ask PROMPT=...   Single Q&A turn"
	@echo "  (no 'diagnosis': the fused decode runs all 28 layers in one dispatch,"
	@echo "   so per-layer intermediates are not observable from the host)"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens for run/profile/chat"
	@echo "  PROMPT=\"...\"          Input prompt text (run/profile)"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo "  MODEL_SOURCE=...      Q4NX weight source (default: $(MODEL_SOURCE))"
	@echo "  LBUILD=2048           ATTN_MAXL the decode templates are built for"

## Build the fused-decode kernels + ATTN_MAXL=2048 templates (~15 min; no weights).
## This example has no separate prefill engine, so `compile` == `compile-decode`.
compile: compile-decode
	@echo "Compilation passed."

## Build the fused decode at the 3B geometry: Peano kernels (LLAMA_3_2_3B) + two
## same-ATTN_MAXL templates (decode_L$(LBUILD) + decode_L$(LREF)) into THIS dir.
## ~15 min, skipped when both are already present. Weight-free; the llvm-link major
## must equal Peano's (the fused_decode preflight enforces it). The driver loads
## the templates from here at run time.
compile-decode:
	@if [ -f $(srcdir)/decode_L$(LBUILD).xclbin ] && [ -f $(srcdir)/decode_L$(LBUILD).insts.bin ] \
	    && [ -f $(srcdir)/decode_L$(LREF).xclbin ] && [ -f $(srcdir)/decode_L$(LREF).insts.bin ]; then \
	  echo "3B decode templates already built (decode_L$(LBUILD) + decode_L$(LREF)); skipping. Run 'make clean' to force a rebuild."; \
	else \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build; \
	fi

_compile_decode_build:
	@# Delegate the llvm-link/Peano version preflight to the engine that owns it:
	@# an llvm-link older than Peano links quietly and miscompiles the attn kernels
	@# into fluent garbage.
	@$(MAKE) --no-print-directory -C $(DEC) preflight-llvm-link \
	  PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)
	@echo ">>> 3B non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.o || exit 1; done
	@echo ">>> 3B attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.ll || exit 1; done
	@echo ">>> 3B template decode_L$(LBUILD) (+ L$(LREF) slope ref)"
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LBUILD) python3 $(DECODE_DRIVER) >/tmp/llama3b_decode_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ]; then mv -f decode.xclbin $(srcdir)/decode_L$(LBUILD).xclbin; mv -f decode.insts.bin $(srcdir)/decode_L$(LBUILD).insts.bin; \
	  else echo "decode_L$(LBUILD) FAILED (see /tmp/llama3b_decode_L$(LBUILD).log)"; tail -30 /tmp/llama3b_decode_L$(LBUILD).log; exit 1; fi
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LREF) python3 $(DECODE_DRIVER) >/tmp/llama3b_decode_L$(LREF).log 2>&1; \
	  if [ -f decode.insts.bin ]; then mv -f decode.insts.bin $(srcdir)/decode_L$(LREF).insts.bin; mv -f decode.xclbin $(srcdir)/decode_L$(LREF).xclbin; \
	  else echo "decode_L$(LREF) FAILED (see /tmp/llama3b_decode_L$(LREF).log)"; tail -30 /tmp/llama3b_decode_L$(LREF).log; exit 1; fi
	@echo "3B decode build complete: decode_L$(LBUILD) + decode_L$(LREF) templates."

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

# No `diagnosis` target: the fused decode runs all 28 layers + the LM head inside a
# single dispatch, so the per-layer intermediates that lens needs are not observable
# from the host. `make verify` (token-set vs HF bf16) is the PASS/FAIL gate.

## Single Q&A turn:  make ask PROMPT="What is the capital of France?"
ask:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(DRIVER) --run-only --prompt "$(PROMPT)" --model $(MODEL) --n-tokens $(N_TOKENS)

## Paris gate: greedy-decode "The capital of France is"; first token must be 12366.
gen:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(DRIVER) --run-only --model base --n-tokens 12 \
		--prompt "The capital of France is"

## Remove build artifacts. Deliberately KEEPS $(srcdir)/.decode_wcache: that is the
## ~2 GB q4k-cascade requant of the Q4NX weights and takes ~15 min to rebuild, while
## the lit tests run `clean` before every gate. Delete it by hand to force a re-pack.
clean:
	rm -f $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin 2>/dev/null || true
	rm -f $(addprefix $(DEC)/,$(addsuffix .o,$(NONATTN_KERNELS))) \
	      $(addprefix $(DEC)/,$(addsuffix .ll,$(ATTN_LL_KERNELS))) 2>/dev/null || true
	rm -rf $(BUILD_DIR) $(srcdir)/__pycache__ $(srcdir)/_q4nx_cache_seq* 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports 2>/dev/null || true
	@echo "Decode templates, build dirs, shared-dir kernels and verify/reports/ removed."
	@echo "Kept .decode_wcache (requant cache); remove it by hand to force a re-pack."
