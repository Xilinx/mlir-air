# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Gemma3-4B Q4NX end-to-end decode on NPU2 (MLIR-AIR), FLM-faithful.
#
# Same driver contract as the other llms/ examples (run/profile/ask drive the inference
# driver; compile/compile-decode are weight-free builds). Where it differs from them is
# `verify`: see the comment on that target.
#
# ONE dispatch = 34 decoder layers + tied LM head, reading and appending a shared per-layer
# KV cache. Drives the shared fused_decode engine (DECODE_MODEL=gemma3-4b): 4-norm sandwich,
# dual-theta RoPE + qk-norm, GELU-tanh GLU, DH=256 GQA2 attention.
#
# The KV seed + greedy first-token golden come from a numpy reference forward
# (gemma3_4b_q4nx_weights.forward_prompt); the fused per-token decode then generates.
#
# Python deps beyond the mlir-air base environment: see requirements.txt.
#
# Quick start:
#   make compile-decode LBUILD=16   Build the gemma decode templates (small, for the Paris gate)
#   make gen                        Run the Paris gate (first token 9079 " Paris")
#   make compile-decode             Build production templates (ATTN_MAXL=2048, ~15 min)

srcdir     := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DEC        := $(srcdir)/../../fused_decode
DECODE_DRIVER := $(DEC)/fused_decode.py
PREFILL_DRIVER := $(srcdir)/gemma3_4b_q4nx_prefill.py
AIEOPT_DIR := $(shell realpath $(dir $(shell which aie-opt))/..)

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Prefill context length (padded). The GEMM registry carries the Gemma shapes at
# M=2048; other lengths need matching registry entries.
CTX ?= 2048
export Q4NX_SEQ_LEN   := $(CTX)
export Q4NX_CACHE_DIR := $(srcdir)/$(BUILD_DIR)/q4nx_kernel_cache_$(CTX)

# Per-kernel -O is LOAD-BEARING (see fused_decode/Makefile): proj_qmm/rms_residual/glu/rope
# -O2; attn_qk/attn_kv inline .ll -O1. Gemma differs only by -DMODEL_TYPE=GEMMA3_4B (the
# header auto-sets A_FUNC=A_GELU, HAS_QK_NORM, ATTN_IMPL, dims/vocab).
NONATTN_KERNELS := proj_qmm rms_residual glu rope
ATTN_LL_KERNELS := attn_qk attn_kv
PEANO_KBASE := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=GEMMA3_4B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(DEC)/kernels -I $(DEC)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

# Gemma decode env (module-level constants read these): 34 layers, vocab-262208 chunking.
DECODE_ENV := DECODE_MODEL=gemma3-4b VOCAB_CHUNK_I2=5 UNIFIED=1 LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1

# ATTN_MAXL to build for. 16 = the Paris gate (P + a few tokens <= 16); 2048 = production.
LBUILD ?= 2048
LREF   := $(shell echo $$(($(LBUILD) - 1)))

# Shared driver contract (same variable names as the other llms/ examples).
# NTOK is kept as a back-compat alias for N_TOKENS.
NTOK     ?=
N_TOKENS ?= $(if $(NTOK),$(NTOK),64)
# Prompt for `make run` / `make ask`. Empty = the canonical Paris prompt, which is
# the only prompt the *** PARIS *** gate is defined for.
PROMPT   ?=
PROMPT_ARG := $(if $(PROMPT),--prompt "$(PROMPT)",)

# Weight source: HF repo id (model.q4nx bundle) or a local dir/file. Read by the
# driver as Q4NX_MODEL_SOURCE.
MODEL_SOURCE ?= FastFlowLM/Gemma3-4B-NPU2
export Q4NX_MODEL_SOURCE := $(MODEL_SOURCE)

# Dual-MM2S weight feed for the fused decode (fused_decode/fused_decode.py).
# ON by default. The BUILD and the RUN must agree -- the flag selects both the
# shim channel split and the DDR weight cascade order -- so export it here and let
# it reach the delegated decode build AND the inference driver. The requant cache
# is keyed on it, so flipping it needs no manual cache purge.
W_DUAL_CHAN ?= 1
export W_DUAL_CHAN


.PHONY: help compile compile-prefill compile-decode _compile_decode_build prefill \
        run ask verify verify-paris profile profile-prefill gen clean

help:
	@echo "============================================================"
	@echo " GEMMA3-4B Q4NX decode on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "  make compile LBUILD=16   Prefill ELFs + decode templates, small (Paris gate)"
	@echo "  make compile             Production build (ATTN_MAXL=2048, ~15 min)"
	@echo "  make compile-prefill     Prefill ELFs only (weight-free)"
	@echo "  make compile-decode      Decode templates only (weight-free)"
	@echo "  make run                 Decode 'The capital of France is' -> Paris gate"
	@echo "  make ask PROMPT=\"...\"    Single Q&A turn"
	@echo "  make verify              Same Paris argmax gate (9079) as the CI check"
	@echo "  make verify-paris        Fast prefill-only weight-integrity gate"
	@echo "  make profile             TTFT + decode throughput over N_TOKENS tokens"
	@echo "  make profile-prefill     Warm prefill TTFT at CTX"
	@echo "  make clean               Remove build artifacts, caches, gemma kernels"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  LBUILD=2048              ATTN_MAXL to build the templates for"
	@echo "  N_TOKENS=64              Tokens generated by 'make profile' / 'make ask'"
	@echo "  PROMPT=\"...\"             Prompt for 'make run' / 'make ask'"
	@echo "  CTX=2048                 Prefill context length (padded)"
	@echo "  MODEL_SOURCE=<repo|dir>  model.q4nx source (default FastFlowLM/Gemma3-4B-NPU2)"

## Build the gemma decode: kernels (GEMMA3_4B) + two same-ATTN_MAXL templates
## (decode_L$(LBUILD) + decode_L$(LREF)) into this dir. Weight-free build.
##
## Skips when both templates are already present, matching fused_decode's
## compile-decode. The guard is keyed on the LBUILD-encoded filenames, so a build
## at a different LBUILD is never silently reused.
# W_DUAL_CHAN changes the DDR weight layout, so a warm template built with the
# other setting must NOT be reused -- it would be fed the wrong blocks and produce
# silent garbage. The stamp makes the skip below flag-aware.
DECODE_FLAGS_STAMP := $(srcdir)/.decode_flags
DECODE_FLAGS       := W_DUAL_CHAN=$(W_DUAL_CHAN)

compile-decode:
	@if [ -f $(srcdir)/decode_L$(LBUILD).xclbin ] && [ -f $(srcdir)/decode_L$(LBUILD).insts.bin ] \
	    && [ -f $(srcdir)/decode_L$(LREF).xclbin ] && [ -f $(srcdir)/decode_L$(LREF).insts.bin ] \
	    && [ "`cat $(DECODE_FLAGS_STAMP) 2>/dev/null`" = "$(DECODE_FLAGS)" ]; then \
	  echo "Gemma decode templates already built (decode_L$(LBUILD) + decode_L$(LREF), $(DECODE_FLAGS)); skipping. Run 'make clean' to force a rebuild."; \
	else \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build \
	    && echo "$(DECODE_FLAGS)" > $(DECODE_FLAGS_STAMP); \
	fi

_compile_decode_build:
	@# Delegate the llvm-link version preflight to the engine that owns it rather
	@# than keeping a copy here: the rule (LLVM <23) must stay in one place, in
	@# sync with lit.cfg.py's llvm_link_pre23 feature that gates this example's
	@# lit tests (fused_decode/Makefile: preflight-llvm-link).
	@$(MAKE) --no-print-directory -C $(DEC) preflight-llvm-link \
	  PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)
	@echo ">>> gemma non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.o || exit 1; done
	@echo ">>> gemma attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.ll || exit 1; done
	@echo ">>> gemma template decode_L$(LBUILD) (+ L$(LREF) slope ref)"
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LBUILD) python3 $(DECODE_DRIVER) >/tmp/gemma_decode_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ]; then mv -f decode.xclbin $(srcdir)/decode_L$(LBUILD).xclbin; mv -f decode.insts.bin $(srcdir)/decode_L$(LBUILD).insts.bin; \
	  else echo "decode_L$(LBUILD) FAILED (see /tmp/gemma_decode_L$(LBUILD).log)"; tail -30 /tmp/gemma_decode_L$(LBUILD).log; exit 1; fi
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LREF) python3 $(DECODE_DRIVER) >/tmp/gemma_decode_L$(LREF).log 2>&1; \
	  if [ -f decode.insts.bin ]; then mv -f decode.insts.bin $(srcdir)/decode_L$(LREF).insts.bin; mv -f decode.xclbin $(srcdir)/decode_L$(LREF).xclbin; \
	  else echo "decode_L$(LREF) FAILED (see /tmp/gemma_decode_L$(LREF).log)"; tail -30 /tmp/gemma_decode_L$(LREF).log; exit 1; fi
	@echo "Gemma decode build complete: decode_L$(LBUILD) + decode_L$(LREF) templates."

## Build/cache the 8 prefill ELFs (rms_qkv_qknorm_rope + 2 flash-attention
## variants + o_norm_res_norm + gate + up + gelu_mul + down_norm_add +
## lm_head_gemv). Weight-free build, no NPU dispatch.
compile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(PREFILL_DRIVER) --compile-only

## Prefill-only Paris gate: PASS iff the first token argmax is 9079 (" Paris").
prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(PREFILL_DRIVER)

## Build everything the end-to-end chatbot needs: the prefill ELFs and the
## decode templates.
compile: compile-prefill compile-decode

## Run inference (NPU prefill + fused NPU decode). With no PROMPT this is the
## canonical Paris gate: PASS iff the first token is 9079 (" Paris").
run:
	@cd $(srcdir) && python3 gemma3_4b_q4nx_inference.py --n-tokens 9 $(PROMPT_ARG)

## Single Q&A turn:  make ask PROMPT="What is the capital of France?"
ask:
	@cd $(srcdir) && python3 gemma3_4b_q4nx_inference.py \
		--n-tokens $(N_TOKENS) $(PROMPT_ARG)

## Correctness gate: the greedy first-token argmax (" Paris", 9079), matching
## FastFlowLM's own Gemma3 NPU output.
##
## Unlike the other llms/ examples this example has no verify_adapter.py and does
## not drive the shared verify/ subsystem (top-k token-set inclusion vs an HF bf16
## reference). That subsystem needs an HF bf16 checkpoint of the same weights; for
## Gemma3-4B that is google/gemma-3-4b-pt, a gated multimodal checkpoint the shared
## HfRunner (AutoModelForCausalLM) does not load as a text-only model. So the gate
## here is greedy-token parity against the reference implementation instead.
## `run` and `verify` are the same gate; `verify` is the name CI expects.
verify: run

## Fast weight-integrity smoke: prefill only, PASS iff the first-token argmax is
## 9079 (" Paris"). Cheap -- no decode templates needed.
verify-paris: prefill

## Warm prefill TTFT at CTX (warmup pass, then a timed one). Isolates the
## prefill dispatch from the one-time weight load the e2e path also pays.
profile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && Q4NX_BENCH_L=$(CTX) python3 $(PREFILL_DRIVER)

## End-to-end: TTFT + decode throughput over N_TOKENS tokens (prints the TTFT and
## Tokens/second lines the nightly LLM benchmark extracts). Needs production
## templates: make compile.
profile:
	@cd $(srcdir) && python3 gemma3_4b_q4nx_inference.py --n-tokens $(N_TOKENS)

## Back-compat alias for the Paris gate.
gen: run

## Remove build artifacts, kernel/weight caches, and the decode templates. Also
## removes the GEMMA3_4B-compiled kernels this Makefile writes into the shared
## fused_decode/ dir -- leaving them there makes a later Llama build silently
## link Gemma objects.
clean:
	rm -f $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin 2>/dev/null || true
	rm -f $(DECODE_FLAGS_STAMP) 2>/dev/null || true
	rm -f $(addprefix $(DEC)/,$(addsuffix .o,$(NONATTN_KERNELS))) \
	      $(addprefix $(DEC)/,$(addsuffix .ll,$(ATTN_LL_KERNELS))) 2>/dev/null || true
	rm -rf $(srcdir)/build_peano $(srcdir)/build_chess $(srcdir)/__pycache__ \
	       $(srcdir)/_q4nx_* $(srcdir)/.wcache 2>/dev/null || true
	@echo "Gemma decode templates, build dirs and shared-dir kernels removed."
