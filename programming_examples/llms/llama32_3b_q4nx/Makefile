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
# The prompt runs as one batched NPU prefill whose per-layer roped-K / raw-V seed
# the decode's device KV cache (warm start); generation then costs one fused
# dispatch per token.
#
# Quick start:
#   make compile        — Compile the prefill ELFs (~4 min; no weights needed)
#   make compile-decode — Build the fused decode templates (~15 min; no weights)
#   make run            — Run inference: 28 layers + LM head on-device, per token
#   make verify         — Top-k token-set inclusion gate: NPU q4nx vs HF bf16
#
# Config: 28 layers, emb=3072, 24 heads / 8 KV heads, head_dim=128,
#         hidden=8192, vocab=128256 (NPU2, AMD Strix).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

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

# Dual-MM2S weight feed for the fused decode (fused_decode/fused_decode.py).
# ON by default. The BUILD and the RUN must agree -- the flag selects both the
# shim channel split and the DDR weight cascade order -- so export it here and let
# it reach the delegated decode build AND the inference driver. The requant cache
# is keyed on it, so flipping it needs no manual cache purge.
W_DUAL_CHAN ?= 1
export W_DUAL_CHAN


DRIVER := $(srcdir)/llama32_3b_q4nx_inference.py
# The fused per-token decode engine is a SEPARATE reusable example (one dispatch =
# 28 layers + LM head); this Makefile drives it with the 3B model selection and
# builds the templates INTO THIS DIRECTORY, the same way gemma3_4b_q4nx does. Every
# model emits the same decode_L<N>.* names, so per-example output is what keeps a
# 1B build from silently satisfying a 3B run.
DEC           := $(srcdir)/../../fused_decode
DECODE_DRIVER := $(DEC)/fused_decode.py
AIEOPT_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))
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


# Prefill engine (weight-integrity Paris gate + the CTX prefill latency sweep).
PREFILL := $(srcdir)/llama32_3b_q4nx_prefill.py
# Padded prefill length for profile-prefill.
CTX ?= 2048
# CTX sets the length the prefill ELFs are BUILT for, not just the length the
# bench times. Without this export the driver defaults its seq_len to 2048 and
# a CTX=512 run pads 512 real tokens into a 2048-row prefill -- a number that
# parses fine and is ~4x too slow. bench/sweep_prefill.py guards against it too.
export Q4NX_SEQ_LEN := $(CTX)

.PHONY: help compile compile-decode compile-decode-dynseq compile-decode-windows _compile_decode_build run profile chat verify \
        verify-full ask gen verify-paris diagnosis profile-prefill clean

help:
	@echo "============================================================"
	@echo " LLAMA-3.2-3B Q4NX Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Compile prefill ELFs (~4 min, one-time; no weights)"
	@echo "  make compile-decode   Decode templates, built into this dir (~15 min, no weights)"
	@echo "  make compile-decode-windows  Staircase: one template pair per ATTN_MAXL ($(WINDOWS)); ~1.11x decode"
	@echo "  make run              Run inference ($(N_TOKENS) tokens)"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "More targets:"
	@echo "  make verify           Top-k token-set inclusion gate: NPU q4nx vs HF bf16 (2 prompts x 32 tokens, k=5)"
	@echo "  make verify-full      Same gate over the full 8-prompt set"
	@echo "  make diagnosis        Single-prompt lens through the shared runner (informational)"
	@echo "  make gen              Paris gate: greedy decode of \"The capital of France is\""
	@echo "  make verify-paris     Prefill-only weight-integrity smoke (first token == Paris)"
	@echo "  make profile-prefill  Warm prefill TTFT benchmark at CTX=$(CTX)"
	@echo "  make ask PROMPT=...   Single Q&A turn"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens for run/profile/chat"
	@echo "  PROMPT=\"...\"          Input prompt text (run/profile)"
	@echo "  MODEL=base|instruct   Model variant (default: instruct)"
	@echo "  MODEL_SOURCE=...      Q4NX weight source (default: $(MODEL_SOURCE))"
	@echo "  LBUILD=2048           ATTN_MAXL the decode templates are built for"

## Compile the prefill ELFs (~4 min, cached; no weights needed). The fused decode
## templates are a separate ~15 min build -- see `make compile-decode`.
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) --compile-only

## Build the fused decode at the 3B geometry: Peano kernels (LLAMA_3_2_3B) + two
## same-ATTN_MAXL templates (decode_L$(LBUILD) + decode_L$(LREF)) into THIS dir.
## ~15 min, skipped when both are already present. Weight-free; needs an llvm-link
## from LLVM <23 (the fused_decode preflight enforces it). The driver loads
## the templates from here at run time.
# W_DUAL_CHAN changes the DDR weight layout, so a warm template built with the
# other setting must NOT be reused -- it would be fed the wrong blocks and produce
# silent garbage. The stamp makes the skip below flag-aware.
DECODE_FLAGS_STAMP := $(srcdir)/.decode_flags
DECODE_FLAGS       := W_DUAL_CHAN=$(W_DUAL_CHAN)

compile-decode:
	@if [ -f $(srcdir)/decode_L$(LBUILD).xclbin ] && [ -f $(srcdir)/decode_L$(LBUILD).insts.bin ] \
	    && [ -f $(srcdir)/decode_L$(LREF).xclbin ] && [ -f $(srcdir)/decode_L$(LREF).insts.bin ] \
	    && [ "`cat $(DECODE_FLAGS_STAMP) 2>/dev/null`" = "$(DECODE_FLAGS)" ]; then \
	  echo "3B decode templates already built (decode_L$(LBUILD) + decode_L$(LREF), $(DECODE_FLAGS)); skipping. Run 'make clean' to force a rebuild."; \
	else \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build \
	    && echo "$(DECODE_FLAGS)" > $(DECODE_FLAGS_STAMP); \
	fi

## Staircase templates: one pair per ATTN_MAXL window. The compiled KV readback streams
## ATTN_MAXL positions whatever the real context length is, so running each token on the
## smallest covering window moves proportionally less DDR per token (measured 1.11x over
## 300 tokens, token stream bit-identical). Run the driver with --staircase to use them.
## Writes $(DECODE_WINDOWS_STAMP) so the driver can reject a directory whose templates do
## not match what was built -- template discovery is a directory scan, and a stray pair
## silently changes which window is selected.
WINDOWS ?= 64 128 256 512 1024 2048
DECODE_WINDOWS_STAMP := $(srcdir)/.decode_windows

compile-decode-windows:
	@for W in $(WINDOWS); do \
	  echo ">>> staircase window ATTN_MAXL=$$W"; \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build LBUILD=$$W \
	    || exit 1; \
	done
	@echo "$(WINDOWS)" > $(DECODE_WINDOWS_STAMP)
	@echo "$(DECODE_FLAGS)" > $(DECODE_FLAGS_STAMP)
	@echo "Staircase templates built (ATTN_MAXL: $(WINDOWS)). Run with --staircase."

## Dynamic-runtime-sequence template: ONE build for every context length (see
## fused_decode/Makefile compile-decode-dynseq). Run with DECODE_DYNSEQ=1.
compile-decode-dynseq:
	@$(MAKE) --no-print-directory -C $(DEC) preflight-llvm-link \
	  PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)
	@echo ">>> 3B non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.o || exit 1; done
	@echo ">>> 3B attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.ll || exit 1; done
	@echo ">>> 3B dynseq template decode_dyn_L$(LBUILD)"
	@cd $(DEC) && $(DECODE_ENV) DECODE_DYNSEQ=1 DECODE_GOLDEN_L=$(LBUILD) $(PYTHON) $(DECODE_DRIVER) >/tmp/llama3b_decode_dyn_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ] && [ -f air_project/npu.air.txn.h ]; then \
	    mv -f decode.xclbin $(srcdir)/decode_dyn_L$(LBUILD).xclbin; \
	    cp -f air_project/npu.air.txn.h $(srcdir)/decode_dyn_L$(LBUILD).txn.h; \
	    rm -f npu_seq_*.mlir; \
	  else echo "decode_dyn_L$(LBUILD) FAILED (see /tmp/llama3b_decode_dyn_L$(LBUILD).log)"; tail -30 /tmp/llama3b_decode_dyn_L$(LBUILD).log; exit 1; fi
	@echo "3B dynseq build complete: decode_dyn_L$(LBUILD). Run with DECODE_DYNSEQ=1."

_compile_decode_build:
	@# Delegate the llvm-link version preflight to the engine that owns it:
	@# a >=23 llvm-link rewrites llvm.lifetime.* and Peano opt rejects the merge.
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
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LBUILD) $(PYTHON) $(DECODE_DRIVER) >/tmp/llama3b_decode_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ]; then mv -f decode.xclbin $(srcdir)/decode_L$(LBUILD).xclbin; mv -f decode.insts.bin $(srcdir)/decode_L$(LBUILD).insts.bin; \
	  else echo "decode_L$(LBUILD) FAILED (see /tmp/llama3b_decode_L$(LBUILD).log)"; tail -30 /tmp/llama3b_decode_L$(LBUILD).log; exit 1; fi
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LREF) $(PYTHON) $(DECODE_DRIVER) >/tmp/llama3b_decode_L$(LREF).log 2>&1; \
	  if [ -f decode.insts.bin ]; then mv -f decode.insts.bin $(srcdir)/decode_L$(LREF).insts.bin; mv -f decode.xclbin $(srcdir)/decode_L$(LREF).xclbin; \
	  else echo "decode_L$(LREF) FAILED (see /tmp/llama3b_decode_L$(LREF).log)"; tail -30 /tmp/llama3b_decode_L$(LREF).log; exit 1; fi
	@echo "3B decode build complete: decode_L$(LBUILD) + decode_L$(LREF) templates."

## Run unified inference
run:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with detailed profiling breakdown. --no-eos-stop so throughput is averaged
## over the full N_TOKENS budget: the default prompt answers in ~7 tokens, and a
## 7-sample average is dominated by the first (cold) dispatch. Mirrors
## llama32_1b_q4nx, whose profile target has carried this since it was written.
profile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) \
		--run-only --n-tokens $(N_TOKENS) --profile --no-eos-stop \
		--prompt "$(PROMPT)" --model $(MODEL)

## Interactive chat REPL
chat:
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) \
		--run-only --interactive --n-tokens $(N_TOKENS) --model $(MODEL)

# The shared verify framework lives under programming_examples/llms/verify/.
VERIFY_RUNNER = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = llama32_3b_q4nx.verify_adapter

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

## Diagnosis lens (single prompt through the shared runner, informational).
## The fused decode runs all 28 layers + the LM head inside one dispatch, so the
## per-layer intermediates that lens wants are not observable from the host; this
## reports the token-level view instead. `make verify` is the PASS/FAIL gate.
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

## Paris gate: greedy-decode "The capital of France is"; first token must be 12366.
gen:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) --run-only --model base --n-tokens 12 \
		--prompt "The capital of France is"

## Remove build artifacts. Deliberately KEEPS $(srcdir)/.decode_wcache: that is the
## ~2 GB q4k-cascade requant of the Q4NX weights and takes ~15 min to rebuild, while
## the lit tests run `clean` before every gate. Delete it by hand to force a re-pack.
clean:
	rm -f $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin 2>/dev/null || true
	rm -f $(DECODE_FLAGS_STAMP) $(DECODE_WINDOWS_STAMP) 2>/dev/null || true
	rm -f $(addprefix $(DEC)/,$(addsuffix .o,$(NONATTN_KERNELS))) \
	      $(addprefix $(DEC)/,$(addsuffix .ll,$(ATTN_LL_KERNELS))) 2>/dev/null || true
	rm -rf $(BUILD_DIR) $(srcdir)/__pycache__ $(srcdir)/_q4nx_cache_seq* 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports 2>/dev/null || true
	@echo "Decode templates, build dirs, shared-dir kernels and verify/reports/ removed."
	@echo "Kept .decode_wcache (requant cache); remove it by hand to force a re-pack."
