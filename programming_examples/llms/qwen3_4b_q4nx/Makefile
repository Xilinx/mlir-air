# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-4B Q4NX end-to-end decode on NPU2 (MLIR-AIR), FLM-faithful.
#
# Qwen3-4B is the DFlash TARGET model (docs/DFlashFeasibility.md). This is
# qwen3_8b_q4nx/Makefile at Qwen3-4B's dimensions: same driver contract as the
# other llms/ examples (run/profile/ask drive the inference driver;
# compile/compile-decode are weight-free builds).
#
# ONE dispatch = 36 decoder layers + the TIED LM head, reading and appending a shared
# per-layer KV cache. Drives the shared fused_decode engine (DECODE_MODEL=qwen3-4b):
# 2-norm pre-norm, single-theta RoPE + qk-norm, SwiGLU, DH=128 GQA4 attention.
#
# Unlike qwen3_8b_q4nx, the total decode weight is ~2.1 GiB (36 layers at
# K=2560, vs 8B's K=4096 / 4.04 GiB) -- under the 4 GiB one-BO shim-BD-offset
# limit, so there is no DECODE_WGROUP weight split here by default.
#
# The KV seed + greedy first-token golden come from a numpy reference forward
# (qwen3_8b_q4nx_weights.forward_prompt, dimension-generic and reused via
# qwen3_4b_q4nx_weights.forward_prompt); the fused per-token decode then generates.
#
# Python deps beyond the mlir-air base environment: see requirements.txt.
#
# Quick start:
#   make compile-decode LBUILD=16   Build the Qwen3-4B decode templates (small, for the Paris gate)
#   make gen                        Run the Paris gate (first token 12095 " Paris")
#   make compile-decode             Build production templates (ATTN_MAXL=2048, ~15 min)

srcdir     := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DEC        := $(srcdir)/../../fused_decode
DECODE_DRIVER := $(DEC)/fused_decode.py
PREFILL_DRIVER := $(srcdir)/qwen3_4b_q4nx_prefill.py
AIEOPT_DIR := $(shell realpath $(dir $(shell which aie-opt))/..)

ifdef PEANO_INSTALL_DIR
  BUILD_DIR := build_peano
else
  BUILD_DIR := build_chess
endif

# Prefill context length (padded). The GEMM registry carries the Qwen3-4B shapes at
# M=2048; other lengths need matching registry entries.
CTX ?= 2048
export Q4NX_SEQ_LEN   := $(CTX)
export Q4NX_CACHE_DIR := $(srcdir)/$(BUILD_DIR)/q4nx_kernel_cache_$(CTX)

# Per-kernel -O is LOAD-BEARING (see fused_decode/Makefile): proj_qmm/rms_residual/glu/rope
# -O2; attn_qk/attn_kv inline .ll -O1. Qwen3-4B differs only by -DMODEL_TYPE=QWEN3_4B (the
# header auto-sets A_FUNC=A_SILU, HAS_QK_NORM, ATTN_IMPL, dims/vocab).
NONATTN_KERNELS := proj_qmm rms_residual glu rope
ATTN_LL_KERNELS := attn_qk attn_kv
PEANO_KBASE := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=QWEN3_4B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(DEC)/kernels -I $(DEC)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

# Qwen3-4B decode env (module-level constants read these): 36 layers, vocab-151936
# chunking. VOCAB_CHUNK_I2=30 is this model's own value (build_template.sh) -- the
# vocab-chunk divisibility constraints depend on NCX/NCY/PAIR_ROWS, which do not
# transfer from qwen3_8b_q4nx's 8.
#
# DECODE_WGROUP: decode layers per weight BO, only needed past the 4 GiB one-BO
# limit (qwen3-4b's 2.1 GiB total does not need it -- see the module docstring in
# qwen3_4b_q4nx_inference.py). 0 = disabled. If nonzero, must match
# Q4NX_DECODE_WGROUP in qwen3_4b_q4nx_inference.py -- the host slices the weights
# the same way.
DECODE_WGROUP ?= 0
DECODE_ENV := DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 UNIFIED=1 LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1 \
              $(if $(filter-out 0,$(DECODE_WGROUP)),DECODE_WGROUP=$(DECODE_WGROUP),)

# GLU_SLICE is derived from this model's own egress-round parity, not looked up,
# so a header copied from another model can disagree with the IR -- silently, at
# runtime. Hand the builder's value to the kernel compile; glu.cc asserts on it.
# Empty (and the assert inert) if the query fails, so this cannot break a build
# that works today.
GLU_SLICE_EXPECTED := $(shell $(DECODE_ENV) FUSED_DECODE_PRINT_CONST=GLU_SLICE python3 $(DECODE_DRIVER) 2>/dev/null)
PEANO_KBASE += $(if $(GLU_SLICE_EXPECTED),-DGLU_SLICE_EXPECTED=$(GLU_SLICE_EXPECTED))

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
# `profile` publishes context_len to the nightly dashboard, so it measures a real
# prompt rather than the 5-token Paris gate one. PROMPT overrides it.
PROFILE_PROMPT := $(if $(PROMPT),$(PROMPT),What is the capital of France?)

# Weight source: HF repo id (model.q4nx bundle) or a local dir/file. Read by the
# driver as Q4NX_MODEL_SOURCE.
MODEL_SOURCE ?= FastFlowLM/Qwen3-4B-NPU2
export Q4NX_MODEL_SOURCE := $(MODEL_SOURCE)

# Dual-MM2S weight feed for the fused decode (fused_decode/fused_decode.py).
# ON by default. The BUILD and the RUN must agree -- the flag selects both the
# shim channel split and the DDR weight cascade order -- so export it here and let
# it reach the delegated decode build AND the inference driver. The requant cache
# is keyed on it, so flipping it needs no manual cache purge.
W_DUAL_CHAN ?= 1
export W_DUAL_CHAN


# ---- shared verify subsystem (programming_examples/llms/verify/) ----
VERIFY_RUNNER  := $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER := qwen3_4b_q4nx.verify_adapter
MODEL_CHOICE   ?= instruct
MAX_PROMPTS    ?= 2

.PHONY: help compile compile-prefill compile-decode compile-decode-dynseq compile-decode-windows _compile_decode_build prefill \
        run ask chat verify verify-full verify-paris diagnosis profile profile-prefill gen clean

help:
	@echo "============================================================"
	@echo " QWEN3-4B Q4NX decode on NPU2 (MLIR-AIR) -- the DFlash target model"
	@echo "============================================================"
	@echo ""
	@echo "  make compile LBUILD=16   Prefill ELFs + decode templates, small (Paris gate)"
	@echo "  make compile             Production build (ATTN_MAXL=2048, ~15 min)"
	@echo "  make compile-prefill     Prefill ELFs only (weight-free)"
	@echo "  make compile-decode      Decode templates only (weight-free)"
	@echo "  make run                 Decode 'The capital of France is' -> Paris gate"
	@echo "  make ask PROMPT=\"...\"    Single Q&A turn"
	@echo "  make chat                Interactive chat REPL (streaming output)"
	@echo "  make verify              Top-k token-set gate vs HF bf16 (2 prompts x 32 tokens, k=5)"
	@echo "  make verify-full         Same over every prompt in the shared prompt file"
	@echo "  make verify-paris        Fast prefill-only weight-integrity gate"
	@echo "  make diagnosis           Single-prompt lens through the shared runner (informational)"
	@echo "  make profile             TTFT + decode throughput over N_TOKENS tokens"
	@echo "  make profile-prefill     Warm prefill TTFT at CTX"
	@echo "  make clean               Remove build artifacts, caches, Qwen3-4B kernels"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  LBUILD=2048              ATTN_MAXL to build the templates for"
	@echo "  N_TOKENS=64              Tokens generated by 'make profile' / 'make ask'"
	@echo "  PROMPT=\"...\"             Prompt for 'make run' / 'make ask'"
	@echo "  CTX=2048                 Prefill context length (padded)"
	@echo "  MODEL_SOURCE=<repo|dir>  model.q4nx source (default FastFlowLM/Qwen3-4B-NPU2)"
	@echo "  DECODE_WGROUP=$(DECODE_WGROUP)               Decode layers per weight BO (0 = disabled; not needed at 4B's size)"

## Build the Qwen3-4B decode: kernels (QWEN3_4B) + two same-ATTN_MAXL templates
## (decode_L$(LBUILD) + decode_L$(LREF)) into this dir. Weight-free build.
##
## Skips when both templates are already present, matching fused_decode's
## compile-decode. The guard is keyed on the LBUILD-encoded filenames, so a build
## at a different LBUILD is never silently reused.
# W_DUAL_CHAN changes the DDR weight layout, so a warm template built with the
# other setting must NOT be reused -- it would be fed the wrong blocks and produce
# silent garbage. The stamp makes the skip below flag-aware.
DECODE_FLAGS_STAMP := $(srcdir)/.decode_flags
DECODE_FLAGS       := W_DUAL_CHAN=$(W_DUAL_CHAN) DECODE_WGROUP=$(DECODE_WGROUP)

compile-decode:
	@if [ -f $(srcdir)/decode_L$(LBUILD).xclbin ] && [ -f $(srcdir)/decode_L$(LBUILD).insts.bin ] \
	    && [ -f $(srcdir)/decode_L$(LREF).xclbin ] && [ -f $(srcdir)/decode_L$(LREF).insts.bin ] \
	    && [ "`cat $(DECODE_FLAGS_STAMP) 2>/dev/null`" = "$(DECODE_FLAGS)" ]; then \
	  echo "Qwen3-4B decode templates already built (decode_L$(LBUILD) + decode_L$(LREF), $(DECODE_FLAGS)); skipping. Run 'make clean' to force a rebuild."; \
	else \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build \
	    && echo "$(DECODE_FLAGS)" > $(DECODE_FLAGS_STAMP); \
	fi

## Staircase templates: one pair per ATTN_MAXL window; each token dispatches on the
## smallest one covering the context. Writes $(DECODE_WINDOWS_STAMP) so a stray pair is
## rejected at load rather than silently changing which window is selected.
WINDOWS ?= 64 128 256 512 1024 2048
DECODE_WINDOWS_STAMP := $(srcdir)/.decode_windows

compile-decode-windows:
	@for W in $(WINDOWS); do \
	  echo ">>> staircase window ATTN_MAXL=$$W"; \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build LBUILD=$$W \
	    || exit 1; \
	done
	@echo "$(WINDOWS)" > $(DECODE_WINDOWS_STAMP)
	@echo "Staircase templates built (ATTN_MAXL: $(WINDOWS)). Run with DECODE_STAIRCASE=1."

## Dynamic-runtime-sequence template: ONE build for every context length. The
## context length becomes a runtime scalar, so a single xclbin plus the emitted
## TXN builder serves any L and the readback moves only this token's context.
## Run with DECODE_DYNSEQ=1. No pair: there is no slope to calibrate.
compile-decode-dynseq:
	@$(MAKE) --no-print-directory -C $(DEC) preflight-llvm-link \
	  PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)
	@echo ">>> qwen3-4b non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.o || exit 1; done
	@echo ">>> qwen3-4b attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.ll || exit 1; done
	@echo ">>> qwen3-4b dynseq template decode_dyn_L$(LBUILD)"
	@cd $(DEC) && $(DECODE_ENV) DECODE_DYNSEQ=1 DECODE_GOLDEN_L=$(LBUILD) python3 $(DECODE_DRIVER) >/tmp/q4b_decode_dyn_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ] && [ -f air_project/npu.air.txn.h ]; then \
	    mv -f decode.xclbin $(srcdir)/decode_dyn_L$(LBUILD).xclbin; \
	    cp -f air_project/npu.air.txn.h $(srcdir)/decode_dyn_L$(LBUILD).txn.h; \
	    rm -f npu_seq_*.mlir; \
	  else echo "decode_dyn_L$(LBUILD) FAILED (see /tmp/q4b_decode_dyn_L$(LBUILD).log)"; tail -30 /tmp/q4b_decode_dyn_L$(LBUILD).log; exit 1; fi
	@echo "Qwen3-4B dynseq build complete: decode_dyn_L$(LBUILD). Run with DECODE_DYNSEQ=1."

_compile_decode_build:
	@# Delegate the llvm-link version preflight to the engine that owns it rather
	@# than keeping a copy here: the rule (LLVM <23) must stay in one place, in
	@# sync with lit.cfg.py's llvm_link_pre23 feature that gates this example's
	@# lit tests (fused_decode/Makefile: preflight-llvm-link).
	@$(MAKE) --no-print-directory -C $(DEC) preflight-llvm-link \
	  PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)
	@echo ">>> qwen3-4b non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.o || exit 1; done
	@echo ">>> qwen3-4b attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.ll || exit 1; done
	@echo ">>> qwen3-4b template decode_L$(LBUILD) (+ L$(LREF) slope ref)"
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LBUILD) python3 $(DECODE_DRIVER) >/tmp/q4b_decode_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ]; then mv -f decode.xclbin $(srcdir)/decode_L$(LBUILD).xclbin; mv -f decode.insts.bin $(srcdir)/decode_L$(LBUILD).insts.bin; \
	  else echo "decode_L$(LBUILD) FAILED (see /tmp/q4b_decode_L$(LBUILD).log)"; tail -30 /tmp/q4b_decode_L$(LBUILD).log; exit 1; fi
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LREF) python3 $(DECODE_DRIVER) >/tmp/q4b_decode_L$(LREF).log 2>&1; \
	  if [ -f decode.insts.bin ]; then mv -f decode.insts.bin $(srcdir)/decode_L$(LREF).insts.bin; mv -f decode.xclbin $(srcdir)/decode_L$(LREF).xclbin; \
	  else echo "decode_L$(LREF) FAILED (see /tmp/q4b_decode_L$(LREF).log)"; tail -30 /tmp/q4b_decode_L$(LREF).log; exit 1; fi
	@echo "Qwen3-4B decode build complete: decode_L$(LBUILD) + decode_L$(LREF) templates."

## Build/cache the 8 prefill ELFs (rms_qkv_qknorm_rope + flash_attn + o_res_norm +
## gate + up + swiglu + down_add + lm_head_gemv). Weight-free build, no NPU dispatch.
compile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(PREFILL_DRIVER) --compile-only

## Prefill-only Paris gate: PASS iff the first token argmax is 12095 (" Paris").
prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && python3 $(PREFILL_DRIVER)

## Build everything the end-to-end chatbot needs: the prefill ELFs and the
## decode templates.
compile: compile-prefill compile-decode

## Run inference (NPU prefill + fused NPU decode). With no PROMPT this is the
## canonical Paris gate: PASS iff the first token is 12095 (" Paris").
run:
	@cd $(srcdir) && python3 qwen3_4b_q4nx_inference.py --n-tokens 9 $(PROMPT_ARG)

## Single Q&A turn:  make ask PROMPT="What is the capital of France?"
ask:
	@cd $(srcdir) && python3 qwen3_4b_q4nx_inference.py \
		--n-tokens $(N_TOKENS) $(PROMPT_ARG)

## Interactive chat REPL (streaming). The prefill engine and decode templates are
## built once and held across turns, so only the first turn pays the model load.
chat:
	@cd $(srcdir) && python3 qwen3_4b_q4nx_inference.py \
		--interactive --n-tokens $(N_TOKENS)

## Top-k token-set inclusion gate (NPU q4nx vs HF bf16, 2 prompts x 32 tokens,
## k=5) -- the same gate, through the same shared verify/ subsystem, as the other
## quantized examples.
##
## REFERENCE. Qwen3 ships a single checkpoint (no separate base/instruct
## split), so both MODEL_CHOICE keys map to Qwen/Qwen3-4B. It is ungated and
## loads directly with AutoModelForCausalLM.
verify:
	cd $(srcdir)/.. && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL_CHOICE) --max-prompts $(MAX_PROMPTS)

## Full-sweep variant of `make verify` (all prompts in the prompt file).
verify-full:
	cd $(srcdir)/.. && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL_CHOICE)

## Fast weight-integrity smoke: prefill only, PASS iff the first-token argmax is
## 12095 (" Paris"). Cheap -- no decode templates, no HF reference needed.
verify-paris: prefill

## Diagnosis lens (single prompt through the shared runner, informational).
## The Q4NX prefill does not expose per-layer ffn_out and the decode runs all 36
## layers in one dispatch, so this reports the token-level view, not per-layer
## cosines.
##
## Its own prompt variable: PROMPT is empty by default (that is what selects the
## canonical Paris prompt for `make run`), and an empty --prompt is not a prompt.
DIAG_PROMPT ?= What is the capital of France?
diagnosis:
	cd $(srcdir)/.. && python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(if $(PROMPT),$(PROMPT),$(DIAG_PROMPT))" \
		--model $(MODEL_CHOICE)

## Warm prefill TTFT at CTX (warmup pass, then a timed one). Isolates the
## prefill dispatch from the one-time weight load the e2e path also pays.
profile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && Q4NX_BENCH_L=$(CTX) python3 $(PREFILL_DRIVER)

## End-to-end: TTFT + decode throughput over N_TOKENS tokens (prints the TTFT and
## Tokens/second lines the nightly LLM benchmark extracts). Needs production
## templates: make compile.
profile:
	@cd $(srcdir) && python3 qwen3_4b_q4nx_inference.py \
		--n-tokens $(N_TOKENS) --no-eos-stop --prompt "$(PROFILE_PROMPT)"

## Back-compat alias for the Paris gate.
gen: run

## Remove build artifacts, kernel/weight caches, and the decode templates. Also
## removes the QWEN3_4B-compiled kernels this Makefile writes into the shared
## fused_decode/ dir -- leaving them there makes another model's build silently
## link Qwen3-4B objects.
clean:
	rm -f $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin 2>/dev/null || true
	rm -f $(srcdir)/decode_dyn_L*.xclbin $(srcdir)/decode_dyn_L*.insts.bin \
	      $(srcdir)/decode_dyn_L*.txn.h 2>/dev/null || true
	rm -f $(DECODE_FLAGS_STAMP) $(DECODE_WINDOWS_STAMP) 2>/dev/null || true
	rm -f $(addprefix $(DEC)/,$(addsuffix .o,$(NONATTN_KERNELS))) \
	      $(addprefix $(DEC)/,$(addsuffix .ll,$(ATTN_LL_KERNELS))) 2>/dev/null || true
	rm -rf $(srcdir)/build_peano $(srcdir)/build_chess $(srcdir)/__pycache__ \
	       $(srcdir)/_q4nx_* 2>/dev/null || true
	@echo "Qwen3-4B decode templates, build dirs and shared-dir kernels removed."
