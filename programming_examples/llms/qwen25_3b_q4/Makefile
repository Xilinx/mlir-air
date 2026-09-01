# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen2.5-3B Q4NX on NPU2 (MLIR-AIR): op-by-op prefill + the fused
# single-dispatch decode.
#
# Same driver contract as the other Q4NX examples (llama32_1b_q4nx,
# llama32_3b_q4nx, gemma3_4b_q4nx): compile/compile-decode are weight-free
# builds, run/gen/bench dispatch on the NPU.
#
#   make compile         — build/cache the prefill ELFs (no weights, no NPU dispatch)
#   make compile-decode  — build the fused decode kernels + the decode_L<N> template pair
#   make run             — prefill "The capital of France is" -> expect " Paris"
#   make gen             — end-to-end: NPU prefill -> NPU fused decode
#   make bench           — warm TTFT at BENCH_L (default 2048)
#
# Config: 36 layers, emb=2048, 16 heads / 2 KV heads (GQA group 8), head_dim=128,
#         hidden=11008, vocab=151936, rope_theta=1e6, tied lm_head.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

PYTHON ?= $(if $(filter Windows_NT,$(OS)),python,python3)

SEQ_LEN  ?= 2048
N_LAYERS ?= 36
BENCH_L  ?= 2048
# NPU weight source: FastFlowLM's pre-quantized Q4NX bundle. A local dir holding
# model.q4nx, a direct path, or a full-precision HF checkpoint (quantized on load)
# also work.
MODEL    ?= FastFlowLM/Qwen2.5-3B-Instruct-NPU2

PREFILL   := $(srcdir)/qwen25_3b_q4_prefill.py
INFERENCE := $(srcdir)/qwen25_3b_q4_inference.py

# ---- fused decode (the shared programming_examples/fused_decode engine) ----
# Qwen2.5-3B is one of the fused_decode engine's DECODE_MODEL entries, like the
# other nine models; it differs from qwen2.5-7b in dimensions only (2 attention
# CUs of 1x8x1 instead of 4). The kernels, the inline-attn merge and the
# llvm-link preflight are shared too.
DEC            := $(srcdir)/../../fused_decode
DECODE_DRIVER  := $(DEC)/fused_decode.py
AIEOPT_DIR ?= $(if $(MLIR_AIE_INSTALL_DIR),$(subst \,/,$(MLIR_AIE_INSTALL_DIR)),$(shell realpath $(dir $(shell which aie-opt))/..))
# Decode context CAP. The attention herd takes L as a runtime operand (an RTP the
# instruction stream writes), so the device attends over the real context and
# appends at the real position; LBUILD only sizes the KV cache. Two templates one
# apart are built so DecodeInstsGen can calibrate the per-token L slope. Any
# prompt up to LBUILD works -- there is no minimum.
#
# 2048 matches the three Llama/Gemma Q4NX decodes. It costs 2.5 ms/token against
# a 256 build (35.5 vs 33.0 ms/tok, same box and power mode, 16-token `make gen`,
# byte-identical output), because the readback streams the padded ATTN_MAXL --
# `make compile-decode-dynseq` removes exactly that.
LBUILD      ?= 2048
LREF        := $(shell echo $$(($(LBUILD) - 1)))
ATTN_L      ?= $(LBUILD)
# W_DUAL_CHAN feeds each proj column's weights on BOTH of its shim MM2S channels
# and reorders the DDR cascade accordingly, so the build and the weight packer
# must agree. The packer reads the flag off the loaded engine rather than the
# environment, so the two cannot disagree.
#
# OFF for this model, unlike the other nine. The dual feed wedges every decode
# dispatch on a Krackan NPU -- 0 of 13 complete, at every context from 1024 to
# 32768, with no prefill and no weights involved, while llama32_3b_q4nx runs at
# 28 tok/s on the same part minutes apart. Turning it off completes 10 of 10
# across that whole range. Costs ~16% decode on Strix (23.4 -> 19.6 tok/s).
# EXPORTED because the builder reads it from the environment: a plain `?=`
# reaches the stamp but never the engine, so the value here would be cosmetic.
W_DUAL_CHAN ?= 0
export W_DUAL_CHAN

# Per-kernel -O is LOAD-BEARING (see fused_decode/Makefile): proj_qmm/rms_residual/
# glu/rope MUST be -O2 (rope.cc at -O1 miscompiles -> the decode HANGS on device);
# attn_qk/attn_kv MUST be -O1 (a -O2 attn do-while deadlock). Qwen differs from the
# Llama build only by -DMODEL_TYPE=QWEN2_5_3B.
NONATTN_KERNELS := proj_qmm rms_residual glu rope   # object .o, -O2
ATTN_LL_KERNELS := attn_qk attn_kv                  # inline .ll only, -O1 (merge-linked)

# Extra kernel defines, as in fused_decode/Makefile -- read that file's header
# for what these are for. The other three Q4NX examples delegate their kernel
# builds to it and so inherit the hook; this Makefile compiles its own kernels
# and had dropped it. Empty by default, so the shipped path is byte-identical. Example:
#     make compile-decode NONATTN_EXTRA=-DQ4_SFIX_MODE=0
# rebuilds against the pre-xor-bias int4 sign fix, which is how the 2.4x in
# README.md's decode table was measured on one box in one session.
ATTN_EXTRA      ?=
NONATTN_EXTRA   ?=
PEANO_KBASE     := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=QWEN2_5_3B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(DEC)/kernels -I $(DEC)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
# Shared-engine decode env, as for the other nine models. VOCAB_CHUNK_I2 is NOT
# free: it must pair with the model entry's UNI_LM=25 (product covers the padded
# vocab). 20/15 also satisfies every divisibility rule and still deadlocks the
# vocab wave on device, so do not 'simplify' it.
# NLAYERS=1 as for every other model: it is the multi-launch outer stitch, which
# the unified wave loop does not use. N_LAYERS (36) reaches the engine through
# the model entry's UNI_DEC, not through here.
DECODE_ENV      := DECODE_MODEL=qwen2.5-3b VOCAB_CHUNK_I2=12 UNIFIED=1 LM_HEAD=0 \
                   NLAYERS=1 DECODE_GOLDEN=1

# GLU_SLICE is derived from this model's own egress-round parity, not looked up,
# so a header copied from another model can disagree with the IR -- silently, at
# runtime. Hand the builder's value to the kernel compile; glu.cc asserts on it.
# Empty (and the assert inert) if the query fails, so this cannot break a build
# that works today.
GLU_SLICE_EXPECTED := $(shell $(DECODE_ENV) FUSED_DECODE_PRINT_CONST=GLU_SLICE python3 $(DECODE_DRIVER) 2>/dev/null)
PEANO_KBASE += $(if $(GLU_SLICE_EXPECTED),-DGLU_SLICE_EXPECTED=$(GLU_SLICE_EXPECTED))

# Build-flag stamp for the idempotent skip. Every variable below changes either the
# xclbin's dataflow, its DDR weight layout or its core ELFs, so a warm build made
# with different settings must NOT be silently reused.
#
# The kernel defines are in the stamp for the same reason the rest are: without
# them `make compile-decode NONATTN_EXTRA=...` would find the templates present,
# skip the build and hand back the PREVIOUS kernels -- an A/B that silently
# measures the same binary twice.
DECODE_STAMP := $(srcdir)/.decode_build_flags
DECODE_FLAGS := MODEL_TYPE=QWEN2_5_3B N_LAYERS=$(N_LAYERS) LBUILD=$(LBUILD) W_DUAL_CHAN=$(W_DUAL_CHAN) \
                ATTN_EXTRA=$(ATTN_EXTRA) NONATTN_EXTRA=$(NONATTN_EXTRA)

# End-to-end generation demo. The only bound now is the cache cap: the prompt
# plus the generated tokens must fit in LBUILD positions. There is no minimum --
# the decode attends over whatever context it actually has.
N_GEN      ?= 16
GEN_PROMPT ?= The capital city of France is called Paris, and the most famous landmark in the entire world is probably the very tall iron tower that stands right in the middle of
KV_NPZ     ?= $(srcdir)/.kv_handoff.npz

# ---- shared verify subsystem (programming_examples/llms/verify/) ----
VERIFY_RUNNER  := $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER := qwen25_3b_q4.verify_adapter
MODEL_CHOICE   ?= instruct
MAX_PROMPTS    ?= 2

.PHONY: help compile compile-decode compile-decode-dynseq _compile_decode_build run gen dump-kv bench \
        weights verify verify-full verify-paris diagnosis profile profile-prefill clean

help:
	@echo "============================================================"
	@echo " Qwen2.5-3B Q4NX on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "  make compile         Build/cache the prefill ELFs (CI-runnable, no weights)"
	@echo "  make compile-decode  Build the fused decode (kernels + decode_L<N> templates, no weights)"
	@echo "  make run             Prefill gate: 'The capital of France is' -> ' Paris'"
	@echo "  make gen             End-to-end NPU prefill -> NPU fused decode"
	@echo "  make bench           Warm prefill TTFT at BENCH_L=$(BENCH_L) (alias: profile-prefill)"
	@echo "  make weights         Q4NX weight-load smoke for layer 0 (no NPU)"
	@echo ""
	@echo "Correctness:"
	@echo "  make verify          Top-k token-set gate vs HF bf16 (2 prompts x 32 tokens, k=5)"
	@echo "  make verify-full     Same over every prompt in the shared prompt file"
	@echo "  make verify-paris    Fast prefill-only weight-integrity smoke (no decode build)"
	@echo "  make diagnosis       Single-prompt lens through the shared runner (informational)"
	@echo "  make profile         Perf capture (prefill TTFT + decode tok/s) for bench/extract_perf.py"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_LAYERS=$(N_LAYERS)        Transformer layers (prefill and decode)"
	@echo "  LBUILD=$(LBUILD)           Decode KV cache cap (max prompt + generated tokens)"
	@echo "  W_DUAL_CHAN=$(W_DUAL_CHAN)        Two-MM2S-channel weight feed per proj column"
	@echo "  N_GEN=$(N_GEN)            Tokens generated by 'make gen'"

## Build/cache the prefill ELFs and exit. No weights, no NPU dispatch.
compile:
	cd $(srcdir) && $(PYTHON) $(PREFILL) --compile-only \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS)

## Build the fused Qwen decode: the Peano kernels (QWEN2_5_3B) + two templates
## one L apart, which is what calibrates the per-token L patch.
## Weight-free (everything is a runtime BO), like `make compile`.
##
## Idempotent: skipped when the xclbin, its instruction stream and a matching
## build-flag stamp are all present. `make clean` forces a rebuild.
compile-decode:
	@if [ -f $(srcdir)/decode_L$(LBUILD).xclbin ] && [ -f $(srcdir)/decode_L$(LBUILD).insts.bin ] \
	    && [ -f $(srcdir)/decode_L$(LREF).xclbin ] && [ -f $(srcdir)/decode_L$(LREF).insts.bin ] \
	    && [ "`cat $(DECODE_STAMP) 2>/dev/null`" = "$(DECODE_FLAGS)" ]; then \
	  echo "Qwen decode already built (decode_L$(LBUILD) + decode_L$(LREF), $(DECODE_FLAGS)); skipping. Run 'make clean' to force a rebuild."; \
	else \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build \
	    && echo "$(DECODE_FLAGS)" > $(DECODE_STAMP); \
	fi

## Dynamic-runtime-sequence template: ONE build for every context length (see
## fused_decode/Makefile compile-decode-dynseq). Run with DECODE_DYNSEQ=1.
compile-decode-dynseq:
	@$(MAKE) --no-print-directory -C $(DEC) preflight-llvm-link \
	  PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)
	@echo ">>> qwen non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(NONATTN_EXTRA) -O2 -c $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.o || exit 1; done
	@echo ">>> qwen attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(ATTN_EXTRA) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.ll || exit 1; done
	@echo ">>> qwen dynseq template decode_dyn_L$(LBUILD)"
	@cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LBUILD) DECODE_DYNSEQ=1 \
	  $(PYTHON) $(DECODE_DRIVER) >/tmp/qwen_decode_dyn_L$(LBUILD).log 2>&1; \
	  if [ -f $(DEC)/decode.xclbin ] && [ -f $(DEC)/decode.insts.bin ]; then \
	    mv -f $(DEC)/decode.xclbin $(srcdir)/decode_dyn_L$(LBUILD).xclbin; \
	    mv -f $(DEC)/decode.insts.bin $(srcdir)/decode_dyn_L$(LBUILD).insts.bin; \
	    rm -f $(DEC)/npu_seq_*.mlir; \
	  else echo "decode_dyn_L$(LBUILD) FAILED (see /tmp/qwen_decode_dyn_L$(LBUILD).log)"; \
	    tail -30 /tmp/qwen_decode_dyn_L$(LBUILD).log; exit 1; fi
	@echo "Qwen dynseq build complete: decode_dyn_L$(LBUILD). Run with DECODE_DYNSEQ=1."

_compile_decode_build:
	@# Delegate the llvm-link version preflight to the engine that owns it rather
	@# than keeping a copy here: the rule (LLVM <23) must stay in one place, in
	@# sync with lit.cfg.py's llvm_link_pre23 feature that gates this example's
	@# lit tests (fused_decode/Makefile: preflight-llvm-link).
	@$(MAKE) --no-print-directory -C $(DEC) preflight-llvm-link \
	  PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)
	@echo ">>> qwen non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(NONATTN_EXTRA) -O2 -c $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.o || exit 1; done
	@echo ">>> qwen attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@# Merge-linked INTO the core (mlir-aie #3399), not object-linked: the builder
	@# tags the kernel decls link_with="<name>.ll" + link_with_mode="merge".
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(ATTN_EXTRA) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.ll || exit 1; done
	@# Two templates one L apart, into THIS example's dir. They must not land in
	@# $(DEC): DecodeInstsGen discovers templates by scanning for decode_L*, so a
	@# Qwen pair sitting next to the Llama decode_L2048 pair would be picked up as
	@# another window of the same model.
	@for W in $(LBUILD) $(LREF); do \
	  echo ">>> qwen decode template decode_L$$W (N_LAYERS=$(N_LAYERS))"; \
	  cd $(DEC) && $(DECODE_ENV) DECODE_GOLDEN_L=$$W \
	    $(PYTHON) $(DECODE_DRIVER) >/tmp/qwen_decode_L$$W.log 2>&1; \
	  if [ -f $(DEC)/decode.xclbin ] && [ -f $(DEC)/decode.insts.bin ]; then \
	    mv -f $(DEC)/decode.xclbin $(srcdir)/decode_L$$W.xclbin; \
	    mv -f $(DEC)/decode.insts.bin $(srcdir)/decode_L$$W.insts.bin; \
	  else echo "decode_L$$W FAILED (see /tmp/qwen_decode_L$$W.log)"; \
	    tail -30 /tmp/qwen_decode_L$$W.log; exit 1; fi; \
	done
	@echo "Qwen decode build complete: decode_L$(LBUILD) + decode_L$(LREF) templates."

## First-token gate: prefill "The capital of France is"; PASS iff argmax == 12095.
run:
	cd $(srcdir) && $(PYTHON) $(PREFILL) \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS) --model $(MODEL)

## Prefill GEN_PROMPT and dump the per-layer roped-K / raw-V cache (debug aid).
dump-kv:
	cd $(srcdir) && $(PYTHON) $(PREFILL) \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS) --model $(MODEL) \
		--prompt "$(GEN_PROMPT)" --dump-kv $(KV_NPZ)

## End-to-end: NPU prefill -> N_GEN tokens on the fused NPU decode. No reference
## model in the loop. Prints the "Time to first token (TTFT)" and "Tokens/second"
## lines llms/bench/extract_perf.py parses.
## The default prompt is the Paris gate, so this also reports whether the decode
## reproduces the recorded greedy continuation.
gen:
	cd $(srcdir) && $(PYTHON) $(INFERENCE) --model $(MODEL) \
		--n-tokens $(N_GEN) --no-eos-stop

## Warm TTFT benchmark (+ per-ELF profile).
bench:
	cd $(srcdir) && $(PYTHON) $(PREFILL) \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS) --model $(MODEL) \
		--bench-l $(BENCH_L)

## The name the other three use for the same thing. `bench` stays because
## `profile` composes it and llms/bench/extract_perf.py parses its output.
profile-prefill: bench

## Q4NX weight-load smoke for layer 0 (no NPU).
weights:
	cd $(srcdir) && $(PYTHON) $(srcdir)/qwen25_3b_q4_weights.py --model $(MODEL)

# ---- correctness gate (identical to the other Q4NX examples) ----
#
# No DECODE_ENV here: the inference driver sets the engine's geometry itself,
# before importing fused_decode.py, so the runner cannot be handed a different
# model than the one the xclbin was built for.

## Top-k token-set inclusion gate (NPU q4 vs HF bf16, 2 prompts x 32 tokens, k=5).
verify:
	cd $(srcdir)/.. && Q4NX_MODEL_SOURCE=$(MODEL) $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL_CHOICE) --max-prompts $(MAX_PROMPTS)

## Full-sweep variant of `make verify` (all prompts in the prompt file).
verify-full:
	cd $(srcdir)/.. && Q4NX_MODEL_SOURCE=$(MODEL) $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL_CHOICE)

## Fast weight-integrity smoke: prefill "The capital of France is"; PASS iff the
## first-token argmax == 12095 (" Paris"). Cheap (prefill only, no decode build,
## no HF reference) -- runnable when the top-k gate above is not.
verify-paris: run

## Diagnosis lens (single prompt through the shared runner, informational).
## The Q4NX prefill does not expose per-layer ffn_out, so this reports the
## token-level view rather than per-layer cosines.
diagnosis:
	cd $(srcdir)/.. && Q4NX_MODEL_SOURCE=$(MODEL) $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(GEN_PROMPT)" --model $(MODEL_CHOICE)

## Perf capture for llms/bench/extract_perf.py: prefill TTFT + fused decode
## throughput. `bench` prints "Time to first token (TTFT): X s" and `gen` prints
## "Tokens/second: X"; extract_perf.py parses both.
profile: bench gen

## Remove this example's caches AND the QWEN2_5_3B-compiled kernels + decode
## artifacts it writes into the shared fused_decode/ dir -- leaving those there
## makes a later Llama or Gemma build silently link Qwen objects.
clean:
	rm -rf $(srcdir)/_q4_cache_seq* $(srcdir)/__pycache__ 2>/dev/null || true
	rm -f $(DECODE_STAMP) $(KV_NPZ) 2>/dev/null || true
	rm -f $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin \
	      $(srcdir)/decode_L*.flags 2>/dev/null || true
	rm -f $(DEC)/decode.xclbin $(DEC)/decode.insts.bin 2>/dev/null || true
	rm -f $(addprefix $(DEC)/,$(addsuffix .o,$(NONATTN_KERNELS))) \
	      $(addprefix $(DEC)/,$(addsuffix .ll,$(ATTN_LL_KERNELS))) 2>/dev/null || true
	@echo "Qwen caches, decode artifacts and shared-dir kernels removed."
