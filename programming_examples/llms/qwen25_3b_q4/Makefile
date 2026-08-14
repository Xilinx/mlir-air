# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen2.5-3B Q4_0 on NPU2 (MLIR-AIR): op-by-op prefill + the fused
# single-dispatch decode.
#
# Same driver contract as the other Q4NX examples (llama32_1b_q4nx,
# llama32_3b_q4nx, gemma3_4b_q4nx): compile/compile-decode are weight-free
# builds, run/gen/bench dispatch on the NPU.
#
#   make compile         — build/cache the prefill ELFs (no weights, no NPU dispatch)
#   make compile-decode  — build the fused decode kernels + decode_qwen.xclbin
#   make run             — prefill "The capital of France is" -> expect " Paris"
#   make gen             — end-to-end: NPU prefill -> KV hand-off -> NPU fused decode
#   make bench           — warm TTFT at BENCH_L (default 2048)
#
# Config: 36 layers, emb=2048, 16 heads / 2 KV heads (GQA group 8), head_dim=128,
#         hidden=11008, vocab=151936, rope_theta=1e6, tied lm_head.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

SEQ_LEN  ?= 2048
N_LAYERS ?= 36
BENCH_L  ?= 2048
MODEL    ?= Qwen/Qwen2.5-3B-Instruct

PREFILL := $(srcdir)/qwen25_3b_q4_prefill.py

# ---- fused decode (the shared programming_examples/fused_decode engine) ----
# Qwen drives its own builder (fused_decode_qwen.py) rather than fused_decode.py's
# DECODE_MODEL switch: its attention topology is 2 CUs of 1x8x1 on column 7 at
# DH=128, not the Llama/Gemma layout. Everything else -- the kernels, the
# inline-attn merge, the llvm-link preflight -- is shared.
DEC            := $(srcdir)/../../fused_decode
DECODE_DRIVER  := $(DEC)/fused_decode_qwen.py
HANDOFF        := $(DEC)/qwen_prefill_to_decode.py
AIEOPT_DIR     := $(shell realpath $(dir $(shell which aie-opt))/..)

# Decode context CAP. The attention herd takes L as a runtime operand (an RTP the
# instruction stream writes), so the device attends over the real context and
# appends at the real position; LBUILD only sizes the KV cache. Two templates one
# apart are built so DecodeInstsGen can calibrate the per-token L slope. Any
# prompt up to LBUILD works -- there is no minimum.
LBUILD      ?= 256
LREF        := $(shell echo $$(($(LBUILD) - 1)))
ATTN_L      ?= $(LBUILD)
# W_DUAL_CHAN feeds each proj column's weights on BOTH of its shim MM2S channels
# and reorders the DDR cascade accordingly, so the build and the weight packer
# must agree. The builder stamps decode_qwen.flags and the hand-off refuses a
# mismatch; this variable is what keeps the two in sync from one place.
W_DUAL_CHAN ?= 1

# Per-kernel -O is LOAD-BEARING (see fused_decode/Makefile): proj_qmm/rms_residual/
# glu/rope MUST be -O2 (rope.cc at -O1 miscompiles -> the decode HANGS on device);
# attn_qk/attn_kv MUST be -O1 (a -O2 attn do-while deadlock). Qwen differs from the
# Llama build only by -DMODEL_TYPE=QWEN2_5_3B.
NONATTN_KERNELS := proj_qmm rms_residual glu rope   # object .o, -O2
ATTN_LL_KERNELS := attn_qk attn_kv                  # inline .ll only, -O1 (merge-linked)
PEANO_KBASE     := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=QWEN2_5_3B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(DEC)/kernels -I $(DEC)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
DECODE_ENV      := QWEN_NLAYERS=$(N_LAYERS) ATTN_L=$(ATTN_L) W_DUAL_CHAN=$(W_DUAL_CHAN)

# Build-flag stamp for the idempotent skip. Every variable below changes either the
# xclbin's dataflow or its DDR weight layout, so a warm build made with different
# settings must NOT be silently reused. (Distinct from the builder's own
# decode_qwen.flags, which records W_DUAL_CHAN for the *runtime* packer check and
# whose single-line format the hand-off parses.)
DECODE_STAMP := $(srcdir)/.decode_build_flags
DECODE_FLAGS := MODEL_TYPE=QWEN2_5_3B NLAYERS=$(N_LAYERS) LBUILD=$(LBUILD) W_DUAL_CHAN=$(W_DUAL_CHAN)

# End-to-end generation demo. The only bound now is the cache cap: the prompt
# plus the generated tokens must fit in LBUILD positions. There is no minimum --
# the decode attends over whatever context it actually has.
N_GEN      ?= 16
GEN_PROMPT ?= The capital city of France is called Paris, and the most famous landmark in the entire world is probably the very tall iron tower that stands right in the middle of
KV_NPZ     ?= $(srcdir)/.kv_handoff.npz

# ---- shared verify subsystem (programming_examples/llms/verify/) ----
VERIFY_RUNNER  := $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER := qwen25_3b_q4.verify_adapter
# This example's own prompt file. It predates the growing KV cache (it exists
# because every line had to be >= ATTN_L tokens); kept so the gate's prompt set
# stays comparable across runs.
VERIFY_PROMPTS := $(srcdir)/verify_prompts.txt
MODEL_CHOICE   ?= instruct
MAX_PROMPTS    ?= 2

.PHONY: help compile compile-decode _compile_decode_build run gen dump-kv bench \
        weights verify verify-topk verify-topk-full verify-paris diagnosis \
        profile clean

help:
	@echo "============================================================"
	@echo " Qwen2.5-3B Q4_0 on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "  make compile         Build/cache the prefill ELFs (CI-runnable, no weights)"
	@echo "  make compile-decode  Build the fused decode (kernels + decode_qwen.xclbin, no weights)"
	@echo "  make run             Prefill gate: 'The capital of France is' -> ' Paris'"
	@echo "  make gen             End-to-end NPU prefill -> KV hand-off -> NPU fused decode"
	@echo "  make bench           Warm prefill TTFT at BENCH_L=$(BENCH_L)"
	@echo "  make weights         Q4_0 quant-error report for layer 0 (no NPU)"
	@echo ""
	@echo "Correctness:"
	@echo "  make verify          CI gate: prefill argmax (12095 ' Paris') + a full fused-decode generation"
	@echo "  make verify-paris    Fast prefill-only weight-integrity smoke (no decode build)"
	@echo "  make verify-topk     Top-k token-set gate vs HF bf16 (informational; 4/5 prompts pass today)"
	@echo "  make verify-topk-full  Same over every prompt in verify_prompts.txt"
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
	cd $(srcdir) && python3 $(PREFILL) --compile-only \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS)

## Build the fused Qwen decode: the Peano kernels (QWEN2_5_3B) + decode_qwen.xclbin.
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

_compile_decode_build:
	@# Delegate the llvm-link version preflight to the engine that owns it rather
	@# than keeping a copy here: the rule (LLVM <23) must stay in one place, in
	@# sync with lit.cfg.py's llvm_link_pre23 feature that gates this example's
	@# lit tests (fused_decode/Makefile: preflight-llvm-link).
	@$(MAKE) --no-print-directory -C $(DEC) preflight-llvm-link \
	  PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)
	@echo ">>> qwen non-attn decode kernels (Peano -O2): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O2 -c $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.o || exit 1; done
	@echo ">>> qwen attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@# Merge-linked INTO the core (mlir-aie #3399), not object-linked: the builder
	@# tags the kernel decls link_with="<name>.ll" + link_with_mode="merge".
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(DEC)/$$k.ll || exit 1; done
	@# Two templates one L apart, into THIS example's dir. They must not land in
	@# $(DEC): DecodeInstsGen discovers templates by scanning for decode_L*, so a
	@# Qwen pair sitting next to the Llama decode_L2048 pair would be picked up as
	@# another window of the same model.
	@for W in $(LBUILD) $(LREF); do \
	  echo ">>> qwen decode template decode_L$$W (NLAYERS=$(N_LAYERS) W_DUAL_CHAN=$(W_DUAL_CHAN))"; \
	  cd $(DEC) && QWEN_NLAYERS=$(N_LAYERS) ATTN_L=$$W W_DUAL_CHAN=$(W_DUAL_CHAN) \
	    python3 $(DECODE_DRIVER) >/tmp/qwen_decode_L$$W.log 2>&1; \
	  if [ -f $(DEC)/decode_qwen.xclbin ] && [ -f $(DEC)/decode_qwen.insts.bin ]; then \
	    mv -f $(DEC)/decode_qwen.xclbin $(srcdir)/decode_L$$W.xclbin; \
	    mv -f $(DEC)/decode_qwen.insts.bin $(srcdir)/decode_L$$W.insts.bin; \
	    mv -f $(DEC)/decode_qwen.flags $(srcdir)/decode_L$$W.flags; \
	  else echo "decode_L$$W FAILED (see /tmp/qwen_decode_L$$W.log)"; \
	    tail -30 /tmp/qwen_decode_L$$W.log; exit 1; fi; \
	done
	@echo "Qwen decode build complete: decode_L$(LBUILD) + decode_L$(LREF) templates."

## First-token gate: prefill "The capital of France is"; PASS iff argmax == 12095.
run:
	cd $(srcdir) && python3 $(PREFILL) \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS) --model $(MODEL)

## Prefill GEN_PROMPT and dump the per-layer roped-K / biased-V hand-off cache.
dump-kv:
	cd $(srcdir) && python3 $(PREFILL) \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS) --model $(MODEL) \
		--prompt "$(GEN_PROMPT)" --dump-kv $(KV_NPZ)

## End-to-end: NPU prefill -> KV hand-off -> N_GEN tokens on the fused NPU decode.
## No reference model in the loop. The decode's first token equals the prefill's
## own argmax, which is the cross-check that the hand-off is exact -- and now
## holds for ANY prompt that fits the cache, since both sides see the same
## context (they used to diverge once the prompt outgrew the decode's window).
gen: dump-kv
	cd $(DEC) && $(DECODE_ENV) python3 $(HANDOFF) --kv $(KV_NPZ) --n-gen $(N_GEN) \
		--templates $(srcdir)

## Warm TTFT benchmark (+ per-ELF profile).
bench:
	cd $(srcdir) && python3 $(PREFILL) \
		--seq-len $(SEQ_LEN) --n-layers $(N_LAYERS) --model $(MODEL) \
		--bench-l $(BENCH_L)

## Q4_0 quant-error report for layer 0 (no NPU).
weights:
	cd $(srcdir) && python3 $(srcdir)/qwen25_3b_q4_weights.py --model $(MODEL)

# ---- correctness gates ----
#
# WHY `verify` IS NOT THE TOP-K GATE HERE (it is for the Llama Q4NX examples).
# The shared top-k gate demands that, at the first token where NPU and HF bf16
# disagree, each side's pick is in the other's top-5. Measured on this design:
# 4/5 prompts pass (2/5 before the decode grew a real KV cache; the head of a
# 32+ token prompt used to slide out of context). The decode itself is sound --
# `make gen` produces "the city. It is called the Eiffel Tower, and it was
# built" -- but greedy runs of a
# Q4_0 36-layer model and a bf16 reference diverge early on open-ended
# continuations, and near-ties there fall outside top-5. Gating CI on that would
# be red, and loosening k or hand-picking prompts until it passed would make the
# gate meaningless. So `verify` gates on two things that DO hold deterministically,
# and the top-k path stays available (and honest) as `verify-topk`.
# gemma3_4b_q4nx made the same call for the same reason.

## CI correctness gate: prefill first-token argmax + a full fused-decode
## generation. Between them these cover both halves on device:
##   run  36 NPU prefill layers -> argmax 12095 (" Paris")
##   gen  36 NPU decode layers, 16 one-dispatch tokens over the KV hand-off
## `gen` is also the guard against an under-primed fill lock, i.e. a decode
## deadlock -- the failure mode the refeed mechanism exists to prevent, and the
## reason the other three Q4NX designs each have a decode gate.
verify: run gen

## Full production top-k token-set inclusion gate (NPU q4 vs HF bf16, k=5) via
## the shared verify subsystem and verify_adapter.py. Informational today; see
## the note above for why it is not the CI gate.
##
## DECODE_ENV is passed here too, not just to the build: fused_decode_qwen reads
## QWEN_NLAYERS / ATTN_L / W_DUAL_CHAN from the environment at import and they
## default to a 1-layer lowering check. Without them the runner would pack a
## 1-layer weight stream for a 36-layer xclbin.
verify-topk:
	cd $(srcdir)/.. && $(DECODE_ENV) python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --prompts-file $(VERIFY_PROMPTS) \
		--model $(MODEL_CHOICE) --max-prompts $(MAX_PROMPTS)

## Full-sweep variant of `make verify-topk` (every prompt in verify_prompts.txt).
verify-topk-full:
	cd $(srcdir)/.. && $(DECODE_ENV) python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --prompts-file $(VERIFY_PROMPTS) \
		--model $(MODEL_CHOICE)

## Fast weight-integrity smoke: prefill only, no decode build required.
## PASS iff the first-token argmax is 12095 (" Paris"). Alias for `make run`.
verify-paris: run

## Diagnosis lens (single prompt through the shared runner, informational).
## The Q4_0 prefill does not expose per-layer ffn_out, so this reports the
## token-level view rather than per-layer cosines.
diagnosis:
	cd $(srcdir)/.. && $(DECODE_ENV) python3 $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
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
	rm -f $(DEC)/decode_qwen.xclbin $(DEC)/decode_qwen.insts.bin \
	      $(DEC)/decode_qwen.flags 2>/dev/null || true
	rm -f $(addprefix $(DEC)/,$(addsuffix .o,$(NONATTN_KERNELS))) \
	      $(addprefix $(DEC)/,$(addsuffix .ll,$(ATTN_LL_KERNELS))) 2>/dev/null || true
	@echo "Qwen caches, decode artifacts and shared-dir kernels removed."
