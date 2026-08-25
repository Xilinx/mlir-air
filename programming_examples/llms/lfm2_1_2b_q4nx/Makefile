# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# LFM2-1.2B Q4_0 Inference on NPU2 (MLIR-AIR)
#
# Same driver contract as the other llms/ examples: run/profile/chat drive the
# unified prefill+decode inference driver, and verify plugs into the shared
# verify/ subsystem (top-k token-set inclusion, NPU q4_0 vs HF bf16).
#
# LFM2 is a HYBRID conv-attention decoder: of its 16 layers only 6 are attention
# (indices 2,5,8,10,12,14) and the other 10 are Lfm2ShortConv, a gated causal
# depthwise convolution. Both layer types run in ONE fused decode dispatch, the
# layer type chosen per wave by a runtime arm.
#
# Weights are quantized from the full-precision HF checkpoint rather than from a
# pre-quantized bundle: the device is -DQ4_0 (symmetric) and quantizing an
# already-affine-quantized bundle would quantize twice, so going from fp is both
# simpler and strictly more accurate.
#
# Quick start:
#   make compile        — Build/cache the prefill ELFs (no weights needed)
#   make compile-decode — Build the fused decode templates (~15 min; no weights)
#   make run            — Run inference: NPU prefill + fused NPU decode
#   make verify         — Top-k token-set inclusion gate: NPU q4_0 vs HF bf16
#
# Config: 16 layers (6 attention + 10 ShortConv), emb=2048, 32 heads / 8 KV
#         heads, head_dim=64, hidden=8192, conv_dim=2048, k=3 taps,
#         vocab=65536, QK-norm (NPU2, AMD Strix).

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
# Weight source. LFM2 publishes ONE checkpoint (no separate base/instruct), and
# it is BOTH the NPU weight source (quantized to Q4_0 on load) and the bf16
# reference `make verify` compares against.
MODEL_SOURCE ?= LiquidAI/LFM2-1.2B
export LFM2_MODEL_SOURCE := $(MODEL_SOURCE)

# Prefill context length (see `make profile-prefill`).
CTX ?= 2048
export LFM2_SEQ_LEN   := $(CTX)
export LFM2_CACHE_DIR := $(srcdir)/$(BUILD_DIR)/kernel_cache_$(CTX)

# Dual-MM2S weight feed for the fused decode (see fused_decode/fused_decode.py).
# The BUILD and the RUN must agree -- the flag selects both the shim channel
# split and the weight cascade order -- so export it here and let it reach both
# the decode build and the inference driver.
W_DUAL_CHAN ?= 1
export W_DUAL_CHAN

DEC           := $(srcdir)/../../fused_decode
DECODE_DRIVER := $(DEC)/fused_decode.py
AIEOPT_DIR    := $(shell realpath $(dir $(shell which aie-opt))/..)

# Unified prefill+decode inference driver (run/profile/chat/gen/ask/compile).
DRIVER  := $(srcdir)/lfm2_1_2b_q4nx_inference.py
# Prefill engine (weight-integrity Paris gate + the CTX prefill latency sweep).
PREFILL := $(srcdir)/lfm2_1_2b_q4nx_prefill.py
# Decode-only numerics gate: no prefill, localises a failure to a layer.
DECODE_GATE := $(srcdir)/lfm2_1_2b_q4nx_decode_gate.py

# Shared verify framework (programming_examples/llms/verify/).
VERIFY_RUNNER  = $(srcdir)/../verify/verify_runner.py
RUNNER_ADAPTER = lfm2_1_2b_q4nx.verify_adapter

# Per-kernel -O is LOAD-BEARING (see fused_decode/Makefile): proj_qmm /
# rms_residual / glu / rope / shortconv at -O2; attn_qk / attn_kv as inline .ll
# at -O1 (-O2 deadlocks attn, -O1 miscompiles rope).
#
# Kernels build into THIS directory, never into $(DEC) -- the shared .o/.ll
# there belong to the llama build and would otherwise be silently clobbered
# with LFM2-typed objects.
#
# SC_DIM is the ShortConv channel count; SC_WAVES is the number of ph0 landings
# the assembled [B|C|X] is built from, and MUST match the builder's CONV_WAVES.
SC_DIM   ?= 2048
SC_WAVES ?= 2
export CONV_WAVES := $(SC_WAVES)
NONATTN_KERNELS := proj_qmm rms_residual glu rope
ATTN_LL_KERNELS := attn_qk attn_kv
PEANO_KBASE := -std=c++20 --target=aie2p-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-deprecated-declarations \
  -DNDEBUG -DMODEL_TYPE=LFM2_1_2B -D__AIE_API_AIE_ADF_HPP__ \
  -I $(AIEOPT_DIR)/include -I $(DEC)/kernels -I $(DEC)/models \
  -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
NONATTN_EXTRA ?=
ATTN_EXTRA    ?=

# VOCAB_CHUNK_I2 must satisfy UNI_LM*VOCAB_CHUNK_I2 = VOCAB_FULL_ROWBLKS/32 = 64
# against UNI_LM=4 in the lfm2-1.2b _MODELS entry; fused_decode.py asserts it.
DECODE_ENV := DECODE_MODEL=lfm2-1.2b VOCAB_CHUNK_I2=16 UNIFIED=1 LM_HEAD=0 NLAYERS=1 DECODE_GOLDEN=1

# Build only the first N model layers. The weight, rms and state slabs are all
# per-layer prefixes, so a short build is a prefix of the full one and needs no
# separate weight pack -- which makes this the bisect when the decode gate says
# the model is wrong: LAYERS=1 is ShortConv alone (expect cos >= 0.996) and
# LAYERS=3 adds the first attention layer. Unset = all 16.
# The weight pack deliberately does NOT take LAYERS: the cache is a per-layer
# prefix, so ONE full pack serves the 16-layer model and every short build, and
# re-packing per bisect step would cost minutes for an identical prefix.
WEIGHTS_ENV := $(DECODE_ENV)
LAYERS ?=
ifneq ($(LAYERS),)
  DECODE_ENV += DECODE_UNI_DEC=$(LAYERS)
endif

LBUILD ?= 2048
LREF   := $(shell echo $$(($(LBUILD) - 1)))
DECODE_FLAGS_STAMP := $(srcdir)/.decode_flags
DECODE_FLAGS       := W_DUAL_CHAN=$(W_DUAL_CHAN) LBUILD=$(LBUILD) SC_WAVES=$(SC_WAVES)

.PHONY: help compile compile-decode _compile_decode_build run profile chat \
        verify verify-full diagnosis verify-paris decode-gate weights \
        profile-prefill ask gen clean

help:
	@echo "============================================================"
	@echo " LFM2-1.2B Q4_0 Inference on NPU2 (MLIR-AIR)"
	@echo "============================================================"
	@echo ""
	@echo "Hybrid conv-attention: 16 layers = 6 attention + 10 ShortConv,"
	@echo "all of them in ONE fused decode dispatch."
	@echo ""
	@echo "Quick start:"
	@echo "  make compile          Build/cache the prefill ELFs (no weights)"
	@echo "  make compile-decode   Build the fused decode templates (~15 min, no weights)"
	@echo "  make run              Run inference ($(N_TOKENS) tokens): NPU prefill + decode"
	@echo "  make chat             Interactive chat REPL (streaming output)"
	@echo "  make profile          Run with profiling breakdown"
	@echo ""
	@echo "Correctness:"
	@echo "  make verify           Top-k token-set inclusion gate: NPU q4_0 vs HF bf16 (2 prompts x 32 tokens, k=5)"
	@echo "  make verify-full      Same gate over the full prompt set"
	@echo "  make diagnosis        Per-layer cosine vs HF bf16 (single prompt, informational)"
	@echo "  make verify-paris     Fast weight-integrity smoke: prefill first-token argmax gate"
	@echo "  make decode-gate      Decode-ONLY numerics (no prefill); localises a failure to a layer"
	@echo ""
	@echo "Prefill latency sweep:"
	@echo "  make profile-prefill  Warm prefill TTFT benchmark at CTX (per-ELF breakdown)"
	@echo ""
	@echo "Options (override with make VAR=value):"
	@echo "  N_TOKENS=1000         Max decode tokens for run/profile/chat"
	@echo "  PROMPT=\"...\"          Input prompt text (run/profile/diagnosis)"
	@echo "  CTX=2048              Prefill context length for profile-prefill"
	@echo "  LBUILD=2048           ATTN_MAXL of the decode templates (128 for a fast CI build)"
	@echo "  LAYERS=               Build only the first N layers (bisect; unset = all 16)"
	@echo "  MODEL_SOURCE=...      Weight source (default: $(MODEL_SOURCE))"

## Build/cache the prefill ELFs. No weight data or NPU dispatch required.
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) --compile-only

## Pack the Q4_0 weight cache the fused decode streams (both layer types, one
## uniform per-layer slab). `decode-gate` needs it; the inference driver builds
## it on demand.
weights:
	@mkdir -p $(srcdir)/$(BUILD_DIR)
	cd $(DEC) && $(WEIGHTS_ENV) $(PYTHON) -c "\
import importlib.util, sys; \
spec = importlib.util.spec_from_file_location('fd', 'fused_decode.py'); \
fd = importlib.util.module_from_spec(spec); sys.modules['fd'] = fd; \
exec('try:\n spec.loader.exec_module(fd)\nexcept SystemExit:\n pass'); \
import lfm2_requant as L; \
L.build_requant_cache(fd, '$(srcdir)/$(BUILD_DIR)/lfm2_all_full.npz', layer_kind='all')"

## Build the fused decode templates. Two same-ATTN_MAXL builds (L and L-1)
## calibrate the L-slope so one template serves every context length.
compile-decode:
	@if [ -f $(srcdir)/decode_L$(LBUILD).xclbin ] && [ -f $(srcdir)/decode_L$(LBUILD).insts.bin ] \
	    && [ -f $(srcdir)/decode_L$(LREF).xclbin ] && [ -f $(srcdir)/decode_L$(LREF).insts.bin ] \
	    && [ "`cat $(DECODE_FLAGS_STAMP) 2>/dev/null`" = "$(DECODE_FLAGS)" ]; then \
	  echo "LFM2 decode templates already built ($(DECODE_FLAGS)); skipping. Run 'make clean' to force a rebuild."; \
	else \
	  $(MAKE) --no-print-directory -f $(srcdir)/Makefile _compile_decode_build \
	    && echo "$(DECODE_FLAGS)" > $(DECODE_FLAGS_STAMP); \
	fi

# Entry point bench/sweep_decode.py invokes BY NAME to rebuild at each swept
# context. Renaming it silently breaks the nightly decode sweep.
_compile_decode_build:
	@echo ">>> non-attn decode kernels (Peano -O2, MODEL_TYPE=LFM2_1_2B): $(NONATTN_KERNELS)"
	@for k in $(NONATTN_KERNELS); do echo "    $$k.cc (-O2)"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(NONATTN_EXTRA) -O2 -c $(DEC)/kernels/$$k.cc -o $(srcdir)/$$k.o || exit 1; done
	@echo "    shortconv.cc (-O2, SC_DIM=$(SC_DIM) SC_WAVES=$(SC_WAVES))"
	@$(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(NONATTN_EXTRA) -DSC_DIM=$(SC_DIM) -DSC_WAVES=$(SC_WAVES) \
	  -O2 -c $(DEC)/kernels/shortconv.cc -o $(srcdir)/shortconv.o || exit 1
	@echo ">>> attn decode kernels (Peano -O1, inline .ll only): $(ATTN_LL_KERNELS)"
	@for k in $(ATTN_LL_KERNELS); do echo "    $$k.ll"; \
	  $(PEANO_INSTALL_DIR)/bin/clang++ $(PEANO_KBASE) $(ATTN_EXTRA) -O1 -DDECODE_INLINE_ATTN -S -emit-llvm \
	    $(DEC)/kernels/$$k.cc -o $(srcdir)/$$k.ll || exit 1; done
	@echo ">>> template decode_L$(LBUILD) (+ L$(LREF) slope ref)"
	@cd $(srcdir) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LBUILD) $(PYTHON) $(DECODE_DRIVER) >/tmp/lfm2_decode_L$(LBUILD).log 2>&1; \
	  if [ -f decode.xclbin ]; then mv -f decode.xclbin decode_L$(LBUILD).xclbin; mv -f decode.insts.bin decode_L$(LBUILD).insts.bin; \
	    cp -f air.mlir decode_L$(LBUILD).air.mlir; \
	  else echo "decode_L$(LBUILD) FAILED (see /tmp/lfm2_decode_L$(LBUILD).log)"; exit 1; fi
	@cd $(srcdir) && $(DECODE_ENV) DECODE_GOLDEN_L=$(LREF) $(PYTHON) $(DECODE_DRIVER) >/tmp/lfm2_decode_L$(LREF).log 2>&1; \
	  if [ -f decode.insts.bin ]; then mv -f decode.insts.bin decode_L$(LREF).insts.bin; mv -f decode.xclbin decode_L$(LREF).xclbin; \
	    cp -f air.mlir decode_L$(LREF).air.mlir; \
	  else echo "decode_L$(LREF) FAILED (see /tmp/lfm2_decode_L$(LREF).log)"; exit 1; fi
	@echo "LFM2 decode build complete: decode_L$(LBUILD) + decode_L$(LREF)."

## Run unified inference (NPU prefill + fused NPU decode).
run:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) \
		--run-only --n-tokens $(N_TOKENS) --prompt "$(PROMPT)" --model $(MODEL)

## Run with the decode-throughput / TTFT profiling summary. --no-eos-stop so
## throughput is averaged over the full N_TOKENS budget: the default prompt
## answers in a handful of tokens, and such a short average is dominated by the
## first (cold) dispatch and understates steady-state tok/s.
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

# ---- shared verify subsystem (top-k token-set inclusion, NPU q4_0 vs HF bf16) ----

## Top-k token-set inclusion gate (NPU q4_0 vs HF bf16, 2 prompts x 32 tokens, k=5).
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL) --max-prompts 2

## Full-sweep variant of `make verify` (all prompts in the prompt file).
verify-full:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts topk_token --model $(MODEL)

## Diagnosis lens (per-layer cosine vs HF bf16, single prompt, informational).
diagnosis:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(VERIFY_RUNNER) --runner=$(RUNNER_ADAPTER) \
		--prompts single --prompt "$(PROMPT)" --model $(MODEL)

## Fast weight-integrity smoke: prefill "The capital of France is"; PASS iff the
## first-token argmax is " Paris". Cheap (prefill only, no decode build).
verify-paris:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(PREFILL) --model $(MODEL_SOURCE)

## Decode-ONLY numerics gate: real Q4_0 weights, both layer types interleaved,
## no prefill. Scores logits and the read-back ShortConv state per layer, so a
## failure localises. Run this first after touching the decode builder.
decode-gate: weights
	cd $(srcdir) && $(PYTHON) $(DECODE_GATE) --lbuild $(LBUILD) --seq

## Warm prefill TTFT benchmark at CTX (per-ELF BO-write / NPU-run / BO-read breakdown).
profile-prefill:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && LFM2_BENCH_L=$(CTX) $(PYTHON) $(PREFILL)

## Single Q&A turn:  make ask PROMPT="What is the capital of France?"
ask:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) --run-only --prompt "$(PROMPT)" --model $(MODEL) --n-tokens $(N_TOKENS)

## Full prefill+decode Paris gate: first generated token -> *** PARIS ***.
gen:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(PYTHON) $(DRIVER) --run-only --greedy --n-tokens 12

## Remove all build artifacts, kernel/weight caches, and verify reports. The
## decode templates live in THIS directory (not in fused_decode/), so unlike the
## llama example this clean does not have to reach into that one.
clean:
	rm -rf $(srcdir)/$(BUILD_DIR) $(srcdir)/air_project $(srcdir)/__pycache__ \
	       $(srcdir)/*.o $(srcdir)/*.ll $(srcdir)/air.mlir \
	       $(srcdir)/decode_L*.xclbin $(srcdir)/decode_L*.insts.bin \
	       $(srcdir)/decode_L*.air.mlir $(DECODE_FLAGS_STAMP) 2>/dev/null || true
	rm -rf $(srcdir)/../verify/reports 2>/dev/null || true
	@echo "Build dirs, caches, verify/reports/ and decode templates removed. Rebuild with 'make compile'."
