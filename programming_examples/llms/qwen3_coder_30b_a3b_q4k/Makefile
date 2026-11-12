# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3-Coder-30B-A3B decode and prefill on NPU2 (MLIR-AIR): the 48 layers in
# one dispatch and the Q8 LM head in a second, the router choosing and fetching
# the experts on device; the prompt prefilled on the shared fused prefill
# device. The weights are the HF checkpoint quantized once into a Q4_K bundle
# (QWEN3C_BUNDLE, default under HF_HOME).
#
#   make prepare         Quantize the HF checkpoint into the bundle (once)
#   make compile         Build the decode and LM head ELFs
#   make compile-fused-prefill  Build the fused prefill device into FUSED_DIR
#   make run             Greedy decode of PROMPT, checked against the CPU reference
#   make prefill-fused   The fused prefill of PROMPT repeated to LONG_LEN tokens
#                        vs the CPU reference
#   make verify          Each stage alone, prefill-fused, the decode, the
#                        reference against HF transformers' Qwen3-MoE layers,
#                        then the top-k token-set gate vs HF bf16
#   make profile         Time per token at context CTX, no reference
#   make profile-decode  Decode tok/s at each of CONTEXTS (one build each)
#   make profile-prefill TTFT of a CTX-token prompt
#
# With FUSED_PREFILL_DIR= the prompt goes through the decode a token at a time.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
DECODE := $(PYTHON) $(srcdir)/decode.py
comma := ,

MAXL      ?= 64
# shorter cache reads below MAXL, each in a layers' ELF of its own
READS     ?=
GEN       ?= 8
PROMPT    ?= The capital of France
CTX       ?= 1024
CONTEXTS  ?= 1024,4096
BENCH     ?= 20
BUILD_DIR ?= build_decode
LONG_LEN  ?= 300
MODEL_CHOICE ?= instruct
MAX_PROMPTS  ?= 2

BUNDLE := $(shell $(PYTHON) -c "import sys; sys.path.insert(0, '$(srcdir)'); \
	import qwen3_coder_30b_a3b_q4k_weights as W; print(W.bundle_path())")
FUSED_DIR ?= $(CURDIR)/_fused_prefill
override FUSED_DIR := $(abspath $(FUSED_DIR))
FUSED_PREFILL_DIR ?= $(FUSED_DIR)
# absolute, as the recipes run in their build directories; empty opts out
override FUSED_PREFILL_DIR := $(if $(FUSED_PREFILL_DIR),$(abspath $(FUSED_PREFILL_DIR)))
export FUSED_PREFILL_DIR
FUSED_DEP := $(if $(FUSED_PREFILL_DIR),$(abspath $(FUSED_PREFILL_DIR))/manifest.json)
STAGES    := attn_layer lm_head

.PHONY: prepare compile compile-fused-prefill run prefill-fused verify profile \
	profile-decode profile-prefill clean

prepare: $(BUNDLE)

$(BUNDLE):
	$(PYTHON) $(srcdir)/qwen3_coder_30b_a3b_q4k_weights.py --prepare --out $@

compile:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DECODE) --compile-only --maxl $(MAXL)

compile-fused-prefill:
	$(PYTHON) $(srcdir)/fused_build.py $(FUSED_DIR)

%/manifest.json:
	$(PYTHON) $(srcdir)/fused_build.py $*

run: $(BUNDLE) $(FUSED_DEP)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DECODE) --maxl $(MAXL) --reads "$(READS)" --gen $(GEN) --prompt "$(PROMPT)"

prefill-fused: $(BUNDLE) $(FUSED_DIR)/manifest.json
	$(PYTHON) $(srcdir)/prefill.py $(FUSED_DIR) --prompt "$(PROMPT)" \
		--prompt-len $(LONG_LEN)

verify: $(BUNDLE)
	for s in $(STAGES); do \
		mkdir -p build_$$s && (cd build_$$s && $(PYTHON) $(srcdir)/$$s.py) \
			|| exit 1; done
	$(MAKE) -f $(srcdir)/Makefile prefill-fused
	$(MAKE) -f $(srcdir)/Makefile run
	$(PYTHON) $(srcdir)/check_vs_hf.py
	cd $(srcdir)/.. && QWEN3C_VERIFY_DIR=$(CURDIR)/build_verify \
		$(PYTHON) verify/verify_runner.py \
		--runner=qwen3_coder_30b_a3b_q4k.verify_adapter --prompts topk_token \
		--model $(MODEL_CHOICE) --max-prompts $(MAX_PROMPTS)

profile: $(BUNDLE)
	mkdir -p build_ctx$(CTX) && cd build_ctx$(CTX) && \
		$(DECODE) --maxl $(CTX) --gen 2 --no-ref --bench $(BENCH) --bench-l $(CTX)

# The context is compiled in: one build per point, each at its own maxl.
profile-decode: $(BUNDLE)
	for c in $(subst $(comma), ,$(CONTEXTS)); do \
		mkdir -p build_ctx$$c && (cd build_ctx$$c && \
			$(DECODE) --maxl $$c --gen 2 --no-ref --bench $(BENCH) --bench-l $$c) \
			|| exit 1; done

profile-prefill: $(BUNDLE) $(FUSED_DEP)
	mkdir -p build_ctx$(CTX) && cd build_ctx$(CTX) && \
		$(DECODE) --maxl $$(( ($(CTX) + 2 + 15) / 16 * 16 )) --prompt-len $(CTX) \
			--gen 2 --no-ref --warm

clean:
	rm -rf $(BUILD_DIR) build_* $(FUSED_DIR) *.o
