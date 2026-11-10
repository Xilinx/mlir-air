# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.6-35B-A3B Q4NX decode on NPU2 (MLIR-AIR), FastFlowLM's layout: the 40
# layers in one dispatch and the Q8 LM head in a second, the router choosing
# and fetching the experts on device. Weights come from FastFlowLM's model.q4nx
# bundle (FastFlowLM/Qwen3.6-35B-A3B-NPU2; QWEN36_FLM_DIR names a local copy).
#
#   make compile         Build the decode and LM head ELFs
#   make compile-fused-prefill  Build the fused prefill device into FUSED_DIR
#   make run             Greedy decode of PROMPT, checked against the CPU reference
#   make prefill-fused   The fused prefill of PROMPT repeated to LONG_LEN tokens
#                        vs the CPU reference
#   make verify          Each stage alone, prefill-fused, the decode, and the
#                        reference against HF transformers' Qwen3.5-MoE layers
#   make profile         Time per token at context CTX, no reference
#   make profile-prefill TTFT of a CTX-token prompt
#   make kernels         Every kernel test
#
# The prompt is prefilled on the shared fused prefill device (fused_spec.py,
# prefill.py) and handed to the decode, whose LM head gives the first token;
# with FUSED_PREFILL_DIR= it goes through the decode a token at a time.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
DECODE := $(PYTHON) $(srcdir)/decode.py

MAXL      ?= 64
GEN       ?= 8
PROMPT    ?= The capital of France
CTX       ?= 1024
BENCH     ?= 20
BUILD_DIR ?= build_decode
LONG_LEN  ?= 300

FUSED_DIR ?= $(CURDIR)/_fused_prefill
override FUSED_DIR := $(abspath $(FUSED_DIR))
FUSED_PREFILL_DIR ?= $(FUSED_DIR)
export FUSED_PREFILL_DIR
FUSED_DEP := $(if $(FUSED_PREFILL_DIR),$(abspath $(FUSED_PREFILL_DIR))/manifest.json)
STAGES    := moe_block gdn_layer attn_layer lm_head
KERNELS   := q4k_mvm router gdn_prep gdn_decode gate rope attn ffn

.PHONY: compile compile-fused-prefill run prefill-fused verify profile \
	profile-prefill kernels clean

compile:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DECODE) --compile-only --maxl $(MAXL)

compile-fused-prefill:
	$(PYTHON) $(srcdir)/fused_build.py $(FUSED_DIR)

%/manifest.json:
	$(PYTHON) $(srcdir)/fused_build.py $*

run: $(FUSED_DEP)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DECODE) --maxl $(MAXL) --gen $(GEN) --prompt "$(PROMPT)"

prefill-fused: $(FUSED_DIR)/manifest.json
	$(PYTHON) $(srcdir)/prefill.py $(FUSED_DIR) --prompt "$(PROMPT)" \
		--prompt-len $(LONG_LEN)

verify:
	for s in $(STAGES); do \
		mkdir -p build_$$s && (cd build_$$s && $(PYTHON) $(srcdir)/$$s.py) \
			|| exit 1; done
	$(MAKE) -f $(srcdir)/Makefile prefill-fused
	$(MAKE) -f $(srcdir)/Makefile run
	$(PYTHON) $(srcdir)/check_vs_hf.py

profile:
	mkdir -p build_ctx$(CTX) && cd build_ctx$(CTX) && \
		$(DECODE) --maxl $(CTX) --gen 2 --no-ref --bench $(BENCH) --bench-l $(CTX)

profile-prefill: $(FUSED_DEP)
	mkdir -p build_ctx$(CTX) && cd build_ctx$(CTX) && \
		$(DECODE) --maxl $$(( ($(CTX) + 2 + 15) / 16 * 16 )) --prompt-len $(CTX) \
			--gen 2 --no-ref --warm

kernels:
	mkdir -p build_kernels && cd build_kernels && \
		for k in $(KERNELS); do \
			$(PYTHON) $(srcdir)/kernel_tests/$$k.py || exit 1; done

clean:
	rm -rf $(BUILD_DIR) build_* $(FUSED_DIR) *.o
